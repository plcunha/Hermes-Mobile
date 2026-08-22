"""Core Agent Bridge - Adapts Hermes Agent for Mobile"""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_mobile.config.settings import get_settings
from hermes_mobile.core.context_compressor import compress_messages, needs_compression
from hermes_mobile.core.prompt_caching import apply_cache_control, supports_caching
from hermes_mobile.providers import ProviderProfile, get_provider_profile
from hermes_mobile.tools.agent_tools import (
    clarify_tool,
    memory_tool,
    session_search_tool,
)
from hermes_mobile.tools.browser_session import (
    browser_back_tool,
    browser_click_tool,
    browser_get_images_tool,
)
from hermes_mobile.tools.desktop_tools import (
    cronjob_tool,
    execute_code_tool,
    patch_tool,
    search_files_tool,
    skill_manage_tool,
    skill_view_tool,
    skills_list_tool,
    todo_tool,
)
from hermes_mobile.tools.home_assistant import (
    ha_call_service_tool,
    ha_get_state_tool,
    ha_list_entities_tool,
    ha_list_services_tool,
)
from hermes_mobile.tools.kanban_tools import (
    kanban_block_tool,
    kanban_comment_tool,
    kanban_complete_tool,
    kanban_create_tool,
    kanban_list_tool,
    kanban_move_tool,
    kanban_show_tool,
    kanban_unblock_tool,
)
from hermes_mobile.tools.media_tools import (
    image_generate_tool,
    text_to_speech_tool,
    vision_analyze_tool,
)
from hermes_mobile.tools.path_security import validate_and_resolve_path
from hermes_mobile.tools.process_tools import MobileProcessRegistry
from hermes_mobile.tools.project_tools import (
    project_create_tool,
    project_list_tool,
    project_switch_tool,
)
from hermes_mobile.tools.security import safe_calculate
from hermes_mobile.tools.web_tools import (
    browser_navigate_tool,
    browser_snapshot_tool,
    web_extract_tool,
    web_search_tool,
)

logger = logging.getLogger(__name__)


class ToolCall:
    """Represents a tool call from the model"""

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
    ):
        self.name = name
        self.arguments = arguments
        self.call_id = call_id or str(uuid.uuid4())
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Message:
    """Represents a chat message"""

    def __init__(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.role = role  # user, assistant, system, tool
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_call_id = tool_call_id
        self.name = name
        self.timestamp = datetime.now()
        self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "id": self.id,
        }

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        return cls("assistant", content, tool_calls=tool_calls)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> "Message":
        return cls("tool", content, tool_call_id=tool_call_id, name=name)


class MobileAgent:
    """Mobile-adapted Hermes Agent"""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        memory_provider: Optional[Any] = None,
        skill_manager: Optional[Any] = None,
        on_tool_call: Optional[Callable[[ToolCall], None]] = None,
        on_tool_result: Optional[Callable[[ToolCall], None]] = None,
        on_message: Optional[Callable[[Message], None]] = None,
        blocked_tools: Optional[set[str]] = None,
    ):
        self.settings = get_settings()
        self.model = model or self.settings.default_model
        self.provider = provider or self.settings.default_provider
        self.system_prompt = system_prompt or self.settings.system_prompt
        self.tools = tools or []
        self.memory_provider = memory_provider
        self.skill_manager = skill_manager
        self.blocked_tools = frozenset(blocked_tools or ())
        self.process_registry = MobileProcessRegistry()

        # Callbacks for UI updates
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_message = on_message

        # Conversation state
        self.messages: List[Message] = []
        self.session_id = str(uuid.uuid4())
        self.iteration = 0
        self.max_iterations = self.settings.max_iterations

        # Initialize OpenAI-compatible client
        self._client: Optional[Any] = None
        self._client_error: Optional[str] = None
        self._init_client()

    def _init_client(self):
        """Initialize the active OpenAI-compatible provider client."""
        from openai import AsyncOpenAI

        profile = self._get_provider_profile()
        if profile is not None and profile.api_mode != "chat_completions":
            self._client = None
            self._client_error = (
                f"Provider '{self.provider}' requires the {profile.api_mode!r} API, "
                "which is not available in this mobile build. Use it through OpenRouter."
            )
            return

        api_key = self._get_api_key()
        base_url = self._get_base_url()

        if not api_key:
            self._client = None
            self._client_error = (
                f"No API key configured for provider '{self.provider}'. "
                "Open Settings and add the provider API key."
            )
            return

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=profile.default_headers if profile else None,
            timeout=self.settings.request_timeout,
            max_retries=self.settings.max_retries,
        )
        self._client_error = None

    def _require_client(self):
        """Return the client or raise a friendly error when not configured."""
        if self._client is None:
            raise RuntimeError(self._client_error or "AI provider not configured.")
        return self._client

    async def transcribe_audio_file(self, path: Path) -> str:
        """Transcribe a local audio file with the active OpenAI-compatible provider.

        Hermes Mobile/Flet does not expose native microphone recording here; the
        chat mic uses this for selected audio files only. Providers without an
        audio transcription endpoint surface a friendly runtime error.
        """
        source = Path(path).expanduser()
        if not source.is_file():
            raise RuntimeError(f"Audio file not found: {source}")
        client = self._require_client()
        audio = getattr(client, "audio", None)
        transcriptions = getattr(audio, "transcriptions", None)
        create = getattr(transcriptions, "create", None)
        if create is None:
            raise RuntimeError(
                "The active provider does not expose OpenAI-compatible audio transcription."
            )
        try:
            with source.open("rb") as file_obj:
                result = await create(model="whisper-1", file=file_obj)
        except Exception as exc:
            raise RuntimeError(f"Audio transcription failed: {exc}") from exc
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text")
        text = str(text or "").strip()
        if not text:
            raise RuntimeError("Audio transcription returned no text.")
        return text

    def _get_provider_profile(self) -> Optional[ProviderProfile]:
        """Resolve the configured provider, including registry aliases."""
        return get_provider_profile(self.provider)

    def _get_api_key(self) -> str:
        """Resolve the API key from persisted settings, then profile env vars."""
        profile = self._get_provider_profile()
        canonical_name = profile.name if profile else self.provider
        try:
            from hermes_mobile.remote.secrets import ProviderSecretStore

            secured = ProviderSecretStore(self.settings.get_data_dir()).get_key(canonical_name)
            if secured:
                return secured
        except Exception:
            logger.warning("Could not read the encrypted provider credential store")
        setting_names = {
            "openrouter": "openrouter_api_key",
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "google": "gemini_api_key",
        }
        setting_name = setting_names.get(canonical_name)
        if setting_name:
            persisted = getattr(self.settings, setting_name, "") or ""
            if persisted:
                return persisted

        if profile:
            for env_var in profile.env_vars:
                value = os.environ.get(env_var, "").strip()
                if value:
                    return value
        return ""

    def reconfigure(self, *, provider: str, model: str) -> None:
        """Apply a local route change without discarding conversation state."""
        self.provider = str(provider).strip()
        self.model = str(model).strip()
        self._init_client()

    def _get_base_url(self) -> str:
        """Resolve the provider endpoint from its declarative profile."""
        profile = self._get_provider_profile()
        if profile and profile.base_url:
            return profile.base_url
        return "https://openrouter.ai/api/v1"

    def add_message(self, message: Message):
        """Add a message to the conversation"""
        self.messages.append(message)
        if self.on_message:
            self.on_message(message)

    def add_user_message(self, content: str):
        """Add a user message"""
        self.add_message(Message.user(content))

    def add_assistant_message(self, content: str, tool_calls: Optional[List[ToolCall]] = None):
        """Add an assistant message"""
        self.add_message(Message.assistant(content, tool_calls))

    def add_tool_result(self, content: str, tool_call_id: str, name: str):
        """Add a tool result message"""
        self.add_message(Message.tool(content, tool_call_id, name))

    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """Get messages in API format with caching if supported."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in self.messages:
            api_msg = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                api_msg["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                api_msg["tool_call_id"] = msg.tool_call_id
            if msg.name:
                api_msg["name"] = msg.name
            messages.append(api_msg)

        if supports_caching(self.settings.default_provider):
            messages = apply_cache_control(messages, self.settings.default_provider)

        return messages

    async def run_conversation(
        self,
        user_input: str,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Run a conversation turn with the agent"""
        self.add_user_message(user_input)
        self.iteration = 0

        while self.iteration < self.max_iterations:
            self.iteration += 1

            api_messages = self.get_messages_for_api()
            if needs_compression(api_messages, self.settings.max_tokens):
                self.messages = self._apply_compression()
                api_messages = self.get_messages_for_api()

            try:
                response = await self._call_model(stream=stream)
                content_parts: List[str] = []

                if stream:
                    streamed_tool_calls: Dict[int, Dict[str, str]] = {}
                    async for chunk in response:
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)
                        if delta is None:
                            continue

                        content_delta = getattr(delta, "content", None)
                        if content_delta:
                            content_parts.append(content_delta)
                            yield content_delta

                        for tc_delta in getattr(delta, "tool_calls", None) or []:
                            index = getattr(tc_delta, "index", 0) or 0
                            call = streamed_tool_calls.setdefault(
                                index,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            call_id = getattr(tc_delta, "id", None)
                            if call_id:
                                call["id"] = call_id
                            function = getattr(tc_delta, "function", None)
                            if function is not None:
                                name = getattr(function, "name", None)
                                arguments = getattr(function, "arguments", None)
                                if name:
                                    call["name"] += name
                                if arguments:
                                    call["arguments"] += arguments

                    content = "".join(content_parts)
                    tool_calls = self._build_stream_tool_calls(streamed_tool_calls)
                else:
                    content = response.choices[0].message.content or ""
                    yield content
                    tool_calls = self._extract_tool_calls(response)

                # Keep the API history valid: an assistant message containing the
                # tool calls must precede their tool result messages.
                if content or tool_calls:
                    self.add_assistant_message(content, tool_calls)

                if tool_calls:
                    await self._execute_tool_calls(tool_calls)
                    continue  # Continue conversation with tool results

                # No tool calls, conversation turn complete
                break

            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                yield f"\n\nError: {str(e)}"
                break

        # Save to memory
        if self.memory_provider:
            await self.memory_provider.save_conversation(
                self.session_id,
                self.messages,
            )

    async def _call_model(self, stream: bool = True):
        """Call the model API"""
        messages = self.get_messages_for_api()
        client = self._require_client()

        return await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools if self.tools else None,
            tool_choice="auto" if self.tools else None,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            stream=stream,
        )

    def _build_stream_tool_calls(
        self,
        streamed_tool_calls: Dict[int, Dict[str, str]],
    ) -> List[ToolCall]:
        """Reconstruct OpenAI tool-call deltas after a streamed response."""
        tool_calls: List[ToolCall] = []
        for index in sorted(streamed_tool_calls):
            raw = streamed_tool_calls[index]
            if not raw["name"]:
                logger.warning("Ignoring streamed tool call without a function name")
                continue
            try:
                arguments = json.loads(raw["arguments"] or "{}")
            except json.JSONDecodeError:
                logger.warning("Invalid JSON arguments for streamed tool %s", raw["name"])
                arguments = {}
            tool_calls.append(
                ToolCall(
                    name=raw["name"],
                    arguments=arguments,
                    call_id=raw["id"] or str(uuid.uuid4()),
                )
            )
        return tool_calls

    def _extract_tool_calls(self, response) -> List[ToolCall]:
        """Extract tool calls from model response"""
        tool_calls = []

        if (
            hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        ):
            for tc in response.choices[0].message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_call = ToolCall(
                    name=tc.function.name,
                    arguments=args,
                    call_id=tc.id,
                )
                tool_calls.append(tool_call)

        return tool_calls

    async def _execute_tool_calls(self, tool_calls: List[ToolCall]):
        """Execute tool calls and add results to conversation"""
        for tool_call in tool_calls:
            tool_call.started_at = datetime.now()

            if self.on_tool_call:
                self.on_tool_call(tool_call)

            try:
                result = await self._execute_tool(tool_call.name, tool_call.arguments)
                tool_call.result = result
                tool_call.completed_at = datetime.now()

                self.add_tool_result(
                    json.dumps(result) if not isinstance(result, str) else result,
                    tool_call.call_id,
                    tool_call.name,
                )

            except Exception as e:
                logger.error(f"Tool {tool_call.name} failed: {e}")
                tool_call.error = str(e)
                tool_call.completed_at = datetime.now()

                self.add_tool_result(
                    f"Error: {str(e)}",
                    tool_call.call_id,
                    tool_call.name,
                )

            if self.on_tool_result:
                self.on_tool_result(tool_call)

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name"""
        # Check built-in tools first
        if name in self._builtin_tools:
            return await self._builtin_tools[name](**arguments)

        # Check skills
        if self.skill_manager:
            skill = self.skill_manager.get_skill(name)
            if skill:
                return await skill.execute(**arguments)

        raise ValueError(f"Unknown tool: {name}")

    @property
    def _builtin_tools(self) -> Dict[str, Callable]:
        tools = {
            "web_search": self._tool_web_search,
            "web_extract": self._tool_web_extract,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "list_files": self._tool_list_files,
            "search_files": self._tool_search_files,
            "patch": self._tool_patch,
            "terminal": self._tool_terminal,
            "process": self._tool_process,
            "run_command": self._tool_run_command,
            "execute_code": self._tool_execute_code,
            "get_time": self._tool_get_time,
            "calculate": self._tool_calculate,
            "session_search": self._tool_session_search,
            "memory": self._tool_memory,
            "clarify": self._tool_clarify,
            "todo": self._tool_todo,
            "skills_list": self._tool_skills_list,
            "skill_view": self._tool_skill_view,
            "skill_manage": self._tool_skill_manage,
            "cronjob": self._tool_cronjob,
            "browser_navigate": self._tool_browser_navigate,
            "browser_snapshot": self._tool_browser_snapshot,
            "browser_back": self._tool_browser_back,
            "browser_click": self._tool_browser_click,
            "browser_get_images": self._tool_browser_get_images,
            "vision_analyze": self._tool_vision_analyze,
            "image_generate": self._tool_image_generate,
            "text_to_speech": self._tool_text_to_speech,
            "ha_list_entities": self._tool_ha_list_entities,
            "ha_get_state": self._tool_ha_get_state,
            "ha_list_services": self._tool_ha_list_services,
            "ha_call_service": self._tool_ha_call_service,
            "project_list": self._tool_project_list,
            "project_create": self._tool_project_create,
            "project_switch": self._tool_project_switch,
            "kanban_list": self._tool_kanban_list,
            "kanban_create": self._tool_kanban_create,
            "kanban_show": self._tool_kanban_show,
            "kanban_move": self._tool_kanban_move,
            "kanban_complete": self._tool_kanban_complete,
            "kanban_block": self._tool_kanban_block,
            "kanban_unblock": self._tool_kanban_unblock,
            "kanban_comment": self._tool_kanban_comment,
            "delegate_tasks": self._tool_delegate_tasks,
            "delegate_task": self._tool_delegate_task,
        }
        return {name: handler for name, handler in tools.items() if name not in self.blocked_tools}

    async def _tool_web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web using DuckDuckGo."""
        return await web_search_tool(query, max_results=max_results)

    async def _tool_read_file(self, path: str) -> str:
        """Read a file with path security validation."""
        resolved, error = validate_and_resolve_path(path)
        if error:
            return f"Error: {error}"
        try:
            return resolved.read_text()
        except Exception as e:
            return f"Error reading file: {e}"

    async def _tool_write_file(self, path: str, content: str) -> str:
        """Write a file with path security validation."""
        resolved, error = validate_and_resolve_path(path)
        if error:
            return f"Error: {error}"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
            return f"File written to {resolved}"
        except Exception as e:
            return f"Error writing file: {e}"

    async def _tool_list_files(self, path: str = ".") -> List[str]:
        """List files in a directory with path security."""
        if path == ".":
            resolved = Path.cwd()
        else:
            resolved, error = validate_and_resolve_path(path)
            if error:
                return [f"Error: {error}"]
        try:
            return [str(p) for p in resolved.iterdir()]
        except Exception as e:
            return [f"Error: {e}"]

    async def _tool_search_files(
        self,
        pattern: str,
        path: str = ".",
        target: str = "content",
        file_glob: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Search file contents or filenames."""
        return await search_files_tool(
            pattern=pattern,
            path=path,
            target=target,
            file_glob=file_glob,
            limit=limit,
        )

    async def _tool_patch(
        self,
        path: str,
        old_string: str,
        new_string: str = "",
        replace_all: bool = False,
    ) -> Dict[str, Any]:
        """Patch a file with a find-and-replace edit."""
        return await patch_tool(
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )

    async def _tool_execute_code(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute Python code in a sandboxed subprocess."""
        return await execute_code_tool(code=code, timeout=timeout)

    async def _tool_todo(
        self,
        action: str,
        item_id: Optional[int] = None,
        content: Optional[str] = None,
        status: str = "pending",
    ) -> Dict[str, Any]:
        """Manage the agent task list."""
        return await todo_tool(
            action=action,
            item_id=item_id,
            content=content,
            status=status,
        )

    async def _tool_skills_list(self) -> Dict[str, Any]:
        """List installed skills."""
        return await skills_list_tool(self.skill_manager)

    async def _tool_skill_view(self, name: str) -> Dict[str, Any]:
        """View a skill's metadata and schema."""
        return await skill_view_tool(name, self.skill_manager)

    async def _tool_skill_manage(
        self,
        action: str,
        name: str,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enable, disable, remove or install a skill."""
        return await skill_manage_tool(
            action=action,
            name=name,
            url=url,
            skill_manager=self.skill_manager,
        )

    async def _tool_cronjob(self, action: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """List, run, pause or resume cron jobs."""
        return await cronjob_tool(action=action, job_id=job_id)

    async def _tool_terminal(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = 180,
        background: bool = False,
    ) -> Dict[str, Any]:
        """Run a shell command now or register it as a background process."""
        return await self.process_registry.terminal(
            command,
            cwd=cwd,
            timeout=timeout,
            background=background,
        )

    async def _tool_process(
        self,
        action: str,
        session_id: Optional[str] = None,
        data: Optional[str] = None,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Inspect or control a process started by the terminal tool."""
        return await self.process_registry.process(
            action,
            session_id=session_id,
            data=data,
            timeout=timeout,
            offset=offset,
            limit=limit,
        )

    async def _tool_run_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Compatibility wrapper for the original foreground terminal API."""
        result = await self._tool_terminal(command, cwd=cwd)
        if "error" in result:
            return {"error": result["error"], "stdout": result.get("output", "")}
        return {
            "stdout": result.get("output", ""),
            "stderr": "",
            "returncode": result.get("exit_code"),
        }

    async def _tool_get_time(self) -> str:
        """Get current time"""
        return datetime.now().isoformat()

    async def _tool_web_extract(self, urls: List[str], format: str = "text") -> Dict[str, Any]:
        """Extract content from web pages."""
        return await web_extract_tool(urls, format=format)

    async def _tool_session_search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search past conversation sessions."""
        return await session_search_tool(query, limit=limit, memory_provider=self.memory_provider)

    async def _tool_memory(
        self,
        action: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Store and retrieve memory entries."""
        return await memory_tool(
            action=action,
            key=key,
            value=value,
            query=query,
            limit=limit,
            memory_provider=self.memory_provider,
        )

    async def _tool_clarify(self, topic: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Get clarification suggestions."""
        return await clarify_tool(topic, context=context)

    async def _tool_browser_navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL and return page content."""
        return await browser_navigate_tool(url)

    async def _tool_browser_snapshot(self, url: str) -> Dict[str, Any]:
        """Return a text snapshot of a web page."""
        return await browser_snapshot_tool(url)

    async def _tool_browser_back(self) -> Dict[str, Any]:
        """Go back to the previous page."""
        return await browser_back_tool()

    async def _tool_browser_click(self, href: str) -> Dict[str, Any]:
        """Click a link by href."""
        return await browser_click_tool(href)

    async def _tool_browser_get_images(self) -> Dict[str, Any]:
        """List images on the current page."""
        return await browser_get_images_tool()

    async def _tool_vision_analyze(
        self, image_url: str, question: str = "Describe this image in detail."
    ) -> Dict[str, Any]:
        """Analyze an image with a vision model."""
        return await vision_analyze_tool(image_url=image_url, question=question, agent=self)

    async def _tool_image_generate(self, prompt: str) -> Dict[str, Any]:
        """Generate an image from a prompt."""
        return await image_generate_tool(prompt=prompt, agent=self)

    async def _tool_text_to_speech(self, text: str, voice: str = "alloy") -> Dict[str, Any]:
        """Generate speech audio from text."""
        return await text_to_speech_tool(text=text, agent=self, voice=voice)

    async def _tool_ha_list_entities(self) -> Dict[str, Any]:
        """List Home Assistant entities."""
        return await ha_list_entities_tool()

    async def _tool_ha_get_state(self, entity_id: str) -> Dict[str, Any]:
        """Read a Home Assistant entity state."""
        return await ha_get_state_tool(entity_id=entity_id)

    async def _tool_ha_list_services(self) -> Dict[str, Any]:
        """List Home Assistant services."""
        return await ha_list_services_tool()

    async def _tool_ha_call_service(
        self,
        domain: str,
        service: str,
        service_data: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call a Home Assistant service."""
        return await ha_call_service_tool(
            domain=domain,
            service=service,
            service_data=service_data,
            entity_id=entity_id,
        )

    async def _tool_project_list(self) -> Dict[str, Any]:
        """List projects and the active one."""
        return await project_list_tool()

    async def _tool_project_create(self, name: str) -> Dict[str, Any]:
        """Create a new project."""
        return await project_create_tool(name=name)

    async def _tool_project_switch(self, name: str) -> Dict[str, Any]:
        """Switch the active project workspace."""
        return await project_switch_tool(name=name, agent=self)

    async def _tool_kanban_list(self, column: Optional[str] = None) -> Dict[str, Any]:
        """List kanban tasks."""
        return await kanban_list_tool(column=column)

    async def _tool_kanban_create(
        self, title: str, description: str = "", column: str = "backlog"
    ) -> Dict[str, Any]:
        """Create a kanban task."""
        return await kanban_create_tool(title=title, description=description, column=column)

    async def _tool_kanban_show(self, task_id: str) -> Dict[str, Any]:
        """Show a kanban task."""
        return await kanban_show_tool(task_id=task_id)

    async def _tool_kanban_move(self, task_id: str, column: str) -> Dict[str, Any]:
        """Move a kanban task between columns."""
        return await kanban_move_tool(task_id=task_id, column=column)

    async def _tool_kanban_complete(self, task_id: str) -> Dict[str, Any]:
        """Complete a kanban task."""
        return await kanban_complete_tool(task_id=task_id)

    async def _tool_kanban_block(self, task_id: str, reason: str = "") -> Dict[str, Any]:
        """Block a kanban task."""
        return await kanban_block_tool(task_id=task_id, reason=reason)

    async def _tool_kanban_unblock(self, task_id: str) -> Dict[str, Any]:
        """Unblock a kanban task."""
        return await kanban_unblock_tool(task_id=task_id)

    async def _tool_kanban_comment(self, task_id: str, text: str) -> Dict[str, Any]:
        """Comment on a kanban task."""
        return await kanban_comment_tool(task_id=task_id, text=text)

    async def _tool_delegate_task(
        self,
        goal: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one task in an isolated child agent with recursion blocked."""
        goal = goal.strip()
        if not goal:
            return {"error": "Goal is required"}

        tool_calls: List[str] = []

        def record_tool_call(call: ToolCall) -> None:
            tool_calls.append(call.name)

        child = MobileAgent(
            model=self.model,
            provider=self.provider,
            system_prompt=(
                f"{self.system_prompt}\n\n"
                "You are an isolated mobile subagent. Complete only the delegated goal, "
                "use available tools when needed, and return a concise evidence-based result."
            ),
            memory_provider=None,
            skill_manager=self.skill_manager,
            on_tool_call=record_tool_call,
            blocked_tools={"delegate_task", "delegate_tasks", "clarify", "cronjob", "memory"},
        )
        child.max_iterations = self.max_iterations
        prompt = goal if not context else f"Context:\n{context}\n\nGoal:\n{goal}"
        chunks: List[str] = []
        try:
            async for chunk in child.run_conversation(prompt, stream=True):
                chunks.append(chunk)
        except Exception as exc:
            return {"status": "failed", "goal": goal, "error": str(exc)}
        return {
            "status": "completed",
            "goal": goal,
            "content": "".join(chunks),
            "tool_calls": tool_calls,
            "session_id": child.session_id,
        }

    async def _tool_delegate_tasks(
        self, tasks: List[str], context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run up to three independent child agents concurrently."""
        goals = [task.strip() for task in tasks if task.strip()]
        if not goals:
            return {"error": "At least one task is required"}
        if len(goals) > 3:
            return {"error": "At most three tasks can run concurrently"}
        results = await asyncio.gather(
            *(self._tool_delegate_task(goal, context=context) for goal in goals)
        )
        return {"status": "completed", "mode": "parallel", "results": results}

    async def _tool_calculate(self, expression: str) -> Any:
        """Calculate a mathematical expression safely."""
        return safe_calculate(expression)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for the model"""
        schemas = []

        # Built-in tools
        schemas.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "max_results": {"type": "integer", "default": 5},
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file from the filesystem",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path"},
                            },
                            "required": ["path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path"},
                                "content": {"type": "string", "description": "Content to write"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "description": "List files in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Directory path",
                                    "default": ".",
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "description": "Run a shell command in foreground or background",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string", "description": "Command to run"},
                                "cwd": {"type": "string", "description": "Working directory"},
                                "timeout": {
                                    "type": "integer",
                                    "description": "Foreground timeout in seconds",
                                    "default": 180,
                                },
                                "background": {
                                    "type": "boolean",
                                    "description": "Return a process session immediately",
                                    "default": False,
                                },
                            },
                            "required": ["command"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "process",
                        "description": "Inspect or control a background terminal process",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "list",
                                        "poll",
                                        "log",
                                        "wait",
                                        "kill",
                                        "write",
                                        "submit",
                                        "close",
                                    ],
                                },
                                "session_id": {"type": "string"},
                                "data": {"type": "string"},
                                "timeout": {"type": "integer"},
                                "offset": {"type": "integer"},
                                "limit": {"type": "integer", "default": 200},
                            },
                            "required": ["action"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "description": "Run a shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string", "description": "Command to run"},
                                "cwd": {"type": "string", "description": "Working directory"},
                            },
                            "required": ["command"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Get current date and time",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Calculate a mathematical expression",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string", "description": "Math expression"},
                            },
                            "required": ["expression"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "web_extract",
                        "description": "Extract text content from web pages",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "urls": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of URLs to extract",
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["text", "markdown"],
                                    "default": "text",
                                },
                            },
                            "required": ["urls"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "session_search",
                        "description": "Search past conversation sessions for relevant context",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "limit": {"type": "integer", "default": 5},
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "memory",
                        "description": "Store and retrieve information in long-term memory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["store", "retrieve", "search", "list", "delete"],
                                    "description": "Memory action",
                                },
                                "key": {"type": "string", "description": "Memory key"},
                                "value": {"type": "string", "description": "Value to store"},
                                "query": {"type": "string", "description": "Search query"},
                                "limit": {"type": "integer", "default": 5},
                            },
                            "required": ["action"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "clarify",
                        "description": "Get clarification suggestions for ambiguous requests",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string", "description": "Topic to clarify"},
                                "context": {"type": "string", "description": "Additional context"},
                            },
                            "required": ["topic"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_navigate",
                        "description": "Navigate to a URL and return page content",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "URL to navigate to"},
                            },
                            "required": ["url"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_snapshot",
                        "description": "Return a cleaned text snapshot of a web page",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "URL to snapshot"},
                            },
                            "required": ["url"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_back",
                        "description": "Go back to the previous page in the browsing session",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_click",
                        "description": "Click a link by href (relative links resolve against the current page)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "href": {"type": "string", "description": "Link href to click"},
                            },
                            "required": ["href"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "browser_get_images",
                        "description": "List images on the current page of the browsing session",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_files",
                        "description": "Search file contents with a regex or find files by name",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "pattern": {
                                    "type": "string",
                                    "description": "Regex pattern to search for",
                                },
                                "path": {
                                    "type": "string",
                                    "description": "Directory to search in",
                                    "default": ".",
                                },
                                "target": {
                                    "type": "string",
                                    "enum": ["content", "files"],
                                    "default": "content",
                                },
                                "file_glob": {
                                    "type": "string",
                                    "description": "Optional filename filter",
                                },
                                "limit": {"type": "integer", "default": 50},
                            },
                            "required": ["pattern"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "patch",
                        "description": "Edit a file by replacing an exact string (old_string) with new_string",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path"},
                                "old_string": {
                                    "type": "string",
                                    "description": "Exact text to find",
                                },
                                "new_string": {"type": "string", "description": "Replacement text"},
                                "replace_all": {"type": "boolean", "default": False},
                            },
                            "required": ["path", "old_string"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "description": "Execute Python code in a sandboxed subprocess and return stdout/stderr",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "description": "Python code to run"},
                                "timeout": {"type": "integer", "default": 60},
                            },
                            "required": ["code"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "todo",
                        "description": "Manage a task list (add/update/remove/list)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["add", "update", "remove", "list"],
                                },
                                "item_id": {
                                    "type": "integer",
                                    "description": "Item id for update/remove",
                                },
                                "content": {"type": "string", "description": "Task text for add"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "default": "pending",
                                },
                            },
                            "required": ["action"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "skills_list",
                        "description": "List installed skills",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "skill_view",
                        "description": "View a skill's metadata and schema",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string", "description": "Skill name"}},
                            "required": ["name"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "skill_manage",
                        "description": "Enable, disable, remove or install a skill",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["enable", "disable", "remove", "install"],
                                },
                                "name": {"type": "string", "description": "Skill name"},
                                "url": {"type": "string", "description": "URL for install"},
                            },
                            "required": ["action", "name"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "cronjob",
                        "description": "List, run, pause or resume cron jobs",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["list", "run", "pause", "resume"],
                                },
                                "job_id": {
                                    "type": "string",
                                    "description": "Job id for run/pause/resume",
                                },
                            },
                            "required": ["action"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vision_analyze",
                        "description": "Analyze an image with a vision model (URL or local path)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_url": {
                                    "type": "string",
                                    "description": "Image URL or local path",
                                },
                                "question": {
                                    "type": "string",
                                    "description": "Question about the image",
                                    "default": "Describe this image in detail.",
                                },
                            },
                            "required": ["image_url"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "image_generate",
                        "description": "Generate an image from a text prompt",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Image prompt"},
                            },
                            "required": ["prompt"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "text_to_speech",
                        "description": "Generate speech audio from text (saved to the device)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text to speak"},
                                "voice": {
                                    "type": "string",
                                    "description": "Voice name",
                                    "default": "alloy",
                                },
                            },
                            "required": ["text"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "ha_list_entities",
                        "description": "List all Home Assistant entity ids",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "ha_get_state",
                        "description": "Read one Home Assistant entity's state and attributes",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "entity_id": {"type": "string", "description": "Entity id"},
                            },
                            "required": ["entity_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "ha_list_services",
                        "description": "List callable Home Assistant services as domain.service",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "ha_call_service",
                        "description": "Call a Home Assistant service (e.g. light.turn_on)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "description": "Service domain (e.g. light)",
                                },
                                "service": {
                                    "type": "string",
                                    "description": "Service name (e.g. turn_on)",
                                },
                                "service_data": {
                                    "type": "object",
                                    "description": "Optional service payload",
                                },
                                "entity_id": {
                                    "type": "string",
                                    "description": "Optional target entity id",
                                },
                            },
                            "required": ["domain", "service"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "project_list",
                        "description": "List projects and the active one",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "project_create",
                        "description": "Create a new project workspace",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Project name"},
                            },
                            "required": ["name"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "project_switch",
                        "description": "Switch the active project workspace",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Project name"},
                            },
                            "required": ["name"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_list",
                        "description": "List kanban tasks, optionally filtered by column (backlog/in_progress/done)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "column": {
                                    "type": "string",
                                    "enum": ["backlog", "in_progress", "done"],
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_create",
                        "description": "Create a kanban task card",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "Task title"},
                                "description": {
                                    "type": "string",
                                    "description": "Task description",
                                },
                                "column": {
                                    "type": "string",
                                    "enum": ["backlog", "in_progress", "done"],
                                    "default": "backlog",
                                },
                            },
                            "required": ["title"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_show",
                        "description": "Show a single kanban task's detail",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task id"},
                            },
                            "required": ["task_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_move",
                        "description": "Move a kanban task to another column",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task id"},
                                "column": {
                                    "type": "string",
                                    "enum": ["backlog", "in_progress", "done"],
                                },
                            },
                            "required": ["task_id", "column"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_complete",
                        "description": "Move a kanban task to done",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task id"},
                            },
                            "required": ["task_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_block",
                        "description": "Block a kanban task with an optional reason",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task id"},
                                "reason": {"type": "string", "description": "Block reason"},
                            },
                            "required": ["task_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_unblock",
                        "description": "Unblock a kanban task",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task id"},
                            },
                            "required": ["task_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_comment",
                        "description": "Add a comment to a kanban task",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string", "description": "Task id"},
                                "text": {"type": "string", "description": "Comment text"},
                            },
                            "required": ["task_id", "text"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "delegate_task",
                        "description": "Run one goal in an isolated child agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string", "description": "Goal to complete"},
                                "context": {
                                    "type": "string",
                                    "description": "Optional background context",
                                },
                            },
                            "required": ["goal"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "delegate_tasks",
                        "description": "Run multiple independent tasks in parallel "
                        "using subagents (max 3)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "tasks": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of task descriptions",
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Optional shared context",
                                },
                            },
                            "required": ["tasks"],
                        },
                    },
                },
            ]
        )

        # Add skill tools
        if self.skill_manager:
            for skill in self.skill_manager.get_active_skills():
                schemas.append(skill.get_schema())

        return [
            schema
            for schema in schemas
            if schema.get("function", {}).get("name") not in self.blocked_tools
        ]

    def set_tools(self, tools: List[Dict[str, Any]]):
        """Set available tools"""
        self.tools = tools

    def _apply_compression(self) -> List[Message]:
        """Compress conversation to save token space.

        Returns new compressed message list.
        """
        api_messages = self.get_messages_for_api()
        compressed = compress_messages(api_messages, self.settings.max_tokens)
        new_messages = []
        for msg_dict in compressed:
            role = msg_dict["role"]
            content = msg_dict.get("content", "")
            if role == "system":
                new_messages.append(Message.system(content))
            elif role == "user":
                new_messages.append(Message.user(content))
            elif role == "assistant":
                new_messages.append(Message.assistant(content))
            elif role == "tool":
                new_messages.append(
                    Message.tool(
                        content,
                        msg_dict.get("tool_call_id", ""),
                        msg_dict.get("name", "unknown"),
                    )
                )
        self.messages = new_messages
        logger.info(
            "Compressed conversation: %d -> %d messages", len(api_messages), len(new_messages)
        )
        return new_messages

    def clear_conversation(self):
        """Clear conversation history"""
        self.messages = []
        self.session_id = str(uuid.uuid4())
        self.iteration = 0


def create_mobile_agent(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs,
) -> MobileAgent:
    """Factory function to create a mobile agent with default configuration"""
    settings = get_settings()

    # Import here to avoid circular imports
    from hermes_mobile.memory.provider import MobileMemoryProvider
    from hermes_mobile.skills.manager import MobileSkillManager

    # Initialize memory provider
    memory_provider = MobileMemoryProvider(
        db_path=settings.get_memory_db_path(),
        encrypt=settings.encrypt_memory,
    )

    # Initialize skill manager
    skill_manager = MobileSkillManager(
        skills_dir=settings.get_skills_dir(),
    )

    agent = MobileAgent(
        model=model or settings.default_model,
        provider=provider or settings.default_provider,
        memory_provider=memory_provider,
        skill_manager=skill_manager,
        **kwargs,
    )

    # Set tools from schemas
    agent.set_tools(agent.get_tool_schemas())

    return agent
