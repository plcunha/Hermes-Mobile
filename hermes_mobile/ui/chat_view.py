"""Chat View - Main conversation interface.

Follows the desktop "nous" chat contract: the transcript is flat — assistant
messages render directly on the surface with real markdown, user messages sit
in a subtle tinted bubble with a hairline border, and tool calls are inline
status rows instead of boxed cards.
"""

import asyncio
import inspect
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import flet as ft

from hermes_mobile.core.agent import Message, ToolCall
from hermes_mobile.locales import t
from hermes_mobile.ui.attachments import (
    MAX_ATTACHMENTS_PER_TURN,
    PendingAttachment,
    attachment_from_picker_file,
    attachments_to_prompt_context,
)
from hermes_mobile.ui.common import MONO_FONT, close_dialog, hermes_welcome_art, open_dialog, snack
from hermes_mobile.ui.theme import mode_colors

logger = logging.getLogger(__name__)


class ChatView:
    """Main chat interface"""

    def __init__(self, app):
        self.app = app
        self.page = app.page
        self.agent = app.agent

        # Message storage
        self.messages: List[Message] = []
        self.current_assistant_text = ""
        self.current_tool_calls: List[ToolCall] = []

        # Track tool call rows by call_id for in-place updates
        self._tool_call_rows: Dict[str, ft.Row] = {}
        # Track the currently streaming assistant control
        self._streaming_control: Optional[ft.Text] = None
        self._streaming_container: Optional[ft.Container] = None

        # UI Components
        self.chat_list = ft.ListView(
            expand=True,
            spacing=4,
            padding=ft.Padding.only(left=16, right=16, top=12, bottom=12),
            auto_scroll=True,
        )

        c = mode_colors(self.app.dark_mode)
        self._sending = False
        self.input_field = ft.TextField(
            hint_text=t("chat.input_placeholder"),
            multiline=True,
            min_lines=1,
            max_lines=6,
            on_submit=self._on_send,
            on_change=self._on_draft_change,
            border=ft.InputBorder.NONE,
            filled=False,
            text_size=14,
            text_style=ft.TextStyle(color=c["foreground"]),
            hint_style=ft.TextStyle(color=c["muted_foreground"]),
            content_padding=ft.Padding.only(left=12, right=8, top=10, bottom=6),
        )

        self.mic_button = ft.IconButton(
            icon=ft.Icons.MIC,
            icon_size=16,
            icon_color=c["muted_foreground"],
            tooltip="Transcribe audio file",
            on_click=self._on_voice,
        )
        self.send_button = ft.IconButton(
            icon=ft.Icons.ARROW_UPWARD,
            on_click=self._on_send,
            icon_color=c["background"],
            bgcolor=c["foreground"],
            tooltip=t("chat.send"),
            icon_size=18,
        )

        self.status_text = ft.Text(
            "",
            size=12,
            color=ft.Colors.OUTLINE,
            visible=False,
        )
        self.pending_attachments: List[PendingAttachment] = []
        self.attachments_row = ft.Row([], spacing=6, wrap=True, visible=False)
        self.file_picker = ft.FilePicker()
        self._register_file_picker()

    def _register_file_picker(self) -> None:
        """Register FilePicker using the active Flet line's extension point.

        Flet 0.86 made FilePicker a Service. Adding it to page.overlay renders a
        red-screen `Unknown control: FilePicker` on Android. Flet 0.28 had no
        page.services and still requires the legacy overlay path.
        """
        services = getattr(self.page, "services", None)
        if services is not None:
            if self.file_picker not in services:
                services.append(self.file_picker)
            return
        if hasattr(self.page, "overlay") and self.file_picker not in self.page.overlay:
            self.page.overlay.append(self.file_picker)

    def build(self) -> ft.Control:
        """Build the flat desktop-derived transcript and docked composer."""
        if not self.messages and not self.chat_list.controls:
            self._show_welcome()

        c = mode_colors(self.app.dark_mode)
        remote_mode = bool(getattr(self.app, "remote_mode", False))
        if remote_mode:
            model = str(getattr(self.app, "remote_model", "") or "")
            short_model = model.split("/")[-1] if model else "Hermes Remote"
            state = self.app.remote_client.state if self.app.remote_client else "closed"
            dot_color = {
                "open": c["success"],
                "connecting": ft.Colors.ORANGE,
                "error": ft.Colors.ERROR,
            }.get(state, c["muted_foreground"])
            context_items = [
                ft.PopupMenuItem(
                    icon=ft.Icons.ATTACH_FILE,
                    content="Attach file",
                    on_click=lambda e: asyncio.create_task(self._pick_attachments()),
                ),
                ft.PopupMenuItem(
                    icon=ft.Icons.HISTORY,
                    content=t("sessions.title"),
                    on_click=lambda e: asyncio.create_task(self.app.show_remote_sessions()),
                ),
                ft.PopupMenuItem(
                    icon=ft.Icons.HUB_OUTLINED,
                    content="Connections",
                    on_click=lambda e: self.app._navigate_to("gateway"),
                ),
            ]
            model_destination = "gateway"
        else:
            model = getattr(self.app.settings, "default_model", "")
            short_model = model.split("/")[-1] if model else t("chat.choose_model")
            dot_color = c["success"]
            context_items = [
                ft.PopupMenuItem(
                    icon=ft.Icons.ATTACH_FILE,
                    content="Attach file",
                    on_click=lambda e: asyncio.create_task(self._pick_attachments()),
                ),
                ft.PopupMenuItem(
                    icon=ft.Icons.FOLDER_OPEN,
                    content="Artifacts",
                    on_click=lambda e: self.app._navigate_to("artifacts"),
                ),
                ft.PopupMenuItem(
                    icon=ft.Icons.PSYCHOLOGY_OUTLINED,
                    content="Memory",
                    on_click=lambda e: self.app._navigate_to("memory"),
                ),
                ft.PopupMenuItem(
                    icon=ft.Icons.BUILD_OUTLINED,
                    content="Tools",
                    on_click=lambda e: self.app._navigate_to("tools"),
                ),
            ]
            model_destination = "settings"

        context_menu = ft.PopupMenuButton(
            icon=ft.Icons.ADD,
            icon_color=c["muted_foreground"],
            tooltip=t("chat.add_context"),
            items=context_items,
        )
        model_pill = ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        width=6,
                        height=6,
                        bgcolor=dot_color,
                        border_radius=ft.BorderRadius.all(6),
                    ),
                    ft.Text(
                        short_model,
                        size=10,
                        color=c["muted_foreground"],
                        max_lines=1,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        no_wrap=True,
                    ),
                ],
                spacing=6,
                tight=True,
            ),
            padding=ft.Padding.symmetric(horizontal=9, vertical=5),
            width=120,
            alignment=ft.Alignment.CENTER_LEFT,
            border=ft.Border.all(1, c["border"]),
            border_radius=ft.BorderRadius.all(12),
            on_click=lambda e: (
                self.app._show_model_picker()
                if hasattr(self.app, "_show_model_picker")
                else self.app._navigate_to(model_destination)
            ),
            ink=True,
            tooltip=t("chat.model_settings"),
        )

        composer = ft.Container(
            content=ft.Column(
                [
                    self.input_field,
                    self.attachments_row,
                    self.status_text,
                    ft.Row(
                        [
                            context_menu,
                            model_pill,
                            ft.Container(expand=True),
                            self.mic_button,
                            self.send_button,
                        ],
                        spacing=4,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=0,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            margin=ft.Margin.only(left=6, right=6, top=4, bottom=6),
            padding=ft.Padding.only(left=8, right=8, top=2, bottom=4),
            bgcolor=c["composer"],
            border=ft.Border.all(1, c["composer_border"]),
            border_radius=ft.BorderRadius.all(12),
        )

        return ft.Column(
            [
                ft.Container(content=self.chat_list, expand=True),
                composer,
            ],
            expand=True,
            spacing=0,
        )

    # ------------------------------------------------------------------
    # Welcome / empty states
    # ------------------------------------------------------------------

    def _show_welcome(self):
        """Show the welcome state in the chat list."""
        c = mode_colors(self.app.dark_mode)
        remote_mode = bool(getattr(self.app, "remote_mode", False))
        has_api_key = bool(
            self.app.settings
            and (
                self.app.settings.openrouter_api_key
                or self.app.settings.openai_api_key
                or self.app.settings.anthropic_api_key
                or self.app.settings.gemini_api_key
            )
        )

        if remote_mode:
            state = self.app.remote_client.state if self.app.remote_client else "closed"
            subtitle_key = (
                "chat.remote_ready_help" if state == "open" else "chat.remote_offline_help"
            )
            subtitle = t(subtitle_key)
        elif has_api_key:
            subtitle = t("chat.ready_help")
        else:
            subtitle = t("chat.no_api_key_hint")

        welcome = ft.Container(
            content=ft.Column(
                [
                    hermes_welcome_art(136),
                    ft.Container(height=10),
                    ft.Text(
                        "Hermes",
                        size=27,
                        weight=ft.FontWeight.W_700,
                        color=c["foreground"],
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        t("chat.tagline"),
                        size=13,
                        color=c["foreground"],
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Container(height=8),
                    ft.Text(
                        subtitle,
                        size=12,
                        color=c["muted_foreground"],
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=3,
            ),
            alignment=ft.Alignment.CENTER,
            height=max(360, int((self.page.height or 720) - 260)),
            padding=ft.Padding.symmetric(horizontal=32, vertical=20),
        )

        self.chat_list.controls.append(welcome)

    def refresh_welcome(self):
        """Rebuild the empty state when runtime connectivity changes."""
        if self.messages:
            return
        self.chat_list.controls.clear()
        self._show_welcome()
        self.page.update()

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def _on_draft_change(self, e):
        """Persist draft text for process-restart and session-switch recovery."""
        if hasattr(self.app, "save_current_draft"):
            self.app.save_current_draft(self.input_field.value or "")

    def _on_send(self, e):
        """Submit, queue, or stop depending on composer state.

        Busy + non-empty text queues a follow-up turn. Busy + empty composer
        keeps the Desktop-like Stop behavior.
        """
        text = self.input_field.value
        if self._sending:
            if text and text.strip():
                prompt = self._consume_pending_attachments(text.strip())
                self.input_field.value = ""
                self.page.update()
                asyncio.create_task(self.app.send_message(prompt))
            else:
                asyncio.create_task(self.app.interrupt_turn())
            return
        if text and text.strip():
            prompt = self._consume_pending_attachments(text.strip())
            self.input_field.value = ""
            self.page.update()
            asyncio.create_task(self.app.send_message(prompt))

    def set_busy(self, busy: bool):
        """Synchronize composer affordances with the active agent turn."""
        self._sending = busy
        # Keep the composer editable while Hermes works.  Android users often
        # type a follow-up while the remote session is running; disabling the
        # TextField lets the IME accept text that Flet never commits, which
        # looks like the message vanished.  The send handler already queues
        # non-empty text and treats an empty tap as Stop, so input can stay live.
        self.input_field.disabled = False
        self.send_button.disabled = False
        self.send_button.icon = ft.Icons.STOP_ROUNDED if busy else ft.Icons.ARROW_UPWARD
        self.send_button.tooltip = "Queue message (empty = Stop)" if busy else t("chat.send")
        pet = getattr(self.app, "pet_view", None)
        if pet is not None:
            pet.set_activity("run" if busy else "idle")
        self.page.update()

    def set_status(self, text: str):
        """Show the backend's current operational state below the composer."""
        value = str(text or "").strip()
        self.status_text.value = value
        self.status_text.visible = bool(value)
        self.page.update()

    def show_remote_request(self, event):
        """Resolve a blocking Hermes remote prompt without stalling the agent."""
        client = self.app.remote_client
        if client is None:
            return
        payload = event.payload
        request_id = str(payload.get("request_id") or "")
        title = "Hermes needs input"
        content_controls: list[ft.Control] = []
        actions: list[ft.Control] = []

        async def submit(coro, dialog):
            try:
                await coro
                close_dialog(self.page, dialog)
            except Exception as exc:
                snack(self.page, str(exc), error=True)

        if event.type == "approval.request":
            title = "Approve remote command?"
            command = str(payload.get("command") or payload.get("description") or "")
            description = str(payload.get("description") or "")
            content_controls = [
                ft.Text(description, size=13)
                if description and description != command
                else ft.Container(),
                ft.Container(
                    content=ft.Text(command, selectable=True, font_family=MONO_FONT, size=12),
                    padding=ft.Padding.all(10),
                    bgcolor=mode_colors(self.app.dark_mode)["muted"],
                    border_radius=ft.BorderRadius.all(8),
                ),
            ]
            dialog = ft.AlertDialog(modal=True)

            def respond(choice):
                return lambda e: asyncio.create_task(
                    submit(client.respond_approval(choice), dialog)
                )

            actions = [
                ft.TextButton("Deny", on_click=respond("deny")),
                ft.TextButton("Allow once", on_click=respond("once")),
            ]
            if payload.get("allow_permanent", payload.get("allowPermanent", True)):
                actions.append(ft.TextButton("Always allow", on_click=respond("always")))
        elif event.type == "clarify.request":
            title = str(payload.get("question") or "Hermes needs clarification")
            choices = [str(item) for item in payload.get("choices") or []]
            multi_select = bool(payload.get("multi_select"))
            answer_field = ft.TextField(
                hint_text="Your answer",
                multiline=not choices,
                min_lines=1,
                max_lines=4,
                autofocus=True,
            )
            selected: dict[str, ft.Checkbox] = {}
            radio = None
            if choices and multi_select:
                selected = {choice: ft.Checkbox(label=choice) for choice in choices}
                content_controls = list(selected.values())
            elif choices:
                radio = ft.RadioGroup(
                    content=ft.Column([ft.Radio(value=choice, label=choice) for choice in choices])
                )
                content_controls = [radio]
            else:
                content_controls = [answer_field]
            dialog = ft.AlertDialog(modal=True)

            def answer_value():
                if selected:
                    return [choice for choice, control in selected.items() if control.value]
                if radio is not None:
                    return radio.value
                return answer_field.value or ""

            actions = [
                ft.TextButton("Cancel", on_click=lambda e: close_dialog(self.page, dialog)),
                ft.TextButton(
                    "Send",
                    on_click=lambda e: asyncio.create_task(
                        submit(client.respond_clarify(request_id, answer_value()), dialog)
                    ),
                ),
            ]
        else:
            is_sudo = event.type == "sudo.request"
            title = "Sudo password" if is_sudo else str(payload.get("prompt") or "Secret required")
            secret_field = ft.TextField(
                hint_text="Password" if is_sudo else "Secret value",
                password=True,
                can_reveal_password=True,
                autofocus=True,
            )
            content_controls = [
                ft.Text(
                    "Sent only to the connected Hermes backend and never saved on this device.",
                    size=12,
                    color=mode_colors(self.app.dark_mode)["muted_foreground"],
                ),
                secret_field,
            ]
            dialog = ft.AlertDialog(modal=True)

            def send_secret():
                value = secret_field.value or ""
                if is_sudo:
                    return client.respond_sudo(request_id, value)
                return client.respond_secret(request_id, value)

            actions = [
                ft.TextButton("Cancel", on_click=lambda e: close_dialog(self.page, dialog)),
                ft.TextButton(
                    "Send",
                    on_click=lambda e: asyncio.create_task(submit(send_secret(), dialog)),
                ),
            ]

        dialog.title = ft.Text(title)
        dialog.content = ft.Container(
            content=ft.Column(content_controls, tight=True, scroll=ft.ScrollMode.AUTO),
            width=420,
        )
        dialog.actions = actions
        open_dialog(self.page, dialog)

    # ------------------------------------------------------------------
    # Message lifecycle
    # ------------------------------------------------------------------

    def add_user_message(self, text: str):
        """Add a user message to the chat"""
        first_message = not self.messages
        message = Message.user(text)
        self.messages.append(message)
        if first_message:
            self.chat_list.controls.clear()
        self._add_message_bubble(message)
        self._scroll_to_bottom()

    def append_assistant_message(self, chunk: str):
        """Append a chunk to the current assistant message (streaming)"""
        if self._streaming_control is None:
            self._start_streaming()
        stream = self._streaming_control
        if stream is None:
            return
        self.current_assistant_text += chunk
        stream.value = self.current_assistant_text
        self.page.update()

    def finalize_assistant_message(self):
        """Finalize the current assistant message — swap plain text for markdown"""
        if self._streaming_container is None:
            return
        text = self.current_assistant_text.strip()
        if text:
            message = Message.assistant(text, list(self.current_tool_calls))
            self.messages.append(message)
            self._streaming_container.content = self._build_markdown(text)
        else:
            # Empty response (e.g. no API key): drop the container
            if self._streaming_container in self.chat_list.controls:
                self.chat_list.controls.remove(self._streaming_container)
        self._streaming_container = None
        self._streaming_control = None
        self.current_assistant_text = ""
        self.current_tool_calls = []
        self._scroll_to_bottom()
        self.page.update()

    def _start_streaming(self):
        """Insert the streaming assistant container (plain text while streaming)."""
        c = mode_colors(self.app.dark_mode)
        self._streaming_control = ft.Text(
            "",
            selectable=True,
            size=14,
            color=c["foreground"],
        )
        self._streaming_container = ft.Container(
            content=self._streaming_control,
            padding=ft.Padding.only(left=2, right=2, top=6, bottom=6),
        )
        self.chat_list.controls.append(self._streaming_container)
        self._scroll_to_bottom()

    def _copy_to_clipboard(self, text: str) -> None:
        self.page.set_clipboard(text)
        snack(self.page, "Copied")

    def _attachment_storage_dir(self) -> Path:
        settings = getattr(self.app, "settings", None)
        if settings is not None and hasattr(settings, "get_data_dir"):
            return Path(settings.get_data_dir())
        return Path(tempfile.gettempdir()) / "hermes-mobile"

    def _attachment_label(self, item: PendingAttachment) -> str:
        suffix = "path" if item.local_path else "inline"
        return f"{item.name} · {item.byte_count // 1024 or 1} KB · {suffix}"

    def _refresh_attachments_row(self) -> None:
        c = mode_colors(self.app.dark_mode)
        self.attachments_row.controls.clear()
        for item in self.pending_attachments:
            self.attachments_row.controls.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(
                                ft.Icons.IMAGE_OUTLINED
                                if item.kind == "image"
                                else ft.Icons.DESCRIPTION_OUTLINED,
                                size=13,
                                color=c["muted_foreground"],
                            ),
                            ft.Text(
                                self._attachment_label(item),
                                size=11,
                                color=c["muted_foreground"],
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.CLOSE,
                                icon_size=12,
                                padding=ft.Padding.all(0),
                                tooltip="Remove attachment",
                                on_click=lambda e, aid=item.id: self._remove_attachment(aid),
                            ),
                        ],
                        spacing=3,
                        tight=True,
                    ),
                    padding=ft.Padding.only(left=8, right=2, top=3, bottom=3),
                    border=ft.Border.all(1, c["border"]),
                    border_radius=ft.BorderRadius.all(999),
                )
            )
        self.attachments_row.visible = bool(self.pending_attachments)

    def _remove_attachment(self, attachment_id: str) -> None:
        self.pending_attachments = [a for a in self.pending_attachments if a.id != attachment_id]
        self._refresh_attachments_row()
        self.page.update()

    async def _pick_attachments(self) -> None:
        try:
            selected = await self.file_picker.pick_files(
                dialog_title="Attach files to this turn",
                allow_multiple=True,
                with_data=True,
            )
        except Exception as exc:
            snack(self.page, f"Attachment picker failed: {exc}", error=True)
            return
        if not selected:
            return
        added = 0
        for file in selected:
            if len(self.pending_attachments) >= MAX_ATTACHMENTS_PER_TURN:
                snack(
                    self.page,
                    f"Attachment limit is {MAX_ATTACHMENTS_PER_TURN} per turn",
                    error=True,
                )
                break
            try:
                attachment = attachment_from_picker_file(file, self._attachment_storage_dir())
                if bool(getattr(self.app, "remote_mode", False)) and attachment.kind != "text":
                    if attachment.local_path:
                        try:
                            Path(attachment.local_path).unlink(missing_ok=True)
                        except OSError:
                            pass
                    snack(
                        self.page,
                        f"Remote attachments currently support text only: {attachment.name}",
                        error=True,
                    )
                    continue
                self.pending_attachments.append(attachment)
                added += 1
            except Exception as exc:
                name = getattr(file, "name", "attachment")
                snack(self.page, f"Could not attach {name}: {exc}", error=True)
        if added:
            self._refresh_attachments_row()
            snack(self.page, f"Attached {added} file{'s' if added != 1 else ''}")
            self.page.update()

    def _consume_pending_attachments(self, text: str) -> str:
        if not self.pending_attachments:
            return text
        context = attachments_to_prompt_context(self.pending_attachments)
        self.pending_attachments = []
        self._refresh_attachments_row()
        return f"{context}\n\nUser message:\n{text}"

    def _on_voice(self, e) -> None:
        asyncio.create_task(self._pick_voice_audio())

    async def _pick_voice_audio(self) -> None:
        """Pick an audio file and transcribe it into the composer.

        This is intentionally not a fake microphone. Current Flet Python APIs expose
        FilePicker but not a native recorder control, so press-to-talk remains a
        native/bridge task.
        """
        if bool(getattr(self.app, "remote_mode", False)):
            snack(
                self.page,
                "Remote voice needs backend /api/audio/transcribe support; use text or attach a transcript for now.",
                error=True,
            )
            return
        agent = getattr(self.app, "agent", None)
        if agent is None or not hasattr(agent, "transcribe_audio_file"):
            snack(self.page, "Local audio transcription is not available", error=True)
            return
        try:
            selected = await self.file_picker.pick_files(
                dialog_title="Choose an audio file to transcribe",
                file_type=ft.FilePickerFileType.ANY,
                allowed_extensions=["wav", "mp3", "m4a", "aac", "ogg", "oga", "webm", "flac"],
                allow_multiple=False,
                with_data=True,
            )
        except Exception as exc:
            snack(self.page, f"Audio picker failed: {exc}", error=True)
            return
        if not selected:
            return
        try:
            attachment = attachment_from_picker_file(selected[0], self._attachment_storage_dir())
            suffix = Path(attachment.name).suffix.lower().lstrip(".")
            if not (
                attachment.mime_type.startswith("audio/")
                or suffix in {"wav", "mp3", "m4a", "aac", "ogg", "oga", "webm", "flac"}
            ):
                raise ValueError(f"Not an audio file: {attachment.name}")
            if not attachment.local_path:
                raise ValueError(f"Audio file could not be staged: {attachment.name}")
            snack(self.page, "Transcribing audio…")
            transcript = await agent.transcribe_audio_file(Path(attachment.local_path))
        except Exception as exc:
            snack(self.page, f"Audio transcription failed: {exc}", error=True)
            return
        existing = str(self.input_field.value or "").strip()
        self.input_field.value = f"{existing}\n{transcript}".strip() if existing else transcript
        self._on_draft_change(None)
        snack(self.page, "Audio transcribed into composer")
        self.page.update()

    def _render_messages(self) -> None:
        """Re-render the chat list from self.messages (used by /undo, /retry)."""
        self.chat_list.controls.clear()
        for msg in self.messages:
            self._add_message_bubble(msg)
        self._scroll_to_bottom()

    def _build_markdown(self, text: str) -> ft.Control:
        """Render an assistant message as markdown."""
        c = mode_colors(self.app.dark_mode)
        code_theme = (
            ft.MarkdownCodeTheme.ATELIER_CAVE_DARK
            if self.app.dark_mode
            else ft.MarkdownCodeTheme.ATELIER_CAVE_LIGHT
        )
        fg = ft.TextStyle(size=14, color=c["foreground"], height=1.45)
        return ft.Markdown(
            value=text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            code_theme=code_theme,
            soft_line_break=True,
            md_style_sheet=ft.MarkdownStyleSheet(
                p_text_style=fg,
                strong_text_style=ft.TextStyle(
                    size=14, weight=ft.FontWeight.W_600, color=c["foreground"]
                ),
                em_text_style=ft.TextStyle(size=14, italic=True, color=c["foreground"]),
                h1_text_style=ft.TextStyle(
                    size=20, weight=ft.FontWeight.W_700, color=c["foreground"]
                ),
                h2_text_style=ft.TextStyle(
                    size=17, weight=ft.FontWeight.W_700, color=c["foreground"]
                ),
                h3_text_style=ft.TextStyle(
                    size=15, weight=ft.FontWeight.W_600, color=c["foreground"]
                ),
                code_text_style=ft.TextStyle(
                    size=13, font_family="monospace", color=ft.Colors.PRIMARY
                ),
                codeblock_padding=ft.Padding.all(10),
                blockquote_text_style=ft.TextStyle(
                    size=14, color=c["muted_foreground"], italic=True
                ),
                list_bullet_text_style=ft.TextStyle(size=14, color=c["foreground"]),
                a_text_style=ft.TextStyle(size=14, color=ft.Colors.PRIMARY),
            ),
        )

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    def on_tool_call(self, tool_call: ToolCall):
        """Handle tool call from agent"""
        self.current_tool_calls.append(tool_call)
        self._show_tool_call(tool_call)

    def on_tool_result(self, tool_call: ToolCall):
        """Handle tool result from agent"""
        self._update_tool_call(tool_call)

    def on_message(self, message: Message):
        """Handle new message from agent"""
        pass

    def _show_tool_call(self, tool_call: ToolCall):
        """Show a tool call as a flat inline status row."""
        c = mode_colors(self.app.dark_mode)
        status_text = ft.Text(
            tool_call.name,
            size=12,
            weight=ft.FontWeight.W_500,
            color=c["muted_foreground"],
            font_family=MONO_FONT,
        )
        spinner = ft.ProgressRing(width=14, height=14, stroke_width=2)

        row = ft.Row(
            [
                ft.Icon(ft.Icons.TERMINAL, size=14, color=c["muted_foreground"]),
                ft.Container(width=6),
                status_text,
                ft.Container(expand=True),
                spinner,
            ],
            spacing=0,
        )
        self._tool_call_rows[tool_call.call_id] = row

        container = ft.Container(
            content=row,
            padding=ft.Padding.only(left=2, right=2, top=4, bottom=4),
        )
        self.chat_list.controls.append(container)
        self.page.update()

    def _update_tool_call(self, tool_call: ToolCall):
        """Update a tool call row with its result."""
        row = self._tool_call_rows.get(tool_call.call_id)
        if row is None:
            return

        c = mode_colors(self.app.dark_mode)
        status_text = row.controls[2]
        if isinstance(status_text, ft.Text):
            if tool_call.error:
                status_text.value = f"{tool_call.name} — failed"
                status_text.color = ft.Colors.ERROR
            else:
                status_text.value = f"{tool_call.name} ✓"
                status_text.color = c["muted_foreground"]

        # Replace the spinner with a status icon
        row.controls[-1] = ft.Icon(
            ft.Icons.CHECK_CIRCLE_OUTLINE if not tool_call.error else ft.Icons.ERROR_OUTLINE,
            size=15,
            color=ft.Colors.PRIMARY if not tool_call.error else ft.Colors.ERROR,
        )

        # Append a compact result preview below the row
        if tool_call.result is not None and not tool_call.error:
            preview = str(tool_call.result)
            if len(preview) > 120:
                preview = preview[:120] + "…"
            parent = row.parent
            if parent is not None:
                preview_text = ft.Text(
                    preview,
                    size=11,
                    font_family="monospace",
                    color=c["muted_foreground"],
                    selectable=True,
                )
                # Keep it inside the same container: wrap row + preview
                if isinstance(parent, ft.Container):
                    parent.content = ft.Column(
                        [row, preview_text],
                        spacing=2,
                        horizontal_alignment=ft.CrossAxisAlignment.START,
                    )
        self.page.update()

    def _add_message_bubble(self, message: Message):
        """Add a message bubble to the chat list."""
        c = mode_colors(self.app.dark_mode)
        is_user = message.role == "user"

        if is_user:
            bubble = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            message.content,
                            selectable=True,
                            size=14,
                            color=c["foreground"],
                        ),
                        ft.Row(
                            [
                                ft.Text(
                                    message.timestamp.strftime("%H:%M"),
                                    size=10,
                                    color=c["muted_foreground"],
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ],
                    spacing=3,
                ),
                padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                border_radius=ft.BorderRadius.all(10),
                bgcolor=c["user_bubble"],
                border=ft.Border.all(1, c["user_bubble_border"]),
                margin=ft.Margin.only(left=48, right=0),
            )
            self.chat_list.controls.append(bubble)
        else:
            container = ft.Container(
                content=self._build_markdown(message.content),
                padding=ft.Padding.only(left=2, right=2, top=6, bottom=6),
            )
            self.chat_list.controls.append(container)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _scroll_to_bottom(self, delay: float = 0):
        """Scroll chat to bottom"""
        if delay > 0:

            async def scroll_after_layout():
                await asyncio.sleep(delay)
                result = self.chat_list.scroll_to(offset=-1, duration=0)
                if inspect.isawaitable(result):
                    await result

            try:
                asyncio.get_running_loop().create_task(scroll_after_layout())
            except RuntimeError:
                pass
            return
        try:
            result = self.chat_list.scroll_to(offset=-1, duration=120)
            if inspect.isawaitable(result):
                try:
                    asyncio.get_running_loop().create_task(result)
                except RuntimeError:
                    # Structural tests build controls without Flet's event loop.
                    result.close()
        except Exception:
            pass

    def load_remote_history(self, messages):
        """Hydrate a resumed Desktop session into the mobile transcript."""
        self.clear_chat(show_welcome=False)
        for item in messages or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "assistant")
            content = str(item.get("content") or item.get("text") or "")
            if not content or role not in {"user", "assistant"}:
                continue
            message = Message(role=role, content=content)
            self.messages.append(message)
            self._add_message_bubble(message)
        if not self.messages:
            self._show_welcome()
        self.page.update()
        # Markdown controls report their final height only after the first
        # layout. Scrolling before that can land beyond the measured transcript
        # and show an apparently empty resumed session until the user swipes.
        self._scroll_to_bottom(delay=0.12)

    def clear_chat(self, show_welcome: bool = True):
        """Start a clean session in both the UI and agent runtime."""
        self.chat_list.controls.clear()
        self.messages.clear()
        self.current_assistant_text = ""
        self.current_tool_calls = []
        self._tool_call_rows.clear()
        self._streaming_container = None
        self._streaming_control = None
        self._sending = False
        self.input_field.disabled = False
        self.send_button.disabled = False
        self.send_button.icon = ft.Icons.ARROW_UPWARD
        if self.agent:
            self.agent.clear_conversation()
        if show_welcome:
            self._show_welcome()
        self.page.update()
