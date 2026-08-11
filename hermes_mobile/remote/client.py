"""Hermes Desktop-compatible remote backend client.

The mobile app talks to the same ``/api/ws`` JSON-RPC gateway used by Hermes
Desktop.  REST is used for probing/authentication/session listings; live agent
turns always use the canonical WebSocket event stream.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import connect as websocket_connect


class RemoteHermesError(RuntimeError):
    """Base error for remote backend failures."""


class RemoteAuthenticationError(RemoteHermesError):
    """Authentication is missing, expired, or rejected."""


class RemoteConnectionError(RemoteHermesError):
    """The backend could not be reached or its transport closed."""


class RemoteProtocolError(RemoteHermesError):
    """The backend returned an invalid or unsupported protocol response."""


@dataclass(frozen=True)
class RemoteStatus:
    version: str = ""
    auth_required: bool = False
    auth_providers: tuple[str, ...] = ()
    overall: str = "unknown"
    profiles: tuple[str, ...] = ()
    gateway_running: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class RemoteEvent:
    type: str
    payload: Mapping[str, Any]
    session_id: str | None = None


EventHandler = Callable[[RemoteEvent], Optional[Awaitable[None]]]
StateHandler = Callable[[str], Optional[Awaitable[None]]]
WebSocketFactory = Callable[..., Awaitable[Any]]


def normalize_remote_base_url(raw_url: str) -> str:
    """Normalize a Desktop-compatible remote URL without losing path prefixes."""
    value = str(raw_url or "").strip()
    if not value:
        raise ValueError("Remote gateway URL is required")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError(f"Remote gateway URL is invalid: {exc}") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Remote gateway URL must use http:// or https://")
    if not parsed.hostname:
        raise ValueError("Remote gateway URL requires a host")
    if parsed.username or parsed.password:
        raise ValueError("Credentials must not be embedded in the remote URL")
    if parsed.query or parsed.fragment:
        raise ValueError("Remote gateway URL must not contain a query or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Remote gateway URL has an invalid port: {exc}") from exc
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    authority = f"{host}:{port}" if port is not None else host
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, authority, path, "", ""))


def insecure_transport_is_private(base_url: str) -> bool:
    """Allow plain HTTP automatically only on loopback/private/Tailscale networks."""
    parsed = urlsplit(normalize_remote_base_url(base_url))
    if parsed.scheme == "https":
        return True
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    tailscale = ipaddress.ip_network("100.64.0.0/10")
    return address.is_loopback or address.is_private or address in tailscale


def build_gateway_ws_url(base_url: str, *, token: str = "", ticket: str = "") -> str:
    """Build the canonical ``/api/ws`` URL used by Hermes Desktop."""
    parsed = urlsplit(normalize_remote_base_url(base_url))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/api/ws"
    if ticket:
        query = f"ticket={quote(ticket, safe='')}"
    elif token:
        query = f"token={quote(token, safe='')}"
    else:
        query = ""
    return urlunsplit((scheme, parsed.netloc, path, query, ""))


def redact_transport_error(error: BaseException | str, *secrets: str) -> str:
    """Prevent credentials or one-time WS tickets from leaking into UI/log errors."""
    text = str(error)
    text = re.sub(
        r"(?i)(ticket|token)=([^\s&#]+)",
        lambda match: f"{match.group(1)}=<redacted>",
        text,
    )
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1<redacted>", text)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "<redacted>")
    return text


class RemoteHermesClient:
    """Async JSON-RPC client for ``hermes serve`` / dashboard backends."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_mode: str = "auto",
        token: str = "",
        username: str = "",
        password: str = "",
        profile: str = "",
        timeout: float = 120.0,
        allow_insecure: bool = False,
        on_event: EventHandler | None = None,
        on_state: StateHandler | None = None,
        http_client: httpx.AsyncClient | None = None,
        websocket_factory: WebSocketFactory | None = None,
    ) -> None:
        self.base_url = normalize_remote_base_url(base_url)
        self.auth_mode = auth_mode if auth_mode in {"auto", "basic", "token"} else "auto"
        self.token = str(token or "")
        self.username = str(username or "")
        self.password = str(password or "")
        self.profile = str(profile or "").strip()
        self.timeout = max(1.0, float(timeout))
        self.allow_insecure = bool(allow_insecure)
        self.on_event = on_event
        self.on_state = on_state
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=self.timeout, follow_redirects=False)
        self._websocket_factory = websocket_factory or websocket_connect
        self._ws: Any | None = None
        self._receiver_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._send_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._closed = False
        self.state = "idle"
        self.status: RemoteStatus | None = None
        self.session_id: str | None = None
        self.stored_session_id: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _token_headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        if self.auth_mode == "token":
            return {"X-Hermes-Session-Token": self.token}
        return {"Authorization": f"Bearer {self.token}"}

    async def _notify(self, callback: EventHandler | StateHandler | None, value: Any) -> None:
        if callback is None:
            return
        result = callback(value)
        if inspect.isawaitable(result):
            await result

    async def _set_state(self, state: str) -> None:
        if self.state == state:
            return
        self.state = state
        await self._notify(self.on_state, state)

    async def probe(self) -> RemoteStatus:
        """Fetch the public health/auth advertisement from the backend."""
        try:
            response = await self._http.get(self._url("/api/status"), headers=self._token_headers())
        except httpx.HTTPError as exc:
            raise RemoteConnectionError(f"Could not reach remote backend: {exc}") from exc
        if response.status_code >= 400:
            raise RemoteConnectionError(f"Remote status probe failed: HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise RemoteProtocolError("Remote status endpoint did not return JSON") from exc
        if not isinstance(body, dict):
            raise RemoteProtocolError("Remote status response must be an object")
        status = RemoteStatus(
            version=str(body.get("version") or ""),
            auth_required=bool(body.get("auth_required")),
            auth_providers=tuple(str(v) for v in body.get("auth_providers") or ()),
            overall=str(body.get("overall") or "unknown"),
            profiles=tuple(str(v) for v in body.get("profiles") or ()),
            gateway_running=bool(body.get("gateway_running")),
            raw=body,
        )
        self.status = status
        return status

    async def _authenticate(self, status: RemoteStatus) -> str:
        if not status.auth_required:
            return build_gateway_ws_url(self.base_url, token=self.token)

        if self.auth_mode == "basic" or (
            self.auth_mode == "auto"
            and self.username
            and self.password
            and "basic" in status.auth_providers
        ):
            if not self.username or not self.password:
                raise RemoteAuthenticationError("Username and password are required")
            try:
                login = await self._http.post(
                    self._url("/auth/password-login"),
                    json={
                        "provider": "basic",
                        "username": self.username,
                        "password": self.password,
                        "next": "/",
                    },
                )
            except httpx.HTTPError as exc:
                raise RemoteConnectionError(f"Remote sign-in failed: {exc}") from exc
            if login.status_code in {401, 403}:
                raise RemoteAuthenticationError("Remote username or password was rejected")
            if login.status_code >= 400:
                raise RemoteAuthenticationError(f"Remote sign-in failed: HTTP {login.status_code}")
        elif not self.token:
            providers = ", ".join(status.auth_providers) or "none advertised"
            raise RemoteAuthenticationError(
                f"Remote backend requires authentication ({providers}); "
                "configure basic credentials or a bearer token"
            )

        try:
            ticket_response = await self._http.post(
                self._url("/api/auth/ws-ticket"), headers=self._token_headers()
            )
        except httpx.HTTPError as exc:
            raise RemoteConnectionError(f"Could not mint WebSocket ticket: {exc}") from exc
        if ticket_response.status_code in {401, 403}:
            raise RemoteAuthenticationError(
                "Remote session was rejected while minting a WebSocket ticket"
            )
        if ticket_response.status_code >= 400:
            raise RemoteConnectionError(
                f"WebSocket ticket request failed: HTTP {ticket_response.status_code}"
            )
        try:
            ticket = str(ticket_response.json().get("ticket") or "")
        except (ValueError, AttributeError) as exc:
            raise RemoteProtocolError("WebSocket ticket response was invalid") from exc
        if not ticket:
            raise RemoteProtocolError("WebSocket ticket response omitted the ticket")
        return build_gateway_ws_url(self.base_url, ticket=ticket)

    async def connect(self) -> RemoteStatus:
        """Probe, authenticate, open the canonical WebSocket, and await gateway.ready."""
        if self.state == "open" and self._ws is not None:
            return self.status or await self.probe()
        if urlsplit(self.base_url).scheme == "http" and not (
            self.allow_insecure or insecure_transport_is_private(self.base_url)
        ):
            raise RemoteConnectionError(
                "Plain HTTP is blocked for public remote hosts; use HTTPS, "
                "a private/VPN address, or explicitly allow insecure transport"
            )
        self._closed = False
        self._ready.clear()
        await self._set_state("connecting")
        try:
            status = await self.probe()
            ws_url = await self._authenticate(status)
            self._ws = await self._websocket_factory(
                ws_url,
                open_timeout=min(self.timeout, 15.0),
                close_timeout=5.0,
                max_size=None,
                ping_interval=20.0,
                ping_timeout=20.0,
            )
            self._receiver_task = asyncio.create_task(self._receive_loop())
            await asyncio.wait_for(self._ready.wait(), timeout=min(self.timeout, 15.0))
        except (RemoteHermesError, asyncio.TimeoutError) as exc:
            await self._set_state("error")
            await self._close_transport()
            if isinstance(exc, asyncio.TimeoutError):
                raise RemoteConnectionError("Remote gateway did not become ready in time") from exc
            raise
        except Exception as exc:
            await self._set_state("error")
            await self._close_transport()
            safe_error = redact_transport_error(exc, self.token, self.password)
            raise RemoteConnectionError(f"Could not open remote gateway: {safe_error}") from exc
        await self._set_state("open")
        return status

    async def _receive_loop(self) -> None:
        error: BaseException | None = None
        ws = self._ws
        if ws is None:
            return
        try:
            async for raw in ws:
                try:
                    frame = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(frame, dict):
                    continue
                request_id = frame.get("id")
                if request_id is not None:
                    future = self._pending.pop(request_id, None)
                    if future is None or future.done():
                        continue
                    if isinstance(frame.get("error"), dict):
                        future.set_exception(
                            RemoteProtocolError(
                                str(frame["error"].get("message") or "Remote RPC failed")
                            )
                        )
                    else:
                        future.set_result(frame.get("result"))
                    continue
                if frame.get("method") != "event" or not isinstance(frame.get("params"), dict):
                    continue
                params = frame["params"]
                event_type = str(params.get("type") or "")
                if not event_type:
                    continue
                payload = params.get("payload")
                event = RemoteEvent(
                    type=event_type,
                    payload=payload if isinstance(payload, dict) else {},
                    session_id=str(params.get("session_id")) if params.get("session_id") else None,
                )
                if event.type == "gateway.ready":
                    self._ready.set()
                await self._notify(self.on_event, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        finally:
            self._ready.clear()
            if not self._closed:
                await self._set_state("error" if error else "closed")
            self._reject_pending(
                RemoteConnectionError(
                    f"Remote gateway connection closed{f': {error}' if error else ''}"
                )
            )

    def _reject_pending(self, error: BaseException) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None, *, timeout: float | None = None
    ) -> Any:
        if self.state != "open" or self._ws is None:
            raise RemoteConnectionError("Remote gateway is not connected")
        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        frame = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": str(method),
            "params": dict(params or {}),
        }
        try:
            async with self._send_lock:
                await self._ws.send(json.dumps(frame, ensure_ascii=False))
            return await asyncio.wait_for(future, timeout=timeout or self.timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise RemoteConnectionError(f"Remote request timed out: {method}") from exc
        except Exception:
            self._pending.pop(request_id, None)
            raise

    async def create_session(self) -> Mapping[str, Any]:
        params: dict[str, Any] = {"cols": 52, "source": "mobile"}
        if self.profile:
            params["profile"] = self.profile
        result = await self.request("session.create", params)
        if not isinstance(result, dict) or not result.get("session_id"):
            raise RemoteProtocolError("session.create returned no session id")
        self.session_id = str(result["session_id"])
        self.stored_session_id = str(result.get("stored_session_id") or "") or None
        return result

    async def resume_session(self, stored_session_id: str) -> Mapping[str, Any]:
        params: dict[str, Any] = {
            "session_id": str(stored_session_id),
            "cols": 52,
            "source": "mobile",
        }
        if self.profile:
            params["profile"] = self.profile
        result, rest_messages = await asyncio.gather(
            self.request("session.resume", params),
            self.get_session_messages(stored_session_id),
            return_exceptions=True,
        )
        if isinstance(result, BaseException):
            raise result
        if not isinstance(result, dict) or not result.get("session_id"):
            raise RemoteProtocolError("session.resume returned no live session id")
        if not result.get("messages") and isinstance(rest_messages, list) and rest_messages:
            result = {**result, "messages": rest_messages}
        self.session_id = str(result["session_id"])
        self.stored_session_id = str(
            result.get("session_key") or result.get("resumed") or stored_session_id
        )
        return result

    async def get_session_messages(self, stored_session_id: str) -> list[Mapping[str, Any]]:
        """Fetch the durable transcript using the same REST fallback as Desktop."""
        session_id = quote(str(stored_session_id), safe="")
        try:
            response = await self._http.get(
                self._url(self._profile_api_path(f"/api/sessions/{session_id}/messages")),
                headers=self._token_headers(),
            )
        except httpx.HTTPError:
            return []
        if response.status_code >= 400:
            return []
        try:
            payload = response.json()
        except ValueError:
            return []
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list) and isinstance(payload, dict):
            messages = payload.get("data")
        if not isinstance(messages, list):
            return []
        return [item for item in messages if isinstance(item, dict)]

    def _profile_api_path(self, path: str) -> str:
        """Scope REST endpoints using the backend's canonical profile prefix."""
        clean_path = f"/{str(path).lstrip('/')}"
        profile = str(self.profile or "").strip()
        if not profile or profile == "default":
            return clean_path
        return f"/p/{quote(profile, safe='')}{clean_path}"

    async def _session_rest_request(
        self,
        method: str,
        stored_session_id: str,
        *,
        suffix: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Call the authenticated Desktop session REST API with safe errors."""
        session_id = str(stored_session_id or "").strip()
        if not session_id:
            raise ValueError("stored_session_id is required")
        path = self._profile_api_path(f"/api/sessions/{quote(session_id, safe='')}{suffix}")
        kwargs: dict[str, Any] = {"headers": self._token_headers()}
        if body is not None:
            kwargs["json"] = dict(body)
        try:
            response = await self._http.request(method, self._url(path), **kwargs)
        except httpx.HTTPError as exc:
            safe_error = redact_transport_error(exc, self.token, self.password)
            raise RemoteConnectionError(f"Remote session request failed: {safe_error}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            if response.status_code >= 400:
                raise RemoteProtocolError(
                    f"Remote session request failed: HTTP {response.status_code}"
                ) from exc
            raise RemoteProtocolError("Remote session endpoint did not return JSON") from exc
        if response.status_code >= 400:
            message = ""
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or "")
                elif error:
                    message = str(error)
                if not message:
                    message = str(payload.get("detail") or payload.get("message") or "")
            safe_message = redact_transport_error(message, self.token, self.password)
            detail = safe_message or f"HTTP {response.status_code}"
            raise RemoteProtocolError(f"Remote session request failed: {detail}")
        if not isinstance(payload, dict):
            raise RemoteProtocolError("Remote session response must be an object")
        return payload

    async def rename_session(self, stored_session_id: str, title: str) -> Mapping[str, Any]:
        clean_title = str(title or "").strip()
        if not clean_title:
            raise ValueError("Session title cannot be empty")
        payload = await self._session_rest_request(
            "PATCH",
            stored_session_id,
            body={"title": clean_title},
        )
        session = payload.get("session")
        if not isinstance(session, dict):
            raise RemoteProtocolError("Rename response omitted the session")
        return session

    async def branch_active_session(self, *, title: str = "") -> Mapping[str, Any]:
        """Branch the connected live session without resuming it a second time."""
        live_session_id = str(self.session_id or "").strip()
        if not live_session_id:
            raise RemoteConnectionError("There is no active remote session to branch")
        params: dict[str, Any] = {"session_id": live_session_id}
        clean_title = str(title or "").strip()
        if clean_title:
            params["name"] = clean_title
        result = await self.request("session.branch", params)
        if not isinstance(result, dict):
            raise RemoteProtocolError("session.branch response must be an object")
        branch_session_id = str(result.get("session_id") or "").strip()
        stored_session_id = str(result.get("stored_session_id") or "").strip()
        if not branch_session_id or not stored_session_id:
            raise RemoteProtocolError("session.branch returned incomplete session identifiers")
        self.session_id = branch_session_id
        self.stored_session_id = stored_session_id
        return result

    async def delete_session(self, stored_session_id: str) -> bool:
        payload = await self._session_rest_request("DELETE", stored_session_id)
        return bool(payload.get("deleted"))

    async def fork_session(self, stored_session_id: str, *, title: str = "") -> Mapping[str, Any]:
        body = {"title": str(title).strip()} if str(title).strip() else {}
        payload = await self._session_rest_request(
            "POST",
            stored_session_id,
            suffix="/fork",
            body=body,
        )
        session = payload.get("session")
        if not isinstance(session, dict):
            raise RemoteProtocolError("Branch response omitted the session")
        return session

    async def submit_prompt(self, text: str) -> Mapping[str, Any]:
        """Submit a prompt. Resumes the stored session if one is set but no
        live session exists - this prevents the common bug where tapping a
        session and sending a message silently creates a new session."""
        if not str(text or "").strip():
            raise ValueError("Prompt cannot be empty")
        if self.session_id is None and self.stored_session_id:
            await self.resume_session(self.stored_session_id)
        if self.session_id is None:
            await self.create_session()
        result = await self.request(
            "prompt.submit", {"session_id": self.session_id, "text": str(text).strip()}
        )
        return result if isinstance(result, dict) else {"status": result}

    async def execute_slash_command(self, command: str) -> Mapping[str, Any]:
        """Execute a Desktop/TUI slash command against the active live session.

        Mobile owns a small set of local shell commands, but runtime commands
        such as ``/goal`` live in the connected Hermes backend.  Route command
        families through the same canonical RPCs used by Desktop/TUI so the APK
        does not drift from backend behavior.  Like prompt submission,
        resuming/creating the live session here prevents a selected stored
        session from being silently bypassed.
        """
        clean = str(command or "").strip()
        if not clean:
            raise ValueError("Slash command cannot be empty")
        if not clean.startswith("/"):
            clean = f"/{clean}"
        if self.session_id is None and self.stored_session_id:
            await self.resume_session(self.stored_session_id)
        if self.session_id is None:
            await self.create_session()
        command_text = clean.lstrip("/")
        command_parts = command_text.split(maxsplit=1)
        command_name = command_parts[0].lower() if command_parts else ""
        command_arg = command_parts[1] if len(command_parts) > 1 else ""
        if command_name == "goal":
            result = await self.request(
                "command.dispatch",
                {"session_id": self.session_id, "name": command_name, "arg": command_arg},
            )
        else:
            result = await self.request(
                "slash.exec",
                {"session_id": self.session_id, "command": clean},
            )
        return result if isinstance(result, dict) else {"output": str(result)}

    async def interrupt(self) -> Mapping[str, Any]:
        if not self.session_id:
            return {"interrupted": False}
        result = await self.request("session.interrupt", {"session_id": self.session_id})
        return result if isinstance(result, dict) else {"result": result}

    async def list_sessions(self, limit: int = 50) -> list[Mapping[str, Any]]:
        result = await self.request("session.list", {"limit": max(1, min(int(limit), 200))})
        if not isinstance(result, dict) or not isinstance(result.get("sessions"), list):
            return []
        return [item for item in result["sessions"] if isinstance(item, dict)]

    async def get_projects_tree(self) -> Mapping[str, Any]:
        """Fetch Desktop's authoritative project overview snapshot."""
        result = await self.request(
            "projects.tree",
            {"preview_limit": 3, "session_limit": 2000},
        )
        if not isinstance(result, dict):
            return {"projects": [], "active_id": None, "scoped_session_ids": []}
        projects = result.get("projects")
        return {
            "projects": [item for item in projects if isinstance(item, dict)]
            if isinstance(projects, list)
            else [],
            "active_id": result.get("active_id"),
            "scoped_session_ids": result.get("scoped_session_ids") or [],
        }

    async def get_project_sessions(self, project_id: str) -> Optional[Mapping[str, Any]]:
        """Hydrate one Desktop project into repos, lanes, and session rows."""
        if not str(project_id or "").strip():
            raise ValueError("project_id is required")
        result = await self.request(
            "projects.project_sessions",
            {"project_id": str(project_id), "session_limit": 5000},
        )
        if not isinstance(result, dict) or not isinstance(result.get("project"), dict):
            return None
        return result["project"]

    async def get_pet_info(self) -> Mapping[str, Any]:
        """Return the active Remote pet using the canonical Desktop RPC."""
        params = {"profile": self.profile} if self.profile else {}
        result = await self.request("pet.info", params)
        return result if isinstance(result, dict) else {"enabled": False}

    async def get_model_options(self, *, refresh: bool = False) -> Mapping[str, Any]:
        """Return the backend-owned provider/model inventory."""
        result = await self.request(
            "model.options",
            {
                "include_unconfigured": False,
                "explicit_only": False,
                **({"refresh": True} if refresh else {}),
            },
        )
        return result if isinstance(result, dict) else {"providers": []}

    async def get_remote_skills(self) -> list[Mapping[str, Any]]:
        """Return skills installed in the connected Hermes runtime."""
        result = await self.request("skills.manage", {"action": "list"})
        skills = result.get("skills") if isinstance(result, dict) else None
        rows: list[Mapping[str, Any]] = []
        if isinstance(skills, dict):
            for category, items in sorted(skills.items(), key=lambda item: str(item[0]).lower()):
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, str) and item.strip():
                        rows.append(
                            {
                                "name": item.strip(),
                                "description": "",
                                "category": str(category),
                            }
                        )
        elif isinstance(skills, list):
            for item in skills:
                if isinstance(item, dict):
                    rows.append(item)
                elif isinstance(item, str) and item.strip():
                    rows.append({"name": item.strip(), "description": ""})
        return rows

    async def get_pet_gallery(self, *, local_only: bool = False) -> Mapping[str, Any]:
        """Return the profile-scoped Desktop petdex gallery."""
        params: dict[str, Any] = {"localOnly": bool(local_only)}
        if self.profile:
            params["profile"] = self.profile
        result = await self.request("pet.gallery", params)
        if not isinstance(result, dict):
            return {"enabled": False, "active": "", "pets": []}
        return result

    async def select_pet(self, slug: str) -> Mapping[str, Any]:
        """Install if needed and select a pet in the active Remote profile."""
        params: dict[str, Any] = {"slug": str(slug).strip()}
        if not params["slug"]:
            raise ValueError("slug is required")
        if self.profile:
            params["profile"] = self.profile
        result = await self.request("pet.select", params)
        return result if isinstance(result, dict) else {"ok": False}

    async def disable_pet(self) -> Mapping[str, Any]:
        """Disable the active pet in the connected Remote profile."""
        params = {"profile": self.profile} if self.profile else {}
        result = await self.request("pet.disable", params)
        return result if isinstance(result, dict) else {"ok": False}

    async def respond_approval(self, choice: str) -> Any:
        return await self.request(
            "approval.respond", {"choice": choice, "session_id": self.session_id}
        )

    async def respond_clarify(self, request_id: str, answer: Any) -> Any:
        return await self.request("clarify.respond", {"request_id": request_id, "answer": answer})

    async def respond_secret(self, request_id: str, value: str) -> Any:
        return await self.request("secret.respond", {"request_id": request_id, "value": value})

    async def respond_sudo(self, request_id: str, password: str) -> Any:
        return await self.request("sudo.respond", {"request_id": request_id, "password": password})

    async def _close_transport(self) -> None:
        task = self._receiver_task
        self._receiver_task = None
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def close(self) -> None:
        self._closed = True
        await self._close_transport()
        self._reject_pending(RemoteConnectionError("Remote gateway connection closed"))
        if self._owns_http:
            await self._http.aclose()
        await self._set_state("closed")
