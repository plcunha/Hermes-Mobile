"""Desktop-parity session browser for Hermes Mobile.

The Desktop keeps sessions in a structured sidebar rather than a modal picker.
On a phone the equivalent is a full-height destination: persistent search,
source-aware sections, durable local pins and a large row target for resuming.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import flet as ft

from hermes_mobile.locales import t
from hermes_mobile.ui.common import (
    MONO_FONT,
    close_dialog,
    empty_state,
    open_dialog,
    section_label,
    snack,
)
from hermes_mobile.ui.theme import mode_colors

logger = logging.getLogger(__name__)

_SOURCE_LABELS = {
    "api_server": "API",
    "bluebubbles": "iMessage",
    "cli": "CLI",
    "codex": "Codex",
    "desktop": "Desktop",
    "discord": "Discord",
    "email": "Email",
    "gateway": "Gateway",
    "local": "Local",
    "matrix": "Matrix",
    "mattermost": "Mattermost",
    "mobile": "Mobile",
    "qqbot": "QQ",
    "signal": "Signal",
    "slack": "Slack",
    "sms": "SMS",
    "telegram": "Telegram",
    "tui": "TUI",
    "webhook": "Webhook",
    "weixin": "WeChat",
    "whatsapp": "WhatsApp",
    "yuanbao": "Yuanbao",
}

_SOURCE_ICONS = {
    "cli": ft.Icons.TERMINAL,
    "desktop": ft.Icons.DESKTOP_WINDOWS_OUTLINED,
    "discord": ft.Icons.FORUM_OUTLINED,
    "email": ft.Icons.MAIL_OUTLINE,
    "mobile": ft.Icons.PHONE_ANDROID,
    "telegram": ft.Icons.SEND_ROUNDED,
    "tui": ft.Icons.TERMINAL,
    "webhook": ft.Icons.WEBHOOK,
    "whatsapp": ft.Icons.CHAT_OUTLINED,
}


class SessionPinStore:
    """Small per-remote pin store matching Desktop's device-local pin semantics."""

    def __init__(self, path: Path, scope: str):
        self.path = Path(path)
        self.scope = scope or "default"

    def load(self) -> list[str]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            scopes = payload.get("scopes") if isinstance(payload, dict) else None
            values = scopes.get(self.scope, []) if isinstance(scopes, dict) else []
            if not isinstance(values, list):
                return []
            return list(dict.fromkeys(str(value) for value in values if value))[:200]
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            return []

    def save(self, session_ids: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            payload = {}
        scopes = payload.get("scopes")
        if not isinstance(scopes, dict):
            scopes = {}
        scopes[self.scope] = list(dict.fromkeys(session_ids))[:200]
        payload = {"version": 1, "scopes": scopes}

        fd, raw_temp = tempfile.mkstemp(prefix="session-pins-", dir=self.path.parent)
        temp_path = Path(raw_temp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


class SessionsView:
    """Full-height mobile translation of the Desktop session sidebar."""

    def __init__(self, app):
        self.app = app
        self.page = app.page
        settings = app.settings
        pin_dir = settings.get_data_dir() / "ui"
        pin_dir.mkdir(parents=True, exist_ok=True)
        self.pin_store = SessionPinStore(
            pin_dir / "session-pins.json",
            self._pin_scope(),
        )
        self.pinned_ids = self.pin_store.load()
        self.sessions: list[Mapping[str, Any]] = []
        self.query = ""
        self.active_filter = "all"
        self.loading = False
        self.error = ""

        c = mode_colors(self.app.dark_mode)
        self.search_field = ft.TextField(
            hint_text=t("sessions.search"),
            prefix_icon=ft.Icons.SEARCH,
            on_change=self._on_search,
            text_size=13,
            border=ft.InputBorder.NONE,
            bgcolor=None,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        )
        self.filter_strip = ft.Container()
        self.session_list = ft.ListView(
            expand=True,
            spacing=0,
            padding=ft.Padding.only(left=14, right=14, bottom=12),
        )
        self.count_text = ft.Text("", size=11, color=c["muted_foreground"])

    def _pin_scope(self) -> str:
        settings = self.app.settings
        remote_url = str(getattr(settings, "remote_url", "") or "").rstrip("/")
        if not remote_url:
            remote_url = "local"
        profile = str(getattr(settings, "remote_profile", "") or "default").strip() or "default"
        return f"{remote_url}|{profile}"

    def _sync_pin_store(self) -> None:
        scope = self._pin_scope()
        if self.pin_store.scope == scope:
            return
        self.pin_store = SessionPinStore(self.pin_store.path, scope)
        self.pinned_ids = self.pin_store.load()

    def build(self) -> ft.Control:
        self._sync_pin_store()
        c = mode_colors(self.app.dark_mode)
        # These controls outlive shell rebuilds, so retokenize them whenever
        # the effective theme changes in place.
        self.count_text.color = c["muted_foreground"]
        self.filter_strip.content = self._build_filters()
        self._render()
        return ft.Column(
            [
                ft.Container(
                    content=self.search_field,
                    margin=ft.Margin.only(left=14, right=14, top=12, bottom=7),
                    bgcolor=c["input"],
                    border=ft.Border.all(1, c["border"]),
                    border_radius=ft.BorderRadius.all(9),
                ),
                self.filter_strip,
                ft.Container(
                    content=ft.Row(
                        [
                            self.count_text,
                            ft.Container(expand=True),
                            ft.IconButton(
                                icon=ft.Icons.REFRESH,
                                icon_size=17,
                                icon_color=c["muted_foreground"],
                                tooltip=t("sessions.refresh"),
                                on_click=lambda e: asyncio.create_task(self.refresh()),
                                visual_density=ft.VisualDensity.COMPACT,
                            ),
                        ],
                        spacing=4,
                    ),
                    padding=ft.Padding.only(left=16, right=8, top=4, bottom=2),
                ),
                self.session_list,
            ],
            expand=True,
            spacing=0,
        )

    async def refresh(self) -> None:
        self._sync_pin_store()
        client = getattr(self.app, "remote_client", None)
        if client is None or client.state != "open":
            self.error = t("sessions.remote_offline")
            self.loading = False
            self._render(update=True)
            return

        self.loading = True
        self.error = ""
        self._render(update=True)
        session_result = await asyncio.gather(
            client.list_sessions(limit=100),
            return_exceptions=True,
        )
        session_result = session_result[0]
        if isinstance(session_result, Exception):
            self.sessions = []
            self.error = str(session_result)
        else:
            self.sessions = sorted(
                session_result,
                key=lambda item: float(item.get("started_at") or 0),
                reverse=True,
            )
        self.loading = False
        self.filter_strip.content = self._build_filters()
        self._render(update=True)

    def _on_search(self, event) -> None:
        self.query = str(getattr(event.control, "value", "") or "").strip().lower()
        self._render(update=True)

    def _filtered(self) -> list[Mapping[str, Any]]:
        result = []
        for item in self.sessions:
            source = self._source(item)
            terms = " ".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("preview") or ""),
                    source,
                    self._source_label(source),
                ]
            ).lower()
            if self.query and self.query not in terms:
                continue
            session_id = self._id(item)
            if self.active_filter == "pinned" and session_id not in self.pinned_ids:
                continue
            if self.active_filter.startswith("source:"):
                selected_source = self.active_filter.split(":", 1)[1]
                if source != selected_source:
                    continue
            result.append(item)
        return result

    def _build_filters(self) -> ft.Control:
        sources = []
        for item in self.sessions:
            source = self._source(item)
            if source not in sources:
                sources.append(source)
        filters = [("all", t("sessions.filter_all")), ("pinned", t("sessions.pinned"))]
        filters.extend((f"source:{source}", self._source_label(source)) for source in sources)
        return ft.ListView(
            controls=[self._filter_pill(key, label) for key, label in filters],
            horizontal=True,
            height=38,
            spacing=7,
            padding=ft.Padding.symmetric(horizontal=14),
        )

    def _filter_pill(self, key: str, label: str) -> ft.Control:
        c = mode_colors(self.app.dark_mode)
        selected = self.active_filter == key
        return ft.Container(
            content=ft.Text(
                label,
                size=11,
                weight=ft.FontWeight.W_600 if selected else ft.FontWeight.W_500,
                color=c["primary_foreground"] if selected else c["muted_foreground"],
            ),
            height=34,
            padding=ft.Padding.symmetric(horizontal=12, vertical=7),
            bgcolor=c["primary"] if selected else None,
            border=ft.Border.all(1, c["primary"] if selected else c["border"]),
            border_radius=ft.BorderRadius.all(18),
            ink=True,
            on_click=lambda e, value=key: self._set_filter(value),
        )

    def _set_filter(self, value: str) -> None:
        self.active_filter = value
        self.filter_strip.content = self._build_filters()
        self._render(update=True)

    def _render(self, *, update: bool = False) -> None:
        if self.loading:
            self.count_text.value = t("common.loading")
            self.session_list.controls = self._loading_rows()
        elif self.error:
            self.count_text.value = ""
            self.session_list.controls = [
                empty_state(
                    self.app.dark_mode,
                    t("sessions.could_not_load"),
                    self.error,
                    icon=ft.Icons.CLOUD_OFF_OUTLINED,
                    action=ft.Button(
                        content=t("sessions.retry"),
                        icon=ft.Icons.REFRESH,
                        on_click=lambda e: asyncio.create_task(self.refresh()),
                    ),
                )
            ]
        else:
            filtered = self._filtered()
            count_key = "sessions.count_one" if len(filtered) == 1 else "sessions.count"
            self.count_text.value = t(count_key).format(count=len(filtered))
            if not filtered:
                self.session_list.controls = [
                    empty_state(
                        self.app.dark_mode,
                        t("sessions.empty"),
                        t("sessions.empty_help"),
                        icon=ft.Icons.FORUM_OUTLINED,
                    )
                ]
            else:
                controls: list[ft.Control] = []
                if self.active_filter == "all":
                    pinned = [item for item in filtered if self._id(item) in self.pinned_ids]
                    recent = [item for item in filtered if self._id(item) not in self.pinned_ids]
                    if pinned:
                        controls.append(self._list_heading(t("sessions.pinned"), len(pinned)))
                        controls.extend(self._session_row(item) for item in pinned)
                    for label, items in self._date_groups(recent):
                        controls.append(self._list_heading(label, len(items)))
                        controls.extend(self._session_row(item) for item in items)
                else:
                    label = (
                        t("sessions.pinned")
                        if self.active_filter == "pinned"
                        else self._source_label(self.active_filter.split(":", 1)[1])
                    )
                    controls.append(self._list_heading(label, len(filtered)))
                    controls.extend(self._session_row(item) for item in filtered)
                self.session_list.controls = controls
        if update:
            try:
                self.page.update()
            except Exception:
                logger.debug("Could not update sessions view", exc_info=True)

    def _loading_rows(self) -> list[ft.Control]:
        c = mode_colors(self.app.dark_mode)
        controls: list[ft.Control] = []
        for index in range(5):
            controls.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(
                                width=34,
                                height=34,
                                bgcolor=c["muted"],
                                border_radius=8,
                            ),
                            ft.Column(
                                [
                                    ft.Container(
                                        width=160 + index * 18,
                                        height=9,
                                        bgcolor=c["muted"],
                                        border_radius=4,
                                    ),
                                    ft.Container(
                                        width=225 - index * 12,
                                        height=7,
                                        bgcolor=c["muted"],
                                        border_radius=4,
                                    ),
                                ],
                                spacing=8,
                                expand=True,
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=ft.Padding.symmetric(horizontal=4, vertical=12),
                    border=ft.Border.only(bottom=ft.BorderSide(1, c["border"])),
                )
            )
        return controls

    def _list_heading(self, title: str, count: int) -> ft.Control:
        return ft.Container(
            content=section_label(self.app.dark_mode, title, str(count)),
            padding=ft.Padding.only(left=2, right=2, top=14, bottom=5),
        )

    def _date_groups(
        self, items: list[Mapping[str, Any]]
    ) -> list[tuple[str, list[Mapping[str, Any]]]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        order = [
            t("sessions.today"),
            t("sessions.yesterday"),
            t("sessions.last_seven_days"),
            t("sessions.older"),
        ]
        now = datetime.now()
        for item in items:
            try:
                stamp = datetime.fromtimestamp(float(item.get("started_at") or 0))
            except (TypeError, ValueError, OSError):
                stamp = datetime.fromtimestamp(0)
            if stamp.date() == now.date():
                label = order[0]
            elif stamp.date() == (now - timedelta(days=1)).date():
                label = order[1]
            elif stamp >= now - timedelta(days=7):
                label = order[2]
            else:
                label = order[3]
            grouped.setdefault(label, []).append(item)
        return [(label, grouped[label]) for label in order if grouped.get(label)]

    def _session_row(self, item: Mapping[str, Any]) -> ft.Control:
        c = mode_colors(self.app.dark_mode)
        session_id = self._id(item)
        source = self._source(item)
        pinned = session_id in self.pinned_ids
        title = str(item.get("title") or "").strip()
        preview = " ".join(str(item.get("preview") or "").split())
        if not title:
            title = preview[:72] or t("sessions.untitled")
        message_count = int(item.get("message_count") or 0)
        when = self._format_when(item.get("started_at"))
        badge = self._source_badge(source)
        client = getattr(self.app, "remote_client", None)
        selected = bool(client and getattr(client, "stored_session_id", None) == session_id)

        meta: list[ft.Control] = [
            badge,
            ft.Icon(ft.Icons.CHAT_BUBBLE_OUTLINE, size=11, color=c["muted_foreground"]),
            ft.Text(
                str(message_count),
                size=9,
                color=c["muted_foreground"],
                font_family=MONO_FONT,
                tooltip=t("sessions.messages"),
            ),
        ]
        if selected:
            meta.extend(
                [
                    ft.Container(width=3),
                    ft.Container(
                        content=ft.Text(
                            t("sessions.active"),
                            size=8,
                            weight=ft.FontWeight.W_700,
                            color=c["success"],
                            font_family=MONO_FONT,
                        ),
                        padding=ft.Padding.symmetric(horizontal=5, vertical=2),
                        border=ft.Border.all(1, c["success"]),
                        border_radius=4,
                    ),
                ]
            )

        return ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Icon(
                            _SOURCE_ICONS.get(source, ft.Icons.CHAT_BUBBLE_OUTLINE),
                            size=18,
                            color=c["primary"] if source == "telegram" else c["muted_foreground"],
                        ),
                        width=34,
                        height=34,
                        alignment=ft.Alignment.CENTER,
                        bgcolor=c["accent"] if source == "telegram" else c["muted"],
                        border_radius=8,
                    ),
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(
                                        title,
                                        size=13,
                                        weight=ft.FontWeight.W_600,
                                        color=c["foreground"],
                                        max_lines=1,
                                        overflow=ft.TextOverflow.ELLIPSIS,
                                        expand=True,
                                    ),
                                    ft.Text(
                                        when,
                                        size=10,
                                        color=c["muted_foreground"],
                                        font_family=MONO_FONT,
                                    ),
                                ],
                                spacing=8,
                            ),
                            ft.Text(
                                preview or t("sessions.no_preview"),
                                size=10.5,
                                color=c["muted_foreground"],
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Row(
                                meta,
                                spacing=5,
                            ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    ft.PopupMenuButton(
                        icon=ft.Icons.MORE_HORIZ,
                        icon_size=18,
                        icon_color=c["muted_foreground"],
                        tooltip=t("sessions.actions"),
                        items=[
                            ft.PopupMenuItem(
                                icon=ft.Icons.PLAY_ARROW_OUTLINED,
                                content=t("sessions.continue"),
                                on_click=lambda e, sid=session_id, name=title: asyncio.create_task(
                                    self.app.resume_remote_session(sid, name)
                                ),
                            ),
                            ft.PopupMenuItem(
                                icon=ft.Icons.PUSH_PIN if pinned else ft.Icons.PUSH_PIN_OUTLINED,
                                content=t("sessions.unpin") if pinned else t("sessions.pin"),
                                on_click=lambda e, sid=session_id: self._toggle_pin(sid),
                            ),
                            ft.PopupMenuItem(
                                icon=ft.Icons.EDIT_OUTLINED,
                                content=t("sessions.rename"),
                                on_click=lambda e, sid=session_id, name=title: (
                                    self._show_rename_dialog(sid, name)
                                ),
                            ),
                            ft.PopupMenuItem(
                                icon=ft.Icons.CALL_SPLIT,
                                content=t("sessions.branch"),
                                on_click=lambda e, sid=session_id, name=title: (
                                    self._show_branch_dialog(sid, name)
                                ),
                            ),
                            ft.PopupMenuItem(
                                icon=ft.Icons.ARCHIVE_OUTLINED,
                                content=t("sessions.archive"),
                                on_click=lambda e, sid=session_id, name=title: (
                                    self._show_archive_dialog(sid, name)
                                ),
                            ),
                            *(
                                []
                                if selected
                                else [
                                    ft.PopupMenuItem(
                                        icon=ft.Icons.DELETE_OUTLINE,
                                        content=t("sessions.delete"),
                                        on_click=lambda e, sid=session_id, name=title: (
                                            self._show_delete_dialog(sid, name)
                                        ),
                                    ),
                                ]
                            ),
                        ],
                        width=44,
                        height=44,
                        padding=10,
                    ),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.only(left=7, right=0, top=8, bottom=8),
            bgcolor=c["accent"] if selected else None,
            border=ft.Border.only(
                left=ft.BorderSide(3, c["primary"] if selected else c["background"]),
                bottom=ft.BorderSide(1, c["border"]),
            ),
            border_radius=6,
            ink=True,
            on_click=lambda e, sid=session_id, name=title: asyncio.create_task(
                self.app.resume_remote_session(sid, name)
            ),
        )

    def _show_rename_dialog(self, session_id: str, title: str) -> None:
        field = ft.TextField(
            label=t("sessions.rename_label"),
            value=title,
            autofocus=True,
            max_length=200,
        )

        async def submit() -> None:
            if await self._rename_session(session_id, str(field.value or "")):
                close_dialog(self.page, dialog)

        dialog = ft.AlertDialog(
            title=ft.Text(t("sessions.rename")),
            content=field,
            actions=[
                ft.TextButton(
                    t("common.cancel"),
                    on_click=lambda e: close_dialog(self.page, dialog),
                ),
                ft.Button(
                    t("common.save"),
                    on_click=lambda e: asyncio.create_task(submit()),
                ),
            ],
        )
        open_dialog(self.page, dialog)

    def _show_branch_dialog(self, session_id: str, title: str) -> None:
        field = ft.TextField(
            label=t("sessions.branch_label"),
            value=t("sessions.branch_default").format(title=title),
            autofocus=True,
            max_length=200,
        )

        async def submit() -> None:
            if await self._branch_session(session_id, str(field.value or "")):
                close_dialog(self.page, dialog)

        dialog = ft.AlertDialog(
            title=ft.Text(t("sessions.branch")),
            content=ft.Column(
                [
                    ft.Text(t("sessions.branch_help"), size=12),
                    field,
                ],
                tight=True,
                spacing=12,
            ),
            actions=[
                ft.TextButton(
                    t("common.cancel"),
                    on_click=lambda e: close_dialog(self.page, dialog),
                ),
                ft.Button(
                    t("sessions.branch_action"),
                    on_click=lambda e: asyncio.create_task(submit()),
                ),
            ],
        )
        open_dialog(self.page, dialog)

    def _show_delete_dialog(self, session_id: str, title: str) -> None:
        self._show_destructive_dialog(
            session_id,
            title,
            heading=t("sessions.delete"),
            body=t("sessions.delete_confirm").format(title=title),
            button=t("sessions.delete_action"),
        )

    def _show_archive_dialog(self, session_id: str, title: str) -> None:
        async def submit() -> None:
            if await self._archive_session(session_id, title):
                close_dialog(self.page, dialog)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("sessions.archive")),
            content=ft.Text(t("sessions.archive_confirm").format(title=title)),
            actions=[
                ft.TextButton(
                    t("common.cancel"),
                    on_click=lambda e: close_dialog(self.page, dialog),
                ),
                ft.Button(
                    t("sessions.archive_action"),
                    on_click=lambda e: asyncio.create_task(submit()),
                ),
            ],
        )
        open_dialog(self.page, dialog)

    def _show_destructive_dialog(
        self,
        session_id: str,
        title: str,
        *,
        heading: str,
        body: str,
        button: str,
    ) -> None:
        async def submit() -> None:
            if await self._delete_session(session_id, title):
                close_dialog(self.page, dialog)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(heading),
            content=ft.Text(body),
            actions=[
                ft.TextButton(
                    t("common.cancel"),
                    on_click=lambda e: close_dialog(self.page, dialog),
                ),
                ft.Button(button, on_click=lambda e: asyncio.create_task(submit())),
            ],
        )
        open_dialog(self.page, dialog)

    def _remove_session_locally(self, session_id: str) -> None:
        self.sessions = [item for item in self.sessions if self._id(item) != session_id]
        if session_id in self.pinned_ids:
            self.pinned_ids = [value for value in self.pinned_ids if value != session_id]
            self.pin_store.save(self.pinned_ids)
        self.filter_strip.content = self._build_filters()
        self._render(update=True)

    def _remote_client(self) -> Any:
        client = getattr(self.app, "remote_client", None)
        if client is None or getattr(client, "state", "") != "open":
            raise RuntimeError(t("sessions.remote_offline"))
        return client

    async def _rename_session(self, session_id: str, title: str) -> bool:
        clean_title = str(title or "").strip()
        if not clean_title:
            snack(self.page, t("sessions.title_required"), error=True)
            return False
        try:
            client = self._remote_client()
            await client.rename_session(session_id, clean_title)
        except Exception as exc:
            logger.warning("Could not rename remote session: %s", exc)
            snack(self.page, t("sessions.action_error").format(error=exc), error=True)
            return False
        self.sessions = [
            {**item, "title": clean_title} if self._id(item) == session_id else item
            for item in self.sessions
        ]
        self._render(update=True)
        snack(self.page, t("sessions.renamed"))
        return True

    async def _delete_session(self, session_id: str, title: str = "") -> bool:
        try:
            client = self._remote_client()
            deleted = await client.delete_session(session_id)
            if not deleted:
                raise RuntimeError(t("sessions.delete_rejected"))
        except Exception as exc:
            logger.warning("Could not delete remote session: %s", exc)
            snack(self.page, t("sessions.action_error").format(error=exc), error=True)
            return False
        self._remove_session_locally(session_id)
        snack(self.page, t("sessions.deleted").format(title=title))
        return True

    async def _archive_session(self, session_id: str, title: str = "") -> bool:
        try:
            client = self._remote_client()
            archived = await client.archive_session(session_id, archived=True)
            if not archived:
                raise RuntimeError(t("sessions.branch_rejected"))
        except Exception as exc:
            logger.warning("Could not archive remote session: %s", exc)
            snack(self.page, t("sessions.action_error").format(error=exc), error=True)
            return False
        self._remove_session_locally(session_id)
        snack(self.page, t("sessions.archived").format(title=title))
        return True

    async def _branch_session(self, session_id: str, title: str) -> bool:
        try:
            client = self._remote_client()
            is_active = bool(
                str(getattr(client, "session_id", "") or "")
                and session_id == str(getattr(client, "stored_session_id", "") or "")
            )
            if is_active:
                branch = await client.branch_active_session(title=title)
                branch_id = str(branch.get("stored_session_id") or "")
            else:
                branch = await client.fork_session(session_id, title=title)
                branch_id = str(branch.get("id") or "")
            if not branch_id:
                raise RuntimeError(t("sessions.branch_rejected"))
            branch_title = str(branch.get("title") or title)
            if is_active:
                activation = self.app.activate_remote_session_result(branch, branch_title)
                if inspect.isawaitable(activation):
                    await activation
            else:
                resume = self.app.resume_remote_session(branch_id, branch_title)
                if inspect.isawaitable(resume):
                    await resume
        except Exception as exc:
            logger.warning("Could not branch remote session: %s", exc)
            snack(self.page, t("sessions.action_error").format(error=exc), error=True)
            return False
        snack(self.page, t("sessions.branched"))
        return True

    def _source_badge(self, source: str) -> ft.Control:
        c = mode_colors(self.app.dark_mode)
        return ft.Container(
            content=ft.Text(
                self._source_label(source).upper(),
                size=8,
                color=c["primary"] if source == "telegram" else c["muted_foreground"],
                font_family=MONO_FONT,
                weight=ft.FontWeight.W_600,
            ),
            padding=ft.Padding.symmetric(horizontal=5, vertical=2),
            bgcolor=c["accent"] if source == "telegram" else c["muted"],
            border=ft.Border.all(1, c["border"]),
            border_radius=4,
        )

    def _toggle_pin(self, session_id: str) -> None:
        if session_id in self.pinned_ids:
            self.pinned_ids = [value for value in self.pinned_ids if value != session_id]
            now_pinned = False
        else:
            self.pinned_ids = [session_id, *self.pinned_ids]
            now_pinned = True
        try:
            self.pin_store.save(self.pinned_ids)
        except OSError as exc:
            logger.warning("Could not persist session pin: %s", exc)
            snack(self.page, t("sessions.pin_error"), error=True)
        self._mirror_pin_remote(session_id, now_pinned)
        self._render(update=True)

    def _mirror_pin_remote(self, session_id: str, pinned: bool) -> None:
        """Best-effort mirror of a device-local pin to the backend keep flag.

        Matches Desktop: the sidebar pin stays local, but the backend's
        auto-archive sweep must know a pinned chat is never to be hidden.
        """
        client = getattr(self.app, "remote_client", None)
        if client is None or getattr(client, "state", "") != "open":
            return

        async def _mirror() -> None:
            try:
                await client.pin_session_remote(session_id, pinned=pinned)
            except Exception as exc:
                logger.debug("Could not mirror remote pin: %s", exc)

        asyncio.create_task(_mirror())

    @staticmethod
    def _id(item: Mapping[str, Any]) -> str:
        return str(item.get("id") or "")

    @staticmethod
    def _source(item: Mapping[str, Any]) -> str:
        return str(item.get("source") or "local").strip().lower() or "local"

    @staticmethod
    def _source_label(source: str) -> str:
        return _SOURCE_LABELS.get(source, source.replace("_", " ").replace("-", " ").title())

    @staticmethod
    def _format_when(value: Any) -> str:
        try:
            stamp = datetime.fromtimestamp(float(value))
        except (TypeError, ValueError, OSError):
            return ""
        now = datetime.now()
        if stamp.date() == now.date():
            return stamp.strftime("%H:%M")
        if stamp.year == now.year:
            return stamp.strftime("%b %d")
        return stamp.strftime("%Y-%m-%d")
