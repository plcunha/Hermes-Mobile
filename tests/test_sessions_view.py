"""Regression coverage for the Desktop-parity mobile session browser."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import flet as ft
import pytest

from hermes_mobile.remote.client import RemoteHermesClient
from hermes_mobile.ui.sessions_view import SessionPinStore, SessionsView
from hermes_mobile.ui.theme import mode_colors


class FakePage:
    theme_mode = ft.ThemeMode.DARK

    def __init__(self):
        self.updates = 0
        self.overlay = []

    def update(self):
        self.updates += 1


def walk_controls(control: ft.Control):
    seen: set[int] = set()
    stack = [control]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        fields = set(getattr(type(current), "__dataclass_fields__", {}))
        fields.update(
            {
                "controls",
                "content",
                "leading",
                "trailing",
                "title",
                "subtitle",
                "label",
                "actions",
                "items",
            }
        )
        for name in fields:
            try:
                value = getattr(current, name)
            except Exception:
                continue
            if isinstance(value, ft.Control):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(item for item in value if isinstance(item, ft.Control))


def make_app(tmp_path: Path):
    return SimpleNamespace(
        page=FakePage(),
        settings=SimpleNamespace(
            remote_url="https://hermes.example.test",
            remote_profile="default",
            get_data_dir=lambda: tmp_path,
        ),
        dark_mode=True,
        current_view="sessions",
        _start_new_session=lambda event=None: None,
        resume_remote_session=lambda session_id, title: None,
        activate_remote_session_result=lambda result, title: None,
    )


def test_build_retokens_persistent_controls_after_theme_change(tmp_path):
    app = make_app(tmp_path)
    view = SessionsView(app)
    view.build()

    app.dark_mode = False
    view.build()

    assert view.count_text.color == mode_colors(False)["muted_foreground"]


def text_values(control: ft.Control) -> list[str]:
    return [
        str(item.value)
        for item in walk_controls(control)
        if isinstance(item, ft.Text) and item.value
    ]


def test_session_browser_groups_pins_and_recency_with_source_filters(tmp_path):
    view = SessionsView(make_app(tmp_path))
    view.sessions = [
        {
            "id": "pinned-telegram",
            "title": "Pinned bot chat",
            "preview": "From Telegram",
            "source": "telegram",
            "started_at": 100,
            "message_count": 4,
        },
        {
            "id": "telegram-chat",
            "title": "Telegram support",
            "preview": "Gateway conversation",
            "source": "telegram",
            "started_at": 90,
            "message_count": 8,
        },
        {
            "id": "desktop-chat",
            "title": "Desktop work",
            "preview": "Local workspace",
            "source": "desktop",
            "started_at": 80,
            "message_count": 3,
        },
    ]
    view.pinned_ids = ["pinned-telegram"]

    root = view.build()
    texts = text_values(root)

    assert "PINNED" in texts
    assert "TELEGRAM" in texts
    assert "OLDER" in texts
    assert "All" in texts
    assert "Desktop" in texts
    assert texts.count("Pinned bot chat") == 1
    assert texts.count("Telegram support") == 1
    assert texts.count("Desktop work") == 1
    assert texts.count("TELEGRAM") >= 2  # section + visible source badges
    actions = next(
        item
        for item in walk_controls(root)
        if isinstance(item, ft.PopupMenuButton) and item.icon == ft.Icons.MORE_HORIZ
    )
    assert actions.width == 44
    assert actions.height == 44


def test_session_browser_marks_active_session_and_filters_by_source(tmp_path):
    app = make_app(tmp_path)
    app.remote_client = SimpleNamespace(stored_session_id="desktop-chat")
    view = SessionsView(app)
    view.sessions = [
        {"id": "telegram-chat", "title": "From phone", "source": "telegram"},
        {"id": "desktop-chat", "title": "Active work", "source": "desktop"},
    ]
    view.active_filter = "source:desktop"

    root = view.build()
    labels = text_values(root)

    assert [view._id(item) for item in view._filtered()] == ["desktop-chat"]
    assert "Active work" in labels
    assert "From phone" not in labels
    assert "ACTIVE" in labels


def test_pin_store_is_scoped_deduplicated_and_private(tmp_path):
    path = tmp_path / "ui" / "session-pins.json"
    first = SessionPinStore(path, "server-a|default")
    second = SessionPinStore(path, "server-b|default")

    first.save(["one", "one", "two"])
    second.save(["other"])

    assert first.load() == ["one", "two"]
    assert second.load() == ["other"]
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_refresh_reloads_pins_when_remote_scope_changes(tmp_path):
    app = make_app(tmp_path)
    view = SessionsView(app)
    pin_path = tmp_path / "ui" / "session-pins.json"
    SessionPinStore(pin_path, "https://second.example.test|work").save(["pinned-session"])

    class Client:
        state = "open"

        async def list_sessions(self, limit=100):
            return [
                {
                    "id": "pinned-session",
                    "title": "Pinned on second backend",
                    "source": "desktop",
                }
            ]

        async def get_pet_info(self):
            return {"enabled": False}

    app.settings.remote_url = "https://second.example.test/"
    app.settings.remote_profile = "work"
    app.remote_client = Client()

    await view.refresh()

    assert view.pin_store.scope == "https://second.example.test|work"
    assert view.pinned_ids == ["pinned-session"]
    assert [view._id(item) for item in view._filtered()] == ["pinned-session"]


def test_single_search_result_uses_singular_count(tmp_path):
    view = SessionsView(make_app(tmp_path))
    view.sessions = [{"id": "one", "title": "Only", "source": "desktop"}]
    view.build()

    assert view.count_text.value == "1 session"


@pytest.mark.asyncio
async def test_remote_client_uses_canonical_pet_info_rpc(monkeypatch):
    client = object.__new__(RemoteHermesClient)
    client.profile = "work"
    calls = []

    async def request(method, params):
        calls.append((method, params))
        return {"enabled": True, "displayName": "Pool Dog"}

    monkeypatch.setattr(client, "request", request)

    result = await client.get_pet_info()

    assert calls == [("pet.info", {"profile": "work"})]
    assert result["displayName"] == "Pool Dog"


@pytest.mark.asyncio
async def test_remote_client_uses_desktop_project_tree_contract(monkeypatch):
    client = object.__new__(RemoteHermesClient)
    calls = []

    async def request(method, params):
        calls.append((method, params))
        if method == "projects.tree":
            return {"projects": [{"id": "alpha"}], "active_id": "alpha"}
        return {"project": {"id": params["project_id"], "repos": []}}

    monkeypatch.setattr(client, "request", request)

    tree = await client.get_projects_tree()
    project = await client.get_project_sessions("alpha")

    assert tree["active_id"] == "alpha"
    assert project["id"] == "alpha"
    assert calls == [
        ("projects.tree", {"preview_limit": 3, "session_limit": 2000}),
        (
            "projects.project_sessions",
            {"project_id": "alpha", "session_limit": 5000},
        ),
    ]


@pytest.mark.asyncio
async def test_resume_uses_desktop_rest_transcript_when_live_rpc_is_empty(monkeypatch):
    client = object.__new__(RemoteHermesClient)
    client.profile = ""
    client.session_id = None
    client.stored_session_id = None

    async def request(method, params):
        assert method == "session.resume"
        return {"session_id": "live-1", "session_key": "stored-1", "messages": []}

    async def get_session_messages(session_id):
        assert session_id == "stored-1"
        return [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "get_session_messages", get_session_messages)

    result = await client.resume_session("stored-1")

    assert [item["content"] for item in result["messages"]] == ["Question", "Answer"]
    assert client.session_id == "live-1"
    assert client.stored_session_id == "stored-1"


def _menu_item_labels(root: ft.Control) -> list[str]:
    labels: list[str] = []
    for item in walk_controls(root):
        if not isinstance(item, ft.PopupMenuItem):
            continue
        content = getattr(item, "content", None)
        if content is None:
            continue
        if isinstance(content, ft.Control):
            if isinstance(content, ft.Text):
                labels.append(str(content.value or ""))
        elif isinstance(content, str):
            labels.append(content)
    return labels


def test_session_menu_contains_all_lifecycle_actions(tmp_path):
    """Only lifecycle actions supported by the current Remote API are shown."""
    app = make_app(tmp_path)
    app.remote_client = SimpleNamespace(
        stored_session_id=None,
        state="open",
    )
    view = SessionsView(app)
    view.sessions = [{"id": "s1", "title": "Testing", "source": "desktop", "message_count": 1}]

    root = view.build()
    items = _menu_item_labels(root)

    assert "Rename session" in items or "Renomear sessão" in items
    assert "Branch session" in items or "Criar ramificação" in items
    assert "Archive session" in items or "Arquivar sessão" in items
    assert "Delete session" in items or "Excluir sessão" in items


def test_active_session_menu_hides_delete(tmp_path):
    """Delete is absent for the currently active session."""
    app = make_app(tmp_path)
    app.remote_client = SimpleNamespace(
        stored_session_id="active-1",
        state="open",
    )
    view = SessionsView(app)
    view.sessions = [
        {"id": "active-1", "title": "Active work", "source": "desktop", "message_count": 3}
    ]

    root = view.build()
    menu_items = _menu_item_labels(root)

    assert "Continue session" in menu_items or "Continuar sessão" in menu_items
    assert "Rename session" in menu_items or "Renomear sessão" in menu_items
    assert "Branch session" in menu_items or "Criar ramificação" in menu_items
    assert "Archive session" in menu_items or "Arquivar sessão" in menu_items
    assert "Delete session" not in menu_items and "Excluir sessão" not in menu_items


def test_remote_client_guard_raises_when_offline(tmp_path):
    app = make_app(tmp_path)
    app.remote_client = SimpleNamespace(state="closed")
    view = SessionsView(app)
    view.sessions = [{"id": "s1", "title": "Offline test", "source": "desktop"}]
    view.build()

    with pytest.raises(RuntimeError, match="offline|Offline"):
        view._remote_client()


def test_remove_session_locally_also_removes_pin(tmp_path):
    app = make_app(tmp_path)
    app.remote_client = SimpleNamespace(state="open")
    view = SessionsView(app)
    view.sessions = [
        {"id": "keep", "title": "Keep", "source": "desktop"},
        {"id": "remove", "title": "Remove me", "source": "telegram"},
    ]
    view.pinned_ids = ["remove", "keep"]
    view.build()

    view._remove_session_locally("remove")

    assert [view._id(item) for item in view.sessions] == ["keep"]
    assert view.pinned_ids == ["keep"]


@pytest.mark.asyncio
async def test_archive_session_removes_from_list_and_shows_success(tmp_path):
    app = make_app(tmp_path)

    class Client:
        state = "open"

        async def archive_session(self, sid, archived=True):
            assert archived is True
            return True

    app.remote_client = Client()
    view = SessionsView(app)
    view.sessions = [
        {"id": "s1", "title": "Archive me", "source": "desktop"},
        {"id": "s2", "title": "Keep", "source": "desktop"},
    ]
    view.build()

    ok = await view._archive_session("s1", "Archive me")

    assert ok is True
    assert [view._id(item) for item in view.sessions] == ["s2"]


@pytest.mark.asyncio
async def test_archive_session_failure_keeps_list(tmp_path):
    app = make_app(tmp_path)

    class Client:
        state = "open"

        async def archive_session(self, sid, archived=True):
            return False

    app.remote_client = Client()
    view = SessionsView(app)
    view.sessions = [{"id": "s1", "title": "Archive me", "source": "desktop"}]
    view.build()

    ok = await view._archive_session("s1", "Archive me")

    assert ok is False
    assert [view._id(item) for item in view.sessions] == ["s1"]


@pytest.mark.asyncio
async def test_branch_active_session_uses_live_rpc_without_second_resume(tmp_path):
    app = make_app(tmp_path)
    calls = []

    class Client:
        state = "open"
        session_id = "live-parent"
        stored_session_id = "stored-parent"

        async def branch_active_session(self, *, title):
            calls.append(("branch-active", title))
            return {
                "session_id": "live-branch",
                "stored_session_id": "stored-branch",
                "title": title,
                "messages": [{"role": "user", "content": "history"}],
            }

        async def fork_session(self, session_id, *, title):
            raise AssertionError("active branch must not use stored REST fork")

    app.remote_client = Client()
    app.activate_remote_session_result = lambda result, title: calls.append(
        ("activate", result["stored_session_id"], title)
    )
    app.resume_remote_session = lambda session_id, title: calls.append(("resume", session_id))
    view = SessionsView(app)

    assert await view._branch_session("stored-parent", "Mobile branch") is True
    assert calls == [
        ("branch-active", "Mobile branch"),
        ("activate", "stored-branch", "Mobile branch"),
    ]


@pytest.mark.asyncio
async def test_branch_inactive_session_uses_stored_fork_then_resume(tmp_path):
    app = make_app(tmp_path)
    calls = []

    class Client:
        state = "open"
        session_id = "live-current"
        stored_session_id = "stored-current"

        async def branch_active_session(self, *, title):
            raise AssertionError("inactive branch must not replace the active runtime")

        async def fork_session(self, session_id, *, title):
            calls.append(("fork", session_id, title))
            return {"id": "stored-branch", "title": title}

    app.remote_client = Client()

    async def resume(session_id, title):
        calls.append(("resume", session_id, title))

    app.resume_remote_session = resume
    view = SessionsView(app)

    assert await view._branch_session("stored-other", "Stored branch") is True
    assert calls == [
        ("fork", "stored-other", "Stored branch"),
        ("resume", "stored-branch", "Stored branch"),
    ]
