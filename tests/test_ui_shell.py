"""Structural regression tests for the mobile shell and flat UI contract."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import pytest

from hermes_mobile.config.settings import HermesMobileSettings, save_settings
from hermes_mobile.core.agent import Message
from hermes_mobile.main import HermesMobileApp
from hermes_mobile.ui.chat_view import ChatView
from hermes_mobile.ui.gateway_view import GatewayView
from hermes_mobile.ui.tools_view import ToolsView


class FakePage:
    height = 844
    width = 430
    platform = ft.PagePlatform.ANDROID
    theme_mode = ft.ThemeMode.DARK
    on_resize = None

    def __init__(self):
        self.updates = 0
        self.clean_calls = 0
        self.controls = []
        self.overlay = []

    def update(self):
        self.updates += 1

    def clean(self):
        self.clean_calls += 1
        self.controls.clear()

    def add(self, *controls):
        self.controls.extend(controls)


class FakeAgent:
    def __init__(self):
        self.messages = ["stale"]
        self.tools = []
        self.clears = 0

    def clear_conversation(self):
        self.messages = []
        self.clears += 1

    def set_tools(self, tools):
        self.tools = tools


def fake_app():
    page = FakePage()
    agent = FakeAgent()
    settings = SimpleNamespace(
        default_model="openai/gpt-test",
        default_provider="openai",
        openrouter_api_key="",
        openai_api_key="configured",
        anthropic_api_key="",
        gemini_api_key="",
    )
    app = SimpleNamespace(
        page=page,
        agent=agent,
        settings=settings,
        dark_mode=True,
        destinations=[],
    )
    app._navigate_to = lambda destination: app.destinations.append(destination)
    return app


def walk_controls(control: ft.Control):
    """Walk Flet's dataclass graph without depending on private serializers."""
    seen: set[int] = set()
    stack = [control]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        fields = set(getattr(type(current), "__dataclass_fields__", {}))
        # Flet 0.28 controls predate the dataclass control graph used by 0.86.
        # Include the public composition slots so the same assertions inspect
        # the rendered tree on both supported generations.
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
                "tabs",
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


def test_chat_has_single_shell_header_and_desktop_derived_composer():
    app = fake_app()
    view = ChatView(app)

    root = view.build()

    assert isinstance(root, ft.Column)
    assert len(root.controls) == 2  # transcript + composer; app shell owns header
    all_controls = list(walk_controls(root))
    assert not any(isinstance(control, ft.Card) for control in all_controls)
    hero_images = {
        control.src: control
        for control in all_controls
        if isinstance(control, ft.Image)
        and control.src
        in {"hermes-mascot.png", "hermes-mobile-sigil.svg", "hermes-welcome-bg.webp"}
    }
    assert set(hero_images) == {
        "hermes-mascot.png",
        "hermes-mobile-sigil.svg",
        "hermes-welcome-bg.webp",
    }
    assert hero_images["hermes-mascot.png"].semantics_label == "Hermes"
    assert hero_images["hermes-mobile-sigil.svg"].opacity == 0.78
    assert hero_images["hermes-welcome-bg.webp"].opacity == 0.18
    assert view.send_button.bgcolor == "#FFE6CB"


def test_remote_composer_shows_backend_state_and_remote_destinations():
    app = fake_app()
    app.remote_mode = True
    app.remote_model = "openai/gpt-5.6-sol"
    app.remote_client = SimpleNamespace(state="open")
    app.show_remote_sessions = lambda: None
    root = ChatView(app).build()

    texts = [
        control.value
        for control in walk_controls(root)
        if isinstance(control, ft.Text) and control.value
    ]
    assert "gpt-5.6-sol" in texts
    assert "Hermes Remote" not in texts
    assert "Connected to Hermes Remote. Send a message to start." in texts
    assert "Go to Settings > API Key to configure." not in texts

    app.remote_client.state = "closed"
    view = ChatView(app)
    view.build()
    app.remote_client.state = "open"
    view.refresh_welcome()
    refreshed = [
        control.value
        for control in walk_controls(view.chat_list)
        if isinstance(control, ft.Text) and control.value
    ]
    assert "Connected to Hermes Remote. Send a message to start." in refreshed
    assert "Hermes Remote is offline. Open Connections to reconnect." not in refreshed

    view.add_user_message("First turn")
    after_first_turn = [
        control.value
        for control in walk_controls(view.chat_list)
        if isinstance(control, ft.Text) and control.value
    ]
    assert "First turn" in after_first_turn
    assert "Connected to Hermes Remote. Send a message to start." not in after_first_turn


def test_connections_surface_separates_remote_from_messaging_gateway():
    app = fake_app()
    app.remote_mode = True
    app.remote_model = ""
    app.remote_client = SimpleNamespace(state="open")
    app.remote_status = SimpleNamespace(version="0.19.1")
    app.remote_secret_store = SimpleNamespace(load=lambda: {}, save=lambda **kwargs: None)
    app.settings.runtime_mode = "remote"
    app.settings.remote_url = "http://100.98.210.62:9119"
    app.settings.remote_auth_mode = "basic"
    app.settings.remote_username = "joao"
    app.settings.remote_profile = ""
    app.settings.remote_allow_insecure = False
    app.gateway_manager = SimpleNamespace(
        _running=False,
        config=SimpleNamespace(
            enabled=False,
            port=8080,
            platforms={},
            pairing_enabled=True,
        ),
    )
    root = GatewayView(app).build()

    texts = [
        str(control.value)
        for control in walk_controls(root)
        if isinstance(control, ft.Text) and control.value
    ]
    assert "HERMES REMOTE" in texts
    assert "MESSAGING GATEWAY" in texts
    assert "Hermes 0.19.1" in texts
    assert not any(isinstance(control, ft.Card) for control in walk_controls(root))
    if not issubclass(ft.Button, ft.ElevatedButton):
        assert not any(isinstance(control, ft.ElevatedButton) for control in walk_controls(root))


@pytest.mark.asyncio
async def test_save_and_connect_persists_reconnects_and_surfaces_success(tmp_path):
    app = fake_app()
    settings = HermesMobileSettings(data_dir=str(tmp_path))
    settings.runtime_mode = "remote"
    app.settings = settings
    app.remote_mode = True
    app.remote_model = ""
    app.remote_client = SimpleNamespace(state="open")
    app.remote_status = SimpleNamespace(version="0.19.0")
    saved_secrets = {}
    app.remote_secret_store = SimpleNamespace(
        load=lambda: {},
        save=lambda **values: saved_secrets.update(values),
    )
    app.gateway_manager = SimpleNamespace(
        _running=False,
        config=SimpleNamespace(
            enabled=False,
            port=8080,
            platforms={},
            pairing_enabled=True,
        ),
    )
    app.chat_view = SimpleNamespace(clear_chat=lambda **kwargs: None)
    app.content_area = SimpleNamespace(content=None)
    app.remote_error = ""
    events = []
    busy_observations = []

    async def disconnect_remote():
        events.append("disconnect")
        app.remote_client = None

    async def connect_remote(announce=False):
        events.append(("connect", announce))
        busy_observations.append(
            (
                view._saving,
                view._connect_button.disabled,
                view._connection_feedback,
            )
        )
        app.remote_client = SimpleNamespace(state="open")
        app.remote_status = SimpleNamespace(version="0.19.1")
        return True

    app.disconnect_remote = disconnect_remote
    app.connect_remote = connect_remote

    view = GatewayView(app)
    view.build()
    view._runtime_field.value = "remote"
    view._url_field.value = "http://100.109.170.51:9119"
    view._auth_field.value = "basic"
    view._username_field.value = "admin"
    view._password_field.value = "private-value"
    view._profile_field.value = "default"
    view._allow_insecure_field.value = False

    await view._save_remote(connect=True)

    persisted = json.loads(settings.settings_file().read_text(encoding="utf-8"))
    assert persisted["runtime_mode"] == "remote"
    assert persisted["remote_url"] == "http://100.109.170.51:9119"
    assert persisted["remote_auth_mode"] == "basic"
    assert saved_secrets == {"password": "private-value", "token": ""}
    assert events == ["disconnect", ("connect", False)]
    assert busy_observations == [(True, True, "Connecting to Hermes Remote…")]
    assert view._connection_feedback == "Connected to Hermes 0.19.1"
    assert view._saving is False


@pytest.mark.asyncio
async def test_save_and_connect_surfaces_transport_failure(tmp_path):
    app = fake_app()
    settings = HermesMobileSettings(data_dir=str(tmp_path))
    settings.runtime_mode = "local"
    settings.remote_url = "https://working.example.test"
    settings.remote_auth_mode = "token"
    settings.remote_username = "previous-user"
    settings.remote_profile = "previous-profile"
    assert save_settings(settings)
    app.settings = settings
    app.remote_mode = False
    app.remote_model = ""
    app.remote_client = None
    app.remote_status = None
    app.remote_error = "Remote username or password was rejected"
    stored_secrets = {"password": "previous-password", "token": "previous-token"}
    app.remote_secret_store = SimpleNamespace(
        load=lambda: dict(stored_secrets),
        save=lambda **values: stored_secrets.update(values),
    )
    app.gateway_manager = SimpleNamespace(
        _running=False,
        config=SimpleNamespace(
            enabled=False,
            port=8080,
            platforms={},
            pairing_enabled=True,
        ),
    )
    cleared_chat = []
    app.chat_view = SimpleNamespace(clear_chat=lambda **kwargs: cleared_chat.append(kwargs))
    app.content_area = SimpleNamespace(content=None)
    app.current_view = "messaging"

    async def disconnect_remote():
        app.remote_client = None

    async def connect_remote(announce=False):
        return False

    app.disconnect_remote = disconnect_remote
    app.connect_remote = connect_remote

    view = GatewayView(app)
    view.build()
    view._runtime_field.value = "remote"
    view._url_field.value = "http://100.109.170.51:9119"
    view._auth_field.value = "basic"
    view._username_field.value = "admin"
    view._password_field.value = "wrong-value"
    view._profile_field.value = "default"
    view._allow_insecure_field.value = False

    await view._save_remote(connect=True)

    assert view._connection_feedback == (
        "Connection failed: Remote username or password was rejected"
    )
    assert view._connection_feedback_error is True
    assert view._saving is False
    persisted = json.loads(settings.settings_file().read_text(encoding="utf-8"))
    assert persisted["runtime_mode"] == "local"
    assert persisted["remote_url"] == "https://working.example.test"
    assert persisted["remote_auth_mode"] == "token"
    assert persisted["remote_username"] == "previous-user"
    assert persisted["remote_profile"] == "previous-profile"
    assert stored_secrets == {
        "password": "previous-password",
        "token": "previous-token",
    }
    assert cleared_chat == []
    assert app.current_view == "messaging"
    snackbars = [control for control in app.page.overlay if isinstance(control, ft.SnackBar)]
    assert snackbars[-1].content.value == "Remote username or password was rejected"


def test_new_session_clears_ui_and_agent_synchronously():
    app = fake_app()
    view = ChatView(app)
    view.build()
    view.messages.append(Message.user("stale"))
    view.chat_list.controls.append(ft.Text("stale"))

    view.clear_chat()

    assert view.messages == []
    assert app.agent.messages == []
    assert app.agent.clears == 1
    assert len(view.chat_list.controls) == 1  # fresh branded welcome state


def test_busy_state_prevents_duplicate_turns_and_recovers():
    app = fake_app()
    view = ChatView(app)

    view.set_busy(True)
    assert view.input_field.disabled is True
    # Match Desktop: the composer is locked but the send affordance stays
    # available as an explicit interrupt button.
    assert view.send_button.disabled is False
    assert view.send_button.icon == ft.Icons.STOP_ROUNDED

    view.set_busy(False)
    assert view.input_field.disabled is False
    assert view.send_button.disabled is False
    assert view.send_button.icon == ft.Icons.ARROW_UPWARD


def test_tools_surface_uses_flat_rows_not_material_cards():
    app = fake_app()
    root = ToolsView(app).build()

    assert not any(isinstance(control, ft.Card) for control in walk_controls(root))


def test_switch_view_builds_only_requested_surface():
    app = cast(Any, HermesMobileApp.__new__(HermesMobileApp))
    calls: list[str] = []

    def view(name: str):
        return SimpleNamespace(build=lambda: calls.append(name) or ft.Text(name))

    app.chat_view = view("chat")
    app.sessions_view = view("sessions")
    app.tools_view = view("tools")
    app.memory_view = view("memory")
    app.skills_view = view("skills")
    app.gateway_view = view("gateway")
    app.artifacts_view = view("artifacts")
    app.cron_view = view("cron")
    app.plugins_view = view("plugins")
    app.terminal_view = view("terminal")
    app.kanban_view = view("kanban")
    app.settings_view = view("settings")
    app.content_area = SimpleNamespace(content=None)
    app.page = FakePage()
    app._update_app_bar_title = lambda name: None

    app._switch_view("tools")

    assert calls == ["tools"]
    assert isinstance(app.content_area.content, ft.Text)


def test_operational_destination_is_reachable_outside_bottom_bar():
    app = cast(Any, HermesMobileApp.__new__(HermesMobileApp))
    app._views = list(HermesMobileApp.MOBILE_VIEWS)
    app.page = FakePage()
    app.nav = SimpleNamespace(selected_index=0)
    switched: list[str] = []
    app._switch_view = switched.append

    app._navigate_to("terminal")

    assert switched == ["terminal"]
    assert app.nav.selected_index == 0


def test_android_enum_selects_mobile_shell():
    app = cast(Any, HermesMobileApp.__new__(HermesMobileApp))
    app.page = FakePage()
    app.settings = SimpleNamespace(theme="system")

    app._setup_page()

    assert app.is_mobile is True
    assert app.page.on_resize == app._on_page_resize


def test_theme_change_rebuilds_shell_and_preserves_active_view():
    app = cast(Any, HermesMobileApp.__new__(HermesMobileApp))
    app.page = FakePage()
    app.chat_view = object()
    app.current_view = "artifacts"
    events = []
    app._build_ui = lambda: events.append("build")
    app._switch_view = lambda view: events.append(("view", view))
    app._refresh_connection_chrome = lambda: events.append("chrome")

    app.apply_theme("light")

    assert app.page.theme_mode == ft.ThemeMode.LIGHT
    assert app.page.clean_calls == 1
    assert events == ["build", ("view", "artifacts"), "chrome"]


def test_resize_crossing_breakpoint_rebuilds_shell_and_preserves_view():
    app = cast(Any, HermesMobileApp.__new__(HermesMobileApp))
    app.page = FakePage()
    app.page.platform = ft.PagePlatform.LINUX
    app.is_mobile = False
    app.chat_view = object()
    app.current_view = "tools"
    rebuilt: list[bool] = []
    switched: list[str] = []
    app._build_ui = lambda: rebuilt.append(app.is_mobile)
    app._switch_view = switched.append
    chrome_refreshes = []
    app._refresh_connection_chrome = lambda: chrome_refreshes.append(True)
    event = SimpleNamespace(width=430)

    app._on_page_resize(event)

    assert app.is_mobile is True
    assert rebuilt == [True]
    assert switched == ["tools"]
    assert chrome_refreshes == [True]
    assert app.page.clean_calls == 1
    assert app.page.updates == 1


@pytest.mark.asyncio
async def test_remote_connect_attempts_are_serialized():
    app = cast(Any, HermesMobileApp.__new__(HermesMobileApp))
    app._remote_connect_lock = asyncio.Lock()
    active = 0
    max_active = 0

    async def connect_locked(announce=False):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return announce

    app._connect_remote_locked = connect_locked

    results = await asyncio.gather(app.connect_remote(False), app.connect_remote(True))

    assert results == [False, True]
    assert max_active == 1


def test_setup_page_registers_official_hermes_fonts():
    app = cast(Any, HermesMobileApp.__new__(HermesMobileApp))
    app.page = FakePage()
    app.settings = SimpleNamespace(theme="light")

    app._setup_page()

    assert app.page.fonts == {
        "Rules": "fonts/RulesVariable.woff2",
        "Sigurd": "fonts/SigurdVariable.woff2",
        "Courier Prime": "fonts/CourierPrime-Regular.woff2",
    }
    assert app.page.theme.font_family == "Rules"
    assert app.page.dark_theme.font_family == "Rules"
    assert app.page.theme.scaffold_bgcolor == "#F8FAFF"
    assert app.page.dark_theme.scaffold_bgcolor == "#0D2F86"


def test_common_mono_font_uses_hermes_website_font():
    from hermes_mobile.ui.common import MONO_FONT

    assert MONO_FONT == "Courier Prime"


def test_build_theme_preserves_flet_tokens_with_vararg_signature():
    from hermes_mobile.ui.theme import build_theme

    light = build_theme(False)
    dark = build_theme(True)

    assert light.font_family == "Rules"
    assert light.scaffold_bgcolor == "#F8FAFF"
    assert light.color_scheme.primary == "#0053FD"
    assert dark.font_family == "Rules"
    assert dark.scaffold_bgcolor == "#0D2F86"
    assert dark.color_scheme.primary == "#FFE6CB"
