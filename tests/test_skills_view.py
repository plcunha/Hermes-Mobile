from types import SimpleNamespace

import pytest

from hermes_mobile.config.settings import HermesMobileSettings
from hermes_mobile.ui.skills_view import SkillsView


class Page:
    width = 430
    platform = "android"
    theme_mode = None

    def __init__(self):
        self.updated = 0
        self.overlay = []

    def update(self):
        self.updated += 1


class RemoteClient:
    state = "open"

    async def get_remote_skills(self):
        return [
            {"name": "github", "description": "", "category": "GitHub"},
            {"name": "memory", "description": "", "category": "General"},
        ]


def _text_values(control):
    values = []
    value = getattr(control, "value", None)
    if isinstance(value, str):
        values.append(value)
    content = getattr(control, "content", None)
    if content is not None and not isinstance(content, str):
        values.extend(_text_values(content))
    for child in getattr(control, "controls", []) or []:
        values.extend(_text_values(child))
    return values


def make_app(tmp_path, client):
    return SimpleNamespace(
        page=Page(),
        settings=HermesMobileSettings(
            data_dir=str(tmp_path),
            runtime_mode="remote",
            remote_url="https://hermes.example.test",
            remote_profile="default",
        ),
        dark_mode=True,
        skill_manager=SimpleNamespace(get_all_skills=lambda: []),
        remote_client=client,
        current_view="skills",
        content_area=SimpleNamespace(content=None),
    )


@pytest.mark.asyncio
async def test_remote_skills_are_loaded_from_backend_catalog(tmp_path):
    app = make_app(tmp_path, RemoteClient())
    view = SkillsView(app)

    await view.refresh_remote()

    assert [row["name"] for row in view.remote_skills] == ["github", "memory"]
    assert "github" in _text_values(view.build())
    assert "memory" in _text_values(view.build())
    assert view.remote_error == ""


@pytest.mark.asyncio
async def test_remote_skills_offline_state_is_explicit(tmp_path):
    app = make_app(tmp_path, None)
    view = SkillsView(app)

    await view.refresh_remote()

    assert view.remote_skills == []
    assert "Connect to Hermes Remote" in view.remote_error
    assert "Remote skills unavailable" in _text_values(view.build())


class ExportManager:
    def __init__(self):
        self.calls = []

    def get_all_skills(self):
        return []

    def export_skill(self, name, export_path):
        self.calls.append((name, export_path))
        export_path.mkdir(parents=True, exist_ok=True)
        (export_path / name).mkdir(exist_ok=True)
        return True


def test_export_skill_writes_to_app_export_directory(tmp_path):
    manager = ExportManager()
    app = make_app(tmp_path, None)
    app.settings.runtime_mode = "local"
    app.skill_manager = manager
    view = SkillsView(app)
    skill = SimpleNamespace(name="demo", path=tmp_path / "skills" / "demo")
    skill.path.mkdir(parents=True)

    view._export_skill(skill)

    assert manager.calls == [("demo", tmp_path / "exports" / "skills")]
    assert (tmp_path / "exports" / "skills" / "demo").exists()
    assert any(
        "Exported to" in getattr(getattr(item, "content", None), "value", "")
        for item in app.page.overlay
    )
