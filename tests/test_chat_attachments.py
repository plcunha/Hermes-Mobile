"""Tests for chat attachment prompt assembly."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import flet as ft

from hermes_mobile.ui.attachments import (
    MAX_INLINE_TEXT_CHARS,
    attachment_from_picker_file,
    attachments_to_prompt_context,
    copy_path_attachment,
)
from hermes_mobile.ui.chat_view import ChatView


class FakePage:
    height = 844
    width = 430
    platform = ft.PagePlatform.ANDROID
    theme_mode = ft.ThemeMode.DARK

    def __init__(self):
        self.overlay = []
        self.updates = 0

    def update(self):
        self.updates += 1

    def set_clipboard(self, text):
        self.clipboard = text


class Picked:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.bytes = data
        self.path = None


def fake_app(tmp_path: Path):
    settings = SimpleNamespace(
        default_model="openai/gpt-test",
        default_provider="openai",
        openrouter_api_key="configured",
        openai_api_key="configured",
        anthropic_api_key="",
        gemini_api_key="",
        get_data_dir=lambda: tmp_path,
    )
    return SimpleNamespace(
        page=FakePage(),
        agent=SimpleNamespace(),
        settings=settings,
        dark_mode=True,
        remote_mode=False,
        destinations=[],
        _navigate_to=lambda destination: None,
    )


def test_text_attachment_is_inlined_with_metadata(tmp_path: Path):
    attachment = attachment_from_picker_file(Picked("notes.md", b"# hello\nworld"), tmp_path)

    context = attachments_to_prompt_context([attachment])

    assert "notes.md" in context
    assert "kind='text'" in context
    assert "# hello" in context
    assert "</attachment>" in context


def test_large_text_attachment_is_truncated(tmp_path: Path):
    attachment = attachment_from_picker_file(
        Picked("large.txt", b"a" * (MAX_INLINE_TEXT_CHARS + 100)), tmp_path
    )

    context = attachments_to_prompt_context([attachment])

    assert attachment.truncated is True
    assert "truncated at mobile inline limit" in context


def test_image_attachment_is_saved_and_referenced_by_path(tmp_path: Path):
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    attachment = attachment_from_picker_file(Picked("photo.png", png), tmp_path)

    assert attachment.kind == "image"
    assert attachment.local_path
    assert Path(attachment.local_path).exists()
    context = attachments_to_prompt_context([attachment])
    assert "vision_analyze" in context
    assert attachment.local_path in context


def test_blocked_attachment_type_is_rejected(tmp_path: Path):
    try:
        attachment_from_picker_file(Picked("app.apk", b"not really"), tmp_path)
    except ValueError as exc:
        assert "Blocked attachment type" in str(exc)
    else:
        raise AssertionError("apk attachment should be rejected")


def test_chat_consumes_attachments_into_outgoing_prompt(tmp_path: Path):
    app = fake_app(tmp_path)
    view = ChatView(app)
    view.pending_attachments.append(
        copy_path_attachment(tmp_path / "note.txt", tmp_path)
        if False
        else attachment_from_picker_file(Picked("note.txt", b"body"), tmp_path)
    )
    view._refresh_attachments_row()

    prompt = view._consume_pending_attachments("summarize it")

    assert "Attached files for this turn" in prompt
    assert "body" in prompt
    assert "User message:\nsummarize it" in prompt
    assert view.pending_attachments == []
    assert view.attachments_row.visible is False


def test_chat_build_adds_file_picker_to_overlay(tmp_path: Path):
    app = fake_app(tmp_path)
    view = ChatView(app)
    root = view.build()

    assert root is not None
    assert view.file_picker in app.page.overlay


class FakePicker:
    def __init__(self, files):
        self.files = files

    async def pick_files(self, **kwargs):
        return self.files


async def test_remote_chat_rejects_binary_attachment_paths(tmp_path: Path):
    app = fake_app(tmp_path)
    app.remote_mode = True
    view = ChatView(app)
    view.file_picker = FakePicker([Picked("photo.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64)])

    await view._pick_attachments()

    assert view.pending_attachments == []
    snack_texts = [
        getattr(getattr(item, "content", None), "value", "") for item in app.page.overlay
    ]
    assert any("Remote attachments currently support text only" in text for text in snack_texts)


class AudioAgent:
    def __init__(self):
        self.paths = []

    async def transcribe_audio_file(self, path: Path) -> str:
        self.paths.append(path)
        return "hello from audio"


async def test_voice_button_transcribes_audio_file_into_composer(tmp_path: Path):
    app = fake_app(tmp_path)
    app.agent = AudioAgent()
    view = ChatView(app)
    view.file_picker = FakePicker([Picked("voice.mp3", b"ID3" + b"0" * 64)])

    await view._pick_voice_audio()

    assert view.input_field.value == "hello from audio"
    assert len(app.agent.paths) == 1
    assert app.agent.paths[0].exists()
    assert any(
        getattr(getattr(item, "content", None), "value", "") == "Audio transcribed into composer"
        for item in app.page.overlay
    )


async def test_voice_button_is_honest_in_remote_mode(tmp_path: Path):
    app = fake_app(tmp_path)
    app.remote_mode = True
    app.agent = AudioAgent()
    view = ChatView(app)
    view.file_picker = FakePicker([Picked("voice.mp3", b"ID3" + b"0" * 64)])

    await view._pick_voice_audio()

    assert app.agent.paths == []
    assert any(
        "Remote voice needs backend /api/audio/transcribe"
        in getattr(getattr(item, "content", None), "value", "")
        for item in app.page.overlay
    )


class FakeServicesPage(FakePage):
    def __init__(self):
        super().__init__()
        self.services = []


def test_file_picker_registers_as_service_when_available(tmp_path: Path):
    app = fake_app(tmp_path)
    app.page = FakeServicesPage()

    view = ChatView(app)

    assert view.file_picker in app.page.services
    assert view.file_picker not in app.page.overlay


def test_file_picker_uses_overlay_for_legacy_flet_page(tmp_path: Path):
    app = fake_app(tmp_path)

    view = ChatView(app)

    assert view.file_picker in app.page.overlay
