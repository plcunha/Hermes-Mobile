"""Tests for media tools (text-to-speech parity)."""

from types import SimpleNamespace

from hermes_mobile.tools.media_tools import text_to_speech_tool


class FakeResponse:
    def __init__(self, status_code, content=b"", text="", json_data=None):
        self.status_code = status_code
        self.content = content
        self.text = text
        self._json = json_data

    def json(self):
        return self._json


class FakeAsyncClient:
    response: FakeResponse = FakeResponse(200, content=b"MP3DATA")

    def __init__(self, *args, **kwargs):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append((url, headers, json))
        return type(self).response


def _no_key_settings():
    return SimpleNamespace(
        openrouter_api_key=None,
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
    )


def _key_settings(tmp_path):
    return SimpleNamespace(
        openrouter_api_key=None,
        openai_api_key="secret",
        anthropic_api_key=None,
        gemini_api_key=None,
        get_data_dir=lambda: tmp_path,
    )


def test_text_to_speech_degrades_without_key(monkeypatch):
    monkeypatch.setattr("hermes_mobile.tools.media_tools.get_settings", _no_key_settings)
    import asyncio

    result = asyncio.run(text_to_speech_tool("hello"))
    assert "error" in result
    assert "API key" in result["error"]


def test_text_to_speech_requires_text(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_mobile.tools.media_tools.get_settings", lambda: _key_settings(tmp_path)
    )
    import asyncio

    result = asyncio.run(text_to_speech_tool("   "))
    assert result == {"error": "text is required"}


def test_text_to_speech_saves_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_mobile.tools.media_tools.get_settings", lambda: _key_settings(tmp_path)
    )
    FakeAsyncClient.response = FakeResponse(200, content=b"MP3DATA")

    captured = {}

    def factory(*args, **kwargs):
        client = FakeAsyncClient()
        captured["client"] = client
        return client

    monkeypatch.setattr("hermes_mobile.tools.media_tools.httpx.AsyncClient", factory)

    import asyncio

    result = asyncio.run(text_to_speech_tool("hello world", agent=None))

    assert result["bytes"] == 7
    assert result["path"].endswith(".mp3")
    assert result["path"].startswith(str(tmp_path))
    url, headers, payload = captured["client"].calls[0]
    assert url.endswith("/audio/speech")
    assert headers["Authorization"] == "Bearer secret"
    assert payload["model"] == "openai/gpt-4o-mini-tts"
    assert payload["input"] == "hello world"
    assert payload["voice"] == "alloy"
