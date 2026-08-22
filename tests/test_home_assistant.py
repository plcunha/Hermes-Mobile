"""Tests for Home Assistant smart-home tools (desktop parity)."""

import asyncio
from types import SimpleNamespace

from hermes_mobile.tools.home_assistant import (
    ha_call_service_tool,
    ha_get_state_tool,
    ha_list_entities_tool,
    ha_list_services_tool,
)


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


class FakeAsyncClient:
    response: FakeResponse = FakeResponse(200, json_data={})

    def __init__(self, *args, **kwargs):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def request(self, method, url, headers=None, json=None):
        self.calls.append((method, url, headers, json))
        return type(self).response


def _no_config():
    return SimpleNamespace(ha_url="", ha_token="")


def _config():
    return SimpleNamespace(ha_url="https://ha.example", ha_token="secret-token")


def test_all_tools_degrades_without_config(monkeypatch):
    monkeypatch.setattr("hermes_mobile.tools.home_assistant.get_settings", _no_config)

    assert asyncio.run(ha_list_entities_tool())["error"].startswith("Home Assistant")
    assert asyncio.run(ha_list_services_tool())["error"].startswith("Home Assistant")
    assert asyncio.run(ha_get_state_tool("light.x"))["error"].startswith("Home Assistant")
    assert asyncio.run(ha_call_service_tool("light", "turn_on"))["error"].startswith(
        "Home Assistant"
    )


def test_ha_get_state_requires_entity(monkeypatch):
    monkeypatch.setattr("hermes_mobile.tools.home_assistant.get_settings", _config)
    assert asyncio.run(ha_get_state_tool("")) == {"error": "entity_id is required"}


def test_ha_call_service_requires_domain_and_service(monkeypatch):
    monkeypatch.setattr("hermes_mobile.tools.home_assistant.get_settings", _config)
    assert asyncio.run(ha_call_service_tool("", "turn_on")) == {
        "error": "domain and service are required"
    }


def test_ha_list_entities(monkeypatch):
    monkeypatch.setattr("hermes_mobile.tools.home_assistant.get_settings", _config)
    FakeAsyncClient.response = FakeResponse(
        200, json_data=[{"entity_id": "light.kitchen"}, {"entity_id": "switch.desk"}]
    )

    captured = {}

    def factory(*args, **kwargs):
        client = FakeAsyncClient()
        captured["client"] = client
        return client

    monkeypatch.setattr("hermes_mobile.tools.home_assistant.httpx.AsyncClient", factory)

    result = asyncio.run(ha_list_entities_tool())

    assert result["count"] == 2
    assert result["entities"] == ["light.kitchen", "switch.desk"]
    method, url, headers, _ = captured["client"].calls[0]
    assert method == "GET"
    assert url == "https://ha.example/api/states"
    assert headers["Authorization"] == "Bearer secret-token"


def test_ha_call_service_posts_body_and_entity(monkeypatch):
    monkeypatch.setattr("hermes_mobile.tools.home_assistant.get_settings", _config)
    FakeAsyncClient.response = FakeResponse(200, json_data=[])

    captured = {}

    def factory(*args, **kwargs):
        client = FakeAsyncClient()
        captured["client"] = client
        return client

    monkeypatch.setattr("hermes_mobile.tools.home_assistant.httpx.AsyncClient", factory)

    result = asyncio.run(
        ha_call_service_tool(
            "light",
            "turn_on",
            service_data={"brightness_pct": 80},
            entity_id="light.kitchen",
        )
    )

    assert result["ok"] is True
    method, url, headers, body = captured["client"].calls[0]
    assert method == "POST"
    assert url == "https://ha.example/api/services/light/turn_on"
    assert headers["Authorization"] == "Bearer secret-token"
    assert body == {"brightness_pct": 80, "entity_id": "light.kitchen"}
