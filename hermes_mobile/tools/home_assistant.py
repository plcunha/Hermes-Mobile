"""Home Assistant smart-home tools for Hermes Mobile (desktop parity).

Ports the Desktop smart-home tool family over the HA REST API:

- ``ha_list_entities`` — enumerate every entity id.
- ``ha_get_state`` — read one entity's state and attributes.
- ``ha_list_services`` — enumerate callable ``domain.service`` pairs.
- ``ha_call_service`` — invoke a service (turn a light on, set a climate, ...).

Authentication uses a long-lived Home Assistant token from the environment
(``HA_URL`` / ``HA_TOKEN``), read via :mod:`hermes_mobile.config.settings`.
The token is a secret and is never persisted to the settings JSON.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import httpx

from hermes_mobile.config.settings import get_settings


def _ha_config() -> Tuple[Optional[str], Optional[str]]:
    settings = get_settings()
    url = str(getattr(settings, "ha_url", "") or "").strip().rstrip("/")
    token = str(getattr(settings, "ha_token", "") or "").strip()
    return (url or None), (token or None)


def _headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def _ha_request(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    url, token = _ha_config()
    if not url or not token:
        return None, {"error": "Home Assistant is not configured (set HA_URL and HA_TOKEN)."}
    full_url = f"{url}{path}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.request(method, full_url, headers=_headers(token), json=json_body)
    except httpx.HTTPError as exc:
        return None, {"error": str(exc)}
    if resp.status_code in (200, 201):
        try:
            return resp.json(), None
        except ValueError:
            return None, {"error": "Home Assistant returned a non-JSON response"}
    return None, {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}


async def ha_list_entities_tool() -> Dict[str, Any]:
    """List all Home Assistant entity ids."""
    data, err = await _ha_request("GET", "/api/states")
    if err:
        return err
    if not isinstance(data, list):
        return {"error": "Unexpected Home Assistant response"}
    entities = [
        entry.get("entity_id")
        for entry in data
        if isinstance(entry, dict) and entry.get("entity_id")
    ]
    return {"count": len(entities), "entities": entities}


async def ha_get_state_tool(entity_id: str) -> Dict[str, Any]:
    """Read one entity's state and attributes."""
    if not str(entity_id or "").strip():
        return {"error": "entity_id is required"}
    data, err = await _ha_request("GET", f"/api/states/{entity_id}")
    if err:
        return err
    if not isinstance(data, dict):
        return {"error": "Unexpected Home Assistant response"}
    return {
        "entity_id": data.get("entity_id"),
        "state": data.get("state"),
        "attributes": data.get("attributes", {}),
    }


async def ha_list_services_tool() -> Dict[str, Any]:
    """List callable Home Assistant services as ``domain.service`` strings."""
    data, err = await _ha_request("GET", "/api/services")
    if err:
        return err
    services: list[str] = []
    if isinstance(data, dict):
        for domain, svcs in data.items():
            if isinstance(svcs, dict):
                services.extend(f"{domain}.{svc}" for svc in svcs)
    services.sort()
    return {"count": len(services), "services": services}


async def ha_call_service_tool(
    domain: str,
    service: str,
    service_data: Optional[Dict[str, Any]] = None,
    entity_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call a Home Assistant service (e.g. domain=light, service=turn_on)."""
    if not str(domain or "").strip() or not str(service or "").strip():
        return {"error": "domain and service are required"}
    body = dict(service_data or {})
    if entity_id:
        body["entity_id"] = str(entity_id)
    data, err = await _ha_request(
        "POST",
        f"/api/services/{domain.strip()}/{service.strip()}",
        json_body=body,
    )
    if err:
        return err
    return {"ok": True, "result": data}
