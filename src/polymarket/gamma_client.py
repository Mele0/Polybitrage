from __future__ import annotations

import json
import asyncio
from typing import Any

import httpx


class GammaClient:
    """Tiny Gamma helper used only when a pair YAML does not already contain token IDs."""

    def __init__(self, base_url: str = "https://gamma-api.polymarket.com", timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GammaClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def market_by_slug(self, slug: str) -> dict[str, Any] | None:
        response = None
        for attempt in range(4):
            response = await self._client.get("/markets", params={"slug": slug})
            if response.status_code == 429 and attempt < 3:
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if response.status_code == 404:
                return None
            response.raise_for_status()
            break
        if response is None:
            return None
        payload = response.json()
        if isinstance(payload, list):
            return payload[0] if payload else None
        if isinstance(payload, dict):
            markets = payload.get("markets") or payload.get("data")
            if isinstance(markets, list):
                return markets[0] if markets else None
            return payload
        return None

    async def event_by_slug(self, slug: str) -> dict[str, Any] | None:
        response = None
        for attempt in range(4):
            response = await self._client.get(f"/events/slug/{slug}")
            if response.status_code == 429 and attempt < 3:
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if response.status_code == 404:
                return None
            response.raise_for_status()
            break
        if response is None:
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) else None

    async def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        response = None
        for attempt in range(4):
            response = await self._client.get(path, params=params)
            if response.status_code == 429 and attempt < 3:
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if response.status_code == 404:
                return None
            response.raise_for_status()
            break
        return None if response is None else response.json()

    async def list_events(
        self,
        *,
        tag_id: str | int | None = None,
        series_id: str | int | None = None,
        closed: bool | None = False,
        active: bool | None = True,
        limit: int = 100,
        max_pages: int = 20,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Page through the Gamma `/events` listing.

        Used by World Cup discovery to enumerate every event under a tag/series
        (e.g. tag_id=102232 "FIFA World Cup", series_id=11433 "soccer-fifwc").
        The public endpoint ignores `?gameId=`, so callers group by gameId locally.
        """
        out: list[dict[str, Any]] = []
        offset = 0
        for _ in range(max(1, max_pages)):
            params: dict[str, Any] = {"limit": max(1, min(limit, 500)), "offset": offset}
            if tag_id is not None:
                params["tag_id"] = str(tag_id)
            if series_id is not None:
                params["series_id"] = str(series_id)
            if closed is not None:
                params["closed"] = "true" if closed else "false"
            if active is not None:
                params["active"] = "true" if active else "false"
            if extra_params:
                params.update(extra_params)
            payload = await self._get_json("/events", params)
            rows = payload if isinstance(payload, list) else (payload or {}).get("data") if isinstance(payload, dict) else None
            if not rows:
                break
            out.extend(row for row in rows if isinstance(row, dict))
            if len(rows) < params["limit"]:
                break
            offset += params["limit"]
        return out


def resolve_binary_token_ids_from_market(market: dict[str, Any]) -> tuple[str | None, str | None]:
    token_ids = _parse_jsonish(
        market.get("clobTokenIds") or market.get("clob_token_ids") or market.get("tokenIds") or market.get("tokens")
    )
    if token_ids and isinstance(token_ids[0], dict):
        yes = no = None
        for item in token_ids:
            outcome = str(item.get("outcome") or item.get("name") or "").strip().lower()
            token_id = str(item.get("token_id") or item.get("tokenId") or item.get("id") or "")
            if outcome == "yes":
                yes = token_id
            elif outcome == "no":
                no = token_id
        return yes, no
    yes = str(token_ids[0]) if len(token_ids) >= 1 else None
    no = str(token_ids[1]) if len(token_ids) >= 2 else None
    return yes, no


def _parse_jsonish(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return [part.strip() for part in stripped.split(",") if part.strip()]
    return []
