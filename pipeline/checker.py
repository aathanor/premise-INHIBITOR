"""
Async premise-checking functions.

All functions call Ollama directly via httpx (no subprocess).
They parse the JSON verdict returned by the checker model and
return typed dataclass instances so the loop can act on them cleanly.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx

from config import OLLAMA_BASE_URL, CHECKER_MODEL, CHECKER_OPTIONS
from pipeline.prompts import (
    INBOUND_CHECK_SYSTEM,
    INBOUND_CHECK_USER,
    OUTBOUND_CHECK_SYSTEM,
    OUTBOUND_CHECK_USER,
)

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────

@dataclass
class InboundResult:
    has_premise_error: bool
    errors: list[dict] = field(default_factory=list)
    rewritten_query: str = ""      # equals original query when no errors found
    raw: str = ""                  # raw model output, for debugging


@dataclass
class OutboundResult:
    verdict: str                   # "PASS" | "FLAG"
    issues: list[dict] = field(default_factory=list)
    raw: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ── Ollama helper ─────────────────────────────────────────────────────────

async def _ollama_chat(
    system: str,
    user: str,
    model: str = CHECKER_MODEL,
    options: dict | None = None,
) -> str:
    """
    Send a single-turn chat request to Ollama and return the response text.
    Uses /api/chat (non-streaming).
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options or CHECKER_OPTIONS,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


def _parse_json_reply(raw: str) -> dict:
    """
    Extract and parse a JSON object from a model reply that may contain
    surrounding prose or markdown fences.
    """
    # Strip common markdown fences
    text = raw.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first { … } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from model reply: %s", raw[:300])
    return {}


# ── Public API ────────────────────────────────────────────────────────────

async def check_inbound(query: str) -> InboundResult:
    """
    Examine the user's query for false/faulty premises.
    Returns an InboundResult with a (possibly rewritten) query.
    """
    user_msg = INBOUND_CHECK_USER.format(query=query)
    try:
        raw = await _ollama_chat(INBOUND_CHECK_SYSTEM, user_msg)
        parsed = _parse_json_reply(raw)
    except Exception as exc:
        logger.error("Inbound check failed: %s", exc)
        return InboundResult(has_premise_error=False, rewritten_query=query)

    return InboundResult(
        has_premise_error=bool(parsed.get("has_premise_error", False)),
        errors=parsed.get("errors", []),
        rewritten_query=parsed.get("rewritten_query") or query,
        raw=raw,
    )


async def check_outbound(query: str, response: str) -> OutboundResult:
    """
    Examine the model's response for premise errors or false claims.
    Returns an OutboundResult with verdict PASS or FLAG.
    """
    user_msg = OUTBOUND_CHECK_USER.format(query=query, response=response)
    try:
        raw = await _ollama_chat(OUTBOUND_CHECK_SYSTEM, user_msg)
        parsed = _parse_json_reply(raw)
    except Exception as exc:
        logger.error("Outbound check failed: %s", exc)
        return OutboundResult(verdict="PASS")   # fail open on checker error

    verdict = parsed.get("verdict", "PASS").upper()
    if verdict not in ("PASS", "FLAG"):
        verdict = "PASS"

    return OutboundResult(
        verdict=verdict,
        issues=parsed.get("issues", []),
        raw=raw,
    )
