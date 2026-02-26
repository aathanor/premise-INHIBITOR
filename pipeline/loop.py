"""
Agentic inhibition loop.

Flow for each user turn:
  1. [INBOUND]  Check user query for bad premises → optionally rewrite it.
  2. [GENERATE] Send (rewritten) query + history to Ollama.
  3. [OUTBOUND] Check the response for false claims / premise acceptance.
  4. If FLAG   → build correction prompt, go to step 2 (up to MAX_ATTEMPTS).
  5. Return final response + metadata about what was inhibited.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from config import (
    OLLAMA_BASE_URL,
    GENERATION_MODEL,
    OLLAMA_OPTIONS,
    MAX_ATTEMPTS,
    REWRITE_INBOUND,
)
from pipeline.checker import check_inbound, check_outbound, InboundResult, OutboundResult
from pipeline.prompts import CORRECTION_SYSTEM, CORRECTION_USER, format_issues

logger = logging.getLogger(__name__)

# ── Result type ───────────────────────────────────────────────────────────

@dataclass
class LoopResult:
    response: str
    attempts: int = 0
    inbound: InboundResult | None = None
    final_outbound: OutboundResult | None = None
    # Human-readable summary of what the inhibitor did this turn
    status_lines: list[str] = field(default_factory=list)

    @property
    def status_badge(self) -> str:
        if not self.status_lines:
            return ""
        return "\n\n---\n*Inhibitor:* " + " · ".join(self.status_lines)


# ── Ollama generation helper ──────────────────────────────────────────────

async def _generate(
    messages: list[dict],
    model: str = GENERATION_MODEL,
) -> str:
    """Non-streaming /api/chat call for the main generation model."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": OLLAMA_OPTIONS,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ── History helpers ───────────────────────────────────────────────────────

def _build_messages(
    query: str,
    history: list[tuple[str, str]],
) -> list[dict]:
    """
    Convert Gradio-style history [(user, assistant), ...] + current query
    into Ollama messages list.
    """
    messages: list[dict] = []
    for user_turn, assistant_turn in history:
        messages.append({"role": "user", "content": user_turn})
        if assistant_turn:
            messages.append({"role": "assistant", "content": assistant_turn})
    messages.append({"role": "user", "content": query})
    return messages


# ── Main loop ─────────────────────────────────────────────────────────────

async def run_agentic(
    user_query: str,
    history: list[tuple[str, str]],
) -> LoopResult:
    """
    Run the full premise-inhibition loop for one user turn.

    Args:
        user_query: The raw text the user just submitted.
        history:    List of (user, assistant) string pairs from prior turns.

    Returns:
        LoopResult with the final response and inhibition metadata.
    """
    result = LoopResult(response="")
    status: list[str] = []

    # ── Step 1: Inbound check ─────────────────────────────────────────────
    working_query = user_query
    if REWRITE_INBOUND:
        inbound = await check_inbound(user_query)
        result.inbound = inbound

        if inbound.has_premise_error:
            n = len(inbound.errors)
            status.append(
                f"query rewritten ({n} premise {'error' if n == 1 else 'errors'} corrected)"
            )
            logger.info(
                "Inbound rewrite: %d error(s). Original: %r  →  Rewritten: %r",
                n, user_query, inbound.rewritten_query,
            )
            working_query = inbound.rewritten_query
        else:
            status.append("query OK")

    # ── Steps 2–4: Generate + outbound loop ───────────────────────────────
    prompt = working_query
    response = ""
    attempt = 0

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        messages = _build_messages(prompt, history)

        try:
            response = await _generate(messages)
        except httpx.HTTPError as exc:
            logger.error("Generation request failed: %s", exc)
            result.response = f"[Error contacting Ollama: {exc}]"
            result.attempts = attempt
            result.status_lines = status
            return result

        # ── Step 3: Outbound check ────────────────────────────────────────
        verdict = await check_outbound(user_query, response)
        result.final_outbound = verdict

        if verdict.passed:
            if attempt > 1:
                status.append(f"output corrected after {attempt} attempt(s)")
            else:
                status.append("output OK")
            break

        # FLAG: build correction and loop
        logger.info(
            "Outbound FLAG on attempt %d/%d: %s",
            attempt, MAX_ATTEMPTS, verdict.issues,
        )
        if attempt < MAX_ATTEMPTS:
            issues_block = format_issues(verdict.issues)
            prompt = CORRECTION_USER.format(
                issues_block=issues_block,
                original_query=user_query,
            )
            # Insert the previous (flagged) response as assistant turn so the
            # model has context when self-correcting.
            history = history + [(user_query, response)]
        else:
            n = len(verdict.issues)
            status.append(
                f"⚠ {n} unresolved issue{'s' if n != 1 else ''} after {MAX_ATTEMPTS} attempt(s)"
            )

    result.response = response
    result.attempts = attempt
    result.status_lines = status
    return result
