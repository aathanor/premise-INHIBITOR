"""
FastAPI backend for the Premise-INHIBITOR app.

Exposes:
  POST /chat       – run the full agentic loop for one turn
  GET  /health     – basic health + Ollama reachability check
  GET  /models     – list models available in the local Ollama instance

The Gradio UI (app.py) mounts on this same FastAPI app at /.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import OLLAMA_BASE_URL, GENERATION_MODEL, CHECKER_MODEL, SHOW_INHIBITOR_STATUS
from pipeline.loop import run_agentic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Premise-INHIBITOR",
    description="Agentic LLM pipeline that detects and corrects premise errors.",
    version="0.1.0",
)


# ── Request / response models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    # Gradio-style history: list of [user, assistant] pairs
    history: list[list[str]] = []


class ChatResponse(BaseModel):
    response: str
    attempts: int
    inbound_rewritten: bool
    outbound_passed: bool
    status: str          # human-readable inhibition summary


# ── Routes ────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Run one conversational turn through the full premise-inhibition loop.
    """
    # Convert [[user, assistant], ...] to [(user, assistant), ...]
    history = [tuple(pair) for pair in req.history if len(pair) == 2]

    loop_result = await run_agentic(req.message, history)

    response_text = loop_result.response
    if SHOW_INHIBITOR_STATUS and loop_result.status_badge:
        response_text += loop_result.status_badge

    return ChatResponse(
        response=response_text,
        attempts=loop_result.attempts,
        inbound_rewritten=bool(
            loop_result.inbound and loop_result.inbound.has_premise_error
        ),
        outbound_passed=bool(
            loop_result.final_outbound and loop_result.final_outbound.passed
        ),
        status=" · ".join(loop_result.status_lines),
    )


@app.get("/health")
async def health() -> JSONResponse:
    """Check that Ollama is reachable and the configured models exist."""
    details: dict = {
        "ollama_url": OLLAMA_BASE_URL,
        "generation_model": GENERATION_MODEL,
        "checker_model": CHECKER_MODEL,
        "ollama_reachable": False,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            tags = resp.json()
            available = [m["name"] for m in tags.get("models", [])]
            details["ollama_reachable"] = True
            details["available_models"] = available
            details["generation_model_found"] = any(
                GENERATION_MODEL in m for m in available
            )
            details["checker_model_found"] = any(
                CHECKER_MODEL in m for m in available
            )
    except Exception as exc:
        details["error"] = str(exc)

    status_code = 200 if details["ollama_reachable"] else 503
    return JSONResponse(content=details, status_code=status_code)


@app.get("/models")
async def list_models() -> JSONResponse:
    """Proxy /api/tags from Ollama for the UI model selector."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
