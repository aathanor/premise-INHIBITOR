"""
Central configuration for the Premise-INHIBITOR app.
Edit these values to match your local setup.
"""
from __future__ import annotations
import os

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Model used for BOTH generation and premise-checking.
# Override with a lighter model for the checker if speed matters.
# qwen3:8b — fast generation; qwen2.5:14b — reliable structured JSON for checker.
# Override either with env vars: GENERATION_MODEL=... CHECKER_MODEL=... python app.py
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "qwen3:8b")
CHECKER_MODEL: str = os.getenv("CHECKER_MODEL", "qwen2.5:14b")

# Ollama generation parameters
OLLAMA_OPTIONS: dict = {
    "temperature": 0.7,
    "num_predict": 4096,   # generous — qwen3 thinking blocks can be long
}
CHECKER_OPTIONS: dict = {
    "temperature": 0.0,   # deterministic for judgements
    "num_predict": 2048,
}

# ── Agentic loop ───────────────────────────────────────────────────────────
MAX_ATTEMPTS: int = int(os.getenv("MAX_ATTEMPTS", "3"))

# If True, rewrite the user's query before sending to the main model.
# If False, only the outbound responses are checked/corrected.
REWRITE_INBOUND: bool = os.getenv("REWRITE_INBOUND", "true").lower() == "true"

# ── Server ─────────────────────────────────────────────────────────────────
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "7860"))

# ── UI ─────────────────────────────────────────────────────────────────────
SHOW_INHIBITOR_STATUS: bool = True   # append per-turn status badge to replies
