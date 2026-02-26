"""
Gradio chat UI for the Premise-INHIBITOR app.

Mounts the Gradio interface onto the FastAPI app defined in backend.py,
so everything runs on a single port (default 7860).

Run with:
    uvicorn app:app --host 0.0.0.0 --port 7860 --reload
or simply:
    python app.py
"""
from __future__ import annotations

import asyncio
import logging

import gradio as gr
import uvicorn

from backend import app as fastapi_app         # FastAPI instance
from config import APP_HOST, APP_PORT, GENERATION_MODEL, MAX_ATTEMPTS, REWRITE_INBOUND
from pipeline.loop import run_agentic

logger = logging.getLogger(__name__)


# ── Gradio handler ────────────────────────────────────────────────────────

async def respond(
    message: str,
    chat_history: list[dict],
) -> tuple[str, list[dict]]:
    """
    Called by Gradio on each user submission.

    Gradio 6 uses the "messages" format: each entry is a dict with
    {"role": "user"|"assistant", "content": "..."}.

    Returns:
        Tuple of (cleared_input, updated_history).
    """
    if not message.strip():
        return "", chat_history

    # Convert Gradio messages list → [(user, assistant), ...] tuples for the loop.
    # Pair up consecutive user/assistant messages from history.
    history_tuples: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in chat_history:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant" and pending_user is not None:
            history_tuples.append((pending_user, msg["content"]))
            pending_user = None

    loop_result = await run_agentic(message, history_tuples)

    display = loop_result.response
    if loop_result.status_badge:
        display += loop_result.status_badge

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": display},
    ]
    return "", chat_history


# ── Gradio UI layout ──────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Premise INHIBITOR") as demo:

        gr.Markdown(
            f"""
# Premise INHIBITOR
**Agentic pipeline** that detects and corrects false premises — both in your
questions and in the model's answers — before they propagate.

| Setting | Value |
|---|---|
| Model | `{GENERATION_MODEL}` |
| Max correction attempts | `{MAX_ATTEMPTS}` |
| Inbound rewriting | `{"on" if REWRITE_INBOUND else "off"}` |

Replies annotated with *Inhibitor: …* show what was caught and corrected this turn.
"""
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
        )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask something…",
                label="Your message",
                show_label=False,
                scale=9,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        clear_btn = gr.Button("Clear conversation", variant="secondary")

        # Wire up submit (Enter key or button click)
        msg_box.submit(
            fn=respond,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot],
        )
        send_btn.click(
            fn=respond,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot],
        )
        clear_btn.click(
            fn=lambda: ([], ""),   # empty messages list + clear textbox
            inputs=[],
            outputs=[chatbot, msg_box],
        )

    return demo


# ── Mount Gradio on FastAPI and launch ────────────────────────────────────

demo = build_ui()

# gr.mount_gradio_app mounts the Gradio app into the FastAPI instance,
# so /chat, /health, /models and the UI all live on the same port.
app = gr.mount_gradio_app(fastapi_app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        log_level="info",
    )
