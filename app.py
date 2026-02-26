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

import logging

import gradio as gr
import uvicorn

from backend import app as fastapi_app         # FastAPI instance
from config import APP_HOST, APP_PORT, GENERATION_MODEL, CHECKER_MODEL, MAX_ATTEMPTS, REWRITE_INBOUND
from pipeline.loop import run_agentic, LoopResult

logger = logging.getLogger(__name__)


# ── Gradio content helper ─────────────────────────────────────────────────

def _content_str(content) -> str:
    """
    Gradio 6 stores chatbot message content as either a plain string or a
    list of content-part dicts, e.g. [{"type": "text", "text": "..."}].
    Ollama expects a plain string, so normalise here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


# ── Trace formatter ───────────────────────────────────────────────────────

def format_trace(original: str, result: LoopResult) -> str:
    """
    Build a markdown string showing exactly what the inhibitor did this turn.
    Displayed in the trace panel below the chatbot.
    """
    lines: list[str] = ["## Inhibitor trace — last turn\n"]

    # ── INBOUND ──────────────────────────────────────────────────────────
    lines.append("### Inbound (query)")
    if not REWRITE_INBOUND:
        lines.append("*Inbound checking is disabled.*\n")
    elif result.inbound is None:
        lines.append("*Not run.*\n")
    else:
        inbound = result.inbound
        lines.append(f"- **Original query:** {original}")
        if inbound.has_premise_error:
            lines.append(f"- **Rewritten query:** {inbound.rewritten_query}")

        premises = inbound.all_premises or [
            {"premise": e.get("premise", "?"), "correct": False, "correction": e.get("correction", "?")}
            for e in inbound.errors
        ]
        if premises:
            lines.append(f"- **Premises identified ({len(premises)}):**")
            for i, p in enumerate(premises, 1):
                if p.get("correct", True):
                    lines.append(f"  {i}. ✅ _{p.get('premise', '?')}_")
                else:
                    lines.append(f"  {i}. ❌ _{p.get('premise', '?')}_")
                    lines.append(f"     **Correction:** {p.get('correction', '?')}")
        else:
            lines.append("- **Verdict:** ✅ no premises flagged")
        lines.append("")

    # ── OUTBOUND ─────────────────────────────────────────────────────────
    lines.append("### Outbound (response)")
    lines.append(f"- **Attempts:** {result.attempts} / {MAX_ATTEMPTS}\n")

    if not result.all_outbounds:
        lines.append("*Not run (generation failed).*\n")
    else:
        for attempt_num, outbound in enumerate(result.all_outbounds, 1):
            verdict_label = "✅ PASS" if outbound.passed else "⚠️ FLAG"
            lines.append(f"#### Attempt {attempt_num} — {verdict_label}")

            # For flagged attempts, show the rejected response so the reader
            # can see what was generated and why it was sent back for correction.
            if not outbound.passed and attempt_num <= len(result.all_responses):
                rejected_text = result.all_responses[attempt_num - 1]
                lines.append("<details><summary>Rejected response</summary>\n")
                lines.append(f"{rejected_text}\n")
                lines.append("</details>\n")

            claims = outbound.all_claims or [
                {"claim": iss.get("claim", "?"), "verdict": "FLAG", "reason": iss.get("reason", "?")}
                for iss in outbound.issues
            ]
            if claims:
                for i, c in enumerate(claims, 1):
                    if c.get("verdict", "PASS") == "PASS":
                        lines.append(f"  {i}. ✅ _{c.get('claim', '?')}_")
                    else:
                        lines.append(f"  {i}. ❌ _{c.get('claim', '?')}_")
                        lines.append(f"     **Reason:** {c.get('reason', '?')}")
            else:
                lines.append("  *No claims identified.*")
            lines.append("")

    return "\n".join(lines)


# ── Gradio handler ────────────────────────────────────────────────────────

async def respond(
    message: str,
    chat_history: list[dict],
) -> tuple[str, list[dict], str]:
    """
    Called by Gradio on each user submission.

    Returns:
        (cleared_input, updated_history, trace_markdown)
    """
    if not message.strip():
        return "", chat_history, ""

    # Convert Gradio messages list → [(user, assistant), ...] tuples for the loop.
    # _content_str handles Gradio 6 storing content as list-of-parts instead of str.
    history_tuples: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in chat_history:
        if msg["role"] == "user":
            pending_user = _content_str(msg["content"])
        elif msg["role"] == "assistant" and pending_user is not None:
            history_tuples.append((pending_user, _content_str(msg["content"])))
            pending_user = None

    loop_result = await run_agentic(message, history_tuples)

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": loop_result.response},
    ]
    trace = format_trace(message, loop_result)
    return "", chat_history, trace


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
| Generation model | `{GENERATION_MODEL}` |
| Checker model | `{CHECKER_MODEL}` |
| Max correction attempts | `{MAX_ATTEMPTS}` |
| Inbound rewriting | `{"on" if REWRITE_INBOUND else "off"}` |
"""
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
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

        with gr.Accordion("Inhibitor trace", open=True):
            trace_panel = gr.Markdown(
                value="*Send a message to see the inhibitor trace.*"
            )

        # Wire up submit (Enter key or button click)
        msg_box.submit(
            fn=respond,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot, trace_panel],
        )
        send_btn.click(
            fn=respond,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot, trace_panel],
        )
        clear_btn.click(
            fn=lambda: ([], "", "*Send a message to see the inhibitor trace.*"),
            inputs=[],
            outputs=[chatbot, msg_box, trace_panel],
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
