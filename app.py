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
from pipeline.checker import check_inbound, InboundResult
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


def _history_tuples(chat_history: list[dict]) -> list[tuple[str, str]]:
    """Convert Gradio messages list → [(user, assistant), ...] tuples for the loop."""
    tuples: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in chat_history:
        if msg["role"] == "user":
            pending_user = _content_str(msg["content"])
        elif msg["role"] == "assistant" and pending_user is not None:
            tuples.append((pending_user, _content_str(msg["content"])))
            pending_user = None
    return tuples


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


# ── Gradio handlers ───────────────────────────────────────────────────────

# Shared sentinel for "clear this state" returns
_HIDE = gr.update(visible=False)
_SHOW = gr.update(visible=True)


async def handle_submit(
    message: str,
    chat_history: list[dict],
) -> tuple:
    """
    First step: run inbound check only.

    - If false premises are found → show the review panel and pause.
    - Otherwise → run the full pipeline immediately.

    Outputs (in order):
        msg_box, chatbot, trace_panel,
        premise_review (Group), orig_label, rewrite_box,
        pending_query (State), pending_hist (State), pending_inbound (State)
    """
    _blank = ("", chat_history, "", _HIDE, "", "", None, None, None)
    if not message.strip():
        return _blank

    hist = _history_tuples(chat_history)

    if REWRITE_INBOUND:
        inbound = await check_inbound(message)
        if inbound.has_premise_error:
            # Pause: show review panel so the user can accept, edit, or discard
            # the suggested rewrite before generation runs.
            return (
                "",                          # clear input
                chat_history,                # chatbot unchanged
                "",                          # trace unchanged
                _SHOW,                       # show review panel
                f"**Original:** {message}",  # orig_label
                inbound.rewritten_query,     # rewrite_box (editable)
                message,                     # pending_query state
                hist,                        # pending_hist state
                inbound,                     # pending_inbound state
            )
        # Inbound passed — fall through to full pipeline with the checked result
        result = await run_agentic(message, hist, precomputed_inbound=inbound)
    else:
        result = await run_agentic(message, hist)

    new_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.response},
    ]
    return (
        "",
        new_history,
        format_trace(message, result),
        _HIDE, "", "",
        None, None, None,
    )


async def handle_keep(
    pending_query: str,
    pending_hist: list,
    pending_inbound: InboundResult,
    chat_history: list[dict],
) -> tuple:
    """User chose to keep their original query despite the flagged premises."""
    result = await run_agentic(
        pending_query, pending_hist,
        precomputed_inbound=pending_inbound,
        generation_query=pending_query,   # original, not the rewrite
    )
    new_history = chat_history + [
        {"role": "user", "content": pending_query},
        {"role": "assistant", "content": result.response},
    ]
    return (
        new_history,
        format_trace(pending_query, result),
        _HIDE, "", "",
        None, None, None,
    )


async def handle_use_rewrite(
    rewrite_text: str,
    pending_query: str,
    pending_hist: list,
    pending_inbound: InboundResult,
    chat_history: list[dict],
) -> tuple:
    """User chose to send the (possibly hand-edited) rewritten query."""
    result = await run_agentic(
        pending_query, pending_hist,
        precomputed_inbound=pending_inbound,
        generation_query=rewrite_text,    # rewrite (may have been edited)
    )
    new_history = chat_history + [
        {"role": "user", "content": pending_query},
        {"role": "assistant", "content": result.response},
    ]
    return (
        new_history,
        format_trace(pending_query, result),
        _HIDE, "", "",
        None, None, None,
    )


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

        # ── Premise review panel (hidden until a false premise is detected) ──
        with gr.Group(visible=False) as premise_review:
            gr.Markdown("### ⚠️ False premise detected")
            orig_label = gr.Markdown("")
            gr.Markdown("**Suggested rewrite** *(you can edit this before sending)*:")
            rewrite_box = gr.Textbox(
                label="",
                show_label=False,
                interactive=True,
                lines=2,
            )
            with gr.Row():
                keep_btn = gr.Button("Keep original", variant="secondary")
                use_btn = gr.Button("Use this rewrite", variant="primary")

        clear_btn = gr.Button("Clear conversation", variant="secondary")

        with gr.Accordion("Inhibitor trace", open=True):
            trace_panel = gr.Markdown(
                value="*Send a message to see the inhibitor trace.*"
            )

        # ── Shared state for the two-step review flow ──────────────────────
        pending_query   = gr.State(None)
        pending_hist    = gr.State(None)
        pending_inbound = gr.State(None)

        # Outputs shared by submit (first step) ─────────────────────────────
        _submit_outputs = [
            msg_box, chatbot, trace_panel,
            premise_review, orig_label, rewrite_box,
            pending_query, pending_hist, pending_inbound,
        ]

        msg_box.submit(fn=handle_submit,
                       inputs=[msg_box, chatbot],
                       outputs=_submit_outputs)
        send_btn.click(fn=handle_submit,
                       inputs=[msg_box, chatbot],
                       outputs=_submit_outputs)

        # Outputs shared by keep / use buttons (second step) ─────────────────
        _review_outputs = [
            chatbot, trace_panel,
            premise_review, orig_label, rewrite_box,
            pending_query, pending_hist, pending_inbound,
        ]

        keep_btn.click(
            fn=handle_keep,
            inputs=[pending_query, pending_hist, pending_inbound, chatbot],
            outputs=_review_outputs,
        )
        use_btn.click(
            fn=handle_use_rewrite,
            inputs=[rewrite_box, pending_query, pending_hist, pending_inbound, chatbot],
            outputs=_review_outputs,
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
