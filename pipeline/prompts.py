"""
Prompt templates for the premise-inhibition pipeline.

Three roles:
  INBOUND  – detect & rewrite bad premises in the user's query
  OUTBOUND – detect premise errors baked into a model response
  CORRECT  – build a targeted correction prompt to feed back into the loop
"""
from __future__ import annotations

# ── Inbound: check + rewrite user query ───────────────────────────────────

INBOUND_CHECK_SYSTEM = """\
You are a premise auditor. Your job is to identify ALL presuppositions embedded \
in a user's question — both correct and false.

A "premise error" is a presupposition the question takes for granted that is \
actually false or contested — e.g. "Why did Einstein fail maths as a child?" \
presupposes he did, but he did not.

Respond with VALID JSON only, no markdown fences, in this exact schema:
{
  "has_premise_error": true | false,
  "all_premises": [
    {
      "premise": "<a presupposition found in the question>",
      "correct": true | false,
      "correction": "<factual correction — include only when correct is false>"
    }
  ],
  "errors": [
    {
      "premise": "<the false presupposition>",
      "correction": "<the factual correction>"
    }
  ],
  "rewritten_query": "<the query with false premises neutralised, or the original if none>"
}

"errors" must contain only the false premises (those entries in all_premises where correct is false).
"""

INBOUND_CHECK_USER = """\
Audit the following question for premise errors.

Question: {query}
"""

# ── Outbound: check model response ────────────────────────────────────────

OUTBOUND_CHECK_SYSTEM = """\
You are a fact-checking assistant. You will be given an AI-generated response \
and the original question that prompted it.

Your task: evaluate ALL distinct factual claims in the response, deciding for \
each whether it is accurate or contains a false / misleading premise.

Do NOT flag:
  - opinions or predictions clearly marked as such
  - hedged statements ("may", "might", "some argue")
  - minor stylistic issues

Respond with VALID JSON only, no markdown fences:
{
  "verdict": "PASS" | "FLAG",
  "all_claims": [
    {
      "claim": "<a factual claim from the response>",
      "verdict": "PASS" | "FLAG",
      "reason": "<why it is false — include only when verdict is FLAG>"
    }
  ],
  "issues": [
    {
      "claim": "<the problematic claim from the response>",
      "reason": "<why it is false or premise-dependent>"
    }
  ]
}

Top-level "verdict" is FLAG if any claim is FLAG, PASS otherwise.
"issues" must contain only the flagged claims (a subset of all_claims where verdict is FLAG).
"""

OUTBOUND_CHECK_USER = """\
Original question: {query}

AI response to check:
{response}
"""

# ── Correction: build a follow-up prompt ─────────────────────────────────

CORRECTION_SYSTEM = """\
You are a helpful, accurate assistant. You previously gave a response that \
contained one or more factual errors or false presuppositions. \
You must now provide a corrected answer.
"""

CORRECTION_USER = """\
Your previous response contained the following issue(s):

{issues_block}

Please provide a corrected, accurate answer to the original question:
{original_query}
"""

def format_issues(issues: list[dict]) -> str:
    """Turn the list of issue dicts from the outbound check into a readable block."""
    lines = []
    for i, issue in enumerate(issues, 1):
        lines.append(f"{i}. Claim: {issue.get('claim', '')}")
        lines.append(f"   Reason: {issue.get('reason', '')}")
    return "\n".join(lines)
