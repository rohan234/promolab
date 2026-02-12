"""LLM explanation layer for PromoLab."""

from __future__ import annotations

import json
import os
from urllib import error, request


def _build_prompt(summary_dict: dict) -> str:
    return (
        "You are an analyst assistant for promotion performance.\n"
        "Use only the provided numbers; if not present, say you don't know.\n"
        "Do not invent metrics, dates, or causal claims.\n"
        "Cite concrete values from the input when making statements.\n"
        "Output markdown with sections:\n"
        "1) Explanation\n"
        "2) Recommended experiments (exactly 2 bullet points)\n"
        "3) Suggested promo copy (2 options)\n\n"
        "INPUT_JSON:\n"
        f"{json.dumps(summary_dict, indent=2, default=str)}"
    )


def generate_explanation(summary_dict: dict) -> str:
    """Generate AI explanation from computed summary data.

    Uses OpenAI Responses API if `PROMOLAB_LLM_API_KEY` is set.
    Returns a safe placeholder when key is absent.
    """
    api_key = os.getenv("PROMOLAB_LLM_API_KEY")
    if not api_key:
        return (
            "### AI explanation (stub)\n"
            "LLM key not configured. Set `PROMOLAB_LLM_API_KEY` to enable AI explanation.\n\n"
            "**Experiments**\n"
            "- Test a tighter discount depth on top 1-2 driver items.\n"
            "- Run a short A/B holdout by weekday to isolate incremental lift.\n\n"
            "**Suggested promo copy**\n"
            "- \"Weekend deal: save on your favorite drink lineup.\"\n"
            "- \"Limited-time offer: more value on best-selling menu items.\""
        )

    model = os.getenv("PROMOLAB_LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "input": _build_prompt(summary_dict),
        "temperature": 0.2,
    }

    req = request.Request(
        url="https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return f"AI explanation unavailable (HTTP {exc.code}).\n\n{detail[:500]}"
    except Exception as exc:  # pragma: no cover
        return f"AI explanation unavailable ({exc})."

    text = body.get("output_text")
    if isinstance(text, str) and text.strip():
        return text

    return "AI explanation unavailable (empty model output)."
