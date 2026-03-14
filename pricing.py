"""OpenRouter pricing and cost tracking for LLM pipelines.

Fetches per-token pricing from OpenRouter's model catalog API and tracks
cumulative token usage and costs across pipeline phases by inspecting
DSPy's LM history.

Usage:
    from pricing import CostTracker

    tracker = CostTracker("qwen/qwen3.5-35b-a3b")

    with tracker.phase("classification"):
        for note in notes:
            classify(note)

    with tracker.phase("decomposition"):
        for note in notes:
            decompose(note)

    tracker.report()
"""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.request import Request, urlopen

import dspy


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class ModelRate:
    """Per-token pricing from OpenRouter's model catalog."""

    model_id: str
    name: str
    prompt_per_token: float  # USD per token
    completion_per_token: float  # USD per token
    context_length: int = 0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_per_token
            + completion_tokens * self.completion_per_token
        )


@dataclass
class PhaseUsage:
    """Token usage and cost for one pipeline phase."""

    name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    rlm_iterations: int = 0
    rlm_sub_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ── OpenRouter API ────────────────────────────────────────────────────────────


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_model_rate(model_id: str, api_key: str | None = None) -> ModelRate:
    """Fetch per-token pricing from OpenRouter's model catalog.

    Returns zero-rate ModelRate if the API is unreachable or model not found.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return _zero_rate(model_id, "(no API key)")

    try:
        req = Request(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {key}"},
        )
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        for model in data.get("data", []):
            if model.get("id") == model_id:
                pricing = model.get("pricing", {})
                return ModelRate(
                    model_id=model_id,
                    name=model.get("name", model_id),
                    prompt_per_token=float(pricing.get("prompt", "0")),
                    completion_per_token=float(pricing.get("completion", "0")),
                    context_length=model.get("context_length", 0),
                )

        return _zero_rate(model_id, "not found in catalog")
    except Exception as exc:
        return _zero_rate(model_id, str(exc))


def _zero_rate(model_id: str, reason: str = "") -> ModelRate:
    if reason:
        print(f"pricing: {model_id} -- {reason}, costs will show as $0")
    return ModelRate(
        model_id=model_id,
        name=model_id,
        prompt_per_token=0.0,
        completion_per_token=0.0,
    )


# ── LM history inspection ────────────────────────────────────────────────────


def _extract_history_usage(lm, start_index: int) -> tuple[int, int, int, float]:
    """Extract token counts from DSPy LM history entries added since start_index.

    Returns (prompt_tokens, completion_tokens, calls, litellm_cost).

    DSPy stores usage data in two places per entry:
      - entry["usage"]: dict with prompt_tokens, completion_tokens
      - entry["response"].usage: LiteLLM ModelResponse usage object
    We check both, preferring the top-level dict.
    """
    prompt_tokens = 0
    completion_tokens = 0
    calls = 0
    litellm_cost = 0.0

    if not lm or not hasattr(lm, "history"):
        return prompt_tokens, completion_tokens, calls, litellm_cost

    for entry in lm.history[start_index:]:
        if not isinstance(entry, dict):
            continue
        calls += 1

        entry_prompt = 0
        entry_completion = 0

        # Primary: top-level usage dict
        usage = entry.get("usage")
        if isinstance(usage, dict):
            entry_prompt = usage.get("prompt_tokens", 0) or 0
            entry_completion = usage.get("completion_tokens", 0) or 0

        # Fallback: response object's usage attribute
        if entry_prompt == 0 and entry_completion == 0:
            resp = entry.get("response")
            if resp and hasattr(resp, "usage") and resp.usage:
                entry_prompt = getattr(resp.usage, "prompt_tokens", 0) or 0
                entry_completion = getattr(resp.usage, "completion_tokens", 0) or 0

        prompt_tokens += entry_prompt
        completion_tokens += entry_completion

        # DSPy also stores LiteLLM's calculated cost
        cost = entry.get("cost")
        if isinstance(cost, (int, float)):
            litellm_cost += cost

    return prompt_tokens, completion_tokens, calls, litellm_cost


# ── Cost tracker ──────────────────────────────────────────────────────────────


class CostTracker:
    """Track token usage and costs across pipeline phases.

    Uses DSPy's LM history to extract per-call token counts, and
    OpenRouter's pricing API for accurate cost calculation.
    """

    def __init__(self, model_id: str, api_key: str | None = None):
        self.rate = fetch_model_rate(model_id, api_key)
        self.phases: list[PhaseUsage] = []
        self._litellm_total: float = 0.0

    @contextmanager
    def phase(self, name: str):
        """Context manager that tracks all LM calls within its scope."""
        usage = PhaseUsage(name=name)
        lm = dspy.settings.lm
        start = len(lm.history) if lm and hasattr(lm, "history") else 0

        yield usage

        pt, ct, calls, lcost = _extract_history_usage(lm, start)
        usage.prompt_tokens += pt
        usage.completion_tokens += ct
        usage.calls += calls
        self._litellm_total += lcost
        self.phases.append(usage)

    @property
    def total_cost(self) -> float:
        return sum(
            self.rate.cost(p.prompt_tokens, p.completion_tokens) for p in self.phases
        )

    @property
    def total_tokens(self) -> int:
        return sum(p.total_tokens for p in self.phases)

    def report(self):
        """Print a formatted cost report to stdout."""
        pm = self.rate.prompt_per_token * 1_000_000
        cm = self.rate.completion_per_token * 1_000_000

        print(f"\n{'=' * 70}")
        print(f"COST REPORT: {self.rate.name}")
        print(f"Model: {self.rate.model_id}")
        if pm > 0 or cm > 0:
            print(f"Pricing: ${pm:.4f}/M input, ${cm:.4f}/M output")
            print(f"Context window: {self.rate.context_length:,} tokens")
        print(f"{'=' * 70}")

        header = f"{'Phase':<25} {'Calls':>6} {'Input':>10} {'Output':>10} {'Cost':>12}"
        print(header)
        print("-" * 70)

        for p in self.phases:
            cost = self.rate.cost(p.prompt_tokens, p.completion_tokens)
            rlm = ""
            if p.rlm_iterations > 0:
                rlm = f"  [{p.rlm_iterations} iters, {p.rlm_sub_calls} sub]"
            print(
                f"{p.name:<25} {p.calls:>6} "
                f"{p.prompt_tokens:>10,} {p.completion_tokens:>10,} "
                f"${cost:>11.6f}{rlm}"
            )

        print("-" * 70)
        tc = sum(p.calls for p in self.phases)
        tp = sum(p.prompt_tokens for p in self.phases)
        tcomp = sum(p.completion_tokens for p in self.phases)
        total = self.total_cost
        print(f"{'TOTAL':<25} {tc:>6} {tp:>10,} {tcomp:>10,} ${total:>11.6f}")

        # LiteLLM cost is more accurate (includes DSPy internal retries)
        if self._litellm_total > 0:
            print(f"{'TOTAL (provider)':>25}{'':>28}${self._litellm_total:>11.6f}")

        print(f"{'=' * 70}")
