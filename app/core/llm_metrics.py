"""
app/core/llm_metrics.py

Per-turn LLM observability accumulator.

Each route handler instantiates an LLMMetrics, threads it into every service
call that hits an LLM, and spreads the resulting dict into the assistant
message before persisting. The "last non-empty model wins" rule means the
generation model (the one that produced the final stored content) is what
gets recorded, even when pre-classification calls fire first.

Scope: chat completions only (per spec). Embeddings are intentionally NOT
instrumented.
"""

from dataclasses import dataclass, field
from typing import Optional  # re-exported for convenience


@dataclass
class LLMMetrics:
    """
    Per-turn LLM observability accumulator.

    The four top-level fields are SUMS / LAST-WINS across every call in the
    turn — kept for backward-compatible dashboards and "rough volume" reads.

    The `llm_calls` array is the per-call breakdown, used for cost
    reconstruction: each chat-completion call appends one entry with its own
    model id, token counts, and latency. Iterate this array to compute exact
    USD cost at each model's pricing tier.
    """
    model: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Per-call breakdown — populated by add(). Entries are dicts with keys
    # {model, input_tokens, output_tokens, latency_ms}.
    llm_calls: list = field(default_factory=list)

    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int = 0,
    ) -> None:
        """
        Accumulate one LLM call into the running turn totals AND append a
        per-call entry to ``llm_calls``.

        - Token counts SUM across calls.
        - Model is OVERWRITTEN by the last non-empty value, so the final
          generation model wins over upstream classifiers.
        - None / falsy inputs are coerced to 0 / no-op so callers can pass
          `getattr(resp, "usage", None) and resp.usage.prompt_tokens` without
          a separate None-check.
        - ``latency_ms`` is the duration of THIS call (wall-clock around the
          ``client.chat.completions.create(...)`` invocation). Defaults to 0
          so the helper signature stays backward-compatible.
        """
        if model:
            self.model = model
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0
        self.llm_calls.append({
            "model": model or "",
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "latency_ms": latency_ms or 0,
        })

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.llm_calls,
        }
