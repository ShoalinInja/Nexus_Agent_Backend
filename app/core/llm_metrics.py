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

from dataclasses import dataclass
from typing import Optional  # re-exported for convenience


@dataclass
class LLMMetrics:
    model: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """
        Accumulate one LLM call into the running turn totals.

        - Token counts SUM across calls.
        - Model is OVERWRITTEN by the last non-empty value, so the final
          generation model wins over upstream classifiers.
        - None / falsy inputs are coerced to 0 / no-op so callers can pass
          `getattr(resp, "usage", None) and resp.usage.prompt_tokens` without
          a separate None-check.
        """
        if model:
            self.model = model
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
