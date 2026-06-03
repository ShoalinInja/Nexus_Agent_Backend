"""
Tests for app.core.llm_metrics.LLMMetrics.

Covers the per-call breakdown added in the follow-up cycle:
  - add() appends an entry to llm_calls with the four expected keys
  - flat fields sum correctly across multiple calls
  - last non-empty model wins (preserves earlier behaviour)
  - latency_ms is optional (back-compat for callers not yet migrated)
  - to_dict() includes the new llm_calls array
"""

from app.core.llm_metrics import LLMMetrics


def test_add_appends_a_per_call_entry_with_all_four_keys():
    m = LLMMetrics()
    m.add("gpt-4o-mini-2024-07-18", 800, 300, latency_ms=350)
    assert len(m.llm_calls) == 1
    entry = m.llm_calls[0]
    assert entry == {
        "model": "gpt-4o-mini-2024-07-18",
        "input_tokens": 800,
        "output_tokens": 300,
        "latency_ms": 350,
    }


def test_two_adds_produce_two_entries_in_order():
    m = LLMMetrics()
    m.add("gpt-4o-mini", 800, 300, latency_ms=350)
    m.add("gpt-4.1", 5000, 600, latency_ms=2400)
    assert len(m.llm_calls) == 2
    assert m.llm_calls[0]["model"] == "gpt-4o-mini"
    assert m.llm_calls[1]["model"] == "gpt-4.1"
    # Flat fields sum across calls
    assert m.input_tokens == 5800
    assert m.output_tokens == 900
    # Top-level model = last non-empty
    assert m.model == "gpt-4.1"


def test_to_dict_includes_llm_calls():
    m = LLMMetrics()
    m.add("gpt-4o-mini", 100, 50, latency_ms=120)
    m.latency_ms = 1500
    d = m.to_dict()
    assert set(d.keys()) == {
        "model",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "llm_calls",
    }
    assert d["llm_calls"] == [{
        "model": "gpt-4o-mini",
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 120,
    }]


def test_latency_ms_defaults_to_zero_for_back_compat():
    """Callers that haven't migrated to per-call timing can still use the 3-arg form."""
    m = LLMMetrics()
    m.add("gpt-4.1", 5000, 600)   # no latency_ms kwarg
    assert m.llm_calls[0]["latency_ms"] == 0


def test_empty_model_still_appends_entry_but_does_not_overwrite_top_model():
    """An empty model arg is recorded as-is (with empty string) but doesn't clobber the running top-level model."""
    m = LLMMetrics()
    m.add("gpt-4.1", 5000, 600, latency_ms=2400)
    m.add("", 7, 3, latency_ms=10)            # e.g. a defensive fallback path
    assert m.model == "gpt-4.1"               # not overwritten
    assert len(m.llm_calls) == 2
    assert m.llm_calls[1]["model"] == ""      # entry IS appended; choice belongs to the caller


def test_none_token_inputs_coerce_to_zero_at_both_levels():
    m = LLMMetrics()
    m.add("gpt-4.1", None, None, latency_ms=None)
    assert m.input_tokens == 0
    assert m.output_tokens == 0
    assert m.llm_calls[0]["input_tokens"] == 0
    assert m.llm_calls[0]["output_tokens"] == 0
    assert m.llm_calls[0]["latency_ms"] == 0


def test_cost_reconstruction_use_case():
    """The whole point of the array: reconstruct per-model spend."""
    m = LLMMetrics()
    m.add("gpt-4o-mini", 800, 300, latency_ms=350)     # cheap classifier
    m.add("gpt-4o-mini", 600, 200, latency_ms=280)     # cheap extractor
    m.add("gpt-4.1", 5000, 600, latency_ms=2400)       # generation
    # Naive single-rate calc using top-level only is the wrong number
    # for cost; iterating llm_calls gives per-model rollups.
    by_model: dict = {}
    for c in m.llm_calls:
        agg = by_model.setdefault(c["model"], {"input": 0, "output": 0})
        agg["input"] += c["input_tokens"]
        agg["output"] += c["output_tokens"]
    assert by_model == {
        "gpt-4o-mini": {"input": 1400, "output": 500},
        "gpt-4.1": {"input": 5000, "output": 600},
    }
