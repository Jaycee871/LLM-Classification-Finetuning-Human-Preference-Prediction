"""Tests for aggregate preference-length diagnostics with synthetic data only."""
import numpy as np
from src.baseline import normalized_frame, fit_baseline, get_labels
from src.diagnostics import length_audit, swap_audit
from tests_fixture import synthetic_data


def test_length_audit_synthetic():
    data = synthetic_data()
    result = length_audit(data, seed=42)
    assert result["observations"] == len(data)
    assert sum(v["n"] for v in result["by_relative_length"].values()) == len(data)
    assert result["n_unequal_non_tie"] >= 0
    if result["n_unequal_non_tie"]:
        low, high = result["selected_longer_95pct_row_bootstrap"]
        assert 0 <= low <= high <= 1


def test_order_swap_synthetic():
    raw = synthetic_data()
    df = normalized_frame(raw)
    model = fit_baseline(df, get_labels(raw))
    result = swap_audit(model, df, batch_size=8)
    assert result["n"] == len(raw)
    assert np.isfinite(result["mean_abs_probability_disagreement"])
    assert 0 <= result["max_abs_probability_disagreement"] <= 1
