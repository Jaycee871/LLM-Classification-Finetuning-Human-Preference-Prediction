"""Synthetic-only checks for saved-adapter inference diagnostics (no GPU needed)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.saved_adapter_inference import (
    aggregate_predictions, exact_pilot_validation, token_truncation_report,
    character_cap_report, validated_swap_rows,
)


def fake_labeled_frame(n_each=48):
    rows = []
    for k in range(3):
        for i in range(n_each):
            rows.append({
                "prompt": json.dumps([f"Artificial request class {k}, item {i}"]),
                "response_a": json.dumps(["Synthetic A " * (k + 1)]),
                "response_b": json.dumps(["Synthetic B " * (3 - k)]),
                "winner_model_a": int(k == 0),
                "winner_model_b": int(k == 1),
                "winner_tie": int(k == 2),
            })
    return pd.DataFrame(rows)


def test_exact_validation_split_reproduces_prior_two_step_seed():
    frame = fake_labeled_frame()
    val, y = exact_pilot_validation(frame, seed=42, max_validation_rows=12)
    again, y2 = exact_pilot_validation(frame, seed=42, max_validation_rows=12)
    assert len(y) == 12
    assert list(val.index) == list(again.index)
    assert np.array_equal(y, y2)
    assert set(y) == {0, 1, 2}
    assert val.index.is_unique


def test_aggregates_include_classwise_calibration_and_swap_without_rows():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.array([
        [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.1, 0.1, 0.8]
    ])
    swapped = p.copy()
    swapped[0, :] = [0.2, 0.7, 0.1]
    stats = aggregate_predictions(y, p, swapped_aligned=swapped)
    assert stats["n"] == 6
    assert stats["predicted_counts"] == {
        "winner_model_a": 2, "winner_model_b": 2, "winner_tie": 2
    }
    assert stats["confusion_matrix_true_rows_predicted_columns"] == [
        [2, 0, 0], [0, 2, 0], [0, 0, 2]
    ]
    assert sum(bin["n"] for bin in stats["confidence_reliability_bins"]) == 6
    assert stats["swap_diagnostic"]["fraction_rows_with_changed_predicted_winner"] == pytest.approx(1 / 6)
    assert stats["swap_diagnostic"]["mean_absolute_probability_disagreement"] > 0
    assert "probabilities" not in stats
    assert "prompt" not in stats
    assert 0 < stats["multiclass_log_loss"] < 1
    assert 0 <= stats["expected_calibration_error_10_bin"] <= 1


def test_rejects_bad_probability_matrix():
    with pytest.raises(ValueError, match="Invalid class probabilities"):
        aggregate_predictions([0], [[0.4, 0.2, 0.2]])


class StubTokenizer:
    def __call__(self, text, add_special_tokens, truncation):
        assert not truncation
        return {"input_ids": list(range(len(text.split())))}


def test_token_truncation_and_raw_char_caps_are_aggregate_only():
    frame = fake_labeled_frame(n_each=12)
    val, _ = exact_pilot_validation(frame, max_validation_rows=0)
    token_report = token_truncation_report(StubTokenizer(), val, max_length=2)
    caps = character_cap_report(frame)
    assert token_report["evaluated_rows"] == len(val)
    assert token_report["fraction_rendered_sequence_exceeds_token_limit"] == 1
    assert set(caps) == {"prompt", "response_a", "response_b"}
    assert all(0 <= record["fraction_raw_fields_exceed_cap"] <= 1 for record in caps.values())


def test_swap_probe_rejects_empty_input():
    assert validated_swap_rows(128, 100) == 100
    assert validated_swap_rows(128, 200) == 128
    for requested, available in [(0, 10), (-1, 10), (10, 0)]:
        with pytest.raises(ValueError, match="must be positive"):
            validated_swap_rows(requested, available)
