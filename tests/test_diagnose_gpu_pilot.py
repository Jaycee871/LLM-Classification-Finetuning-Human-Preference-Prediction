"""Aggregate-only GPU pilot diagnostics using synthetic mock metric dictionaries."""
import math

import pytest

from scripts.diagnose_gpu_pilot import assess


@pytest.fixture
def sample_metrics():
    return {
        "validation_log_loss": 1.963874524651205,
        "matched_length_reference_log_loss": 1.066152731793071,
        "validation_rows": 1200,
        "train_rows_after_augmentation": 4000,
        "pipeline_status": "submission_completed",
        "swap_probe_mean_abs_difference": 0.13,
    }


def test_flags_underperformance_without_claiming_cause(sample_metrics):
    report = assess(sample_metrics)
    assert report["flags"]["qwen_worse_than_matched_length_reference"]
    assert report["flags"]["qwen_worse_than_uniform_baseline"]
    assert report["flags"]["length_reference_better_than_uniform"]
    assert report["observations"]["uniform_three_class_log_loss"] == pytest.approx(
        math.log(3)
    )
    assert report["observations"]["qwen_minus_matched_length_log_loss"] == pytest.approx(
        0.897721792858134
    )
    assert len(report["diagnostic_limits"]) >= 3
    assert not any("cause is" in text for text in report["diagnostic_limits"])


def test_missing_and_nonfinite_metrics_are_rejected(sample_metrics):
    del sample_metrics["validation_rows"]
    with pytest.raises(ValueError, match="validation_rows"):
        assess(sample_metrics)
    sample_metrics["validation_rows"] = 1200
    sample_metrics["validation_log_loss"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        assess(sample_metrics)


def test_rejects_invalid_swap_probe(sample_metrics):
    sample_metrics["swap_probe_mean_abs_difference"] = 1.1
    with pytest.raises(ValueError, match="between 0 and 1"):
        assess(sample_metrics)
