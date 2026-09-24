"""CPU-only checks for the saved-Qwen context-length ablation."""
import pytest

from src.context_length_ablation import delta_from_baseline, validate_contexts


def test_context_validation():
    assert validate_contexts([384, 768, 1024]) == (384, 768, 1024)
    with pytest.raises(ValueError, match="positive"):
        validate_contexts([384, 0])
    with pytest.raises(ValueError, match="unique"):
        validate_contexts([384, 384])


def test_deltas_use_384_as_reference():
    def record(loss, acc, ece, swap, changed, trunc):
        return {
            "multiclass_log_loss": loss,
            "accuracy": acc,
            "expected_calibration_error_10_bin": ece,
            "swap_diagnostic": {
                "mean_absolute_probability_disagreement": swap,
                "fraction_rows_with_changed_predicted_winner": changed,
            },
            "rendered_token_truncation": {
                "fraction_rendered_sequence_exceeds_token_limit": trunc,
            },
        }

    values = {
        "384": record(1.96, .33, .34, .29, .66, .65),
        "768": record(1.80, .36, .28, .20, .50, .27),
        "1024": record(1.75, .37, .25, .18, .45, .10),
    }
    delta = delta_from_baseline(values, 384)
    assert set(delta) == {"768", "1024"}
    assert delta["768"]["delta_log_loss_vs_baseline"] == pytest.approx(-.16)
    assert delta["1024"]["delta_accuracy_vs_baseline"] == pytest.approx(.04)
    assert delta["1024"]["delta_truncation_fraction_vs_baseline"] == pytest.approx(-.55)
