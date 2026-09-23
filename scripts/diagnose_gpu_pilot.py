"""Audit existing aggregate Kaggle Qwen pilot metrics without using GPU.

Input must be the previously saved gpu_pilot_metrics.json, downloaded from
the user's private Kaggle kernel. Only aggregate quantities are emitted.
No new training, model inference, Kaggle competition submission, raw data,
or per-example predictions are involved.
"""
import argparse
import json
import math
from pathlib import Path


def assess(metrics):
    required = (
        "validation_log_loss",
        "matched_length_reference_log_loss",
        "validation_rows",
        "train_rows_after_augmentation",
        "pipeline_status",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"Missing required aggregate metrics: {missing}")

    qwen = float(metrics["validation_log_loss"])
    reference = float(metrics["matched_length_reference_log_loss"])
    validation_rows = int(metrics["validation_rows"])
    if not (math.isfinite(qwen) and math.isfinite(reference)):
        raise ValueError("Non-finite multiclass log loss")
    if validation_rows <= 0:
        raise ValueError("No held-out validation rows")
    if qwen < 0 or reference < 0:
        raise ValueError("Log loss cannot be negative")

    uniform = math.log(3.0)
    swap = metrics.get("swap_probe_mean_abs_difference")
    if swap is not None:
        swap = float(swap)
        if not (math.isfinite(swap) and 0 <= swap <= 1):
            raise ValueError("Swap disagreement must be between 0 and 1")

    return {
        "type": "existing_kaggle_gpu_pilot_aggregate_diagnostic",
        "observations": {
            "qwen_validation_log_loss": qwen,
            "matched_length_reference_log_loss": reference,
            "uniform_three_class_log_loss": uniform,
            "qwen_minus_matched_length_log_loss": qwen - reference,
            "qwen_minus_uniform_log_loss": qwen - uniform,
            "matched_length_minus_uniform_log_loss": reference - uniform,
            "validation_rows": validation_rows,
            "training_rows_after_augmentation": int(
                metrics["train_rows_after_augmentation"]
            ),
            "swap_probe_mean_absolute_probability_difference": swap,
            "pipeline_status": str(metrics["pipeline_status"]),
        },
        "flags": {
            "qwen_worse_than_matched_length_reference": qwen > reference,
            "qwen_worse_than_uniform_baseline": qwen > uniform,
            "length_reference_better_than_uniform": reference < uniform,
        },
        "diagnostic_limits": [
            "Differences are on the same validation rows but no paired prediction "
            "matrix is available, so paired uncertainty cannot be calculated.",
            "The existing aggregate JSON does not contain class-wise losses, "
            "prediction histogram, entropy, calibration or truncation fractions.",
            "This report cannot isolate the effect of the random classification "
            "head, training duration, learning rate, label imbalance, or truncation.",
            "The private Kaggle pilot is different from the public competition "
            "leaderboard population; this is not a new leaderboard score.",
        ],
        "next_measurements_without_training": [
            "On the saved adapter and base model, regenerate validation "
            "predictions using the original fixed seed and split.",
            "Measure per-class counts and losses, mean predicted probabilities, "
            "confidence histogram, entropy, logit ranges, and order-swap behavior.",
            "Measure the token-truncation fraction and inspect training log "
            "loss history without exposing raw prompts or examples.",
            "Retain predictions privately. Publish only aggregate diagnostics "
            "and run an independently reserved validation split before tuning.",
        ],
        "source": "Existing successful private Kaggle Qwen 0.5B LoRA pilot version 2",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = assess(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Existing GPU output assessed; no new GPU training or submission.")
    print(json.dumps(report["observations"], indent=2))


if __name__ == "__main__":
    main()
