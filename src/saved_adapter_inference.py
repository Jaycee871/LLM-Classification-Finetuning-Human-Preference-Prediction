"""Inference-only diagnosis of an EXISTING Kaggle Qwen-LoRA adapter.

No model training, raw-row export, checkpoint publishing or competition
submission. Only this module's aggregate JSON is suitable for GitHub artifacts.
The original 2026-09-24 GPU pilot used seed 42, a 15% row-stratified outer
split and a stratified 1,200-row subsample of that held-out fold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

from src.baseline import TARGETS, flatten_messages, flip_pairs, get_labels, normalized_frame
from src.finetune_lora import render_pair


def exact_pilot_validation(raw, seed=42, max_validation_rows=1200):
    """Reproduce *both* split calls used by the successful Kaggle pilot."""
    frame = normalized_frame(raw)
    labels = get_labels(raw)
    _, val_frame, _, val_labels = train_test_split(
        frame, labels, test_size=0.15, random_state=seed, stratify=labels
    )
    if max_validation_rows and len(val_frame) > max_validation_rows:
        val_frame, _, val_labels, _ = train_test_split(
            val_frame, val_labels, train_size=max_validation_rows,
            random_state=seed, stratify=val_labels
        )
    return val_frame, np.asarray(val_labels, dtype=int)


def validated_swap_rows(requested, available):
    """Reject empty swap probes before touching expensive model inference."""
    if requested <= 0 or available <= 0:
        raise ValueError("--swap-rows and available validation rows must be positive")
    return min(requested, available)


def predict_in_batches(model, tokenizer, frame, device, max_length=384, batch_size=8):
    """Use saved adapter in eval/inference mode only; return in-memory probabilities."""
    import torch
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(frame) == 0:
        raise ValueError("Inference batch frame must not be empty")
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            chunk = frame.iloc[start:start + batch_size]
            rendered = [render_pair(row) for row in chunk.to_dict("records")]
            enc = tokenizer(
                rendered, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt"
            )
            batch = {key: value.to(device) for key, value in enc.items()}
            logits = model(**batch).logits.detach().float().cpu().numpy()
            if logits.ndim != 2 or logits.shape[1] != 3:
                raise ValueError(f"Expected exactly 3 logits, received {logits.shape}")
            result.append(softmax(logits.astype(np.float64), axis=1))
    return np.vstack(result)


def token_truncation_report(tokenizer, frame, max_length=384):
    """Count truncated rendered text locally; never include any text in the report."""
    lengths = np.array([
        len(tokenizer(render_pair(row), add_special_tokens=True, truncation=False)["input_ids"])
        for row in frame.to_dict("records")
    ], dtype=int)
    return {
        "evaluated_rows": int(len(lengths)),
        "sequence_max_tokens": max_length,
        "rendered_token_length_mean": float(lengths.mean()),
        "rendered_token_length_median": float(np.median(lengths)),
        "rendered_token_length_p90": float(np.percentile(lengths, 90)),
        "rendered_token_length_max": int(lengths.max()),
        "fraction_rendered_sequence_exceeds_token_limit":
            float(np.mean(lengths > max_length)),
        "warning": "Also includes earlier character caps in normalized_frame (2400 per column), "
                   "render_pair (1200 prompt, 2400 per response); full original content "
                   "beyond those caps is not included in token-length counts.",
    }


def character_cap_report(raw_rows):
    """Measure the raw character caps applied before the tokenizer."""
    caps = {"prompt": 1200, "response_a": 2400, "response_b": 2400}
    results = {}
    for field, cap in caps.items():
        lengths = np.asarray([
            len(flatten_messages(v, max_chars=10_000_000))
            for v in raw_rows[field]
        ])
        results[field] = {
            "cap_characters": cap,
            "fraction_raw_fields_exceed_cap": float(np.mean(lengths > cap)),
        }
    return results


def aggregate_predictions(y, probabilities, swapped_aligned=None):
    """Compute only class-wise and overall diagnostics, no row-level exports."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), 3) or not np.isfinite(p).all():
        raise ValueError("Expected finite Nx3 probabilities")
    if np.any(p < -1e-7) or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    if not np.isin(y, [0, 1, 2]).all() or not len(y):
        raise ValueError("Expected nonempty validation labels 0,1,2")
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    predicted = np.argmax(p, axis=1)
    counts = np.bincount(predicted, minlength=3)
    confusion = confusion_matrix(y, predicted, labels=[0, 1, 2])
    confidence = p.max(axis=1)
    correct = (predicted == y).astype(float)
    entropy = -(p * np.log(p)).sum(axis=1)
    raw_bins = np.minimum((confidence * 10).astype(int), 9)
    calibration = []
    for k in range(10):
        mask = raw_bins == k
        calibration.append({
            "lower_confidence": k / 10,
            "upper_confidence": (k + 1) / 10,
            "n": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        })
    ece = sum(
        (row["n"] / len(y)) * abs(row["mean_confidence"] - row["accuracy"])
        for row in calibration if row["n"]
    )
    classwise = {}
    for idx, name in enumerate(TARGETS):
        mask = y == idx
        classwise[name] = {
            "true_n": int(mask.sum()),
            "predicted_n": int(counts[idx]),
            "mean_true_class_probability":
                float(p[mask, idx].mean()) if mask.any() else None,
            "mean_negative_log_likelihood":
                float(-np.log(p[mask, idx]).mean()) if mask.any() else None,
            "recall": float((predicted[mask] == idx).mean()) if mask.any() else None,
        }
    output = {
        "n": int(len(y)),
        "multiclass_log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "accuracy": float(correct.mean()),
        "multiclass_brier_score": float(np.mean(np.sum(
            (p - np.eye(3)[y]) ** 2, axis=1
        ))),
        "mean_prediction_entropy_nats": float(entropy.mean()),
        "mean_max_probability": float(confidence.mean()),
        "median_max_probability": float(np.median(confidence)),
        "p90_max_probability": float(np.percentile(confidence, 90)),
        "predicted_counts": {
            name: int(counts[i]) for i, name in enumerate(TARGETS)
        },
        "mean_predicted_probabilities": {
            name: float(p[:, i].mean()) for i, name in enumerate(TARGETS)
        },
        "true_counts": {
            name: int((y == i).sum()) for i, name in enumerate(TARGETS)
        },
        "confusion_matrix_true_rows_predicted_columns": confusion.astype(int).tolist(),
        "classwise": classwise,
        "confidence_reliability_bins": calibration,
        "expected_calibration_error_10_bin": float(ece),
        "notes": [
            "ECE uses the maximum predicted probability and ten equal-width bins. "
            "Small-bin estimates are noisy.",
            "Scores are for the original Qwen pilot row-random validation subset, "
            "not the prompt-grouped reporting holdout or Kaggle leaderboard.",
        ],
    }
    if swapped_aligned is not None:
        aligned = np.asarray(swapped_aligned, dtype=np.float64)
        if aligned.shape != p.shape or not np.isfinite(aligned).all():
            raise ValueError("Swapped probabilities must match original shape")
        err = np.abs(p - aligned)
        output["swap_diagnostic"] = {
            "n": int(len(y)),
            "mean_absolute_probability_disagreement": float(err.mean()),
            "fraction_rows_with_any_class_difference_above_0p2":
                float(np.mean(err.max(axis=1) > 0.2)),
            "fraction_rows_with_changed_predicted_winner":
                float(np.mean(predicted != np.argmax(aligned, axis=1))),
            "meaning": "Model order instability after relabeling swapped A/B. "
                       "Not a measurement of human presentation bias.",
        }
    return output


def main(args):
    # Model loading is intentionally lazy; synthetic tests run entirely on CPU.
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base_dir = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError("Missing local pretrained base model config")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError("Missing saved adapter configuration")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Missing saved adapter weights")
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt",
                   device="cpu") as handle:
        keys = list(handle.keys())
    head_keys = [key for key in keys if "score" in key and key.endswith(".weight")]
    if not head_keys:
        raise ValueError("Saved adapter has no classifier score weight; "
                         "cannot claim original head was restored")
    if not torch.cuda.is_available():
        raise RuntimeError("Inference-only Notebook requires Kaggle CUDA accelerator")
    raw = pd.read_csv(args.train)
    x_val, y_val = exact_pilot_validation(raw, seed=args.seed,
                                         max_validation_rows=args.max_validation_rows)
    # Map back only to these original raw rows for character-cap counts.
    raw_val = raw.loc[x_val.index]
    n_swap = validated_swap_rows(args.swap_rows, len(x_val))
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True,
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad nor EOS")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForSequenceClassification.from_pretrained(
        base_dir, num_labels=3, torch_dtype=dtype,
        local_files_only=True, trust_remote_code=False
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=False, local_files_only=True
    )
    model.to("cuda:0")
    model.eval()

    probs = predict_in_batches(model, tokenizer, x_val, device="cuda:0",
                               max_length=args.max_length, batch_size=args.batch_size)
    swapped = predict_in_batches(
        model, tokenizer, flip_pairs(x_val.iloc[:n_swap]), "cuda:0",
        max_length=args.max_length, batch_size=args.batch_size
    )[:, [1, 0, 2]]
    aggregated = aggregate_predictions(y_val, probs)
    swap_aggregated = aggregate_predictions(
        y_val[:n_swap], probs[:n_swap], swapped_aligned=swapped
    )["swap_diagnostic"]
    aggregated["swap_diagnostic"] = swap_aggregated
    aggregated["rendered_token_truncation"] = token_truncation_report(
        tokenizer, x_val, max_length=args.max_length
    )
    aggregated["original_character_cap_rates"] = character_cap_report(raw_val)
    aggregated["model_check"] = {
        "loaded_saved_classifier_weight_tensors": len(head_keys),
        "num_classes": 3,
        "adapter_source": "previous private Kaggle Qwen LoRA v2 output",
        "no_new_training": True,
    }
    prior = Path(args.previous_metrics)
    if prior.is_file():
        old = json.loads(prior.read_text(encoding="utf-8"))
        old_loss = old.get("validation_log_loss")
        old_swap = old.get("swap_probe_mean_abs_difference")
        aggregated["reproducibility"] = {
            "previous_validation_log_loss": old_loss,
            "difference_in_log_loss":
                aggregated["multiclass_log_loss"] - float(old_loss)
                if old_loss is not None else None,
            "previous_swap_mean_abs_disagreement": old_swap,
            "difference_in_swap_mean_abs_disagreement":
                aggregated["swap_diagnostic"]["mean_absolute_probability_disagreement"] -
                float(old_swap) if old_swap is not None else None,
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(aggregated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    # No raw prompts, text, IDs or per-row probabilities are written to disk.
    print(json.dumps({
        "validation_rows": aggregated["n"],
        "log_loss": aggregated["multiclass_log_loss"],
        "predicted_counts": aggregated["predicted_counts"],
        "mean_confidence": aggregated["mean_max_probability"],
        "swap_mean_absolute_disagreement":
            aggregated["swap_diagnostic"]["mean_absolute_probability_disagreement"],
        "token_truncation_fraction":
            aggregated["rendered_token_truncation"]["fraction_rendered_sequence_exceeds_token_limit"],
    }, indent=2))
    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--previous-metrics", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-rows", type=int, default=1200)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--swap-rows", type=int, default=128)
    main(parser.parse_args())
