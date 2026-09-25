"""Inference-only A/B swap ensemble for the existing Qwen2.5-0.5B LoRA adapter.

The same saved adapter is evaluated twice on each held-out pair:
1) original response order A/B;
2) swapped order B/A, with class columns mapped back to the original labels.

No training and no competition submission happen here. Only aggregate JSON is written.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.baseline import TARGETS, flatten_messages, get_labels, normalized_frame


def render_pair(row):
    """Exact text template used in the original Qwen pilot."""
    question = flatten_messages(row["prompt"], max_chars=1200)
    a = flatten_messages(row["response_a"], max_chars=2400)
    b = flatten_messages(row["response_b"], max_chars=2400)
    return (
        "A human gave two chatbots the same user request.\n"
        f"User request: {question}\n"
        f"Response A: {a}\n"
        f"Response B: {b}\n"
        "Predict whether the human prefers response A, response B, or a tie."
    )


def exact_validation(raw, seed=42, max_validation_rows=1200):
    """Reproduce the exact two-step validation sampling from the original pilot."""
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
    return val_frame, np.asarray(val_labels, dtype=np.int64)


def swap_frame(frame):
    out = frame.copy()
    out["response_a"], out["response_b"] = (
        frame["response_b"].to_numpy(copy=True),
        frame["response_a"].to_numpy(copy=True),
    )
    return out


def predict_probs(model, tokenizer, frame, device="cuda:0", max_length=1024, batch_size=2):
    import torch
    if len(frame) == 0:
        raise ValueError("Inference frame is empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            chunk = frame.iloc[start:start + batch_size]
            texts = [render_pair(row) for row in chunk.to_dict("records")]
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().float().cpu().numpy()
            if logits.ndim != 2 or logits.shape[1] != 3:
                raise ValueError(f"Expected Nx3 logits, got {logits.shape}")
            result.append(softmax(logits.astype(np.float64), axis=1))
    probs = np.vstack(result)
    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def align_swapped(probabilities):
    """Map swapped-order classes B/A/tie back onto original A/B/tie."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Expected Nx3 probabilities")
    return p[:, [1, 0, 2]]


def arithmetic_ensemble(original, swapped_aligned):
    p = 0.5 * (np.asarray(original) + np.asarray(swapped_aligned))
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def geometric_ensemble(original, swapped_aligned):
    a = np.clip(np.asarray(original, dtype=np.float64), 1e-12, 1.0)
    b = np.clip(np.asarray(swapped_aligned, dtype=np.float64), 1e-12, 1.0)
    p = np.sqrt(a * b)
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def summarize(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), 3):
        raise ValueError("Probability matrix shape mismatch")
    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for k in range(10):
        mask = bins == k
        if not mask.any():
            continue
        acc = np.mean(pred[mask] == y[mask])
        conf = np.mean(confidence[mask])
        ece += (mask.sum() / len(y)) * abs(acc - conf)
    return {
        "multiclass_log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "accuracy": float(np.mean(pred == y)),
        "mean_max_probability": float(np.mean(confidence)),
        "expected_calibration_error_10_bin": float(ece),
        "predicted_counts": {
            TARGETS[i]: int(np.sum(pred == i)) for i in range(3)
        },
    }


def disagreement(original, swapped_aligned):
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(swapped_aligned, dtype=np.float64)
    err = np.abs(a - b)
    return {
        "mean_absolute_probability_disagreement": float(err.mean()),
        "fraction_rows_with_any_class_difference_above_0p2":
            float(np.mean(err.max(axis=1) > 0.2)),
        "fraction_rows_with_changed_predicted_winner":
            float(np.mean(a.argmax(axis=1) != b.argmax(axis=1))),
    }


def run(args):
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base_dir = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError("Missing Qwen base config")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError("Missing adapter config")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Missing adapter weights")
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        head_keys = [key for key in handle.keys() if "score" in key and key.endswith(".weight")]
    if not head_keys:
        raise ValueError("Saved adapter does not include classifier score weights")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for inference-only ensemble audit")

    raw = pd.read_csv(args.train)
    x_val, y_val = exact_validation(raw, seed=args.seed, max_validation_rows=args.max_validation_rows)

    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer lacks pad and EOS")
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForSequenceClassification.from_pretrained(
        base_dir,
        num_labels=3,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=False, local_files_only=True
    )
    model.to("cuda:0")
    model.eval()

    original = predict_probs(
        model, tokenizer, x_val,
        max_length=args.max_length, batch_size=args.batch_size
    )
    swapped_raw = predict_probs(
        model, tokenizer, swap_frame(x_val),
        max_length=args.max_length, batch_size=args.batch_size
    )
    swapped = align_swapped(swapped_raw)
    arithmetic = arithmetic_ensemble(original, swapped)
    geometric = geometric_ensemble(original, swapped)

    result = {
        "type": "saved_qwen_1024_full_swap_ensemble",
        "no_new_training": True,
        "no_competition_submission": True,
        "validation_rows": int(len(y_val)),
        "context_tokens": int(args.max_length),
        "seed": int(args.seed),
        "original": summarize(y_val, original),
        "swapped_aligned": summarize(y_val, swapped),
        "arithmetic_swap_ensemble": summarize(y_val, arithmetic),
        "geometric_swap_ensemble": summarize(y_val, geometric),
        "original_vs_swapped": disagreement(original, swapped),
        "deltas_vs_original": {
            "arithmetic_log_loss":
                summarize(y_val, arithmetic)["multiclass_log_loss"] -
                summarize(y_val, original)["multiclass_log_loss"],
            "geometric_log_loss":
                summarize(y_val, geometric)["multiclass_log_loss"] -
                summarize(y_val, original)["multiclass_log_loss"],
        },
        "limits": [
            "The 1,200-row validation set has already been used for context-length selection; "
            "ensemble selection here is exploratory and not independent confirmation.",
            "Averaging original and swapped predictions enforces order symmetry by construction "
            "but cannot prove the human labels themselves are order-invariant.",
            "The saved adapter was trained on only 2,000 original pairs for one epoch.",
            "Any competition submission based on this ensemble should be labeled experimental.",
        ],
    }
    best_name = min(
        ("original", "arithmetic_swap_ensemble", "geometric_swap_ensemble"),
        key=lambda name: result[name]["multiclass_log_loss"],
    )
    result["lowest_observed_log_loss_method"] = best_name
    result["lowest_observed_log_loss"] = result[best_name]["multiclass_log_loss"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "original": result["original"],
        "swapped_aligned": result["swapped_aligned"],
        "arithmetic": result["arithmetic_swap_ensemble"],
        "geometric": result["geometric_swap_ensemble"],
        "disagreement": result["original_vs_swapped"],
        "best_method": result["lowest_observed_log_loss_method"],
        "best_log_loss": result["lowest_observed_log_loss"],
    }, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="/kaggle/working/qwen_swap_ensemble.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-rows", type=int, default=1200)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    run(parser.parse_args())
