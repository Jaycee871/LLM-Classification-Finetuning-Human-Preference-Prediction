"""Aggregate observational length audit and A/B-swap stability test.

No raw Kaggle text or row-level predictions are written by this script.
Character counts are proxies for length, not tokenizer token counts.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baseline import TARGETS, flip_pairs, flatten_messages, get_labels, normalized_frame, predict_prob


def _interval(values, seed=42, draws=1000):
    """Nonparametric 95% bootstrap interval, resampling rows."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for j in range(draws):
        means[j] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def length_audit(raw, seed=42):
    """Descriptive preference–length association; cannot estimate causal effects.

    Crucially, count the *original* serialized message length, rather than the
    truncated inputs used by the quick TF-IDF baseline.
    """
    labels = get_labels(raw)
    a = np.asarray([len(flatten_messages(v, max_chars=2_000_000))
                    for v in raw["response_a"]], dtype=np.int64)
    b = np.asarray([len(flatten_messages(v, max_chars=2_000_000))
                    for v in raw["response_b"]], dtype=np.int64)
    group = np.where(a > b, "a_longer",
                     np.where(b > a, "b_longer", "equal_char_length"))
    groups = {}
    for name in ("a_longer", "b_longer", "equal_char_length"):
        idx = group == name
        counts = np.bincount(labels[idx], minlength=3).astype(int)
        groups[name] = {
            "n": int(idx.sum()),
            "winner_model_a": int(counts[0]),
            "winner_model_b": int(counts[1]),
            "tie": int(counts[2]),
        }
    valid = a != b
    decided = valid & (labels != 2)
    chosen_longer = np.where(a > b, labels == 0, labels == 1)[decided]
    equal = int((a == b).sum())
    return {
        "observations": len(raw),
        "unit": "Unicode characters (joined conversational turns)",
        "n_equal_length": equal,
        "n_unequal_non_tie": int(decided.sum()),
        "selected_longer_rate_among_unequal_non_tie":
            float(chosen_longer.mean()) if len(chosen_longer) else None,
        "selected_longer_95pct_row_bootstrap": _interval(chosen_longer, seed=seed),
        "by_relative_length": groups,
        "limits": [
            "Observational: length is confounded with quality, topic, and model identity.",
            "The logged choice is not evidence of an individual user's stable preference.",
            "Character count is not tokenizer token count.",
            "CI assumes exchangeable rows; correlated prompts can make it too narrow.",
        ],
    }


def swap_audit(bundle, frame, batch_size=512):
    """Mean disagreement of P(A), P(B), P(tie) under swapped response order."""
    diffs = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start:start + batch_size]
        p = predict_prob(bundle, batch)
        swapped = predict_prob(bundle, flip_pairs(batch))[:, [1, 0, 2]]
        diffs.append(np.abs(p - swapped))
    if not diffs:
        return {"n": 0, "mean_abs_probability_disagreement": None}
    error = np.vstack(diffs)
    return {
        "n": int(len(frame)),
        "mean_abs_probability_disagreement": float(error.mean()),
        "mean_abs_disagreement_by_class":
            {name: float(error[:, j].mean()) for j, name in enumerate(TARGETS)},
        "max_abs_probability_disagreement": float(error.max()),
        "note": "Diagnostic of model order sensitivity, not human position bias.",
    }


def run(input_csv, output_json, model_path=None, seed=42):
    raw = pd.read_csv(input_csv)
    result = {"length_audit": length_audit(raw, seed=seed)}
    if model_path:
        # Only load trusted artifacts produced by the local baseline trainer.
        result["model_swap_audit"] = swap_audit(
            joblib.load(model_path), normalized_frame(raw)
        )
        result["model_swap_audit"]["caution"] = (
            "For unbiased generalization estimates run swap_audit on a held-out "
            "partition; running it over training data is only a sanity check."
        )
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2) + "\n",
                                 encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/train.csv")
    p.add_argument("--out", default="artifacts/aggregate_diagnostics.json")
    p.add_argument("--model", default=None,
                   help="Optional trusted joblib baseline; order-stability audit")
    args = p.parse_args()
    print(json.dumps(run(args.data, args.out, args.model), indent=2))
