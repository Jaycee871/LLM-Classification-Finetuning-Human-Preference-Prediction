"""Run nested prompt-grouped robustness checks for the length-only baseline.

Official data are downloaded only into the ephemeral Actions runner. No
prompts, identifiers, trained weights, or per-row predictions are uploaded.
This intentionally uses a NEW prompt-grouped split, rather than repeatedly
tuning against the original exploratory random-row validation fold.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

from src.baseline import TARGETS, flatten_messages, get_labels
from src.length_baseline import fit_length_model, predict_length_model, logloss_delta_interval


def dataset_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prompt_group_keys(df):
    """Use complete normalized prompts for grouping, store SHA-256 only in RAM.

    Exact prompt matches cannot cross splits. Similar/near-duplicate prompts
    can still cross, so this is not near-duplicate decontamination.
    """
    return np.array([
        hashlib.sha256(
            flatten_messages(value, max_chars=2_000_000).encode("utf-8")
        ).hexdigest() for value in df["prompt"]
    ])


def grouped_indices(groups, test_size, seed):
    train, heldout = next(GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    ).split(np.arange(len(groups)), groups=groups))
    assert not set(groups[train]) & set(groups[heldout]), (
        "Grouped split leaked a prompt hash"
    )
    return train, heldout


def run_grouped_audit(train_csv, out_dir, seed=20260924, inner_seed=20260925):
    raw = pd.read_csv(train_csv)
    if len(raw) < 120:
        raise ValueError("Need at least 120 labeled rows for grouped audit")
    y = get_labels(raw)
    groups = prompt_group_keys(raw)
    if len(set(groups)) < 20:
        raise ValueError("Need at least 20 distinct prompts for grouped validation")
    outer_train, outer_test = grouped_indices(groups, test_size=0.15, seed=seed)
    inner_train_local, inner_val_local = grouped_indices(
        groups[outer_train], test_size=0.18, seed=inner_seed
    )
    inner_train = outer_train[inner_train_local]
    inner_val = outer_train[inner_val_local]

    for name, indices in (
        ("inner_train", inner_train), ("inner_validation", inner_val),
        ("outer_train", outer_train), ("outer_test", outer_test)
    ):
        if len(np.unique(y[indices])) < 3:
            raise ValueError(f"Grouped split missing a class in {name}; change seed")
    inner_losses = {}
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = fit_length_model(raw.iloc[inner_train], y[inner_train], c=c)
        pred = predict_length_model(model, raw.iloc[inner_val])
        inner_losses[str(c)] = float(log_loss(y[inner_val], pred, labels=[0, 1, 2]))
    best_c = min([0.01, 0.1, 1.0, 10.0, 100.0],
                 key=lambda c: inner_losses[str(c)])
    model = fit_length_model(raw.iloc[outer_train], y[outer_train], c=best_c)
    out = predict_length_model(model, raw.iloc[outer_test])

    uniform = np.full_like(out, 1.0 / 3.0)
    counts = np.bincount(y[outer_train], minlength=3).astype(float)
    prior = np.tile(counts / counts.sum(), (len(outer_test), 1))
    ref_loss = float(log_loss(y[outer_test], prior, labels=[0, 1, 2]))
    model_loss = float(log_loss(y[outer_test], out, labels=[0, 1, 2]))
    result = {
        "data_source": "Official Kaggle llm-classification-finetuning train.csv",
        "source_sha256": dataset_digest(train_csv),
        "total_rows": int(len(raw)),
        "distinct_exact_prompt_hashes": int(len(set(groups))),
        "outer_training_rows": int(len(outer_train)),
        "outer_testing_rows": int(len(outer_test)),
        "inner_training_rows": int(len(inner_train)),
        "inner_validation_rows": int(len(inner_val)),
        "outer_training_group_count": int(len(set(groups[outer_train]))),
        "outer_testing_group_count": int(len(set(groups[outer_test]))),
        "exact_group_overlap": int(len(
            set(groups[outer_train]) & set(groups[outer_test])
        )),
        "label_counts_outer_test": {
            key: int((y[outer_test] == j).sum())
            for j, key in enumerate(TARGETS)
        },
        "tuned_c_inner_only": best_c,
        "inner_loss_by_c": inner_losses,
        "outer_log_loss_length_only": model_loss,
        "outer_log_loss_training_prior": ref_loss,
        "outer_log_loss_uniform": float(log_loss(y[outer_test], uniform, labels=[0, 1, 2])),
        "outer_length_minus_prior_bootstrap_95pct": logloss_delta_interval(
            y[outer_test], out, prior, draws=1000, seed=seed
        ),
        "outer_split_seed": seed,
        "inner_split_seed": inner_seed,
        "split_rule": "GroupShuffleSplit, grouping by SHA256 of full normalized exact prompt",
        "limitations": [
            "Exact prompt overlap controlled; paraphrased/near-duplicate prompts may cross groups.",
            "Nested C selection uses only inner data; outer test is for reporting only.",
            "Only one outer grouped split; additional seeds needed for uncertainty.",
            "Character lengths are correlated with other features and have no causal interpretation.",
            "This is a local validation result, not Kaggle public/private leaderboard performance."
        ],
        "environment": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "commit": os.environ.get("GITHUB_SHA", "local"),
            "run_utc": datetime.now(timezone.utc).isoformat()
        }
    }
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: result[key] for key in (
            "total_rows", "distinct_exact_prompt_hashes",
            "outer_training_rows", "outer_testing_rows",
            "exact_group_overlap", "tuned_c_inner_only",
            "outer_log_loss_length_only", "outer_log_loss_training_prior",
            "outer_log_loss_uniform", "outer_length_minus_prior_bootstrap_95pct"
        )
    }, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--out", default="artifacts/grouped_length_robustness")
    args = parser.parse_args()
    run_grouped_audit(args.train, args.out)
