"""Run an aggregate-only CPU pilot against the official Kaggle training data.

This script never uploads Kaggle CSVs, model artifacts or row-level predictions.
Set --sample-size 0 for the full labeled set after checking runner resources.
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
from sklearn.model_selection import train_test_split

from src.baseline import TARGETS, get_labels, train
from src.diagnostics import length_audit


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(train_csv, out_dir, sample_size=12000, seed=42):
    train_csv = Path(train_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv)
    labels = get_labels(raw)
    if len(raw) < 30 or np.min(np.bincount(labels, minlength=3)) < 3:
        raise ValueError("Need at least 30 examples and three labels for this pilot")
    if sample_size < 0:
        raise ValueError("--sample-size must be nonnegative")

    if sample_size and sample_size < len(raw):
        if sample_size < 30:
            raise ValueError("--sample-size must be 0 or at least 30")
        indices, _ = train_test_split(
            np.arange(len(raw)), train_size=sample_size, random_state=seed,
            stratify=labels,
        )
        sampled = raw.iloc[indices].reset_index(drop=True)
    else:
        sampled = raw.reset_index(drop=True)
    pilot_labels = get_labels(sampled)

    # This file stays on the ephemeral runner and is never uploaded as an artifact.
    local_sample = out_dir / "local_train_sample.csv"
    sampled.to_csv(local_sample, index=False)
    local_model_dir = out_dir / "local_models"
    metrics = train(local_sample, local_model_dir)

    # Recreate precisely the validation indices used by src.baseline.train.
    train_idx, validation_idx = train_test_split(
        np.arange(len(sampled)), test_size=0.15, random_state=42,
        stratify=pilot_labels,
    )
    training_prior = np.bincount(pilot_labels[train_idx], minlength=3).astype(float)
    training_prior /= training_prior.sum()
    val_labels = pilot_labels[validation_idx]
    uniform_probs = np.full((len(val_labels), 3), 1 / 3)
    constant_probs = np.tile(training_prior, (len(val_labels), 1))
    uniform_loss = float(log_loss(val_labels, uniform_probs, labels=[0, 1, 2]))
    prior_loss = float(log_loss(val_labels, constant_probs, labels=[0, 1, 2]))

    stats = {
        "dataset": "Kaggle: llm-classification-finetuning",
        "source_file": "train.csv",
        "source_file_sha256": file_sha256(train_csv),
        "official_training_rows": int(len(raw)),
        "pilot_rows": int(len(sampled)),
        "class_counts_pilot": {
            name: int((pilot_labels == idx).sum()) for idx, name in enumerate(TARGETS)
        },
        "validation_rows": int(metrics["validation_rows"]),
        "validation_log_loss_tfidf": metrics["validation_log_loss"],
        "validation_log_loss_uniform": uniform_loss,
        "validation_log_loss_training_class_prior": prior_loss,
        "seed": seed,
        "internal_validation_seed": 42,
        "split": "15% seeded stratified random rows; NOT grouped by prompt",
        "validation_limitations": [
            "Repeated prompts or near duplicates may cross row-random split.",
            "Pilot truncates text to 2400 characters per field.",
            "No claim of competition leaderboard performance or causal preference effects.",
            "Observational response-length audit uses pilot sample, not hidden test data.",
        ],
        "length_audit": length_audit(sampled, seed=seed),
        "environment": {
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "github_commit": os.getenv("GITHUB_SHA", "local"),
            "run_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    # Avoid posting competition text or row-level predictions to public CI logs.
    print("Official train rows:", stats["official_training_rows"])
    print("Pilot rows:", stats["pilot_rows"])
    print("TF-IDF validation log loss:", stats["validation_log_loss_tfidf"])
    print("Uniform log loss:", stats["validation_log_loss_uniform"])
    print("Train-prior log loss:", stats["validation_log_loss_training_class_prior"])
    print("Aggregate summary:", path)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--out", default="artifacts/official_cpu_pilot")
    p.add_argument("--sample-size", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    execute(a.train, a.out, a.sample_size, a.seed)
