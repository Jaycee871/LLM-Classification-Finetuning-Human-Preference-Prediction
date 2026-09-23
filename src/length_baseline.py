"""Nested-tuned, length-only A/B/tie baseline, exploratory validation.

This tests whether simple observable character lengths provide predictive signal.
It neither measures warmth nor establishes a causal preference for verbosity.
Only aggregate JSON may be published; official Kaggle data stay local.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.baseline import TARGETS, flatten_messages, flip_pairs, get_labels
from scripts.run_official_cpu_pilot import file_sha256


def feature_matrix(df):
    """No semantic word features: lengths of complete joined conversation fields."""
    def lengths(col):
        return np.asarray(
            [len(flatten_messages(value, max_chars=2_000_000)) for value in df[col]],
            dtype=np.float64,
        )
    a = np.log1p(lengths("response_a"))
    b = np.log1p(lengths("response_b"))
    q = np.log1p(lengths("prompt"))
    delta = a - b
    return np.column_stack([
        delta,
        np.abs(delta),
        (a + b) / 2,
        q,
        delta * (q / 10),
        (a - b) / np.maximum((a + b), 1.0),
    ])


def fit_length_model(frame, labels, c=1.0):
    original = feature_matrix(frame)
    swapped = feature_matrix(flip_pairs(frame))
    swapped_labels = np.where(labels == 0, 1, np.where(labels == 1, 0, 2))
    model = LogisticRegression(C=c, max_iter=700, random_state=42)
    model.fit(
        np.vstack([original, swapped]),
        np.concatenate([labels, swapped_labels])
    )
    return model


def predict_length_model(model, frame):
    raw = model.predict_proba(feature_matrix(frame))
    out = np.zeros((len(frame), 3), dtype=np.float64)
    for idx, label in enumerate(model.classes_):
        out[:, int(label)] = raw[:, idx]
    return out


def logloss_delta_interval(labels, prediction, reference, draws=1000, seed=42):
    """Bootstrap per-row excess log loss; negative means model lower loss."""
    y = np.asarray(labels, dtype=np.int64)
    selected_model = np.clip(prediction[np.arange(len(y)), y], 1e-15, 1)
    selected_reference = np.clip(reference[np.arange(len(y)), y], 1e-15, 1)
    delta = -np.log(selected_model) + np.log(selected_reference)
    rng = np.random.default_rng(seed)
    draws_arr = np.array([
        delta[rng.integers(0, len(delta), len(delta))].mean()
        for _ in range(draws)
    ])
    return [float(x) for x in np.quantile(draws_arr, [0.025, 0.975])]


def run_pilot(train_csv, out_dir, sample_size=12000, seed=42):
    source = Path(train_csv)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source)
    labels = get_labels(raw)
    if len(raw) < 100 or np.min(np.bincount(labels, minlength=3)) < 10:
        raise ValueError("Need >=100 labeled examples and >=10 per class")
    if sample_size < 0 or (sample_size > 0 and sample_size < 100):
        raise ValueError("Pilot size should be 0 for all rows or >=100")
    if sample_size and sample_size < len(raw):
        chosen, _ = train_test_split(
            np.arange(len(raw)), train_size=sample_size,
            stratify=labels, random_state=seed,
        )
        pilot = raw.iloc[chosen].reset_index(drop=True)
    else:
        pilot = raw.reset_index(drop=True)
    pilot_y = get_labels(pilot)
    # Exactly the outer split used in the 2026-09-23 TF-IDF benchmark.
    outer_train, outer_val, y_train, y_val = train_test_split(
        pilot, pilot_y, test_size=0.15, random_state=42, stratify=pilot_y
    )
    # Hyperparameter selection must be based only on an INNER calibration fold.
    inner_train, inner_val, inner_y, inner_val_y = train_test_split(
        outer_train, y_train, test_size=0.20,
        random_state=1337, stratify=y_train
    )
    candidate_c = [0.01, 0.1, 1.0, 10.0]
    inner_results = {}
    for c in candidate_c:
        model = fit_length_model(inner_train, inner_y, c=c)
        inner_results[str(c)] = float(
            log_loss(inner_val_y, predict_length_model(model, inner_val),
                     labels=[0, 1, 2])
        )
    best_c = min(candidate_c, key=lambda c: inner_results[str(c)])
    model = fit_length_model(outer_train, y_train, c=best_c)
    prediction = predict_length_model(model, outer_val)
    uniform = np.full_like(prediction, 1 / 3)
    prior = np.bincount(y_train, minlength=3).astype(np.float64)
    prior /= prior.sum()
    prior_probs = np.tile(prior, (len(y_val), 1))
    metrics = {
        "source":"Official Kaggle LLM Classification Finetuning training CSV",
        "source_file_sha256":file_sha256(source),
        "official_training_rows":int(len(raw)),
        "pilot_rows":int(len(pilot)),
        "outer_validation_rows":int(len(y_val)),
        "outer_validation_log_loss_length_only":float(log_loss(y_val, prediction,labels=[0,1,2])),
        "outer_validation_log_loss_training_prior":float(log_loss(y_val, prior_probs,labels=[0,1,2])),
        "outer_validation_log_loss_uniform":float(log_loss(y_val, uniform,labels=[0,1,2])),
        "length_minus_prior_log_loss_bootstrap_95pct":logloss_delta_interval(
            y_val, prediction, prior_probs, seed=seed,
        ),
        "inner_validation_log_loss_by_c":inner_results,
        "selected_c":best_c,
        "outer_seed":42,
        "inner_seed":1337,
        "pilot_seed":seed,
        "note":"EXPLORATORY comparison. Outer fold overlaps initial TF-IDF benchmark; do not reuse it indefinitely for model selection. Not a Kaggle submission.",
        "constraints":["Row-random splitting; repeated prompts may cross splits.","Character lengths are not measures of style or warmth.","Selected C tuned on inner fold only.","No text features: performance reflects length correlates, not causal effects."],
        "environment":{
            "scikit_learn":sklearn.__version__,
            "pandas":pd.__version__,
            "github_sha":os.getenv("GITHUB_SHA","local"),
            "run_utc":datetime.now(timezone.utc).isoformat(),
        }
    }
    (output/"summary.json").write_text(
        json.dumps(metrics,indent=2)+"\n",encoding="utf-8"
    )
    print("Official training rows:",metrics["official_training_rows"])
    print("Pilot training rows:",metrics["pilot_rows"])
    print("Chosen inner-fold regularization C:",best_c)
    print("Outer length-only log loss:",metrics["outer_validation_log_loss_length_only"])
    print("Outer train-prior log loss:",metrics["outer_validation_log_loss_training_prior"])
    print("95% bootstrap (length minus prior):",metrics["length_minus_prior_log_loss_bootstrap_95pct"])
    print("Aggregate summary saved.")
    return metrics


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",default="data/train.csv")
    p.add_argument("--out",default="artifacts/official_length_pilot")
    p.add_argument("--sample-size",type=int,default=12000)
    a=p.parse_args()
    run_pilot(a.train,a.out,a.sample_size)
