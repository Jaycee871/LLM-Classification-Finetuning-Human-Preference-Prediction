"""Leakage-controlled, swap-augmented TF-IDF baseline for Kaggle LLM preference prediction.

This is a classical ML baseline, not an LLM fine-tuning run.
"""
import argparse
import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]
TEXT_COLUMNS = ["prompt", "response_a", "response_b"]


def flatten_messages(value, max_chars=2400):
    """Normalize Kaggle's serialized lists of turns; cap length for a CPU starter."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except (ValueError, TypeError):
                try:
                    value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    value = text
        else:
            value = text
    if isinstance(value, (list, tuple)):
        text = " ".join("" if item is None else str(item) for item in value)
    else:
        text = str(value)
    return text[:max_chars]


def normalized_frame(df):
    missing = [c for c in TEXT_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"Missing text columns: {missing}")
    return pd.DataFrame(
        {col: [flatten_messages(v) for v in df[col]] for col in TEXT_COLUMNS},
        index=df.index,
    )


def get_labels(df):
    missing = [c for c in TARGETS if c not in df]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    y = df[TARGETS].to_numpy(dtype=int)
    if not np.all(y.sum(axis=1) == 1) or not np.all((y == 0) | (y == 1)):
        raise ValueError("Expected exactly one binary winner label per training row")
    return y.argmax(axis=1)


def flip_pairs(df):
    flipped = df.copy()
    flipped["response_a"], flipped["response_b"] = (
        df["response_b"].copy(), df["response_a"].copy()
    )
    return flipped


def make_vectorizer(df):
    # Only fit on training-partition texts; do not fit on held-out validation/test.
    min_df = 2 if len(df) >= 30 else 1
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=35000, min_df=min_df,
        strip_accents="unicode", sublinear_tf=True, dtype=np.float32,
    )
    vectorizer.fit(
        df["prompt"].tolist() + df["response_a"].tolist() +
        df["response_b"].tolist()
    )
    return vectorizer


def pair_features(df, vectorizer):
    q = vectorizer.transform(df["prompt"])
    a = vectorizer.transform(df["response_a"])
    b = vectorizer.transform(df["response_b"])
    len_a = df["response_a"].str.len().to_numpy(dtype=np.float32)
    len_b = df["response_b"].str.len().to_numpy(dtype=np.float32)
    len_q = df["prompt"].str.len().to_numpy(dtype=np.float32)
    numeric = np.column_stack([
        np.log1p(len_a) - np.log1p(len_b),
        (np.log1p(len_a) + np.log1p(len_b)) / 2,
        np.log1p(len_q),
    ]) / 10.0
    return hstack([q, a - b, (a + b) * 0.5, csr_matrix(numeric)],
                  format="csr", dtype=np.float32)


def fit_baseline(df, y):
    vectorizer = make_vectorizer(df)
    x_original = pair_features(df, vectorizer)
    x_flipped = pair_features(flip_pairs(df), vectorizer)
    swapped_labels = np.where(y == 0, 1, np.where(y == 1, 0, 2))
    model = LogisticRegression(C=2.0, max_iter=300, random_state=42)
    model.fit(vstack([x_original, x_flipped], format="csr"),
              np.concatenate([y, swapped_labels]))
    return {"vectorizer": vectorizer, "model": model, "targets": TARGETS}


def predict_prob(bundle, df):
    features = pair_features(df, bundle["vectorizer"])
    raw = bundle["model"].predict_proba(features)
    out = np.zeros((len(df), len(TARGETS)), dtype=np.float64)
    for col_idx, class_idx in enumerate(bundle["model"].classes_):
        out[:, int(class_idx)] = raw[:, col_idx]
    return out / out.sum(axis=1, keepdims=True)


def train(train_csv, out_dir, validation_fraction=0.15):
    raw = pd.read_csv(train_csv)
    df = normalized_frame(raw)
    y = get_labels(raw)
    if len(np.unique(y)) != 3 or np.min(np.bincount(y, minlength=3)) < 2:
        raise ValueError("Each class must have at least two examples for validation")
    x_tr, x_val, y_tr, y_val = train_test_split(
        df, y, test_size=validation_fraction, random_state=42, stratify=y
    )
    validation_bundle = fit_baseline(x_tr, y_tr)
    val_probs = predict_prob(validation_bundle, x_val)
    metrics = {
        "validation_log_loss": float(log_loss(y_val, val_probs, labels=[0, 1, 2])),
        "train_rows": int(len(x_tr)),
        "validation_rows": int(len(x_val)),
        "full_rows": int(len(df)),
        "seed": 42,
        "note": "Random stratified split; not a competition leaderboard result.",
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    # Refit using all labeled data only after holding out validation above.
    joblib.dump(fit_baseline(df, y), output / "baseline.joblib")
    return metrics


def predict(test_csv, model_path, out_csv):
    test = pd.read_csv(test_csv)
    if "id" not in test.columns:
        raise ValueError("Test CSV must contain id")
    bundle = joblib.load(model_path)  # Load only artifacts you created/trust.
    probabilities = predict_prob(bundle, normalized_frame(test))
    result = pd.DataFrame(probabilities, columns=TARGETS)
    result.insert(0, "id", test["id"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--train", default="data/train.csv")
    fit.add_argument("--out", default="artifacts")
    infer = sub.add_parser("predict")
    infer.add_argument("--test", default="data/test.csv")
    infer.add_argument("--model", default="artifacts/baseline.joblib")
    infer.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.train, args.out), indent=2))
    else:
        print(f"Wrote {len(predict(args.test, args.model, args.out))} rows: {args.out}")


if __name__ == "__main__":
    main()
