"""Synthetic smoke tests; no Kaggle download or API token is required."""
import numpy as np
import pandas as pd

from src.baseline import TARGETS, flatten_messages, normalized_frame, get_labels
from src.baseline import fit_baseline, predict_prob, train, predict


def make_fake_train():
    rows = []
    for cls in range(3):
        for idx in range(12):
            rows.append({
                "id": cls * 12 + idx,
                "prompt": '["Explain this example clearly"]',
                "response_a": '["Detailed and clear explanation number %d"]' % idx
                if cls == 0 else '["Short ambiguous reply %d"]' % idx,
                "response_b": '["Detailed and clear explanation number %d"]' % idx
                if cls == 1 else '["Short ambiguous reply %d"]' % idx,
                **{name: int(cls == j) for j, name in enumerate(TARGETS)},
            })
    return pd.DataFrame(rows)


def test_serialized_messages_and_labels():
    df = make_fake_train()
    assert flatten_messages('["hello", null, "world"]') == "hello  world"
    assert normalized_frame(df).iloc[0]["prompt"] == "Explain this example clearly"
    assert sorted(set(get_labels(df))) == [0, 1, 2]


def test_probability_and_submission(tmp_path):
    raw = make_fake_train()
    normalized = normalized_frame(raw)
    bundle = fit_baseline(normalized, get_labels(raw))
    probs = predict_prob(bundle, normalized.iloc[:5])
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    raw.to_csv(train_csv, index=False)
    raw[["id", "prompt", "response_a", "response_b"]].iloc[:3].to_csv(
        test_csv, index=False
    )
    artifacts = tmp_path / "models"
    metrics = train(train_csv, artifacts)
    assert np.isfinite(metrics["validation_log_loss"])
    output = predict(test_csv, artifacts / "baseline.joblib", tmp_path / "submission.csv")
    assert output.columns.tolist() == ["id"] + TARGETS
    assert len(output) == 3
    assert np.allclose(output[TARGETS].sum(axis=1), 1.0)
