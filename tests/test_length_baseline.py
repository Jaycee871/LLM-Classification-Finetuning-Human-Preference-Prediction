"""Synthetic nested-fit length-only baseline tests."""
import numpy as np
import pandas as pd

from src.baseline import get_labels
from src.length_baseline import (
    feature_matrix, fit_length_model, predict_length_model, run_pilot,
)
from tests_fixture import synthetic_data


def test_length_only_features_swap_sign():
    raw = synthetic_data()
    a = feature_matrix(raw)
    from src.baseline import flip_pairs
    b = feature_matrix(flip_pairs(raw))
    assert np.allclose(a[:, 0], -b[:, 0])
    assert np.allclose(a[:, 1:4], b[:, 1:4])


def test_exploratory_nested_validation(tmp_path):
    raw = pd.concat([synthetic_data()] * 4, ignore_index=True)
    frame = raw[["prompt","response_a","response_b"]]
    model=fit_length_model(frame,get_labels(raw),c=0.1)
    pred=predict_length_model(model,frame)
    assert pred.shape==(len(raw),3)
    assert np.allclose(pred.sum(axis=1),1)
    train_csv=tmp_path/"train.csv"
    raw.to_csv(train_csv,index=False)
    stats=run_pilot(train_csv,tmp_path/"out",sample_size=100)
    assert stats["official_training_rows"]==len(raw)
    assert stats["pilot_rows"]==100
    assert stats["outer_validation_rows"]>=10
    assert stats["selected_c"] in [0.01,0.1,1.0,10.0]
    assert np.isfinite(stats["outer_validation_log_loss_length_only"])
    assert (tmp_path/"out"/"summary.json").is_file()


def test_reject_sampled_pilot_with_insufficient_minority_class(tmp_path):
    import pytest
    raw = synthetic_data()
    a = raw[raw["winner_model_a"] == 1]
    b = raw[raw["winner_model_b"] == 1]
    tie = raw[raw["winner_tie"] == 1]
    skewed = pd.concat([a] * 15 + [b] + [tie], ignore_index=True)
    path = tmp_path / "skewed.csv"
    skewed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Sampled pilot must contain at least 10"):
        run_pilot(path, tmp_path / "out", sample_size=100)
