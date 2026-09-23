"""CPU official-data pilot using synthetic data, never uses Kaggle credentials."""
import numpy as np
import pandas as pd
from scripts.run_official_cpu_pilot import execute
from tests_fixture import synthetic_data


def test_official_cpu_pilot_simulated_data(tmp_path):
    raw = pd.concat([synthetic_data()] * 3, ignore_index=True)
    data = tmp_path / "train.csv"
    raw.to_csv(data, index=False)
    summary = execute(data, tmp_path / "out", sample_size=36)
    assert summary["official_training_rows"] == len(raw)
    assert summary["pilot_rows"] == 36
    assert summary["validation_rows"] >= 3
    for key in (
        "validation_log_loss_tfidf",
        "validation_log_loss_uniform",
        "validation_log_loss_training_class_prior",
    ):
        assert np.isfinite(summary[key])
    assert (tmp_path / "out" / "summary.json").exists()
    assert "prompt" not in summary and "response_a" not in summary
