"""Synthetic regression tests for the exact-prompt grouped holdout."""
import json

import numpy as np
import pandas as pd

from src.baseline import TARGETS
from scripts.run_grouped_length_robustness import (
    grouped_indices, prompt_group_keys, run_grouped_audit
)


def fake_grouped_data(n_groups=30):
    # Two labeled rows of each class per prompt, repeated prompts across rows.
    rows = []
    for g in range(n_groups):
        for cl in range(3):
            for repeat in range(2):
                rows.append({
                    "id": len(rows),
                    "prompt": json.dumps([f"Artificial prompt number {g}"]),
                    "response_a": json.dumps(["Long synthetic explanation " * (cl + 1)]),
                    "response_b": json.dumps(["Brief example " * (4-cl)]),
                    **{name: int(cl == j) for j, name in enumerate(TARGETS)}
                })
    return pd.DataFrame(rows)


def test_prompt_group_prevents_exact_leakage():
    frame = fake_grouped_data()
    keys = prompt_group_keys(frame)
    a, b = grouped_indices(keys, test_size=0.2, seed=17)
    assert set(keys[a]).isdisjoint(set(keys[b]))
    assert len(a) + len(b) == len(frame)


def test_grouped_audit_on_synthetic_data(tmp_path):
    frame = fake_grouped_data()
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    result = run_grouped_audit(path, tmp_path / "metrics")
    assert result["total_rows"] == len(frame)
    assert result["distinct_exact_prompt_hashes"] == 30
    assert result["exact_group_overlap"] == 0
    assert sum(result["label_counts_outer_test"].values()) == result["outer_testing_rows"]
    assert result["tuned_c_inner_only"] in [0.01, 0.1, 1.0, 10.0, 100.0]
    assert np.isfinite(result["outer_log_loss_length_only"])
    assert (tmp_path / "metrics" / "summary.json").is_file()
