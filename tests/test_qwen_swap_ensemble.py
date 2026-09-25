"""CPU-only tests for the Qwen 1024 A/B swap ensemble."""
import numpy as np
import pytest

from src.qwen_swap_ensemble import (
    align_swapped,
    arithmetic_ensemble,
    geometric_ensemble,
    disagreement,
    summarize,
)


def test_align_swapped_reverses_a_b_columns_only():
    p=np.array([[0.1,0.7,0.2],[0.8,0.1,0.1]])
    got=align_swapped(p)
    assert np.allclose(got,[[0.7,0.1,0.2],[0.1,0.8,0.1]])


def test_ensemble_probabilities_are_normalized_and_symmetric():
    a=np.array([[0.8,0.1,0.1],[0.2,0.5,0.3]])
    b=np.array([[0.4,0.5,0.1],[0.5,0.2,0.3]])
    for fn in (arithmetic_ensemble,geometric_ensemble):
        p=fn(a,b)
        q=fn(b,a)
        assert np.allclose(p.sum(axis=1),1)
        assert np.allclose(p,q)


def test_summary_and_disagreement():
    y=np.array([0,1,2])
    p=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.1,0.2,0.7]])
    s=summarize(y,p)
    assert s["accuracy"]==1.0
    assert s["multiclass_log_loss"]>0
    assert sum(s["predicted_counts"].values())==3
    other=np.array([[0.1,0.8,0.1],[0.8,0.1,0.1],[0.1,0.2,0.7]])
    d=disagreement(p,other)
    assert d["fraction_rows_with_changed_predicted_winner"]==pytest.approx(2/3)
    assert d["mean_absolute_probability_disagreement"]>0
