"""Pure-function checks for GPU starter, no transformers or GPU required."""
import numpy as np

from src.finetune_lora import render_pair, swap_labels


def test_render_pair():
    text = render_pair({
        "prompt": '["Describe a dummy example"]',
        "response_a": '["First invented answer"]',
        "response_b": '["Second invented answer"]',
    })
    assert "First invented answer" in text
    assert "Second invented answer" in text
    assert "Describe a dummy example" in text


def test_swapped_labels():
    assert np.array_equal(swap_labels([0, 1, 2]), [1, 0, 2])
