"""Exercise real train_and_predict control flow without expensive GPU weights.

Stubs cover hardware/transformers APIs only; dataset splitting, prompt rendering,
probabilities, CSV submission and JSON metrics persistence are real.
"""
from argparse import Namespace
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.finetune_lora import train_and_predict
from tests_fixture import synthetic_data


def install_fake_gpu_stack(monkeypatch):
    torch = ModuleType("torch")
    torch.__path__ = []
    torch.cuda = SimpleNamespace(is_available=lambda: True,
                                 is_bf16_supported=lambda: False)
    torch.manual_seed = lambda value: None
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torchutils = ModuleType("torch.utils")
    torchutils.__path__ = []
    torchdata = ModuleType("torch.utils.data")
    torchdata.Dataset = type("Dataset", (), {})

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "[EOS]"
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
            return cls()
        def __call__(self, text, truncation, max_length):
            assert "Response A:" in text and "Response B:" in text
            assert truncation and max_length > 0
            return {"input_ids": [1, 2, 3]}
        def save_pretrained(self, path):
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=None, use_cache=True)
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"] and kwargs["num_labels"] == 3
            return cls()
        def save_pretrained(self, path):
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.train_dataset = kwargs["train_dataset"]
            self.tokenizer = kwargs["processing_class"]
        def train(self):
            assert len(self.train_dataset) >= 30
            assert "labels" in self.train_dataset[0]
        def predict(self, dataset):
            if len(dataset):
                assert "input_ids" in dataset[0]
            return SimpleNamespace(predictions=np.tile(
                [[0.4, 0.1, 0.2]], (len(dataset), 1)
            ))

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    transformers.AutoModelForSequenceClassification = FakeModel
    transformers.DataCollatorWithPadding = lambda **kwargs: kwargs
    transformers.TrainingArguments = lambda **kwargs: kwargs
    transformers.Trainer = FakeTrainer
    peft = ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: kwargs
    peft.TaskType = SimpleNamespace(SEQ_CLS="SEQ_CLS")
    peft.get_peft_model = lambda model, config: model
    for key, module in {
        "torch": torch,
        "torch.utils": torchutils,
        "torch.utils.data": torchdata,
        "transformers": transformers,
        "peft": peft,
    }.items():
        monkeypatch.setitem(sys.modules, key, module)


@pytest.mark.parametrize("produce_submission", [False, True, "invalid"])
def test_gpu_training_flow_persists_reported_metrics(tmp_path, monkeypatch,
                                                     produce_submission):
    install_fake_gpu_stack(monkeypatch)
    raw = pd.concat([synthetic_data()] * 3, ignore_index=True)
    train_csv = tmp_path / "train.csv"
    raw.to_csv(train_csv, index=False)
    test_csv = tmp_path / "test.csv"
    cols = ["id", "prompt", "response_a", "response_b"]
    if produce_submission == "invalid":
        cols.remove("id")
    raw[cols].iloc[:4].to_csv(test_csv, index=False)
    model_dir = tmp_path / "offline_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "experiment"
    submission = tmp_path / "submission.csv"
    args = Namespace(
        train=str(train_csv),
        test=str(test_csv) if produce_submission else None,
        base_model=str(model_dir),
        output=str(output_dir),
        submission=str(submission),
        pilot_rows=30,
        max_length=64,
        epochs=1.0,
        batch_size=2,
        eval_batch_size=4,
        grad_accum=8,
        learning_rate=2e-4,
        seed=42,
        swap_train=True,
    )
    if produce_submission == "invalid":
        with pytest.raises(ValueError, match="Test data require id column"):
            train_and_predict(args)
        partial = json.loads(
            (output_dir / "gpu_pilot_metrics.json").read_text(encoding="utf-8")
        )
        assert np.isfinite(partial["validation_log_loss"])
        assert partial["pipeline_status"] == "downstream_failed"
        assert partial["downstream_error_type"] == "ValueError"
        assert "submission_rows" not in partial
        return

    reported = train_and_predict(args)
    persisted = json.loads(
        (output_dir / "gpu_pilot_metrics.json").read_text(encoding="utf-8")
    )
    assert persisted == reported
    assert persisted["validation_rows"] > 0
    assert persisted["train_rows_after_augmentation"] == 60
    assert (output_dir / "adapter").is_dir()
    assert np.isfinite(persisted["validation_log_loss"])
    assert np.isfinite(persisted["matched_length_reference_log_loss"])
    if produce_submission:
        output = pd.read_csv(submission)
        assert len(output) == 4
        assert reported["submission_rows"] == 4
        assert reported["pipeline_status"] == "submission_completed"
        assert np.allclose(output.iloc[:, 1:].sum(axis=1), 1)
    else:
        assert not submission.exists()
        assert "submission_rows" not in reported
        assert reported["pipeline_status"] == "adapter_saved"
