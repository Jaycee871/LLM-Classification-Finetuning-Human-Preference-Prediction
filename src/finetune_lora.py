"""Optional GPU pilot: fine-tune an offline Qwen2.5-0.5B sequence classifier with LoRA.

Only run after attaching legitimately accessible model weights and official Kaggle
competition data. This is a pilot; no actual GPU experiment is claimed here.

With offline Kaggle submissions, attach the *complete* base model directory
(config, tokenizer and weights) as an Input; never fetch from Hugging Face at runtime.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from src.baseline import TARGETS, flatten_messages, flip_pairs, get_labels, normalized_frame


def render_pair(row):
    """Fixed prompt template and truncation; do not include train-only model names."""
    question = flatten_messages(row["prompt"], max_chars=1200)
    a = flatten_messages(row["response_a"], max_chars=2400)
    b = flatten_messages(row["response_b"], max_chars=2400)
    return (
        "A human gave two chatbots the same user request.\n"
        f"User request: {question}\n"
        f"Response A: {a}\n"
        f"Response B: {b}\n"
        "Predict whether the human prefers response A, response B, or a tie."
    )


def swap_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Expected A=0, B=1, tie=2")
    return np.where(labels == 0, 1, np.where(labels == 1, 0, 2))


def train_and_predict(args):
    # Lazy imports ensure that unit tests do not need GPU-only dependencies.
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, Trainer, TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This LoRA pilot requires a CUDA GPU; use the CPU baseline otherwise.")
    model_dir = Path(args.base_model).expanduser()
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            "Supply the complete offline base model directory via --base-model; "
            f"no config.json found in {model_dir}"
        )
    if args.pilot_rows != 0 and args.pilot_rows < 30:
        raise ValueError("pilot_rows must be 0 (all rows) or at least 30")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    raw = pd.read_csv(args.train)
    df = normalized_frame(raw)
    y = get_labels(raw)
    if np.min(np.bincount(y, minlength=3)) < 2:
        raise ValueError("Each class needs at least two rows")
    x_train, x_val, y_train, y_val = train_test_split(
        df, y, test_size=0.15, stratify=y, random_state=args.seed
    )
    # Cap *training only*. Keep validation untouched for honest pilot comparison.
    if args.pilot_rows and len(x_train) > args.pilot_rows:
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=args.pilot_rows,
            stratify=y_train, random_state=args.seed
        )
    if args.swap_train:
        x_original, y_original = x_train.copy(), y_train.copy()
        x_train = pd.concat(
            [x_original, flip_pairs(x_original)], ignore_index=True
        )
        y_train = np.concatenate([y_original, swap_labels(y_original)])

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=3, torch_dtype=dtype,
        local_files_only=True, trust_remote_code=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["score"],
        ),
    )

    class PairDataset(Dataset):
        def __init__(self, frame, labels=None):
            self.texts = [render_pair(row) for row in frame.to_dict("records")]
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            encoded = tokenizer(
                self.texts[i], truncation=True, max_length=args.max_length
            )
            if self.labels is not None:
                encoded["labels"] = int(self.labels[i])
            return encoded

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = TrainingArguments(
        output_dir=str(output / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=25,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=PairDataset(x_train, y_train),
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    val_logits = trainer.predict(PairDataset(x_val)).predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    val_prob = softmax(np.asarray(val_logits, dtype=np.float64), axis=-1)
    metrics = {
        "validation_log_loss": float(log_loss(y_val, val_prob, labels=[0, 1, 2])),
        "validation_rows": len(x_val),
        "train_rows_after_augmentation": len(x_train),
        "pilot_rows": args.pilot_rows,
        "seed": args.seed,
        "max_length_tokens": args.max_length,
        "base_model_dir": model_dir.name,
        "training_type": "Qwen2.5-0.5B sequence classification head + LoRA",
        "caution": "Preliminary pilot; independent replication and full-data run pending.",
    }
    # Probe original/swap consistency on a bounded held-out subset.
    subset = x_val.iloc[: min(128, len(x_val))]
    original_logits = trainer.predict(PairDataset(subset)).predictions
    swapped_logits = trainer.predict(PairDataset(flip_pairs(subset))).predictions
    if isinstance(original_logits, tuple):
        original_logits = original_logits[0]
    if isinstance(swapped_logits, tuple):
        swapped_logits = swapped_logits[0]
    original_prob = softmax(np.asarray(original_logits, dtype=np.float64), axis=-1)
    swapped_prob = softmax(np.asarray(swapped_logits, dtype=np.float64), axis=-1)[:, [1, 0, 2]]
    metrics["swap_probe_mean_abs_difference"] = float(
        np.abs(original_prob - swapped_prob).mean()
    )
    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    if args.test:
        test = pd.read_csv(args.test)
        if "id" not in test.columns:
            raise ValueError("Test data require id column")
        test_frame = normalized_frame(test)
        test_logits = trainer.predict(PairDataset(test_frame)).predictions
        if isinstance(test_logits, tuple):
            test_logits = test_logits[0]
        probs = softmax(np.asarray(test_logits, dtype=np.float64), axis=-1)
        submission = pd.DataFrame(probs, columns=TARGETS)
        submission.insert(0, "id", test["id"])
        submission_path = Path(args.submission)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)
        metrics["submission_rows"] = int(len(submission))
        print(f"Submission written to {submission_path}")
    # Persist after optional test inference so the artifact includes submission_rows.
    (output / "gpu_pilot_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--test", default=None)
    p.add_argument("--base-model", required=True,
                   help="Complete *local* Qwen2.5-0.5B-Instruct weights/tokenizer folder")
    p.add_argument("--output", default="artifacts/gpu_pilot")
    p.add_argument("--submission", default="submission.csv")
    p.add_argument("--pilot-rows", type=int, default=4000,
                   help="Training cap excluding held-out validation; 0 uses all train rows")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-swap-train", action="store_false", dest="swap_train")
    p.set_defaults(swap_train=True)
    return p.parse_args()


if __name__ == "__main__":
    train_and_predict(parse_args())
