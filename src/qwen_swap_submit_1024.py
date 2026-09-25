"""Authorized experimental Kaggle submission: Qwen 1024-token A/B arithmetic ensemble.

Uses the existing saved LoRA adapter. Each test pair is inferred twice, once in
original A/B order and once with responses swapped. Swapped probabilities are
mapped back to the original label order and averaged arithmetically.
No training occurs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import TARGETS, normalized_frame
from src.qwen_swap_ensemble import (
    align_swapped,
    arithmetic_ensemble,
    predict_probs,
    swap_frame,
)


def run(args):
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base_dir = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError("Missing Qwen base config")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError("Missing saved adapter config")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Missing saved adapter weights")
    with safe_open(
        adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        head_keys = [k for k in handle.keys() if "score" in k and k.endswith(".weight")]
    if not head_keys:
        raise ValueError("Saved adapter does not contain classifier score weights")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is required for ensemble competition inference")

    test = pd.read_csv(args.test)
    if "id" not in test.columns:
        raise ValueError("Competition test.csv requires id")
    frame = normalized_frame(test)

    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer lacks pad and EOS")
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForSequenceClassification.from_pretrained(
        base_dir,
        num_labels=3,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=False, local_files_only=True
    )
    model.to("cuda:0")
    model.eval()

    original = predict_probs(
        model, tokenizer, frame,
        max_length=args.max_length, batch_size=args.batch_size
    )
    swapped_raw = predict_probs(
        model, tokenizer, swap_frame(frame),
        max_length=args.max_length, batch_size=args.batch_size
    )
    swapped_aligned = align_swapped(swapped_raw)
    ensemble = arithmetic_ensemble(original, swapped_aligned)

    if ensemble.shape != (len(test), 3):
        raise ValueError(f"Expected {len(test)}x3 ensemble probabilities, got {ensemble.shape}")
    if not np.isfinite(ensemble).all() or np.any(ensemble < 0):
        raise ValueError("Invalid ensemble probabilities")
    if not np.allclose(ensemble.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Ensemble probabilities are not normalized")

    submission = pd.DataFrame(ensemble, columns=TARGETS)
    submission.insert(0, "id", test["id"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)

    disagreement = np.abs(original - swapped_aligned)
    print(f"Wrote {len(submission)} ensemble rows to {out}")
    print("Context tokens:", args.max_length)
    print("Batch size:", args.batch_size)
    print("Preview mean original-vs-swapped probability disagreement:", float(disagreement.mean()))
    print("Preview changed-winner fraction:", float(
        np.mean(original.argmax(axis=1) != swapped_aligned.argmax(axis=1))
    ))
    return len(submission)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", default="/kaggle/working/submission.csv")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    run(p.parse_args())
