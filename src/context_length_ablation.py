"""Inference-only context-length ablation for the SAVED Qwen-LoRA adapter.

This module NEVER trains model weights and NEVER submits to Kaggle. It evaluates
one previously trained private adapter on the exact same 1,200-row pilot
validation subset while changing only the tokenizer max_length. Aggregate-only
JSON is written; no prompts, IDs, per-row predictions, or model artifacts.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.saved_adapter_inference import (
    aggregate_predictions,
    character_cap_report,
    exact_pilot_validation,
    predict_in_batches,
    token_truncation_report,
    validated_swap_rows,
)
from src.baseline import flip_pairs


DEFAULT_CONTEXTS = (384, 768, 1024)
DEFAULT_BATCHES = {384: 8, 768: 4, 1024: 2}


def validate_contexts(contexts):
    values = tuple(int(x) for x in contexts)
    if not values or any(x <= 0 for x in values):
        raise ValueError("Context lengths must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("Context lengths must be unique")
    return values


def delta_from_baseline(results, baseline_context):
    base = results[str(baseline_context)]
    out = {}
    for key, record in results.items():
        if int(key) == baseline_context:
            continue
        out[key] = {
            "delta_log_loss_vs_baseline":
                record["multiclass_log_loss"] - base["multiclass_log_loss"],
            "delta_accuracy_vs_baseline":
                record["accuracy"] - base["accuracy"],
            "delta_ece_vs_baseline":
                record["expected_calibration_error_10_bin"] -
                base["expected_calibration_error_10_bin"],
            "delta_swap_mean_abs_disagreement_vs_baseline":
                record["swap_diagnostic"]["mean_absolute_probability_disagreement"] -
                base["swap_diagnostic"]["mean_absolute_probability_disagreement"],
            "delta_fraction_swap_changed_winner_vs_baseline":
                record["swap_diagnostic"]["fraction_rows_with_changed_predicted_winner"] -
                base["swap_diagnostic"]["fraction_rows_with_changed_predicted_winner"],
            "delta_truncation_fraction_vs_baseline":
                record["rendered_token_truncation"]["fraction_rendered_sequence_exceeds_token_limit"] -
                base["rendered_token_truncation"]["fraction_rendered_sequence_exceeds_token_limit"],
        }
    return out


def run_ablation(args):
    # Lazy GPU imports preserve CPU-only synthetic CI.
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    contexts = validate_contexts(args.contexts)
    base_dir = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError("Missing local Qwen base config")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError("Missing existing adapter_config.json")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Missing existing adapter_model.safetensors")
    with safe_open(
        adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        keys = list(handle.keys())
    head_keys = [k for k in keys if "score" in k and k.endswith(".weight")]
    if not head_keys:
        raise ValueError("Saved adapter does not contain classifier score weights")
    if not torch.cuda.is_available():
        raise RuntimeError("Context ablation requires Kaggle CUDA for inference only")

    raw = pd.read_csv(args.train)
    x_val, y_val = exact_pilot_validation(
        raw, seed=args.seed, max_validation_rows=args.max_validation_rows
    )
    raw_val = raw.loc[x_val.index]
    n_swap = validated_swap_rows(args.swap_rows, len(x_val))

    tokenizer = AutoTokenizer.from_pretrained(
        base_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model_max = getattr(tokenizer, "model_max_length", None)
    finite_model_max = (
        int(model_max) if isinstance(model_max, (int, np.integer))
        and model_max < 10**9 else None
    )
    if finite_model_max and max(contexts) > finite_model_max:
        raise ValueError(
            f"Requested context {max(contexts)} exceeds tokenizer model_max_length "
            f"{finite_model_max}"
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForSequenceClassification.from_pretrained(
        base_dir, num_labels=3, torch_dtype=dtype,
        local_files_only=True, trust_remote_code=False,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=False, local_files_only=True
    )
    model.to("cuda:0")
    model.eval()

    results = {}
    for max_length in contexts:
        batch_size = int(args.batch_size_overrides.get(
            max_length, DEFAULT_BATCHES.get(max_length, 2)
        ))
        started = time.perf_counter()
        probabilities = predict_in_batches(
            model, tokenizer, x_val, "cuda:0",
            max_length=max_length, batch_size=batch_size
        )
        swapped = predict_in_batches(
            model, tokenizer, flip_pairs(x_val.iloc[:n_swap]), "cuda:0",
            max_length=max_length, batch_size=batch_size
        )[:, [1, 0, 2]]
        metrics = aggregate_predictions(y_val, probabilities)
        metrics["swap_diagnostic"] = aggregate_predictions(
            y_val[:n_swap], probabilities[:n_swap],
            swapped_aligned=swapped
        )["swap_diagnostic"]
        metrics["rendered_token_truncation"] = token_truncation_report(
            tokenizer, x_val, max_length=max_length
        )
        metrics["inference_batch_size"] = batch_size
        metrics["wall_clock_seconds_including_swap_probe"] = (
            time.perf_counter() - started
        )
        results[str(max_length)] = metrics
        torch.cuda.empty_cache()
        print(json.dumps({
            "context": max_length,
            "log_loss": metrics["multiclass_log_loss"],
            "accuracy": metrics["accuracy"],
            "ece": metrics["expected_calibration_error_10_bin"],
            "swap_changed_winner":
                metrics["swap_diagnostic"]["fraction_rows_with_changed_predicted_winner"],
            "truncation":
                metrics["rendered_token_truncation"]["fraction_rendered_sequence_exceeds_token_limit"],
            "seconds": metrics["wall_clock_seconds_including_swap_probe"],
        }, indent=2))

    baseline = int(args.baseline_context)
    if baseline not in contexts:
        raise ValueError("--baseline-context must be one of --contexts")
    best_context = min(
        contexts, key=lambda c: results[str(c)]["multiclass_log_loss"]
    )
    report = {
        "type": "saved_qwen_context_length_ablation",
        "no_new_training": True,
        "no_competition_submission": True,
        "validation_rows": int(len(y_val)),
        "swap_probe_rows": int(n_swap),
        "seed": int(args.seed),
        "contexts_tokens": list(contexts),
        "baseline_context_tokens": baseline,
        "tokenizer_reported_model_max_length": finite_model_max,
        "saved_classifier_weight_tensors": len(head_keys),
        "character_cap_report": character_cap_report(raw_val),
        "results": results,
        "deltas_from_baseline": delta_from_baseline(results, baseline),
        "lowest_observed_validation_log_loss_context": int(best_context),
        "lowest_observed_validation_log_loss":
            float(results[str(best_context)]["multiclass_log_loss"]),
        "interpretation_limits": [
            "The same previously observed validation set is reused across all contexts; "
            "choosing the best context from this table is exploratory tuning, not independent evidence.",
            "Increasing tokenizer max_length cannot recover text already removed by earlier "
            "character caps in normalized_frame/render_pair.",
            "The adapter was trained with 384-token inputs. Evaluating it at longer context "
            "lengths changes inference exposure without retraining positional usage.",
            "Any improvement or deterioration cannot be attributed solely to truncation without "
            "a prospectively isolated validation set and balanced-template controls.",
            "A/B swap diagnostics describe model order sensitivity, not human presentation bias.",
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-validation-rows", type=int, default=1200)
    parser.add_argument("--swap-rows", type=int, default=128)
    parser.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--baseline-context", type=int, default=384)
    args = parser.parse_args()
    args.batch_size_overrides = DEFAULT_BATCHES
    run_ablation(args)
