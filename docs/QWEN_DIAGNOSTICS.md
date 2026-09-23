# Existing Qwen pilot: zero-GPU diagnostic audit

The first successful Qwen2.5-0.5B LoRA Kaggle T4 pilot finished with multiclass log loss **1.963875**, versus **1.066153** for the length-only classifier trained on the same 2,000 original examples and evaluated on the same 1,200 held-out examples. Both scores came from Kaggle GPU notebook **version 2**, archived in [first GPU results](FIRST_GPU_RESULTS.md).

Before another GPU experiment, this diagnostic workflow **reuses the existing private Kaggle `gpu_pilot_metrics.json`**. It makes no new Kaggle kernel version, uses no accelerator, does not download official competition text or model weights, and makes **no competition submission**.

The script `scripts/diagnose_gpu_pilot.py` computes actual differences against a same-validation-set matched length model and a constant uniform 3-class baseline (natural-log loss = log(3)). If swap-probe mean absolute difference was saved, it reports it without presuming that it caused the high loss.

## What these metrics cannot tell us yet

Aggregate log loss by itself cannot distinguish a newly initialized sequence-classification head from undertraining, input truncation, calibration failure or predicted-class collapse. The correct next measurement is a **validation-only, inference-only** diagnostic that reloads the **existing saved adapter** and the exact base model, on exactly the original held-out split, and emits class-wise counts, log loss, confidence, entropy, token truncation rate and order-swap statistics. Keep all individual prompts, tokens and predictions private.

Avoid tuning models on the newly completed **8,504-row exact-prompt-grouped reporting holdout**. That analysis is independent robustness evidence for the length model, not a free repeated tuning set for GPU experiments.

[Zero-GPU diagnostic workflow](../.github/workflows/diagnose-existing-gpu.yml) runs once upon its intentionally named merge commit or through explicit manual dispatch, and uploads only aggregate derived metrics.
