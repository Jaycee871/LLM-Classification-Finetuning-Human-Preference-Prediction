# Kaggle GPU pilot: reproducibility and setup

**GPU experiment prepared for real Kaggle execution. Actual GPU performance remains unverified until the run completes.** This experiment is independent of the GitHub API-secret check. Kaggle Notebooks already provide official competition Inputs after joining; never pass a GitHub token into a public Notebook.

## Goal
Pilot a locally attached **Qwen2.5-0.5B-Instruct** model with a three-class sequence-classification head and LoRA adapters (`q_proj`, `v_proj`). Report multiclass log loss for an untouched, seeded stratified 15% holdout; probe A/B-swap inconsistency on up to 128 held-out examples.

## Prerequisites
1. Join the [competition](https://www.kaggle.com/competitions/llm-classification-finetuning) and accept its data/rules.
2. In a Kaggle GPU Notebook, attach the official `llm-classification-finetuning` dataset as Input.
3. Attach a legitimate, complete `Qwen/Qwen2.5-0.5B-Instruct` model snapshot as a *separate Kaggle Input*, including tokenizer, config and all weight files. Check the model's terms and license. Edit `BASE` to the actual Input folder containing `config.json`.
4. Ensure `transformers`, `peft`, `accelerate`, `sentencepiece` are available. If absent, prepare dependencies using the Kaggle Dependency Manager **before** submitting with Internet disabled. See `requirements-gpu.txt`. Use Kaggle's CUDA-enabled PyTorch.
5. Enable GPU; disable Notebook Internet for your submission. Import `notebooks/kaggle_gpu_lora_pilot.ipynb` into Kaggle, attach Inputs, edit `BASE`, then **Run All**.

The GPU Notebook **embeds its own copies** of `src/baseline.py` and `src/finetune_lora.py`, and writes them under `/kaggle/working/src`. It does not clone GitHub or download weights at runtime. The model itself is not included in this repo. **After changing either source module, run `python scripts/sync_gpu_notebook.py` and commit the regenerated Notebook.** CI runs `python scripts/sync_gpu_notebook.py --check` and extracts/compiles both embedded modules, so stale copies cannot silently reach a submission.

## Pilot hyperparameters
- Fixed seed 42; 85/15 stratified split before any augmentation; validation remains untouched.
- Pilot caps *train-only* original examples at 4,000, then adds A/B-inverted pairs with reversed labels. The cap is adjustable; full run is `pilot_rows=0` once runtime and resource budgets have been verified.
- User prompt is limited to 1,200 characters, each response to 2,400 characters, and joint tokenization to 384 tokens. This sacrifices long-context fidelity for a first resource-bounded pilot. No model names are included because test data do not provide them.
- LoRA rank 8, alpha 16, dropout 0.05; batch 2, accumulation 8, one epoch, initial learning rate 2e-4. Evaluation is from an untouched validation partition.
- Output: `/kaggle/working/gpu_pilot/gpu_pilot_metrics.json`, `/kaggle/working/gpu_pilot/adapter/`, and `/kaggle/working/submission.csv`.

## Important caveats
- **Runtime**: preview `test.csv` is small; hidden evaluation is approximately 25K rows. Budget for hidden-test inference within the official **9-hour GPU limit**. Do not claim submission compatibility until a full commit/test has completed. If runtime fails, reduce `max_length`, eval batch size, or train-only pilot size first, then document changes.
- **Software**: the optional GPU stack has not yet been tested on an actual Kaggle GPU. If Kaggle's provided libraries conflict, capture exact versions and error logs (without credentials) before changing requirements.
- **Inference with saved adapter**: `adapter/` contains LoRA parameters and classification head, **not** the base model. Keep the same base model and tokenizer when reloading.
- **Methods**: pair reversal during training is a regularization technique; the swap diagnostic quantifies *model* instability under a controlled input perturbation, not whether human judges have presentation bias. Length analysis is observational.
- **Privacy and licensing**: do not commit Kaggle CSVs, raw conversations, checkpoint weights, or personal API credentials. Keep aggregate metrics and stripped training configuration in the repo only after checking disclosure rights.

## CPU fallback and auditing
The CPU starter `notebooks/kaggle_starter.ipynb` requires no pretrained weights and writes the same three target probabilities. Once official `train.csv` is accessible, run:

```bash
python -m src.baseline train --train data/train.csv --out artifacts
python -m src.diagnostics --data data/train.csv --out artifacts/aggregate_diagnostics.json
```

The aggregate diagnostic reports length-binned A/B/tie counts and a row-level bootstrap CI; it cannot infer causal effects or generalize individual preferences. For **held-out** order-stability analysis, pass a model trained on a non-overlapping partition and run `swap_audit` on the validation partition. Passing a full-data-trained baseline plus `--model` to the CLI only provides a pipeline sanity check.


## 2026-09-24 first-run setup

See [the dedicated GPU first-run guide](GPU_FIRST_RUN.md). This pilot uses the officially published [QwenLM 0.5B base Kaggle model](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/0.5b/1) under Apache 2.0. It trains on 2,000 original rows plus A/B reversal and evaluates on a separate 1,200-row held-out validation subset. We also fit a length-only reference on the identical training partition and evaluate the two models on the same held-out rows. The workflow does not automatically submit an additional competition entry.
