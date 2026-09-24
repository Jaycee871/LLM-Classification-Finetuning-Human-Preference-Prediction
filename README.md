# LLM Classification Finetuning: Human Preference Prediction

Reproducible study of **A/B/tie human preference prediction** on Kaggle's [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning) dataset, with a longer-term research direction on response length, presentation-order effects, and conversational style.

**Status:** CPU TF-IDF baseline, synthetic tests, aggregate preference diagnostics, and optional GPU LoRA pilot code are implemented. **Kaggle API verification and two official-data CPU pilots are complete. The length-only Kaggle entry scored 1.06530 (rank 157 when observed on 2026-09-24). Private Qwen2.5 0.5B LoRA GPU pilot version 2 also completed successfully; its held-out multiclass log loss was 1.963875 versus 1.066153 for the length-only reference on identical validation rows. No Qwen competition entry was submitted.**

## Competition task

Given a prompt and responses from two anonymized language models, estimate three probabilities: `winner_model_a`, `winner_model_b`, `winner_tie`. Kaggle evaluates **multiclass log loss**. The official training data contain roughly 55K rows. The competition is a **code competition**: submit a Kaggle Notebook that creates `submission.csv` without internet access during scoring.

## Quick start (local)

1. Join the competition and accept its rules on Kaggle.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Download the competition data using `bash scripts/download_competition.sh` (requires Kaggle authentication), or use the Kaggle input dataset directly in a Notebook.
4. Train: `python -m src.baseline train --train data/train.csv --out artifacts`
5. Predict: `python -m src.baseline predict --test data/test.csv --model artifacts/baseline.joblib --out submission.csv`

A self-contained, **internet-off compatible** Kaggle starter Notebook is in [notebooks/kaggle_starter.ipynb](notebooks/kaggle_starter.ipynb). Upload it to Kaggle and attach the official competition dataset; its final cell creates `/kaggle/working/submission.csv`.

Run local tests with `python -m pytest -q`. GitHub Actions runs these tests without using secrets or downloading restricted data.

## Authentication and data safety

- Use Kaggle's current API token via environment variable `KAGGLE_API_TOKEN` or your authenticated CLI. Do **not** share your token in issues, chats, notebooks, source files, or commits.
- The already-added GitHub repository secret is **not needed** by the local baseline or Kaggle-hosted Notebook. In GitHub Actions, a secret is available only to workflows that explicitly reference its exact name; the manual verification, official-data CPU and length-only pilot workflows explicitly read the configured secret for authenticated Kaggle requests. These workflows do not print the token, and only authorized GitHub collaborators can dispatch them.
- `data/*.csv`, checkpoints, `submission.csv` and credentials are gitignored. The competition dataset uses **CC BY-NC 4.0**; do not commit or republish it without checking the license and rules.
- **This repository is currently public.** Change Settings > General > Danger Zone > Change repository visibility if you intended a private project.

## Verify Kaggle API access (optional manual workflow)

Open **Actions > Verify Kaggle access (manual) > Run workflow**. This check reads the repository secret **KAGGLE_API_TOKEN** (preferred) or **KAGGLE_TOKEN** (legacy alias), then tries listing the competition's files. No token is displayed, and no competition data are downloaded or published. If your existing secret has another name, rename it to one of these; do not paste its value into source code. A successful check verifies this API request only; accepting the competition rules and running/submitting a notebook are separate steps.

## New: Preference-length audit and optional GPU pilot

Once competition data are downloaded, run the **aggregate** observational preference-length audit (no raw records in the output):

```bash
python -m src.diagnostics --data data/train.csv --out artifacts/aggregate_diagnostics.json
```

The script describes how often the longer response won, separately counts A/B/ties, reports a bootstrap interval, and can measure model order sensitivity with an optional trusted joblib artifact. These statistics are observational; they cannot establish that length caused the preference.

For an **optional GPU** pilot, see [the Kaggle LoRA guide](docs/GPU_PILOT.md) and the [offline GPU Notebook](notebooks/kaggle_gpu_lora_pilot.ipynb). Attach official competition data and a complete base model as Kaggle Inputs. The Notebook includes all required project source code, trains LoRA adapters for a Qwen2.5-0.5B three-way sequence classifier, evaluates a held-out subset, and prepares the submission. **This version has now executed successfully on official data using Kaggle T4; see [the actual GPU results](docs/FIRST_GPU_RESULTS.md).** Install dependencies in advance via Kaggle's Dependency Manager; the scoring Notebook must keep Internet disabled. Pilot size is capped deliberately, so its metrics are preliminary.

## Next experiment: real-data aggregate CPU pilot

A new [official-data CPU pilot workflow](.github/workflows/official-data-cpu-pilot.yml) downloads `train.csv` from Kaggle into an ephemeral GitHub runner, trains the CPU TF-IDF model on a **seeded, stratified 12,000-row pilot sample**, and uploads **aggregate JSON only** (no raw text, checkpoint or API token). The initial run has completed successfully; all future experimental runs require an explicit **Actions > Kaggle official-data CPU pilot > Run workflow** dispatch. A verified API key does **not** guarantee download permissions if competition rules were not accepted. See [setup, limits and failure guide](docs/OFFICIAL_CPU_PILOT.md).

The workflow's aggregate validation numbers are a research *pilot*, not Kaggle leaderboard results. The first Qwen GPU experiment completed via a separate private Kaggle Notebook, with results documented independently.

## Actual results and competition-ready CPU Notebook

Two CPU experiments have now **executed on the official competition dataset** (57,477 labeled rows, identical 12,000-row exploratory pilot, 1,800 outer-validation rows). See [the first-results discussion](docs/FIRST_RESULTS.md) and the tracked aggregate JSON in [`results/`](results/).

| Exploratory predictor | Held-out multiclass log loss |
| --- | ---: |
| Original TF-IDF / pair-swap logistic regression | 1.174414 |
| Uniform three-class probabilities | 1.098612 |
| Training-class-prior probabilities | 1.097219 |
| **Inner-tuned length-only logistic regression** | **1.057783** |

**Caution:** These exploratory results share a row-random outer fold; prompts may repeat across train/validation. Repeated model development against that fold is *not* independent scientific confirmation. Character length does not measure warmth or causally determine preference.

A new [self-contained Kaggle length-only submission Notebook](notebooks/kaggle_length_submission.ipynb) is prepared. On Kaggle, attach the official competition dataset through **Add Input**, keep **Internet Off**, run on CPU, then commit the Notebook to create `/kaggle/working/submission.csv`. **Version 2 successfully generated `submission.csv` and was submitted via Kaggle CLI; the displayed public leaderboard score was 1.06530 (rank 157 at screenshot time).** The Notebook is synchronized against authoritative modules via `python scripts/sync_length_notebook.py`; CI checks for drift.

## Experimental design

1. Establish a leak-controlled random stratified validation split and record three-class log loss.
2. Compare the initial baseline with a transformer sequence classifier using parameter-efficient fine-tuning, conditional on available GPU/runtime and model license.
3. Independently examine response-length differences, A/B presentation order, and style markers on held-out data. Observational correlations do **not** establish that a style caused preferences. Document any annotation protocol.
4. Track seed, data version, split IDs (not original copyrighted text), hyperparameters, runtime, and failure cases.

See [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for evaluation and reproducibility safeguards.

## Project structure

- `src/baseline.py`: reproducible three-class starter model and inference.
- `src/diagnostics.py`: aggregate response-length and A/B-order stability auditing.
- `src/finetune_lora.py`: optional GPU LoRA pilot; expects user-attached local model weights.
- `src/length_baseline.py`: exploratory length-only predictor with inner-fold regularization selection.
- `notebooks/kaggle_starter.ipynb`: offline Kaggle Notebook submission starter.
- `notebooks/kaggle_gpu_lora_pilot.ipynb`: offline Kaggle GPU pilot with bundled source.
- `notebooks/kaggle_length_submission.ipynb`: self-contained offline CPU Kaggle submission starter based on the observed length-only baseline.
- `scripts/download_competition.sh`: authenticated CLI download (local only).
- `tests/`: synthetic smoke tests, safe to run in public CI.
- `docs/`: research protocol and offline GPU pilot setup guide, plus official-data CPU pilot procedure.
- `data/`, `artifacts/`: local, ignored experiment inputs and outputs.

**Competition reference:** Chiang et al., *LLM Classification Finetuning*, Kaggle (2024).

## Kaggle Qwen2.5 0.5B GPU pilot

The [private Kaggle GPU run workflow](.github/workflows/launch-gpu-pilot.yml) created `packkwanlow/llm-preference-qwen05b-lora-pilot` on T4 and uses the officially published [QwenLM 0.5B base model](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/0.5b/1). It compared LoRA to a length-only classifier trained on exactly the same 2,000 original rows and evaluated on exactly the same 1,200-row held-out validation subset. Aggregate metrics only are copied back to GitHub. No automatic second competition submission is made.

**GPU environment note (first Kaggle attempt):** Private GPU Notebook v1 mounted successfully and started T4 but PEFT rejected Kaggle's bundled optional torchao 0.10.0 (requires >0.16.0). A guarded [GPU retry](docs/GPU_FIRST_RUN.md) now removes the incompatible optional package **offline**, before PyTorch/PEFT import. No additional competition submission will be made automatically.

## Completed Qwen GPU result and independent robustness follow-up

The [first actual Qwen GPU pilot](docs/FIRST_GPU_RESULTS.md) completed on Kaggle T4: **1.963875** log loss for LoRA versus **1.066153** for a length-only model on the **same 1,200 held-out examples**. The Qwen model had only 2,000 original training pairs and one epoch; the result is not evidence that all fine-tuned language models underperform. **Its preview output was not submitted to the competition.**

Before committing further GPU time, the new [exact-prompt grouped validation audit](docs/GROUPED_ROBUSTNESS.md) tests the existing length-only approach on non-overlapping prompt groups with inner-only regularization selection, comparing against simple class-prior and uniform baselines. The audit uses the official labeled data only inside an ephemeral runner; results shared to GitHub are aggregate-only. [Workflow](.github/workflows/grouped-length-robustness.yml). **The first grouped run has now completed** on all 57,477 official labeled rows: 8,504 rows in an outer exact-prompt-separated holdout, zero exact-prompt overlap, **length-only log loss 1.065217**, versus **1.097199** for a training-prior baseline. See [verified grouped results](docs/GROUPED_RESULTS.md). This grouped local validation is distinct from the public Kaggle score.

## Post-GPU diagnostic before further training

The first successful Qwen LoRA pilot's **1.963875** validation log loss was worse than its matched length-only reference **1.066153**. The new [zero-GPU diagnosis](docs/QWEN_DIAGNOSTICS.md) reuses the previously saved private Kaggle aggregate JSON without retraining, publishing raw data or submitting another competition entry. It separates the **observed** underperformance from hypotheses (new classification head, short exposure, truncation, calibration) that require further measurements.

## Saved-Qwen-adapter inference-only diagnosis

The first completed Qwen LoRA GPU pilot ran on 2,000 original pairs plus reversals and scored **1.963875** multiclass log loss on 1,200 held-out rows versus **1.066153** for the matched length-only reference. Its saved A/B-swap probe disagreement was **0.294159**. Rather than spending another GPU training run or another competition submission, the separate [private saved-adapter diagnostic Notebook](notebooks/diagnose_saved_qwen_adapter.ipynb) reloads the **previous private Kaggle kernel's saved LoRA adapter and classification head**, reproduces the same validation split, and measures classwise loss, prediction distribution, confidence/calibration, input truncation and A/B swap sensitivity. [Method and access restrictions](docs/SAVED_ADAPTER_INFERENCE_DIAGNOSTICS.md); [one-time validation-only workflow](.github/workflows/diagnose-saved-adapter.yml). Its only public artifact is an aggregate metrics JSON, never a dataset, row-level prediction or model weights. **No new Qwen competition submission is made.**

## Completed saved-adapter inference diagnosis (2026-09-24)

The separate **private** Kaggle Notebook successfully restored the original Qwen LoRA adapter **without any retraining** and analyzed the same 1,200-row pilot validation set. The restored model's multiclass loss was **1.964393** (previous run **1.963875**, difference **0.000519**). Diagnostics found **33.08% accuracy**, mean predicted confidence **66.98%**, ten-bin calibration error **0.3390**, **65.5% of rendered inputs above the 384-token cap**, and on 128 A/B-swap probes **66.41% changed winning class** after relabeling. These observations suggest specific follow-up tests but do not prove the cause of poor performance. [Full findings and limits](docs/QWEN_SAVED_ADAPTER_FINDINGS.md), [permanent aggregate JSON](results/qwen_saved_adapter_inference_2026-09-24.json), [actual successful Actions run](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/35904556787). **No additional Kaggle competition entry was submitted.**

## Inference-only Qwen context-length ablation

The next controlled diagnostic keeps the saved Qwen LoRA weights and the exact same 1,200-row pilot validation subset fixed, changing only tokenizer context length: **384, 768 and 1,024 tokens**. Each condition measures log loss, calibration, truncation and A/B swap instability. This directly tests whether the observed **65.5% truncation rate at 384 tokens** is associated with better or worse behavior when more context is exposed at inference. Because the adapter was trained at 384 tokens and the same validation set is reused, this is exploratory diagnosis rather than independent confirmation. [Protocol](docs/QWEN_CONTEXT_ABLATION.md), [private one-time workflow](.github/workflows/qwen-context-ablation.yml). No retraining or competition submission occurs.
