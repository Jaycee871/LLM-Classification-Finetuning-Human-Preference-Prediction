# LLM Classification Finetuning: Human Preference Prediction

Reproducible study of **A/B/tie human preference prediction** on Kaggle's [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning) dataset, with a longer-term research direction on response length, presentation-order effects, and conversational style.

**Status:** CPU TF-IDF baseline, synthetic tests, aggregate preference diagnostics, and optional GPU LoRA pilot code are implemented. **The real-data evaluation, GPU run, competition submission and Kaggle API verification have not yet been completed.**

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
- The already-added GitHub repository secret is **not needed** by the local baseline or Kaggle-hosted Notebook. In GitHub Actions, a secret is available only to workflows that explicitly reference its exact name; this project has only an **optional manually triggered** verification workflow that reads the configured secret but does not print it.
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

For an **optional GPU** pilot, see [the Kaggle LoRA guide](docs/GPU_PILOT.md) and the [offline GPU Notebook](notebooks/kaggle_gpu_lora_pilot.ipynb). Attach official competition data and a complete base model as Kaggle Inputs. The Notebook includes all required project source code, trains LoRA adapters for a Qwen2.5-0.5B three-way sequence classifier, evaluates a held-out subset, and prepares the submission. **It has not yet been run on official data or Kaggle GPUs.** Install dependencies in advance via Kaggle's Dependency Manager; the scoring Notebook must keep Internet disabled. Pilot size is capped deliberately, so its metrics are preliminary.

## Next experiment: real-data aggregate CPU pilot

A new [official-data CPU pilot workflow](.github/workflows/official-data-cpu-pilot.yml) downloads `train.csv` from Kaggle into an ephemeral GitHub runner, trains the CPU TF-IDF model on a **seeded, stratified 12,000-row pilot sample**, and uploads **aggregate JSON only** (no raw text, checkpoint or API token). It is set to run once when its workflow file first reaches `main`; future runs are available through **Actions > Kaggle official-data CPU pilot > Run workflow**. A verified API key does **not** guarantee download permissions if competition rules were not accepted. See [setup, limits and failure guide](docs/OFFICIAL_CPU_PILOT.md).

The workflow's aggregate validation numbers are a research *pilot*, not Kaggle leaderboard results. Qwen GPU training still requires a separate Kaggle GPU Notebook execution.

## Actual results and competition-ready CPU Notebook

Two CPU experiments have now **executed on the official competition dataset** (57,477 labeled rows, identical 12,000-row exploratory pilot, 1,800 outer-validation rows). See [the first-results discussion](docs/FIRST_RESULTS.md) and the tracked aggregate JSON in [`results/`](results/).

| Exploratory predictor | Held-out multiclass log loss |
| --- | ---: |
| Original TF-IDF / pair-swap logistic regression | 1.174414 |
| Uniform three-class probabilities | 1.098612 |
| Training-class-prior probabilities | 1.097219 |
| **Inner-tuned length-only logistic regression** | **1.057783** |

**Caution:** These exploratory results share a row-random outer fold; prompts may repeat across train/validation. Repeated model development against that fold is *not* independent scientific confirmation. Character length does not measure warmth or causally determine preference.

A new [self-contained Kaggle length-only submission Notebook](notebooks/kaggle_length_submission.ipynb) is prepared. On Kaggle, attach the official competition dataset through **Add Input**, keep **Internet Off**, run on CPU, then commit the Notebook to create `/kaggle/working/submission.csv`. **No leaderboard submission or score is claimed.** The Notebook is synchronized against authoritative modules via `python scripts/sync_length_notebook.py`; CI checks for drift.

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
