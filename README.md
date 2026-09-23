# LLM Classification Finetuning: Human Preference Prediction

Reproducible study of **A/B/tie human preference prediction** on Kaggle's [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning) dataset, with a longer-term research direction on response length, presentation-order effects, and conversational style.

**Status:** repository initialized. A CPU TF-IDF + logistic regression benchmark is implemented; **no leaderboard result or LLM fine-tuning result has been claimed**.

## Competition task

Given a prompt and responses from two anonymized language models, estimate three probabilities: `winner_model_a`, `winner_model_b`, `winner_tie`. Kaggle evaluates **multiclass log loss**. The official training data contain roughly 55K rows. The competition is a **code competition**: submit a Kaggle Notebook that creates `submission.csv` without internet access during scoring.

## Quick start (local)

1. Join the competition and accept its rules on Kaggle.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Download the competition data using `bash scripts/download_competition.sh` (requires Kaggle authentication), or use the Kaggle input dataset directly in a Notebook.
4. Train: `python -m src.baseline train --train data/train.csv --out artifacts`
5. Predict: `python -m src.baseline predict --test data/test.csv --model artifacts/baseline.joblib --out submission.csv`

A self-contained, **internet-off compatible** Kaggle starter Notebook is in [notebooks/kaggle_starter.ipynb](notebooks/kaggle_starter.ipynb). Upload it to Kaggle and attach the official competition dataset; its final cell creates `/kaggle/working/submission.csv`.

Run local tests with `pytest -q`. GitHub Actions runs these tests without using secrets or downloading restricted data.

## Authentication and data safety

- Use Kaggle's current API token via environment variable `KAGGLE_API_TOKEN` or your authenticated CLI. Do **not** share your token in issues, chats, notebooks, source files, or commits.
- The already-added GitHub repository secret is **not needed** by the local baseline or Kaggle-hosted Notebook. In GitHub Actions, a secret is available only to workflows that explicitly reference its exact name; this project deliberately has **no workflow that prints or uses tokens**.
- `data/*.csv`, checkpoints, `submission.csv` and credentials are gitignored. The competition dataset uses **CC BY-NC 4.0**; do not commit or republish it without checking the license and rules.
- **This repository is currently public.** Change Settings > General > Danger Zone > Change repository visibility if you intended a private project.

## Experimental design

1. Establish a leak-controlled random stratified validation split and record three-class log loss.
2. Compare the initial baseline with a transformer sequence classifier using parameter-efficient fine-tuning, conditional on available GPU/runtime and model license.
3. Independently examine response-length differences, A/B presentation order, and style markers on held-out data. Observational correlations do **not** establish that a style caused preferences. Document any annotation protocol.
4. Track seed, data version, split IDs (not original copyrighted text), hyperparameters, runtime, and failure cases.

See [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for evaluation and reproducibility safeguards.

## Project structure

- `src/baseline.py`: reproducible three-class starter model and inference.
- `notebooks/kaggle_starter.ipynb`: offline Kaggle Notebook submission starter.
- `scripts/download_competition.sh`: authenticated CLI download (local only).
- `tests/`: synthetic smoke tests, safe to run in public CI.
- `docs/`: research protocol and work log.
- `data/`, `artifacts/`: local, ignored experiment inputs and outputs.

**Competition reference:** Chiang et al., *LLM Classification Finetuning*, Kaggle (2024).
