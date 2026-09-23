# Official Kaggle training data: first aggregate CPU run

## What this automated run does
The GitHub Actions workflow `official-data-cpu-pilot.yml` downloads the official competition training CSV on an ephemeral GitHub runner using the **existing** secret `KAGGLE_API_TOKEN` (or `KAGGLE_TOKEN`). The download stays local to the runner. A fixed random seed draws a stratified 12,000-row subset (or all data if the dataset is smaller), then `src/baseline.py` fits and validates a TF-IDF logistic regression classifier. It compares a three-class uniform predictor and training-class-prior predictor **on the same validation split**.

The run additionally calculates aggregate descriptive associations between response length and selected answer. It saves the SHA-256 fingerprint of the official `train.csv` for reproducibility and uploads only `summary.json` as a GitHub Actions artifact.

**No model checkpoint, Kaggle CSV, conversation excerpts, row IDs, row-level predictions or secret values are uploaded.** The repository is public, so aggregate results may be publicly downloadable. The initial merge-triggered run is complete. To avoid accidental repeat downloads during workflow maintenance, the workflow is now **manual-dispatch only** under **Actions > Kaggle official-data CPU pilot > Run workflow**.

## Interpretation

The CPU model intentionally truncates each of the three text fields to 2,400 characters. Its default 85/15 stratified split is row-random, and duplicate or near-duplicate prompts may appear in both train and validation. Results are a pilot and must not be presented as a deduplicated generalization estimate, as Kaggle leaderboard results, or as evidence that answer length **causes** preference.

The official competition uses multiclass log loss and accepts code submissions only, with a nine-hour CPU/GPU notebook runtime limit and Internet turned off. This GitHub-run validation **does not submit to Kaggle**. Source: [competition overview](https://www.kaggle.com/competitions/llm-classification-finetuning/overview) and [data](https://www.kaggle.com/competitions/llm-classification-finetuning/data).

## If the run fails

1. `KAGGLE_API_TOKEN` or `KAGGLE_TOKEN` must be a GitHub repository secret, never a variable or committed credential.
2. A successful `competitions files` test does **not** guarantee that the competition rules were accepted for downloading `train.csv`. Confirm competition participation in the browser if download is rejected.
3. Look at the Actions failure step; avoid copying logs containing personal data or tokens.
4. When it succeeds, read the *aggregate* summary artifact from the workflow run, then run the offline CPU Notebook on Kaggle using official Inputs to test competition submission mechanics. GPU LoRA training remains separate and unexecuted until actually run on Kaggle GPU.
