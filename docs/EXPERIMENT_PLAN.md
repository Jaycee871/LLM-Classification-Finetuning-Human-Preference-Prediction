# Experiment plan (v0.1, 2026-09-23)

## Research question
How well can computational models predict A/B/tie choices from matched prompt-and-response pairs? Which associations between response length, order and response style persist under held-out evaluation?

## Data and access
Use only the Kaggle competition's officially provided `train.csv` and `test.csv` after accepting the rules. Never publish raw competition data or identifiers linked to user text. Note the published CC BY-NC 4.0 license and potentially offensive material.

## Current baseline (implemented)
- Random state: 42; stratified 85/15 train/validation split.
- Parse serialized conversational turns; truncate to 2,400 characters per field for an inexpensive CPU run.
- TF-IDF (1–2 grams, at most 35K terms) trained on **only the training partition**.
- Pairwise A-minus-B features, prompt features, symmetric response features, length covariates.
- Swap A and B with label reversal **only in the training partition** to reduce presentation-side dependence.
- Logistic regression (3-way), report multiclass log loss on untouched validation.
- Refit on the full labeled dataset after validation, then export three calibrated-ish probabilities (calibration itself not yet validated) for Kaggle inference.
- Synthetic tests in public CI; actual competition evaluation pending dataset access and Kaggle execution.

## Next experiments (not yet implemented)
1. Stronger model: transformer sequence classifier; LoRA/QLoRA if hardware permits. Maintain exactly the same validation split; compare log loss, per-class precision/recall, calibration error and runtime.
2. Paired order-swap diagnostic on held-out samples. Report mean absolute change after mapping A/B probabilities back to original response identities, plus tie consistency.
3. Length diagnostic: compare bins for relative response length. Treat associations as descriptive because quality, topic and length are confounded.
4. Annotated warmth/style subset: pre-register style annotation instructions and compare inter-rater agreement if human annotation is used. Do not infer warmth simply from which answer wins.
5. Replicate across seeds and report uncertainty; save config, deterministic run information, and aggregate tables.

## Constraints
The competition is notebook-only with **internet disabled** for final evaluation and a **9-hour CPU/GPU limit**. Any model weights needed in a final submission must be made available as attached Kaggle inputs. The starter Notebook runs a small offline baseline and builds `submission.csv` itself.

## Progress
- [x] Create repository and secret storage
- [x] CPU baseline, local tests, offline Kaggle starter Notebook
- [ ] Confirm competition rule acceptance / dataset download in Kaggle
- [ ] Run real data and archive aggregate validation results
- [ ] Conduct GPU-based LLM fine-tuning
- [ ] Run independent preference-bias analyses
