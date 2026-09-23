# Experiment plan (v0.2, 2026-09-23)

## Primary competition task
Predict which response wins for a given prompt with three class probabilities, A, B, or tie. Use multiclass log loss and the exact schema in the official competition rules.

## Stage A: Implemented CPU baseline
- Stratified, fixed-seed 85/15 split with train-only TF-IDF fitting and train-only pair-order reversal. **Identical prompts could appear across splits**; before drawing strong generalization conclusions, add a grouped-by-prompt robustness split.
- Logistic-regression probabilities and `submission.csv` creation.
- Offline, self-contained Kaggle CPU Notebook; synthetic tests run in GitHub Actions.
- **Status:** synthetic CI passed; Kaggle real-data run pending.

## Stage B: Implemented preference diagnostics (execution pending)
- `src/diagnostics.py` calculates character-length strata and longer-answer choice rate, with 1,000 bootstrap replicates.
- Optional `swap_audit` compares model predictions under response-order inversion; run on an untouched validation partition for interpretable held-out results.
- These measures do not establish what caused preferences or measure subjective warmth.

## Stage C: Optional GPU LoRA pilot (execution pending)
- `src/finetune_lora.py`: locally attached Qwen2.5-0.5B base, LoRA sequence classification, 4,000 train-only pilot rows, three-class log loss, bounded held-out swap probe.
- Self-contained offline notebook `notebooks/kaggle_gpu_lora_pilot.ipynb` with source bundled in cells. Requires separately attached base weights, available packages, and GPU.
- Save adapter and aggregate metrics; do not commit weights or raw data. See `docs/GPU_PILOT.md`.

## Stage D: Independent robustness (planned)
1. Add prompt-grouped split and multi-seed reruns; compare with the original random split to assess near-duplicate leakage.
2. Run full training only after validating pilot runtime on genuine Kaggle hardware, accounting for the hidden ~25K test rows.
3. Independently annotate response warmth with a preregistered codebook and agreement checks. Do not equate response length or preference with warmth.
4. Provide bias, calibration and subgroup analyses with clear limitations; maintain provenance for code, random seeds and data version.

## Milestones
- [x] Repository initialized; synthetic CPU test pipeline passes.
- [x] Aggregate preference-length and model-order diagnostics implemented.
- [x] Optional LoRA GPU training and self-contained offline Notebook added.
- [ ] Verify user's Kaggle Secret via the **manual** GitHub workflow (not run automatically).
- [ ] Obtain official competition data and report actual validation scores.
- [ ] Run Kaggle GPU pilot successfully and record model/runtime metrics.
- [ ] Complete grouped prompt validation and independent style annotation.
