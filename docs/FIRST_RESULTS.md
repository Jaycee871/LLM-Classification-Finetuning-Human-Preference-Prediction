# First official-data CPU pilot | 2026-09-23

**Actual execution:** [GitHub Actions run 35883696153](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/35883696153) completed successfully. Numbers below are derived from the official Kaggle `train.csv`. The exact aggregate-only output is [`results/official_cpu_pilot_2026-09-23.json`](../results/official_cpu_pilot_2026-09-23.json). No raw records, checkpoints, private credentials, or row-level model outputs were committed.

## Dataset and experiment

- Official available labeled rows: **57,477**. Reproducible stratified training subset: **12,000** rows using seed 42.
- Class counts in pilot: model A winner **4,189**; model B winner **4,103**; tie **3,708**.
- Row-random validation holdout: **1,800** of the 12,000 pilot rows. **This split is not grouped by prompt**, so near-duplicate or repeated prompts may cross the split.
- Three-class log-loss, evaluated on the *same* held-out rows:

| Model | Validation log loss |
| --- | ---: |
| CPU TF-IDF + pair-reversal logistic regression (default C=2) | **1.174414** |
| Uniform 1/3 probability per class | **1.098612** |
| Constant training-class-prior probability per class | **1.097219** |

**Interpretation:** This initial TF-IDF model performed **worse than the trivial constant baselines** under the tested split and settings. Do not submit this as competitive performance. Investigate regularization and probability calibration using an *inner* calibration split while keeping a separate outer evaluation fold, then compare against stronger models. Text fields were truncated to 2,400 characters for the CPU model.

## Descriptive length observation

Among **8,273** pilot rows where the two responses had different character lengths and the judge selected A or B (not tie), the **longer response was selected 61.10%** of the time. The exploratory nonparametric row-bootstrap 95% interval was **60.00% to 62.17%** using 1,000 draws. There were also **82** pairs with equal character length (including ties). This is **not** evidence that adding words *causes* preference; quality, prompt, context, model identity and other factors are not controlled. Row-bootstrap intervals may be too narrow when examples share prompts, and character lengths are not tokenizer lengths.

## Next experiments

1. Improve the CPU reference through inner-fold regularization and probability calibration. Report outer-held-out log loss and compare to both constant baselines.
2. Add prompt-grouped and repeated-seed robustness checks to reduce row-leakage risk.
3. Run the offline Kaggle GPU Qwen LoRA pilot with a legitimate attached base model, no runtime Internet, and enough hidden-test runtime for the competition.
4. Treat warmth/style hypotheses as separate preregistered annotations rather than inferring them from response length.

**Provenance:** `train.csv` SHA-256: `0692154aaf20fc6649090c3f49b6b5dd1e693a765ecb5f0856ae0da5946f5be2`; runner commit `e1800265332720551c237c632d54c64b0f33d101`; scikit-learn `1.9.1`; pandas `2.3.3`.
