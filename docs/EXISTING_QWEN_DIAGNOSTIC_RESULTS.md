# Aggregate-only Qwen GPU diagnosis | Completed 2026-09-24

[Successful zero-GPU diagnostic workflow](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/35903590417) reloaded **only the existing private Kaggle kernel v2 aggregate JSON**. No new GPU training, no raw competition file download and no further competition submission occurred.

On **1,200 identical held-out rows**, Qwen LoRA had multiclass log loss **1.963875**, the matched length-only model **1.066153**, and a uniform three-class reference **1.098612**. The LoRA loss exceeded the matched length reference by **0.897722** and even the uniform reference by **0.865262**.

The previously recorded **A/B swap probe mean absolute probability difference** was **0.294159** on up to 128 held-out examples after aligning winner A and B columns after swapping the input order. This describes sensitivity to answer order in the particular trained model; it does not establish that the human-judgment data suffer order bias, and it is not a causal explanation of the high loss.

**What we still cannot diagnose from these aggregates:** predicted class distribution, class-wise error, probability confidence or calibration, truncation frequency, head convergence. The correct next experiment should **reload the existing adapter for validation-only inference**, preserve the identical original split and compute aggregate-only statistics. Keep predictions and raw prompts private. In particular, measure how frequently A/B swaps change the winning class or yield large probability disagreements; check label mappings and whether newly initialized classification head is actually trainable. Do not retrain from scratch or make a new Kaggle competition submission as a substitute for these measurements.

The grouped independent length-only check ([completed audit](GROUPED_RESULTS.md)) remains a separate 8,504-row reporting holdout and should not be reused to tune Qwen.
