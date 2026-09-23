# Saved Qwen2.5-0.5B LoRA adapter: validation-only forensic audit

**Date:** 2026-09-24 (Taipei). **Actual Kaggle inference completed.** [Successful GitHub Actions run 35904556787](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/35904556787) published a **separate private Kaggle diagnostic Notebook version 1**, attached our earlier **private Qwen pilot version 2** as a kernel input, and reloaded its saved LoRA adapter and classification head. No additional training or competition submission occurred. The [entire aggregate JSON](../results/qwen_saved_adapter_inference_2026-09-24.json) is permanently archived in the repo; no raw prompts, row-level predictions, model weights, or private keys were published.

## Reproduction of the original saved model

The original GPU pilot's exact two seed-42 stratified validation selections were reconstructed from the official competition `train.csv`. Restoring the previous adapter's saved score-head weights was verified by finding **one score-weight tensor** in the adapter file. The model was evaluated using `model.eval()` and `torch.inference_mode()` at the original **384-token** cap. This is the *previous model*, not a freshly initialized or retrained classifier.

| Same 1,200-row validation subset | Multiclass log loss |
| --- | ---: |
| Inference-only restored Qwen adapter | **1.964393** |
| Original Qwen pilot reported | **1.963875** |
| Original, identical-holdout length-only reference | **1.066153** |
| Uniform three-class predictor | **1.098612** |

The restored adapter's log loss differs from the original reported value by only **0.000519**, consistent with a faithful reconstruction of the same model and validation split under inference differences. This is not an independent generalization result. The separately observed public Kaggle **1.06530** score came from the CPU length-only competition entry on a **different evaluation population**.

## Three specific diagnostic observations

**1. Confidence is much higher than realized accuracy.** Overall three-class accuracy is **33.08%**. Mean maximum predicted probability is **66.98%**. Among **286/1,200** validation examples with predicted confidence between 90% and 100%, mean confidence was **96.14%**, whereas observed accuracy was **34.27%**. Ten-bin expected calibration error (ECE) was **0.3390**. The model is severely miscalibrated on *this* validation set. A majority-class benchmark from the true counts is 419/1,200 = **34.92%** accuracy; accuracy and log loss answer different questions. There is **no complete single-class collapse**: predicted counts are A **469**, B **390**, and tie **341**, against true counts A **419**, B **410**, tie **371**.

| True outcome | Count | Per-class negative log likelihood | Recall |
| --- | ---: | ---: | ---: |
| A preferred | 419 | 1.74055 | 41.05% |
| B preferred | 410 | 2.04337 | 28.78% |
| Tie | 371 | 2.12992 | 28.84% |

**2. The model is highly sensitive to A/B order.** On the **first 128 held-out examples**, swap-response analysis *re-aligned* the labels after reversing A and B: mean absolute class-probability disagreement was **0.29499**; at least one class probability changed by over 0.2 in **71.09%** of pairs; and the predicted winning class changed in **66.41%**. The original pilot recorded **0.29416**, a difference of **0.00083** from the restored adapter test. These are **model-level** observations, not evidence that human judges have answer-order bias. The model had originally been trained with A/B-swap augmentation; that alone did not ensure swap-invariant predictions.

**3. Input truncation is common.** The existing template concatenates prompt, response A, response B and instruction text, then restricts the result to **384 tokens**. After the model's *earlier character caps*, the rendered text exceeded 384 tokens for **65.5%** of validation examples. Its median length was **554.5 tokens**, and 90th percentile **1,017.1 tokens**. Even before tokenization, **6%** of raw prompts exceeded their 1,200-character cap; raw response A and B exceeded 2,400 characters in **11.83%** and **11.75%** of rows, respectively. Since A appears before B in the prompt template, token truncation could disproportionately remove later content, but **that mechanism has not been experimentally isolated**.

## Scientific interpretation and next controlled experiment

These findings identify *observable failure modes*, not proven causes. The presence of saved score weights shows that the classification head was restored; it does not establish the head converged during the original one-epoch pilot. Overconfidence, severe order sensitivity and frequent truncation are compatible with multiple interacting explanations.

A sensible next controlled experiment is **inference only on the same saved adapter**, changing the maximum context from 384 tokens to **768** and possibly **1,024**. Hold fixed the model weights, same seed-42 1,200-row pilot validation subset, evaluation code and prompt template; record log loss, calibration and A/B-swap consistency for each context limit. Also consider a **balanced prompt template** with explicitly bounded A/B token budgets (e.g. equal response allocations). This is hypothesis testing on a previously observed validation set; improvements here are *exploratory*, not independent evidence or a Kaggle leaderboard score. A final scientific claim requires a newly isolated, prompt-grouped evaluation design and stronger repeated-run evidence.

This result does **not** justify spending more training GPU hours or submitting the existing Qwen predictions to the competition without controlled follow-up.
