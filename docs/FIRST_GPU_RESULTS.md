# Qwen2.5-0.5B first completed Kaggle GPU pilot

**Date (Taipei): 2026-09-24.** Source: [successful GitHub Actions run 35898823108](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/35898823108), private Kaggle notebook `packkwanlow/llm-preference-qwen05b-lora-pilot` **version 2**. Aggregate metrics are archived in [`results/qwen05b_gpu_pilot_v2_2026-09-24.json`](../results/qwen05b_gpu_pilot_v2_2026-09-24.json).

| Same-split pilot (1,200 held-out rows) | Multiclass log loss |
| --- | ---: |
| Qwen2.5-0.5B base, newly created 3-class head + LoRA | **1.963875** |
| Length-only logistic regression, C=10, matched training rows | **1.066153** |

Lower multiclass log loss is better. **The LoRA pilot was substantially worse than the simple length model on the same validation rows.** These results do not explain *why* without further diagnostics. Reasonable hypotheses to test include a newly initialized head, inadequate training exposure, suboptimal learning rate, and context truncation, without assuming any one is the cause.

The experiment used **2,000 original training pairs**, 2,000 A/B-inverted training augmentations, a separate stratified 1,200-row held-out validation subset, **one epoch**, and a 384-token input cap. It produced a **three-row preview `submission.csv`**, but **no second competition entry** was submitted. Do not confuse the **1.066153 local matched-split validation loss** with the earlier **1.06530 public leaderboard** score of the length model on a different population.

The first GPU version failed **before training** because Kaggle bundled `torchao 0.10.0`, incompatible with the installed PEFT version. The fixed **version 2** removed the unused optional package before imports, trained and completed successfully. Preserve both attempts in the methods audit.

## Next research decision

**Do not submit the current LoRA output to Kaggle** based on these findings. First test whether the length model generalizes under a *prompt-grouped* outer holdout (rather than our previously reused row-random split), and measure class-prior baselines and class distribution. Only consider another GPU run after defining dedicated tuning and evaluation subsets and diagnostic metrics such as predicted-class proportions, prediction entropy, class-wise losses and calibration. A future GPU experiment must not tune on the held-out reporting split and should include a matched same-training-size length baseline.

## Provenance and scope

- Successful run ID: `35898823108`
- Kaggle kernel: `packkwanlow/llm-preference-qwen05b-lora-pilot`, version 2
- Synthetic GitHub tests passed separately. Real Kaggle execution and aggregate JSON upload also succeeded.
- Notebook version 2 is **private**, and the three-row preview output is not a hidden-test score.
- The earlier CPU leaderboard entry remains the **only confirmed scored competition submission** in this project.
