# Qwen 1,024-token A/B swap ensemble | Completed 2026-09-25

The private Kaggle inference-only audit completed successfully in [GitHub Actions run 36080336884](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/36080336884). It reloaded the existing saved Qwen2.5-0.5B LoRA adapter and evaluated the **same seed-42 1,200-row pilot validation subset** twice per example: original A/B order and swapped B/A order, with swapped output classes mapped back to the original A/B/tie labels.

| Method | Local multiclass log loss | Accuracy | 10-bin ECE |
| --- | ---: | ---: | ---: |
| Original 1,024-token Qwen | 1.320317 | 33.58% | 0.17595 |
| Swapped-aligned Qwen | 1.321815 | 34.42% | 0.17133 |
| **Arithmetic A/B ensemble** | **1.190046** | 34.83% | **0.09199** |
| Geometric A/B ensemble | 1.225452 | 34.92% | 0.10222 |

The original and swapped predictions remained strongly order-sensitive: **72.25%** of rows changed predicted winner and mean absolute class-probability disagreement was **0.16282**. Arithmetic averaging reduced local log loss by **0.13027** relative to the single-pass 1,024-token Qwen.

The previously submitted **single-pass** 1,024-token Qwen received **1.35009** on the Kaggle leaderboard, while the project's best length-only entry remained **1.06530**. The ensemble has not yet been independently validated on hidden data at the time this audit was recorded.

## Interpretation

The improvement supports a narrow engineering conclusion: enforcing prediction symmetry at inference can partially mitigate the model's observed A/B positional instability. It does **not** establish that answer order causes human preference, and it does not make the reused local validation set independent. Because 1,024 tokens and the arithmetic ensemble were both selected after inspecting this same held-out subset, the next hidden Kaggle score is best treated as an **experimental external check** rather than confirmation of a general result.

The competition submission created from this finding must use the same saved adapter, 1,024-token limit, and arithmetic mean of original plus aligned-swapped probabilities, with no retraining.
