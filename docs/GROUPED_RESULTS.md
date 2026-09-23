# Independent exact-prompt grouped validation | Completed 2026-09-24

**Status:** Successfully executed on the **official 57,477-row training CSV**. [GitHub Actions run 35901558177](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/35901558177). Aggregate-only metrics are archived at [`results/grouped_length_2026-09-24.json`](../results/grouped_length_2026-09-24.json).

A new 15% outer holdout was created **by exact normalized prompt hash**, not by individual row. Among 57,477 rows there were **51,731 distinct exact prompt hashes**. The resulting outer train/test sets contained **48,973 and 8,504 rows**, with **zero exact-prompt overlap**. Only the inner training-partition groups were used to choose regularization; `C=10` was selected.

| Model on the new grouped outer holdout | Multiclass log loss |
| --- | ---: |
| Length-only logistic regression | **1.065217** |
| Constant training-prior baseline | 1.097199 |
| Uniform baseline | 1.098612 |

The row-bootstrap 95% interval for the *length-only minus training-prior per-example log loss* was **[-0.03802, -0.02550]**. Under this sampled grouped outer split, the length-only model had lower observed loss than either simple constant baseline.

These numbers are on a **different population** than both the original 1,800-row local pilot (length-only 1.057783) and the previous Kaggle public leaderboard score **1.06530**. They should not be interpreted as new leaderboard measurements or pooled indiscriminately. Because this run groups only *exact* prompts and uses one outer split, paraphrased prompts may still cross partitions. The bootstrap interval resamples rows and may underestimate variability when several records share a prompt. The result does not establish that answer length causes preference or that any AI style is objectively superior.

The independent grouped check strengthens the evidence that simple length features carry **some predictive association** in these data. It does not establish the general superiority of length features over language models: our separate Qwen pilot had only 2,000 original training pairs, one epoch, a newly initialized classification head and 384-token truncation. See [first actual Qwen pilot results](FIRST_GPU_RESULTS.md) and the executable [grouped robustness protocol](GROUPED_ROBUSTNESS.md).

**Next priority:** diagnose Qwen's class probability distribution, head convergence and truncation before funding another large GPU run; ensure new experiments include distinct calibration and reporting partitions and avoid repeatedly optimizing against this grouped test fold.
