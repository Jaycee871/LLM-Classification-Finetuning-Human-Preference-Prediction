# First Qwen 0.5B LoRA Kaggle GPU pilot

## Aim and prior benchmark
Our first, completed **length-only** competition submission obtained **1.06530** on the then-visible public leaderboard (screenshot from 2026-09-24, rank 157 at that moment). Its 12K-row exploratory local validation loss was 1.057783. Those two figures are **different populations**; neither is the same evaluation split as the new GPU experiment.

The GPU pilot fits Qwen2.5-0.5B **base** (QwenLM, Apache 2.0) as a three-way sequence classifier with rank-8 LoRA adapters. A length-only C=10 model fits **the exact same 2,000 original training examples**; both are evaluated on the **same separate, stratified 1,200-row validation subset**. We train one epoch on 2,000 original A/B pairs and their 2,000 label-inverted copies, with max 384 tokens per example. The main GPU output is a **matched-split comparison**, not an independently confirmed improvement over the public leaderboard.

## Model and runtime
- Official [QwenLM Qwen2.5 0.5B base, Kaggle Transformers version 1](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/0.5b/1) through `model_sources`, **not** an unverifiable GitHub-downloaded weights file.
- [Competition](https://www.kaggle.com/competitions/llm-classification-finetuning): offline code submission, 9-hour GPU limit. The notebook's `competition_sources` metadata attaches official data; our first CPU submission demonstrated that **input mount paths should be discovered at runtime**.
- GPU: NvidiaTeslaT4 accelerator, Internet disabled, private Notebook. The notebook finds mounted `train.csv`/`test.csv` and Qwen model `config.json` under `/kaggle/input`. Missing packages or model source raise an explicit error; never embed credentials or download weights in a scoring Notebook.
- These packages must exist in the Kaggle image: `torch`, `transformers`, `peft`, `accelerate`, `scipy`, `sklearn`, `pandas`, `joblib`. If a dependency is absent, capture the error and attach an offline wheel dataset before retrying. Do not silently enable Notebook Internet.
- Prior test data `test.csv` are only a public preview; any later **competition code submission** will rerun with a substantially larger hidden test, so inference must remain within the competition's runtime limit.

## Automated run
The [GPU Kaggle pilot GitHub workflow](../.github/workflows/launch-gpu-pilot.yml) uses the existing `KAGGLE_API_TOKEN` (or `KAGGLE_TOKEN`) only to **push and monitor** a separate Kaggle private kernel, `packkwanlow/llm-preference-qwen05b-lora-pilot`. The workflow is triggered when first merged to main, or deliberately via manual dispatch. It records the actual Kaggle kernel version from the upload response, waits for its reported `KernelWorkerStatus.COMPLETE`, and retrieves **only** the aggregate `gpu_pilot_metrics.json`; saved adapters, competition data and generated `submission.csv` remain inside private Kaggle outputs and are **not** copied to the public GitHub repository. It does **not** send a new competition entry until metrics and a hidden-test runtime plan have been reviewed.

**Contingencies:** if Kaggle rejects an accelerator, the model input identifier, or a missing library, the workflow fails and prints a small bounded Kaggle diagnostic. For GPU pilot metrics, use Kaggle versioned run results. The synthetic GitHub CI suite tests data labeling, notebook source synchronization, and fake-GPU control flow but cannot prove runtime compatibility with actual Kaggle hardware.

## Why we keep the results separate
Human preference labels reflect observed comparisons, not absolute quality or causality. The original data can share prompts across row-random splits; a future publication requires prompt-grouped re-evaluation and independent multi-seed testing. Do not infer warmth or subjective states from response lengths.
