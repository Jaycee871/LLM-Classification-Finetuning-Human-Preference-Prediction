# Qwen saved-adapter context-length ablation

This experiment changes **only inference context length** for the already trained private Qwen2.5-0.5B LoRA adapter. It reuses the exact original seed-42 1,200-row pilot validation subset and compares **384, 768 and 1,024 tokens**. The first 128 held-out rows are also reused for aligned A/B swap tests at every context limit.

The motivation is the completed forensic audit: **65.5%** of rendered validation inputs exceeded 384 tokens, the restored adapter had log loss **1.964393**, accuracy **33.08%**, ECE **0.3390**, and **66.41%** of 128 swap probes changed predicted winner. Those observations identify failure modes but do not establish that truncation caused them.

Each context condition reports log loss, accuracy, ECE, class counts, confidence, entropy, truncation rate, A/B swap instability and wall-clock inference time. The saved model weights do not change. The adapter itself was originally trained at 384 tokens, so longer-context inference is an **exploratory ablation** and cannot substitute for retraining or independent validation.

The private Kaggle Notebook attaches the original private adapter kernel as a `kernel_source`, the official Qwen 0.5B base as a `model_source`, and official competition data. Internet is off. No row-level outputs, model files or competition submissions are published. The first successful merge intentionally launches the private run once; future reruns are manual.
