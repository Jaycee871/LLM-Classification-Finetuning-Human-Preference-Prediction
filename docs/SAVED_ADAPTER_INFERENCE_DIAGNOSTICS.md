# Saved Qwen 0.5B adapter: inference-only diagnostic run

This experiment follows the completed private Kaggle T4 Qwen LoRA run. That run produced **1.963875** validation log loss, versus **1.066153** for a matched length-only model on the same 1,200 examples. An aggregate-only follow-up also found an A/B swap probability difference of **0.294159**. These observations do not identify their root causes.

## Exact inputs and controls

A separate **private** Kaggle Notebook attaches three sources through its versioned metadata:
- Official competition source `llm-classification-finetuning`, providing the original labeled `train.csv`.
- Official pretrained model source `qwen-lm/qwen2.5/transformers/0.5b/1`.
- Our own **private prior Notebook output**: `packkwanlow/llm-preference-qwen05b-lora-pilot`, containing the previously trained LoRA adapter, saved classification head and earlier aggregate metrics.

This uses Kaggle's documented `kernel_sources` mechanism. It does not download or publish the private adapter through a public GitHub artifact. The new Notebook has Internet **off** and uses a T4 GPU **for inference only**.

## Analytic procedure

The [source implementation](../src/saved_adapter_inference.py) reconstructs the previous Notebook's exact row-stratified validation steps using seed 42, its subsequent stratified 1,200-row subsample, and the identical 384-token preprocessing. It loads the base model with three labels, verifies that the saved LoRA adapter includes a classifier-score weight, then loads the prior adapter with `is_trainable=False` and runs `model.eval()` under `torch.inference_mode()`.

Only aggregate diagnostics leave Kaggle: true and predicted label counts, confusion matrix, per-class negative log likelihood, mean predicted probabilities, 10-bin confidence/accuracy calibration (and ECE), entropy, fraction of rendered sequences exceeding the token limit, raw-character-cap frequencies, and 128-row A/B-swap diagnostics after swapping the label columns back. It also compares the new inference loss and swap statistic with the **existing** saved values to detect a mismatch of adapter or validation subset. **No individual prompts, IDs, per-example probabilities or model weights are uploaded.**

Even if a high fraction exceeds 384 tokens, that alone cannot prove the truncation caused poor predictions. Likewise, a newly initialized head and one epoch of training are *hypotheses to investigate*, not established explanations.

## Execution safeguards

[The one-time diagnostic workflow](../.github/workflows/diagnose-saved-adapter.yml) creates **only the new private diagnostic kernel** `packkwanlow/qwen-saved-adapter-diagnostics` on the intentionally named initial merge commit. It requires the existing authorized `KAGGLE_API_TOKEN` or `KAGGLE_TOKEN`; it never creates a competition submission or retrains Qwen. Future runs are **manual only**, and the workflow refuses to overwrite an existing diagnostic kernel automatically. If Kaggle cannot attach the private prior output, the workflow fails without trying to retrain.

Do not reuse the previously published **8,504-row prompt-grouped** length-only outer reporting fold to tune Qwen. This is a diagnostic on the original pilot validation rows, not an independent prospective generalization experiment.
