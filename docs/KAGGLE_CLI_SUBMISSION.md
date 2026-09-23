# Kaggle CLI code competition submission

The competition uses Kaggle-hosted Notebook versions, **not** a locally generated CSV uploaded from GitHub. Kaggle's documented command is:

```bash
kaggle competitions submit llm-classification-finetuning \
  -f submission.csv \
  -k packkwanlow/llm-preference-length-baseline-2026 \
  -v 1 \
  -m "Offline length-only baseline"
```

Here, `-f submission.csv` names the file produced by the **Kaggle Notebook**, `-k` names the Kaggle-hosted Notebook (not the GitHub repository), and `-v` names a **completed version** of that Notebook. Running this command before the Notebook is uploaded and saved/executed will not work.

To avoid manually using Kaggle's Notebook editor, GitHub has an **on-demand** workflow: **Actions > Publish and submit Kaggle notebook > Run workflow**. The default `push_and_submit` mode copies our verified, offline CPU submission Notebook, attaches the official competition dataset through Kaggle metadata, creates a *private* Notebook under the Kaggle account `packkwanlow`, polls until execution is complete and confirms the output filename, then submits its first version using the CLI.

**Before running:** the existing GitHub `KAGGLE_API_TOKEN` or `KAGGLE_TOKEN` must belong to the Kaggle account `packkwanlow` and have permission to join this competition. If the actual Kaggle username is different, change both the kernel metadata `id` and workflow `KERNEL` before executing. The workflow does not print or commit the secret.

The workflow automatically runs **once when this new workflow file is first added to `main`**. Future workflow maintenance edits do not trigger another submission; manual reruns are available via **Actions > Publish and submit Kaggle notebook > Run workflow**. After a successful push, avoid choosing `push_and_submit` again because the Notebook now exists. If Kaggle execution succeeded but the submission failed, choose `submit_existing` and enter that successfully executed Notebook version. Avoid repeated competition submissions unless necessary.

The Kaggle Notebook uses CPU-only character-length features and `C=10`, selected during our earlier exploratory evaluation. It has **not yet been executed or scored by the competition** at the time this workflow is prepared. Our exploratory held-out validation score does not predict the public leaderboard score.

References: [Kaggle CLI code competition instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md); [Kaggle CLI kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

## Actual Kaggle submission result (2026-09-24, Taipei time)

- [Failed first-run logs](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/35887420239): Kaggle Notebook **v1** returned `KernelWorkerStatus.ERROR` because the first cell required a hardcoded `/kaggle/input/llm-classification-finetuning/train.csv` path that was not present. A GitHub shell poller also incorrectly expected `status: ERROR` and consequently waited ~44 minutes after the failure. No competition submission occurred from that run.
- The notebook now discovers the actual competition input root by locating a matched `train.csv` and `test.csv` pair inside Kaggle's input mounts. The existing Notebook was updated to **version 2**.
- [Successful code-submission run](https://github.com/Jaycee871/LLM-Classification-Finetuning-Human-Preference-Prediction/actions/runs/35893885041): v2 status `KernelWorkerStatus.COMPLETE`; its output lists `submission.csv`; `kaggle competitions submit llm-classification-finetuning -f submission.csv -k packkwanlow/llm-preference-length-baseline-2026 -v 2` exited successfully and reported **9 submissions remaining today**.
- The official public/private scoring results remain pending/unverified. Do not describe our previously observed 1.057783 *local* validation loss as the Kaggle score.

**Rerun safeguards:** The original workflow recognizes explicit, specially named push commits or a manual workflow dispatch only. Do not rerun an already submitted Notebook version unnecessarily. The Kaggle CLI `kaggle competitions submit` command takes the competition **positionally** in the installed 2.2.4 CLI (not `-c`).
