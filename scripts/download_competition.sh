#!/usr/bin/env bash
set -euo pipefail
# Use authenticated Kaggle CLI on your own machine.
# Current CLI supports KAGGLE_API_TOKEN, or 'kaggle auth login'.
# Accept competition rules in the browser before downloading.
mkdir -p data
kaggle competitions download llm-classification-finetuning -p data --unzip
echo "Downloaded to data/; source data are gitignored. Do not commit them."
