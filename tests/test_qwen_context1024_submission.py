"""Static safety checks for the authorized Qwen 1024 code submission."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_qwen_1024_submission_notebook_is_inference_only():
    nb=json.loads((ROOT/"notebooks"/"qwen_context1024_submission.ipynb").read_text())
    code="\n".join("".join(c.get("source",[])) for c in nb["cells"] if c["cell_type"]=="code")
    run_cell="".join(nb["cells"][-1]["source"])
    assert "max_length=1024" in run_cell
    assert "batch_size=2" in run_cell
    assert "run(args)" in run_cell
    assert "train_and_predict(" not in run_cell
    assert "trainer.train(" not in run_cell
    assert "submission.csv" in run_cell
    # Training code may be embedded only as an offline dependency for render_pair,
    # but it must never be invoked by the submission execution cell.
    assert "from src.qwen_submit_1024 import run" in code

def test_qwen_1024_metadata_is_private_offline_and_uses_prior_adapter():
    m=json.loads((ROOT/"kaggle_qwen_submit"/"kernel-metadata.json").read_text())
    assert m["is_private"]=="true"
    assert m["enable_internet"]=="false"
    assert m["enable_gpu"]=="true"
    assert m["kernel_sources"]==["packkwanlow/llm-preference-qwen05b-lora-pilot"]
    assert m["model_sources"]==["qwen-lm/qwen2.5/transformers/0.5b/1"]
