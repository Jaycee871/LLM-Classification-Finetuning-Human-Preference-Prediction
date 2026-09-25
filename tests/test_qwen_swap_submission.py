"""Static safety checks for the authorized Qwen swap-ensemble submission."""
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/"notebooks"/"qwen_swap_ensemble_submission.ipynb"

def test_swap_submission_notebook_sync_and_execution_cell():
    sys.path.insert(0,str(ROOT/"scripts"))
    from sync_qwen_swap_submit_notebook import generated_source_cell
    nb=json.loads(NB.read_text())
    assert nb["cells"][2]["source"]==generated_source_cell()
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code":
            ast.parse("".join(c["source"]),filename=f"cell-{i}")
    run="".join(nb["cells"][3]["source"])
    assert "max_length=1024" in run
    assert "batch_size=2" in run
    assert "run(args)" in run
    assert "train_and_predict(" not in run

def test_swap_submission_metadata_is_private_offline():
    m=json.loads((ROOT/"kaggle_qwen_swap_submit"/"kernel-metadata.json").read_text())
    assert m["is_private"]=="true"
    assert m["enable_internet"]=="false"
    assert m["enable_gpu"]=="true"
    assert m["kernel_sources"]==["packkwanlow/llm-preference-qwen05b-lora-pilot"]
