"""Static checks for private Qwen 1024 swap-ensemble audit Notebook."""
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks"/"qwen_swap_ensemble_1024.ipynb"

def test_swap_ensemble_notebook_is_synced_and_inference_only():
    sys.path.insert(0,str(ROOT/"scripts"))
    from sync_qwen_swap_ensemble_notebook import generated_source_cell
    nb=json.loads(NOTEBOOK.read_text())
    assert nb["cells"][2]["source"]==generated_source_cell()
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code":
            ast.parse("".join(c["source"]),filename=f"cell-{i}")
    run="".join(nb["cells"][3]["source"])
    assert "max_length=1024" in run
    assert "max_validation_rows=1200" in run
    assert "run(args)" in run
    assert "train_and_predict(" not in run
    assert "kaggle competitions submit" not in run

def test_swap_ensemble_metadata_is_private_offline_and_uses_prior_adapter():
    m=json.loads((ROOT/"kaggle_swap_ensemble"/"kernel-metadata.json").read_text())
    assert m["is_private"]=="true"
    assert m["enable_internet"]=="false"
    assert m["enable_gpu"]=="true"
    assert m["kernel_sources"]==["packkwanlow/llm-preference-qwen05b-lora-pilot"]
    assert m["model_sources"]==["qwen-lm/qwen2.5/transformers/0.5b/1"]
