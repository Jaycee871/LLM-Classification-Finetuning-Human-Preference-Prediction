"""Validate private context-ablation Notebook and Kaggle metadata."""
import ast, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks"/"qwen_context_length_ablation.ipynb"


def test_context_ablation_notebook_sync_and_no_training_calls():
    sys.path.insert(0,str(ROOT/"scripts"))
    from sync_context_ablation_notebook import generated_source_cell
    nb=json.loads(NOTEBOOK.read_text())
    assert nb["cells"][2]["source"]==generated_source_cell()
    for i,cell in enumerate(nb["cells"]):
        if cell["cell_type"]=="code":
            ast.parse("".join(cell["source"]),filename=f"cell-{i}")
    run="".join(nb["cells"][3]["source"])
    assert "contexts=[384, 768, 1024]" in run
    assert "run_ablation(params)" in run
    assert "train_and_predict" not in run
    assert "kaggle competitions submit" not in run


def test_context_ablation_kaggle_sources_are_private_and_versioned():
    meta=json.loads((ROOT/"kaggle_context_ablation"/"kernel-metadata.json").read_text())
    assert meta["is_private"]=="true"
    assert meta["enable_internet"]=="false"
    assert meta["enable_gpu"]=="true"
    assert meta["kernel_sources"]==["packkwanlow/llm-preference-qwen05b-lora-pilot"]
    assert meta["model_sources"]==["qwen-lm/qwen2.5/transformers/0.5b/1"]
