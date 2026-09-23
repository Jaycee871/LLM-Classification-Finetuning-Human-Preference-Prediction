"""Ensure the private inference-only Kaggle Notebook uses current source modules."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "diagnose_saved_qwen_adapter.ipynb"
MODULES = ("baseline.py", "finetune_lora.py", "length_baseline.py", "saved_adapter_inference.py")


def test_notebook_cells_compile_and_modules_are_current():
    sys.path.insert(0, str(ROOT / "scripts"))
    from sync_saved_adapter_notebook import generated_source_cell

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 4
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{idx}")
    embedded = nb["cells"][2]["source"]
    assert embedded == generated_source_cell(), (
        "Run python scripts/sync_saved_adapter_notebook.py after editing src/"
    )
    captured = {}
    for node in ast.walk(ast.parse("".join(embedded))):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "write_text":
            continue
        value = ast.unparse(node.func.value)
        for filename in MODULES:
            if f"'{filename}'" in value:
                captured[filename] = ast.literal_eval(node.args[0])
    assert set(captured) == set(MODULES)
    for filename, embedded_source in captured.items():
        expected = (ROOT / "src" / filename).read_text(encoding="utf-8")
        assert embedded_source == expected
        ast.parse(embedded_source, filename=filename)


def test_preflight_uninstalls_only_old_optional_torchao_before_import():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join(nb["cells"][1]["source"])
    assert "importlib.metadata.version('torchao')" in code
    assert "'-m', 'pip', 'uninstall', '-y', 'torchao'" in code
    assert code.index("subprocess.run(") < code.index("import torch, transformers, peft")
    assert "adapter_model.safetensors" in code
    assert "llm-preference-qwen05b-lora-pilot" in code
    assert "kernel_sources" in code
    run_cell = "".join(nb["cells"][3]["source"])
    assert "main(params)" in run_cell
    assert "seed=42" in run_cell
    assert "max_validation_rows=1200" in run_cell
    assert "max_length=384" in run_cell
    assert "train_and_predict(" not in run_cell
    assert "kaggle competitions submit" not in run_cell


def test_private_notebook_has_official_model_and_prior_kernel_sources():
    metadata = json.loads(
        (ROOT / "kaggle_diagnostics" / "kernel-metadata.json").read_text()
    )
    assert metadata["is_private"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["enable_gpu"] == "true"
    assert metadata["competition_sources"] == ["llm-classification-finetuning"]
    assert metadata["kernel_sources"] == [
        "packkwanlow/llm-preference-qwen05b-lora-pilot"
    ]
    assert metadata["model_sources"] == [
        "qwen-lm/qwen2.5/transformers/0.5b/1"
    ]
