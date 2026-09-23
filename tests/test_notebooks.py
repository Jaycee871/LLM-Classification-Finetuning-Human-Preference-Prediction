"""Validate checked-in notebooks without executing costly competition runs."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "kaggle_starter.ipynb",
    ROOT / "notebooks" / "kaggle_gpu_lora_pilot.ipynb",
]


def test_notebook_json_and_python_syntax():
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert nb["cells"]
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                ast.parse(code, filename=f"{path.name}:cell_{i}")


def test_gpu_notebook_embeds_modules_offline():
    nb = json.loads(NOTEBOOKS[1].read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "write_text(" in code
    assert "baseline.py" in code and "finetune_lora.py" in code
    assert "KAGGLE_API_TOKEN" not in code
