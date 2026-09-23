"""Validate checked-in notebooks without executing costly competition runs."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "kaggle_starter.ipynb",
    ROOT / "notebooks" / "kaggle_gpu_lora_pilot.ipynb",
    ROOT / "notebooks" / "kaggle_length_submission.ipynb",
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


def test_gpu_notebook_embeds_current_and_valid_python_modules():
    # Generated cell must match src/ on every CI run; parse actual literals too.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from sync_gpu_notebook import generated_source_cell

    nb = json.loads(NOTEBOOKS[1].read_text(encoding="utf-8"))
    cell = nb["cells"][2]["source"]
    assert cell == generated_source_cell(), (
        "Run python scripts/sync_gpu_notebook.py after editing src/"
    )
    tree = ast.parse("".join(cell))
    captured = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "write_text":
            continue
        location = ast.unparse(node.func.value)
        for filename in ("baseline.py", "finetune_lora.py"):
            if filename in location:
                captured[filename] = ast.literal_eval(node.args[0])

    assert set(captured) == {"baseline.py", "finetune_lora.py"}
    for filename, embedded in captured.items():
        assert embedded == (ROOT / "src" / filename).read_text(encoding="utf-8")
        ast.parse(embedded, filename=filename)
    assert "KAGGLE_API_TOKEN" not in "".join(cell)


def test_length_notebook_source_matches_authoritative_modules():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from sync_length_notebook import generated_source_cell

    nb = json.loads(NOTEBOOKS[2].read_text(encoding="utf-8"))
    code = nb["cells"][2]["source"]
    assert code == generated_source_cell(), (
        "Run python scripts/sync_length_notebook.py after editing src/"
    )
    tree = ast.parse("".join(code))
    captured = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "write_text":
            continue
        path = ast.unparse(node.func.value)
        for filename in ("baseline.py", "length_baseline.py"):
            if filename in path:
                captured[filename] = ast.literal_eval(node.args[0])
    assert set(captured) == {"baseline.py", "length_baseline.py"}
    for filename, source in captured.items():
        assert source == (ROOT / "src" / filename).read_text(encoding="utf-8")
        ast.parse(source, filename=filename)
