"""The committed eval notebooks: valid, output-free, machine-independent and logic-free.

Outputs are cleared because they can embed nuScenes or RDD2022 images, which must never reach git.
Logic belongs in eval/ and tools/ (where it is tested); a notebook only calls it and displays it.
"""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0[34]_eval_*.ipynb"))
FORBIDDEN_TEXT = ("/Users/", "/home/", "joshu", "image/png", "image/jpeg")
DRIVE_PATH = re.compile(r"[A-Za-z]:[\\/]")  # any Windows drive path, e.g. C:\ or D:/


def test_both_eval_notebooks_exist() -> None:
    assert [p.name for p in NOTEBOOKS] == ["03_eval_classifier.ipynb", "04_eval_detector.ipynb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_nbformat(path: Path) -> None:
    nbformat = pytest.importorskip(
        "nbformat"
    )  # the `notebook` extra; the checks below need no extra
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs_or_execution_counts(path: Path) -> None:
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "clear outputs before committing"
            assert cell["execution_count"] is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_machine_specific_text(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # Decoded cell sources, so JSON escaping (a colon then \" in a string) cannot look like a path.
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    text += json.dumps(notebook["metadata"])
    assert not [bad for bad in FORBIDDEN_TEXT if bad in text]
    assert not DRIVE_PATH.search(text)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse_and_define_no_functions_or_classes(path: Path) -> None:
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        tree = ast.parse(source if isinstance(source, str) else "".join(source))
        defined = [
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.ClassDef | ast.Lambda)
        ]
        assert not defined, "move logic into eval/ or tools/ and call it from the notebook"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_not_git_ignored(path: Path) -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs git and a checkout")
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path.relative_to(ROOT))], cwd=ROOT, check=False
    )
    assert result.returncode == 1, "notebooks/0[34]_eval_*.ipynb must be tracked"
