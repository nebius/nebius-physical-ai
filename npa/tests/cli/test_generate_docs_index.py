from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts" / "_generate_docs_index.py"


def _load_docs_index_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_generate_docs_index", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_title_accepts_indented_markdown_h1(tmp_path: Path) -> None:
    page = tmp_path / "fallback-name.md"
    page.write_text("  # npa demo\n\ncontent\n")

    module = _load_docs_index_module()

    assert module._title(page) == "npa demo"
