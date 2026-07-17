"""Keep the human tool catalog in sync with TOOL_CATALOG."""

from __future__ import annotations

import re
from pathlib import Path

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

ROOT = Path(__file__).resolve().parents[4]
DOC_PATH = ROOT / "docs" / "workbench" / "npa-workflow-tool-catalog.md"


def _doc_tool_refs() -> set[str]:
    text = DOC_PATH.read_text(encoding="utf-8")
    # Match backtick-wrapped toolRefs, including train_* / eval_* wildcards.
    refs = set(re.findall(r"`([a-z0-9_.]+(?:_\*)?)`", text))
    return {ref for ref in refs if "." in ref}


def _catalog_keys_for_doc() -> set[str]:
    keys = set(TOOL_CATALOG)
    # Doc may use wildcard rows for detection_training train/eval variants.
    train_keys = {k for k in keys if k.startswith("workbench.detection_training.train_")}
    eval_keys = {k for k in keys if k.startswith("workbench.detection_training.eval_")}
    if train_keys:
        keys -= train_keys
        keys.add("workbench.detection_training.train_*")
    if eval_keys:
        keys -= eval_keys
        keys.add("workbench.detection_training.eval_*")
    return keys


def test_catalog_doc_lists_every_tool_ref() -> None:
    assert DOC_PATH.is_file()
    documented = _doc_tool_refs()
    expected = _catalog_keys_for_doc()
    missing = sorted(expected - documented)
    assert not missing, f"toolRefs missing from {DOC_PATH.name}: {missing}"


def test_stub_flags_match_echo_stubs() -> None:
    for name, entry in TOOL_CATALOG.items():
        uses_echo = entry.argv_template and entry.argv_template[0] == "echo"
        if uses_echo:
            assert entry.stub, f"{name} uses echo but stub=False"
        if entry.stub:
            assert "stub" in entry.description.lower() or uses_echo, (
                f"{name} marked stub without stub language in description"
            )


def test_byof_toolref_uses_cli() -> None:
    argv = TOOL_CATALOG["workbench.byof.repo"].argv_template
    assert argv[:4] == ["npa", "workbench", "byof", "run"]
