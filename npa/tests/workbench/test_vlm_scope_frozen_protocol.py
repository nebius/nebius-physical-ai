"""Keep historical scope controls separate from the evolving default sample."""

import hashlib
import json
from pathlib import Path
import runpy

from npa.workbench import vlm_eval


def _live_lane():
    path = Path(__file__).parents[1] / "e2e/test_vlm_benchmark_scope_live_e2e.py"
    return runpy.run_path(str(path))


def test_frozen_scope_protocol_retains_exact_original_manifest(tmp_path) -> None:
    lane = _live_lane()
    path = lane["_frozen_protocol"](tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "25dee38f69e05f6b1b236b8a2a9c598aa219fdc54a1525eb65a8abeac5f31595"
    )
    dataset = vlm_eval.load_benchmark_dataset(str(path))
    assert [item.id for item in dataset.items] == [
        "place-block-clear-pass",
        "align-tool-pass",
        "missed-target-fail",
        "unstable-end-state-fail",
    ]
    assert [item.expected_label for item in dataset.items] == [True, True, False, False]
    assert dataset.evidence_scope == "illustrative_only"
    assert len(dataset.limitations) == 3
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["thresholds"] == [0.8]
    assert protocol["use_fixture_scores"] is False
    assert "not the original request payload" in protocol["request_scope"]


def test_current_sample_keeps_new_terminal_case_and_updated_scope() -> None:
    current = vlm_eval.load_benchmark_dataset(str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH))
    assert len(current.items) == 5
    current_items = {item.id: item for item in current.items}
    assert current_items["progress-without-terminal-fail"].expected_label is False
    assert current.rubrics["default"] == vlm_eval.DEFAULT_RUBRIC
    assert current.evidence_scope == "illustrative_only"
    assert "truncated-progress sequence" in current.limitations[0]
    assert "additional omitted-terminal label" in current.limitations[1]
    assert "not task-validation" in current.limitations[2]
    frozen = vlm_eval.load_benchmark_dataset(str(_live_lane()["FROZEN_SCOPE_PROTOCOL_PATH"]))
    assert current.rubrics["default"] != frozen.rubrics["default"]
    assert [current_items[item.id] for item in frozen.items] == frozen.items
