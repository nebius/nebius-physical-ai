"""Keep ordinary task discovery tolerant without weakening strict audit readers."""

import json

import pytest

from npa.workbench import vlm_eval


@pytest.mark.parametrize("relative", ["meta/info.json", "info.json", "manifest.json"])
@pytest.mark.parametrize("payload", [[], None, 42, True, "not an object"])
def test_nonobject_task_metadata_uses_next_candidate(tmp_path, relative, payload):
    metadata = tmp_path / relative
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(payload))
    expected = "sim-to-real"
    if relative != "manifest.json":
        expected = "complete the fallback task"
        (tmp_path / "manifest.json").write_text(json.dumps({"task": expected}))

    assert vlm_eval._resolve_task_text(tmp_path, "sim-to-real") == expected


def test_explicit_task_does_not_read_discovery_metadata(monkeypatch, tmp_path):
    def unexpected(_path):
        pytest.fail("explicit task reached metadata discovery")

    assert (
        vlm_eval._resolve_task_text(
            tmp_path, "explicit task", metadata_reader=unexpected
        )
        == "explicit task"
    )


def test_strict_metadata_reader_error_does_not_become_fallback(tmp_path):
    (tmp_path / "manifest.json").write_text("[]")

    def strict_reader(_path):
        raise ValueError("invalid_task_metadata")

    with pytest.raises(ValueError, match="invalid_task_metadata"):
        vlm_eval._resolve_task_text(
            tmp_path, "sim-to-real", metadata_reader=strict_reader
        )
