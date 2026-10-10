"""Compose shared agency task precedence with fail-closed paired audit readers."""

import json
from pathlib import Path

import pytest

from npa.workbench import vlm_eval


@pytest.mark.parametrize("task", ["operator task", " "])
def test_shared_explicit_task_precedence_skips_strict_reader(tmp_path, task):
    def refuse(_path):
        pytest.fail("explicit task must not inspect metadata")

    assert vlm_eval._explicit_task_text(task) == task
    assert vlm_eval._resolve_task_text(tmp_path, task, metadata_reader=refuse) == task


@pytest.mark.parametrize("task", ["", "sim-to-real"])
def test_default_task_uses_strict_reader_bytes_not_disk_fallback(tmp_path, task):
    metadata = tmp_path / "manifest.json"
    metadata.write_text(json.dumps({"task": "untrusted fallback"}))
    visited: list[Path] = []

    def reader(path):
        visited.append(path)
        return json.dumps({"task": "strict retained task"}).encode()

    assert vlm_eval._explicit_task_text(task) is None
    assert (
        vlm_eval._resolve_task_text(tmp_path, task, metadata_reader=reader)
        == "strict retained task"
    )
    assert visited == [metadata]


@pytest.mark.parametrize("relative", ["meta/tasks.parquet", "manifest.json"])
@pytest.mark.parametrize("symlink", [False, True])
def test_strict_reader_refusal_precedes_permissive_fallback(
    tmp_path, relative, symlink
):
    metadata = tmp_path / relative
    metadata.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        metadata.symlink_to(tmp_path / "absent-private-metadata")
    else:
        metadata.write_bytes(b"malformed retained metadata")
    if relative != "manifest.json":
        (tmp_path / "manifest.json").write_text(json.dumps({"task": "fallback"}))
    visited: list[Path] = []

    def refuse(path):
        visited.append(path)
        raise ValueError("invalid_task_metadata")

    with pytest.raises(ValueError, match="invalid_task_metadata"):
        vlm_eval._resolve_task_text(tmp_path, "sim-to-real", metadata_reader=refuse)
    assert visited == [metadata]
