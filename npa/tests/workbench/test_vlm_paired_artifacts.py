"""Verify private paired evidence publication independently of provider fixtures."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workbench import vlm_eval


def _payload():
    return {"schema_version": vlm_eval.JUDGE_COMPARISON_SCHEMA_VERSION}


def test_pair_write_is_private_even_under_permissive_umask(tmp_path):
    path = tmp_path / vlm_eval.JUDGE_COMPARISON_RESULT_FILENAME
    previous = os.umask(0)
    try:
        vlm_eval.write_result(_payload(), result_uri=str(path))
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == _payload()


def test_pair_write_resolves_directory_to_canonical_artifact(tmp_path):
    directory = tmp_path / "report"
    result = vlm_eval.write_result(_payload(), result_uri=str(directory))
    assert Path(result) == directory / vlm_eval.JUDGE_COMPARISON_RESULT_FILENAME
    assert json.loads(Path(result).read_text()) == _payload()


@pytest.mark.parametrize("symlink", [False, True])
def test_pair_write_cannot_replace_evidence_or_follow_symlink(tmp_path, symlink):
    path = tmp_path / vlm_eval.JUDGE_COMPARISON_RESULT_FILENAME
    original = tmp_path / "original.json" if symlink else path
    original.write_text("original evidence")
    if symlink:
        path.symlink_to(original)
    with pytest.raises(vlm_eval.VlmEvalError, match="new writable destination"):
        vlm_eval.write_result(_payload(), result_uri=str(path))
    assert original.read_text() == "original evidence"
    assert not list(tmp_path.glob(".vlm-pair-*"))


def test_pair_concurrent_publication_keeps_one_complete_outcome(tmp_path):
    path = tmp_path / vlm_eval.JUDGE_COMPARISON_RESULT_FILENAME

    def publish(index):
        try:
            vlm_eval.write_result(
                {**_payload(), "attempt": index}, result_uri=str(path)
            )
            return index
        except vlm_eval.VlmEvalError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        winners = [
            value for value in executor.map(publish, range(2)) if value is not None
        ]
    assert len(winners) == 1
    assert json.loads(path.read_text())["attempt"] == winners[0]
    assert not list(tmp_path.glob(".vlm-pair-*"))


def test_pair_s3_write_is_conditional_and_never_falls_back():
    client = Mock()
    uri = "s3://example-bucket/run/vlm_judge_disagreement.json"
    assert (
        vlm_eval.write_result(_payload(), result_uri=uri, storage_client=client) == uri
    )
    args, kwargs = client.put_bytes_conditional.call_args
    assert json.loads(args[0]) == _payload()
    assert args[1] == uri
    assert kwargs == {"if_none_match": True, "content_type": "application/json"}
    client.put_bytes_conditional.side_effect = StoragePreconditionFailed("collision")
    with pytest.raises(vlm_eval.VlmEvalError):
        vlm_eval.write_result(_payload(), result_uri=uri, storage_client=client)
    client.upload_file.assert_not_called()


def test_pair_invalid_filename_fails_before_input_or_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vlm_eval, "_materialized_input", lambda *_: pytest.fail("input was prepared")
    )
    with pytest.raises(vlm_eval.VlmEvalError, match="filename"):
        vlm_eval.compare_vlm_judges(
            vlm_eval.VlmJudgeComparisonRequest(
                input_path=str(tmp_path / "frame.png"),
                output_path=str(tmp_path / "wrong.json"),
                primary_model="first/model",
                secondary_model="second/model",
            )
        )


def test_pair_existing_artifact_fails_before_input_or_provider(monkeypatch, tmp_path):
    path = tmp_path / vlm_eval.JUDGE_COMPARISON_RESULT_FILENAME
    path.write_text("prior evidence")
    monkeypatch.setattr(
        vlm_eval, "_materialized_input", lambda *_: pytest.fail("input was prepared")
    )
    with pytest.raises(vlm_eval.VlmEvalError, match="already exists"):
        vlm_eval.compare_vlm_judges(
            vlm_eval.VlmJudgeComparisonRequest(
                input_path="unused.png",
                output_path=str(tmp_path),
                primary_model="first/model",
                secondary_model="second/model",
            )
        )
    assert path.read_text() == "prior evidence"


def test_pair_write_rejects_parent_symlinks(tmp_path):
    directory = tmp_path / "actual"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(vlm_eval.VlmEvalError):
        vlm_eval.write_result(
            _payload(),
            result_uri=str(alias / vlm_eval.JUDGE_COMPARISON_RESULT_FILENAME),
        )
    assert list(Path(directory).iterdir()) == []
