"""Regeneration ownership and derived-index replay use only owned fixtures."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from npa.workflows import sim2real_rerun_regen as regen
from npa.workflows import sim2real_viz as viz


@pytest.mark.parametrize("redirect", ["root", "checkpoints", "candidate", "leaf"])
def test_load_failure_does_not_modify_redirected_candidate(tmp_path, redirect):
    work, outside = tmp_path / "run", tmp_path / "outside"
    candidate = outside / "checkpoints/candidate/candidate.json"
    candidate.parent.mkdir(parents=True)
    original = b'{"deployable_policy":true,"marker":"outside"}\n'
    candidate.write_bytes(original)
    if redirect == "root":
        work.symlink_to(outside, target_is_directory=True)
    else:
        work.mkdir()
        relative = {
            "checkpoints": "checkpoints",
            "candidate": "checkpoints/candidate",
            "leaf": "checkpoints/candidate/candidate.json",
        }[redirect]
        link = work / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside / relative, target_is_directory=redirect != "leaf")
    config = SimpleNamespace(run_id="fixture-run", s3_bucket="unit", s3_prefix="runs")
    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._load_regen_state(config, work, object(), sync_inputs=False)
    assert candidate.read_bytes() == original


def test_candidate_write_rechecks_ancestor_after_read(tmp_path):
    work, outside = tmp_path / "run", tmp_path / "outside"
    candidate = work / "checkpoints/candidate/candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text('{"deployable_policy":true}')
    path, payload = regen._read_candidate_manifest(work)
    candidate.parent.rename(outside)
    candidate.parent.symlink_to(outside, target_is_directory=True)
    before = (outside / "candidate.json").read_bytes()
    with pytest.raises(regen.Sim2RealRerunRegenError, match="symlinked ancestor"):
        regen._write_candidate_manifest(path, payload)
    regen._persist_failed_candidate(path, payload)
    assert (outside / "candidate.json").read_bytes() == before


def test_contained_candidate_is_still_failed_closed(tmp_path):
    candidate = tmp_path / "checkpoints/candidate/candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        json.dumps({"deployable_policy": True, "policy_download_command": "old access"})
    )
    regen._fail_closed_local_candidate(tmp_path)
    payload = json.loads(candidate.read_text())
    assert payload["deployable_policy"] is False
    assert payload["policy_bytes_available"] is False
    assert "policy_download_command" not in payload


class _MemoryStorage:
    def __init__(self):
        self.objects = {}

    def read_bytes_with_etag(self, uri):
        data = self.objects.get(uri)
        return None if data is None else (data, hashlib.sha256(data).hexdigest())

    def put_bytes_conditional(self, data, uri, *, if_none_match=False):
        assert if_none_match and uri not in self.objects
        self.objects[uri] = data


def _publish_index(storage, local):
    index = local / "reports/sim2real-visual-index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    decision = local / "outer_loop/decision.json"
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text('{"run_id":"fixture-run"}')
    payload = viz._build_visual_index(
        local_dir=local, inner_evidence={}, heldout_report={}, critique_panel_rows=[]
    )
    index.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    publication = SimpleNamespace(
        publish_renders=False,
        report_path=local / "missing",
        visual_index_path=index,
        candidate_path=local / "missing-candidate",
        prefix="s3://unit/run/",
        generation_prefix="s3://unit/run/reports/generations/fixture/",
    )
    regen._publish_regen_supporting_artifacts(storage, publication)
    return publication


def test_real_index_replay_and_relocation_preserve_canonical_bytes(tmp_path):
    storage = _MemoryStorage()
    canonical = "s3://unit/run/reports/sim2real-visual-index.json"
    storage.objects[canonical] = b"canonical index is not owned by regeneration"
    first = _publish_index(storage, tmp_path / "first")
    first_uri = regen._regen_visual_index_uri(first)
    first_bytes = storage.objects[first_uri]
    second = _publish_index(storage, tmp_path / "first")
    second_uri = regen._regen_visual_index_uri(second)
    moved = _publish_index(storage, tmp_path / "relocated")
    moved_uri = regen._regen_visual_index_uri(moved)
    assert len({first_uri, second_uri, moved_uri}) == 3
    assert storage.objects[first_uri] == first_bytes
    assert storage.objects[canonical] == b"canonical index is not owned by regeneration"
    regen._publish_regen_supporting_artifacts(storage, moved)
    assert len(storage.objects) == 4
    report = tmp_path / "report.json"
    report.write_text('{"run_id":"fixture-run"}')
    regen._seal_regen_publication_report(
        report,
        publication_id="fixture",
        rrd_uri="s3://unit/run/recording.rrd",
        mcap_uri="",
        report_uri="s3://unit/run/report.json",
        journal_uri="s3://unit/run/journal.json",
        visual_index_uri=moved_uri,
    )
    assert (
        json.loads(report.read_text())["publication"]["visual_index_uri"] == moved_uri
    )
