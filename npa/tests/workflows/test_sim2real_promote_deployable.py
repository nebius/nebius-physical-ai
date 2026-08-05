"""threshold_decision: promote references real weights + is deployable when a BYO
trained checkpoint exists; falls back to reference-metadata stub otherwise."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from npa.workflows.sim2real.config import build_config_from_env
from npa.workflows.sim2real.engine import threshold_decision


def _cfg(tmp_path):
    return build_config_from_env(
        threshold=0.45, s3_bucket="", run_id="t", output_dir=str(tmp_path)
    )


def _candidate(tmp_path):
    return json.loads(
        (tmp_path / "checkpoints" / "candidate" / "candidate.json").read_text()
    )


def test_engine_preserves_threshold_decision_import() -> None:
    from npa.workflows.sim2real import decision, engine

    assert engine.threshold_decision is decision.threshold_decision


def test_promote_deployable_with_real_checkpoint(tmp_path):
    report = {
        "success_rate": 1.0,
        "policy_checkpoint": "s3://b/run/byo-trainer/model_latest.pt",
    }
    d = threshold_decision(
        _cfg(tmp_path), local_dir=tmp_path, heldout_report=report, outer_iteration=1
    )
    assert d["decision"] == "promote_checkpoint"
    assert d["checkpoint_uri"] == "s3://b/run/byo-trainer/model_latest.pt"
    cand = _candidate(tmp_path)
    assert cand["deployable_policy"] is True
    assert cand["source"] == "isaac-rsl-rl-ppo"
    assert cand["policy_artifact_kind"] == "isaac_rsl_rl_checkpoint"
    assert cand["policy_checkpoint_uri"].endswith("model_latest.pt")


def test_promote_stub_without_real_checkpoint(tmp_path):
    report = {"success_rate": 1.0}  # reference path: no policy_checkpoint
    d = threshold_decision(
        _cfg(tmp_path), local_dir=tmp_path, heldout_report=report, outer_iteration=1
    )
    assert d["decision"] == "promote_checkpoint"
    cand = _candidate(tmp_path)
    assert cand["deployable_policy"] is False
    assert cand["policy_artifact_kind"] == "reference_metadata"


def test_below_threshold_real_checkpoint_remains_deployable_candidate(tmp_path):
    report = {
        "success_rate": 0.125,
        "policy_checkpoint": "s3://b/run/byo-trainer/model_latest.pt",
    }

    decision = threshold_decision(
        _cfg(tmp_path),
        local_dir=tmp_path,
        heldout_report=report,
        outer_iteration=1,
    )

    assert decision["decision"] == "loop_back_to_inner_loop"
    candidate = _candidate(tmp_path)
    assert candidate["deployable_policy"] is True
    assert candidate["policy_artifact_kind"] == "isaac_rsl_rl_checkpoint"
    assert candidate["policy_checkpoint_uri"].endswith("model_latest.pt")
    assert candidate["threshold_met"] is False
    assert candidate["promotion_decision"] == "loop_back_to_inner_loop"
    assert candidate["candidate_status"] == "below_threshold_deployable_candidate"
    assert candidate["promoted_at"] == ""


def test_real_checkpoint_candidate_hashes_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.workflows.sim2real import decision

    payload = bytes(range(251)) * 8192

    class FakeStorage:
        def download_file(self, _uri: str, local_path: str) -> str:
            Path(local_path).write_bytes(payload)
            return local_path

    monkeypatch.setattr(
        decision.StorageClient,
        "from_environment",
        lambda **_kwargs: FakeStorage(),
    )

    def forbidden_read_bytes(_path: Path) -> bytes:
        raise AssertionError("Path.read_bytes() must not hash candidate checkpoints")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    report = {
        "success_rate": 1.0,
        "policy_checkpoint": "s3://b/run/byo-trainer/model_latest.pt",
    }

    threshold_decision(
        _cfg(tmp_path),
        local_dir=tmp_path,
        heldout_report=report,
        outer_iteration=1,
    )

    candidate = _candidate(tmp_path)
    assert set(candidate) == {
        "candidate_status",
        "deployable_policy",
        "evaluated_at",
        "handoff_doc",
        "heldout_success_rate",
        "policy_artifact_kind",
        "policy_checkpoint_identity",
        "policy_checkpoint_sha256",
        "policy_checkpoint_size_bytes",
        "policy_checkpoint_uri",
        "policy_download_command",
        "policy_ui_action",
        "promoted_at",
        "promotion_decision",
        "run_id",
        "schema",
        "source",
        "threshold",
        "threshold_met",
    }
    assert candidate["schema"] == "npa.sim2real.candidate_checkpoint.v1"
    assert candidate["policy_checkpoint_identity"] == "model_latest.pt"
    assert candidate["policy_checkpoint_sha256"] == hashlib.sha256(payload).hexdigest()
    assert candidate["policy_checkpoint_size_bytes"] == len(payload)
