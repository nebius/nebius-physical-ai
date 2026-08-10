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


def _real_report(success_rate: float) -> dict:
    return {
        "success_rate": success_rate,
        "evaluation_split": "gold_heldout",
        "policy_checkpoint": "s3://b/run/byo-trainer/model_latest.pt",
        "per_env": [{"env_id": f"gold-{index:04d}"} for index in range(64)],
        "policy_inference_provenance": {
            "loaded_for_inference": True,
            "stock_or_scripted_policy": False,
        },
        "applied_scenario_proof": {"exact_digest_match": True},
    }


def test_engine_preserves_threshold_decision_import() -> None:
    from npa.workflows.sim2real import decision, engine

    assert engine.threshold_decision is decision.threshold_decision


def test_promote_deployable_with_real_checkpoint(tmp_path):
    report = _real_report(1.0)
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
    assert cand["effective_learning_rate"] == 0.08
    assert not (tmp_path / "outer_loop" / "loopback.json").exists()


def test_promote_stub_without_real_checkpoint(tmp_path):
    report = {"success_rate": 1.0, "per_env": [{"env_id": "reference-0"}]}
    d = threshold_decision(
        _cfg(tmp_path), local_dir=tmp_path, heldout_report=report, outer_iteration=1
    )
    assert d["decision"] == "promote_checkpoint"
    cand = _candidate(tmp_path)
    assert cand["deployable_policy"] is False
    assert cand["policy_artifact_kind"] == "reference_metadata"
    assert not (tmp_path / "outer_loop" / "loopback.json").exists()


def test_below_threshold_real_checkpoint_remains_packaged_candidate(tmp_path):
    report = _real_report(0.125)

    decision = threshold_decision(
        _cfg(tmp_path),
        local_dir=tmp_path,
        heldout_report=report,
        outer_iteration=1,
    )

    assert decision["decision"] == "loop_back_to_inner_loop"
    candidate = _candidate(tmp_path)
    assert candidate["deployable_policy"] is False
    assert candidate["policy_bytes_available"] is True
    assert candidate["policy_artifact_kind"] == "isaac_rsl_rl_checkpoint"
    assert candidate["policy_checkpoint_uri"].endswith("model_latest.pt")
    assert candidate["threshold_met"] is False
    assert candidate["promotion_decision"] == "loop_back_to_inner_loop"
    assert candidate["candidate_status"] == "below_threshold_policy_artifact"
    assert decision["candidate"] == candidate
    assert candidate["promoted_at"] == ""
    loopback = json.loads((tmp_path / "outer_loop" / "loopback.json").read_text())
    assert loopback["schema"] == "npa.sim2real.loopback.v1"
    assert loopback["real_policy"] is True
    assert loopback["policy_checkpoint_uri"] == report["policy_checkpoint"]
    assert loopback["candidate_path"] == str(
        tmp_path / "checkpoints" / "candidate" / "candidate.json"
    )
    assert loopback["score"] == 0.125
    assert loopback["threshold"] == 0.45
    assert loopback["outer_iteration"] == 1
    assert loopback["remaining_outer_iterations"] == 2
    assert loopback["remaining_work"] == "run_next_outer_iteration"
    assert loopback["decision"]["decision"] == "loop_back_to_inner_loop"


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
    report = _real_report(1.0)

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
        "effective_learning_rate",
        "learning_rate_scope",
        "policy_artifact_kind",
        "policy_bytes_available",
        "policy_checkpoint_identity",
        "policy_checkpoint_sha256",
        "policy_checkpoint_size_bytes",
        "policy_checkpoint_uri",
        "policy_download_command",
        "policy_ui_action",
        "promoted_at",
        "promotion_decision",
        "promotion_gates",
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
