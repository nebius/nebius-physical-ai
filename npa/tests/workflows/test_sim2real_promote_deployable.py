"""threshold_decision: promote references real weights + is deployable when a BYO
trained checkpoint exists; falls back to reference-metadata stub otherwise."""
from __future__ import annotations

import json

from npa.workflows.sim2real.config import build_config_from_env
from npa.workflows.sim2real.engine import threshold_decision


def _cfg(tmp_path):
    return build_config_from_env(threshold=0.45, s3_bucket="", run_id="t",
                                 output_dir=str(tmp_path))


def _candidate(tmp_path):
    return json.loads((tmp_path / "checkpoints" / "candidate" / "candidate.json").read_text())


def test_promote_deployable_with_real_checkpoint(tmp_path):
    report = {"success_rate": 1.0,
              "policy_checkpoint": "s3://b/run/byo-trainer/model_latest.pt"}
    d = threshold_decision(_cfg(tmp_path), local_dir=tmp_path, heldout_report=report, outer_iteration=1)
    assert d["decision"] == "promote_checkpoint"
    assert d["checkpoint_uri"] == "s3://b/run/byo-trainer/model_latest.pt"
    cand = _candidate(tmp_path)
    assert cand["deployable_policy"] is True
    assert cand["source"] == "isaac-rsl-rl-ppo"
    assert cand["policy_artifact_kind"] == "isaac_rsl_rl_checkpoint"
    assert cand["policy_checkpoint_uri"].endswith("model_latest.pt")


def test_promote_stub_without_real_checkpoint(tmp_path):
    report = {"success_rate": 1.0}  # reference path: no policy_checkpoint
    d = threshold_decision(_cfg(tmp_path), local_dir=tmp_path, heldout_report=report, outer_iteration=1)
    assert d["decision"] == "promote_checkpoint"
    cand = _candidate(tmp_path)
    assert cand["deployable_policy"] is False
    assert cand["policy_artifact_kind"] == "reference_metadata"
