"""Verify same-run native training, GPU serving and offline media from the turnkey recipe."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from npa.workflows.policy_training.turnkey_report import reconcile

pytestmark = pytest.mark.e2e


@pytest.fixture
def collected():
    if not os.environ.get("NPA_POLICY_PUBLIC_RESULTS"):
        pytest.skip("requires collected standard-runtime public policy artifacts")
    return Path(os.environ["NPA_POLICY_PUBLIC_RESULTS"])


def test_same_run_native_training_and_both_measured_gates(collected):
    proof = json.loads((collected / "reports/proof.json").read_text())
    assert proof["qualified"] and proof["all_actions_reconciled"]
    assert len(proof["workflow_sha256"]) == 64
    history = proof["history"]
    assert any(row["engine"] == "fiftyone" for row in history)
    for phase in ("pretrain", "finetune"):
        training = [
            r
            for r in history
            if r["stage"].startswith(phase) and r["stage"].endswith("-train")
        ]
        gates = [
            r
            for r in history
            if r.get("phase") == phase and r["stage"].endswith("-gate")
        ]
        assert training and any(r["decision"] == "promote_checkpoint" for r in gates)
        for row in training:
            assert row["engine"] == "lerobot-smolvla-torchrun"
            assert row["checkpoint_sha256"] != row["input_checkpoint_sha256"]
            assert (
                row["training_frames"] > 0
                and row["runtime"]["parameters"] > 100_000_000
            )
        evaluation = next(
            r
            for r in history
            if r.get("phase") == phase and "reserved_demonstrations" in r
        )
        assert evaluation["reserved_demonstrations"]["frames"] > 0


def test_independent_gpu_workers_and_every_applied_http_action(collected):
    client = collected / "serving/client"
    summary = json.loads((client / "summary.json").read_text())
    actions = reconcile(collected / "serving/server", client, summary)
    assert len(actions) == summary["steps"] > 0
    assert summary["new_action_chunks"] > 0
    assert summary["simulation_rendering"] == "NVIDIA EGL"
    assert summary["model"]["cuda"] and summary["model"]["gpu"]
    assert summary["initial_state_offset"] == 2 * summary["trials"]
    roles = [
        json.loads((collected / f"serving/{role}/stage.json").read_text())
        for role in ("server", "client")
    ]
    assert {r["rank"] for r in roles} == {0, 1}
    assert all(r["workers"] == 2 for r in roles)


def test_all_native_videos_decode_and_html_is_standalone(collected):
    import av

    report = collected / "reports"
    html = (report / "demo.html").read_text()
    assert "data:video/mp4;base64," in html and "data:image/png;base64," in html
    assert "__EVIDENCE_JSON__" not in html and "__VIDEO_BASE64__" not in html
    assert "connect-src 'none'" in html
    for forbidden in (
        "Bearer ",
        "session.json",
        "s3://",
        "KUBERNETES_SERVICE_HOST",
        "https://",
    ):
        assert forbidden not in html
    with av.open(str(report / "demo.mp4")) as video:
        frames = list(video.decode(video=0))
        assert len(frames) > 20 and frames[0].width == 1280
        assert frames[0].to_ndarray().std() > 1
