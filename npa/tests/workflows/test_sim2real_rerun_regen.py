from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.reporting import build_progress_metrics
from npa.workflows.sim2real_rerun_regen import (
    Sim2RealRerunRegenError,
    _ensure_policy_access_metadata,
    regen_sim2real_rrd,
    resolve_local_rrd_path,
    sync_heldout_renders,
)


def _config(run_id: str = "sim2real-staged-20260616t093101z") -> Sim2RealLoopConfig:
    return Sim2RealLoopConfig(
        run_id=run_id,
        s3_bucket="demo-bucket",
        s3_prefix="sim2real-b",
        s3_endpoint="https://storage.example",
    )


def test_resolve_local_rrd_path_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCAL_RRD_PATH", str(tmp_path / "custom.rrd"))
    assert (
        resolve_local_rrd_path("sim2real-staged-20260616t093101z")
        == tmp_path / "custom.rrd"
    )


def test_regen_sim2real_rrd_requires_heldout_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_dir = tmp_path / "run"
    (local_dir / "inner_loop/outer-01").mkdir(parents=True)
    (local_dir / "eval/heldout").mkdir(parents=True)
    (local_dir / "inner_loop/outer-01/evidence.json").write_text(
        json.dumps({"iterations": []}),
        encoding="utf-8",
    )
    (local_dir / "eval/heldout/report.json").write_text(
        json.dumps({"success_rate": 1.0, "render_manifest": {"episodes": []}}),
        encoding="utf-8",
    )

    class FakeResult:
        output_rrd_path = str(local_dir / "reports" / "sim2real.rrd")
        heldout_frame_count = 0
        rollout_count = 0
        frame_count = 0

    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen.emit_sim2real_rerun",
        lambda **_kwargs: FakeResult(),
    )

    with pytest.raises(Sim2RealRerunRegenError, match="heldout_frame_count=0"):
        regen_sim2real_rrd(_config(), local_dir=local_dir, sync_inputs=False)


def test_regen_sim2real_rrd_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_dir = tmp_path / "run"
    (local_dir / "inner_loop/outer-01").mkdir(parents=True)
    (local_dir / "eval/heldout").mkdir(parents=True)
    (local_dir / "inner_loop/outer-01/evidence.json").write_text(
        json.dumps({"iterations": []}),
        encoding="utf-8",
    )
    (local_dir / "eval/heldout/report.json").write_text(
        json.dumps({"success_rate": 1.0}),
        encoding="utf-8",
    )

    class FakeResult:
        output_rrd_path = str(local_dir / "reports" / "sim2real.rrd")
        heldout_frame_count = 4
        rollout_count = 0
        frame_count = 0

    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen.emit_sim2real_rerun",
        lambda **_kwargs: FakeResult(),
    )

    result = regen_sim2real_rrd(
        _config(), local_dir=local_dir, sync_inputs=False, upload=False
    )
    assert result.heldout_frame_count == 4
    assert result.local_rrd_path.endswith("sim2real.rrd")


@pytest.mark.parametrize(
    ("provenance", "expected_loaded", "expected_stock_or_scripted"),
    [
        pytest.param(
            {
                "checkpoint_uri": "s3://demo-bucket/run/model.pt",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_size_bytes": 128,
                "loaded_for_inference": True,
                "stock_or_scripted_policy": False,
                "actor_is_learned": True,
                "scripted_post_actor_controller": False,
                "policy_composition": "learned_actor_only",
                "post_actor_controller": None,
            },
            True,
            False,
            id="proven",
        ),
        pytest.param("true", False, None, id="malformed-object"),
    ],
)
def test_regen_preserves_strict_heldout_policy_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provenance: object,
    expected_loaded: bool,
    expected_stock_or_scripted: bool | None,
) -> None:
    local_dir = tmp_path / "run"
    (local_dir / "inner_loop/outer-01").mkdir(parents=True)
    (local_dir / "eval/heldout").mkdir(parents=True)
    (local_dir / "inner_loop/outer-01/evidence.json").write_text(
        json.dumps({"iterations": []}),
        encoding="utf-8",
    )
    (local_dir / "eval/heldout/report.json").write_text(
        json.dumps(
            {
                "success_rate": 1.0,
                "policy_checkpoint_sha256": "a" * 64,
                "policy_checkpoint_size_bytes": 128,
                "policy_inference_provenance": provenance,
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeResult:
        output_rrd_path = str(local_dir / "reports" / "sim2real.rrd")
        heldout_frame_count = 4
        rollout_count = 0
        frame_count = 0

    def fake_emit(**kwargs):
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen.emit_sim2real_rerun", fake_emit
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen.emit_sim2real_mcap_if_enabled",
        lambda **_kwargs: {"status": "skipped"},
    )

    regen_sim2real_rrd(_config(), local_dir=local_dir, sync_inputs=False)

    metadata = captured["run_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["heldout_policy_loaded_for_inference"] is expected_loaded
    assert (
        metadata["heldout_policy_stock_or_scripted_policy"]
        is expected_stock_or_scripted
    )
    assert metadata["heldout_policy_identity_verified"] is expected_loaded
    assert metadata["heldout_policy_learned_actor_only"] is (
        expected_loaded and expected_stock_or_scripted is False
    )


def _regen_fixture(tmp_path: Path) -> Path:
    local_dir = tmp_path / "run"
    (local_dir / "inner_loop/outer-01").mkdir(parents=True)
    (local_dir / "eval/heldout").mkdir(parents=True)
    (local_dir / "inner_loop/outer-01/evidence.json").write_text(
        json.dumps({"iterations": [], "reward_trend": [0.1, 0.2]}), encoding="utf-8"
    )
    (local_dir / "eval/heldout/report.json").write_text(
        json.dumps({"success_rate": 1.0}), encoding="utf-8"
    )
    return local_dir


def _patch_rrd_emit(monkeypatch: pytest.MonkeyPatch, local_dir: Path) -> None:
    class FakeResult:
        output_rrd_path = str(local_dir / "reports" / "sim2real.rrd")
        heldout_frame_count = 4
        rollout_count = 1
        frame_count = 4

    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen.emit_sim2real_rerun",
        lambda **_kwargs: FakeResult(),
    )


def test_regen_also_refreshes_the_mcap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finalize writes both recordings from one set of inputs, so regen must too.

    Refreshing only the .rrd leaves the run's MCAP frozen at whatever the emitter
    produced when the run first completed, so viewer-side fixes never reach it.
    """

    local_dir = _regen_fixture(tmp_path)
    _patch_rrd_emit(monkeypatch, local_dir)

    seen: dict[str, object] = {}

    def fake_mcap(*, local_dir, inner_evidence, heldout_report, output_mcap):
        seen["output_mcap"] = output_mcap
        seen["reward_trend"] = inner_evidence.get("reward_trend")
        Path(output_mcap).parent.mkdir(parents=True, exist_ok=True)
        Path(output_mcap).write_bytes(b"\x89MCAP0\r\n")
        return {"status": "written", "output_mcap_path": str(output_mcap)}

    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen.emit_sim2real_mcap_if_enabled", fake_mcap
    )

    uploaded: list[tuple[str, str]] = []

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            uploaded.append((local, uri))
            return uri

        def upload_directory(self, local: str, uri: str) -> str:
            uploaded.append((local, uri))
            return uri

    (local_dir / "reports").mkdir(parents=True, exist_ok=True)
    (local_dir / "reports" / "sim2real.rrd").write_bytes(b"rrd")

    result = regen_sim2real_rrd(
        _config(),
        local_dir=local_dir,
        sync_inputs=False,
        upload=True,
        client=FakeStorage(),
    )

    # Emitted next to the .rrd, from the same synced inputs.
    assert Path(str(seen["output_mcap"])).name == "sim2real.mcap"
    assert seen["reward_trend"] == [0.1, 0.2]
    assert result.mcap_status == "written"
    assert result.local_mcap_path.endswith("sim2real.mcap")
    # And published to the run prefix so the agent/viewer picks it up.
    assert result.mcap_upload_uri.endswith("/reports/sim2real.mcap")
    assert any(uri.endswith("/reports/sim2real.mcap") for _local, uri in uploaded)


def test_regen_mcap_failure_never_breaks_the_rrd_regen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MCAP emission is best-effort here, exactly as in the finalize stage."""

    local_dir = _regen_fixture(tmp_path)
    _patch_rrd_emit(monkeypatch, local_dir)
    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen.emit_sim2real_mcap_if_enabled",
        lambda **_kwargs: {"status": "skipped", "reason": "mcap not installed"},
    )

    result = regen_sim2real_rrd(
        _config(), local_dir=local_dir, sync_inputs=False, upload=False
    )
    assert result.heldout_frame_count == 4
    assert result.mcap_status == "skipped"
    assert result.local_mcap_path == ""
    assert result.mcap_upload_uri == ""


def test_policy_access_metadata_hashes_real_checkpoint_without_secrets(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "checkpoints" / "candidate" / "candidate.json"
    candidate_path.parent.mkdir(parents=True)
    checkpoint_uri = "s3://demo-bucket/run/model_latest.pt"
    candidate_path.write_text(
        json.dumps(
            {
                "deployable_policy": True,
                "policy_checkpoint_uri": checkpoint_uri,
            }
        ),
        encoding="utf-8",
    )

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            assert uri == checkpoint_uri
            Path(destination).write_bytes(b"real-policy-bytes")

    access = _ensure_policy_access_metadata(
        _config(), tmp_path, storage=FakeStorage(), report={}
    )

    assert access["deployable_policy"] is True
    assert access["identity"] == "model_latest.pt"
    assert len(access["sha256"]) == 64
    assert access["size_bytes"] == len(b"real-policy-bytes")
    assert checkpoint_uri in access["authenticated_download_command"]
    assert "$AWS_ENDPOINT_URL" in access["authenticated_download_command"]
    assert access["viewer_executes_policy"] is False
    assert "secret" not in json.dumps(access).lower()


@pytest.mark.parametrize(
    ("field", "claim"),
    [
        ("deployable_policy", "true"),
        ("deployable_policy", 1),
        ("policy_bytes_available", "true"),
        ("policy_bytes_available", 1),
    ],
)
def test_policy_access_metadata_rejects_non_boolean_deployment_claims(
    tmp_path: Path,
    field: str,
    claim: object,
) -> None:
    candidate_path = tmp_path / "checkpoints" / "candidate" / "candidate.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps(
            {
                field: claim,
                "policy_checkpoint_uri": "s3://demo-bucket/run/model_latest.pt",
            }
        ),
        encoding="utf-8",
    )
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(b"unexpected-policy-bytes")

    access = _ensure_policy_access_metadata(
        _config(), tmp_path, storage=FakeStorage(), report={}
    )

    assert access["deployable_policy"] is False
    assert access["policy_bytes_available"] is False
    assert downloads == []


def test_policy_access_metadata_consumes_canonical_stage14_decision(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    checkpoint_sha256 = "a" * 64

    class UnusedStorage:
        def download_file(self, *_args, **_kwargs) -> None:
            raise AssertionError("complete retained identity must not be redownloaded")

    access = _ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=UnusedStorage(),
        report={
            "outer_loop": {
                "decision": {
                    "decision": "promote_checkpoint",
                    "checkpoint_uri": checkpoint_uri,
                },
                "latest_heldout_report": {
                    "policy_checkpoint_sha256": checkpoint_sha256,
                    "policy_checkpoint_size_bytes": 128,
                },
            },
            "checkpoint_selection": {
                "checkpoint_uri": checkpoint_uri,
                "checkpoint_sha256": checkpoint_sha256,
            },
        },
    )

    assert access["checkpoint_uri"] == checkpoint_uri
    assert access["sha256"] == checkpoint_sha256
    assert access["size_bytes"] == 128
    assert access["deployable_policy"] is True
    assert access["identity"] == "model_selected.pt"


@pytest.mark.parametrize(
    "conflict",
    ["checkpoint_uri", "checkpoint_sha256", "checkpoint_size_bytes"],
)
def test_policy_access_metadata_rejects_conflicting_retained_identity(
    tmp_path: Path,
    conflict: str,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    checkpoint_sha256 = "a" * 64
    candidate_path = tmp_path / "checkpoints/candidate/candidate.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps(
            {
                "deployable_policy": True,
                "policy_checkpoint_uri": checkpoint_uri,
                "policy_checkpoint_sha256": checkpoint_sha256,
                "policy_checkpoint_size_bytes": 128,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "outer_loop": {
            "decision": {
                "decision": "promote_checkpoint",
                "checkpoint_uri": checkpoint_uri,
            },
            "latest_heldout_report": {
                "policy_checkpoint_sha256": checkpoint_sha256,
                "policy_checkpoint_size_bytes": 128,
            },
        },
        "checkpoint_selection": {
            "checkpoint_uri": checkpoint_uri,
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
    if conflict == "checkpoint_uri":
        report["outer_loop"]["decision"]["checkpoint_uri"] = (
            "s3://demo-bucket/run/other.pt"
        )
    elif conflict == "checkpoint_sha256":
        report["checkpoint_selection"]["checkpoint_sha256"] = "b" * 64
    else:
        report["outer_loop"]["latest_heldout_report"][
            "policy_checkpoint_size_bytes"
        ] = 256

    with pytest.raises(Sim2RealRerunRegenError, match="sources disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=object(),
            report=report,
        )


def test_progress_metrics_embed_all_outer_reward_loss_and_success(
    tmp_path: Path,
) -> None:
    history = []
    for outer, success in ((1, 0.5), (2, 1.0)):
        evidence_path = tmp_path / "inner_loop" / f"outer-{outer:02d}" / "evidence.json"
        evidence_path.parent.mkdir(parents=True)
        evidence_path.write_text(
            json.dumps(
                {
                    "reward_trend": [0.1 * outer, 0.2 * outer],
                    "final_quality": 0.7 + outer / 10,
                    "iterations": [
                        {
                            "iteration": 1,
                            "mean_reward": 0.1 * outer,
                            "quality_after": 0.7,
                            "update": {
                                "loss_before": 0.8,
                                "loss_after": 0.4,
                                "checkpoint_path": f"s3://run/model_{outer}.pt",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        history.append(
            {
                "outer_iteration": outer,
                "inner_loop": str(evidence_path),
                "checkpoint_uri": f"s3://run/model_{outer}.pt",
                "decision": {
                    "success_rate": success,
                    "threshold": 0.8,
                    "decision": "promote_checkpoint" if success >= 0.8 else "loop_back",
                },
            }
        )

    metrics = build_progress_metrics(tmp_path, history)
    assert metrics["outer_iteration_count"] == 2
    assert [
        item["evaluation_success_rate"] for item in metrics["outer_iterations"]
    ] == [
        0.5,
        1.0,
    ]
    assert metrics["outer_iterations"][0]["loss_trend"] == [
        {"before": 0.8, "after": 0.4}
    ]
    assert metrics["outer_iterations"][1]["reward_trend"] == [0.2, 0.4]
    assert metrics["stage_12_external_stub"]["tier"] == "SEAM"


def test_sync_heldout_renders_falls_back_to_byo_eval_tree(tmp_path: Path) -> None:
    run_id = "s2r-real-0725t222636z"
    config = _config(run_id=run_id)
    local_dir = tmp_path / "run"
    report_path = local_dir / "eval" / "heldout" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")

    class FakePaginator:
        def paginate(self, *, Bucket: str, Prefix: str, Delimiter: str):
            if Prefix.endswith(f"{run_id}/byo-eval/"):
                return [
                    {
                        "CommonPrefixes": [
                            {
                                "Prefix": (
                                    f"sim2real-b/{run_id}/byo-eval/"
                                    f"s2r-byo-isaac-eval-{run_id}/"
                                )
                            }
                        ]
                    }
                ]
            return [{"CommonPrefixes": []}]

    class FakeS3:
        def get_paginator(self, name: str) -> FakePaginator:
            assert name == "list_objects_v2"
            return FakePaginator()

    class FakeStorage:
        _s3 = FakeS3()

        def download_directory(self, uri: str, dest: str) -> None:
            if uri.endswith(f"/byo-eval/s2r-byo-isaac-eval-{run_id}/renders/"):
                env_dir = Path(dest) / "env-00006"
                env_dir.mkdir(parents=True)
                (env_dir / "camera-000.png").write_bytes(b"png")
                return
            raise OSError(uri)

        def download_path(self, uri: str, local_path: str) -> None:
            if uri.endswith(
                f"/byo-eval/s2r-byo-isaac-eval-{run_id}/render-manifest.json"
            ):
                Path(local_path).parent.mkdir(parents=True)
                Path(local_path).write_text(
                    json.dumps(
                        {
                            "schema": "npa.sim2real.heldout_renders.v1",
                            "episodes": [
                                {"env_id": "env-00006", "frames": ["camera-000.png"]}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return
            raise OSError(uri)

    assert sync_heldout_renders(
        config, local_dir, heldout_report={"success_rate": 1.0}, client=FakeStorage()
    )
    assert (
        local_dir / "eval" / "heldout" / "renders" / "env-00006" / "camera-000.png"
    ).is_file()
    updated_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert updated_report["render_manifest"]["episodes"][0]["env_id"] == "env-00006"


def test_gold_render_sync_uses_only_explicit_lineage(tmp_path: Path) -> None:
    config = _config("gold-run")
    local_dir = tmp_path / "run"
    exact_uri = "s3://demo-bucket/sim2real-b/gold-run/byo-eval/gold-exact/renders/"
    downloads: list[str] = []

    class FakeStorage:
        def download_directory(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            assert uri == exact_uri
            episode = Path(destination) / "gold-0001"
            episode.mkdir(parents=True)
            (episode / "camera-000.png").write_bytes(b"gold-frame")

    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 3,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "renders_s3_uri": exact_uri,
            "checkpoint_uri": "s3://demo-bucket/checkpoints/model_500.pt",
        },
    }
    assert sync_heldout_renders(
        config, local_dir, heldout_report=report, client=FakeStorage()
    )
    assert downloads == [exact_uri]
    assert (
        local_dir
        / "eval"
        / "gold-heldout"
        / "outer-03"
        / "renders"
        / "gold-0001"
        / "camera-000.png"
    ).read_bytes() == b"gold-frame"


def test_gold_render_sync_rejects_missing_or_validation_lineage(tmp_path: Path) -> None:
    class UnusedStorage:
        def download_directory(self, *_args, **_kwargs):
            raise AssertionError("must fail before S3")

    with pytest.raises(Sim2RealRerunRegenError, match="no exact render_lineage"):
        sync_heldout_renders(
            _config("gold-run"),
            tmp_path,
            heldout_report={"evaluation_split": "gold_heldout"},
            client=UnusedStorage(),
        )
    with pytest.raises(Sim2RealRerunRegenError, match="wrong evaluation split"):
        sync_heldout_renders(
            _config("gold-run"),
            tmp_path,
            heldout_report={
                "evaluation_split": "gold_heldout",
                "render_lineage": {
                    "evaluation_split": "validation",
                    "renders_s3_uri": "s3://bucket/validation/renders/",
                },
            },
            client=UnusedStorage(),
        )
