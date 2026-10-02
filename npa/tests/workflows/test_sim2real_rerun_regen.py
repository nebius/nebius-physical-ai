from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.reporting import build_progress_metrics
from npa.workflows.sim2real_rerun_regen import (
    Sim2RealRerunRegenError,
    _ensure_policy_access_metadata,
    _gold_report_path,
    _policy_access_record,
    _stage_components,
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


def _write_candidate_manifest(tmp_path: Path, candidate: dict) -> Path:
    candidate_path = tmp_path / "checkpoints" / "candidate" / "candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    return candidate_path


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


def test_gold_report_path_tracks_latest_completed_outer_iteration(
    tmp_path: Path,
) -> None:
    for outer in (1, 2):
        evidence = tmp_path / "inner_loop" / f"outer-{outer:02d}" / "evidence.json"
        report = (
            tmp_path / "eval" / "gold-heldout" / f"outer-{outer:02d}" / "report.json"
        )
        evidence.parent.mkdir(parents=True)
        report.parent.mkdir(parents=True)
        evidence.write_text("{}", encoding="utf-8")
        report.write_text(json.dumps({"outer_iteration": outer}), encoding="utf-8")

    assert _gold_report_path(_config(), tmp_path) == (
        tmp_path / "eval" / "gold-heldout" / "outer-02" / "report.json"
    )


def test_stage_components_preserves_canonical_component_records() -> None:
    report = {"component_records": [{"name": "stage_14_rerun_viz", "tier": "SEAM"}]}

    components = _stage_components(report)
    components[0]["tier"] = "WORKS"

    assert report == {
        "component_records": [{"name": "stage_14_rerun_viz", "tier": "WORKS"}]
    }
    assert "components" not in report


def test_stage_components_keeps_equal_compatibility_aliases_synchronized() -> None:
    canonical = [{"name": "stage_14_rerun_viz", "tier": "SEAM"}]
    report = {
        "component_records": canonical,
        "components": [dict(canonical[0])],
    }

    components = _stage_components(report)
    components[0]["tier"] = "WORKS"

    assert report["component_records"] == report["components"]
    assert report["component_records"] is report["components"]


@pytest.mark.parametrize(
    ("provenance", "expected_loaded", "expected_stock_or_scripted"),
    [
        pytest.param(
            {
                "checkpoint_uri": "s3://demo-bucket/run/model.pt",
                "checkpoint_sha256": "a" * 64,
                "generator_policy_sha256": "a" * 64,
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

    if not isinstance(provenance, dict):
        with pytest.raises(
            Sim2RealRerunRegenError,
            match="policy_inference_provenance must be a JSON object",
        ):
            regen_sim2real_rrd(_config(), local_dir=local_dir, sync_inputs=False)
        return

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
    checkpoint_uri = "s3://demo-bucket/run/model_latest.pt"
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_checkpoint_uri": checkpoint_uri,
        },
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


def test_policy_access_metadata_rejects_downloaded_digest_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_latest.pt"
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_checkpoint_uri": checkpoint_uri,
            "policy_checkpoint_sha256": "a" * 64,
        },
    )

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            assert uri == checkpoint_uri
            Path(destination).write_bytes(b"different-policy-bytes")

    with pytest.raises(
        Sim2RealRerunRegenError,
        match="downloaded checkpoint SHA-256 disagrees",
    ):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=FakeStorage(),
            report={},
        )


@pytest.mark.parametrize(
    "checkpoint_uri",
    [
        pytest.param("not-a-uri", id="relative"),
        pytest.param("https://demo-bucket/run/model.pt", id="wrong-scheme"),
        pytest.param("s3://bad bucket/model.pt", id="authority-space"),
        pytest.param("s3://Demo-bucket/run/model.pt", id="uppercase-bucket"),
        pytest.param("s3://demo-bucket./run/model.pt", id="trailing-dot-bucket"),
        pytest.param("s3://dëmo-bucket/run/model.pt", id="unicode-bucket"),
        pytest.param("s3://demo-bucket/run/../model.pt", id="parent-segment"),
        pytest.param(
            "s3://demo-bucket/run/%2e%2e/model.pt",
            id="encoded-parent-segment",
        ),
        pytest.param("s3://demo-bucket/run//model.pt", id="empty-segment"),
        pytest.param("s3://demo-bucket/run\\model.pt", id="backslash"),
        pytest.param("s3://demo-bucket/run/\u202emodel.pt", id="bidi-control"),
    ],
)
def test_policy_access_metadata_rejects_malformed_uri_before_download(
    tmp_path: Path,
    checkpoint_uri: str,
) -> None:
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_checkpoint_uri": checkpoint_uri,
        },
    )
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(b"unexpected-policy-bytes")

    with pytest.raises(Sim2RealRerunRegenError, match="URI.*malformed"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=FakeStorage(),
            report={},
        )

    assert downloads == []


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
    candidate_path = _write_candidate_manifest(
        tmp_path,
        {
            field: claim,
            "policy_checkpoint_uri": "s3://demo-bucket/run/model_latest.pt",
        },
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
    persisted = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert persisted["deployable_policy"] is False
    assert persisted["policy_bytes_available"] is False


def test_policy_access_metadata_consumes_canonical_stage14_decision(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    checkpoint_bytes = b"selected-policy"
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(checkpoint_bytes)

    access = _ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=FakeStorage(),
        report={
            "outer_loop": {
                "decision": {
                    "decision": "promote_checkpoint",
                    "checkpoint_uri": checkpoint_uri,
                },
                "latest_heldout_report": {
                    "policy_checkpoint_sha256": checkpoint_sha256,
                    "policy_checkpoint_size_bytes": len(checkpoint_bytes),
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
    assert access["size_bytes"] == len(checkpoint_bytes)
    assert access["deployable_policy"] is True
    assert access["identity"] == "model_selected.pt"
    assert downloads == [checkpoint_uri]


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
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_checkpoint_uri": checkpoint_uri,
            "policy_checkpoint_sha256": checkpoint_sha256,
            "policy_checkpoint_size_bytes": 128,
        },
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


def test_policy_access_metadata_rejects_conflicting_decision_aliases(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    report = {
        "outer_loop": {
            "latest_decision": {
                "decision": "promote_checkpoint",
                "checkpoint_uri": checkpoint_uri,
            },
            "decision": {
                "decision": "loop_back_to_inner_loop",
                "checkpoint_uri": checkpoint_uri,
            },
        }
    }

    with pytest.raises(Sim2RealRerunRegenError, match="decision.*disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=object(),
            report=report,
        )


def test_policy_access_metadata_never_promotes_loop_back_candidate_claims(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    candidate_path = _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_bytes_available": True,
            "policy_checkpoint_uri": checkpoint_uri,
        },
    )
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(b"below-threshold-policy-bytes")

    access = _ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=FakeStorage(),
        report={
            "outer_loop": {
                "decision": {
                    "decision": "loop_back_to_inner_loop",
                    "checkpoint_uri": checkpoint_uri,
                }
            }
        },
    )

    assert access["deployable_policy"] is False
    assert access["policy_bytes_available"] is True
    assert downloads == [checkpoint_uri]
    persisted = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert persisted["deployable_policy"] is False
    assert persisted["policy_bytes_available"] is True
    assert access["authenticated_download_command"] == ""
    assert access["ui_action"] == ""
    assert "policy_download_command" not in persisted
    assert "policy_ui_action" not in persisted


def test_policy_access_record_never_returns_instructions_when_not_deployable() -> None:
    access = _policy_access_record(
        _config(),
        {
            "policy_download_command": "stale download command",
            "policy_ui_action": "stale UI action",
        },
        checkpoint_uri="s3://demo-bucket/run/model.pt",
        bytes_available=True,
        deployable=False,
    )

    assert access["authenticated_download_command"] == ""
    assert access["ui_action"] == ""


def test_policy_access_metadata_reconciles_producer_before_download(
    tmp_path: Path,
) -> None:
    candidate_uri = "s3://demo-bucket/run/model_selected.pt"
    producer_uri = "s3://demo-bucket/run/other.pt"
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_bytes_available": True,
            "policy_checkpoint_uri": candidate_uri,
        },
    )
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(b"unexpected-policy-bytes")

    with pytest.raises(Sim2RealRerunRegenError, match="sources disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=FakeStorage(),
            report={
                "outer_loop": {
                    "decision": {
                        "decision": "promote_checkpoint",
                        "checkpoint_uri": candidate_uri,
                    },
                    "latest_heldout_report": {
                        "policy_checkpoint_sha256": "a" * 64,
                        "policy_checkpoint_size_bytes": 128,
                        "policy_inference_provenance": {
                            "checkpoint_uri": producer_uri,
                            "checkpoint_sha256": "a" * 64,
                            "checkpoint_size_bytes": 128,
                            "loaded_for_inference": True,
                            "stock_or_scripted_policy": False,
                            "actor_is_learned": True,
                            "scripted_post_actor_controller": False,
                            "policy_composition": "learned_actor_only",
                            "post_actor_controller": None,
                        },
                    },
                }
            },
        )

    assert downloads == []


def _promoted_candidate_report(
    checkpoint_uri: str,
    candidate: dict | None = None,
) -> dict:
    decision = {
        "decision": "promote_checkpoint",
        "checkpoint_uri": checkpoint_uri,
    }
    if candidate is not None:
        decision["candidate"] = candidate
    return {"outer_loop": {"decision": decision}}


def test_policy_access_metadata_reconciles_embedded_decision_candidate(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_bytes_available": True,
            "policy_checkpoint_uri": checkpoint_uri,
            "policy_checkpoint_sha256": "a" * 64,
            "policy_checkpoint_size_bytes": 128,
        },
    )
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(b"unexpected-policy-bytes")

    with pytest.raises(Sim2RealRerunRegenError, match="sources disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=FakeStorage(),
            report=_promoted_candidate_report(
                checkpoint_uri,
                {
                    "deployable_policy": True,
                    "policy_bytes_available": True,
                    "policy_checkpoint_uri": "s3://demo-bucket/run/other.pt",
                    "policy_checkpoint_sha256": "b" * 64,
                    "policy_checkpoint_size_bytes": 256,
                },
            ),
        )

    assert downloads == []


def _embedded_candidate_report(
    checkpoint_uri: str,
    checkpoint_sha256: str,
    checkpoint_size: int,
) -> dict:
    return _promoted_candidate_report(
        checkpoint_uri,
        {
            "deployable_policy": False,
            "policy_bytes_available": True,
            "policy_checkpoint_uri": checkpoint_uri,
            "policy_checkpoint_sha256": checkpoint_sha256,
            "generator_policy_sha256": checkpoint_sha256,
            "policy_checkpoint_size_bytes": checkpoint_size,
        },
    )


def test_policy_access_metadata_reconciles_embedded_candidate_deployment_claim(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    checkpoint_bytes = b"selected-policy"
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(checkpoint_bytes)

    access = _ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=FakeStorage(),
        report=_embedded_candidate_report(
            checkpoint_uri, digest, len(checkpoint_bytes)
        ),
    )

    assert access["policy_bytes_available"] is True
    assert access["deployable_policy"] is False
    assert downloads == [checkpoint_uri]


@pytest.mark.parametrize(
    ("alias", "value"),
    [
        pytest.param("checkpoint_uri", "s3://demo-bucket/run/other.pt", id="uri"),
        pytest.param("sha256", "b" * 64, id="digest"),
        pytest.param("checkpoint_sha256", "b" * 64, id="checkpoint-digest"),
        pytest.param("size_bytes", 256, id="size"),
        pytest.param("checkpoint_size_bytes", 256, id="checkpoint-size"),
        pytest.param("identity", "other.pt", id="identity"),
    ],
)
def test_policy_access_metadata_rejects_embedded_candidate_alias_conflicts(
    tmp_path: Path,
    alias: str,
    value: object,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    candidate = {
        "deployable_policy": True,
        "policy_bytes_available": True,
        "policy_checkpoint_uri": checkpoint_uri,
        "policy_checkpoint_identity": "model_selected.pt",
        "policy_checkpoint_sha256": "a" * 64,
        "policy_checkpoint_size_bytes": 128,
        alias: value,
    }

    with pytest.raises(Sim2RealRerunRegenError, match="disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=object(),
            report={
                "outer_loop": {
                    "decision": {
                        "decision": "promote_checkpoint",
                        "checkpoint_uri": checkpoint_uri,
                        "candidate": candidate,
                    }
                },
                "checkpoint_selection": {
                    "checkpoint_uri": checkpoint_uri,
                    "checkpoint_sha256": "a" * 64,
                    "checkpoint_size_bytes": 128,
                },
            },
        )


@pytest.mark.parametrize(
    ("alias", "value"),
    [
        pytest.param("checkpoint_uri", "s3://demo-bucket/run/other.pt", id="uri"),
        pytest.param("sha256", "b" * 64, id="digest"),
        pytest.param("checkpoint_sha256", "b" * 64, id="checkpoint-digest"),
        pytest.param("size_bytes", 256, id="size"),
        pytest.param("checkpoint_size_bytes", 256, id="checkpoint-size"),
        pytest.param("identity", "other.pt", id="identity"),
    ],
)
def test_policy_access_metadata_rejects_manifest_candidate_alias_conflicts(
    tmp_path: Path,
    alias: str,
    value: object,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    candidate = {
        "deployable_policy": True,
        "policy_bytes_available": True,
        "policy_checkpoint_uri": checkpoint_uri,
        "policy_checkpoint_identity": "model_selected.pt",
        "policy_checkpoint_sha256": "a" * 64,
        "policy_checkpoint_size_bytes": 128,
        alias: value,
    }
    _write_candidate_manifest(tmp_path, candidate)

    with pytest.raises(Sim2RealRerunRegenError, match="disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=object(),
            report={},
        )


def test_policy_access_metadata_reconciles_current_heldout_before_download(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_bytes_available": True,
            "policy_checkpoint_uri": checkpoint_uri,
            "policy_checkpoint_sha256": "a" * 64,
            "policy_checkpoint_size_bytes": 128,
        },
    )
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(b"unexpected-policy-bytes")

    with pytest.raises(Sim2RealRerunRegenError, match="sources disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=FakeStorage(),
            report=_promoted_candidate_report(checkpoint_uri),
            heldout_report={
                "policy_inference_provenance": {
                    "checkpoint_uri": "s3://demo-bucket/run/other.pt",
                    "checkpoint_sha256": "a" * 64,
                    "checkpoint_size_bytes": 128,
                }
            },
        )

    assert downloads == []


@pytest.mark.parametrize("source", ["inner_selection", "current_decision"])
def test_policy_access_metadata_reconciles_current_control_plane_before_download(
    tmp_path: Path,
    source: str,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    other_uri = "s3://demo-bucket/run/other.pt"
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_bytes_available": True,
            "policy_checkpoint_uri": checkpoint_uri,
            "policy_checkpoint_sha256": "a" * 64,
            "policy_checkpoint_size_bytes": 128,
        },
    )
    downloads: list[str] = []

    class FakeStorage:
        def download_file(self, uri: str, destination: str) -> None:
            downloads.append(uri)
            Path(destination).write_bytes(b"unexpected-policy-bytes")

    kwargs: dict[str, object] = {}
    if source == "inner_selection":
        kwargs["inner_evidence"] = {
            "selected_checkpoint_uri": other_uri,
            "final_checkpoint_uri": other_uri,
            "checkpoint_selection": {
                "checkpoint_uri": other_uri,
                "checkpoint_sha256": "a" * 64,
                "checkpoint_size_bytes": 128,
                "generator_policy_sha256": "a" * 64,
            },
            "checkpoint_candidates": [
                {
                    "checkpoint_uri": other_uri,
                    "checkpoint_sha256": "a" * 64,
                    "checkpoint_size_bytes": 128,
                    "generator_policy_sha256": "a" * 64,
                }
            ],
        }
    else:
        kwargs["current_decision"] = {
            "decision": "promote_checkpoint",
            "checkpoint_uri": other_uri,
        }

    with pytest.raises(Sim2RealRerunRegenError, match="sources disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=FakeStorage(),
            report={},
            **kwargs,
        )
    assert downloads == []


def test_policy_access_metadata_clears_stale_instructions_when_unavailable(
    tmp_path: Path,
) -> None:
    candidate_path = _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": False,
            "policy_bytes_available": False,
            "policy_checkpoint_uri": "s3://demo-bucket/run/model.pt",
            "policy_download_command": "stale download command",
            "policy_ui_action": "stale UI action",
        },
    )

    access = _ensure_policy_access_metadata(
        _config(),
        tmp_path,
        storage=object(),
        report={},
    )

    assert access["authenticated_download_command"] == ""
    assert access["ui_action"] == ""
    persisted = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert "policy_download_command" not in persisted
    assert "policy_ui_action" not in persisted


def test_policy_access_metadata_reconciles_prior_access_record(
    tmp_path: Path,
) -> None:
    candidate_uri = "s3://demo-bucket/run/model_selected.pt"
    candidate_path = _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_bytes_available": True,
            "policy_checkpoint_uri": candidate_uri,
        },
    )

    with pytest.raises(Sim2RealRerunRegenError, match="sources disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=object(),
            report={
                "policy_access": {
                    "checkpoint_uri": "s3://demo-bucket/run/other.pt",
                    "sha256": "a" * 64,
                    "size_bytes": 128,
                    "deployable_policy": True,
                    "policy_bytes_available": True,
                }
            },
        )
    persisted = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert persisted["deployable_policy"] is False
    assert persisted["policy_bytes_available"] is False
    assert "policy_download_command" not in persisted
    assert "policy_ui_action" not in persisted


def test_policy_access_metadata_reconciles_selected_candidate_generator_alias(
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model_selected.pt"
    report = {
        "selected_checkpoint_candidate": {
            "policy_checkpoint_uri": checkpoint_uri,
            "policy_checkpoint_sha256": "a" * 64,
            "generator_policy_sha256": "a" * 64,
            "policy_generator_sha256": "b" * 64,
            "policy_checkpoint_size_bytes": 128,
        }
    }

    with pytest.raises(Sim2RealRerunRegenError, match="sources disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=object(),
            report=report,
        )


def test_policy_access_metadata_rejects_duplicate_identity_fields(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "checkpoints/candidate/candidate.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        (
            '{"deployable_policy":true,'
            '"policy_checkpoint_uri":"s3://demo-bucket/run/model.pt",'
            '"policy_checkpoint_uri":"s3://demo-bucket/run/other.pt"}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(Sim2RealRerunRegenError, match="duplicate.*field"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=object(),
            report={},
        )


def test_failure_state_persistence_never_masks_primary_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_uri = "s3://demo-bucket/run/model.pt"
    _write_candidate_manifest(
        tmp_path,
        {
            "deployable_policy": True,
            "policy_bytes_available": True,
            "policy_checkpoint_uri": checkpoint_uri,
        },
    )

    def fail_write(*_args, **_kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen._write_candidate_manifest",
        fail_write,
    )

    with pytest.raises(Sim2RealRerunRegenError, match="sources disagree"):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=object(),
            report={
                "policy_access": {
                    "checkpoint_uri": "s3://demo-bucket/run/other.pt",
                }
            },
        )


def test_process_control_exception_does_not_rewrite_candidate_state(
    tmp_path: Path,
) -> None:
    candidate = {
        "deployable_policy": True,
        "policy_bytes_available": True,
        "policy_checkpoint_uri": "s3://demo-bucket/run/model.pt",
        "policy_download_command": "retained command",
        "policy_ui_action": "retained action",
    }
    candidate_path = _write_candidate_manifest(tmp_path, candidate)
    original = candidate_path.read_bytes()

    class InterruptedStorage:
        def download_file(self, _uri: str, _destination: str) -> None:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _ensure_policy_access_metadata(
            _config(),
            tmp_path,
            storage=InterruptedStorage(),
            report={},
        )

    assert candidate_path.read_bytes() == original


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


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError("current render tree unavailable"), id="os-error"),
        pytest.param(
            ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}},
                "ListObjectsV2",
            ),
            id="provider-client-error",
        ),
    ],
)
def test_gold_render_sync_never_reuses_stale_frames_after_download_failure(
    tmp_path: Path,
    error: Exception,
) -> None:
    config = _config("gold-run")
    local_dir = tmp_path / "run"
    renders_dir = local_dir / "eval" / "gold-heldout" / "outer-03" / "renders"
    stale_frame = renders_dir / "stale" / "camera-000.png"
    stale_frame.parent.mkdir(parents=True)
    stale_frame.write_bytes(b"stale-frame")

    class FailingStorage:
        def download_directory(self, _uri: str, _destination: str) -> None:
            raise error

    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 3,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "renders_s3_uri": ("s3://demo-bucket/sim2real-b/gold-run/current/renders/"),
        },
    }

    assert not sync_heldout_renders(
        config,
        local_dir,
        heldout_report=report,
        client=FailingStorage(),
    )
    assert not stale_frame.exists()
    assert not renders_dir.exists()


def test_gold_render_sync_unlinks_stale_symlink_after_client_error(
    tmp_path: Path,
) -> None:
    config = _config("gold-run")
    local_dir = tmp_path / "run"
    renders_dir = local_dir / "eval" / "gold-heldout" / "outer-03" / "renders"
    outside = tmp_path / "outside-renders"
    outside_frame = outside / "stale" / "camera-000.png"
    outside_frame.parent.mkdir(parents=True)
    outside_frame.write_bytes(b"outside-frame")
    renders_dir.parent.mkdir(parents=True)
    renders_dir.symlink_to(outside, target_is_directory=True)

    class FailingStorage:
        def download_directory(self, _uri: str, _destination: str) -> None:
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}},
                "ListObjectsV2",
            )

    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 3,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "canonical_s3_uri": (
                "s3://demo-bucket/sim2real-b/gold-run/current/renders/"
            ),
        },
    }

    assert not sync_heldout_renders(
        config,
        local_dir,
        heldout_report=report,
        client=FailingStorage(),
    )
    assert not renders_dir.is_symlink()
    assert outside_frame.read_bytes() == b"outside-frame"


def test_gold_render_sync_preserves_provider_error_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config("gold-run")
    local_dir = tmp_path / "run"
    renders_dir = local_dir / "eval" / "gold-heldout" / "outer-03" / "renders"
    stale_frame = renders_dir / "stale" / "camera-000.png"
    stale_frame.parent.mkdir(parents=True)
    stale_frame.write_bytes(b"stale-frame")

    class FailingStorage:
        def download_directory(self, _uri: str, _destination: str) -> None:
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}},
                "ListObjectsV2",
            )

    def fail_cleanup(_path: Path) -> None:
        raise OSError("read-only render tree")

    monkeypatch.setattr(
        "npa.workflows.sim2real_rerun_regen._remove_tree",
        fail_cleanup,
    )
    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 3,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "canonical_s3_uri": (
                "s3://demo-bucket/sim2real-b/gold-run/current/renders/"
            ),
        },
    }

    with pytest.raises(ClientError, match="AccessDenied") as caught:
        sync_heldout_renders(
            config,
            local_dir,
            heldout_report=report,
            client=FailingStorage(),
        )
    assert isinstance(caught.value.__cause__, OSError)
    assert stale_frame.read_bytes() == b"stale-frame"


def test_gold_render_sync_reconciles_canonical_and_legacy_lineage_uri(
    tmp_path: Path,
) -> None:
    exact_uri = "s3://demo-bucket/sim2real-b/gold-run/exact/renders/"

    class FakeStorage:
        def download_directory(self, uri: str, destination: str) -> None:
            assert uri == exact_uri
            episode = Path(destination) / "gold-0001"
            episode.mkdir(parents=True)
            (episode / "camera-000.png").write_bytes(b"gold-frame")

    report = {
        "evaluation_split": "gold_heldout",
        "outer_iteration": 3,
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "canonical_s3_uri": exact_uri,
            "renders_s3_uri": exact_uri,
        },
    }
    assert sync_heldout_renders(
        _config("gold-run"),
        tmp_path,
        heldout_report=report,
        client=FakeStorage(),
    )

    report["render_lineage"]["renders_s3_uri"] = (
        "s3://demo-bucket/sim2real-b/gold-run/other/renders/"
    )
    with pytest.raises(Sim2RealRerunRegenError, match="URI sources disagree"):
        sync_heldout_renders(
            _config("gold-run"),
            tmp_path,
            heldout_report=report,
            client=FakeStorage(),
        )


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
