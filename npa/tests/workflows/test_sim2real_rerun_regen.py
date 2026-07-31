from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real_rerun_regen import (
    Sim2RealRerunRegenError,
    regen_sim2real_rrd,
    resolve_local_rrd_path,
)


def _config(run_id: str = "sim2real-staged-20260616t093101z") -> Sim2RealLoopConfig:
    return Sim2RealLoopConfig(
        run_id=run_id,
        s3_bucket="demo-bucket",
        s3_prefix="sim2real-b",
        s3_endpoint="https://storage.example",
    )


def test_resolve_local_rrd_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCAL_RRD_PATH", str(tmp_path / "custom.rrd"))
    assert resolve_local_rrd_path("sim2real-staged-20260616t093101z") == tmp_path / "custom.rrd"


def test_regen_sim2real_rrd_requires_heldout_frames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_regen_sim2real_rrd_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    result = regen_sim2real_rrd(_config(), local_dir=local_dir, sync_inputs=False, upload=False)
    assert result.heldout_frame_count == 4
    assert result.local_rrd_path.endswith("sim2real.rrd")


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


def test_regen_also_refreshes_the_mcap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    result = regen_sim2real_rrd(_config(), local_dir=local_dir, sync_inputs=False, upload=False)
    assert result.heldout_frame_count == 4
    assert result.mcap_status == "skipped"
    assert result.local_mcap_path == ""
    assert result.mcap_upload_uri == ""
