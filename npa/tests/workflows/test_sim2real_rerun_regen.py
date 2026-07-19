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
