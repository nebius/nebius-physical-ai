"""Unit tests for SONIC retargeting source-format auto-detection and output mode.

These are hermetic: they exercise the pure detection/output helpers without
cloning the upstream converter or touching S3.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from npa.workbench.retargeting import (
    RetargetingError,
    _combined_pkl_output,
    _detect_source_format,
    _effective_individual,
    _local_output_target,
)


def _write_soma_clip(clip_dir: Path, frames: int = 4) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    header_joint = ",".join(f"joint_{i}" for i in range(29))
    (clip_dir / "joint_pos.csv").write_text(
        header_joint + "\n" + "\n".join(",".join("0.0" for _ in range(29)) for _ in range(frames)),
        encoding="utf-8",
    )
    (clip_dir / "body_pos.csv").write_text(
        ",".join(f"b{i}" for i in range(42)) + "\n"
        + "\n".join(",".join("0.0" for _ in range(42)) for _ in range(frames)),
        encoding="utf-8",
    )
    (clip_dir / "body_quat.csv").write_text(
        ",".join(f"q{i}" for i in range(56)) + "\n"
        + "\n".join(",".join("0.0" for _ in range(56)) for _ in range(frames)),
        encoding="utf-8",
    )


def test_detect_soma_csv_single_dir(tmp_path: Path) -> None:
    _write_soma_clip(tmp_path)
    assert _detect_source_format(tmp_path) == "soma-csv"


def test_detect_soma_csv_parent_dir(tmp_path: Path) -> None:
    _write_soma_clip(tmp_path / "macarena_001")
    _write_soma_clip(tmp_path / "squat_002")
    assert _detect_source_format(tmp_path) == "soma-csv"


def test_detect_bones_seed_csv(tmp_path: Path) -> None:
    header = "Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ"
    header += "," + ",".join(f"j{i}_dof" for i in range(29))
    (tmp_path / "session.csv").write_text(header + "\n" + ",".join("0.0" for _ in range(36)), encoding="utf-8")
    assert _detect_source_format(tmp_path) == "bones-seed-csv"


def test_detect_deploy_pkl(tmp_path: Path) -> None:
    pkl = tmp_path / "deploy.pkl"
    joblib.dump(
        {
            "walk": {
                "joint_pos": np.zeros((4, 29), dtype=np.float32),
                "body_pos_w": np.zeros((4, 14, 3), dtype=np.float32),
                "body_quat_w": np.zeros((4, 14, 4), dtype=np.float32),
            }
        },
        pkl,
    )
    assert _detect_source_format(pkl) == "deploy-pkl"


def test_detect_motion_lib_pkl(tmp_path: Path) -> None:
    pkl = tmp_path / "already.pkl"
    joblib.dump(
        {
            "walk": {
                "root_trans_offset": np.zeros((4, 3), dtype=np.float32),
                "pose_aa": np.zeros((4, 30, 3), dtype=np.float32),
                "dof": np.zeros((4, 29), dtype=np.float32),
                "root_rot": np.zeros((4, 4), dtype=np.float32),
                "fps": 30,
            }
        },
        pkl,
    )
    assert _detect_source_format(pkl) == "motion-lib"


def test_detect_rejects_unknown_directory(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(RetargetingError):
        _detect_source_format(tmp_path)


def test_soma_csv_runs_combined_not_individual() -> None:
    # SOMA CSV directories must never run the Bones-SEED-only --individual path.
    assert _effective_individual("soma-csv", True) is False
    assert _combined_pkl_output("soma-csv", False, input_is_file=False) is True


def test_bones_seed_honors_individual() -> None:
    assert _effective_individual("bones-seed-csv", True) is True
    assert _combined_pkl_output("bones-seed-csv", True, input_is_file=False) is False
    assert _combined_pkl_output("bones-seed-csv", False, input_is_file=False) is True


def test_soma_csv_local_output_is_single_pkl(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_soma_clip(src / "clip")
    out = tmp_path / "retargeted"
    target = _local_output_target(
        output_path=str(out),
        work_dir=tmp_path,
        source_format="soma-csv",
        input_path=src,
        individual=False,
    )
    assert target == out / "motion_lib.pkl"


def test_soma_csv_s3_output_is_single_pkl(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_soma_clip(src / "clip")
    target = _local_output_target(
        output_path="s3://bucket/retargeted/",
        work_dir=tmp_path,
        source_format="soma-csv",
        input_path=src,
        individual=False,
    )
    assert target == tmp_path / "output" / "motion_lib.pkl"
