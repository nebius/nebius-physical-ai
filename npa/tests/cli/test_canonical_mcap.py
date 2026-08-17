from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.agent_backend.canonical_mcap import (
    canonical_key_for_run,
    clear_cross_run_mcap_state,
    has_rich_visualization_contract,
    prepare_canonical_mcap,
    rich_run_provenance_from_manifest,
)


class _S3:
    def __init__(self) -> None:
        self.objects = {"runs/run-1/camera/frame.png": b"real-image-bytes"}

    def put_object(self, *, Bucket, Key, Body, **_kwargs):
        del Bucket
        self.objects[Key] = Body.read() if hasattr(Body, "read") else bytes(Body)

    def get_object(self, *, Bucket, Key):
        del Bucket
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket, Key):
        del Bucket
        return {"ContentLength": len(self.objects[Key])}


def _safe_key(value: str) -> str:
    if ".." in value.split("/"):
        raise ValueError("unsafe key")
    return value.strip("/")


def test_prepare_canonical_mcap_persists_and_reuses_exact_s3_bytes(
    tmp_path: Path,
) -> None:
    s3 = _S3()
    invalidations: list[bool] = []

    def find(_buckets, **_kwargs):
        items = [
            SimpleNamespace(key=key, s3_uri=f"s3://bucket/{key}")
            for key in sorted(s3.objects)
        ]
        return "bucket", items

    def download(uri: str, destination: Path, **_kwargs):
        key = uri.removeprefix("s3://bucket/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(s3.objects[key])
        return destination

    class _Converted:
        def to_dict(self):
            return {"message_count": 2, "channels": {"/camera": 2}}

    def convert(*, output_path: Path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x89MCAP0\r\ncanonical-payload\x89MCAP0\r\n")
        return _Converted()

    def summarize(path: Path):
        return SimpleNamespace(
            to_dict=lambda: {
                "valid_magic": True,
                "size_bytes": path.stat().st_size,
                "message_count": 2,
                "channels": {"/camera": 2},
                "channel_time_ranges": {"/camera": [0, 100_000_000]},
                "start_time_ns": 0,
                "end_time_ns": 100_000_000,
                "duration_s": 0.1,
                "metadata": {"npa": {"timestamps": "synthetic-fps", "fps": "10"}},
            }
        )

    kwargs = {
        "run_id": "run-1",
        "fps": 10.0,
        "max_frames": 10,
        "validate_run_id": lambda value: value,
        "s3_client": lambda: (s3, {"prefix": "runs"}),
        "list_buckets": lambda *_args: ["bucket"],
        "find_artifacts": find,
        "safe_key": _safe_key,
        "download": download,
        "convert": convert,
        "summarize": summarize,
        "invalidate_cache": lambda: invalidations.append(True),
        "now_iso": lambda: "2026-08-09T18:53:07+00:00",
        "recordings_dir": tmp_path / "recordings",
    }

    first = prepare_canonical_mcap(**kwargs)
    exact = s3.objects["runs/run-1/reports/sim2real.mcap"]
    second = prepare_canonical_mcap(**kwargs)

    assert first["created"] is True
    assert second["created"] is False
    assert first["source"] == second["source"] == "generated-from-s3-artifacts"
    assert Path(second["local_path"]).read_bytes() == exact
    assert first["sha256"] == second["sha256"]
    provenance = json.loads(
        s3.objects["runs/run-1/reports/sim2real.mcap.provenance.json"]
    )
    assert provenance["schema"] == "npa.canonical-mcap.v2"
    assert (
        provenance["canonical_s3_uri"] == "s3://bucket/runs/run-1/reports/sim2real.mcap"
    )
    assert provenance["channel_time_ranges"]["/camera"] == [0, 100_000_000]
    assert provenance["timestamps"] == "synthetic-fps"
    assert provenance["duration_s"] == 0.1
    assert len(invalidations) == 2


def test_canonical_key_and_cross_run_state_are_strict() -> None:
    assert (
        canonical_key_for_run("prefix/run-1/camera/a.png", "run-1", safe_key=_safe_key)
        == "prefix/run-1/reports/sim2real.mcap"
    )
    with pytest.raises((RuntimeError, ValueError)):
        canonical_key_for_run("prefix/run-2/../secret", "run-1", safe_key=_safe_key)

    state = {
        "run_id": "run-1",
        "mcap_uri": "/old.mcap",
        "canonical_mcap_sha256": "old",
        "foxglove_cloud": {"recording_id": "old"},
    }
    clear_cross_run_mcap_state(state, "run-2")
    assert state["mcap_uri"] == ""
    assert state["canonical_mcap_sha256"] == ""
    assert state["foxglove_cloud"] == {}


def test_rich_visualization_contract_requires_every_meaningful_topic() -> None:
    schemas = {
        "/camera": "foxglove.CompressedImage",
        "/robot/diagnostic_scene": "foxglove.SceneUpdate",
        "/robot/diagnostic_pose": "foxglove.PoseInFrame",
        "/robot/diagnostic_trajectory": "foxglove.PosesInFrame",
        "/robot/diagnostic_joint_states": "foxglove.JointStates",
        "/actuators/commands": "npa.ActuatorCommands",
        "/run/state": "npa.RunState",
        "/metrics/execution": "npa.RunMetrics.execution",
        "/log": "foxglove.Log",
    }
    info = {
        "schemas": schemas,
        "metadata": {"npa": {"visualization_contract": "npa.foxglove.robot-motion.v3"}},
    }

    assert has_rich_visualization_contract(info) is True
    assert (
        has_rich_visualization_contract(
            {
                "schemas": schemas,
                "visualization_contract": "npa.foxglove.robot-motion.v3",
            }
        )
        is True
    )
    for missing in schemas:
        incomplete = {
            **info,
            "schemas": {k: v for k, v in schemas.items() if k != missing},
        }
        assert has_rich_visualization_contract(incomplete) is False
    assert (
        has_rich_visualization_contract(
            {**info, "metadata": {"npa": {"visualization_contract": "v1"}}}
        )
        is False
    )
    assert (
        has_rich_visualization_contract(
            {
                **info,
                "metadata": {
                    "npa": {"visualization_contract": "npa.foxglove.robot-motion.v2"}
                },
            }
        )
        is False
    )


def test_prepare_canonical_mcap_replaces_stale_v2_native_bytes(
    tmp_path: Path,
) -> None:
    s3 = _S3()
    canonical_key = "runs/run-1/reports/sim2real.mcap"
    old_bytes = b"old-v2-canonical"
    new_bytes = b"new-v3-canonical"
    s3.objects[canonical_key] = old_bytes
    s3.objects["runs/run-1/reports/rich-run-manifest.json"] = json.dumps(
        {
            "schema": "npa.foxglove.rich-run.v1",
            "run_id": "run-1",
            "engine_provenance": {"engine": "Isaac"},
            "duration_seconds": 3.2,
            "sample_count": 33,
            "fps": 10,
            "camera_counts": {"primary": 33, "side": 33, "workspace": 33},
            "limitations": ["RGB only; no calibrated depth or extrinsics."],
        }
    ).encode()

    def find(_buckets, **_kwargs):
        return "bucket", [
            SimpleNamespace(key=key, s3_uri=f"s3://bucket/{key}")
            for key in sorted(s3.objects)
        ]

    def download(uri: str, destination: Path, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(s3.objects[uri.removeprefix("s3://bucket/")])
        return destination

    def convert(*, output_path: Path, **_kwargs):
        output_path.write_bytes(new_bytes)
        return SimpleNamespace(to_dict=lambda: {"message_count": 32})

    schemas = {
        "/camera": "foxglove.CompressedImage",
        "/robot/diagnostic_scene": "foxglove.SceneUpdate",
        "/robot/diagnostic_pose": "foxglove.PoseInFrame",
        "/robot/diagnostic_trajectory": "foxglove.PosesInFrame",
        "/robot/diagnostic_joint_states": "foxglove.JointStates",
        "/actuators/commands": "npa.ActuatorCommands",
        "/run/state": "npa.RunState",
        "/metrics/execution": "npa.RunMetrics.execution",
        "/log": "foxglove.Log",
    }

    def summarize(path: Path):
        contract = (
            "npa.foxglove.robot-motion.v2"
            if path.read_bytes() == old_bytes
            else "npa.foxglove.robot-motion.v3"
        )
        return SimpleNamespace(
            to_dict=lambda: {
                "valid_magic": True,
                "size_bytes": path.stat().st_size,
                "message_count": 32,
                "channels": {topic: 1 for topic in schemas},
                "schemas": schemas,
                "metadata": {
                    "npa": {
                        "visualization_contract": contract,
                        "scene_update_schema_source": "@foxglove/schemas@2.1.0",
                    }
                },
            }
        )

    result = prepare_canonical_mcap(
        run_id="run-1",
        fps=10,
        max_frames=100,
        validate_run_id=lambda value: value,
        s3_client=lambda: (s3, {"prefix": "runs"}),
        list_buckets=lambda *_args: ["bucket"],
        find_artifacts=find,
        safe_key=_safe_key,
        download=download,
        convert=convert,
        summarize=summarize,
        invalidate_cache=lambda: None,
        now_iso=lambda: "2026-08-14T00:00:00+00:00",
        recordings_dir=tmp_path,
    )

    assert s3.objects[canonical_key] == new_bytes
    assert result["created"] is True
    assert result["source"] == "regenerated-rich-visualization-v3"
    assert result["provenance"]["visualization_contract"].endswith(".v3")
    assert result["provenance"]["scene_update_schema_source"] == (
        "@foxglove/schemas@2.1.0"
    )


def test_rich_run_manifest_compacts_honest_engine_and_limitations() -> None:
    rich = rich_run_provenance_from_manifest(
        {
            "schema": "npa.foxglove.rich-run.v1",
            "run_id": "foxglove-visual-run",
            "engine_provenance": {
                "engine": "NVIDIA Isaac Sim + Isaac Lab via LeIsaac",
                "task": "LeIsaac-SO101-LiftCube-v0",
            },
            "duration_seconds": 9.9375,
            "sample_count": 160,
            "fps": 16,
            "camera_counts": {"overview": 160, "workspace": 160},
            "timestamp_semantics": "episode-relative at source-recorded FPS",
            "trajectory_semantics": "real observation state; not world geometry",
            "limitations": ["No calibrated depth or world-frame object pose."],
            "source": {"frames": ["large source detail must not be copied"]},
        },
        run_id="foxglove-visual-run",
        manifest_key="runs/foxglove-visual-run/reports/rich-run-manifest.json",
        manifest_sha256="a" * 64,
    )

    assert rich["engine_provenance"]["task"] == "LeIsaac-SO101-LiftCube-v0"
    assert rich["camera_counts"] == {"overview": 160, "workspace": 160}
    assert rich["duration_seconds"] == 9.9375
    assert "source" not in rich
    assert "not world geometry" in rich["trajectory_semantics"]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("run_id", "other", "run_id"),
        ("engine_provenance", {}, "engine provenance"),
        ("camera_counts", {}, "camera counts"),
        ("limitations", "none", "limitations"),
        ("sample_count", 0, "duration and sample count"),
    ],
)
def test_rich_run_manifest_rejects_false_or_incomplete_contracts(
    field: str, value, error: str
) -> None:
    manifest = {
        "schema": "npa.foxglove.rich-run.v1",
        "run_id": "run",
        "engine_provenance": {"engine": "real engine"},
        "duration_seconds": 8,
        "sample_count": 80,
        "camera_counts": {"front": 80},
        "limitations": [],
    }
    manifest[field] = value
    with pytest.raises(RuntimeError, match=error):
        rich_run_provenance_from_manifest(
            manifest,
            run_id="run",
            manifest_key="runs/run/reports/rich-run-manifest.json",
            manifest_sha256="b" * 64,
        )
