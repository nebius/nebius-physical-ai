#!/usr/bin/env python3
"""Build a factual Rerun recording from validated HY-World output only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ENTITY_ROOT = "hy_world_2"
VIDEO_ENTITY = f"{ENTITY_ROOT}/generated_scene/camera_render"
FRAME_ENTITY = f"{ENTITY_ROOT}/generated_scene/camera_frame"
REPORT_SCHEMA = "npa.workbench.hy_world_2.rrd_manifest.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "evidence is not a JSON object")
    _require(
        payload.get("pipeline_mode") == "image_to_world", "not image-to-world evidence"
    )
    _require(
        payload.get("capability") == "hy_world_2_image_conditioned_world_generation",
        "wrong capability",
    )
    return payload


def _verify_cli(rrd: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for command in ("verify", "stats"):
        result = subprocess.run(
            [sys.executable, "-m", "rerun", "rrd", command, str(rrd)],
            capture_output=True,
            check=False,
            text=True,
        )
        _require(
            result.returncode == 0,
            f"rerun rrd {command} failed: {result.stderr.strip()}",
        )
        if command == "stats":
            for name in ("num_chunks", "num_entity_paths", "num_rows"):
                match = re.search(
                    rf"^{name}\s*=\s*([0-9]+)", result.stdout, re.MULTILINE
                )
                _require(
                    match is not None and int(match.group(1)) > 0,
                    f"RRD stats missing {name}",
                )
                values[name] = int(match.group(1))
    return values


def _record(
    rrd: Path,
    evidence: dict[str, Any],
    video: Path,
    frame_count: int,
    run_id: str,
) -> str:
    import rerun as rr
    import rerun.blueprint as rrb

    recording_id = hashlib.sha256(
        f"{run_id}\0{_sha256(video)}".encode("utf-8")
    ).hexdigest()[:32]
    recording = rr.RecordingStream("npa_hy_world_2", recording_id=recording_id)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin=f"{ENTITY_ROOT}/generated_scene",
                contents=f"{ENTITY_ROOT}/generated_scene/**",
            ),
            rrb.TextDocumentView(
                origin=f"{ENTITY_ROOT}/evidence", name="Generated-scene evidence"
            ),
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
    )
    recording.save(rrd, default_blueprint=blueprint)
    try:
        asset = rr.AssetVideo(path=video)
        recording.log(VIDEO_ENTITY, asset, static=True)
        recording.log(
            f"{ENTITY_ROOT}/evidence",
            rr.TextDocument(
                json.dumps(evidence, indent=2, sort_keys=True),
                media_type="application/json",
            ),
            static=True,
        )
        timestamps = asset.read_frame_timestamps_nanos()
        _require(
            len(timestamps) == frame_count,
            "Rerun and decoded-video frame counts disagree",
        )
        for frame, timestamp in enumerate(timestamps):
            rr.set_time("frame", sequence=frame, recording=recording)
            rr.set_time(
                "video_time",
                duration=int(timestamp) / 1_000_000_000,
                recording=recording,
            )
            recording.log(
                FRAME_ENTITY,
                rr.VideoFrameReference(
                    nanoseconds=int(timestamp), video_reference=VIDEO_ENTITY
                ),
            )
        recording.flush()
        return str(rr.__version__)
    finally:
        recording.disconnect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--rrd", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        evidence = _load_evidence(args.evidence)
        render = evidence.get("rendered_camera_dataset") or {}
        _require(
            render.get("sha256") == _sha256(args.video), "rendered MP4 hash mismatch"
        )
        frame_count = int((render.get("decode") or {}).get("decoded_frames") or 0)
        _require(frame_count >= 3, "validated rendered MP4 has too few frames")
        args.rrd.parent.mkdir(parents=True, exist_ok=True)
        _require(bool(args.run_id.strip()), "run ID is empty")
        sdk_version = _record(args.rrd, evidence, args.video, frame_count, args.run_id)
        stats = _verify_cli(args.rrd)
        manifest = {
            "schema": REPORT_SCHEMA,
            "status": "verified",
            "capability": "hy_world_2_factual_scene_report",
            "rrd": {
                "path": args.rrd.name,
                "bytes": args.rrd.stat().st_size,
                "sha256": _sha256(args.rrd),
            },
            "rendered_mp4_sha256": _sha256(args.video),
            "run_id": args.run_id,
            "rerun_sdk_version": sdk_version,
            "rerun_stats": stats,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        ImportError,
    ) as exc:
        print(f"npa-hy-world report: {exc}", file=sys.stderr)
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
