"""Display actual Franka camera frames and named joint telemetry from LeRobot data."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from npa.viz.adapters.lerobot_to_rerun import verify_rerun_entities


def _blueprint(metadata: dict):
    import rerun.blueprint as rrb

    tabs = []
    for index in range(metadata["num_episodes"]):
        root = f"episodes/{index:06d}"
        tabs.append(rrb.Vertical(
            rrb.Spatial2DView(origin=f"/{root}/camera", name="Isaac RTX camera"),
            rrb.Horizontal(rrb.TimeSeriesView(origin=f"/{root}/joints", name="Joint positions"),
                           rrb.TimeSeriesView(origin=f"/{root}/actions", name="Applied actions")),
            rrb.TextDocumentView(origin=f"/{root}/measurement", name="Measured outcome"),
            name=f"Episode {index}", row_shares=[5, 2, 1],
        ))
    return rrb.Blueprint(rrb.Tabs(*tabs), collapse_panels=True)


def _episode(recording, dataset: Path, metadata: dict, info: dict, location: dict, rows: list) -> list[str]:
    import rerun as rr

    index = int(location["episode_index"])
    root, camera = f"episodes/{index:06d}", "observation.images.workspace"
    path = dataset / info["video_path"].format(video_key=camera,
        chunk_index=location[f"videos/{camera}/chunk_index"], file_index=location[f"videos/{camera}/file_index"])
    if not path.is_file():
        raise ValueError("Franka LeRobot dataset is missing its genuine simulator video")
    rr.log(f"{root}/video", rr.AssetVideo(path=path), static=True, recording=recording)
    outcome = metadata["episode_results"][index]
    rr.log(f"{root}/measurement", rr.TextDocument(json.dumps(outcome, indent=2)), static=True, recording=recording)
    required = [f"{root}/camera"]
    for row in rows:
        rr.set_time("frame", sequence=int(row["frame_index"]), recording=recording)
        rr.set_time("time", duration=float(row["timestamp"]), recording=recording)
        seconds = float(row["timestamp"]) + location[f"videos/{camera}/from_timestamp"]
        rr.log(f"{root}/camera", rr.VideoFrameReference(seconds=seconds, video_reference=f"{root}/video"), recording=recording)
        for field, group, names in (("observation.state", "joints", metadata["state_names"]),
                                    ("action", "actions", metadata["action_names"])):
            if len(row[field]) != len(names):
                raise ValueError("Franka LeRobot telemetry does not match its named embodiment")
            for name, value in zip(names, row[field], strict=True):
                rr.log(f"{root}/{group}/{name}", rr.Scalars(float(value)), recording=recording)
    for group, names in (("joints", metadata["state_names"]), ("actions", metadata["action_names"])):
        required.extend(f"{root}/{group}/{name}" for name in names)
    return required


def write_recording(dataset: Path, output: Path, metadata: dict) -> dict[str, int]:
    """Record decoded LeRobot rows and embedded simulator videos with run provenance.

    Args:
        dataset: Converted Franka LeRobotDataset with genuine workspace videos.
        output: Destination Rerun recording.
        metadata: Actual capture outcomes, feature names, checkpoint, and run ID.
    Returns:
        Decoded dynamic row counts for every camera, joint, and action entity.
    Raises:
        ValueError: Capture provenance or named telemetry is incomplete.
        OSError: Dataset or recording files cannot be read or written.
    """
    import rerun as rr

    if not metadata.get("genuine_simulator_pixels") or not metadata.get("run_id"):
        raise ValueError("Franka Rerun export requires actual capture provenance and a run ID")
    info = json.loads((dataset / "meta/info.json").read_text())
    rows = [row for path in sorted((dataset / "data").rglob("*.parquet")) for row in pq.read_table(path).to_pylist()]
    episodes = [row for path in sorted((dataset / "meta/episodes").rglob("*.parquet")) for row in pq.read_table(path).to_pylist()]
    recording = rr.RecordingStream(f"npa-{metadata.get('robot_type', 'franka')}-rl", recording_id=metadata["run_id"])
    rr.save(output, default_blueprint=_blueprint(metadata), recording=recording)
    rr.log("provenance", rr.TextDocument(json.dumps(metadata, indent=2)), static=True, recording=recording)
    required = []
    for location in episodes:
        selected = sorted((row for row in rows if row["episode_index"] == location["episode_index"]),
                          key=lambda row: row["frame_index"])
        required.extend(_episode(recording, dataset, metadata, info, location, selected))
    recording.flush()
    return verify_rerun_entities(output, required)
