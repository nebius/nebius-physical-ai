"""Give independent photographs exact poses in NRE's single-rig calibration."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from npa.workbench.nurec.nurec import (
    DEFAULT_CONFIG_NAME,
    NurecError,
    read_rig_sidecar,
)

POSES_GROUP = "npa_photographic"


def prepare_photographic_training(config, source: str, *, dry_run: bool = False):
    """Prepare a separate virtual timeline without modifying the source capture.

    Args:
        config: Resolved reconstruction settings and selected cameras.
        source: Verified source NCore metadata with photographic rig provenance.
        dry_run: Return the intended native arguments without writing data.
    Returns:
        Configuration, training metadata path, and optional provenance path.
    Raises:
        NurecError: Photographs cannot be represented without losing source data.
        OSError: The fresh training generation cannot be written.
    """
    if not _needs_photographic_timeline(config, source):
        return config, source, ""
    root = config.resolved_out_dir / "photographic-input"
    evidence = config.resolved_out_dir / "photographic-timeline.json"
    if dry_run:
        target = root / Path(source).name
    else:
        root.mkdir(parents=True, exist_ok=True)
        generation = Path(tempfile.mkdtemp(prefix="generation-", dir=root))
        target = _write_capture(Path(source), generation, evidence)
    return (
        replace(config, poses_component_group=POSES_GROUP),
        str(target),
        str(evidence),
    )


def _needs_photographic_timeline(config, source):
    sidecar = read_rig_sidecar(source)
    if config.config_name != DEFAULT_CONFIG_NAME or not sidecar.get("reference_camera"):
        return False
    if len(sidecar.get("cameras", ())) < 2:
        return False
    if any(
        value.partition("=")[0].lstrip("+~") == "dataset.poses_component_group"
        for value in config.extra_overrides
    ):
        raise NurecError(
            "photographic training cannot use an overriding rig pose group"
        )
    return True


def _write_capture(source: Path, target: Path, evidence: Path) -> Path:
    from ncore.data.v4 import CameraSensorComponent, SequenceComponentGroupsWriter
    from upath import UPath
    from npa.workbench.nurec.ncore_rig import _open_reader, camera_world_trajectories

    reader = _open_reader(source)
    cameras = reader.open_component_readers(CameraSensorComponent.Reader)
    retained = _retained_stores(reader)
    frames = _plan_frames(reader, cameras, camera_world_trajectories(source))
    writer = SequenceComponentGroupsWriter.from_reader(
        output_dir_path=UPath(target),
        store_base_name=reader.sequence_id,
        sequence_reader=reader,
        store_type="itar",
    )
    _write_cameras(writer, cameras, frames)
    _write_poses(writer, reader, cameras, frames, read_rig_sidecar(str(source)))
    stores = [Path(str(path)) for path in writer.finalize()]
    metadata = _combine_stores(source, target, retained, stores)
    _verify_roundtrip(metadata, cameras, frames)
    _write_evidence(source, metadata, cameras, frames, evidence, retained)
    return metadata


def _write_evidence(source, metadata, cameras, frames, evidence, retained):
    target = metadata.parent
    sidecar = {
        **read_rig_sidecar(str(source)),
        "poses_component_group": POSES_GROUP,
        "timeline": "unique virtual photograph timestamps; not capture time",
    }
    (target / "npa-rig.json").write_text(json.dumps(sidecar, indent=2))
    payload = {
        "schema": "npa.nurec.photographic-timeline.v1",
        "source_meta_sha256": _sha256(source.read_bytes()),
        "training_meta_sha256": _sha256(metadata.read_bytes()),
        "timestamp_semantics": sidecar["timeline"],
        "camera_ids": sorted(cameras),
        "frame_count": len(frames),
        "frames": frames,
        "retained_stores": {path.name: _file_sha256(path) for path in retained},
    }
    evidence.write_text(json.dumps(payload, indent=2) + "\n")


def _retained_stores(reader):
    metadata = reader.get_sequence_meta().to_dict()
    retained = []
    paths = {
        Path(str(path)).name: Path(str(path)) for path in reader.component_store_paths
    }
    for store in metadata["component_stores"]:
        components = store["components"]
        if "cameras" in components:
            if set(components) != {"cameras"}:
                raise NurecError(
                    "photographic camera stores must not co-locate other components"
                )
        else:
            retained.append(paths[store["path"]])
    return retained


def _plan_frames(reader, cameras, trajectories):
    frames = []
    for camera_id, camera in sorted(cameras.items()):
        frames.extend(_photograph_frames(camera_id, camera, trajectories))
    interval = reader.sequence_timestamp_interval_us
    start, stop = int(interval.start), int(interval.stop)
    if len(frames) < 2 or stop - start < len(frames):
        raise NurecError("sequence interval cannot hold unique photographic timestamps")
    for index, frame in enumerate(frames):
        frame["training_timestamp_us"] = start + index * (stop - start - 1) // (
            len(frames) - 1
        )
    return frames


def _photograph_frames(camera_id, camera, trajectories):
    from npa.workbench.nurec.capture_trajectory import _exact_frame_poses
    from npa.workbench.nurec.ncore_rig import _frame_timestamps

    timestamps = _frame_timestamps(camera_id, camera)
    intervals = np.asarray(camera.frames_timestamps_us)
    if not np.array_equal(intervals[:, 0], timestamps):
        raise NurecError("photographic retiming requires instantaneous source frames")
    if camera_id not in trajectories:
        raise NurecError(f"photographic capture has no poses for {camera_id!r}")
    poses = _exact_frame_poses(camera_id, trajectories[camera_id], intervals)[:, 0]
    return list(
        {
            "camera_id": camera_id,
            "frame_index": index,
            "source_timestamp_us": int(timestamp),
            "T_camera_world": pose.tolist(),
        }
        for index, (timestamp, pose) in enumerate(zip(timestamps, poses, strict=True))
    )


def _write_cameras(writer, cameras, frames):
    from ncore.data.v4 import CameraSensorComponent

    for camera_id, camera in sorted(cameras.items()):
        output = writer.register_component_writer(
            CameraSensorComponent.Writer,
            component_instance_name=camera_id,
            group_name=f"photographic-{camera_id}",
            generic_meta_data=camera.generic_meta_data,
        )
        output.set_generic_data(
            {
                name: camera.get_generic_data(name)
                for name in camera.get_generic_data_names()
            },
            camera.generic_meta_data,
        )
        for frame in frames:
            if frame["camera_id"] != camera_id:
                continue
            _copy_frame(output, camera, frame)


def _copy_frame(output, camera, frame):
    timestamp = frame["source_timestamp_us"]
    image = camera.get_frame_data(timestamp)
    data = image.get_encoded_image_data()
    frame["image_sha256"] = _sha256(data)
    output.store_frame(
        image_binary_data=data,
        image_format=image.get_encoded_image_format(),
        frame_timestamps_us=np.full(2, frame["training_timestamp_us"], dtype=np.uint64),
        generic_data={
            name: camera.get_frame_generic_data(timestamp, name)
            for name in camera.get_frame_generic_data_names(timestamp)
        },
        generic_meta_data=camera.get_frame_generic_meta_data(timestamp),
    )


def _write_poses(writer, reader, cameras, frames, sidecar):
    from ncore.data.v4 import PosesComponent

    output = writer.register_component_writer(
        PosesComponent.Writer,
        component_instance_name=POSES_GROUP,
        group_name=POSES_GROUP,
        generic_meta_data={"derivation": "one virtual rig pose per photograph"},
    )
    poses = np.asarray([frame["T_camera_world"] for frame in frames], dtype=np.float32)
    timestamps = np.asarray(
        [frame["training_timestamp_us"] for frame in frames], dtype=np.uint64
    )
    output.store_dynamic_pose(
        source_frame_id="rig",
        target_frame_id="world",
        poses=poses,
        timestamps_us=timestamps,
        require_sequence_time_coverage=False,
    )
    for camera_id in sorted(cameras):
        output.store_static_pose(
            source_frame_id=camera_id,
            target_frame_id="rig",
            pose=np.eye(4, dtype=np.float32),
        )
    _copy_other_poses(output, reader, cameras, sidecar)


def _copy_other_poses(output, reader, cameras, sidecar):
    from ncore.data.v4 import PosesComponent

    original = reader.open_component_readers(PosesComponent.Reader)[
        sidecar["poses_component_group"]
    ]
    excluded = {"rig", *cameras}
    for (source, target), (values, times) in original.get_dynamic_poses():
        if not excluded.intersection((source, target)):
            output.store_dynamic_pose(
                source_frame_id=source,
                target_frame_id=target,
                poses=values,
                timestamps_us=times,
                require_sequence_time_coverage=False,
            )
    for (source, target), value in original.get_static_poses():
        if not excluded.intersection((source, target)):
            output.store_static_pose(
                source_frame_id=source, target_frame_id=target, pose=value
            )


def _combine_stores(source, target, retained, generated):
    from ncore.data.v4 import SequenceComponentGroupsReader
    from upath import UPath

    for path in retained:
        (target / path.name).symlink_to(path.resolve())
    reader = SequenceComponentGroupsReader(
        [UPath(path) for path in [*retained, *generated]]
    )
    metadata = reader.get_sequence_meta().to_dict()
    if any(
        Path(store["path"]).name != store["path"]
        for store in metadata["component_stores"]
    ):
        raise NurecError("photographic metadata contains nonportable store paths")
    path = target / source.name
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def _verify_roundtrip(metadata, original_cameras, frames):
    from ncore.data.v4 import CameraSensorComponent, PosesComponent
    from npa.workbench.nurec.ncore_rig import _open_reader

    reader = _open_reader(metadata)
    cameras = reader.open_component_readers(CameraSensorComponent.Reader)
    if set(cameras) != set(original_cameras):
        raise NurecError("photographic preparation changed the camera inventory")
    poses = reader.open_component_readers(PosesComponent.Reader)[POSES_GROUP]
    rig, times = dict(poses.get_dynamic_poses())[("rig", "world")]
    static = dict(poses.get_static_poses())
    for index, frame in enumerate(frames):
        _verify_frame(cameras[frame["camera_id"]], frame, rig[index], times[index])
    for camera_id in cameras:
        if not np.array_equal(static[(camera_id, "rig")], np.eye(4)):
            raise NurecError(
                "photographic training camera has a non-identity rig transform"
            )
    for name, camera in cameras.items():
        if camera.frames_count != original_cameras[name].frames_count:
            raise NurecError("photographic preparation changed the frame inventory")


def _verify_frame(camera, frame, pose, timestamp):
    image = camera.get_frame_data(frame["training_timestamp_us"])
    if _sha256(image.get_encoded_image_data()) != frame["image_sha256"]:
        raise NurecError("photographic preparation changed encoded image bytes")
    if int(timestamp) != frame["training_timestamp_us"] or not np.allclose(
        pose, frame["T_camera_world"], atol=1e-5, rtol=0
    ):
        raise NurecError("photographic training rig differs from a source camera pose")


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
