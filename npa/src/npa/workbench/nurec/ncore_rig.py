"""Derive the ``rig -> world`` pose edge an NRE NCore dataset needs.

NRE's NCore data source requires a ``("rig", "world")`` pose-graph edge to
determine scene extent::

    # nre/datasets/ncore.py (nre-ga 26.04)
    # TODO: frame-pose only data might fail here as there are no rig poses and
    #       might require refined logic
    rig_world_edge = unpack_optional(
        sequence_loader.pose_graph.get_edge("rig", "world"),
        msg="Rig-to-world poses are currently required to determine scene extend")

Object-centric photographic captures do not have a vehicle rig, so NVIDIA's own
COLMAP -> NCore converter stores per-camera ``("<camera>", "world")`` poses and no
rig node at all (``tools/data_converter/colmap/converter.py``: ``reference_frame
= "world"``). Every such sequence - including the NCore export shipped in
``nvidia/PhysicalAI-NuRec-PPISP`` - therefore fails to load in NRE 26.04.

For a **single-camera** capture the fix is exact rather than approximate: the
rig *is* the camera, so ``rig -> world`` is precisely that camera's pose
trajectory. This module reads an existing NCore V4 sequence with the public
``nvidia-ncore`` library, writes ONE extra component store holding that derived
edge, and emits a new sequence meta-file that references the original stores plus
the new one. The source data is never modified.

Requires ``nvidia-ncore`` (Apache-2.0, https://github.com/NVIDIA/ncore); the
import is local to each function so the rest of the workbench tool stays
importable without it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npa.workbench.nurec.nurec import NurecError

#: Frame name NRE looks for. Not configurable upstream.
RIG_FRAME = "rig"
WORLD_FRAME = "world"
#: NCore V4 poses *component instance* name for the derived edge set. It must not
#: be ``default``: ``open_component_readers`` asserts that an instance name occurs
#: only once across a sequence's stores, so re-using ``default`` collides with the
#: sequence's own poses component. NRE selects it with
#: ``dataset.poses_component_group=<this>``, which is why the derived component
#: carries a COMPLETE copy of the original edges plus the rig edge — selecting a
#: group replaces the pose set rather than merging with it.
DERIVED_POSES_GROUP = "npa_rig"
#: Store-file suffix for the derived store: ``<sequence>.ncore4-<group>.zarr.itar``.
DERIVED_GROUP_NAME = "npa_rig"
#: Sidecar written next to the derived meta-file. It makes the sequence
#: SELF-DESCRIBING: whoever consumes it later -- including a stage in a different
#: pod that only received an S3 prefix -- can recover which camera became the rig
#: and which poses group to select, without re-opening the zarr stores or needing
#: nvidia-ncore installed. ``publish_ncore_sequence`` uploads it automatically
#: because it lives in the sequence directory.
RIG_SIDECAR_NAME = "npa-rig.json"


@dataclass(frozen=True)
class RigPoseResult:
    """Outcome of deriving the rig edge for one NCore sequence."""

    ok: bool
    sequence_id: str
    source_meta: str
    output_meta: str
    reference_camera: str
    pose_count: int
    cameras: tuple[str, ...] = ()
    already_present: bool = False
    poses_component_group: str = ""
    copied_dynamic_edges: tuple[str, ...] = ()
    copied_static_edges: tuple[str, ...] = ()
    store_paths: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "sequence_id": self.sequence_id,
            "source_meta": self.source_meta,
            "output_meta": self.output_meta,
            "reference_camera": self.reference_camera,
            "pose_count": self.pose_count,
            "cameras": list(self.cameras),
            "already_present": self.already_present,
            "poses_component_group": self.poses_component_group,
            "copied_dynamic_edges": list(self.copied_dynamic_edges),
            "copied_static_edges": list(self.copied_static_edges),
            "store_paths": list(self.store_paths),
            "errors": list(self.errors),
        }


def camera_world_trajectories(ncore_json: Path | str) -> dict[str, tuple[Any, Any]]:
    """Return ``{camera_id: (poses[N,4,4], timestamps_us[N])}`` for ``<cam> -> world``.

    Reads the sequence's ``poses`` components and keeps every dynamic edge whose
    target frame is ``world``, which is how the COLMAP converter stores camera
    extrinsics.
    """
    reader = _open_reader(ncore_json)
    trajectories: dict[str, tuple[Any, Any]] = {}
    from ncore.data.v4 import PosesComponent

    for poses_reader in reader.open_component_readers(PosesComponent.Reader).values():
        for (source, target), (poses, timestamps) in poses_reader.get_dynamic_poses():
            if target == WORLD_FRAME and source != RIG_FRAME:
                trajectories[str(source)] = (poses, timestamps)
    return trajectories


def has_rig_edge(ncore_json: Path | str) -> bool:
    """True when the sequence already carries a ``rig -> world`` (or inverse) edge."""
    reader = _open_reader(ncore_json)
    from ncore.data.v4 import PosesComponent

    for poses_reader in reader.open_component_readers(PosesComponent.Reader).values():
        for (source, target), _payload in poses_reader.get_dynamic_poses():
            if {source, target} == {RIG_FRAME, WORLD_FRAME}:
                return True
        for (source, target), _pose in poses_reader.get_static_poses():
            if {source, target} == {RIG_FRAME, WORLD_FRAME}:
                return True
    return False


def select_reference_camera(
    trajectories: dict[str, tuple[Any, Any]], preferred: str = ""
) -> str:
    """Pick the camera whose trajectory becomes the rig trajectory.

    ``preferred`` wins when it has poses; otherwise the camera with the most
    poses is used (the longest, best-constrained trajectory), ties broken by id
    so the choice is deterministic.
    """
    if not trajectories:
        raise NurecError(
            "sequence has no '<camera> -> world' dynamic poses; cannot derive a rig frame"
        )
    if preferred:
        if preferred not in trajectories:
            raise NurecError(
                f"reference camera {preferred!r} has no '-> world' poses "
                f"(available: {sorted(trajectories)})"
            )
        return preferred
    return sorted(trajectories, key=lambda name: (-len(trajectories[name][1]), name))[0]


def derive_rig_poses(
    ncore_json: Path | str,
    *,
    output_dir: Path | str,
    reference_camera: str = "",
    sequence_meta_name: str = "",
) -> RigPoseResult:
    """Write a derived NCore sequence that adds the ``rig -> world`` edge.

    The original stores are left untouched and referenced in place by the new
    meta-file, so this costs one small extra store rather than a copy of the
    multi-hundred-megabyte camera shards.
    """
    source = Path(ncore_json)
    target_dir = Path(output_dir)
    if not source.is_file():
        return RigPoseResult(
            ok=False,
            sequence_id="",
            source_meta=str(source),
            output_meta="",
            reference_camera="",
            pose_count=0,
            errors=(f"NCore sequence meta-file not found: {source}",),
        )

    try:
        reader = _open_reader(source)
        sequence_id = str(reader.sequence_id)
        trajectories = camera_world_trajectories(source)
        if has_rig_edge(source):
            return RigPoseResult(
                ok=True,
                sequence_id=sequence_id,
                source_meta=str(source),
                output_meta=str(source),
                reference_camera="",
                pose_count=0,
                cameras=tuple(sorted(trajectories)),
                already_present=True,
            )
        chosen = select_reference_camera(trajectories, preferred=reference_camera)
        poses, timestamps = trajectories[chosen]
    except NurecError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface library/IO failures as a result
        return RigPoseResult(
            ok=False,
            sequence_id="",
            source_meta=str(source),
            output_meta="",
            reference_camera="",
            pose_count=0,
            errors=(f"failed to read NCore sequence: {exc}",),
        )

    if len(timestamps) < 2:
        return RigPoseResult(
            ok=False,
            sequence_id=sequence_id,
            source_meta=str(source),
            output_meta="",
            reference_camera=chosen,
            pose_count=int(len(timestamps)),
            cameras=tuple(sorted(trajectories)),
            errors=(
                "a rig trajectory needs at least two poses to be interpolatable; "
                f"camera {chosen!r} has {len(timestamps)}",
            ),
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from ncore.data.v4 import (
            PosesComponent,
            SequenceComponentGroupsReader,
            SequenceComponentGroupsWriter,
        )

        writer = SequenceComponentGroupsWriter.from_reader(
            output_dir_path=_upath(target_dir),
            store_base_name=sequence_id,
            sequence_reader=reader,
            store_type="itar",
        )
        poses_writer = writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name=DERIVED_POSES_GROUP,
            group_name=DERIVED_GROUP_NAME,
            generic_meta_data={
                "derived_by": "npa.workbench.nurec.ncore_rig",
                "derivation": (
                    "rig == reference camera; rig->world is that camera's "
                    "sensor-to-world trajectory"
                ),
                "reference_camera": chosen,
            },
        )
        # Selecting a poses component group REPLACES the pose set, so every original
        # edge is copied across before the derived rig edge is appended. A
        # `<camera> -> rig` static identity is deliberately NOT written: the copied
        # `<camera> -> world` edges already position the cameras, and adding both
        # would give the pose graph two routes between the same frames.
        # NCore stores each edge in ONE direction (store_dynamic_pose refuses a pair
        # whose inverse already exists), so copying every edge verbatim cannot
        # duplicate a relation.
        copied_dynamic: list[str] = []
        copied_static: list[str] = []
        for source_reader in reader.open_component_readers(PosesComponent.Reader).values():
            for (edge_from, edge_to), (edge_poses, edge_ts) in source_reader.get_dynamic_poses():
                if {edge_from, edge_to} == {RIG_FRAME, WORLD_FRAME}:
                    continue
                poses_writer.store_dynamic_pose(
                    source_frame_id=edge_from,
                    target_frame_id=edge_to,
                    poses=edge_poses,
                    timestamps_us=edge_ts,
                    require_sequence_time_coverage=False,
                )
                copied_dynamic.append(f"{edge_from}->{edge_to}")
            for (edge_from, edge_to), edge_pose in source_reader.get_static_poses():
                poses_writer.store_static_pose(
                    source_frame_id=edge_from,
                    target_frame_id=edge_to,
                    pose=edge_pose,
                )
                copied_static.append(f"{edge_from}->{edge_to}")
        poses_writer.store_dynamic_pose(
            source_frame_id=RIG_FRAME,
            target_frame_id=WORLD_FRAME,
            poses=poses,
            timestamps_us=timestamps,
            require_sequence_time_coverage=False,
        )
        derived_paths = [Path(str(path)) for path in writer.finalize()]

        combined = [Path(str(path)) for path in reader.component_store_paths] + derived_paths
        merged_reader = SequenceComponentGroupsReader([_upath(path) for path in combined])
        meta = merged_reader.get_sequence_meta().to_dict()
        # get_sequence_meta records each store by BASENAME and the meta-file resolves
        # them relative to its own directory. Assert it rather than trust it: a
        # library change to absolute paths would silently produce a meta-file that
        # only resolves on the machine that wrote it, which would then fail in a
        # different pod after the S3 handoff.
        recorded = [str(store.get("path", "")) for store in meta.get("component_stores", [])]
        unexpected = [name for name in recorded if "/" in name or not name]
        if unexpected:
            raise NurecError(
                "NCore wrote non-basename component-store paths "
                f"{unexpected}; the derived meta-file would not resolve elsewhere"
            )
    except Exception as exc:  # noqa: BLE001
        return RigPoseResult(
            ok=False,
            sequence_id=sequence_id,
            source_meta=str(source),
            output_meta="",
            reference_camera=chosen,
            pose_count=int(len(timestamps)),
            cameras=tuple(sorted(trajectories)),
            errors=(f"failed to write derived rig poses: {exc}",),
        )

    # The meta-file records store paths by BASENAME and resolves them relative to
    # its own directory, so the original shards are symlinked (copied when links
    # are unavailable) next to the derived store instead of being duplicated.
    for store_path in combined:
        link = target_dir / store_path.name
        if link.exists() or link.is_symlink():
            continue
        try:
            link.symlink_to(store_path.resolve())
        except OSError:
            shutil.copy2(store_path, link)

    meta_path = target_dir / (sequence_meta_name or f"{sequence_id}.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (target_dir / RIG_SIDECAR_NAME).write_text(
        json.dumps(
            {
                "derived_by": "npa.workbench.nurec.ncore_rig",
                "sequence_id": sequence_id,
                "reference_camera": chosen,
                "poses_component_group": DERIVED_POSES_GROUP,
                "pose_count": int(len(timestamps)),
                "cameras": sorted(trajectories),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return RigPoseResult(
        ok=True,
        sequence_id=sequence_id,
        source_meta=str(source),
        output_meta=str(meta_path),
        reference_camera=chosen,
        pose_count=int(len(timestamps)),
        cameras=tuple(sorted(trajectories)),
        poses_component_group=DERIVED_POSES_GROUP,
        copied_dynamic_edges=tuple(copied_dynamic),
        copied_static_edges=tuple(copied_static),
        store_paths=tuple(str(path) for path in derived_paths),
    )


def _open_reader(ncore_json: Path | str) -> Any:
    try:
        from ncore.data.v4 import SequenceComponentGroupsReader
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise NurecError(
            "nvidia-ncore is required to derive NCore rig poses "
            "(pip install nvidia-ncore)"
        ) from exc
    return SequenceComponentGroupsReader([_upath(Path(ncore_json))])


def _upath(path: Path) -> Any:
    from upath import UPath

    return UPath(path)
