"""Join independently valid cleanup and visual receipts to the qualified run."""

import hashlib
from pathlib import Path

from image_byte_scan import core as W, prepare as P
from npa.workbench.nurec.qualification_cleanup import _workflow_jobs
from npa.workbench.nurec.qualification_readback import _canonical_sha

from . import vlm_evidence as V
from .process import file_sha


def _json(path):
    return W.bound_json(P.binding(path))


def cleanup_run(cleanup, readback, status_path, build_path):
    """Require cleanup of this build, run, managed attempts and storage prefix.

    Args:
        cleanup: Independently validated cleanup receipt.
        readback: Complete qualification readback receipt.
        status_path: Exact final workflow status from that readback.
        build_path: Build receipt already checked by prepublication.
    Returns:
        None.
    Raises:
        ValueError: A receipt belongs to another qualification or is incomplete.
    """
    status = _json(status_path)
    run_id = status.get("run_id")
    W.require(type(run_id) is str and run_id, "acceptance_cleanup_run_id")
    jobs, status_sha, unlaunched = _workflow_jobs(status_path, run_id)
    W.require(
        cleanup.get("workflow_status_sha256") == status_sha
        and cleanup.get("run_id_sha256") == hashlib.sha256(run_id.encode()).hexdigest()
        and cleanup.get("managed_job_identities_sha256") == _canonical_sha(jobs)
        and cleanup.get("managed_jobs") == len(jobs)
        and cleanup.get("proven_unlaunched_states") == list(unlaunched)
        and cleanup.get("build_receipt_sha256") == file_sha(build_path)
        and isinstance(readback.get("prefix_sha256"), str)
        and cleanup.get("storage_prefix_sha256") == readback["prefix_sha256"],
        "acceptance_cleanup_qualification_binding",
    )


def _qualified_frames(evidence_root, final_manifest):
    render_root = evidence_root / "readback/novel_views"
    cameras = [
        path.relative_to(render_root).as_posix()
        for path in render_root.rglob("*")
        if path.is_dir()
        and not path.is_symlink()
        and hashlib.sha256(
            path.relative_to(render_root).as_posix().encode()
        ).hexdigest()
        == final_manifest.get("render_camera_sha256")
    ]
    W.require(len(cameras) == 1, "acceptance_visual_camera_binding")
    _, frames = V._render_trajectory(render_root, cameras[0])
    readback = _json(evidence_root / "qualification-readback.json")
    inventory = {item["path"]: item for item in readback.get("local_inventory", [])}
    for path in frames:
        relative = path.relative_to(evidence_root / "readback").as_posix()
        W.require(
            inventory.get(relative)
            == {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha(path),
            },
            "acceptance_visual_readback_binding",
        )
    return frames


def visual_run(manifest, evidence_root):
    """Bind the frozen VLM source and sampled pixels to qualified render bytes.

    Args:
        manifest: Qualified publication manifest.
        evidence_root: Private qualification directory containing readback and VLM.
    Returns:
        None.
    Raises:
        ValueError: Source, trajectory, original bytes or normalized pixels differ.
    """
    root = evidence_root / "vlm"
    freeze = _json(root / "freeze.json")
    final = _json(root / "final-frame-manifest.json")
    W.require(
        freeze.get("source_archive_sha256")
        == manifest["conversion"]["source_archive_sha256"],
        "acceptance_visual_source_binding",
    )
    frames = _qualified_frames(evidence_root, final)
    inventory = [
        {"sha256": file_sha(path), "bytes": path.stat().st_size} for path in frames
    ]
    W.require(
        freeze.get("render_inventory_sha256")
        == V._sha_bytes(V._canonical(inventory))
        == final.get("render_inventory_sha256"),
        "acceptance_visual_inventory_binding",
    )
    _selected_pixels(root, frames, final.get("frames"))


def _selected_pixels(root: Path, frames, records):
    indices = V._indices(len(frames))
    W.require(
        type(records) is list and len(records) == len(indices),
        "acceptance_visual_frame_population",
    )
    for order, (index, record) in enumerate(zip(indices, records, strict=True)):
        body, _ = V._image_bytes(frames[index])
        W.require(
            record.get("order") == order
            and record.get("source_index") == index
            and record.get("source_sha256") == file_sha(frames[index])
            and record.get("sha256") == V._sha_bytes(body)
            and record.get("bytes") == len(body)
            and file_sha(root / record["path"]) == V._sha_bytes(body),
            "acceptance_visual_selected_frame_binding",
        )
