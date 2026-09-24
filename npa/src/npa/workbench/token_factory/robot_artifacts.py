"""Publish accepted LeRobot episodes and a video gallery bound to their physics records."""

from __future__ import annotations

import hashlib
from html import escape
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from npa.adapter.sim_to_lerobot import convert
from npa.clients.storage import StorageClient
from .robot_sim import ACTION_NAMES, FPS, JOINT_NAMES


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def export_robot_dataset(root, records):
    """Convert accepted physical episodes through the existing LeRobot v3 adapter.

    Args:
        root: Run directory with recorded per-episode arrays.
        records: Scene and physics provenance, including rejected episodes.
    Returns:
        Number of accepted training episodes; zero leaves no dataset.
    Raises:
        AdapterError: Video encoding or dataset conversion fails.
        OSError: Artifact writing fails.
    """
    accepted = [record for record in records if record["status"] == "accepted"]
    if not accepted:
        return 0
    with TemporaryDirectory(prefix="npa-robot-demos-") as temporary:
        demos = Path(temporary)
        tasks = []
        for index, record in enumerate(accepted):
            (demos / f"episode_{index:04d}").symlink_to(
                (root / record["episode_path"]).resolve(), target_is_directory=True
            )
            record["dataset_episode_index"] = index
            tasks.append({"episode_index": index, "task": record["simulation"]["task"]})
        _write_json(demos / "metadata.json", {"episodes": tasks})
        convert(
            demos,
            root / "dataset",
            fps=FPS,
            robot_type="fetch",
            task_from_metadata=True,
        )
    info_path = root / "dataset/meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"]["names"] = JOINT_NAMES
    info["features"]["action"]["names"] = ACTION_NAMES
    _write_json(info_path, info)
    return len(accepted)


def _episode_card(record):
    identifier, status = escape(record["id"]), escape(record["status"])
    if "simulation" not in record:
        return f"<article><h2>{identifier}</h2><p>{status}: scene generation failed</p></article>"
    result = record["simulation"]
    video = escape(record["episode_path"] + "/preview.mp4", quote=True)
    checks = " · ".join(
        f"{escape(name)}: {'pass' if value else 'FAIL'}"
        for name, value in result["checks"].items()
    )
    return (
        f"<article><h2>{identifier} <small>{status}</small></h2>"
        f'<p>{escape(result["task"])}</p><video controls preload="metadata" src="{video}"></video>'
        '<p class="note">Left: workspace camera. Right: moving wrist camera. Both show the same simulation timestep.</p>'
        f"<p>Planner: {escape(record['served_model'])} · {result['frames']} frames at {result['fps']} Hz</p>"
        f"<p>Lift: {result['lift_height_m']:.3f} m · Final target error: {result['final_distance_m']:.4f} m</p>"
        f"<details><summary>Physics checks and route</summary><p>{checks}</p>"
        f"<p>{escape(record['routing']['reason'])}</p></details></article>"
    )


def write_robot_gallery(root, records):
    """Create a local HTML gallery that plays actual simulator camera recordings.

    Args:
        root: Run output directory.
        records: Completed provenance records.
    Returns:
        None; writes index.html with relative video paths.
    Raises:
        OSError: The gallery cannot be written.
    """
    cards = "\n".join(_episode_card(record) for record in records)
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot SDG — recorded pick-and-place episodes</title>
<style>body{font:16px/1.6 system-ui;background:#f1f5f6;color:#172e35;margin:0}
main{max-width:1120px;margin:auto;padding:32px 20px}article{background:white;padding:24px;border-radius:14px;margin:24px 0}
video{display:block;width:100%;background:#16272b}h1{line-height:1.2}small,.note{font-size:13px;color:#526775}
small{background:#e9eff1;padding:5px 10px;border-radius:5px}summary{cursor:pointer}a{color:#006b67}</style>
<main><h1>Robot demonstrations generated in simulation</h1>
<p>Token Factory routes scene planning to hosted open-weight models. A scripted Fetch controller produces camera frames, joint states and actions in MuJoCo. Simulator state and contacts determine acceptance. Accepted episodes are exported as LeRobot v3.</p>
<p><a href="report.json">Run manifest</a> · <a href="provenance.jsonl">Routing and physics records</a> · <a href="dataset/meta/info.json">Dataset metadata</a></p>
"""
        + cards
        + "</main></html>\n"
    )
    (root / "index.html").write_text(page)


def robot_artifact_hashes(root):
    """Hash all delivered data, videos and provenance before writing the manifest.

    Args:
        root: Run output directory without report.json.
    Returns:
        Relative artifact names mapped to byte lengths and SHA-256 digests.
    Raises:
        OSError: An artifact cannot be read.
    """
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = _file_digest(path)
            result[path.relative_to(root).as_posix()] = {
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
    return result


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_robot_run(root, output_path, report):
    """Publish complete run artifacts, writing the manifest last.

    Args:
        root: Local directory containing generated artifacts.
        output_path: The same local directory or an empty S3 destination prefix.
        report: Final manifest.
    Returns:
        None.
    Raises:
        StorageError: Destination is occupied or upload fails.
        OSError: Manifest writing fails.
    """
    client = None
    if output_path.startswith("s3://"):
        client = StorageClient.from_environment()
        client.upload_directory(str(root), output_path, require_empty=True)
    _write_json(root / "report.json", report)
    if client is not None:
        client.upload_file(
            str(root / "report.json"), output_path.rstrip("/") + "/report.json"
        )
