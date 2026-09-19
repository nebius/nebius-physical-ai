"""Shared Open3D operations for CLI, SDK and SkyPilot stages.

Each operation resolves its inputs, hands local files to the isolated runner,
validates what came back against the published contract, and only then publishes
artifacts with a read-after-write digest check. A stage that cannot prove its own
output does not get to report success.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.workbench.dataset.storage import read_bytes_uri, uri_join, write_bytes_uri
from npa.workbench.storage_scope import authorize_uri

from .artifacts import (
    PAIRS_JOURNAL,
    Open3dError,
    canonical,
    read_journal,
    sha256_bytes,
    summarize,
    validate_mesh,
    validate_pose_graph,
    validate_result,
    validate_support,
    verify_rerun_recording,
)
from .schemas import (
    POINT_CLOUD_SUFFIXES,
    Fragment,
    PrepareRequest,
    ReconstructRequest,
    RegistrationManifest,
    RunRequest,
    StageDemoRequest,
    fragment_id_for,
)

_LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
POSE_GRAPH_FILENAME = "pose_graph.json"
FUSED_FILENAME = "fused.ply"
MESH_FILENAME = "mesh.ply"
UNCROPPED_MESH_FILENAME = "mesh_uncropped.ply"
RECORDING_FILENAME = "point_cloud.rrd"
RESULT_FILENAME = "result.json"
#: Maximum fragments one prefix may contribute. Multiway registration is
#: O(n^2) in pairwise registrations, so an accidentally broad prefix would
#: otherwise turn into an unbounded run.
MAX_FRAGMENTS = 32


def _tool_paths(request: RunRequest) -> None:
    validate_read_path(request.input_path, tool="open3d", allow_hf=False)
    validate_write_path(request.output_path, tool="open3d", required=True)
    authorize_uri(request.input_path, operation="read")
    authorize_uri(request.output_path, operation="write")


def _publish(uri: str, payload: bytes) -> None:
    write_bytes_uri(uri, payload)
    if sha256_bytes(read_bytes_uri(uri)) != sha256_bytes(payload):
        raise Open3dError("artifact S3 read-after-write digest mismatch")


def _work_dir(prefix: str) -> Path:
    root = Path(
        tempfile.mkdtemp(prefix=prefix, dir=os.environ.get("NPA_OPEN3D_WORK_DIR"))
    )
    root.chmod(0o700)
    return root


def _discard(root: Path) -> None:
    """Artifacts are already durable; a cleanup failure must not fail the run."""

    try:
        shutil.rmtree(root)
    except OSError:
        _LOGGER.warning("Open3D artifacts verified; local working-file cleanup failed")


def _run_runner(kind: str, payload: dict[str, Any], root: Path, run_id: str) -> Path:
    """Invoke the isolated runner and keep its log next to the inputs on failure."""

    output = root / "output"
    (root / "input.json").write_bytes(canonical(payload))
    command = [
        os.environ.get("NPA_OPEN3D_PYTHON", sys.executable),
        "-m",
        "npa.workbench.open3d.runner",
        "--kind",
        kind,
        "--input",
        str(root / "input.json"),
        "--output",
        str(output),
        "--run-id",
        run_id,
    ]
    with (root / "runtime.log").open("wb") as log:
        completed = subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        tail = _log_tail(root / "runtime.log")
        raise Open3dError(
            f"Open3D {kind} failed with exit code {completed.returncode}; retained "
            f"local log at {root / 'runtime.log'}{tail}"
        )
    return output


def _log_tail(path: Path, *, lines: int = 12) -> str:
    """Surface the actual upstream error, so triage does not need the pod back.

    A stage that fails with only an exit code forces a rerun to learn anything,
    and the rerun is the expensive part.
    """

    try:
        text = path.read_text(errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    tail = "\n".join(text.splitlines()[-lines:])
    return f"\nlast {lines} log lines:\n{tail}"


def _list_point_clouds(prefix: str) -> list[str]:
    """List readable point-cloud URIs directly under a prefix, sorted by name."""

    target = authorize_uri(prefix.rstrip("/") + "/", operation="read")
    if target.kind == "s3":
        import boto3
        from botocore.config import Config as BotoConfig

        client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("NEBIUS_S3_ENDPOINT")
            or None,
            config=BotoConfig(signature_version="s3v4"),
        )
        base = f"s3://{target.bucket}"
        keys: list[str] = []
        token: str | None = None
        while True:
            page = client.list_objects_v2(
                Bucket=target.bucket,
                Prefix=target.key,
                Delimiter="/",
                **({"ContinuationToken": token} if token else {}),
            )
            keys.extend(item["Key"] for item in page.get("Contents", ()))
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated"):
                break
        found = [f"{base}/{key}" for key in keys]
    else:
        assert target.local_path is not None
        directory = target.local_path
        if not directory.is_dir():
            raise Open3dError(f"{prefix} is not a directory of point clouds")
        found = [str(entry) for entry in directory.iterdir() if entry.is_file()]
    clouds = sorted(uri for uri in found if uri.lower().endswith(POINT_CLOUD_SUFFIXES))
    if len(clouds) < 2:
        raise Open3dError(
            f"{prefix} holds {len(clouds)} readable point cloud(s); registration "
            "needs at least 2 files ending in " + " or ".join(POINT_CLOUD_SUFFIXES)
        )
    if len(clouds) > MAX_FRAGMENTS:
        raise Open3dError(
            f"{prefix} holds {len(clouds)} point clouds; at most {MAX_FRAGMENTS} "
            "may be registered in one run"
        )
    return clouds


def _fragment_for(uri: str) -> Fragment:
    payload = read_bytes_uri(uri)
    return Fragment(
        id=fragment_id_for(uri),
        uri=uri,
        sha256=sha256_bytes(payload),
        bytes=len(payload),
    )


def prepare(request: PrepareRequest) -> dict[str, Any]:
    """Index operator scans under a prefix into a registration manifest."""

    validate_write_path(request.output_path, tool="open3d", required=True)
    authorize_uri(request.output_path, operation="write")
    validate_read_path(request.input_path, tool="open3d", allow_hf=False)
    fragments = [_fragment_for(uri) for uri in _list_point_clouds(request.input_path)]
    return _publish_manifest(request, fragments, demo_release=None)


def stage_demo(request: StageDemoRequest) -> dict[str, Any]:
    """Publish the upstream `open3d.data` fragments as this run's input scans.

    These are real captured indoor scans from the upstream release, not generated
    geometry, which is why staging them is an acceptable self-contained input and
    why the shipped reference spec can start from nothing.
    """

    validate_write_path(request.output_path, tool="open3d", required=True)
    authorize_uri(request.output_path, operation="write")
    root = _work_dir("npa-open3d-stage-demo-")
    try:
        output = _run_runner("stage-demo", {}, root, "stage-demo")
        staged = json.loads((output / "runner.json").read_text())
        fragments = []
        for entry in staged["fragments"]:
            local = output / "fragments" / entry["filename"]
            uri = uri_join(request.output_path, "fragments", entry["filename"])
            _publish(uri, local.read_bytes())
            fragments.append(
                Fragment(
                    id=entry["id"],
                    uri=uri,
                    sha256=entry["sha256"],
                    bytes=entry["bytes"],
                )
            )
        result = _publish_manifest(
            request, fragments, demo_release=staged["demo_data_release"]
        )
        result["staged_points"] = {
            entry["id"]: entry["points"] for entry in staged["fragments"]
        }
        return result
    finally:
        _discard(root)


def _publish_manifest(
    request: StageDemoRequest, fragments: list[Fragment], *, demo_release: str | None
) -> dict[str, Any]:
    manifest = RegistrationManifest(
        fragments=fragments,
        voxel_size=request.voxel_size,
        icp_estimation=request.icp_estimation,
        demo_data_release=demo_release,
    )
    body = canonical(manifest.model_dump(mode="json"))
    _publish(uri_join(request.output_path, MANIFEST_FILENAME), body)
    return {
        "schema_version": "npa.open3d.prepared.v1",
        "output_path": request.output_path,
        "manifest_sha256": sha256_bytes(body),
        "fragment_count": len(fragments),
        "fragment_ids": [fragment.id for fragment in fragments],
        "voxel_size": manifest.voxel_size,
        "icp_estimation": manifest.icp_estimation,
        "demo_data_release": demo_release,
    }


def _load_manifest(input_path: str) -> RegistrationManifest:
    try:
        payload = json.loads(read_bytes_uri(uri_join(input_path, MANIFEST_FILENAME)))
    except ValueError as exc:
        raise Open3dError(
            f"{uri_join(input_path, MANIFEST_FILENAME)} is not valid JSON; run "
            "`npa workbench open3d prepare` against this prefix first"
        ) from exc
    return RegistrationManifest.model_validate(payload)


def _download_fragments(manifest: RegistrationManifest, root: Path) -> dict[str, str]:
    """Fetch each scan and hold it to the digest the manifest recorded."""

    directory = root / "fragments"
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for fragment in manifest.fragments:
        payload = read_bytes_uri(fragment.uri)
        if sha256_bytes(payload) != fragment.sha256:
            raise Open3dError(
                f"fragment {fragment.id} no longer matches the digest recorded at "
                "prepare time; the manifest and the stored scan disagree"
            )
        suffix = Path(urlparse(fragment.uri).path or fragment.uri).suffix
        local = directory / f"{fragment.id}{suffix}"
        local.write_bytes(payload)
        paths[fragment.id] = str(local)
    return paths


def _registration(kind: str, request: RunRequest) -> dict[str, Any]:
    _tool_paths(request)
    manifest = _load_manifest(request.input_path)
    root = _work_dir(f"npa-open3d-{kind}-")
    try:
        started = time.perf_counter()
        output = _run_runner(
            kind,
            {
                "manifest": manifest.model_dump(mode="json"),
                "fragments": _download_fragments(manifest, root),
            },
            root,
            request.run_id,
        )
        rows = read_journal(output / PAIRS_JOURNAL)
        report = json.loads((output / RESULT_FILENAME).read_text())
        validate_result(report, rows, run_id=request.run_id, kind=kind)
        expected = _expected_pairs(kind, manifest)
        if [row["pair_id"] for row in rows] != expected:
            raise Open3dError(
                f"completed {kind} does not exactly cover the requested fragment pairs"
            )
        if kind == "multiway":
            validate_pose_graph(
                report["pose_graph"], fragment_count=len(manifest.fragments)
            )
        report["subprocess_wall_seconds"] = time.perf_counter() - started
        report["manifest_sha256"] = sha256_bytes(
            canonical(manifest.model_dump(mode="json"))
        )
        # Downstream stages need the scale the registration actually ran at and
        # the prefix its scans came from; re-deriving either from a spec would let
        # a later stage disagree with the run it is describing.
        report["input_path"] = request.input_path
        report["voxel_size"] = manifest.voxel_size
        report["icp_estimation"] = manifest.icp_estimation
        report["fragment_ids"] = [fragment.id for fragment in manifest.fragments]
        report["journal_sha256"] = sha256_bytes((output / PAIRS_JOURNAL).read_bytes())
        _publish(
            uri_join(request.output_path, PAIRS_JOURNAL),
            (output / PAIRS_JOURNAL).read_bytes(),
        )
        for name in _extra_artifacts(kind):
            _publish(uri_join(request.output_path, name), (output / name).read_bytes())
        if kind == "register":
            report["aligned_uris"] = _publish_aligned(output, request, rows)
        _publish(uri_join(request.output_path, RESULT_FILENAME), canonical(report))
        return report
    finally:
        _discard(root)


def _expected_pairs(kind: str, manifest: RegistrationManifest) -> list[str]:
    ordered = [fragment.id for fragment in manifest.fragments]
    if kind == "register":
        return [f"{a}__{b}" for a, b in zip(ordered, ordered[1:])]
    return [
        f"{ordered[i]}__{ordered[j]}"
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    ]


def _extra_artifacts(kind: str) -> tuple[str, ...]:
    return (POSE_GRAPH_FILENAME, FUSED_FILENAME) if kind == "multiway" else ()


def _publish_aligned(
    output: Path, request: RunRequest, rows: list[dict[str, Any]]
) -> dict[str, str]:
    published: dict[str, str] = {}
    for row in rows:
        local = output / "aligned" / f"{row['pair_id']}.ply"
        payload = local.read_bytes()
        if sha256_bytes(payload) != row["aligned_sha256"]:
            raise Open3dError(
                f"aligned cloud for {row['pair_id']} does not match its journal digest"
            )
        uri = uri_join(request.output_path, "aligned", local.name)
        _publish(uri, payload)
        published[row["pair_id"]] = uri
    return published


def register(request: RunRequest) -> dict[str, Any]:
    """Global RANSAC/FPFH registration plus ICP refinement, pair by pair."""

    return _registration("register", request)


def multiway(request: RunRequest) -> dict[str, Any]:
    """Full pairwise pose graph, `global_optimization`, and one fused cloud."""

    return _registration("multiway", request)


def validate(request: RunRequest) -> dict[str, Any]:
    """Re-verify a published registration from its artifacts alone."""

    _tool_paths(request)
    journal = read_bytes_uri(uri_join(request.input_path, PAIRS_JOURNAL))
    report = json.loads(read_bytes_uri(uri_join(request.input_path, RESULT_FILENAME)))
    root = _work_dir("npa-open3d-validate-")
    try:
        local = root / PAIRS_JOURNAL
        local.write_bytes(journal)
        rows = read_journal(local)
        kind = report.get("kind") if isinstance(report, dict) else None
        if kind not in {"register", "multiway"}:
            raise Open3dError(
                "published result does not name a known registration kind"
            )
        validate_result(report, rows, run_id=report.get("run_id"), kind=kind)
        if report.get("journal_sha256") != sha256_bytes(journal):
            raise Open3dError("published journal hash does not match its own bytes")
        result = {
            "schema_version": "npa.open3d.validation.v1",
            "run_id": request.run_id,
            "validated_run_id": report["run_id"],
            "kind": kind,
            "pair_count": len(rows),
            "summary": summarize(rows),
            "journal_sha256": report["journal_sha256"],
            "valid": True,
        }
        _publish(uri_join(request.output_path, "validation.json"), canonical(result))
        return result
    finally:
        _discard(root)


def reconstruct(request: ReconstructRequest) -> dict[str, Any]:
    """Poisson surface reconstruction over a published multiway fused cloud."""

    _tool_paths(request)
    manifest_report = json.loads(
        read_bytes_uri(uri_join(request.input_path, RESULT_FILENAME))
    )
    if manifest_report.get("kind") != "multiway":
        raise Open3dError(
            "reconstruct consumes a multiway output prefix; the given prefix holds "
            f"a {manifest_report.get('kind')!r} result"
        )
    pose_graph = manifest_report["pose_graph"]
    root = _work_dir("npa-open3d-reconstruct-")
    try:
        fused = read_bytes_uri(uri_join(request.input_path, FUSED_FILENAME))
        if sha256_bytes(fused) != pose_graph["fused_sha256"]:
            raise Open3dError("fused cloud does not match the published pose graph")
        local = root / FUSED_FILENAME
        local.write_bytes(fused)
        output = _run_runner(
            "reconstruct",
            {
                "request": request.model_dump(mode="json"),
                "fused_path": str(local),
                "voxel_size": _load_manifest_voxel(manifest_report),
            },
            root,
            request.run_id,
        )
        report = json.loads((output / "runner.json").read_text())
        validate_mesh(report["mesh"])
        validate_mesh(report["mesh_uncropped"])
        validate_support(report, voxel_size=_load_manifest_voxel(manifest_report))
        mesh = (output / MESH_FILENAME).read_bytes()
        if sha256_bytes(mesh) != report["mesh"]["sha256"]:
            raise Open3dError("reconstructed mesh does not match its reported digest")
        uncropped = (output / UNCROPPED_MESH_FILENAME).read_bytes()
        if sha256_bytes(uncropped) != report["mesh_uncropped"]["sha256"]:
            raise Open3dError("uncropped mesh does not match its reported digest")
        report["schema_version"] = "npa.open3d.reconstruction.v1"
        # Carry the registration prefix forward so `visualize` can reach the
        # pose graph and fragments without being told twice.
        report["registration_path"] = request.input_path
        _publish(uri_join(request.output_path, MESH_FILENAME), mesh)
        _publish(uri_join(request.output_path, UNCROPPED_MESH_FILENAME), uncropped)
        _publish(uri_join(request.output_path, RESULT_FILENAME), canonical(report))
        return report
    finally:
        _discard(root)


def _load_manifest_voxel(report: dict[str, Any]) -> float:
    """Normals for Poisson must use the scale the registration actually ran at."""

    voxel = report.get("voxel_size")
    if voxel is None:
        raise Open3dError("published registration result does not record voxel_size")
    return float(voxel)


def visualize(request: RunRequest) -> dict[str, Any]:
    """Emit and verify an RRD of the optimized fragments, fusion and surface."""

    _tool_paths(request)
    report = json.loads(read_bytes_uri(uri_join(request.input_path, RESULT_FILENAME)))
    registration_path = report.get("registration_path")
    if report.get("schema_version") != "npa.open3d.reconstruction.v1" or not isinstance(
        registration_path, str
    ):
        raise Open3dError(
            "visualize consumes a reconstruct output prefix; the given prefix does "
            "not hold a reconstruction result"
        )
    registration = json.loads(
        read_bytes_uri(uri_join(registration_path, RESULT_FILENAME))
    )
    pose_graph = json.loads(
        read_bytes_uri(uri_join(registration_path, POSE_GRAPH_FILENAME))
    )
    manifest = _load_manifest(registration["input_path"])
    root = _work_dir("npa-open3d-visualize-")
    try:
        fused = root / FUSED_FILENAME
        fused.write_bytes(read_bytes_uri(uri_join(registration_path, FUSED_FILENAME)))
        mesh = root / MESH_FILENAME
        mesh.write_bytes(read_bytes_uri(uri_join(request.input_path, MESH_FILENAME)))
        uncropped = root / UNCROPPED_MESH_FILENAME
        uncropped.write_bytes(
            read_bytes_uri(uri_join(request.input_path, UNCROPPED_MESH_FILENAME))
        )
        output = _run_runner(
            "visualize",
            {
                "pose_graph": pose_graph,
                "fragments": _download_fragments(manifest, root),
                "fused_path": str(fused),
                "mesh_path": str(mesh),
                "uncropped_mesh_path": str(uncropped),
                "voxel_size": manifest.voxel_size,
            },
            root,
            request.run_id,
        )
        recording = output / RECORDING_FILENAME
        verify_rerun_recording(recording)
        result = json.loads((output / "runner.json").read_text())
        result["schema_version"] = "npa.open3d.recording.v1"
        payload = recording.read_bytes()
        if sha256_bytes(payload) != result["sha256"]:
            raise Open3dError("recording does not match its reported digest")
        _publish(uri_join(request.output_path, RECORDING_FILENAME), payload)
        _publish(uri_join(request.output_path, "rrd-manifest.json"), canonical(result))
        return result
    finally:
        _discard(root)
