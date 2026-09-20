"""Run the exact Habitat-Sim Skokloster RGB-D/Bullet qualification."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import time
from typing import BinaryIO, Callable
import urllib.parse
import urllib.request
import zipfile


SOURCE_REVISION = "57ee4941dc4765240f0f91f70b2c97a919bf9038"
SOURCE_LICENSE = "MIT"
ARCHIVE_URL = "https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip"
ARCHIVE_BYTES = 94_590_970
ARCHIVE_SHA256 = "1231420c6482e79e25beea7ab25121e0421a5fd67b68dd9502145442c288db06"
SCENE_NAME = "skokloster-castle.glb"
NAVMESH_NAME = "skokloster-castle.navmesh"
SCENE_SHA256 = "b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56"
NAVMESH_SHA256 = "1a9a5bd123af8001f0ea2c5c8d326cb3fd39808ca771fc766856af8f0772391d"
ASSET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.en"
ORIGINAL_ASSET_URL = (
    "https://sketchfab.com/3d-models/the-kings-hall-d18155613363445b9b68c0c67196d98d"
)
CAPABILITY = "skokloster_castle_rgb_depth_bullet_traversal"
WORKFLOW_NAME = "habitat-sim-smoke"
READY_MARKER_NAME = "habitat-sim-publication-ready.json"
PUBLICATION_MANIFEST_NAME = "habitat-sim-publication-manifest.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_RGB_SHAPE = [240, 320, 4]
EXPECTED_DEPTH_SHAPE = [240, 320]
MEMBER_SPECS = {
    SCENE_NAME: {
        "archive_member": (
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
        ),
        "bytes": 38_295_764,
        "crc32": "7a0ced74",
        "sha256": SCENE_SHA256,
    },
    NAVMESH_NAME: {
        "archive_member": (
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.navmesh"
        ),
        "bytes": 28_192,
        "crc32": "a694cab0",
        "sha256": NAVMESH_SHA256,
    },
}


class SmokeFailure(RuntimeError):
    """Signal that the exact live capability contract was not satisfied.

    Args:
        *args: Failure details passed to RuntimeError.

    Returns:
        None.

    Raises:
        None.
    """


@dataclass
class _ObjectWriteLedger:
    attempted: list[str] = field(default_factory=list)
    owned: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


@dataclass
class _RuntimeCacheOwnership:
    """Hold a fresh private namespace for one trusted, non-root run identity.

    The worker must not share its uid or these directories with other writers.
    Mode-0700 directories exclude other identities, not a compromised same-uid
    process or root. Descriptor-relative names are not inode-conditional deletes;
    the exclusive private namespace is part of the ownership contract.
    """

    output_dir: Path
    name: str
    descriptors: dict[str, int] = field(default_factory=dict)
    identities: dict[str, tuple[int, int]] = field(default_factory=dict)
    marker_sha256: str = ""
    created: bool = False
    cleanup_attempted: bool = False
    removed: bool = False


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse redirects before urllib contacts the proposed target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        raise SmokeFailure("official archive redirects are not permitted")


def sha256_file(path: Path) -> str:
    """Hash a file without loading its complete contents into memory.

    Args:
        path: File whose bytes are measured.

    Returns:
        Lowercase SHA-256 hex digest.

    Raises:
        OSError: The file cannot be opened or read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_request():
    parsed = urllib.parse.urlsplit(ARCHIVE_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "dl.fbaipublicfiles.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise SmokeFailure("official archive URL is not the pinned HTTPS origin")
    return urllib.request.Request(
        ARCHIVE_URL, headers={"User-Agent": "npa-habitat-sim-smoke/2"}
    )


def _stream_archive(response, stream):
    final_url = str(getattr(response, "geturl", lambda: ARCHIVE_URL)())
    if final_url != ARCHIVE_URL:
        raise SmokeFailure("official archive redirected away from the pinned HTTPS URL")
    headers = getattr(response, "headers", {})
    content_length = headers.get("Content-Length")
    if content_length is not None and int(content_length) > ARCHIVE_BYTES:
        raise SmokeFailure(
            "official archive exceeds its pinned Content-Length boundary"
        )
    digest = hashlib.sha256()
    size = 0
    while chunk := response.read(min(1024 * 1024, ARCHIVE_BYTES - size + 1)):
        if size + len(chunk) > ARCHIVE_BYTES:
            raise SmokeFailure("official archive exceeded its pinned byte boundary")
        digest.update(chunk)
        stream.write(chunk)
        size += len(chunk)
    return (
        size,
        digest.hexdigest(),
        {
            "status": getattr(response, "status", None),
            "content_length": headers.get("Content-Length"),
            "content_type": headers.get("Content-Type"),
            "etag": headers.get("ETag"),
            "last_modified": headers.get("Last-Modified"),
        },
    )


def _download(
    destination: Path,
    opener: Callable[..., BinaryIO] | None = None,
) -> dict[str, object]:
    request = _archive_request()
    if opener is None:
        opener = urllib.request.build_opener(_RefuseRedirects()).open
    try:
        with opener(request, timeout=120) as response, destination.open("xb") as stream:
            size, actual, response_metadata = _stream_archive(response, stream)
        os.chmod(destination, 0o600)
        if size != ARCHIVE_BYTES or actual != ARCHIVE_SHA256:
            raise SmokeFailure(
                f"official archive identity mismatch: bytes={size}, sha256={actual}"
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {
        "url": ARCHIVE_URL,
        "url_role": "mutable official locator only",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_metadata": response_metadata,
        "bytes": size,
        "sha256": actual,
    }


def _validate_zip_member(member: zipfile.ZipInfo, spec: dict[str, object]) -> None:
    unix_mode = (member.external_attr >> 16) & 0xFFFF
    if member.flag_bits & 0x1:
        raise SmokeFailure("official archive member is encrypted")
    if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise SmokeFailure("official archive member has unsupported compression")
    if (unix_mode & 0o170000) == 0o120000:
        raise SmokeFailure("official archive member is a symbolic link")
    if member.file_size != spec["bytes"] or f"{member.CRC:08x}" != spec["crc32"]:
        raise SmokeFailure("official archive member metadata mismatch")


def _extract_member(
    bundle: zipfile.ZipFile,
    root: Path,
    output_name: str,
    spec: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    matches = [
        item for item in bundle.infolist() if item.filename == spec["archive_member"]
    ]
    if len(matches) != 1:
        raise SmokeFailure(
            f"official archive member count for {output_name}: {len(matches)}"
        )
    member = matches[0]
    _validate_zip_member(member, spec)
    destination = root / output_name
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with bundle.open(member) as source, temporary.open("xb") as stream:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                stream.write(chunk)
        actual = digest.hexdigest()
        if actual != spec["sha256"]:
            raise SmokeFailure(
                f"official archive member hash mismatch for {output_name}: {actual}"
            )
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination, _member_record(member, actual)


def _member_record(member: zipfile.ZipInfo, actual: str) -> dict[str, object]:
    return {
        "name": member.filename,
        "bytes": member.file_size,
        "compressed_bytes": member.compress_size,
        "crc32": f"{member.CRC:08x}",
        "sha256": actual,
    }


def _extract_assets(archive_path: Path, root: Path) -> dict[str, dict[str, object]]:
    published: list[Path] = []
    complete = False
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise SmokeFailure(
                    f"official archive ZIP integrity failure: {bad_member}"
                )
            records: dict[str, dict[str, object]] = {}
            for output_name, spec in MEMBER_SPECS.items():
                destination, record = _extract_member(bundle, root, output_name, spec)
                published.append(destination)
                records[output_name] = record
            complete = True
            return records
    except zipfile.BadZipFile as error:
        raise SmokeFailure("official archive is not a valid ZIP") from error
    finally:
        if not complete:
            for destination in published:
                destination.unlink(missing_ok=True)


def fetch_scene_assets(
    root: Path,
    opener: Callable[..., BinaryIO] | None = None,
    *,
    create_root: bool = True,
) -> tuple[Path, Path, dict[str, object], dict[str, dict[str, object]]]:
    """Fetch and verify only the pinned scene members into the owned cache.

    Args:
        root: Run-owned cache directory.
        opener: Optional HTTP opener, allowing inert fixture transport.
        create_root: Whether this invocation must create the cache directory.

    Returns:
        Scene path, navmesh path, archive identity and member identities.

    Raises:
        SmokeFailure: The archive, cache or selected members fail verification.
        OSError: Transport or local filesystem access fails.
    """
    if create_root:
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
    elif not root.is_dir() or root.is_symlink():
        raise SmokeFailure("run-owned scene cache is not a real directory")
    os.chmod(root, 0o700)
    archive_path = root / "habitat-test-scenes.zip.part"
    archive: dict[str, object] | None = None
    try:
        archive = _download(archive_path, opener=opener)
        records = _extract_assets(archive_path, root)
    finally:
        archive_path.unlink(missing_ok=True)
    if archive is None:
        raise SmokeFailure("official archive acquisition did not complete")
    return _fetched_assets(root, archive_path, archive, records)


def _fetched_assets(root, archive_path, archive, records):
    archive.update(
        {
            "url_is_mutable": True,
            "zip_integrity": "pass",
            "ephemeral_copy_removed": not archive_path.exists(),
            "unrelated_members_extracted": False,
        }
    )
    return root / SCENE_NAME, root / NAVMESH_NAME, archive, records


def query_gpu() -> dict[str, object]:
    """Require exactly one locally observed RTX PRO 6000 Blackwell GPU.

    Args:
        None.

    Returns:
        Observed model, architecture, compute capability and count.

    Raises:
        SmokeFailure: GPU count, model or compute capability does not match.
        subprocess.CalledProcessError: The local observation command fails.
        OSError: The local observation command cannot be started.
    """
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _observed_gpu(completed.stdout)


def _observed_gpu(stdout: str) -> dict[str, object]:
    rows = [line.strip().rsplit(",", 1) for line in stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise SmokeFailure(
            f"Habitat-Sim requires exactly one visible GPU, observed {len(rows)}"
        )
    model, compute = (value.strip() for value in rows[0])
    normalized = re.sub(r"[^A-Z0-9]+", "", model.upper())
    if "RTXPRO6000BLACKWELL" not in normalized or "B200" in normalized:
        raise SmokeFailure(f"expected RTX PRO 6000 Blackwell, observed {model!r}")
    if compute != "12.0":
        raise SmokeFailure(
            f"expected Blackwell compute capability 12.0, observed {compute!r}"
        )
    return {
        "count": 1,
        "model": model,
        "architecture": "Blackwell",
        "compute_capability": compute,
        "observation_source": "nvidia-smi inside the workload pod",
    }


def _gl_evidence(simulator: object) -> dict[str, object]:
    simulator.renderer.acquire_gl_context()
    gl = ctypes.CDLL("libGL.so.1")
    gl.glGetString.argtypes = [ctypes.c_uint]
    gl.glGetString.restype = ctypes.c_char_p

    def read_string(enum: int) -> str:
        value = gl.glGetString(enum)
        return value.decode("utf-8", errors="replace") if value else ""

    values = {
        "vendor": read_string(0x1F00),
        "renderer": read_string(0x1F01),
        "version": read_string(0x1F02),
    }
    if "nvidia" not in values["vendor"].lower():
        raise SmokeFailure(f"headless renderer is not NVIDIA: {values}")
    maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
    libraries = sorted(
        {
            line.rsplit(None, 1)[-1]
            for line in maps.splitlines()
            if "/" in line and ("libEGL" in line or "libGLX_nvidia" in line)
        }
    )
    if not any("libEGL.so" in item for item in libraries):
        raise SmokeFailure("EGL dispatch library is not loaded")
    if not any("libEGL_nvidia.so" in item for item in libraries):
        raise SmokeFailure("NVIDIA EGL implementation is not loaded")
    return {
        "backend": "EGL",
        "display_unset": True,
        "gl": values,
        "libraries": libraries,
    }


def _immutable_image() -> tuple[str, str]:
    image = os.environ.get("NPA_TASK_IMAGE", "").removeprefix("docker:").strip()
    match = re.fullmatch(r".+@(?P<digest>sha256:[0-9a-f]{64})", image)
    if match is None:
        raise SmokeFailure(f"workload pod image is not immutable: {image!r}")
    return image, match.group("digest")


def _source_provenance() -> dict[str, object]:
    path = Path("/usr/share/doc/npa-habitat-sim/source-manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source", {}).get("revision") != SOURCE_REVISION:
        raise SmokeFailure("image source manifest does not match pinned Habitat-Sim")
    return {
        "repository": payload["source"]["repository"],
        "requested_revision": SOURCE_REVISION,
        "observed_revision": payload["source"]["revision"],
        "license": SOURCE_LICENSE,
        "manifest_sha256": sha256_file(path),
    }


def _choose_path(simulator: object, agent: object, habitat_sim: object, np: object):
    for _ in range(64):
        start = simulator.pathfinder.get_random_navigable_point()
        goal = simulator.pathfinder.get_random_navigable_point_near(
            start, 3.0, max_tries=100
        )
        shortest = habitat_sim.ShortestPath()
        shortest.requested_start = start
        shortest.requested_end = goal
        if not simulator.pathfinder.find_path(shortest):
            continue
        if (
            not math.isfinite(shortest.geodesic_distance)
            or shortest.geodesic_distance < 1.0
        ):
            continue
        state = agent.get_state()
        state.position = start
        agent.set_state(state, reset_sensors=True)
        follower = simulator.make_greedy_follower(agent_id=0, goal_radius=0.2)
        try:
            actions = [
                action for action in follower.find_path(goal) if action is not None
            ]
        except habitat_sim.errors.GreedyFollowerError:
            continue
        if "move_forward" in actions:
            return (
                np.asarray(start),
                np.asarray(goal),
                float(shortest.geodesic_distance),
                actions,
            )
    raise SmokeFailure("could not construct a real navigable Skokloster traversal")


def _save_frame(
    output_dir: Path, index: int, action: object, rgb: object, depth: object
):
    import numpy as np
    from PIL import Image

    frames = output_dir / "habitat-sim-observations"
    frames.mkdir(exist_ok=True)
    rgb_bytes = rgb.tobytes(order="C")
    depth_bytes = depth.tobytes(order="C")
    rgb_path = frames / f"rgb-{index:04d}.png"
    depth_path = frames / f"depth-{index:04d}.npy"
    preview_path = frames / f"depth-{index:04d}.png"
    Image.fromarray(rgb, mode="RGBA").save(rgb_path)
    np.save(depth_path, depth, allow_pickle=False)
    finite = depth[np.isfinite(depth)]
    scale = max(float(np.percentile(finite, 99)), 1.0e-6)
    preview = np.clip(depth / scale, 0.0, 1.0)
    preview[~np.isfinite(preview)] = 0.0
    Image.fromarray((preview * 255.0).astype(np.uint8), mode="L").save(preview_path)
    return (
        _frame_record(
            output_dir, index, action, rgb, depth, rgb_path, depth_path, preview_path
        ),
        finite.astype(np.float64, copy=False),
        rgb_bytes,
        depth_bytes,
    )


def _frame_record(
    output_dir, index, action, rgb, depth, rgb_path, depth_path, preview_path
):
    return {
        "index": index,
        "action": str(action),
        "rgb_shape": list(rgb.shape),
        "rgb_raw_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
        "rgb_png_path": str(rgb_path.relative_to(output_dir)),
        "rgb_png_media_type": "image/png",
        "rgb_png_bytes": rgb_path.stat().st_size,
        "rgb_png_sha256": sha256_file(rgb_path),
        "depth_shape": list(depth.shape),
        "depth_raw_sha256": hashlib.sha256(depth.tobytes(order="C")).hexdigest(),
        "depth_npy_path": str(depth_path.relative_to(output_dir)),
        "depth_npy_media_type": "application/x-npy",
        "depth_npy_bytes": depth_path.stat().st_size,
        "depth_npy_sha256": sha256_file(depth_path),
        "depth_preview_png_path": str(preview_path.relative_to(output_dir)),
        "depth_preview_png_media_type": "image/png",
        "depth_preview_png_bytes": preview_path.stat().st_size,
        "depth_preview_png_sha256": sha256_file(preview_path),
    }


def _run_traversal(scene: Path, output_dir: Path) -> dict[str, object]:
    import habitat_sim
    import numpy as np
    from habitat_sim.utils.settings import default_sim_settings, make_cfg

    if os.environ.get("DISPLAY"):
        raise SmokeFailure("headless EGL gate refuses a workload with DISPLAY set")
    if not habitat_sim.bindings.built_with_bullet:
        raise SmokeFailure("Habitat-Sim was not built with Bullet physics")
    settings = default_sim_settings.copy()
    settings.update(
        {
            "scene": str(scene),
            "width": EXPECTED_RGB_SHAPE[1],
            "height": EXPECTED_RGB_SHAPE[0],
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": False,
            "enable_physics": True,
            "physics_config_file": str(
                Path(os.environ.get("NPA_HABITAT_SOURCE_ROOT", "/usr/src/habitat-sim"))
                / "data/default.physics_config.json"
            ),
            "seed": 7,
        }
    )
    configuration = make_cfg(settings)
    configuration.sim_cfg.random_seed = 7
    configuration.sim_cfg.gpu_device_id = 0
    return _simulate(habitat_sim, np, configuration, output_dir)


def _observe_traversal(simulator, np, planned, output_dir):
    records, depth_chunks, actions = [], [], []
    rgb_hash, depth_hash = hashlib.sha256(), hashlib.sha256()
    collisions = 0
    started = time.perf_counter()
    for index, action in enumerate(planned):
        observation = simulator.step(action, dt=1.0 / 60.0)
        rgb = np.ascontiguousarray(observation["color_sensor"])
        depth = np.ascontiguousarray(observation["depth_sensor"], dtype=np.float32)
        if (
            list(rgb.shape) != EXPECTED_RGB_SHAPE
            or list(depth.shape) != EXPECTED_DEPTH_SHAPE
        ):
            raise SmokeFailure("rendered RGB/depth shape mismatch")
        if not np.isfinite(depth).any():
            raise SmokeFailure(f"depth frame {index} has no finite samples")
        record, finite, rgb_bytes, depth_bytes = _save_frame(
            output_dir, index, action, rgb, depth
        )
        records.append(record)
        depth_chunks.append(finite)
        rgb_hash.update(rgb_bytes)
        depth_hash.update(depth_bytes)
        actions.append(str(action))
        collisions += int(bool(observation.get("collided", False)))
    elapsed = time.perf_counter() - started
    return records, depth_chunks, actions, rgb_hash, depth_hash, collisions, elapsed


def _simulate(habitat_sim: object, np: object, configuration: object, output_dir: Path):
    with habitat_sim.Simulator(configuration) as simulator:
        if not simulator.pathfinder.is_loaded:
            raise SmokeFailure("the immutable Skokloster Castle navmesh did not load")
        agent = simulator.initialize_agent(0)
        _, goal, geodesic, planned = _choose_path(simulator, agent, habitat_sim, np)
        start_state = np.asarray(agent.get_state().position, dtype=np.float64)
        physics_start = float(simulator.get_world_time())
        records, depth_chunks, actions, rgb_hash, depth_hash, collisions, elapsed = (
            _observe_traversal(simulator, np, planned, output_dir)
        )
        result = {
            "start": start_state,
            "end": np.asarray(agent.get_state().position, dtype=np.float64),
            "goal": goal,
            "geodesic": geodesic,
            "actions": actions,
            "collisions": collisions,
            "physics_start": physics_start,
            "physics_end": float(simulator.get_world_time()),
            "elapsed": elapsed,
            "egl": _gl_evidence(simulator),
            "records": records,
            "finite_depth": np.concatenate(depth_chunks)
            if depth_chunks
            else np.array([]),
            "rgb_hash": rgb_hash.hexdigest(),
            "depth_hash": depth_hash.hexdigest(),
        }
    return _validate_traversal(result, np)


def _validate_traversal(result: dict[str, object], np: object) -> dict[str, object]:
    records = result["records"]
    count = len(records)
    displacement = float(np.linalg.norm(result["end"] - result["start"]))
    if count == 0 or displacement <= 0.1:
        raise SmokeFailure("agent traversal produced no meaningful displacement")
    if len({item["rgb_raw_sha256"] for item in records}) < 2:
        raise SmokeFailure("RGB traversal observations did not change")
    if len({item["depth_raw_sha256"] for item in records}) < 2:
        raise SmokeFailure("depth traversal observations did not change")
    if result["physics_end"] <= result["physics_start"]:
        raise SmokeFailure("Bullet world time did not advance")
    finite = result["finite_depth"]
    if finite.size == 0 or not np.isfinite(finite).all() or float(np.max(finite)) <= 0:
        raise SmokeFailure("depth traversal has no finite positive range")
    fps = count / result["elapsed"] if result["elapsed"] > 0 else 0.0
    if not math.isfinite(fps) or fps <= 0:
        raise SmokeFailure("invalid measured renderer FPS")
    result.update({"count": count, "displacement": displacement, "fps": fps})
    return result


def _asset_proof(archive: dict[str, object], records: dict[str, dict[str, object]]):
    return {
        "source": "official Meta Habitat test-scene archive",
        "archive": archive,
        "id": "habitat_test_scenes/skokloster-castle.glb",
        "sha256": SCENE_SHA256,
        "bytes": records[SCENE_NAME]["bytes"],
        "archive_member": records[SCENE_NAME],
        "license": "CC BY 4.0",
        "license_url": ASSET_LICENSE_URL,
        "attribution": "The King's Hall, Skokloster Castle; scan by Erik Lernestål",
        "original_asset": {
            "name": "The King's Hall",
            "creator": "Skokloster Castle",
            "scan_credit": "Erik Lernestål",
            "url": ORIGINAL_ASSET_URL,
        },
        "modification_notice": (
            "Official Habitat-ready processed GLB and navmesh derived from the original "
            "scan; NPA copies the selected members byte-for-byte and renders observations."
        ),
        "immutability_boundary": (
            "The HTTPS locator is mutable; the complete archive and both selected "
            "members must match their pinned SHA-256 values."
        ),
        "navmesh_sha256": NAVMESH_SHA256,
        "navmesh_bytes": records[NAVMESH_NAME]["bytes"],
        "navmesh_archive_member": records[NAVMESH_NAME],
    }


def _proof(
    source: dict[str, object],
    image: str,
    digest: str,
    gpu: dict[str, object],
    archive: dict[str, object],
    records: dict[str, dict[str, object]],
    traversal: dict[str, object],
    run_id: str,
    plan_sha256: str,
) -> dict[str, object]:
    count = traversal["count"]
    return {
        "schema_version": "npa.habitat-sim.smoke.v1",
        "solution": "habitat-sim",
        "capability": CAPABILITY,
        "capabilities_exercised": [
            CAPABILITY,
            "headless_nvidia_egl_rgb_depth_render",
            "bullet_physics_world_step",
            "greedy_geodesic_agent_traversal",
        ],
        "execution_binding": _execution_binding(run_id, plan_sha256),
        "source_revision": SOURCE_REVISION,
        "source": source,
        "scene_id": "habitat_test_scenes/skokloster-castle.glb",
        "scene_sha256": SCENE_SHA256,
        "scene_license": "CC BY 4.0",
        "scene": _asset_proof(archive, records),
        "rendered_rgb_frame_count": count,
        "rendered_depth_frame_count": count,
        **_render_proof(traversal),
        **_motion_proof(traversal),
        "measured_fps": traversal["fps"],
        "renderer_egl_evidence": traversal["egl"],
        **_runtime_proof(gpu, image, digest),
        "exit_status": 0,
    }


def _execution_binding(run_id, plan_sha256):
    return {
        "workflow_name": WORKFLOW_NAME,
        "run_id": run_id,
        "rendered_plan_sha256": plan_sha256,
        "runtime_uid": os.geteuid(),
        "runtime_gid": os.getegid(),
    }


def _render_proof(traversal):
    import numpy as np

    count = traversal["count"]
    finite = traversal["finite_depth"]
    return {
        "rendered_observations": {
            "rgb": {
                "frame_count": count,
                "shape": EXPECTED_RGB_SHAPE,
                "dtype": "uint8",
                "aggregate_raw_sha256": traversal["rgb_hash"],
            },
            "depth": {
                "frame_count": count,
                "shape": EXPECTED_DEPTH_SHAPE,
                "dtype": "float32",
                "aggregate_raw_sha256": traversal["depth_hash"],
            },
            "frames": traversal["records"],
            "saved_directory": "habitat-sim-observations",
        },
        "finite_depth_statistics": {
            "count": int(finite.size),
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "standard_deviation": float(np.std(finite)),
        },
    }


def _motion_proof(traversal):
    count = traversal["count"]
    return {
        "agent": {
            "start": traversal["start"].tolist(),
            "end": traversal["end"].tolist(),
            "goal": traversal["goal"].tolist(),
            "displacement": traversal["displacement"],
            "planned_geodesic_distance": traversal["geodesic"],
            "actions": traversal["actions"],
            "collision_count": traversal["collisions"],
        },
        "agent_start": traversal["start"].tolist(),
        "agent_end": traversal["end"].tolist(),
        "agent_displacement": traversal["displacement"],
        "bullet": {
            "built_with_bullet": True,
            "enabled": True,
            "step_count": count,
            "world_time_start": traversal["physics_start"],
            "world_time_end": traversal["physics_end"],
        },
        "bullet_step_count": count,
    }


def _runtime_proof(gpu, image, digest):
    return {
        "observed_gpu": gpu,
        "observed_rtx_gpu_model": gpu["model"],
        "observed_rtx_gpu_architecture": gpu["architecture"],
        "observed_rtx_gpu_count": gpu["count"],
        "pod_observed_immutable_image": image,
        "pod_observed_image_digest": digest,
        "image_observation_source": "NPA_TASK_IMAGE set from the submitted exact digest",
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _readback_object(client: object, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def _verify_object(
    client: object, bucket: str, key: str, expected: bytes, label: str
) -> None:
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head["ContentLength"]) != len(expected):
        raise SmokeFailure(f"storage readback size mismatch for {label}")
    if hashlib.sha256(_readback_object(client, bucket, key)).hexdigest() != (
        hashlib.sha256(expected).hexdigest()
    ):
        raise SmokeFailure(f"storage readback hash mismatch for {label}")


def _put_owned_object(
    client: object,
    bucket: str,
    key: str,
    payload: bytes,
    media_type: str,
    label: str,
    ledger: _ObjectWriteLedger,
    *,
    exclusive_key: bool,
) -> None:
    ledger.attempted.append(key)
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=media_type,
            IfNoneMatch="*",
        )
    except Exception:
        del exclusive_key
        ledger.unresolved.append(key)
        raise
    ledger.owned.append(key)
    _verify_object(client, bucket, key, payload, label)


def _listed_exact_keys(client: object, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if token is not None:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        keys.update(
            str(row["Key"])
            for row in response.get("Contents", [])
            if str(row["Key"]).startswith(prefix)
        )
        if not response.get("IsTruncated"):
            return keys
        token = str(response["NextContinuationToken"])


def _cleanup_exact_objects(client: object, bucket: str, keys: list[str]) -> list[str]:
    failures: list[str] = []
    for key in reversed(keys):
        try:
            client.delete_object(Bucket=bucket, Key=key)
        except Exception as error:
            failures.append(f"delete {key}: {type(error).__name__}: {error}")
    for key in keys:
        try:
            if key in _listed_exact_keys(client, bucket, key):
                failures.append(f"verify {key}: object remains")
        except Exception as error:
            failures.append(f"verify {key}: {type(error).__name__}: {error}")
    return failures


def _publication_failure(
    client: object, bucket: str, ledger: _ObjectWriteLedger, error: Exception
) -> None:
    diagnostics = _cleanup_exact_objects(client, bucket, ledger.owned)
    if ledger.unresolved:
        diagnostics.append(
            "unresolved object ownership after failed write: "
            + ", ".join(ledger.unresolved)
        )
    if diagnostics:
        raise SmokeFailure(
            f"publication failed: {type(error).__name__}: {error}; "
            + "; ".join(diagnostics)
        ) from error


def _publication_provenance(proof: dict[str, object]) -> dict[str, object]:
    scene = proof["scene"]
    return {
        "source_revision": proof["source_revision"],
        "source_license": proof["source"]["license"],
        "source_manifest_sha256": proof["source"]["manifest_sha256"],
        "archive_url": scene["archive"]["url"],
        "archive_sha256": scene["archive"]["sha256"],
        "scene_id": proof["scene_id"],
        "scene_sha256": proof["scene_sha256"],
        "navmesh_sha256": scene["navmesh_sha256"],
        "asset_license": proof["scene_license"],
        "asset_license_url": scene["license_url"],
        "attribution": scene["attribution"],
        "original_asset": scene["original_asset"],
        "modification_notice": scene["modification_notice"],
    }


def _load_named_proof(output_dir: Path) -> dict[str, object]:
    proof_path = output_dir / "habitat-sim-smoke.json"
    if not proof_path.is_file() or proof_path.is_symlink():
        raise SmokeFailure("named Habitat-Sim proof is missing from the output root")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != "npa.habitat-sim.smoke.v1":
        raise SmokeFailure("named Habitat-Sim proof schema is invalid")
    return proof


def _stage_output_files(
    client: object,
    bucket: str,
    stage_prefix: str,
    output_dir: Path,
    ledger: _ObjectWriteLedger,
) -> list[dict[str, object]]:
    media_types = {
        ".json": "application/json",
        ".npy": "application/x-npy",
        ".png": "image/png",
    }
    inventory: list[dict[str, object]] = []
    paths = sorted(
        candidate for candidate in output_dir.rglob("*") if candidate.is_file()
    )
    for path in paths:
        relative = path.relative_to(output_dir).as_posix()
        if path.suffix not in media_types:
            raise SmokeFailure(f"unsupported output media type for {relative}")
        key = f"{stage_prefix}/{relative}"
        payload = path.read_bytes()
        _put_owned_object(
            client,
            bucket,
            key,
            payload,
            media_types[path.suffix],
            relative,
            ledger,
            exclusive_key=True,
        )
        inventory.append(
            _object_record(relative, key, payload, media_types[path.suffix])
        )
    return inventory


def _object_record(relative, key, payload, media_type):
    return {
        "path": relative,
        "key": key,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "media_type": media_type,
    }


def _stage_publication_manifest(
    client: object,
    bucket: str,
    manifest_key: str,
    proof: dict[str, object],
    inventory: list[dict[str, object]],
    ledger: _ObjectWriteLedger,
) -> bytes:
    manifest = {
        "schema_version": "npa.habitat-sim.publication-manifest.v1",
        "solution": "habitat-sim",
        "capability": CAPABILITY,
        "execution_binding": proof["execution_binding"],
        "provenance": _publication_provenance(proof),
        "object_count": len(inventory),
        "objects": inventory,
    }
    payload = _json_bytes(manifest)
    _put_owned_object(
        client,
        bucket,
        manifest_key,
        payload,
        "application/json",
        PUBLICATION_MANIFEST_NAME,
        ledger,
        exclusive_key=True,
    )
    return payload


def _publication_receipt(
    stage_prefix: str,
    manifest_key: str,
    manifest_bytes: bytes,
    ready_key: str,
    inventory: list[dict[str, object]],
) -> tuple[dict[str, object], bytes]:
    proof_rows = [row for row in inventory if row["path"] == "habitat-sim-smoke.json"]
    if len(proof_rows) != 1:
        raise SmokeFailure("staged artifact inventory must name exactly one proof")
    ready = {
        "schema_version": "npa.habitat-sim.publication-ready.v1",
        "stage_prefix": stage_prefix,
        "manifest_key": manifest_key,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "inventory_sha256": hashlib.sha256(_json_bytes(inventory)).hexdigest(),
        "object_count": len(inventory),
        "proof_key": proof_rows[0]["key"],
        "proof_sha256": proof_rows[0]["sha256"],
    }
    payload = _json_bytes(ready)
    publication = {
        **ready,
        "ready_key": ready_key,
        "ready_sha256": hashlib.sha256(payload).hexdigest(),
        "objects": inventory,
    }
    return publication, payload


def _commit_ready_marker(
    client: object,
    bucket: str,
    ready_key: str,
    payload: bytes,
    ledger: _ObjectWriteLedger,
) -> None:
    _put_owned_object(
        client,
        bucket,
        ready_key,
        payload,
        "application/json",
        READY_MARKER_NAME,
        ledger,
        exclusive_key=False,
    )


def _publication_destination(output_uri: str, stage_token: str | None):
    bucket, separator, prefix = output_uri.removeprefix("s3://").partition("/")
    if not output_uri.startswith("s3://") or not separator or not bucket or not prefix:
        raise SmokeFailure("output URI requires an s3:// bucket and run-owned prefix")
    token = stage_token or secrets.token_hex(16)
    if re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise SmokeFailure("staging token must be an exact 128-bit lowercase hex value")
    final_prefix = prefix.rstrip("/")
    stage_prefix = f"{final_prefix}/.staging/{token}"
    ready_key = f"{final_prefix}/{READY_MARKER_NAME}"
    manifest_key = f"{stage_prefix}/{PUBLICATION_MANIFEST_NAME}"
    return bucket, stage_prefix, ready_key, manifest_key


def _stage_publication(client, bucket, stage_prefix, manifest_key, output_dir, ledger):
    proof = _load_named_proof(output_dir)
    inventory = _stage_output_files(client, bucket, stage_prefix, output_dir, ledger)
    manifest = _stage_publication_manifest(
        client, bucket, manifest_key, proof, inventory, ledger
    )
    expected = {row["key"] for row in inventory} | {manifest_key}
    if _listed_exact_keys(client, bucket, stage_prefix + "/") != expected:
        raise SmokeFailure("staged artifact inventory is incomplete or contains extras")
    return inventory, manifest


def _upload_directory(
    output_dir: Path,
    output_uri: str,
    *,
    client: object | None = None,
    stage_token: str | None = None,
    after_commit: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    import boto3

    bucket, stage_prefix, ready_key, manifest_key = _publication_destination(
        output_uri, stage_token
    )
    if client is None:
        client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3") or None
        )
    if _listed_exact_keys(client, bucket, stage_prefix + "/"):
        raise SmokeFailure("run-owned staging prefix is not empty")
    if ready_key in _listed_exact_keys(client, bucket, ready_key):
        raise SmokeFailure("immutable publication ready marker already exists")
    ledger = _ObjectWriteLedger()
    try:
        inventory, manifest = _stage_publication(
            client, bucket, stage_prefix, manifest_key, output_dir, ledger
        )
        publication, ready_bytes = _publication_receipt(
            stage_prefix, manifest_key, manifest, ready_key, inventory
        )
        _commit_ready_marker(client, bucket, ready_key, ready_bytes, ledger)
        if after_commit is not None:
            after_commit(publication)
        return publication
    except Exception as error:
        _publication_failure(client, bucket, ledger, error)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Construct the runtime qualification command parser.

    Args:
        None.

    Returns:
        Parser requiring explicit output and execution identities.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-sha256", required=True)
    return parser


def _execution_identity(run_id: str, plan_sha256: str) -> tuple[str, str]:
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", run_id) is None:
        raise SmokeFailure("workflow run ID is not DNS-safe")
    if SHA256_PATTERN.fullmatch(plan_sha256) is None or plan_sha256 == "0" * 64:
        raise SmokeFailure("rendered plan SHA-256 is missing or invalid")
    if os.geteuid() == 0:
        raise SmokeFailure("Habitat-Sim qualification refuses root execution")
    return run_id, plan_sha256


def _write_termination_receipt(
    publication: dict[str, object], proof: dict[str, object], path: Path
) -> None:
    binding = proof["execution_binding"]
    payload = {
        "schema_version": "npa.habitat-sim.pod-termination.v1",
        "workflow_name": binding["workflow_name"],
        "run_id": binding["run_id"],
        "rendered_plan_sha256": binding["rendered_plan_sha256"],
        "image_digest": proof["pod_observed_image_digest"],
        "proof_sha256": publication["proof_sha256"],
        "manifest_key": publication["manifest_key"],
        "manifest_sha256": publication["manifest_sha256"],
        "ready_key": publication["ready_key"],
        "ready_sha256": publication["ready_sha256"],
        "exit_status": 0,
    }
    path.write_bytes(_json_bytes(payload))


def _private_cache_directory(descriptor: int) -> tuple[int, int]:
    """Require an owned private directory on the retained descriptor."""
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
        or observed.st_nlink < 2
    ):
        raise SmokeFailure("runtime cache directory ownership is unresolved")
    return observed.st_dev, observed.st_ino


def _require_cache_directories(owner: _RuntimeCacheOwnership) -> None:
    """Bind the private parent and cache names to their creation descriptors."""
    _require_cache_parent(owner)
    parent = owner.descriptors["parent"]
    cache = owner.descriptors["cache"]
    if _private_cache_directory(cache) != owner.identities["cache"]:
        raise SmokeFailure("runtime cache descriptor identity changed")
    named_cache = os.stat(owner.name, dir_fd=parent, follow_symlinks=False)
    if (
        not stat.S_ISDIR(named_cache.st_mode)
        or (named_cache.st_dev, named_cache.st_ino) != owner.identities["cache"]
        or named_cache.st_uid != os.geteuid()
        or stat.S_IMODE(named_cache.st_mode) != 0o700
    ):
        raise SmokeFailure("runtime cache named ownership is unresolved")


def _require_cache_parent(owner: _RuntimeCacheOwnership) -> None:
    """Reject lost output ownership before creating or removing any child."""
    if (
        _private_cache_directory(owner.descriptors["parent"])
        != owner.identities["parent"]
    ):
        raise SmokeFailure("runtime cache parent descriptor identity changed")
    named = os.stat(owner.output_dir, follow_symlinks=False)
    if (
        not stat.S_ISDIR(named.st_mode)
        or (named.st_dev, named.st_ino) != owner.identities["parent"]
        or named.st_uid != os.geteuid()
        or stat.S_IMODE(named.st_mode) != 0o700
    ):
        raise SmokeFailure("runtime cache parent ownership is unresolved")


def _cache_file_fingerprint(observed: os.stat_result) -> tuple[int, ...]:
    """Exclude access-time updates while binding private regular-file identity."""
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_mode & 0o077
        or observed.st_nlink != 1
    ):
        raise SmokeFailure("runtime cache contains an unowned or nonregular entry")
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _require_cache_ownership(owner: _RuntimeCacheOwnership) -> None:
    """Verify the held marker bytes without reopening the cache pathname."""
    _require_cache_directories(owner)
    marker = owner.descriptors["marker"]
    before = _cache_file_fingerprint(os.fstat(marker))
    payload = os.pread(marker, 129, 0)
    after = _cache_file_fingerprint(os.fstat(marker))
    named = _cache_file_fingerprint(
        os.stat(
            ".npa-run-owner", dir_fd=owner.descriptors["cache"], follow_symlinks=False
        )
    )
    if (
        before != after
        or after != named
        or len(payload) != 64
        or hashlib.sha256(payload).hexdigest() != owner.marker_sha256
    ):
        raise SmokeFailure("runtime cache owner marker identity changed")


def _initialize_cache(owner: _RuntimeCacheOwnership) -> None:
    """Create a no-clobber cache and marker within the fresh private output."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    owner.descriptors["parent"] = os.open(owner.output_dir, flags)
    parent = owner.descriptors["parent"]
    _require_cache_parent(owner)
    if os.listdir(parent):
        raise SmokeFailure("runtime cache requires a fresh empty private output")
    os.mkdir(owner.name, mode=0o700, dir_fd=parent)
    owner.created = True
    owner.descriptors["cache"] = os.open(owner.name, flags, dir_fd=parent)
    cache = owner.descriptors["cache"]
    owner.identities["cache"] = _private_cache_directory(cache)
    _require_cache_directories(owner)
    marker_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    owner.descriptors["marker"] = os.open(
        ".npa-run-owner", marker_flags, 0o600, dir_fd=cache
    )
    marker = owner.descriptors["marker"]
    payload = secrets.token_hex(32).encode("ascii")
    if os.write(marker, payload) != len(payload):
        raise SmokeFailure("runtime cache owner marker write was incomplete")
    os.fsync(marker)
    owner.marker_sha256 = hashlib.sha256(payload).hexdigest()
    _require_cache_ownership(owner)


def _release_cache_descriptors(owner: _RuntimeCacheOwnership) -> list[str]:
    """Attempt every owned close once, retaining independent failure diagnostics."""
    diagnostics = []
    for role in ("marker", "cache", "parent"):
        descriptor = owner.descriptors.pop(role, None)
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except OSError as error:
            diagnostics.append(f"{role} descriptor release: {error}")
    return diagnostics


def _claim_runtime_cache(
    output_dir: Path, created_identity: tuple[int, int]
) -> _RuntimeCacheOwnership:
    """Retain creation authority; a partial claim never authorizes deletion."""
    owner = _RuntimeCacheOwnership(
        output_dir,
        ".habitat-scene-cache-" + secrets.token_hex(16),
        identities={"parent": created_identity},
    )
    try:
        _initialize_cache(owner)
    except BaseException as primary:
        diagnostics = _release_cache_descriptors(owner)
        primary.add_note(
            "Runtime cache claim incomplete; namespace preserved, not adopted."
        )
        for diagnostic in diagnostics:
            primary.add_note(diagnostic)
        raise
    return owner


def _cache_members(owner: _RuntimeCacheOwnership) -> dict[str, tuple[int, ...]]:
    """Accept only the flat, known files written by this run's asset downloader."""
    _require_cache_ownership(owner)
    allowed = {".npa-run-owner", "habitat-test-scenes.zip.part"}
    allowed.update(MEMBER_SPECS)
    allowed.update(name + ".part" for name in MEMBER_SPECS)
    cache = owner.descriptors["cache"]
    names = os.listdir(cache)
    if set(names) - allowed or ".npa-run-owner" not in names:
        raise SmokeFailure("runtime cache has unresolved entries; refusing cleanup")
    return {
        name: _cache_file_fingerprint(
            os.stat(name, dir_fd=cache, follow_symlinks=False)
        )
        for name in names
    }


def _clear_cache_members(owner: _RuntimeCacheOwnership, members: dict) -> None:
    """Remove only observed payload entries while private ownership still holds."""
    diagnostics = []
    cache = owner.descriptors["cache"]
    for name in sorted(set(members) - {".npa-run-owner"}):
        try:
            _require_cache_ownership(owner)
            named = os.stat(name, dir_fd=cache, follow_symlinks=False)
            if _cache_file_fingerprint(named) != members[name]:
                raise SmokeFailure("runtime cache member identity changed")
            os.unlink(name, dir_fd=cache)
        except (OSError, SmokeFailure) as error:
            diagnostics.append(f"{name}: {type(error).__name__}: {error}")
    if diagnostics:
        failure = SmokeFailure("runtime cache cleanup incomplete; ownership unresolved")
        for diagnostic in diagnostics:
            failure.add_note(diagnostic)
        raise failure


def _remove_runtime_cache(owner: _RuntimeCacheOwnership) -> None:
    """Clean the exclusive private namespace, never a caller-supplied path."""
    members = _cache_members(owner)
    _clear_cache_members(owner, members)
    _require_cache_ownership(owner)
    cache = owner.descriptors["cache"]
    if os.listdir(cache) != [".npa-run-owner"]:
        raise SmokeFailure("runtime cache cleanup has unresolved entries")
    os.unlink(".npa-run-owner", dir_fd=cache)
    _require_cache_directories(owner)
    if os.listdir(cache):
        raise SmokeFailure("runtime cache is not empty; refusing directory removal")
    os.rmdir(owner.name, dir_fd=owner.descriptors["parent"])
    _require_cache_parent(owner)
    if owner.name in os.listdir(owner.descriptors["parent"]):
        raise SmokeFailure("runtime cache cleanup did not complete")
    owner.removed = True


def _finish_runtime_cache(
    owner: _RuntimeCacheOwnership, primary: BaseException | None = None
) -> None:
    """Preserve a primary failure and report every cleanup/release diagnostic."""
    failure = None
    try:
        if owner.cleanup_attempted:
            raise SmokeFailure("runtime cache cleanup cannot be retried")
        owner.cleanup_attempted = True
        _remove_runtime_cache(owner)
    except BaseException as error:
        failure = error
    diagnostics = _release_cache_descriptors(owner)
    if not owner.removed:
        diagnostics.append(
            "Runtime cache ownership/removal unresolved; publication refused."
        )
    target = primary if primary is not None else failure
    if target is None and diagnostics:
        target = SmokeFailure("runtime cache descriptor release incomplete")
    if target is not None:
        if failure is not None and primary is not None:
            target.add_note(f"Cache cleanup: {type(failure).__name__}: {failure}")
            diagnostics.extend(getattr(failure, "__notes__", []))
        for diagnostic in diagnostics:
            target.add_note(diagnostic)
        if primary is None:
            raise target


def _prepare_output(args):
    """Create private output and retain authority for its ephemeral cache child."""
    output_dir = args.output_dir.resolve()
    configured_output = os.environ.get("NPA_SMOKE_OUTPUT_DIR")
    if configured_output and Path(configured_output).resolve() != output_dir:
        raise SmokeFailure("NPA_SMOKE_OUTPUT_DIR does not match --output-dir")
    os.environ["NPA_SMOKE_OUTPUT_DIR"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(output_dir, 0o700)
    created = output_dir.stat(follow_symlinks=False)
    ownership = _claim_runtime_cache(output_dir, (created.st_dev, created.st_ino))
    return output_dir, output_dir / ownership.name, ownership


def _produce_proof(cache, output_dir, run_id, plan_sha256):
    source = _source_provenance()
    image, digest = _immutable_image()
    gpu = query_gpu()
    scene, navmesh, archive, records = fetch_scene_assets(cache, create_root=False)
    if scene.with_suffix(".navmesh") != navmesh:
        raise SmokeFailure("scene and navmesh names do not satisfy auto-load contract")
    traversal = _run_traversal(scene, output_dir)
    proof = _proof(
        source, image, digest, gpu, archive, records, traversal, run_id, plan_sha256
    )
    proof_path = output_dir / "habitat-sim-smoke.json"
    proof_path.write_text(
        json.dumps(proof, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(proof_path, 0o600)
    return proof


def main(argv: list[str] | None = None) -> int:
    """Run qualification and acknowledge only a verified immutable publication.

    Args:
        argv: Explicit arguments, or None to parse the process arguments.

    Returns:
        Zero after verified publication and termination-receipt writing.

    Raises:
        SmokeFailure: A qualification, publication or cleanup gate fails.
        Exception: An underlying runtime, transport or filesystem operation fails.
    """
    args = build_parser().parse_args(argv)
    run_id, plan_sha256 = _execution_identity(args.run_id, args.plan_sha256)
    output_dir, cache, ownership = _prepare_output(args)
    try:
        proof = _produce_proof(cache, output_dir, run_id, plan_sha256)
    except BaseException as primary:
        _finish_runtime_cache(ownership, primary)
        raise
    _finish_runtime_cache(ownership)
    _upload_directory(
        output_dir,
        args.output_uri,
        after_commit=lambda publication: _write_termination_receipt(
            publication, proof, Path("/dev/termination-log")
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
