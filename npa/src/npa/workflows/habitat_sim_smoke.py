"""Run the exact Habitat-Sim Skokloster RGB-D/Bullet qualification."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import time
from typing import BinaryIO, Callable
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
    """The exact live capability contract was not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    destination: Path,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, object]:
    request = urllib.request.Request(
        ARCHIVE_URL, headers={"User-Agent": "npa-habitat-sim-smoke/2"}
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with opener(request, timeout=120) as response, destination.open("xb") as stream:
            final_url = str(getattr(response, "geturl", lambda: ARCHIVE_URL)())
            if final_url != ARCHIVE_URL:
                raise SmokeFailure(
                    "official archive redirected away from the pinned HTTPS URL"
                )
            headers = getattr(response, "headers", {})
            content_length = headers.get("Content-Length")
            if content_length is not None and int(content_length) > ARCHIVE_BYTES:
                raise SmokeFailure(
                    "official archive exceeds its pinned Content-Length boundary"
                )
            while chunk := response.read(min(1024 * 1024, ARCHIVE_BYTES - size + 1)):
                if size + len(chunk) > ARCHIVE_BYTES:
                    raise SmokeFailure(
                        "official archive exceeded its pinned byte boundary"
                    )
                digest.update(chunk)
                stream.write(chunk)
                size += len(chunk)
            response_metadata = {
                "status": getattr(response, "status", None),
                "content_length": headers.get("Content-Length"),
                "content_type": headers.get("Content-Type"),
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
            }
        os.chmod(destination, 0o600)
        actual = digest.hexdigest()
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
    return destination, {
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
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
    *,
    create_root: bool = True,
) -> tuple[Path, Path, dict[str, object], dict[str, dict[str, object]]]:
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
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        line.strip().rsplit(",", 1)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
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
        {
            "index": index,
            "action": str(action),
            "rgb_shape": list(rgb.shape),
            "rgb_raw_sha256": hashlib.sha256(rgb_bytes).hexdigest(),
            "rgb_png_path": str(rgb_path.relative_to(output_dir)),
            "rgb_png_media_type": "image/png",
            "rgb_png_bytes": rgb_path.stat().st_size,
            "rgb_png_sha256": sha256_file(rgb_path),
            "depth_shape": list(depth.shape),
            "depth_raw_sha256": hashlib.sha256(depth_bytes).hexdigest(),
            "depth_npy_path": str(depth_path.relative_to(output_dir)),
            "depth_npy_media_type": "application/x-npy",
            "depth_npy_bytes": depth_path.stat().st_size,
            "depth_npy_sha256": sha256_file(depth_path),
            "depth_preview_png_path": str(preview_path.relative_to(output_dir)),
            "depth_preview_png_media_type": "image/png",
            "depth_preview_png_bytes": preview_path.stat().st_size,
            "depth_preview_png_sha256": sha256_file(preview_path),
        },
        finite.astype(np.float64, copy=False),
        rgb_bytes,
        depth_bytes,
    )


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
            "physics_config_file": "/usr/src/habitat-sim/data/default.physics_config.json",
            "seed": 7,
        }
    )
    configuration = make_cfg(settings)
    configuration.sim_cfg.random_seed = 7
    configuration.sim_cfg.gpu_device_id = 0
    return _simulate(habitat_sim, np, configuration, output_dir)


def _simulate(habitat_sim: object, np: object, configuration: object, output_dir: Path):
    records, depth_chunks, actions = [], [], []
    rgb_hash, depth_hash = hashlib.sha256(), hashlib.sha256()
    collisions = 0
    with habitat_sim.Simulator(configuration) as simulator:
        if not simulator.pathfinder.is_loaded:
            raise SmokeFailure("the immutable Skokloster Castle navmesh did not load")
        agent = simulator.initialize_agent(0)
        _, goal, geodesic, planned = _choose_path(simulator, agent, habitat_sim, np)
        start_state = np.asarray(agent.get_state().position, dtype=np.float64)
        physics_start = float(simulator.get_world_time())
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
) -> dict[str, object]:
    import numpy as np

    count = traversal["count"]
    finite = traversal["finite_depth"]
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
        "source_revision": SOURCE_REVISION,
        "source": source,
        "scene_id": "habitat_test_scenes/skokloster-castle.glb",
        "scene_sha256": SCENE_SHA256,
        "scene_license": "CC BY 4.0",
        "scene": _asset_proof(archive, records),
        "rendered_rgb_frame_count": count,
        "rendered_depth_frame_count": count,
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
        "measured_fps": traversal["fps"],
        "renderer_egl_evidence": traversal["egl"],
        "observed_gpu": gpu,
        "observed_rtx_gpu_model": gpu["model"],
        "observed_rtx_gpu_architecture": gpu["architecture"],
        "observed_rtx_gpu_count": gpu["count"],
        "pod_observed_immutable_image": image,
        "pod_observed_image_digest": digest,
        "image_observation_source": "NPA_TASK_IMAGE set from the submitted exact digest",
        "exit_status": 0,
    }


def _upload_directory(output_dir: Path, output_uri: str) -> dict[str, object]:
    if not output_uri.startswith("s3://"):
        raise SmokeFailure("output URI must be an s3:// URI")
    import boto3

    bucket, _, prefix = output_uri.removeprefix("s3://").partition("/")
    if not bucket or not prefix:
        raise SmokeFailure("output URI requires a bucket and run-owned prefix")
    client = boto3.client(
        "s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3") or None
    )
    inventory = []
    for path in sorted(
        candidate for candidate in output_dir.rglob("*") if candidate.is_file()
    ):
        relative = path.relative_to(output_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{relative}"
        client.upload_file(str(path), bucket, key)
        head = client.head_object(Bucket=bucket, Key=key)
        if int(head["ContentLength"]) != path.stat().st_size:
            raise SmokeFailure(f"storage readback size mismatch for {relative}")
        readback = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        if hashlib.sha256(readback).hexdigest() != sha256_file(path):
            raise SmokeFailure(f"storage readback hash mismatch for {relative}")
        media_types = {
            ".json": "application/json",
            ".npy": "application/x-npy",
            ".png": "image/png",
        }
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "media_type": media_types[path.suffix],
            }
        )
    return {"object_count": len(inventory), "objects": inventory}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-uri", required=True)
    return parser


def _claim_runtime_cache(cache: Path) -> tuple[int, int, str]:
    """Create and identify the cache that this invocation may remove."""

    created = False
    try:
        cache.mkdir(parents=False, exist_ok=False, mode=0o700)
        created = True
        os.chmod(cache, 0o700)
        identity = cache.stat(follow_symlinks=False)
        marker = cache / ".npa-run-owner"
        marker.write_text(secrets.token_hex(32), encoding="ascii")
        marker.chmod(0o600)
        return identity.st_dev, identity.st_ino, sha256_file(marker)
    except FileExistsError as error:
        raise SmokeFailure("run-owned scene cache path already exists") from error
    except Exception:
        if created:
            shutil.rmtree(cache)
        raise


def _cache_identity_matches(cache: Path, identity: tuple[int, int, str]) -> bool:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open(cache, directory_flags)
    try:
        observed = os.fstat(directory_fd)
        if (
            (observed.st_dev, observed.st_ino) != identity[:2]
            or observed.st_uid != os.geteuid()
            or observed.st_mode & 0o077 != 0
        ):
            return False
        marker_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        marker_fd = os.open(".npa-run-owner", marker_flags, dir_fd=directory_fd)
        try:
            marker = os.fstat(marker_fd)
            payload = os.read(marker_fd, 129)
            after = os.fstat(marker_fd)
            named = os.stat(
                ".npa-run-owner", dir_fd=directory_fd, follow_symlinks=False
            )
            return (
                stat.S_ISREG(marker.st_mode)
                and marker.st_uid == os.geteuid()
                and marker.st_mode & 0o077 == 0
                and marker.st_size == len(payload) == 64
                and marker == after == named
                and hashlib.sha256(payload).hexdigest() == identity[2]
            )
        finally:
            os.close(marker_fd)
    except OSError:
        return False
    finally:
        os.close(directory_fd)


def _remove_runtime_cache(
    cache: Path, identity: tuple[int, int, str] | None = None
) -> None:
    """Remove the run-owned scene cache and verify the postcondition."""

    if not os.path.lexists(cache):
        return
    if identity is not None and not _cache_identity_matches(cache, identity):
        raise SmokeFailure("run-owned scene cache identity changed; refusing cleanup")
    shutil.rmtree(cache)
    if os.path.lexists(cache):
        raise SmokeFailure("run-owned scene cache cleanup did not complete")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    configured_output = os.environ.get("NPA_SMOKE_OUTPUT_DIR")
    if configured_output and Path(configured_output).resolve() != output_dir:
        raise SmokeFailure("NPA_SMOKE_OUTPUT_DIR does not match --output-dir")
    os.environ["NPA_SMOKE_OUTPUT_DIR"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(output_dir, 0o700)
    cache = output_dir.parent / f".{output_dir.name}-scene-cache"
    cache_identity = _claim_runtime_cache(cache)
    try:
        source = _source_provenance()
        image, digest = _immutable_image()
        gpu = query_gpu()
        scene, navmesh, archive, records = fetch_scene_assets(cache, create_root=False)
        if scene.with_suffix(".navmesh") != navmesh:
            raise SmokeFailure(
                "scene and navmesh names do not satisfy auto-load contract"
            )
        traversal = _run_traversal(scene, output_dir)
        proof = _proof(source, image, digest, gpu, archive, records, traversal)
        proof_path = output_dir / "habitat-sim-smoke.json"
        proof_path.write_text(
            json.dumps(proof, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(proof_path, 0o600)
        upload = _upload_directory(output_dir, args.output_uri)
        print(
            json.dumps({"proof": proof, "storage": upload}, sort_keys=True), flush=True
        )
        return 0
    finally:
        _remove_runtime_cache(cache, cache_identity)


if __name__ == "__main__":
    raise SystemExit(main())
