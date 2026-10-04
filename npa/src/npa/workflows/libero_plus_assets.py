"""Licensed LIBERO-Plus asset camera compatibility through original LIBERO.

This module deliberately does not import, fetch, or execute the unlicensed
``sylvestf/LIBERO-plus`` source repository.  It consumes one allowlisted scene
from the separately MIT-labelled Hugging Face asset archive and renders it with
the original MIT LIBERO executor.  The output is camera compatibility evidence,
not a LIBERO-Plus task, policy, robustness, or robot-success result.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
from http.client import HTTPSConnection
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import urljoin, urlparse, urlsplit
import zipfile


ASSET_REPOSITORY = "Sylvest/LIBERO-plus"
ASSET_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
ASSET_FILE = "assets.zip"
ASSET_ARCHIVE_BYTES = 6_395_849_578
ASSET_ARCHIVE_SHA256 = (
    "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
)
ASSET_MEMBER_COUNT = 457_675
ASSET_LICENSE = "MIT (as declared on the author-published asset card)"
ASSET_CARD_URL = (
    "https://huggingface.co/datasets/Sylvest/LIBERO-plus/blob/"
    f"{ASSET_REVISION}/README.md"
)
ASSET_ARCHIVE_URL = (
    "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/"
    f"{ASSET_REVISION}/{ASSET_FILE}?download=true"
)
# The first URL is the author-published Hugging Face endpoint.  The current
# immutable revision redirects to this exact provider CDN host.  Do not follow a
# redirect to an arbitrary scheme or host: a transport change requires review.
ASSET_DOWNLOAD_HOSTS = frozenset(
    {"huggingface.co", "us.aws.cdn.hf.co", "cas-bridge.xethub.hf.co"}
)
# The author-published archive has a producer-specific leading directory.  Do
# not persist that directory as an NPA contract: it is not the licensed scene
# identity and it is not needed to select the one bounded asset.  The suffix is
# still exact, and ambiguous roots fail closed below.
SCENE_MEMBER = "assets/scenes/libero_tabletop_base_style.xml"
SCENE_SHA256 = "5e69f8568bedf4a71641fcb62285182d0f6dbe498ea18adad86a13706558033f"
MAX_SCENE_BYTES = 1_048_576

NATIVE_EXECUTOR_REPOSITORY = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
NATIVE_EXECUTOR_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
NATIVE_EXECUTOR_LICENSE = "MIT"
NATIVE_EXECUTOR_LICENSE_SHA256 = (
    "e2885fd30a08381b799c4a33385522b23d637b4051b8f9a7f9f2519944b68ff6"
)
DEFAULT_NATIVE_SOURCE_ROOT = "/opt/openwam-libero-source"
CAMERA_SCOPE = "libero-plus-licensed-assets-camera-compatibility"
LIMITATION = (
    "This is native scene and camera compatibility evidence only; it is not the "
    "10,030-task LIBERO-Plus benchmark, a policy rollout, a robustness score, "
    "a convergence result, or physical-robot evidence."
)


class LiberoPlusAssetsError(RuntimeError):
    """Raised when the bounded asset-camera contract cannot be proven."""


def _json_bytes(value: object) -> bytes:
    """Encode deterministic finite JSON for a run-scoped artifact."""
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    """Return the digest of immutable artifact bytes."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a potentially large artifact without materializing it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_uri(uri: str) -> bytes:
    """Read a local file or an exact S3 object URI."""
    if not uri.startswith("s3://"):
        return Path(uri.removeprefix("file://")).read_bytes()
    import boto3

    parsed = urlparse(uri)
    if not parsed.netloc or not parsed.path.strip("/"):
        raise LiberoPlusAssetsError(f"expected exact S3 object URI, got {uri!r}")
    return (
        boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
        .get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
        .read()
    )


def _write_uri(uri: str, payload: bytes) -> None:
    """Write and read back a local or S3 artifact, rejecting changed bytes."""
    if not uri.startswith("s3://"):
        target = Path(uri.removeprefix("file://"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return
    import boto3

    parsed = urlparse(uri)
    if not parsed.netloc or not parsed.path.strip("/"):
        raise LiberoPlusAssetsError(f"expected exact S3 object URI, got {uri!r}")
    client = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
    client.put_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Body=payload)
    observed = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))[
        "Body"
    ].read()
    if _sha256_bytes(observed) != _sha256_bytes(payload):
        raise LiberoPlusAssetsError("S3 read-after-write hash mismatch")


def _asset_cache_root() -> Path:
    """Return the immutable, revision-scoped node cache location."""
    base = Path(os.environ.get("NPA_MODEL_CACHE_DIR", "/workspace/.cache/npa-model"))
    return base / "libero-plus-assets" / ASSET_REVISION


def _asset_download_target(url: str) -> tuple[str, str]:
    """Return a fail-closed HTTPS target for the fixed asset download chain."""
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as error:
        raise LiberoPlusAssetsError(
            "MIT asset download URL has an invalid port"
        ) from error
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or hostname not in ASSET_DOWNLOAD_HOSTS
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/")
        or parsed.fragment
    ):
        raise LiberoPlusAssetsError("MIT asset download target is not approved HTTPS")
    target = parsed.path
    if parsed.query:
        target = f"{target}?{parsed.query}"
    return hostname, target


def _asset_download_response() -> tuple[HTTPSConnection, Any]:
    """Open the pinned archive using only reviewed HTTPS redirect targets."""
    url = ASSET_ARCHIVE_URL
    for _redirect in range(6):
        hostname, target = _asset_download_target(url)
        connection = HTTPSConnection(hostname, timeout=120)
        try:
            connection.request(
                "GET", target, headers={"User-Agent": "npa-libero-assets/1"}
            )
            response = connection.getresponse()
        except BaseException:
            connection.close()
            raise
        if response.status == 200:
            return connection, response
        if response.status not in {301, 302, 303, 307, 308}:
            connection.close()
            raise LiberoPlusAssetsError(
                f"MIT asset download returned unexpected HTTPS status {response.status}"
            )
        location = response.getheader("Location")
        connection.close()
        if not location:
            raise LiberoPlusAssetsError("MIT asset download redirect has no location")
        url = urljoin(url, location)
    raise LiberoPlusAssetsError("MIT asset download exceeded approved redirect limit")


def _download_archive(destination: Path) -> None:
    """Download the public immutable archive to a unique temporary path."""
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    try:
        connection, response = _asset_download_response()
        try:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        finally:
            connection.close()
        if temporary.stat().st_size != ASSET_ARCHIVE_BYTES:
            raise LiberoPlusAssetsError(
                "MIT asset archive size mismatch; refusing incomplete or changed payload"
            )
        if _sha256_file(temporary) != ASSET_ARCHIVE_SHA256:
            raise LiberoPlusAssetsError("MIT asset archive SHA-256 mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verified_archive() -> Path:
    """Return a locked, hash-verified archive with an atomic ready marker."""
    root = _asset_cache_root()
    archive = root / ASSET_FILE
    ready = root / "READY.json"
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{ASSET_REVISION}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            ready.exists()
            and archive.exists()
            and _sha256_file(archive) == ASSET_ARCHIVE_SHA256
        ):
            return archive
        if root.exists():
            shutil.rmtree(root)
        temporary = Path(
            tempfile.mkdtemp(prefix="libero-plus-assets-", dir=root.parent)
        )
        try:
            downloaded = temporary / ASSET_FILE
            _download_archive(downloaded)
            (temporary / "READY.json").write_bytes(
                _json_bytes(
                    {
                        "schema": "npa.libero-plus.mit-assets-cache.v1",
                        "repository": ASSET_REPOSITORY,
                        "revision": ASSET_REVISION,
                        "archive": ASSET_FILE,
                        "archive_sha256": ASSET_ARCHIVE_SHA256,
                        "archive_bytes": ASSET_ARCHIVE_BYTES,
                    }
                )
            )
            os.replace(temporary, root)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return archive


def _extract_allowlisted_scene(archive: Path) -> bytes:
    """Read one verified scene member without extracting task or source bytes."""
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != ASSET_MEMBER_COUNT:
            raise LiberoPlusAssetsError("MIT asset archive member inventory drift")
        matches = [
            member
            for member in members
            if _is_allowlisted_scene_member(member.filename)
        ]
        if len(matches) != 1:
            raise LiberoPlusAssetsError(
                "allowlisted MIT scene must have exactly one normalized archive member"
            )
        member = matches[0]
        mode = member.external_attr >> 16
        if member.is_dir() or (mode & 0o170000) == 0o120000:
            raise LiberoPlusAssetsError("allowlisted MIT scene has unsafe ZIP metadata")
        if member.file_size <= 0 or member.file_size > MAX_SCENE_BYTES:
            raise LiberoPlusAssetsError("allowlisted MIT scene has unexpected size")
        with bundle.open(member) as handle:
            scene = handle.read(MAX_SCENE_BYTES + 1)
    if len(scene) != member.file_size or _sha256_bytes(scene) != SCENE_SHA256:
        raise LiberoPlusAssetsError("allowlisted MIT scene SHA-256 mismatch")
    return scene


def _is_allowlisted_scene_member(name: str) -> bool:
    """Accept the one bounded scene below an opaque archive root.

    The Hugging Face asset archive has an upstream producer directory before the
    asset path.  Keeping it opaque avoids treating that incidental directory as
    a source contract, while rejecting absolute and traversal paths prevents a
    similarly named unsafe entry from satisfying the allowlist.
    """
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        return False
    return normalized == SCENE_MEMBER or normalized.endswith(f"/{SCENE_MEMBER}")


def acquire(output_uri: str, scene_uri: str) -> None:
    """Acquire, hash, attribute, and safely select an MIT scene asset.

    The complete archive is downloaded only to the immutable node cache and is
    not written to run artifacts.  Only the allowlisted XML scene is emitted.
    """
    archive = _verified_archive()
    scene = _extract_allowlisted_scene(archive)
    _write_uri(scene_uri, scene)
    _write_uri(
        output_uri,
        _json_bytes(
            {
                "schema": "npa.libero-plus.licensed-assets.manifest.v1",
                "capability": CAMERA_SCOPE,
                "asset": {
                    "repository": ASSET_REPOSITORY,
                    "revision": ASSET_REVISION,
                    "license": ASSET_LICENSE,
                    "terms_url": ASSET_CARD_URL,
                    "archive": ASSET_FILE,
                    "archive_sha256": ASSET_ARCHIVE_SHA256,
                    "archive_bytes": ASSET_ARCHIVE_BYTES,
                    "member_count": ASSET_MEMBER_COUNT,
                    "selected_member": SCENE_MEMBER,
                    "selected_member_sha256": SCENE_SHA256,
                    "selected_member_uri": scene_uri,
                },
                "exclusions": {
                    "unlicensed_libero_plus_source": "not fetched or executed",
                    "bddl_task_definitions": "not present in asset archive and not fetched",
                    "benchmark_randomizer": "not present in asset archive and not fetched",
                },
                "limitation": LIMITATION,
            }
        ),
    )


def _native_source_root() -> Path:
    """Verify the mounted original-MIT executor before importing it."""
    source = Path(os.environ.get("NPA_LIBERO_SOURCE_ROOT", DEFAULT_NATIVE_SOURCE_ROOT))
    license_path = source / "LICENSE"
    if not (
        source / "libero" / "libero" / "envs" / "arenas" / "table_arena.py"
    ).is_file():
        raise LiberoPlusAssetsError(
            "original LIBERO TableArena executor is unavailable"
        )
    if (
        not license_path.is_file()
        or _sha256_file(license_path) != NATIVE_EXECUTOR_LICENSE_SHA256
    ):
        raise LiberoPlusAssetsError("original LIBERO MIT license identity mismatch")
    return source


def _isolated_libero_config(source: Path, parent: Path) -> Path:
    """Create the LIBERO config before importing native LIBERO modules."""
    config_root = parent / "libero-config"
    config_root.mkdir(parents=True, exist_ok=True)
    libero_root = source / "libero" / "libero"
    (config_root / "config.yaml").write_text(
        "benchmark_root: {root}\n"
        "bddl_files: {root}/bddl_files\n"
        "init_states: {root}/init_files\n"
        "assets: {root}/assets\n"
        "datasets: {datasets}\n".format(root=libero_root, datasets=parent / "datasets")
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    return config_root


def _compile_scene(scene: bytes) -> tuple[Any, Any, Any, Path]:
    """Compile one scene through original TableArena and advance MuJoCo once."""
    if os.environ.get("MUJOCO_GL", "egl") != "egl":
        raise LiberoPlusAssetsError(
            "licensed-assets camera workflow requires MUJOCO_GL=egl"
        )
    os.environ["MUJOCO_GL"] = "egl"
    source = _native_source_root()
    workspace = Path(tempfile.mkdtemp(prefix="libero-plus-assets-scene-"))
    try:
        _isolated_libero_config(source, workspace)
        scene_path = workspace / "scene.xml"
        scene_path.write_bytes(scene)
        import mujoco
        from libero.libero.envs.arenas.table_arena import TableArena

        arena = TableArena(xml=str(scene_path))
        model = mujoco.MjModel.from_xml_string(arena.get_xml())
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return mujoco, model, data, workspace
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def _close_scene(model: Any, workspace: Path) -> None:
    """Release native scene state and remove only this invocation's config."""
    del model
    shutil.rmtree(workspace, ignore_errors=True)


def _camera_pose(mujoco: Any, model: Any, camera: str) -> dict[str, Any]:
    """Return native camera position/quaternion for a named camera."""
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if identifier < 0:
        raise LiberoPlusAssetsError(f"native scene has no requested camera {camera!r}")
    return {
        "name": camera,
        "id": int(identifier),
        "position": [float(value) for value in model.cam_pos[identifier]],
        "quaternion": [float(value) for value in model.cam_quat[identifier]],
    }


def assemble(asset_manifest_uri: str, output_uri: str) -> None:
    """Compile and forward the selected scene with the original MIT executor."""
    manifest_payload = _read_uri(asset_manifest_uri)
    manifest = json.loads(manifest_payload)
    asset = manifest.get("asset", {})
    if (
        manifest.get("schema") != "npa.libero-plus.licensed-assets.manifest.v1"
        or asset.get("revision") != ASSET_REVISION
        or asset.get("selected_member") != SCENE_MEMBER
        or asset.get("selected_member_sha256") != SCENE_SHA256
    ):
        raise LiberoPlusAssetsError(
            "scene assembly requires the exact acquired MIT asset manifest"
        )
    scene = _read_uri(str(asset["selected_member_uri"]))
    if _sha256_bytes(scene) != SCENE_SHA256:
        raise LiberoPlusAssetsError(
            "scene assembly input does not match allowlisted MIT scene"
        )
    mujoco, model, _data, workspace = _compile_scene(scene)
    try:
        cameras = [
            _camera_pose(mujoco, model, name) for name in ("agentview", "agentview_60")
        ]
        _write_uri(
            output_uri,
            _json_bytes(
                {
                    "schema": "npa.libero-plus.licensed-assets.scene-assembly.v1",
                    "capability": CAMERA_SCOPE,
                    "asset_manifest_sha256": _sha256_bytes(manifest_payload),
                    "scene": {
                        "uri": asset["selected_member_uri"],
                        "sha256": SCENE_SHA256,
                    },
                    "native_executor": {
                        "repository": NATIVE_EXECUTOR_REPOSITORY,
                        "revision": NATIVE_EXECUTOR_REVISION,
                        "license": NATIVE_EXECUTOR_LICENSE,
                        "license_sha256": NATIVE_EXECUTOR_LICENSE_SHA256,
                        "arena": "TableArena",
                        "mujoco_version": mujoco.__version__,
                    },
                    "native_model": {
                        "camera_count": int(model.ncam),
                        "geometry_count": int(model.ngeom),
                        "light_count": int(model.nlight),
                        "mj_forward": "passed",
                        "camera_poses": cameras,
                    },
                    "limitation": LIMITATION,
                }
            ),
        )
    finally:
        _close_scene(model, workspace)


def render(
    assembly_uri: str, output_uri: str, camera_a_uri: str, camera_b_uri: str
) -> None:
    """Render two named native EGL camera views from the same scene state."""
    assembly_payload = _read_uri(assembly_uri)
    assembly = json.loads(assembly_payload)
    if assembly.get("schema") != "npa.libero-plus.licensed-assets.scene-assembly.v1":
        raise LiberoPlusAssetsError(
            "renderer requires a native scene-assembly artifact"
        )
    scene_info = assembly.get("scene", {})
    scene = _read_uri(str(scene_info.get("uri", "")))
    if (
        _sha256_bytes(scene) != scene_info.get("sha256")
        or scene_info.get("sha256") != SCENE_SHA256
    ):
        raise LiberoPlusAssetsError("renderer scene does not match native assembly")
    mujoco, model, data, workspace = _compile_scene(scene)
    try:
        import numpy as np
        from PIL import Image

        renderer = mujoco.Renderer(model, height=256, width=256)
        frames: dict[str, Any] = {}
        try:
            for camera, uri in (
                ("agentview", camera_a_uri),
                ("agentview_60", camera_b_uri),
            ):
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
                    raise LiberoPlusAssetsError(
                        "native EGL renderer emitted an unexpected RGB frame"
                    )
                with tempfile.TemporaryDirectory(
                    prefix="libero-plus-assets-png-"
                ) as temporary:
                    png = Path(temporary) / f"{camera}.png"
                    Image.fromarray(frame).save(png)
                    payload = png.read_bytes()
                _write_uri(uri, payload)
                frames[camera] = {
                    "uri": uri,
                    "sha256": _sha256_bytes(payload),
                    "bytes": len(payload),
                    "shape": [int(value) for value in frame.shape],
                    "pose": _camera_pose(mujoco, model, camera),
                }
        finally:
            renderer.close()
        _write_uri(
            output_uri,
            _json_bytes(
                {
                    "schema": "npa.libero-plus.licensed-assets.camera-gallery.v1",
                    "capability": CAMERA_SCOPE,
                    "assembly_sha256": _sha256_bytes(assembly_payload),
                    "render_backend": "egl",
                    "scene_sha256": SCENE_SHA256,
                    "frames": frames,
                    "limitation": LIMITATION,
                }
            ),
        )
    finally:
        _close_scene(model, workspace)


def validate(gallery_uri: str, output_uri: str) -> None:
    """Decode both emitted images and calculate measured pose and pixel deltas."""
    gallery_payload = _read_uri(gallery_uri)
    gallery = json.loads(gallery_payload)
    if gallery.get("schema") != "npa.libero-plus.licensed-assets.camera-gallery.v1":
        raise LiberoPlusAssetsError(
            "validator requires a native camera-gallery artifact"
        )
    frames = gallery.get("frames", {})
    if (
        set(frames) != {"agentview", "agentview_60"}
        or gallery.get("render_backend") != "egl"
    ):
        raise LiberoPlusAssetsError(
            "validator requires exactly the two requested EGL camera frames"
        )
    import numpy as np
    from PIL import Image

    decoded: dict[str, Any] = {}
    pixels: dict[str, Any] = {}
    for camera, evidence in frames.items():
        payload = _read_uri(str(evidence["uri"]))
        if _sha256_bytes(payload) != evidence.get("sha256"):
            raise LiberoPlusAssetsError(f"{camera} PNG digest does not match gallery")
        with tempfile.TemporaryDirectory(
            prefix="libero-plus-assets-decode-"
        ) as temporary:
            png = Path(temporary) / f"{camera}.png"
            png.write_bytes(payload)
            with Image.open(png) as image:
                image.load()
                if image.mode != "RGB" or image.size != (256, 256):
                    raise LiberoPlusAssetsError(
                        f"{camera} is not a decodable 256x256 RGB PNG"
                    )
                pixels[camera] = np.asarray(image, dtype=np.uint8).copy()
        decoded[camera] = {
            "mode": "RGB",
            "width": 256,
            "height": 256,
            "sha256": evidence["sha256"],
        }
    position_a = np.asarray(frames["agentview"]["pose"]["position"], dtype=float)
    position_b = np.asarray(frames["agentview_60"]["pose"]["position"], dtype=float)
    quaternion_a = np.asarray(frames["agentview"]["pose"]["quaternion"], dtype=float)
    quaternion_b = np.asarray(frames["agentview_60"]["pose"]["quaternion"], dtype=float)
    image_mad = float(
        np.abs(
            pixels["agentview"].astype(float) - pixels["agentview_60"].astype(float)
        ).mean()
    )
    position_distance = float(np.linalg.norm(position_a - position_b))
    quaternion_distance = float(
        min(
            np.linalg.norm(quaternion_a - quaternion_b),
            np.linalg.norm(quaternion_a + quaternion_b),
        )
    )
    if image_mad <= 0.0 or position_distance <= 0.0:
        raise LiberoPlusAssetsError(
            "matched native camera views did not differ in pixels and pose"
        )
    _write_uri(
        output_uri,
        _json_bytes(
            {
                "schema": "npa.libero-plus.licensed-assets.camera-metrics.v1",
                "capability": CAMERA_SCOPE,
                "gallery_sha256": _sha256_bytes(gallery_payload),
                "decoded": decoded,
                "metrics": {
                    "rgb_mean_absolute_difference": image_mad,
                    "camera_position_l2_distance": position_distance,
                    "camera_quaternion_sign_invariant_l2_distance": quaternion_distance,
                },
                "limitation": LIMITATION,
            }
        ),
    )


def _emit_rrd(
    gallery: dict[str, Any], metrics: dict[str, Any], target_uri: str
) -> dict[str, Any]:
    """Write a factual gallery and measured camera metrics into a closed RRD."""
    try:
        import rerun as rr
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise LiberoPlusAssetsError(
            "runtime image requires rerun, numpy, and Pillow"
        ) from error
    with tempfile.TemporaryDirectory(prefix="libero-plus-assets-rrd-") as temporary:
        target = Path(temporary) / "licensed-assets-camera-compatibility.rrd"
        recording = rr.RecordingStream(
            "npa_libero_plus_licensed_assets_camera_compatibility"
        )
        recording.save(str(target))
        rr.log("provenance/capability", rr.TextLog(CAMERA_SCOPE), recording=recording)
        rr.log("provenance/limitation", rr.TextLog(LIMITATION), recording=recording)
        rr.log(
            "metrics/rgb_mean_absolute_difference",
            rr.Scalars(float(metrics["metrics"]["rgb_mean_absolute_difference"])),
            recording=recording,
        )
        rr.log(
            "metrics/camera_position_l2_distance",
            rr.Scalars(float(metrics["metrics"]["camera_position_l2_distance"])),
            recording=recording,
        )
        rr.log(
            "metrics/camera_quaternion_sign_invariant_l2_distance",
            rr.Scalars(
                float(
                    metrics["metrics"]["camera_quaternion_sign_invariant_l2_distance"]
                )
            ),
            recording=recording,
        )
        for index, camera in enumerate(("agentview", "agentview_60")):
            payload = _read_uri(str(gallery["frames"][camera]["uri"]))
            with tempfile.TemporaryDirectory(
                prefix="libero-plus-assets-rrd-image-"
            ) as image_tmp:
                image_path = Path(image_tmp) / f"{camera}.png"
                image_path.write_bytes(payload)
                with Image.open(image_path) as image:
                    image.load()
                    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            recording.set_time("camera_frame", sequence=index)
            rr.log(f"gallery/{camera}", rr.Image(pixels), recording=recording)
        recording.flush()
        recording.disconnect()
        payload = target.read_bytes()
    _write_uri(target_uri, payload)
    return {"uri": target_uri, "sha256": _sha256_bytes(payload), "bytes": len(payload)}


def report(
    assembly_uri: str, gallery_uri: str, metrics_uri: str, output_uri: str, rrd_uri: str
) -> None:
    """Publish factual gallery provenance, RRD, and the explicitly narrow scope."""
    assembly_payload = _read_uri(assembly_uri)
    gallery_payload = _read_uri(gallery_uri)
    metrics_payload = _read_uri(metrics_uri)
    assembly = json.loads(assembly_payload)
    gallery = json.loads(gallery_payload)
    metrics = json.loads(metrics_payload)
    if metrics.get("gallery_sha256") != _sha256_bytes(gallery_payload):
        raise LiberoPlusAssetsError(
            "report metrics do not consume the exact native gallery"
        )
    if gallery.get("assembly_sha256") != _sha256_bytes(assembly_payload):
        raise LiberoPlusAssetsError(
            "report gallery does not consume the exact native assembly"
        )
    rrd = _emit_rrd(gallery, metrics, rrd_uri)
    _write_uri(
        output_uri,
        _json_bytes(
            {
                "schema": "npa.libero-plus.licensed-assets.camera-report.v1",
                "capability": CAMERA_SCOPE,
                "scope": "MIT asset scene plus original MIT LIBERO native camera compatibility",
                "assembly_sha256": _sha256_bytes(assembly_payload),
                "gallery_sha256": _sha256_bytes(gallery_payload),
                "metrics_sha256": _sha256_bytes(metrics_payload),
                "asset": {
                    "repository": ASSET_REPOSITORY,
                    "revision": ASSET_REVISION,
                    "terms_url": ASSET_CARD_URL,
                    "archive_sha256": ASSET_ARCHIVE_SHA256,
                    "selected_member": SCENE_MEMBER,
                    "selected_member_sha256": SCENE_SHA256,
                },
                "native_executor": assembly["native_executor"],
                "gallery": gallery["frames"],
                "metrics": metrics["metrics"],
                "rrd": rrd,
                "limitations": LIMITATION,
                "benchmark_equivalence": False,
                "policy_rollout": False,
                "physical_robot": False,
            }
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the native workflow-stage CLI parser for direct invocation tests."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    acquire_parser = commands.add_parser("acquire")
    acquire_parser.add_argument("--output-uri", required=True)
    acquire_parser.add_argument("--scene-uri", required=True)
    assemble_parser = commands.add_parser("assemble")
    assemble_parser.add_argument("--asset-manifest-uri", required=True)
    assemble_parser.add_argument("--output-uri", required=True)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--assembly-uri", required=True)
    render_parser.add_argument("--output-uri", required=True)
    render_parser.add_argument("--camera-a-uri", required=True)
    render_parser.add_argument("--camera-b-uri", required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--gallery-uri", required=True)
    validate_parser.add_argument("--output-uri", required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--assembly-uri", required=True)
    report_parser.add_argument("--gallery-uri", required=True)
    report_parser.add_argument("--metrics-uri", required=True)
    report_parser.add_argument("--output-uri", required=True)
    report_parser.add_argument("--rrd-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch one real licensed-asset camera workflow stage."""
    args = build_parser().parse_args(argv)
    if args.command == "acquire":
        acquire(args.output_uri, args.scene_uri)
    elif args.command == "assemble":
        assemble(args.asset_manifest_uri, args.output_uri)
    elif args.command == "render":
        render(args.assembly_uri, args.output_uri, args.camera_a_uri, args.camera_b_uri)
    elif args.command == "validate":
        validate(args.gallery_uri, args.output_uri)
    else:
        report(
            args.assembly_uri,
            args.gallery_uri,
            args.metrics_uri,
            args.output_uri,
            args.rrd_uri,
        )


if __name__ == "__main__":
    main()
