"""Verify and adapt the pinned Comet12 policy without importing its runtime."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

import numpy as np


SOURCE_COMMIT = "4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5"
MODEL_REPOSITORY = "sunshk/openpi_comet"
MODEL_REVISION = "a3d85eb978b58501c99f6c927a18d52ec6c1532c"
CHECKPOINT_NAME = "pi05-b1kpt12-cs32"
CONFIG_NAME = "pi05_b1k-base"
SUPPORTED_TASK_IDS = (0, 1, 6, 17, 18, 22, 30, 32, 34, 35, 40, 45)
CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")
PROPRIOCEPTION_INDICES = {
    "R1Pro": {
        "base_qvel": slice(0, 3),
        "arm_left_qpos": slice(3, 10),
        "gripper_left_qpos": slice(24, 26),
        "arm_right_qpos": slice(28, 35),
        "gripper_right_qpos": slice(49, 51),
        "trunk_qpos": slice(53, 57),
    }
}
SOURCE_FILES = {
    "pyproject.toml": "3b32ad5b6d6b9d00aa421252d577038bbf47718ec834af7f016f1d755c33b7eb",
    "scripts/serve_b1k.py": "59b3b97698a79540677513cfba74661d1e63f4c5a37250b9957ba6801557d406",
    "src/openpi/policies/b1k_policy.py": "a82c15dece0242a94cb25d94b7fe4ed04c1d0fcfc508ed6d9d5e28b0f32a075e",
    "src/openpi/policies/policy_config.py": "aaf42ab04a33b6c91d2447926211646c33ebd8ebfa427f026d7b6c4f7c45ec52",
    "src/openpi/shared/eval_b1k_wrapper.py": "7afc9498628d3295e3360a7b37eec81e98f82b4cce862409d957af3f78437779",
    "src/openpi/training/config.py": "f5a4a363b19e8956745962af494996efd51088f36692d0560c6b70b1d56c18dd",
}


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_digest(path: Path, size: int) -> str:
    digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, purpose: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{purpose} must be a regular file")


def verify_source(root: Path) -> dict[str, str]:
    """Verify the clean pinned Comet source checkout and critical file bytes.

    Args:
        root: Root of the operator-fetched source checkout.
    Returns:
        Critical source paths mapped to their verified SHA-256 identities.
    Raises:
        ValueError: The revision, cleanliness, or critical bytes differ.
    """
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if revision != SOURCE_COMMIT or changes:
        raise ValueError("Comet source must match the clean pinned checkout")
    for relative, expected in SOURCE_FILES.items():
        path = root / relative
        _regular_file(path, "Pinned Comet source")
        if _digest(path) != expected:
            raise ValueError("Pinned Comet source file bytes differ")
    return dict(SOURCE_FILES)


def load_checkpoint_manifest(path: Path | None = None) -> dict:
    """Load and validate the frozen public checkpoint inventory metadata.

    Args:
        path: Optional inventory path used by focused tests.
    Returns:
        Validated checkpoint inventory.
    Raises:
        ValueError: Metadata identity, totals, paths, or entries are malformed.
    """
    source = path or Path(__file__).with_name("comet12-checkpoint.json")
    _regular_file(source, "Comet checkpoint inventory")
    manifest = json.loads(source.read_text())
    _validate_manifest_header(manifest)
    _validate_manifest_files(manifest)
    return manifest


def _validate_manifest_header(manifest: dict) -> None:
    expected = {
        "schema": "npa.behavior.comet12-checkpoint-inventory.v1",
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "checkpoint": CHECKPOINT_NAME,
        "file_count": 5948,
        "total_bytes": 12_441_382_544,
        "lfs_file_count": 2436,
        "lfs_bytes": 12_411_371_755,
        "task_ids": list(SUPPORTED_TASK_IDS),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Comet checkpoint inventory identity differs")


def _validate_manifest_files(manifest: dict) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != manifest["file_count"]:
        raise ValueError("Comet checkpoint inventory file count differs")
    total = 0
    lfs_count = 0
    lfs_bytes = 0
    for relative, identity in files.items():
        _validate_manifest_entry(relative, identity)
        total += identity["size"]
        if "sha256" in identity:
            lfs_count += 1
            lfs_bytes += identity["size"]
    if (total, lfs_count, lfs_bytes) != (
        manifest["total_bytes"],
        manifest["lfs_file_count"],
        manifest["lfs_bytes"],
    ):
        raise ValueError("Comet checkpoint inventory totals differ")


def _validate_manifest_entry(relative: str, identity: dict) -> None:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Comet checkpoint inventory path is unsafe")
    keys = set(identity)
    if keys not in ({"size", "sha256"}, {"size", "git_blob_sha1"}):
        raise ValueError("Comet checkpoint inventory identity is malformed")
    digest = identity.get("sha256", identity.get("git_blob_sha1", ""))
    if type(identity["size"]) is not int or identity["size"] < 0:
        raise ValueError("Comet checkpoint inventory size is malformed")
    if len(digest) not in (40, 64) or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise ValueError("Comet checkpoint inventory digest is malformed")


def _checkpoint_files(root: Path) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Comet checkpoint must be a regular directory")
    files = {}
    for directory, names, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in names):
            raise ValueError("Comet checkpoint cannot contain symlinks")
        for name in filenames:
            path = base / name
            _regular_file(path, "Comet checkpoint member")
            files[path.relative_to(root).as_posix()] = path
    return files


def _verify_checkpoint_files(root: Path, manifest: dict) -> dict[str, dict]:
    files = _checkpoint_files(root)
    expected = manifest["files"]
    if set(files) != set(expected):
        raise ValueError("Comet checkpoint file set differs")
    for relative, identity in expected.items():
        path = files[relative]
        if path.stat().st_size != identity["size"]:
            raise ValueError("Comet checkpoint file size differs")
        actual = _checkpoint_digest(path, identity)
        expected_digest = identity.get("sha256", identity.get("git_blob_sha1"))
        if actual != expected_digest:
            raise ValueError("Comet checkpoint file bytes differ")
    return expected


def verify_checkpoint(root: Path) -> dict[str, dict]:
    """Verify every fetched Comet checkpoint file against its frozen identity.

    Args:
        root: Extracted checkpoint folder, excluding its Hugging Face parent.
    Returns:
        The exact verified file inventory.
    Raises:
        ValueError: File sets, sizes, digests, or file types differ.
    """
    return _verify_checkpoint_files(root, load_checkpoint_manifest())


def _checkpoint_digest(path: Path, identity: dict) -> str:
    if "sha256" in identity:
        return _digest(path)
    return _git_blob_digest(path, identity["size"])


def _archive_members(archive: Path) -> dict[str, zipfile.ZipInfo]:
    prefix = CHECKPOINT_NAME + "/"
    members = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = member.filename.removeprefix(prefix)
            path = PurePosixPath(relative.removesuffix("/"))
            mode = member.external_attr >> 16
            if (
                relative == member.filename
                or path.is_absolute()
                or ".." in path.parts
                or (mode and (mode & 0o170000) not in {0, 0o040000, 0o100000})
            ):
                raise ValueError("Comet checkpoint archive layout differs")
            if member.is_dir():
                continue
            if relative in members:
                raise ValueError("Comet checkpoint archive layout differs")
            members[relative] = member
    return members


def _archive_matches_file(bundle, member, path: Path) -> bool:
    with bundle.open(member) as archived, path.open("rb") as extracted:
        while True:
            left = archived.read(1024 * 1024)
            right = extracted.read(1024 * 1024)
            if left != right:
                return False
            if not left:
                return True


def verify_checkpoint_archive(
    archive: Path, root: Path, expected_sha256: str
) -> dict[str, dict]:
    """Verify the full private archive, extraction, and frozen HF inventory.

    Args:
        archive: Operator-created Zip64 archive with one checkpoint prefix.
        root: Extracted checkpoint directory.
        expected_sha256: Frozen full archive SHA-256 from the campaign recipe.
    Returns:
        The exact verified checkpoint file inventory.
    Raises:
        ValueError: Archive identity, layout, extraction, or contents differ.
    """
    _regular_file(archive, "Comet checkpoint archive")
    if _digest(archive) != expected_sha256:
        raise ValueError("Comet checkpoint archive differs from the frozen recipe")
    manifest = load_checkpoint_manifest()
    members = _archive_members(archive)
    if set(members) != set(manifest["files"]):
        raise ValueError("Comet checkpoint archive file set differs")
    with zipfile.ZipFile(archive) as bundle:
        for relative, member in members.items():
            if not _archive_matches_file(bundle, member, root / relative):
                raise ValueError("Extracted Comet checkpoint differs from its archive")
    return _verify_checkpoint_files(root, manifest)


def task_identity(source_root: Path, task_id: int, task_name: str) -> dict:
    """Bind an allowed Comet task ID to its pinned prompt metadata.

    Args:
        source_root: Verified Comet source root.
        task_id: Original BEHAVIOR challenge task index.
        task_name: Exact source task name requested by the evaluator plan.
    Returns:
        Pinned task metadata consumed by the upstream wrapper.
    Raises:
        ValueError: The task is unsupported or its mapping differs.
    """
    if task_id not in SUPPORTED_TASK_IDS:
        raise ValueError("Comet checkpoint does not declare this task")
    mapping_path = source_root / "scripts/task_mapping.json"
    _regular_file(mapping_path, "Comet task mapping")
    mapping = json.loads(mapping_path.read_text())
    row = mapping.get(task_name)
    if not isinstance(row, dict) or row.get("task_index") != task_id:
        raise ValueError("Comet task name and ID mapping differ")
    if not isinstance(row.get("task"), str) or not row["task"].strip():
        raise ValueError("Comet task prompt metadata is malformed")
    return row


def policy_observation(observation: dict) -> dict:
    """Select only the three onboard RGB views and 2026 proprioception.

    Args:
        observation: Flattened official RGBDFullResWrapper observation.
    Returns:
        Three RGB images and the 61-element permitted proprioception vector.
    Raises:
        KeyError: A required camera or proprioception field is absent.
        ValueError: An input shape, dtype, or finite-value contract differs.
    """
    state = np.asarray(observation["robot_r1::proprio"])
    if state.shape != (61,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 61-element BEHAVIOR 2026 proprioception")
    result = {"robot_r1::proprio": state}
    for camera in CAMERAS:
        key = f"robot_r1::robot_r1:{camera}:Camera:0::rgb"
        image = np.asarray(observation[key])
        size = 720 if camera == "zed_link" else 480
        if image.shape not in {(size, size, 3), (size, size, 4)}:
            raise ValueError("Expected official full-resolution RGB camera shape")
        if image.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB camera pixels")
        result[key] = image[..., :3]
    return result


def validate_action(value) -> np.ndarray:
    """Convert the upstream one-action tensor to the evaluator's 23-vector.

    Args:
        value: Array or tensor returned by the pinned Comet wrapper.
    Returns:
        Finite numeric action with shape ``(23,)``.
    Raises:
        ValueError: The wrapper output shape, dtype, or values are invalid.
    """
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    action = np.asarray(value)
    if action.shape != (1, 23) or not np.issubdtype(action.dtype, np.number):
        raise ValueError("Comet wrapper must return one 23-element action")
    if not np.isfinite(action).all():
        raise ValueError("Comet wrapper returned a nonfinite action")
    return action[0]


def build_source_overlay(root: Path, overlay: Path) -> Path:
    """Copy pinned OpenPI code and replace its simulation-only index import.

    Args:
        root: Verified Comet source root.
        overlay: Empty destination for the runtime source overlay.
    Returns:
        Overlay root containing the patched ``openpi`` package.
    Raises:
        ValueError: The pinned import seam occurs other than exactly once.
    """
    verify_source(root)
    destination = overlay / "openpi"
    shutil.copytree(
        root / "src/openpi",
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    policy = destination / "policies/b1k_policy.py"
    old = "from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES"
    new = "from comet_policy import PROPRIOCEPTION_INDICES"
    source = policy.read_text()
    if source.count(old) != 1:
        raise ValueError("Unexpected pinned Comet proprioception import")
    policy.write_text(source.replace(old, new))
    return overlay


def _stage_adapters(output: Path) -> dict[str, str]:
    names = ("comet_policy.py", "comet_server.py", "comet12-checkpoint.json")
    identities = {}
    for name in names:
        source = Path(__file__).with_name(name)
        target = output / name
        shutil.copyfile(source, target)
        identities[name] = _digest(target)
    return identities


def _server_command(args, output: Path, task_id: int, task_name: str) -> list[str]:
    return [
        str(args.policy_python),
        str(output / "comet_server.py"),
        "--source-root",
        str(args.policy_root),
        "--checkpoint",
        str(args.policy_checkpoint),
        "--task-id",
        str(task_id),
        "--task-name",
        task_name,
        "--port",
        str(args.port),
    ]


def _campaign_task(args, plan: dict) -> tuple[int, str]:
    tasks = plan["recipe"].get("tasks")
    if plan["recipe"].get("split") not in {"development", "report"}:
        raise ValueError("Comet transfer requires development or report split")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("Comet transfer requires exactly one task")
    if tasks[0] != getattr(args, "policy_task_name", None):
        raise ValueError("Comet policy task must match the campaign task")
    mapping = json.loads((args.policy_root / "scripts/task_mapping.json").read_text())
    row = mapping.get(tasks[0])
    if not isinstance(row, dict) or type(row.get("task_index")) is not int:
        raise ValueError("Comet campaign task metadata is malformed")
    task_id = row["task_index"]
    task_identity(args.policy_root, task_id, tasks[0])
    registry_path = args.upstream_root / "docs/challenge/task_data.json"
    registry = json.loads(registry_path.read_text())
    current = [item.get("id") for item in registry.get("tasks", [])]
    if task_id >= len(current) or current[task_id] != tasks[0]:
        raise ValueError("Comet task differs from the current evaluator registry")
    return task_id, tasks[0]


def _verify_campaign_ports(args, plan: dict) -> None:
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Comet transfer requires at least one planned case")
    if any(case.get("policy_port") not in {None, args.port} for case in cases):
        raise ValueError("Comet policy port must match every planned case")


def prepare_policy(args, plan: dict, output: Path) -> list[str]:
    """Verify and stage Comet12 for the managed policy supervisor.

    Args:
        args: Managed policy source, interpreter, checkpoint, task, and port.
        plan: Frozen single-task development or report evaluation plan.
        output: Owner-only directory for adapters and public provenance.
    Returns:
        Exact subprocess argument vector for the staged Comet server.
    Raises:
        ValueError: Source, task, checkpoint, or plan contracts differ.
    """
    verify_source(args.policy_root)
    _verify_campaign_ports(args, plan)
    task_id, task_name = _campaign_task(args, plan)
    files = verify_checkpoint_archive(
        args.policy_archive,
        args.policy_checkpoint,
        plan["recipe"]["policy_checkpoint_sha256"],
    )
    adapters = _stage_adapters(output)
    command = _server_command(args, output, task_id, task_name)
    evidence = {
        "schema": "npa.behavior.comet12-policy.v1",
        "kind": "comet12",
        "source_commit": SOURCE_COMMIT,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_file_count": len(files),
        "checkpoint_inventory_sha256": adapters["comet12-checkpoint.json"],
        "task_id": task_id,
        "task_name": task_name,
        "adapters": adapters,
        "command": command,
        "redistribution": "private-runtime-checkpoint",
        "evaluation_status": "not_evaluated",
    }
    (output / "policy-provenance.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    return command
