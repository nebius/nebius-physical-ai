"""Run a source-pinned, paired LIBERO comparison for the Sylvest checkpoint.

The workflow intentionally does not treat a mixed-data checkpoint as an
improvement.  It either selects task names outside a supplied training inventory
or explicitly records that training coverage is unknown, runs the upstream
OpenVLA-OFT LIBERO evaluator once per matched task/initial-state pair, then
reports paired success differences and a factual Rerun recording.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from npa.clients.storage import StorageClient

WORKFLOW_SCHEMA = "npa.sylvest-oft-mixdata.v1"
OFT_SOURCE_REPOSITORY = "https://github.com/moojink/openvla-oft"
OFT_LICENSE_SHA256 = "53f449ef3886b1ccd71039d9a75a9f503fda1a6912e4095d1d72e2ac7b8767bd"
LIBERO_PLUS_SOURCE_REPOSITORY = "https://github.com/sylvestf/LIBERO-plus"
LIBERO_SOURCE_REPOSITORY = "https://github.com/Lifelong-Robot-Learning/LIBERO"
LIBERO_SOURCE_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_LICENSE_SHA256 = (
    "e2885fd30a08381b799c4a33385522b23d637b4051b8f9a7f9f2519944b68ff6"
)
LIBERO_CODE_LICENSE = "MIT"
LIBERO_DATASET_LICENSE = "CC-BY-4.0"
_LIBERO_TASK_MAP_FILE = Path("libero/libero/benchmark/libero_suite_task_map.py")
DLIMP_SOURCE_REPOSITORY = "https://github.com/kvablack/dlimp"
DLIMP_SOURCE_REVISION = "92e3eca97af3b14d0b6aa15182c0dc240407698d"
DLIMP_LICENSE_SHA256 = (
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)
_DLIMP_READY_FILE = ".npa-dlimp-ready.json"
_SOURCE_READY_FILE = ".npa-source-ready.json"
_DLIMP_DATASET_FILE = Path("dlimp/dataset.py")
_DLIMP_DETERMINISTIC_BEFORE = "options.deterministic = False"
_DLIMP_DETERMINISTIC_AFTER = "options.deterministic = True"
_CHECKPOINT_READY_FILE = ".npa-checkpoint-ready.json"
_CHECKPOINT_MUTABLE_FILES = {
    "config.json",
    "configuration_prismatic.py",
    "modeling_prismatic.py",
}
_ORIGINAL_LIBERO_SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
}


class SylvestComparisonError(RuntimeError):
    """Raised when paired checkpoint evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ArtifactLocation:
    """Describe a local or S3 workflow artifact location.

    Args:
        value: Original user-supplied URI or filesystem path.

    Returns:
        None.

    Raises:
        None.
    """

    value: str

    @property
    def is_s3(self) -> bool:
        """Return whether the artifact uses the supported S3 scheme.

        Args:
            None.

        Returns:
            True when the location is an S3 URI.

        Raises:
            None.
        """

        return self.value.startswith("s3://")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with temporary:
            temporary.write(_canonical_bytes(payload))
        os.replace(temporary.name, path)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SylvestComparisonError(
            f"invalid required JSON artifact: {path}: {exc}"
        ) from exc


def _storage() -> StorageClient:
    return StorageClient.from_environment()


def _materialize(location: str, destination: Path) -> Path:
    source = ArtifactLocation(location)
    if source.is_s3:
        if source.value.endswith("/"):
            _storage().download_directory(source.value, str(destination))
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        _storage().download_file(source.value, str(destination))
        return destination
    local = Path(source.value)
    if not local.exists():
        raise SylvestComparisonError(f"required local artifact does not exist: {local}")
    if local.is_dir():
        shutil.copytree(local, destination, dirs_exist_ok=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, destination)
    return destination


def _publish_tree(local_dir: Path, output_uri: str) -> None:
    target = ArtifactLocation(output_uri)
    if target.is_s3:
        _storage().upload_directory(str(local_dir), output_uri)
        return
    destination = Path(output_uri)
    shutil.copytree(local_dir, destination, dirs_exist_ok=True)


def _parse_initial_state_indices(value: str) -> list[int]:
    try:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise SylvestComparisonError(
            f"invalid comma-separated initial-state list: {value!r}"
        ) from exc
    if indices != list(range(len(indices))):
        raise SylvestComparisonError(
            "initial-state indices must be contiguous zero-based upstream trial indices"
        )
    return indices


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _task_names(payload: Any) -> set[str]:
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return set(payload)
    if isinstance(payload, dict) and isinstance(payload.get("task_names"), list):
        values = payload["task_names"]
        if all(isinstance(item, str) for item in values):
            return set(values)
    raise SylvestComparisonError(
        "training-task inventory must be a JSON string list or {'task_names': [...]}"
    )


def _verified_source_root(root: str, revision: str, repository: str) -> Path:
    path = Path(root).resolve()
    if not (path / ".git").exists():
        raise SylvestComparisonError(
            f"runtime source must be an operator-provided git checkout: {path}"
        )
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    actual = completed.stdout.strip()
    if completed.returncode or actual != revision:
        raise SylvestComparisonError(
            f"source identity mismatch for {repository}: expected {revision}, found {actual or 'unreadable'}"
        )
    return path


def _runtime_source_metadata(
    root: Path, repository: str, revision: str
) -> dict[str, str]:
    return {
        "repository": repository,
        "revision": revision,
        "tree": str(root),
    }


def _materialize_runtime_source(root: str, revision: str, repository: str) -> Path:
    """Fetch an immutable, source-only runtime checkout with an atomic marker.

    Licensed source is fetched on the operator's worker instead of copied into
    an image. A cache directory is never reused without both its Git identity
    and the marker written after that identity was verified.
    """

    destination = Path(root).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    import subprocess

    lock_path = destination.parent / f".{destination.name}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        marker = destination / _SOURCE_READY_FILE
        if marker.is_file():
            payload = _read_json(marker)
            if isinstance(payload, dict) and payload == _runtime_source_metadata(
                destination, repository, revision
            ):
                return _verified_source_root(str(destination), revision, repository)
            raise SylvestComparisonError(
                "runtime source cache has an incompatible ready marker"
            )
        if destination.exists():
            raise SylvestComparisonError(
                "runtime source cache exists without a matching atomic ready marker"
            )
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            for argv in (
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    repository,
                    str(staging),
                ],
                ["git", "-C", str(staging), "checkout", "--detach", revision],
            ):
                result = subprocess.run(
                    argv, capture_output=True, check=False, text=True
                )
                if result.returncode:
                    raise SylvestComparisonError(
                        f"failed to materialize pinned runtime source for {repository}"
                    )
            _verified_source_root(str(staging), revision, repository)
            _write_json(
                staging / _SOURCE_READY_FILE,
                _runtime_source_metadata(destination, repository, revision),
            )
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination


def _require_libero_plus_license(root: Path) -> Path:
    candidates = [
        root / "LICENSE",
        root / "LICENSE.md",
        root / "COPYING",
        root / "NOTICE",
    ]
    license_file = next((path for path in candidates if path.is_file()), None)
    if license_file is None:
        raise SylvestComparisonError(
            "LIBERO-Plus source has no license file. Do not execute that source "
            "as a benchmark until the upstream authors publish or identify an "
            "applicable license. A separately scoped author-published asset-only "
            "compatibility check through original MIT LIBERO does not license this "
            "source or establish benchmark equivalence; this is not an NPA EULA "
            "requirement."
        )
    return license_file


def _require_original_libero_license(root: Path) -> Path:
    """Verify the exact MIT license of the original LIBERO source."""

    license_file = root / "LICENSE"
    if (
        not license_file.is_file()
        or _sha256_file(license_file) != LIBERO_LICENSE_SHA256
    ):
        raise SylvestComparisonError(
            "original LIBERO source is missing its expected MIT LICENSE"
        )
    return license_file


def _require_oft_license(root: Path) -> Path:
    """Verify the exact OpenVLA-OFT MIT notice before executing its evaluator."""

    license_file = root / "LICENSE"
    if not license_file.is_file() or _sha256_file(license_file) != OFT_LICENSE_SHA256:
        raise SylvestComparisonError(
            "OpenVLA-OFT source is missing its expected MIT LICENSE"
        )
    return license_file


def _original_libero_suite_cases(
    root: Path, suite: str
) -> tuple[list[dict[str, Any]], Path]:
    """Read the source-pinned original LIBERO task map without importing its runtime."""

    if suite not in _ORIGINAL_LIBERO_SUITES:
        raise SylvestComparisonError(
            "original LIBERO comparison supports libero_spatial, libero_object, "
            "libero_goal, or libero_10 only"
        )
    task_map_path = root / _LIBERO_TASK_MAP_FILE
    try:
        tree = ast.parse(task_map_path.read_text(encoding="utf-8"), str(task_map_path))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "libero_task_map"
                for target in node.targets
            )
        )
        task_map = ast.literal_eval(assignment.value)
    except (OSError, StopIteration, SyntaxError, ValueError) as exc:
        raise SylvestComparisonError(
            "unable to read the original LIBERO task map as a literal source mapping"
        ) from exc
    names = task_map.get(suite) if isinstance(task_map, dict) else None
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise SylvestComparisonError(
            f"original LIBERO task map has no usable task list for {suite!r}"
        )
    return (
        [
            {
                "id": index,
                "name": name,
                "category": suite,
                "difficulty_level": 0,
            }
            for index, name in enumerate(names)
        ],
        task_map_path,
    )


def _original_libero_initial_state_hashes(
    root: Path, suite: str, cases: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """Bind each original LIBERO task identity to its upstream init-state bytes."""

    hashes: dict[str, str] = {}
    for case in cases:
        name = str(case["name"])
        path = root / "libero/libero/init_files" / suite / f"{name}.pruned_init"
        if not path.is_file():
            raise SylvestComparisonError(
                f"original LIBERO initial-state payload is missing for {name!r}"
            )
        hashes[name] = _sha256_file(path)
    return hashes


def _require_dlimp_license(root: Path) -> Path:
    """Verify the exact Apache-2.0 dlimp parent license before deriving it."""

    license_file = root / "LICENSE"
    if not license_file.is_file() or _sha256_file(license_file) != DLIMP_LICENSE_SHA256:
        raise SylvestComparisonError(
            "licensed dlimp parent is missing its expected Apache-2.0 LICENSE"
        )
    return license_file


def _tree_inventory(
    root: Path, *, exclude: set[str] | None = None
) -> list[dict[str, Any]]:
    excluded = exclude or set()
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name not in excluded
    ]
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(files)
    ]


def _dlimp_derivative_metadata(source: Path, derivative: Path) -> dict[str, Any]:
    source_dataset = source / _DLIMP_DATASET_FILE
    derivative_dataset = derivative / _DLIMP_DATASET_FILE
    return {
        "schema": WORKFLOW_SCHEMA,
        "kind": "licensed_dlimp_deterministic_derivative",
        "source": {
            "repository": DLIMP_SOURCE_REPOSITORY,
            "revision": DLIMP_SOURCE_REVISION,
            "license_file": "LICENSE",
            "license_sha256": _sha256_file(source / "LICENSE"),
            "dataset_sha256_before": _sha256_file(source_dataset),
        },
        "modification": {
            "path": str(_DLIMP_DATASET_FILE),
            "replacement": f"{_DLIMP_DETERMINISTIC_BEFORE} -> {_DLIMP_DETERMINISTIC_AFTER}",
            "dataset_sha256_after": _sha256_file(derivative_dataset),
            "notice_file": "NPA_MODIFICATIONS.md",
        },
        "inventory": _tree_inventory(derivative, exclude={_DLIMP_READY_FILE}),
    }


def _dlimp_derivative_is_ready(derivative: Path) -> bool:
    ready = derivative / _DLIMP_READY_FILE
    if not ready.is_file():
        return False
    payload = _read_json(ready)
    if not isinstance(payload, dict):
        return False
    source = payload.get("source")
    modification = payload.get("modification")
    if not isinstance(source, dict) or not isinstance(modification, dict):
        return False
    dataset = derivative / _DLIMP_DATASET_FILE
    if not dataset.is_file():
        return False
    text = dataset.read_text(encoding="utf-8")
    return (
        source.get("repository") == DLIMP_SOURCE_REPOSITORY
        and source.get("revision") == DLIMP_SOURCE_REVISION
        and source.get("license_sha256") == DLIMP_LICENSE_SHA256
        and modification.get("replacement")
        == f"{_DLIMP_DETERMINISTIC_BEFORE} -> {_DLIMP_DETERMINISTIC_AFTER}"
        and _DLIMP_DETERMINISTIC_AFTER in text
        and _DLIMP_DETERMINISTIC_BEFORE not in text
        and payload.get("inventory")
        == _tree_inventory(derivative, exclude={_DLIMP_READY_FILE})
    )


def _prepare_deterministic_dlimp_runtime(source: Path, destination: Path) -> Path:
    """Create one Apache-noticed deterministic dlimp derivative atomically.

    The OpenVLA fork changed only this option. NPA applies the same one-line
    behavior to the Apache-2.0 parent instead of importing the unlicensed fork.
    """

    _require_dlimp_license(source)
    dataset = source / _DLIMP_DATASET_FILE
    source_text = dataset.read_text(encoding="utf-8")
    if source_text.count(_DLIMP_DETERMINISTIC_BEFORE) != 1:
        raise SylvestComparisonError(
            "licensed dlimp parent no longer has the expected deterministic option"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock_path = destination.parent / f".{destination.name}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _dlimp_derivative_is_ready(destination):
            return destination
        if destination.exists():
            raise SylvestComparisonError(
                "dlimp runtime cache exists without a matching atomic ready marker"
            )
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            shutil.copytree(
                source,
                staging,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git", _SOURCE_READY_FILE),
            )
            derived_dataset = staging / _DLIMP_DATASET_FILE
            derived_text = derived_dataset.read_text(encoding="utf-8")
            derived_dataset.write_text(
                derived_text.replace(
                    _DLIMP_DETERMINISTIC_BEFORE, _DLIMP_DETERMINISTIC_AFTER, 1
                ),
                encoding="utf-8",
            )
            rendered = derived_dataset.read_text(encoding="utf-8")
            if (
                _DLIMP_DETERMINISTIC_BEFORE in rendered
                or _DLIMP_DETERMINISTIC_AFTER not in rendered
            ):
                raise SylvestComparisonError(
                    "failed to apply the deterministic dlimp runtime override"
                )
            (staging / "NPA_MODIFICATIONS.md").write_text(
                "# NPA modification notice\n\n"
                "Derived from kvablack/dlimp@92e3eca97af3b14d0b6aa15182c0dc240407698d "
                "under Apache-2.0. NPA changes dlimp/dataset.py only: "
                "options.deterministic = False to options.deterministic = True.\n",
                encoding="utf-8",
            )
            metadata = _dlimp_derivative_metadata(source, staging)
            _write_json(staging / _DLIMP_READY_FILE, metadata)
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination


def _prepare_protocol(args: argparse.Namespace, output: Path) -> None:
    source = _comparison_source(args.benchmark, args.libero_revision)
    if args.benchmark == "original_libero":
        libero_root = _materialize_runtime_source(
            args.libero_root, args.libero_revision, source["repository"]
        )
        license_file = _require_original_libero_license(libero_root)
        suite_cases, suite_map_path = _original_libero_suite_cases(
            libero_root, args.task_suite
        )
        initial_state_hashes = _original_libero_initial_state_hashes(
            libero_root, args.task_suite, suite_cases
        )
    else:
        libero_root = _verified_source_root(
            args.libero_root, args.libero_revision, source["repository"]
        )
        license_file = _require_libero_plus_license(libero_root)
        suite_map_path = (
            libero_root / "libero/libero/benchmark/task_classification.json"
        )
        classification = _read_json(suite_map_path)
        suite_cases = (
            classification.get(args.task_suite)
            if isinstance(classification, dict)
            else None
        )
        if not isinstance(suite_cases, list):
            raise SylvestComparisonError(
                f"classification has no task list for {args.task_suite!r}"
            )
        initial_state_hashes = {}
    openvla_root = _materialize_runtime_source(
        args.openvla_oft_root, args.openvla_oft_revision, OFT_SOURCE_REPOSITORY
    )
    openvla_license = _require_oft_license(openvla_root)
    dlimp_root = _materialize_runtime_source(
        args.dlimp_root, args.dlimp_revision, DLIMP_SOURCE_REPOSITORY
    )
    dlimp_license = _require_dlimp_license(dlimp_root)
    training_names, inventory_sha256 = _training_inventory(args.training_task_ids_uri)
    if args.comparison_scope == "held_out":
        if not training_names:
            raise SylvestComparisonError(
                "held_out comparison_scope requires a non-empty authoritative training-task inventory"
            )
        selected_cases = [
            row
            for row in suite_cases
            if str(row.get("name") or "") not in training_names
        ]
        population_definition = (
            "task name absent from the supplied immutable mix-SFT training inventory"
        )
    elif args.comparison_scope == "training_coverage_unknown":
        if training_names:
            raise SylvestComparisonError(
                "training_coverage_unknown comparison_scope must not receive a training-task inventory"
            )
        selected_cases = suite_cases
        population_definition = "all selected suite tasks; candidate training coverage is unknown and may be in-distribution"
    else:
        raise SylvestComparisonError(
            "comparison_scope must be held_out or training_coverage_unknown"
        )
    if not selected_cases:
        raise SylvestComparisonError(
            "selected comparison population has no benchmark tasks"
        )
    protocol_cases = _protocol_cases(
        selected_cases,
        _parse_initial_state_indices(args.initial_state_indices),
        initial_state_hashes,
    )
    checkpoints = _prepare_checkpoints(args)
    protocol = {
        "schema": WORKFLOW_SCHEMA,
        "kind": "paired_libero_protocol",
        "benchmark": args.benchmark,
        "task_suite": args.task_suite,
        "initial_state_indices": _parse_initial_state_indices(
            args.initial_state_indices
        ),
        "cases": protocol_cases,
        "comparison_scope": args.comparison_scope,
        "selected_task_count": len(selected_cases),
        "training_inventory_count": len(training_names),
        "training_inventory_sha256": inventory_sha256,
        "checkpoints": checkpoints,
        "runtime_sources": {
            "openvla_oft": {
                "repository": OFT_SOURCE_REPOSITORY,
                "revision": args.openvla_oft_revision,
                "license_file": openvla_license.name,
                "license_sha256": _sha256_file(openvla_license),
            },
            "dlimp": {
                "repository": DLIMP_SOURCE_REPOSITORY,
                "revision": args.dlimp_revision,
                "license_file": dlimp_license.name,
                "license_sha256": _sha256_file(dlimp_license),
            },
        },
        "source": {
            **source,
            "suite_map_sha256": _sha256_file(suite_map_path),
            "license_file": license_file.name,
            "license_sha256": _sha256_file(license_file),
            **(
                {
                    "code_license": LIBERO_CODE_LICENSE,
                    "initial_state_data_license": LIBERO_DATASET_LICENSE,
                }
                if args.benchmark == "original_libero"
                else {}
            ),
        },
        "evaluation_config": _evaluation_config(args),
        "population_definition": population_definition,
    }
    protocol["protocol_sha256"] = hashlib.sha256(_canonical_bytes(protocol)).hexdigest()
    notices = _third_party_notices(
        protocol, license_file, openvla_license, dlimp_license
    )
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "notices.json", notices)
    _write_json(output / "provenance.json", _prepare_provenance(args, protocol))


def _third_party_notices(
    protocol: Mapping[str, Any],
    benchmark_license: Path,
    oft_license: Path,
    dlimp_license: Path,
) -> dict[str, Any]:
    """Produce attribution carried from preparation to the final report stage."""

    source = protocol["source"]
    runtime_sources = protocol["runtime_sources"]
    checkpoints = protocol["checkpoints"]
    return {
        "schema": WORKFLOW_SCHEMA,
        "kind": "third_party_notices",
        "protocol_sha256": protocol["protocol_sha256"],
        "components": [
            {
                "name": "LIBERO",
                "repository": source["repository"],
                "revision": source["revision"],
                "license": source.get("code_license", "unresolved"),
                "license_file": benchmark_license.name,
                "license_sha256": _sha256_file(benchmark_license),
                "license_text": benchmark_license.read_text(encoding="utf-8"),
                "data_license": source.get("initial_state_data_license", "unresolved"),
                "credit": "Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone",
                "citation": "Liu et al., LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning (2023)",
            },
            {
                "name": "OpenVLA-OFT",
                "repository": runtime_sources["openvla_oft"]["repository"],
                "revision": runtime_sources["openvla_oft"]["revision"],
                "license": "MIT",
                "license_file": oft_license.name,
                "license_sha256": _sha256_file(oft_license),
                "license_text": oft_license.read_text(encoding="utf-8"),
                "credit": "Moo Jin Kim, Chelsea Finn, and Percy Liang",
                "citation": "Kim, Finn, and Liang, Fine-Tuning Vision-Language-Action Models (2025)",
            },
            {
                "name": "dlimp deterministic private derivative",
                "repository": runtime_sources["dlimp"]["repository"],
                "revision": runtime_sources["dlimp"]["revision"],
                "license": "Apache-2.0",
                "license_file": dlimp_license.name,
                "license_sha256": _sha256_file(dlimp_license),
                "license_text": dlimp_license.read_text(encoding="utf-8"),
                "credit": "Kevin Black",
                "modification": "dlimp/dataset.py: options.deterministic = False -> True; preserve LICENSE and NPA_MODIFICATIONS.md",
            },
            {
                "name": "Official OpenVLA-OFT baseline checkpoint",
                "repository": f"https://huggingface.co/{checkpoints['baseline']['repo_id']}",
                "revision": checkpoints["baseline"]["revision"],
                "credit": "OpenVLA-OFT authors",
                "lineage": "stage-one checkpoint inventory",
            },
            {
                "name": "Sylvest mixed-data candidate checkpoint",
                "repository": f"https://huggingface.co/{checkpoints['candidate']['repo_id']}",
                "revision": checkpoints["candidate"]["revision"],
                "credit": "Sylvest",
                "lineage": "stage-one checkpoint inventory",
            },
        ],
    }


def _prepare_checkpoints(args: argparse.Namespace) -> dict[str, Mapping[str, Any]]:
    """Fetch and inventory both exact model inputs before either rollout arm.

    The later stages independently revalidate their local cache against these
    complete inventories. That keeps a successful pre-rollout verification from
    becoming a substitute for validating the bytes a GPU worker actually loads.
    """

    requested = {
        "baseline": (args.baseline_checkpoint_id, args.baseline_checkpoint_revision),
        "candidate": (args.candidate_checkpoint_id, args.candidate_checkpoint_revision),
    }
    if requested["baseline"] == requested["candidate"]:
        raise SylvestComparisonError(
            "baseline and candidate checkpoint identities must be distinct"
        )
    root = Path(args.model_cache_root)
    return {
        role: _checkpoint_provenance(_snapshot_checkpoint(repo_id, revision, root))
        for role, (repo_id, revision) in requested.items()
    }


def _training_inventory(uri: str) -> tuple[set[str], str | None]:
    if not uri:
        return set(), None
    with tempfile.TemporaryDirectory(prefix="sylvest-training-inventory-") as temporary:
        inventory = _materialize(uri, Path(temporary) / "training.json")
        return _task_names(_read_json(inventory)), _sha256_file(inventory)


def _protocol_cases(
    rows: Iterable[Mapping[str, Any]],
    initial_state_indices: Iterable[int],
    initial_state_hashes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in sorted(
        rows, key=lambda item: (int(item.get("id", 0)), str(item.get("name", "")))
    ):
        name = str(row.get("name") or "")
        if not name:
            raise SylvestComparisonError(
                "benchmark task map contains a task without a name"
            )
        for initial_state_index in initial_state_indices:
            cases.append(
                {
                    "case_id": f"{name}::initial-state={initial_state_index}",
                    "task_name": name,
                    "category": str(row.get("category") or "unspecified"),
                    "difficulty_level": int(row.get("difficulty_level") or 0),
                    "initial_state_index": initial_state_index,
                    **(
                        {"initial_state_source_sha256": initial_state_hashes[name]}
                        if initial_state_hashes and name in initial_state_hashes
                        else {}
                    ),
                }
            )
    return cases


def _comparison_source(benchmark: str, revision: str) -> dict[str, str]:
    if benchmark == "original_libero":
        if revision != LIBERO_SOURCE_REVISION:
            raise SylvestComparisonError(
                f"original LIBERO requires revision {LIBERO_SOURCE_REVISION}"
            )
        return {
            "repository": LIBERO_SOURCE_REPOSITORY,
            "revision": LIBERO_SOURCE_REVISION,
        }
    if benchmark == "libero_plus":
        return {
            "repository": LIBERO_PLUS_SOURCE_REPOSITORY,
            "revision": revision,
        }
    raise SylvestComparisonError("benchmark must be original_libero or libero_plus")


def _evaluation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "env_image_resolution": args.env_image_resolution,
        "center_crop": args.center_crop,
        "num_images_in_input": args.num_images_in_input,
        "num_open_loop_steps": args.num_open_loop_steps,
        "action_transform": "openvla_normalize_then_invert_gripper",
    }


def _prepare_provenance(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": WORKFLOW_SCHEMA,
        "producer": "npa.workflows.sylvest_oft_mixdata.prepare",
        "protocol_sha256": protocol["protocol_sha256"],
        "limitations": [
            "A held-out result is available only with an authoritative supplied training inventory.",
            "The published checkpoint card does not itself provide a training-task split.",
            "training_coverage_unknown results may include in-distribution tasks.",
            "Initial-state indices are upstream LIBERO trial indices, not independently sampled seeds.",
            "No benchmark improvement is inferred at preparation time.",
        ],
        "sources": {
            "benchmark": args.benchmark,
            "libero_revision": args.libero_revision,
            "openvla_oft_revision": args.openvla_oft_revision,
            "dlimp_revision": args.dlimp_revision,
            "training_task_ids_uri": args.training_task_ids_uri,
            "comparison_scope": protocol["comparison_scope"],
            "evaluation_config": protocol["evaluation_config"],
        },
    }


def _snapshot_checkpoint(repo_id: str, revision: str, cache_root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SylvestComparisonError(
            "huggingface_hub is required for checkpoint retrieval"
        ) from exc
    safe_id = repo_id.replace("/", "--")
    repository_cache = cache_root / safe_id
    target = repository_cache / revision
    ready = target / _CHECKPOINT_READY_FILE
    repository_cache.mkdir(parents=True, exist_ok=True)
    lock_path = repository_cache / f".{revision}.lock"
    # A run-shared filesystem cache must never expose an unmarked partial
    # snapshot. A lock serializes the fetch; the ready marker is atomically
    # replaced only after its required payload is present. Hugging Face resumes
    # an interrupted local-dir fetch on a later retry.
    import fcntl

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _checkpoint_is_ready(target, ready, repo_id, revision):
            return target
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(target))
        _write_json(ready, _checkpoint_metadata(target, repo_id, revision))
    return target


def _checkpoint_is_ready(
    target: Path, ready: Path, repo_id: str, revision: str
) -> bool:
    if not ready.is_file():
        return False
    metadata = _read_json(ready)
    if not isinstance(metadata, dict):
        return False
    if metadata.get("repo_id") != repo_id or metadata.get("revision") != revision:
        return False
    return metadata.get("inventory") == _checkpoint_inventory(target)


def _checkpoint_metadata(target: Path, repo_id: str, revision: str) -> dict[str, Any]:
    required_files = _required_checkpoint_files(target)
    inventory = _checkpoint_inventory(target)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "required_files": required_files,
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(_canonical_bytes(inventory)).hexdigest(),
    }


def _required_checkpoint_files(target: Path) -> list[str]:
    required = [target / "config.json", target / "dataset_statistics.json"]
    action_heads = sorted(target.glob("action_head--*_checkpoint.pt"))
    proprio_projectors = sorted(target.glob("proprio_projector--*_checkpoint.pt"))
    adapter = target / "lora_adapter"
    required.extend(action_heads + proprio_projectors)
    if len(action_heads) != 1 or len(proprio_projectors) != 1 or not adapter.is_dir():
        raise SylvestComparisonError(
            "checkpoint is not an OpenVLA-OFT package with one action head, one proprio projector, and lora_adapter"
        )
    adapter_files = sorted(path for path in adapter.rglob("*") if path.is_file())
    if not adapter_files:
        raise SylvestComparisonError(
            "OpenVLA-OFT checkpoint has an empty lora_adapter directory"
        )
    required.extend(adapter_files)
    missing = [path for path in required if not path.is_file() or path.is_symlink()]
    if missing:
        names = ", ".join(str(path.relative_to(target)) for path in missing)
        raise SylvestComparisonError(
            f"checkpoint is missing required regular files: {names}"
        )
    return [str(path.relative_to(target)) for path in required]


def _checkpoint_inventory(target: Path) -> list[dict[str, Any]]:
    files = [
        path
        for path in target.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name != _CHECKPOINT_READY_FILE
        and ".cache" not in path.relative_to(target).parts
    ]
    return [
        {
            "path": str(path.relative_to(target)),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(files)
    ]


def _checkpoint_provenance(checkpoint: Path) -> Mapping[str, Any]:
    metadata = _read_json(checkpoint / _CHECKPOINT_READY_FILE)
    if not isinstance(metadata, dict) or "inventory_sha256" not in metadata:
        raise SylvestComparisonError(
            "checkpoint cache has no verified provenance marker"
        )
    return metadata


def _copy_checkpoint_workspace(snapshot: Path, destination: Path) -> Path:
    for source in sorted(path for path in snapshot.rglob("*") if path.is_file()):
        relative = source.relative_to(snapshot)
        if source.name == _CHECKPOINT_READY_FILE or ".cache" in relative.parts:
            continue
        if source.is_symlink():
            raise SylvestComparisonError(
                f"checkpoint contains a symlinked payload: {relative}"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(relative) in _CHECKPOINT_MUTABLE_FILES:
            shutil.copy2(source, target)
        else:
            _link_or_copy(source, target)
    return destination


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _runtime_modules(
    openvla_root: Path, libero_root: Path, deterministic_dlimp_root: Path
) -> Any:
    importlib.invalidate_caches()
    loaded_dlimp = sys.modules.get("dlimp")
    if loaded_dlimp is not None:
        loaded_file = getattr(loaded_dlimp, "__file__", None)
        if not loaded_file or not Path(loaded_file).resolve().is_relative_to(
            deterministic_dlimp_root.resolve()
        ):
            raise SylvestComparisonError(
                "a non-deterministic or unverified dlimp module was imported before rollout"
            )
    paths = [
        str(deterministic_dlimp_root),
        str(libero_root / "libero"),
        str(openvla_root),
    ]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("experiments.robot.libero.run_libero_eval")


def _rollout(args: argparse.Namespace, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sylvest-protocol-") as temporary:
        protocol_path = _materialize(
            args.protocol_uri, Path(temporary) / "protocol.json"
        )
        protocol = _read_json(protocol_path)
    _validate_protocol(protocol)
    source = protocol["source"]
    if not isinstance(source, Mapping):
        raise SylvestComparisonError("protocol has no benchmark-source provenance")
    benchmark = str(protocol.get("benchmark") or "")
    if args.benchmark != benchmark:
        raise SylvestComparisonError(
            "supplied benchmark does not match protocol benchmark provenance"
        )
    expected_source = _comparison_source(benchmark, args.libero_revision)
    observed_source = {
        "repository": source.get("repository"),
        "revision": source.get("revision"),
    }
    if observed_source != expected_source:
        raise SylvestComparisonError(
            "protocol benchmark provenance does not match the supplied source"
        )
    if benchmark == "original_libero":
        libero_root = _materialize_runtime_source(
            args.libero_root, args.libero_revision, expected_source["repository"]
        )
        _require_original_libero_license(libero_root)
    else:
        libero_root = _verified_source_root(
            args.libero_root, args.libero_revision, expected_source["repository"]
        )
        _require_libero_plus_license(libero_root)
    dlimp_source = _materialize_runtime_source(
        args.dlimp_root, args.dlimp_revision, DLIMP_SOURCE_REPOSITORY
    )
    _require_dlimp_license(dlimp_source)
    deterministic_dlimp_root = _prepare_deterministic_dlimp_runtime(
        dlimp_source, Path(args.dlimp_runtime_root)
    )
    openvla_root = _materialize_runtime_source(
        args.openvla_oft_root, args.openvla_oft_revision, OFT_SOURCE_REPOSITORY
    )
    _require_oft_license(openvla_root)
    _require_prepared_runtime_sources(protocol, args)
    checkpoint_snapshot = _snapshot_checkpoint(
        args.checkpoint_id, args.checkpoint_revision, Path(args.model_cache_root)
    )
    checkpoint_provenance = _checkpoint_provenance(checkpoint_snapshot)
    _require_prepared_checkpoint(protocol, checkpoint_provenance)
    with tempfile.TemporaryDirectory(
        prefix="sylvest-checkpoint-workspace-"
    ) as temporary:
        runtime_workspace = Path(temporary)
        checkpoint = _copy_checkpoint_workspace(
            checkpoint_snapshot, runtime_workspace / "checkpoint"
        )
        (runtime_workspace / "prismatic").symlink_to(
            openvla_root / "prismatic", target_is_directory=True
        )
        results = _run_upstream_rollouts(
            protocol,
            checkpoint,
            openvla_root,
            libero_root,
            deterministic_dlimp_root,
            output,
            runtime_workspace,
        )
        _collect_rollout_clips(runtime_workspace, output, expected_count=len(results))
    payload = {
        "schema": WORKFLOW_SCHEMA,
        "kind": "openvla_oft_libero_rollouts",
        "checkpoint": {
            "id": args.checkpoint_id,
            "revision": args.checkpoint_revision,
            "inventory_sha256": checkpoint_provenance["inventory_sha256"],
            "required_files": checkpoint_provenance["required_files"],
        },
        "protocol_sha256": protocol["protocol_sha256"],
        "task_suite": protocol["task_suite"],
        "benchmark": benchmark,
        "source": protocol["source"],
        "runtime_sources": protocol["runtime_sources"],
        "comparison_scope": protocol["comparison_scope"],
        "evaluation_config": protocol["evaluation_config"],
        "episodes": results,
        "summary": _rollout_summary(results),
    }
    _write_json(output / "rollouts.json", payload)
    _write_json(output / "clips" / "manifest.json", _clip_manifest(output / "clips"))
    _write_json(output / "provenance.json", _rollout_provenance(args, payload))


def _require_prepared_checkpoint(
    protocol: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    """Require the GPU-loaded checkpoint bytes to match stage-one inventory."""

    checkpoints = protocol.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise SylvestComparisonError("protocol lacks prepared checkpoint provenance")
    matches = [
        value
        for value in checkpoints.values()
        if isinstance(value, Mapping)
        and value.get("repo_id") == observed.get("repo_id")
        and value.get("revision") == observed.get("revision")
    ]
    if len(matches) != 1 or dict(matches[0]) != dict(observed):
        raise SylvestComparisonError(
            "GPU checkpoint inventory differs from stage-one prepared provenance"
        )


def _require_prepared_runtime_sources(
    protocol: Mapping[str, Any], args: argparse.Namespace
) -> None:
    """Bind evaluator dependencies on the GPU worker to stage-one source pins."""

    sources = protocol.get("runtime_sources")
    expected = {
        "openvla_oft": {
            "repository": OFT_SOURCE_REPOSITORY,
            "revision": args.openvla_oft_revision,
            "license_sha256": OFT_LICENSE_SHA256,
        },
        "dlimp": {
            "repository": DLIMP_SOURCE_REPOSITORY,
            "revision": args.dlimp_revision,
            "license_sha256": DLIMP_LICENSE_SHA256,
        },
    }
    if not isinstance(sources, Mapping):
        raise SylvestComparisonError(
            "protocol lacks prepared runtime source provenance"
        )
    for name, identity in expected.items():
        observed = sources.get(name)
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != value for key, value in identity.items()
        ):
            raise SylvestComparisonError(
                "GPU runtime source differs from stage-one prepared provenance"
            )


def _validate_protocol(protocol: Any) -> None:
    if not isinstance(protocol, dict) or protocol.get("schema") != WORKFLOW_SCHEMA:
        raise SylvestComparisonError("unsupported protocol schema")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SylvestComparisonError("protocol has no evaluation cases")
    comparison_scope = protocol.get("comparison_scope")
    if comparison_scope not in {"held_out", "training_coverage_unknown"}:
        raise SylvestComparisonError("protocol has no valid comparison scope")
    benchmark = protocol.get("benchmark")
    source = protocol.get("source")
    if benchmark not in {"original_libero", "libero_plus"} or not isinstance(
        source, dict
    ):
        raise SylvestComparisonError(
            "protocol has no valid benchmark-source provenance"
        )
    if source.get("repository") != _comparison_source(
        str(benchmark), str(source.get("revision") or "")
    ).get("repository"):
        raise SylvestComparisonError("protocol benchmark source is not canonical")
    if benchmark == "original_libero" and any(
        not isinstance(case, Mapping)
        or not isinstance(case.get("initial_state_source_sha256"), str)
        or not case["initial_state_source_sha256"]
        for case in cases
    ):
        raise SylvestComparisonError(
            "original LIBERO protocol lacks initial-state byte provenance"
        )
    checkpoints = protocol.get("checkpoints")
    if (
        not isinstance(checkpoints, dict)
        or set(checkpoints) != {"baseline", "candidate"}
        or any(
            not isinstance(value, dict)
            or not isinstance(value.get("repo_id"), str)
            or not isinstance(value.get("revision"), str)
            or not isinstance(value.get("inventory_sha256"), str)
            for value in checkpoints.values()
        )
    ):
        raise SylvestComparisonError("protocol lacks prepared checkpoint inventories")
    runtime_sources = protocol.get("runtime_sources")
    expected_sources = {
        "openvla_oft": (OFT_SOURCE_REPOSITORY, OFT_LICENSE_SHA256),
        "dlimp": (DLIMP_SOURCE_REPOSITORY, DLIMP_LICENSE_SHA256),
    }
    if (
        not isinstance(runtime_sources, dict)
        or set(runtime_sources) != set(expected_sources)
        or any(
            not isinstance(runtime_sources[name], dict)
            or runtime_sources[name].get("repository") != repository
            or not isinstance(runtime_sources[name].get("revision"), str)
            or runtime_sources[name].get("license_sha256") != license_sha256
            for name, (repository, license_sha256) in expected_sources.items()
        )
    ):
        raise SylvestComparisonError("protocol lacks prepared runtime source pins")
    evaluation_config = protocol.get("evaluation_config")
    if not isinstance(evaluation_config, dict) or not {
        "seed",
        "env_image_resolution",
        "center_crop",
        "num_images_in_input",
        "num_open_loop_steps",
        "action_transform",
    } <= set(evaluation_config):
        raise SylvestComparisonError("protocol lacks paired evaluator configuration")
    if comparison_scope == "held_out" and not protocol.get("training_inventory_sha256"):
        raise SylvestComparisonError(
            "held-out protocol lacks immutable training-inventory provenance"
        )
    expected = dict(protocol)
    actual = str(expected.pop("protocol_sha256", ""))
    if not actual or hashlib.sha256(_canonical_bytes(expected)).hexdigest() != actual:
        raise SylvestComparisonError(
            "protocol hash does not match its declared content"
        )


def _run_upstream_rollouts(
    protocol: Mapping[str, Any],
    checkpoint: Path,
    openvla_root: Path,
    libero_root: Path,
    deterministic_dlimp_root: Path,
    output: Path,
    runtime_workspace: Path,
) -> list[dict[str, Any]]:
    evaluator = _runtime_modules(openvla_root, libero_root, deterministic_dlimp_root)
    evaluation_config = protocol["evaluation_config"]
    if not isinstance(evaluation_config, Mapping):
        raise SylvestComparisonError("protocol lacks evaluator configuration")
    prior_cwd = Path.cwd()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(runtime_workspace)
    try:
        cfg = evaluator.GenerateConfig(
            pretrained_checkpoint=str(checkpoint),
            task_suite_name=protocol["task_suite"],
            num_trials_per_task=len(protocol["initial_state_indices"]),
            seed=int(evaluation_config["seed"]),
            env_img_res=int(evaluation_config["env_image_resolution"]),
            center_crop=bool(evaluation_config["center_crop"]),
            num_images_in_input=int(evaluation_config["num_images_in_input"]),
            num_open_loop_steps=int(evaluation_config["num_open_loop_steps"]),
            use_wandb=False,
            local_log_dir=str(output / "logs"),
        )
        evaluator.validate_config(cfg)
        model, action_head, proprio, noisy, processor = evaluator.initialize_model(cfg)
        task_suite = evaluator.benchmark.get_benchmark_dict()[protocol["task_suite"]]()
        task_ids = {
            name: index for index, name in enumerate(task_suite.get_task_names())
        }
        log_file, _, _ = evaluator.setup_logging(cfg)
        try:
            results = _run_cases(
                evaluator,
                cfg,
                protocol["cases"],
                task_ids,
                model,
                action_head,
                proprio,
                noisy,
                processor,
                log_file,
            )
        finally:
            log_file.close()
        _raise_if_episode_errors(output / "logs")
        return results
    finally:
        os.chdir(prior_cwd)


def _run_cases(
    evaluator: Any,
    cfg: Any,
    cases: Sequence[Mapping[str, Any]],
    task_ids: Mapping[str, int],
    model: Any,
    action_head: Any,
    proprio: Any,
    noisy: Any,
    processor: Any,
    log_file: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total_episodes = 0
    total_successes = 0
    resize_size = evaluator.get_image_resize_size(cfg)
    for task_name, task_cases in _cases_by_task(cases).items():
        if task_name not in task_ids:
            raise SylvestComparisonError(
                f"selected task is absent from runtime suite: {task_name}"
            )
        task_results, total_episodes, total_successes = _run_task_cases(
            evaluator,
            cfg,
            task_cases,
            task_ids[task_name],
            model,
            resize_size,
            processor,
            action_head,
            proprio,
            noisy,
            total_episodes,
            total_successes,
            log_file,
        )
        results.extend(task_results)
    return results


def _cases_by_task(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case["task_name"]), []).append(case)
    for task_cases in grouped.values():
        task_cases.sort(key=lambda case: int(case["initial_state_index"]))
        indices = [int(case["initial_state_index"]) for case in task_cases]
        if indices != list(range(len(indices))):
            raise SylvestComparisonError(
                "each task must use contiguous upstream initial-state indices"
            )
    return dict(sorted(grouped.items()))


def _run_task_cases(
    evaluator: Any,
    cfg: Any,
    cases: Sequence[Mapping[str, Any]],
    task_id: int,
    model: Any,
    resize_size: Any,
    processor: Any,
    action_head: Any,
    proprio: Any,
    noisy: Any,
    total_episodes: int,
    total_successes: int,
    log_file: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    cfg.num_trials_per_task = len(cases)
    evaluator.set_seed_everywhere(cfg.seed)
    outcomes, total_episodes, total_successes = _capture_upstream_task(
        evaluator,
        cfg,
        task_id,
        model,
        resize_size,
        processor,
        action_head,
        proprio,
        noisy,
        total_episodes,
        total_successes,
        log_file,
    )
    if len(outcomes) != len(cases):
        raise SylvestComparisonError(
            "upstream evaluator did not complete every requested initial-state trial"
        )
    results = [
        {**dict(case), "success": int(success)}
        for case, success in zip(cases, outcomes, strict=True)
    ]
    return results, total_episodes, total_successes


def _capture_upstream_task(
    evaluator: Any,
    cfg: Any,
    task_id: int,
    model: Any,
    resize_size: Any,
    processor: Any,
    action_head: Any,
    proprio: Any,
    noisy: Any,
    total_episodes: int,
    total_successes: int,
    log_file: Any,
) -> tuple[list[bool], int, int]:
    outcomes: list[bool] = []
    original_run_episode = evaluator.run_episode

    def record_episode(*args: Any, **kwargs: Any) -> tuple[bool, Any]:
        success, replay_images = original_run_episode(*args, **kwargs)
        outcomes.append(bool(success))
        return success, replay_images

    evaluator.run_episode = record_episode
    try:
        totals = evaluator.run_task(
            cfg,
            evaluator.benchmark.get_benchmark_dict()[cfg.task_suite_name](),
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio,
            noisy,
            total_episodes,
            total_successes,
            log_file,
        )
    finally:
        evaluator.run_episode = original_run_episode
    if totals[0] != total_episodes + len(outcomes) or totals[
        1
    ] != total_successes + sum(outcomes):
        raise SylvestComparisonError(
            "upstream evaluator returned inconsistent task totals"
        )
    return outcomes, *totals


def _rollout_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(int(row["success"]) for row in results)
    return {
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
    }


def _raise_if_episode_errors(logs: Path) -> None:
    errors = []
    for path in sorted(logs.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Episode error:" in line:
                errors.append(f"{path.name}: {line}")
    if errors:
        raise SylvestComparisonError(
            "upstream evaluator recorded episode errors; refusing to score them as failures: "
            + " | ".join(errors)
        )


def _clip_manifest(clips: Path) -> dict[str, Any]:
    files = (
        sorted(path for path in clips.rglob("*.mp4") if path.is_file())
        if clips.exists()
        else []
    )
    return {
        "schema": WORKFLOW_SCHEMA,
        "clips": [
            {
                "path": str(path.relative_to(clips)),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }


def _collect_rollout_clips(
    runtime_workspace: Path, output: Path, expected_count: int
) -> None:
    """Move upstream-generated rollout MP4s into the declared artifact directory.

    Args:
        runtime_workspace: Temporary upstream evaluator work directory.
        output: Stage-local output root for durable workflow artifacts.
        expected_count: Number of completed upstream simulator episodes.

    Returns:
        None.

    Raises:
        OSError: A generated video cannot be moved into the output contract.
    """

    upstream = runtime_workspace / "rollouts"
    clips = output / "clips"
    if not upstream.is_dir():
        raise SylvestComparisonError(
            "upstream evaluator did not produce a rollout-video directory"
        )
    clips.mkdir(parents=True, exist_ok=True)
    for source in sorted(upstream.rglob("*.mp4")):
        target = clips / source.name
        if target.exists():
            raise SylvestComparisonError(
                f"duplicate upstream rollout filename: {source.name}"
            )
        shutil.move(str(source), target)
    videos = sorted(clips.glob("*.mp4"))
    if len(videos) != expected_count:
        raise SylvestComparisonError(
            f"expected {expected_count} upstream rollout MP4s but found {len(videos)}"
        )
    for video in videos:
        _validate_rollout_video(video)


def _validate_rollout_video(video: Path) -> None:
    try:
        import imageio.v3 as imageio

        first_frame = imageio.imread(video, index=0)
    except Exception as exc:
        raise SylvestComparisonError(
            f"upstream rollout MP4 cannot be decoded: {video.name}: {exc}"
        ) from exc
    if video.stat().st_size <= 0 or getattr(first_frame, "size", 0) <= 0:
        raise SylvestComparisonError(f"upstream rollout MP4 is empty: {video.name}")


def _rollout_provenance(
    args: argparse.Namespace, payload: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": WORKFLOW_SCHEMA,
        "producer": "upstream OpenVLA-OFT experiments.robot.libero.run_libero_eval.run_task",
        "openvla_oft": {
            "repository": OFT_SOURCE_REPOSITORY,
            "revision": args.openvla_oft_revision,
        },
        "benchmark_source": {
            **_comparison_source(args.benchmark, args.libero_revision),
            "benchmark": args.benchmark,
        },
        "dlimp": _dlimp_derivative_provenance(Path(args.dlimp_runtime_root)),
        "checkpoint": payload["checkpoint"],
        "protocol_sha256": payload["protocol_sha256"],
        "limitations": [
            "Results are simulator rollouts, not physical-robot success.",
            "Every scored episode has one decoded upstream save_rollout_video MP4.",
        ],
    }


def _dlimp_derivative_provenance(derivative: Path) -> Mapping[str, Any]:
    if not _dlimp_derivative_is_ready(derivative):
        raise SylvestComparisonError(
            "dlimp deterministic runtime provenance is invalid"
        )
    payload = _read_json(derivative / _DLIMP_READY_FILE)
    if not isinstance(payload, dict):
        raise SylvestComparisonError(
            "dlimp deterministic runtime provenance is invalid"
        )
    return payload


def _compare(args: argparse.Namespace, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sylvest-compare-") as temporary:
        root = Path(temporary)
        baseline = _read_json(
            _materialize(args.baseline_uri, root / "baseline" / "rollouts.json")
        )
        candidate = _read_json(
            _materialize(args.candidate_uri, root / "candidate" / "rollouts.json")
        )
    comparison = _paired_comparison(baseline, candidate)
    _write_json(output / "comparison.json", comparison)
    _write_json(
        output / "provenance.json",
        {
            "schema": WORKFLOW_SCHEMA,
            "producer": "paired exact success comparison",
            "protocol_sha256": comparison["protocol_sha256"],
        },
    )


def _paired_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_rollouts(baseline)
    _validate_rollouts(candidate)
    if baseline["protocol_sha256"] != candidate["protocol_sha256"]:
        raise SylvestComparisonError(
            "baseline and candidate used different comparison protocols"
        )
    comparison_scope = baseline.get("comparison_scope", "unspecified")
    if comparison_scope != candidate.get("comparison_scope", "unspecified"):
        raise SylvestComparisonError(
            "baseline and candidate used different comparison population scopes"
        )
    if baseline.get("benchmark") != candidate.get("benchmark"):
        raise SylvestComparisonError("baseline and candidate used different benchmarks")
    if baseline.get("source") != candidate.get("source") or baseline.get(
        "runtime_sources"
    ) != candidate.get("runtime_sources"):
        raise SylvestComparisonError(
            "baseline and candidate used different benchmark or evaluator sources"
        )
    if baseline.get("evaluation_config") != candidate.get("evaluation_config"):
        raise SylvestComparisonError(
            "baseline and candidate used different evaluator configurations"
        )
    baseline_rows = {str(row["case_id"]): row for row in baseline["episodes"]}
    candidate_rows = {str(row["case_id"]): row for row in candidate["episodes"]}
    if baseline_rows.keys() != candidate_rows.keys():
        raise SylvestComparisonError(
            "baseline and candidate completed different paired cases"
        )
    rows = [
        _paired_row(baseline_rows[key], candidate_rows[key])
        for key in sorted(baseline_rows)
    ]
    deltas = [int(row["delta_success"]) for row in rows]
    wins = sum(delta == 1 for delta in deltas)
    losses = sum(delta == -1 for delta in deltas)
    return {
        "schema": WORKFLOW_SCHEMA,
        "kind": "paired_robustness_difference",
        "protocol_sha256": baseline["protocol_sha256"],
        "comparison_scope": comparison_scope,
        "benchmark": baseline["benchmark"],
        "source": baseline["source"],
        "runtime_sources": baseline["runtime_sources"],
        "evaluation_config": baseline["evaluation_config"],
        "baseline_checkpoint": baseline["checkpoint"],
        "candidate_checkpoint": candidate["checkpoint"],
        "paired_cases": rows,
        "overall": _delta_summary(deltas, wins, losses),
        "by_category": _category_summaries(rows),
        "limitations": [
            "Improvement is supported only when the paired interval and discordant-pair test are reported.",
            "This result does not establish convergence, a full benchmark claim, or physical-robot success.",
            *(
                [
                    "Training coverage is unknown; this comparison may include in-distribution tasks and is not a held-out generalization claim."
                ]
                if comparison_scope == "training_coverage_unknown"
                else []
            ),
        ],
    }


def _validate_rollouts(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKFLOW_SCHEMA or not isinstance(
        payload.get("episodes"), list
    ):
        raise SylvestComparisonError("invalid rollout artifact")
    if (
        not payload.get("protocol_sha256")
        or not payload["episodes"]
        or payload.get("benchmark") not in {"original_libero", "libero_plus"}
        or not isinstance(payload.get("source"), dict)
        or not isinstance(payload.get("runtime_sources"), dict)
        or not isinstance(payload.get("evaluation_config"), dict)
    ):
        raise SylvestComparisonError("rollout artifact lacks paired evidence")
    if payload.get("benchmark") == "original_libero" and any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("initial_state_source_sha256"), str)
        or not row["initial_state_source_sha256"]
        for row in payload["episodes"]
    ):
        raise SylvestComparisonError(
            "original LIBERO rollout lacks initial-state byte provenance"
        )


def _paired_row(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    for key in (
        "task_name",
        "category",
        "difficulty_level",
        "initial_state_index",
        "initial_state_source_sha256",
    ):
        if baseline.get(key) != candidate.get(key):
            raise SylvestComparisonError(
                f"paired case disagrees on {key}: {baseline['case_id']}"
            )
    return {
        "case_id": baseline["case_id"],
        "task_name": baseline["task_name"],
        "category": baseline["category"],
        "difficulty_level": baseline["difficulty_level"],
        "initial_state_index": baseline["initial_state_index"],
        "initial_state_source_sha256": baseline.get("initial_state_source_sha256"),
        "baseline_success": int(baseline["success"]),
        "candidate_success": int(candidate["success"]),
        "delta_success": int(candidate["success"]) - int(baseline["success"]),
    }


def _delta_summary(deltas: Sequence[int], wins: int, losses: int) -> dict[str, Any]:
    count = len(deltas)
    mean = sum(deltas) / count
    variance = (
        sum((delta - mean) ** 2 for delta in deltas) / (count - 1) if count > 1 else 0.0
    )
    margin = 1.96 * math.sqrt(variance / count) if count > 1 else 0.0
    return {
        "n": count,
        "mean_delta_success": mean,
        "normal_95_ci": [mean - margin, mean + margin],
        "candidate_only_successes": wins,
        "baseline_only_successes": losses,
        "mcnemar_exact_pvalue": _mcnemar_pvalue(wins, losses),
    }


def _mcnemar_pvalue(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, value) for value in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _category_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(int(row["delta_success"]))
    return {
        category: _delta_summary(values, values.count(1), values.count(-1))
        for category, values in sorted(grouped.items())
    }


def _report(args: argparse.Namespace, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sylvest-report-") as temporary:
        root = Path(temporary)
        comparison = _read_json(
            _materialize(args.comparison_uri, root / "comparison.json")
        )
        protocol = _read_json(_materialize(args.protocol_uri, root / "protocol.json"))
        notices = _read_json(_materialize(args.notices_uri, root / "notices.json"))
    _validate_protocol(protocol)
    _validate_comparison(comparison)
    _validate_notices(notices, protocol, comparison)
    report = _report_payload(args.run_id, comparison)
    _write_json(output / "report.json", report)
    _write_json(output / "notices.json", notices)
    rrd = output / "comparison.rrd"
    _write_rrd(rrd, args.run_id, comparison)
    checksums = {
        "report.json": _sha256_file(output / "report.json"),
        "comparison.rrd": _sha256_file(rrd),
        "notices.json": _sha256_file(output / "notices.json"),
    }
    _write_json(
        output / "checksums.json", {"schema": WORKFLOW_SCHEMA, "files": checksums}
    )


def _validate_comparison(comparison: Mapping[str, Any]) -> None:
    if comparison.get("schema") != WORKFLOW_SCHEMA or not comparison.get(
        "paired_cases"
    ):
        raise SylvestComparisonError("invalid paired comparison artifact")


def _validate_notices(
    notices: Any, protocol: Mapping[str, Any], comparison: Mapping[str, Any]
) -> None:
    if (
        not isinstance(notices, dict)
        or notices.get("schema") != WORKFLOW_SCHEMA
        or notices.get("kind") != "third_party_notices"
        or notices.get("protocol_sha256") != protocol.get("protocol_sha256")
        or notices.get("protocol_sha256") != comparison.get("protocol_sha256")
        or not isinstance(notices.get("components"), list)
        or len(notices["components"]) < 5
    ):
        raise SylvestComparisonError("invalid prepared third-party notices artifact")


def _report_payload(run_id: str, comparison: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": WORKFLOW_SCHEMA,
        "kind": "verified_sylvest_oft_mixdata_comparison_report",
        "run_id": run_id,
        "protocol_sha256": comparison["protocol_sha256"],
        "comparison_scope": comparison.get("comparison_scope", "unspecified"),
        "benchmark": comparison["benchmark"],
        "source": comparison["source"],
        "runtime_sources": comparison["runtime_sources"],
        "evaluation_config": comparison["evaluation_config"],
        "baseline_checkpoint": comparison["baseline_checkpoint"],
        "candidate_checkpoint": comparison["candidate_checkpoint"],
        "overall": comparison["overall"],
        "by_category": comparison["by_category"],
        "rollout_video_source": "upstream OpenVLA-OFT save_rollout_video outputs from both rollout stages",
        "limitations": comparison["limitations"],
    }


def _write_rrd(path: Path, run_id: str, comparison: Mapping[str, Any]) -> None:
    try:
        import rerun as rr
    except ImportError as exc:
        raise SylvestComparisonError(
            "rerun-sdk is required to emit comparison.rrd"
        ) from exc
    recording = rr.RecordingStream("npa_sylvest_oft_mixdata", recording_id=run_id)
    rr.save(path, recording=recording)
    rr.log(
        "provenance/protocol_sha256",
        rr.TextLog(str(comparison["protocol_sha256"])),
        static=True,
        recording=recording,
    )
    for index, row in enumerate(comparison["paired_cases"]):
        rr.set_time("paired_case", sequence=index, recording=recording)
        rr.log(
            "metrics/baseline_success",
            rr.Scalars(float(row["baseline_success"])),
            recording=recording,
        )
        rr.log(
            "metrics/candidate_success",
            rr.Scalars(float(row["candidate_success"])),
            recording=recording,
        )
        rr.log(
            "metrics/delta_success",
            rr.Scalars(float(row["delta_success"])),
            recording=recording,
        )
    recording.flush()
    recording.disconnect()
    if not path.is_file() or path.stat().st_size <= 0:
        raise SylvestComparisonError(
            "Rerun did not write a non-empty comparison recording"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for each stateless workflow stage.

    Args:
        None.

    Returns:
        Configured command-line parser.

    Raises:
        None.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--benchmark", choices=("original_libero", "libero_plus"), required=True
    )
    prepare.add_argument(
        "--libero-root", "--libero-plus-root", dest="libero_root", required=True
    )
    prepare.add_argument(
        "--libero-revision",
        "--libero-plus-revision",
        dest="libero_revision",
        required=True,
    )
    prepare.add_argument("--training-task-ids-uri", default="")
    prepare.add_argument(
        "--comparison-scope",
        default="training_coverage_unknown",
        choices=("held_out", "training_coverage_unknown"),
    )
    prepare.add_argument("--task-suite", default="libero_spatial")
    prepare.add_argument("--initial-state-indices", default="0,1,2")
    prepare.add_argument("--seed", type=int, default=7)
    prepare.add_argument("--env-image-resolution", type=int, default=256)
    prepare.add_argument("--center-crop", type=_parse_bool, default=True)
    prepare.add_argument("--num-images-in-input", type=int, default=2)
    prepare.add_argument("--num-open-loop-steps", type=int, default=8)
    prepare.add_argument("--model-cache-root", required=True)
    prepare.add_argument("--openvla-oft-root", required=True)
    prepare.add_argument("--openvla-oft-revision", required=True)
    prepare.add_argument("--dlimp-root", required=True)
    prepare.add_argument("--dlimp-revision", required=True)
    prepare.add_argument("--baseline-checkpoint-id", required=True)
    prepare.add_argument("--baseline-checkpoint-revision", required=True)
    prepare.add_argument("--candidate-checkpoint-id", required=True)
    prepare.add_argument("--candidate-checkpoint-revision", required=True)
    rollout = commands.add_parser("rollout")
    rollout.add_argument("--protocol-uri", required=True)
    rollout.add_argument("--checkpoint-id", required=True)
    rollout.add_argument("--checkpoint-revision", required=True)
    rollout.add_argument("--model-cache-root", required=True)
    rollout.add_argument("--openvla-oft-root", required=True)
    rollout.add_argument("--openvla-oft-revision", required=True)
    rollout.add_argument("--dlimp-root", required=True)
    rollout.add_argument("--dlimp-revision", required=True)
    rollout.add_argument("--dlimp-runtime-root", required=True)
    rollout.add_argument(
        "--benchmark", choices=("original_libero", "libero_plus"), required=True
    )
    rollout.add_argument(
        "--libero-root", "--libero-plus-root", dest="libero_root", required=True
    )
    rollout.add_argument(
        "--libero-revision",
        "--libero-plus-revision",
        dest="libero_revision",
        required=True,
    )
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline-uri", required=True)
    compare.add_argument("--candidate-uri", required=True)
    report = commands.add_parser("report")
    report.add_argument("--comparison-uri", required=True)
    report.add_argument("--protocol-uri", required=True)
    report.add_argument("--notices-uri", required=True)
    report.add_argument("--run-id", required=True)
    for command in (prepare, rollout, compare, report):
        command.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one source-pinned comparison stage and publish its artifacts.

    Args:
        argv: Explicit CLI arguments or process arguments when omitted.

    Returns:
        Zero after the declared stage output is published.

    Raises:
        SylvestComparisonError: A required source, input, or evidence invariant fails.
        OSError: Local artifact I/O fails.
    """

    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix=f"sylvest-{args.stage}-") as temporary:
        output = Path(temporary) / "output"
        if args.stage == "prepare":
            _prepare_protocol(args, output)
        elif args.stage == "rollout":
            _rollout(args, output)
        elif args.stage == "compare":
            _compare(args, output)
        else:
            _report(args, output)
        _publish_tree(output, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
