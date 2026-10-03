"""Run-scoped contracts for MolmoAct2 TensorRT evidence on Jetson AGX Thor.

The cloud workflow prepares observations, validates target-produced evidence,
evaluates recorded action traces, and emits an Rerun recording.  The two target
commands deliberately run on a Jetson device: TensorRT plans are not portable
to a Nebius x86 worker, and this module refuses to represent x86 execution as
Thor validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from npa.clients.storage import StorageClient

APPLICATION_ID = "npa.molmoact2.jetson-thor"
OBSERVATIONS_SCHEMA = "npa.molmoact2.jetson_thor.observations.v1"
PREPARED_SCHEMA = "npa.molmoact2.jetson_thor.prepared_observations.v1"
QUALIFICATION_SCHEMA = "npa.molmoact2.jetson_thor.target_qualification.v1"
VERIFIED_QUALIFICATION_SCHEMA = "npa.molmoact2.jetson_thor.verified_qualification.v1"
TRACE_SCHEMA = "npa.molmoact2.jetson_thor.action_trace.v1"
VERIFIED_TRACE_SCHEMA = "npa.molmoact2.jetson_thor.verified_action_trace.v1"
EVALUATION_SCHEMA = "npa.molmoact2.jetson_thor.action_trace_evaluation.v1"
VISUALIZATION_SCHEMA = "npa.molmoact2.jetson_thor.visualization.v1"

VLA_EDGE_REPOSITORY = "https://github.com/Agents2AgentsAI/vla-edge"
VLA_EDGE_REVISION = "747fd96aaf1a386d3873cf6c637e7a51141b66fd"
BUNDLE_REPOSITORY = "https://huggingface.co/agents2agents/MolmoAct2-LIBERO-Jetson-Thor"
BUNDLE_REVISION = "7d2de215036b802c247e3c3c4db7316c99d6adfe"
BASE_MODEL = "allenai/MolmoAct2-LIBERO"
BASE_MODEL_REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
TARGET_REQUIREMENTS = {
    "arch": "aarch64",
    "compute_capability": "sm_110a",
    "cuda_device_name": "NVIDIA Thor",
    "multiprocessor_count": 20,
    "board_model": "NVIDIA Jetson AGX Thor Developer Kit",
    "tensorrt": "10.16.2.10",
    "jetpack": "R39 rev 2.1",
}


class MolmoAct2JetsonThorError(RuntimeError):
    """Raised when a MolmoAct2 Jetson-Thor artifact contract is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_run_id(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or any(not (part.isalnum() or part in "-_") for part in normalized):
        raise MolmoAct2JetsonThorError("run_id must contain only letters, digits, '-' or '_'")
    return normalized


def _require_digest(value: Any, name: str) -> str:
    normalized = str(value or "").strip().lower().removeprefix("sha256:")
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise MolmoAct2JetsonThorError(f"{name} must be a sha256 digest")
    return normalized


def _materialize(uri: str, destination: Path) -> Path:
    if uri.startswith("s3://"):
        StorageClient.from_environment().download_file(uri, str(destination))
    else:
        source = Path(uri)
        if not source.is_file():
            raise MolmoAct2JetsonThorError(f"input is not a regular file: {uri}")
        shutil.copyfile(source, destination)
    return destination


def _publish(path: Path, uri: str) -> str:
    if uri.startswith("s3://"):
        return str(StorageClient.from_environment().upload_file(str(path), uri))
    destination = Path(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    return str(destination)


def _read_json(uri: str, destination: Path) -> dict[str, Any]:
    try:
        value = json.loads(_materialize(uri, destination).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MolmoAct2JetsonThorError(f"invalid JSON at {uri}") from exc
    if not isinstance(value, dict):
        raise MolmoAct2JetsonThorError(f"JSON object required at {uri}")
    return value


def _write_json(root: Path, name: str, value: Mapping[str, Any], uri: str) -> str:
    path = root / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _publish(path, uri)


def _read_npz(uri: str, destination: Path) -> tuple[dict[str, np.ndarray], str]:
    path = _materialize(uri, destination)
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise MolmoAct2JetsonThorError(f"invalid NPZ observations at {uri}") from exc
    return values, _sha256_bytes(path.read_bytes())


def _write_npz(root: Path, name: str, values: Mapping[str, np.ndarray], uri: str) -> tuple[str, str]:
    path = root / name
    np.savez_compressed(path, **values)
    return _publish(path, uri), _sha256_bytes(path.read_bytes())


def _require_observation_arrays(values: Mapping[str, np.ndarray]) -> int:
    required = {"image", "wrist_image", "state", "instruction", "episode_id", "split"}
    missing = sorted(required - set(values))
    if missing:
        raise MolmoAct2JetsonThorError(f"observations missing arrays: {missing}")
    count = int(values["state"].shape[0]) if values["state"].ndim else 0
    if count < 2:
        raise MolmoAct2JetsonThorError("observations need at least two rows")
    for camera in ("image", "wrist_image"):
        image = values[camera]
        if image.shape[0] != count or image.ndim != 4 or image.shape[-1] != 3:
            raise MolmoAct2JetsonThorError(f"{camera} must be [N,H,W,3]")
        if image.dtype != np.uint8:
            raise MolmoAct2JetsonThorError(f"{camera} must contain uint8 RGB pixels")
    state = values["state"]
    if state.shape != (count, 8) or not np.isfinite(state).all():
        raise MolmoAct2JetsonThorError("state must be finite [N,8]")
    for name in ("instruction", "episode_id", "split"):
        if values[name].shape != (count,):
            raise MolmoAct2JetsonThorError(f"{name} must be a length-N vector")
    episodes = [str(value) for value in values["episode_id"]]
    if len(set(episodes)) != count or any(not value for value in episodes):
        raise MolmoAct2JetsonThorError("episode_id values must be nonempty and unique")
    splits = {str(value) for value in values["split"]}
    if splits != {"calibration", "evaluation"}:
        raise MolmoAct2JetsonThorError("split must contain exactly calibration and evaluation rows")
    if any(not str(value).strip() for value in values["instruction"]):
        raise MolmoAct2JetsonThorError("instruction values must be nonempty")
    return count


def prepare_observations(
    *, observations_uri: str, prepared_uri: str, manifest_uri: str, run_id: str
) -> dict[str, Any]:
    """Validate and canonicalize actual LIBERO observations for edge execution.

    Args:
        observations_uri: Source NPZ object containing camera pixels and states.
        prepared_uri: Destination NPZ object for canonicalized observations.
        manifest_uri: Destination JSON provenance manifest.
        run_id: Owning workflow run identifier.

    Returns:
        Published artifact URIs and row counts.

    Raises:
        MolmoAct2JetsonThorError: The observation contract is malformed.
    """
    run_id = _safe_run_id(run_id)
    with tempfile.TemporaryDirectory(prefix="npa-molmoact2-thor-prepare-") as temp:
        root = Path(temp)
        values, source_digest = _read_npz(observations_uri, root / "source.npz")
        count = _require_observation_arrays(values)
        if "seed" in values:
            seed = values["seed"]
            if seed.shape != (count,) or not np.issubdtype(seed.dtype, np.integer):
                raise MolmoAct2JetsonThorError("seed must be a length-N integer vector")
        else:
            values = {**values, "seed": np.arange(count, dtype=np.int64)}
        prepared, prepared_digest = _write_npz(root, "prepared.npz", values, prepared_uri)
        splits = np.asarray(values["split"]).astype(str)
        manifest = {
            "schema": PREPARED_SCHEMA,
            "run_id": run_id,
            "source_observations_uri": observations_uri,
            "source_observations_sha256": source_digest,
            "prepared_observations_uri": prepared,
            "prepared_observations_sha256": prepared_digest,
            "rows": count,
            "calibration_rows": int(np.count_nonzero(splits == "calibration")),
            "evaluation_rows": int(np.count_nonzero(splits == "evaluation")),
            "camera_contract": ["image", "wrist_image"],
            "state_dim": 8,
            "source_schema": OBSERVATIONS_SCHEMA,
        }
        published_manifest = _write_json(root, "prepared-manifest.json", manifest, manifest_uri)
    return {"prepared_uri": prepared, "manifest_uri": published_manifest, **manifest}


def _require_target_fields(report: Mapping[str, Any]) -> None:
    for name, expected in TARGET_REQUIREMENTS.items():
        actual = report.get("target", {}).get(name) if isinstance(report.get("target"), dict) else None
        if actual != expected:
            raise MolmoAct2JetsonThorError(f"target {name}={actual!r}, expected {expected!r}")


def _canonical_compute_capability(value: Any) -> str:
    """Keep upstream's ``sm_110``/``sm_110a`` compatibility equivalence explicit."""

    detected = str(value or "")
    if detected.rstrip("a") == TARGET_REQUIREMENTS["compute_capability"].rstrip("a"):
        return TARGET_REQUIREMENTS["compute_capability"]
    return detected


def _detect_jetpack_release() -> str:
    """Return only the documented JetPack contract marker from the target OS."""

    try:
        release = Path("/etc/nv_tegra_release").read_text(encoding="utf-8")
    except OSError as exc:
        raise MolmoAct2JetsonThorError("JetPack release marker is unavailable on this target") from exc
    if "R39" in release and "REVISION: 2.1" in release:
        return TARGET_REQUIREMENTS["jetpack"]
    return "unrecognized"


def target_qualify(
    *, engine_dir: str, prepared_uri: str, qualification_uri: str, run_id: str
) -> dict[str, Any]:
    """Run upstream TensorRT compatibility and checksum gates on a Thor target.

    Args:
        engine_dir: Locally mounted downloaded engine-bundle root on the Thor.
        prepared_uri: Canonical observations produced by ``prepare_observations``.
        qualification_uri: Destination for the target-produced JSON evidence.
        run_id: Owning workflow run identifier.

    Returns:
        The published target qualification report.

    Raises:
        MolmoAct2JetsonThorError: The device, bundle, or observations do not match.
    """
    run_id = _safe_run_id(run_id)
    try:
        from vla_edge.backends.tensorrt import artifacts
    except ImportError as exc:
        raise MolmoAct2JetsonThorError(
            "target-qualify requires the pinned vla-edge runtime on the Jetson target"
        ) from exc
    with tempfile.TemporaryDirectory(prefix="npa-molmoact2-thor-qualify-") as temp:
        root = Path(temp)
        _values, prepared_digest = _read_npz(prepared_uri, root / "prepared.npz")
        _require_observation_arrays(_values)
        try:
            artifacts.check_compatible(engine_dir)
            artifacts.verify_checksums(engine_dir)
            engine_manifest = artifacts.load_manifest(engine_dir)
            environment = artifacts.Environment.detect()
        except Exception as exc:  # upstream defines the correct compatibility boundary
            raise MolmoAct2JetsonThorError(f"upstream TensorRT qualification failed: {exc}") from exc
        requires = engine_manifest.get("requires")
        if not isinstance(requires, dict):
            raise MolmoAct2JetsonThorError("engine MANIFEST.json omits requirements")
        for name, expected in TARGET_REQUIREMENTS.items():
            if name == "jetpack":
                if expected not in str(requires.get(name, "")):
                    raise MolmoAct2JetsonThorError("engine manifest JetPack revision differs")
            elif requires.get(name) != expected:
                raise MolmoAct2JetsonThorError(f"engine manifest {name} differs from target contract")
        report = {
            "schema": QUALIFICATION_SCHEMA,
            "status": "passed",
            "run_id": run_id,
            "prepared_observations_sha256": prepared_digest,
            "bundle_repository": BUNDLE_REPOSITORY,
            "bundle_revision": BUNDLE_REVISION,
            "vla_edge_repository": VLA_EDGE_REPOSITORY,
            "vla_edge_revision": VLA_EDGE_REVISION,
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_MODEL_REVISION,
            "engine_manifest_sha256": _sha256_bytes(
                (Path(engine_dir) / "MANIFEST.json").read_bytes()
            ),
            "target": {
                "arch": environment.arch,
                "compute_capability": _canonical_compute_capability(environment.compute_capability),
                "cuda_device_name": environment.cuda_device_name,
                "multiprocessor_count": environment.multiprocessor_count,
                "board_model": environment.board_model,
                "tensorrt": environment.tensorrt,
                "jetpack": _detect_jetpack_release(),
            },
        }
        _require_target_fields(report)
        published = _write_json(root, "target-qualification.json", report, qualification_uri)
    return {"qualification_uri": published, **report}


def verify_target_qualification(
    *, prepared_uri: str, qualification_uri: str, verified_uri: str, run_id: str
) -> dict[str, Any]:
    """Bind a target-produced qualification report to this run's prepared data.

    Args:
        prepared_uri: Canonical observation artifact.
        qualification_uri: Report emitted by ``target_qualify`` on the Thor.
        verified_uri: Destination for the bound verification receipt.
        run_id: Owning workflow run identifier.

    Returns:
        The verified receipt containing immutable source and engine identities.

    Raises:
        MolmoAct2JetsonThorError: The report is incomplete or mismatched.
    """
    run_id = _safe_run_id(run_id)
    with tempfile.TemporaryDirectory(prefix="npa-molmoact2-thor-verify-engine-") as temp:
        root = Path(temp)
        _values, prepared_digest = _read_npz(prepared_uri, root / "prepared.npz")
        _require_observation_arrays(_values)
        report = _read_json(qualification_uri, root / "qualification.json")
        if report.get("schema") != QUALIFICATION_SCHEMA or report.get("status") != "passed":
            raise MolmoAct2JetsonThorError("target qualification did not pass")
        if report.get("run_id") != run_id or report.get("prepared_observations_sha256") != prepared_digest:
            raise MolmoAct2JetsonThorError("target qualification is not bound to this run's observations")
        expected = {
            "bundle_repository": BUNDLE_REPOSITORY,
            "bundle_revision": BUNDLE_REVISION,
            "vla_edge_repository": VLA_EDGE_REPOSITORY,
            "vla_edge_revision": VLA_EDGE_REVISION,
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_MODEL_REVISION,
        }
        if any(report.get(name) != value for name, value in expected.items()):
            raise MolmoAct2JetsonThorError("target qualification source provenance differs")
        _require_target_fields(report)
        receipt = {
            "schema": VERIFIED_QUALIFICATION_SCHEMA,
            "run_id": run_id,
            "prepared_observations_uri": prepared_uri,
            "prepared_observations_sha256": prepared_digest,
            "qualification_uri": qualification_uri,
            "qualification_sha256": _sha256_bytes(json.dumps(report, sort_keys=True).encode()),
            "engine_manifest_sha256": _require_digest(report.get("engine_manifest_sha256"), "engine_manifest_sha256"),
            "target": report["target"],
            **expected,
        }
        published = _write_json(root, "verified-qualification.json", receipt, verified_uri)
    return {"verified_uri": published, **receipt}


def _bind_edge_endpoint(endpoint: str, backend: str):
    try:
        from vla_edge.protocol.client import ActClient
    except ImportError as exc:
        raise MolmoAct2JetsonThorError(
            "target-action-trace requires vla-edge on the target device"
        ) from exc
    client = ActClient(endpoint)
    contract = client.bind(
        policy="molmoact2-libero",
        model_family="molmoact2",
        embodiment="libero",
        cameras=("image", "wrist_image"),
        state_dim=8,
    )
    health = client.health()
    if health.get("backend") != backend:
        client.close()
        raise MolmoAct2JetsonThorError(f"endpoint must report backend={backend!r}")
    return client, contract


def target_action_trace(
    *,
    prepared_uri: str,
    qualification_uri: str,
    target_endpoint: str,
    reference_endpoint: str,
    trace_uri: str,
    manifest_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Call target TensorRT and Torch reference servers on evaluation observations.

    Args:
        prepared_uri: Prepared observations containing actual RGB pixels.
        qualification_uri: Target qualification report or its verified receipt.
        target_endpoint: Thor TensorRT ``vla-edge`` `/act` server endpoint.
        reference_endpoint: Matching Torch ``vla-edge`` `/act` server endpoint.
        trace_uri: Destination NPZ containing both returned action tensors.
        manifest_uri: Destination JSON provenance manifest.
        run_id: Owning workflow run identifier.

    Returns:
        Published trace and manifest artifact identities.

    Raises:
        MolmoAct2JetsonThorError: An endpoint or returned action contract fails.
    """
    run_id = _safe_run_id(run_id)
    with tempfile.TemporaryDirectory(prefix="npa-molmoact2-thor-trace-") as temp:
        root = Path(temp)
        values, prepared_digest = _read_npz(prepared_uri, root / "prepared.npz")
        _require_observation_arrays(values)
        qualification = _read_json(qualification_uri, root / "qualification.json")
        qualification_schema = qualification.get("schema")
        if qualification_schema == QUALIFICATION_SCHEMA:
            if qualification.get("status") != "passed":
                raise MolmoAct2JetsonThorError("target qualification did not pass")
            qualification_digest = _sha256_bytes(json.dumps(qualification, sort_keys=True).encode())
        elif qualification_schema == VERIFIED_QUALIFICATION_SCHEMA:
            qualification_digest = _require_digest(
                qualification.get("qualification_sha256"), "qualification_sha256"
            )
        else:
            raise MolmoAct2JetsonThorError("target-action-trace requires a target qualification report")
        if qualification.get("run_id") != run_id or qualification.get("prepared_observations_sha256") != prepared_digest:
            raise MolmoAct2JetsonThorError("qualification is not bound to these observations")
        target, target_contract = _bind_edge_endpoint(target_endpoint, "tensorrt")
        reference, reference_contract = _bind_edge_endpoint(reference_endpoint, "torch")
        try:
            if (target_contract.action_dim, target_contract.action_horizon) != (
                reference_contract.action_dim,
                reference_contract.action_horizon,
            ):
                raise MolmoAct2JetsonThorError("target and reference action contracts differ")
            selected = np.asarray(values["split"]).astype(str) == "evaluation"
            target_actions, reference_actions, target_dt, reference_dt = [], [], [], []
            for index in np.flatnonzero(selected):
                cameras = {name: values[name][index] for name in ("image", "wrist_image")}
                state = values["state"][index]
                instruction = str(values["instruction"][index])
                seed = int(values["seed"][index])
                candidate, candidate_ms = target.act(cameras, instruction, state, seed=seed)
                baseline, baseline_ms = reference.act(cameras, instruction, state, seed=seed)
                target_actions.append(candidate)
                reference_actions.append(baseline)
                target_dt.append(candidate_ms)
                reference_dt.append(baseline_ms)
        finally:
            target.close()
            reference.close()
        if not target_actions:
            raise MolmoAct2JetsonThorError("prepared observations contain no evaluation rows")
        trace_values = {
            "episode_id": values["episode_id"][selected],
            "seed": values["seed"][selected],
            "target_actions": np.stack(target_actions),
            "reference_actions": np.stack(reference_actions),
            "target_dt_ms": np.asarray(target_dt, dtype=np.float64),
            "reference_dt_ms": np.asarray(reference_dt, dtype=np.float64),
        }
        published_trace, trace_digest = _write_npz(root, "target-action-trace.npz", trace_values, trace_uri)
        manifest = {
            "schema": TRACE_SCHEMA,
            "run_id": run_id,
            "execution_mode": "open_loop_observation_trace",
            "closed_loop_rollout_status": "not_evaluated_by_vla_edge",
            "prepared_observations_sha256": prepared_digest,
            "qualification_sha256": qualification_digest,
            "trace_uri": published_trace,
            "trace_sha256": trace_digest,
            "rows": int(trace_values["target_actions"].shape[0]),
            "action_horizon": target_contract.action_horizon,
            "action_dim": target_contract.action_dim,
            "target_backend": "tensorrt",
            "reference_backend": "torch",
        }
        published_manifest = _write_json(root, "target-action-trace.json", manifest, manifest_uri)
    return {"trace_uri": published_trace, "manifest_uri": published_manifest, **manifest}


def _validate_trace(values: Mapping[str, np.ndarray], rows: int, horizon: int, width: int) -> None:
    expected = {"episode_id", "seed", "target_actions", "reference_actions", "target_dt_ms", "reference_dt_ms"}
    if expected - set(values):
        raise MolmoAct2JetsonThorError("action trace omits required arrays")
    action_shape = (rows, horizon, width)
    for name in ("target_actions", "reference_actions"):
        if values[name].shape != action_shape or not np.isfinite(values[name]).all():
            raise MolmoAct2JetsonThorError(f"{name} must be finite {action_shape}")
    for name in ("target_dt_ms", "reference_dt_ms"):
        if values[name].shape != (rows,) or not np.isfinite(values[name]).all() or (values[name] <= 0).any():
            raise MolmoAct2JetsonThorError(f"{name} must contain positive finite milliseconds")


def verify_target_action_trace(
    *, trace_manifest_uri: str, verified_qualification_uri: str, verified_trace_uri: str, run_id: str
) -> dict[str, Any]:
    """Verify target-produced action tensors before numerical evaluation.

    Args:
        trace_manifest_uri: Manifest emitted by ``target_action_trace``.
        verified_qualification_uri: Bound engine qualification receipt.
        verified_trace_uri: Destination for the validated trace receipt.
        run_id: Owning workflow run identifier.

    Returns:
        A receipt that binds the trace digest to qualification evidence.

    Raises:
        MolmoAct2JetsonThorError: The trace is incomplete, stale, or non-finite.
    """
    run_id = _safe_run_id(run_id)
    with tempfile.TemporaryDirectory(prefix="npa-molmoact2-thor-verify-trace-") as temp:
        root = Path(temp)
        manifest = _read_json(trace_manifest_uri, root / "trace-manifest.json")
        qualification = _read_json(verified_qualification_uri, root / "qualification.json")
        if manifest.get("schema") != TRACE_SCHEMA or qualification.get("schema") != VERIFIED_QUALIFICATION_SCHEMA:
            raise MolmoAct2JetsonThorError("target trace or qualification schema differs")
        if manifest.get("run_id") != run_id or qualification.get("run_id") != run_id:
            raise MolmoAct2JetsonThorError("target trace is not bound to this run")
        if manifest.get("qualification_sha256") != qualification.get("qualification_sha256"):
            raise MolmoAct2JetsonThorError("target trace is not bound to engine qualification")
        rows, horizon, width = (int(manifest[name]) for name in ("rows", "action_horizon", "action_dim"))
        if rows < 1 or horizon < 1 or width < 1:
            raise MolmoAct2JetsonThorError("target trace dimensions must be positive")
        values, trace_digest = _read_npz(str(manifest.get("trace_uri", "")), root / "trace.npz")
        if trace_digest != _require_digest(manifest.get("trace_sha256"), "trace_sha256"):
            raise MolmoAct2JetsonThorError("target trace digest differs from its manifest")
        _validate_trace(values, rows, horizon, width)
        if manifest.get("target_backend") != "tensorrt" or manifest.get("reference_backend") != "torch":
            raise MolmoAct2JetsonThorError("trace must compare TensorRT target against Torch reference")
        receipt = {
            "schema": VERIFIED_TRACE_SCHEMA,
            "run_id": run_id,
            "trace_manifest_uri": trace_manifest_uri,
            "trace_uri": manifest["trace_uri"],
            "trace_sha256": trace_digest,
            "qualification_sha256": qualification["qualification_sha256"],
            "rows": rows,
            "action_horizon": horizon,
            "action_dim": width,
            "execution_mode": manifest["execution_mode"],
            "closed_loop_rollout_status": manifest["closed_loop_rollout_status"],
        }
        published = _write_json(root, "verified-action-trace.json", receipt, verified_trace_uri)
    return {"verified_trace_uri": published, **receipt}


def evaluate_action_trace(
    *, verified_trace_uri: str, evaluation_uri: str, run_id: str
) -> dict[str, Any]:
    """Calculate exact action-error and latency statistics from a verified trace.

    Args:
        verified_trace_uri: Receipt emitted by ``verify_target_action_trace``.
        evaluation_uri: Destination for the numerical evaluation JSON.
        run_id: Owning workflow run identifier.

    Returns:
        Published numerical metrics and declared evidence scope.

    Raises:
        MolmoAct2JetsonThorError: The input trace is not verified for this run.
    """
    run_id = _safe_run_id(run_id)
    with tempfile.TemporaryDirectory(prefix="npa-molmoact2-thor-evaluate-") as temp:
        root = Path(temp)
        receipt = _read_json(verified_trace_uri, root / "verified-trace.json")
        if receipt.get("schema") != VERIFIED_TRACE_SCHEMA or receipt.get("run_id") != run_id:
            raise MolmoAct2JetsonThorError("evaluation requires this run's verified trace")
        rows, horizon, width = (int(receipt[name]) for name in ("rows", "action_horizon", "action_dim"))
        values, trace_digest = _read_npz(str(receipt.get("trace_uri", "")), root / "trace.npz")
        if trace_digest != _require_digest(receipt.get("trace_sha256"), "trace_sha256"):
            raise MolmoAct2JetsonThorError("verified trace digest changed")
        _validate_trace(values, rows, horizon, width)
        error = np.asarray(values["target_actions"] - values["reference_actions"], dtype=np.float64)
        absolute = np.abs(error)
        target_latency = np.asarray(values["target_dt_ms"], dtype=np.float64)
        reference_latency = np.asarray(values["reference_dt_ms"], dtype=np.float64)
        result = {
            "schema": EVALUATION_SCHEMA,
            "run_id": run_id,
            "verified_trace_uri": verified_trace_uri,
            "trace_uri": receipt["trace_uri"],
            "trace_sha256": trace_digest,
            "rows": rows,
            "action_horizon": horizon,
            "action_dim": width,
            "action_mae": float(np.mean(absolute)),
            "action_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "action_max_abs_error": float(np.max(absolute)),
            "target_latency_ms": {"mean": float(np.mean(target_latency)), "p50": float(np.percentile(target_latency, 50)), "p95": float(np.percentile(target_latency, 95))},
            "reference_latency_ms": {"mean": float(np.mean(reference_latency)), "p50": float(np.percentile(reference_latency, 50)), "p95": float(np.percentile(reference_latency, 95))},
            "latency_ratio_target_over_reference": float(np.mean(target_latency) / np.mean(reference_latency)),
            "closed_loop_rollout_status": receipt["closed_loop_rollout_status"],
            "scope": "action-trace equivalence and endpoint latency; not LIBERO closed-loop task success",
        }
        published = _write_json(root, "action-trace-evaluation.json", result, evaluation_uri)
    return {"evaluation_uri": published, **result}


def _write_rrd(path: Path, evaluation: Mapping[str, Any], values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    import rerun as rr

    recording = rr.RecordingStream(APPLICATION_ID, recording_id=str(evaluation["run_id"]))
    rr.save(path, recording=recording)
    rr.log(
        "provenance",
        rr.TextDocument(
            "# MolmoAct2 Jetson-Thor action-trace evidence\n\n"
            f"- trace SHA-256: `{evaluation['trace_sha256']}`\n"
            f"- scope: {evaluation['scope']}\n"
            f"- closed-loop status: `{evaluation['closed_loop_rollout_status']}`",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
        recording=recording,
    )
    target = np.asarray(values["target_actions"], dtype=np.float64)
    reference = np.asarray(values["reference_actions"], dtype=np.float64)
    for index, (candidate, baseline, target_ms, reference_ms) in enumerate(
        zip(target, reference, values["target_dt_ms"], values["reference_dt_ms"], strict=True)
    ):
        rr.set_time("evaluation_row", sequence=index, recording=recording)
        rr.log("latency/target_ms", rr.Scalars(float(target_ms)), recording=recording)
        rr.log("latency/reference_ms", rr.Scalars(float(reference_ms)), recording=recording)
        rr.log("actions/mae", rr.Scalars(float(np.mean(np.abs(candidate - baseline)))), recording=recording)
    return {"timeline": "evaluation_row", "samples": int(target.shape[0])}


def visualize_action_trace(
    *, verified_trace_uri: str, evaluation_uri: str, rrd_uri: str, manifest_uri: str, run_id: str
) -> dict[str, Any]:
    """Create a factual Rerun recording from verified actions and evaluation.

    Args:
        verified_trace_uri: Verified trace receipt.
        evaluation_uri: Numerical evaluation JSON.
        rrd_uri: Destination Rerun recording URI.
        manifest_uri: Destination visualization provenance JSON.
        run_id: Owning workflow run identifier.

    Returns:
        Published RRD and its byte-bound manifest.

    Raises:
        MolmoAct2JetsonThorError: The supplied evaluation and trace disagree.
    """
    run_id = _safe_run_id(run_id)
    with tempfile.TemporaryDirectory(prefix="npa-molmoact2-thor-viz-") as temp:
        root = Path(temp)
        receipt = _read_json(verified_trace_uri, root / "verified-trace.json")
        evaluation = _read_json(evaluation_uri, root / "evaluation.json")
        if receipt.get("schema") != VERIFIED_TRACE_SCHEMA or evaluation.get("schema") != EVALUATION_SCHEMA:
            raise MolmoAct2JetsonThorError("visualization requires verified trace and evaluation schemas")
        if receipt.get("run_id") != run_id or evaluation.get("run_id") != run_id:
            raise MolmoAct2JetsonThorError("visualization artifacts belong to another run")
        if receipt.get("trace_sha256") != evaluation.get("trace_sha256"):
            raise MolmoAct2JetsonThorError("evaluation and trace receipt disagree")
        values, trace_digest = _read_npz(str(receipt.get("trace_uri", "")), root / "trace.npz")
        _validate_trace(values, int(receipt["rows"]), int(receipt["action_horizon"]), int(receipt["action_dim"]))
        if trace_digest != receipt["trace_sha256"]:
            raise MolmoAct2JetsonThorError("visualization trace digest changed")
        rrd_path = root / "molmoact2-jetson-thor.rrd"
        timeline = _write_rrd(rrd_path, evaluation, values)
        published_rrd = _publish(rrd_path, rrd_uri)
        manifest = {
            "schema": VISUALIZATION_SCHEMA,
            "run_id": run_id,
            "rrd_uri": published_rrd,
            "rrd_sha256": _sha256_bytes(rrd_path.read_bytes()),
            "rrd_bytes": rrd_path.stat().st_size,
            "trace_sha256": trace_digest,
            "timeline": timeline,
            "closed_loop_rollout_status": evaluation["closed_loop_rollout_status"],
        }
        published_manifest = _write_json(root, "visualization-manifest.json", manifest, manifest_uri)
    return {"rrd_uri": published_rrd, "manifest_uri": published_manifest, **manifest}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--observations-uri", required=True)
    prepare.add_argument("--prepared-uri", required=True)
    prepare.add_argument("--manifest-uri", required=True)
    prepare.add_argument("--run-id", required=True)
    qualify = commands.add_parser("target-qualify")
    qualify.add_argument("--engine-dir", required=True)
    qualify.add_argument("--prepared-uri", required=True)
    qualify.add_argument("--qualification-uri", required=True)
    qualify.add_argument("--run-id", required=True)
    verify_engine = commands.add_parser("verify-qualification")
    verify_engine.add_argument("--prepared-uri", required=True)
    verify_engine.add_argument("--qualification-uri", required=True)
    verify_engine.add_argument("--verified-uri", required=True)
    verify_engine.add_argument("--run-id", required=True)
    trace = commands.add_parser("target-action-trace")
    for name in ("prepared-uri", "qualification-uri", "target-endpoint", "reference-endpoint", "trace-uri", "manifest-uri", "run-id"):
        trace.add_argument(f"--{name}", required=True)
    verify_trace = commands.add_parser("verify-action-trace")
    for name in ("trace-manifest-uri", "verified-qualification-uri", "verified-trace-uri", "run-id"):
        verify_trace.add_argument(f"--{name}", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--verified-trace-uri", required=True)
    evaluate.add_argument("--evaluation-uri", required=True)
    evaluate.add_argument("--run-id", required=True)
    visualize = commands.add_parser("visualize")
    for name in ("verified-trace-uri", "evaluation-uri", "rrd-uri", "manifest-uri", "run-id"):
        visualize.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one argv-safe MolmoAct2 Jetson-Thor pipeline stage.

    Args:
        argv: Optional command-line arguments without the program name.

    Returns:
        Zero after publishing the requested artifact.

    Raises:
        MolmoAct2JetsonThorError: The requested stage fails its input contract.
    """
    args = vars(_parser().parse_args(argv))
    command = args.pop("command")
    handlers = {
        "prepare": prepare_observations,
        "target-qualify": target_qualify,
        "verify-qualification": verify_target_qualification,
        "target-action-trace": target_action_trace,
        "verify-action-trace": verify_target_action_trace,
        "evaluate": evaluate_action_trace,
        "visualize": visualize_action_trace,
    }
    result = handlers[command](**args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
