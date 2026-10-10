"""Run the pinned OpenWAM-alpha to LIBERO workflow with durable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import yaml


OPENWAM_REPOSITORY = "https://github.com/OpenWAM-Official/OpenWAM"
OPENWAM_SOURCE_REF = "48bd67b89d489b14d03b8d92bc66e65d306df32e"
OPENWAM_LICENSE = "Apache-2.0"
FOUNDATION_REPOSITORY = "OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model"
FOUNDATION_REVISION = "52df4e66c82c5c8b480adcc8d01f4db7415dfb56"
WAN_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
WAN_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
LIBERO_DATASET = "OpenWAM/LIBERO"
LIBERO_DATASET_REVISION = "bcb2eaf1121ae4cbd324f8862807abce282500e8"
LIBERO_REPOSITORY = "https://github.com/Lifelong-Robot-Learning/LIBERO"
LIBERO_SOURCE_REF = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_LICENSE = "MIT"
LIBERO_DATA_LICENSE = "CC-BY-4.0"
_NORMALIZATION_OFF_VALUES = (None, "", "none", "null")
_TRAINING_MODE_OVERRIDES = {
    "production": (),
    "diagnostic": ("training.debug=true",),
}
_TRAINING_MODE_DESCRIPTIONS = {
    "production": "upstream production defaults (training.debug=false)",
    "diagnostic": "upstream training.debug=true (20-step diagnostic run)",
}


class OpenWAMPipelineError(RuntimeError):
    """Raised when an OpenWAM workflow artifact violates its contract."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        if not parsed.netloc or not parsed.path.lstrip("/"):
            raise OpenWAMPipelineError(f"S3 URI must contain bucket and key: {uri!r}")
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in {"", "file"}:
        return "", parsed.path if parsed.scheme else uri
    raise OpenWAMPipelineError(f"Unsupported artifact URI: {uri!r}")


def _s3_client():
    import boto3
    from botocore.config import Config

    kwargs: dict[str, object] = {
        "config": Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-central1"),
    }
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def _read_bytes(uri: str) -> bytes:
    bucket, key_or_path = _parse_uri(uri)
    if bucket:
        return _s3_client().get_object(Bucket=bucket, Key=key_or_path)["Body"].read()
    return Path(key_or_path).read_bytes()


def _write_bytes(uri: str, payload: bytes, *, content_type: str) -> None:
    bucket, key_or_path = _parse_uri(uri)
    if bucket:
        _s3_client().put_object(
            Bucket=bucket,
            Key=key_or_path,
            Body=payload,
            ContentType=content_type,
            IfNoneMatch="*",
        )
        return
    path = Path(key_or_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(uri: str, payload: Mapping[str, Any]) -> None:
    _write_bytes(
        uri,
        json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n",
        content_type="application/json",
    )


def _read_json(uri: str) -> dict[str, Any]:
    value = json.loads(_read_bytes(uri))
    if not isinstance(value, dict):
        raise OpenWAMPipelineError(f"Artifact is not a JSON object: {uri}")
    return value


def _upload_file(uri: str, source: Path) -> None:
    bucket, key_or_path = _parse_uri(uri)
    if bucket:
        _s3_client().upload_file(str(source), bucket, key_or_path)
        return
    destination = Path(key_or_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _download_file(uri: str, destination: Path) -> None:
    bucket, key_or_path = _parse_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if bucket:
        _s3_client().download_file(bucket, key_or_path, str(destination))
        return
    shutil.copyfile(key_or_path, destination)


def _runtime_image_provenance(expected_image: str) -> dict[str, str]:
    declared_digest = _image_digest(expected_image)
    if declared_digest is None:
        raise OpenWAMPipelineError(
            "runtime image must be supplied as an immutable digest"
        )
    observed = os.environ.get("NPA_TASK_IMAGE", "").strip()
    observed_digest = _image_digest(observed) if observed else None
    if observed and "@sha256:" in observed and observed_digest is None:
        raise OpenWAMPipelineError("worker image exposes an invalid immutable digest")
    if observed_digest is not None and observed_digest != declared_digest:
        raise OpenWAMPipelineError(
            "worker image digest differs from the declared image digest"
        )
    return {
        "declared": expected_image,
        "declared_digest": declared_digest,
        "observed": observed or "not_exposed",
        "observed_digest": observed_digest or "not_exposed",
    }


def _image_digest(image: str) -> str | None:
    """Return a validated OCI SHA-256 digest from an immutable image reference."""

    prefix, marker, digest = image.rpartition("@sha256:")
    if not prefix or not marker or len(digest) != 64:
        return None
    normalized = digest.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        return None
    return normalized


def _require_repository(root: Path) -> Path:
    required = (root / "scripts" / "train.sh", root / "scripts" / "deploy.py")
    if not all(path.is_file() for path in required):
        raise OpenWAMPipelineError(f"OpenWAM checkout is incomplete at {root}")
    return root


def _copy_repository(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("assets", "outputs", ".git")
    )
    return _require_repository(destination)


def _snapshot_download(
    repo_id: str, revision: str, destination: Path, *, repo_type: str
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(destination),
    )


def _copy_license(source: Path, destination: Path, name: str) -> None:
    candidate = source / "LICENSE"
    if not candidate.is_file():
        raise OpenWAMPipelineError(f"Missing upstream license at {candidate}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, destination / name)


def _clone_libero(destination: Path) -> None:
    subprocess.run(["git", "clone", LIBERO_REPOSITORY, str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", LIBERO_SOURCE_REF],
        check=True,
    )
    observed = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != LIBERO_SOURCE_REF:
        raise OpenWAMPipelineError(f"LIBERO source revision mismatch: {observed}")
    shutil.rmtree(destination / ".git")


def _archive_tree(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        archive.add(source, arcname="payload", recursive=True)
    return {
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise OpenWAMPipelineError(
                f"Archive member escapes destination: {member.name}"
            )
        if member.issym() or member.islnk():
            raise OpenWAMPipelineError(f"Archive links are not accepted: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise OpenWAMPipelineError(
                f"Archive member is not a regular file or directory: {member.name}"
            )
    # ``filter=`` is a Python 3.12 addition.  Validate every member first, then
    # extract each approved regular file/directory individually.  This retains
    # the same safety property on the supported Python 3.10 LIBERO client and
    # deliberately avoids tarfile.extractall's unsafe default API.
    for member in archive.getmembers():
        archive.extract(member, path=destination)


def _restore_archive(manifest: Mapping[str, Any], destination: Path) -> Path:
    archive_uri = str(manifest.get("archive_uri", ""))
    expected = str(manifest.get("archive_sha256", ""))
    if not archive_uri or len(expected) != 64:
        raise OpenWAMPipelineError("Prepared artifact manifest is incomplete")
    archive_path = destination / "input.tar.gz"
    _download_file(archive_uri, archive_path)
    if _sha256_file(archive_path) != expected:
        raise OpenWAMPipelineError("Prepared artifact archive checksum mismatch")
    extracted = destination / "extract"
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, extracted)
    payload = extracted / "payload"
    if not payload.is_dir():
        raise OpenWAMPipelineError("Prepared artifact archive lacks payload directory")
    return payload


def _checkpoint_files(root: Path) -> list[Path]:
    values = sorted(root.rglob("checkpoint_step_*.safetensors"))
    if not values:
        raise OpenWAMPipelineError(f"No OpenWAM checkpoint found below {root}")
    return values


def _normalization_required(config: Path) -> bool:
    """Mirror the pinned OpenWAM loader's saved-config normalization contract."""

    try:
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise OpenWAMPipelineError(
            f"OpenWAM checkpoint config is not valid YAML: {config}"
        ) from exc
    if not isinstance(document, dict):
        raise OpenWAMPipelineError(
            f"OpenWAM checkpoint config is not a mapping: {config}"
        )
    dataloader = document.get("dataloader")
    if dataloader is None:
        return False
    if not isinstance(dataloader, dict):
        raise OpenWAMPipelineError(
            f"OpenWAM checkpoint dataloader config is not a mapping: {config}"
        )
    return dataloader.get("normalize_mode") not in _NORMALIZATION_OFF_VALUES


def _checkpoint_record(root: Path) -> dict[str, Any]:
    checkpoint = _checkpoint_files(root)[-1]
    config = checkpoint.parent / "config.yaml"
    if not config.is_file():
        raise OpenWAMPipelineError("OpenWAM checkpoint is missing config.yaml")
    record: dict[str, Any] = {
        "relative_checkpoint": str(checkpoint.relative_to(root)),
        "sha256": _sha256_file(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "config_sha256": _sha256_file(config),
        "normalization_required": _normalization_required(config),
    }
    if record["normalization_required"]:
        normalization = checkpoint.parent / "normalization_stats.npy"
        if not normalization.is_file():
            raise OpenWAMPipelineError(
                "OpenWAM checkpoint enables action normalization but is missing "
                "normalization_stats.npy"
            )
        record["normalization_sha256"] = _sha256_file(normalization)
    return record


def _command(
    command: list[str], *, cwd: Path, env: Mapping[str, str] | None = None
) -> None:
    merged = dict(os.environ)
    merged.update(env or {})
    subprocess.run(command, cwd=cwd, env=merged, check=True)


def _training_command(foundation: Path, output: Path, training_mode: str) -> list[str]:
    """Build the pinned upstream trainer invocation for one supported mode."""

    try:
        mode_overrides = _TRAINING_MODE_OVERRIDES[training_mode]
    except KeyError as exc:
        raise OpenWAMPipelineError(
            f"Unsupported OpenWAM training mode: {training_mode}"
        ) from exc
    return [
        "bash",
        "scripts/train.sh",
        "dataloader=libero",
        f"training.finetune_ckpt_path={foundation}",
        *mode_overrides,
        f"training.output_path={output}",
    ]


def prepare_assets(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch and archive exact public OpenWAM, Wan, and LIBERO inputs."""

    image = _runtime_image_provenance(args.runtime_image)
    workspace = Path(args.work_dir).resolve()
    repo = _copy_repository(Path(args.openwam_root).resolve(), workspace / "repo")
    assets = repo / "assets"
    foundation = (
        assets
        / "openwam_ckpt"
        / "openwam_alpha"
        / "OpenWAM-Alpha-Pretrain-Foundation-Model"
    )
    backbone = assets / "video_backbone_ckpt" / "Wan2.2-TI2V-5B"
    dataset = assets / "benchmark_data" / "libero"
    _snapshot_download(
        FOUNDATION_REPOSITORY, FOUNDATION_REVISION, foundation, repo_type="model"
    )
    _snapshot_download(WAN_REPOSITORY, WAN_REVISION, backbone, repo_type="model")
    _snapshot_download(
        LIBERO_DATASET, LIBERO_DATASET_REVISION, dataset, repo_type="dataset"
    )
    libero_source = assets / "libero_source"
    _clone_libero(libero_source)
    notices = assets / "notices"
    _copy_license(repo, notices, "OPENWAM_LICENSE")
    _copy_license(libero_source, notices, "LIBERO_LICENSE")
    archive = workspace / "prepared-assets.tar.gz"
    observed = _archive_tree(assets, archive)
    _upload_file(args.archive_uri, archive)
    manifest = {
        "schema": "npa.openwam.prepared-assets.v1",
        "run_id": args.run_id,
        "runtime_image": image,
        "archive_uri": args.archive_uri,
        "archive_sha256": observed["sha256"],
        "archive_size_bytes": observed["size_bytes"],
        "sources": {
            "openwam": {
                "repository": OPENWAM_REPOSITORY,
                "revision": OPENWAM_SOURCE_REF,
                "license": OPENWAM_LICENSE,
            },
            "foundation": {
                "repository": FOUNDATION_REPOSITORY,
                "revision": FOUNDATION_REVISION,
                "license": "Apache-2.0",
                "role": "fine_tune_only",
            },
            "wan22_ti2v_5b": {
                "repository": WAN_REPOSITORY,
                "revision": WAN_REVISION,
                "license": "Apache-2.0",
            },
            "libero_data": {
                "repository": LIBERO_DATASET,
                "revision": LIBERO_DATASET_REVISION,
                "license": LIBERO_DATA_LICENSE,
            },
            "libero_code": {
                "repository": LIBERO_REPOSITORY,
                "revision": LIBERO_SOURCE_REF,
                "license": LIBERO_LICENSE,
            },
        },
        "checkpoint": _checkpoint_record(foundation),
        "redistribution": {
            "image": "operator_private_only",
            "weights": "runtime_fetched_private_artifact",
            "dataset": "runtime_fetched_private_artifact",
        },
    }
    _write_json(args.output_uri, manifest)
    return manifest


def fine_tune(args: argparse.Namespace) -> dict[str, Any]:
    """Run OpenWAM's upstream production or diagnostic fine-tune."""

    image = _runtime_image_provenance(args.runtime_image)
    prepared = _read_json(args.prepared_assets_uri)
    workspace = Path(args.work_dir).resolve()
    assets = _restore_archive(prepared, workspace / "prepared")
    repo = _copy_repository(Path(args.openwam_root).resolve(), workspace / "repo")
    shutil.copytree(assets, repo / "assets")
    foundation = (
        repo
        / "assets"
        / "openwam_ckpt"
        / "openwam_alpha"
        / "OpenWAM-Alpha-Pretrain-Foundation-Model"
    )
    _checkpoint_record(foundation)
    output = workspace / "training-output"
    started = time.monotonic()
    _command(
        _training_command(foundation, output, args.training_mode),
        cwd=repo,
        env={"WANDB_MODE": "offline", "NPROC_PER_NODE": args.gpu_count},
    )
    checkpoint = _checkpoint_record(output)
    archive = workspace / "trained-checkpoint.tar.gz"
    observed = _archive_tree(output, archive)
    _upload_file(args.checkpoint_archive_uri, archive)
    report = {
        "schema": "npa.openwam.fine-tune.v1",
        "run_id": args.run_id,
        "runtime_image": image,
        "prepared_assets_uri": args.prepared_assets_uri,
        "prepared_assets_sha256": prepared["archive_sha256"],
        "architecture": "dual_system/joint_self_attn",
        "video_backbone": "wan22_ti2v_5b",
        "attention_mask": "mutual",
        "training_mode": args.training_mode,
        "mode": _TRAINING_MODE_DESCRIPTIONS[args.training_mode],
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_archive_uri": args.checkpoint_archive_uri,
        "checkpoint_archive_sha256": observed["sha256"],
        "checkpoint_archive_size_bytes": observed["size_bytes"],
        "checkpoint": checkpoint,
    }
    _write_json(args.output_uri, report)
    return report


def _start_server(
    repo: Path, checkpoint_root: Path, port: int
) -> subprocess.Popen[str]:
    record = _checkpoint_record(checkpoint_root)
    process = subprocess.Popen(
        [
            "python",
            "scripts/deploy.py",
            "--ckpt-dir",
            str(checkpoint_root),
            "--ckpt-name",
            Path(record["relative_checkpoint"]).name,
            "--compile-enabled",
            "false",
            "--port",
            str(port),
        ],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process


def _wait_for_port(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise OpenWAMPipelineError(
                f"OpenWAM policy server exited early: {output[-4000:]}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise OpenWAMPipelineError("OpenWAM policy server did not open its WebSocket port")


def _stop_server(process: subprocess.Popen[str]) -> str:
    process.terminate()
    try:
        return process.communicate(timeout=30)[0] or ""
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate(timeout=30)[0] or ""


def _apply_libero_patch(repo: Path, source: Path) -> None:
    """Apply the upstream PyTorch compatibility patch to the copied LIBERO source."""

    patch = repo / "benchmarks" / "libero" / "patches" / "libero-pytorch-load.patch"
    if not patch.is_file():
        raise OpenWAMPipelineError(
            f"Missing OpenWAM LIBERO compatibility patch: {patch}"
        )
    applied = subprocess.run(
        ["patch", "--dry-run", "--reverse", "--strip=1", "--input", str(patch)],
        cwd=source,
        capture_output=True,
        text=True,
    )
    if applied.returncode == 0:
        return
    _command(["patch", "--forward", "--strip=1", "--input", str(patch)], cwd=source)


def _run_libero_trial(
    repo: Path,
    checkpoint_root: Path,
    result: Path,
    *,
    suite: str,
    task_id: int,
    trial_start: int,
    num_trials: int,
    port: int,
) -> dict[str, Any]:
    """Run an explicit upstream LIBERO trial range against the deployed policy."""

    if not suite:
        raise OpenWAMPipelineError("LIBERO suite must not be empty")
    if task_id < 0 or trial_start < 0 or num_trials <= 0:
        raise OpenWAMPipelineError(
            "LIBERO task ID and trial start must be non-negative and num_trials positive"
        )
    source = repo / "assets" / "libero_source"
    _apply_libero_patch(repo, source)
    config = repo / "benchmarks" / "libero" / "policy_config.yml"
    client_python = os.environ.get("LIBERO_PYTHON", "/opt/openwam-libero/bin/python")
    if not Path(client_python).is_file():
        raise OpenWAMPipelineError(
            "LIBERO_PYTHON must name the separately provisioned upstream LIBERO client interpreter"
        )
    process = _start_server(repo, checkpoint_root, port)
    try:
        _wait_for_port(process, port)
        python_path = os.pathsep.join(
            (str(source), str(repo), os.environ.get("PYTHONPATH", ""))
        )
        _command(
            [
                client_python,
                "benchmarks/libero/single_eval.py",
                "--config",
                str(config),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--suite",
                suite,
                "--task-id",
                str(task_id),
                "--trial-start",
                str(trial_start),
                "--num-trials",
                str(num_trials),
                "--result-dir",
                str(result),
            ],
            cwd=repo,
            env={
                "LIBERO_PATH": str(source),
                "LIBERO_CONFIG_ROOT": str(result / "libero-config"),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "PYTHONPATH": python_path,
            },
        )
    finally:
        server_log = _stop_server(process)
        (result / "openwam-server.log").write_text(server_log, encoding="utf-8")
    payload = json.loads((result / "results.json").read_text(encoding="utf-8"))
    expected_stop = trial_start + num_trials
    if (
        payload.get("suite") != suite
        or payload.get("task_id") != task_id
        or payload.get("trial_start") != trial_start
        or payload.get("trial_stop") != expected_stop
        or len(payload.get("trials", [])) != num_trials
        or not isinstance(payload.get("success_rate"), (int, float))
    ):
        raise OpenWAMPipelineError(
            "LIBERO result did not match the requested suite, task, and trial range"
        )
    return payload


def _trial_request(args: argparse.Namespace) -> dict[str, int | str]:
    """Return the explicit upstream evaluation request recorded with each result."""

    return {
        "suite": args.suite,
        "task_id": args.task_id,
        "trial_start": args.trial_start,
        "num_trials": args.num_trials,
    }


def _ensure_heldout(
    request: dict[str, int | str], rollout_report: dict[str, Any]
) -> None:
    """Reject overlapping same-task evaluation ranges before consuming GPU time."""

    prior = rollout_report.get("request")
    if not isinstance(prior, dict):
        raise OpenWAMPipelineError(
            "Rollout report lacks the recorded LIBERO trial request"
        )
    if request["suite"] != prior.get("suite") or request["task_id"] != prior.get(
        "task_id"
    ):
        return
    start = int(request["trial_start"])
    stop = start + int(request["num_trials"])
    prior_start = int(prior.get("trial_start", -1))
    prior_stop = prior_start + int(prior.get("num_trials", 0))
    if start < prior_stop and prior_start < stop:
        raise OpenWAMPipelineError(
            "Held-out LIBERO trials overlap the rollout trial range for the same suite and task"
        )


def _trial_metrics(result: dict[str, Any]) -> tuple[int, float]:
    """Return the trial count and mean policy steps from native evaluator output."""

    trials = result.get("trials")
    if not isinstance(trials, list) or not trials:
        raise OpenWAMPipelineError(
            "LIBERO result does not contain completed trial metrics"
        )
    steps = [trial.get("policy_steps") for trial in trials if isinstance(trial, dict)]
    if len(steps) != len(trials) or not all(
        isinstance(value, (int, float)) for value in steps
    ):
        raise OpenWAMPipelineError(
            "LIBERO trials do not contain numerical policy-step metrics"
        )
    return len(steps), sum(steps) / len(steps)


def rollout(args: argparse.Namespace) -> dict[str, Any]:
    """Deploy the exact trained checkpoint and run an upstream LIBERO trial range."""

    image = _runtime_image_provenance(args.runtime_image)
    prepared = _read_json(args.prepared_assets_uri)
    training = _read_json(args.training_uri)
    workspace = Path(args.work_dir).resolve()
    assets = _restore_archive(prepared, workspace / "prepared")
    trained = _restore_archive(
        {
            "archive_uri": training["checkpoint_archive_uri"],
            "archive_sha256": training["checkpoint_archive_sha256"],
        },
        workspace / "trained",
    )
    repo = _copy_repository(Path(args.openwam_root).resolve(), workspace / "repo")
    shutil.copytree(assets, repo / "assets")
    result_dir = workspace / "rollout"
    result_dir.mkdir(parents=True, exist_ok=True)
    request = _trial_request(args)
    payload = _run_libero_trial(repo, trained, result_dir, port=args.port, **request)
    report = {
        "schema": "npa.openwam.libero-rollout.v1",
        "run_id": args.run_id,
        "runtime_image": image,
        "training_uri": args.training_uri,
        "training_checkpoint_sha256": training["checkpoint"]["sha256"],
        "request": request,
        "result": payload,
        "server_log_sha256": _sha256_file(result_dir / "openwam-server.log"),
    }
    _write_json(args.output_uri, report)
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the trained checkpoint on a non-overlapping LIBERO trial range."""

    image = _runtime_image_provenance(args.runtime_image)
    prepared = _read_json(args.prepared_assets_uri)
    training = _read_json(args.training_uri)
    rollout_report = _read_json(args.rollout_uri)
    workspace = Path(args.work_dir).resolve()
    assets = _restore_archive(prepared, workspace / "prepared")
    trained = _restore_archive(
        {
            "archive_uri": training["checkpoint_archive_uri"],
            "archive_sha256": training["checkpoint_archive_sha256"],
        },
        workspace / "trained",
    )
    repo = _copy_repository(Path(args.openwam_root).resolve(), workspace / "repo")
    shutil.copytree(assets, repo / "assets")
    result_dir = workspace / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    request = _trial_request(args)
    _ensure_heldout(request, rollout_report)
    payload = _run_libero_trial(repo, trained, result_dir, port=args.port, **request)
    report = {
        "schema": "npa.openwam.libero-heldout-evaluation.v1",
        "run_id": args.run_id,
        "runtime_image": image,
        "training_uri": args.training_uri,
        "rollout_uri": args.rollout_uri,
        "request": request,
        "rollout_request": rollout_report["request"],
        "result": payload,
        "server_log_sha256": _sha256_file(result_dir / "openwam-server.log"),
        "limitation": "The configured LIBERO trial range is not a full benchmark aggregate or a training-convergence claim.",
    }
    _write_json(args.output_uri, report)
    return report


def visualize(args: argparse.Namespace) -> dict[str, Any]:
    """Create and independently verify a factual Rerun report from actual stages."""

    training = _read_json(args.training_uri)
    rollout_report = _read_json(args.rollout_uri)
    evaluation = _read_json(args.evaluation_uri)
    local_rrd = Path(args.work_dir).resolve() / "openwam-libero.rrd"
    local_rrd.parent.mkdir(parents=True, exist_ok=True)
    import rerun as rr

    rollout_count, rollout_mean_steps = _trial_metrics(rollout_report["result"])
    evaluation_count, evaluation_mean_steps = _trial_metrics(evaluation["result"])
    recording = rr.RecordingStream("npa.openwam.libero", recording_id=args.run_id)
    recording.save(str(local_rrd))
    recording.log(
        "provenance/run",
        rr.TextDocument(
            json.dumps(
                {
                    "training_uri": args.training_uri,
                    "rollout_uri": args.rollout_uri,
                    "evaluation_uri": args.evaluation_uri,
                },
                sort_keys=True,
            ),
            media_type="application/json",
        ),
        static=True,
    )
    recording.set_time("workflow_stage", sequence=1)
    recording.log(
        "metrics/checkpoint_size_bytes",
        rr.Scalars(training["checkpoint"]["size_bytes"]),
    )
    recording.log(
        "metrics/rollout_success_rate",
        rr.Scalars(rollout_report["result"]["success_rate"]),
    )
    recording.log("metrics/rollout_trial_count", rr.Scalars(rollout_count))
    recording.log("metrics/rollout_mean_policy_steps", rr.Scalars(rollout_mean_steps))
    recording.set_time("workflow_stage", sequence=2)
    recording.log(
        "metrics/heldout_success_rate", rr.Scalars(evaluation["result"]["success_rate"])
    )
    recording.log("metrics/heldout_trial_count", rr.Scalars(evaluation_count))
    recording.log(
        "metrics/heldout_mean_policy_steps", rr.Scalars(evaluation_mean_steps)
    )
    recording.flush()
    recording.disconnect()
    rerun_cli = Path(sys.executable).with_name("rerun")
    if not rerun_cli.is_file():
        raise OpenWAMPipelineError(f"Rerun CLI is unavailable beside {sys.executable}")
    subprocess.run([str(rerun_cli), "rrd", "verify", str(local_rrd)], check=True)
    inspected = subprocess.run(
        [str(rerun_cli), "rrd", "print", "-vv", str(local_rrd)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "metrics/heldout_success_rate" not in inspected:
        raise OpenWAMPipelineError(
            "Rerun inspection did not find held-out metric entity"
        )
    _upload_file(args.rrd_uri, local_rrd)
    report = {
        "schema": "npa.openwam.visualization.v1",
        "run_id": args.run_id,
        "rrd_uri": args.rrd_uri,
        "rrd_sha256": _sha256_file(local_rrd),
        "rrd_size_bytes": local_rrd.stat().st_size,
        "training_checkpoint_sha256": training["checkpoint"]["sha256"],
        "rollout_success_rate": rollout_report["result"]["success_rate"],
        "rollout_trial_count": rollout_count,
        "rollout_mean_policy_steps": rollout_mean_steps,
        "heldout_success_rate": evaluation["result"]["success_rate"],
        "heldout_trial_count": evaluation_count,
        "heldout_mean_policy_steps": evaluation_mean_steps,
        "verification": {
            "rerun_rrd_verify": "passed",
            "entity": "metrics/heldout_success_rate",
        },
    }
    _write_json(args.output_uri, report)
    return report


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--openwam-root", default="/opt/openwam")
    parser.add_argument("--work-dir", required=True)


def _add_stage_parser(
    subparsers: argparse._SubParsersAction, name: str
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    _add_common(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for real OpenWAM workflow stages."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = _add_stage_parser(subparsers, "prepare")
    prepare.add_argument("--archive-uri", required=True)
    prepare.add_argument("--output-uri", required=True)
    train = _add_stage_parser(subparsers, "fine-tune")
    train.add_argument("--prepared-assets-uri", required=True)
    train.add_argument("--checkpoint-archive-uri", required=True)
    train.add_argument("--output-uri", required=True)
    train.add_argument("--gpu-count", default="1")
    train.add_argument(
        "--training-mode",
        choices=tuple(_TRAINING_MODE_OVERRIDES),
        default="production",
        help="Use upstream production defaults or the explicit 20-step diagnostic override.",
    )
    for name in ("rollout", "evaluate"):
        stage = _add_stage_parser(subparsers, name)
        stage.add_argument("--prepared-assets-uri", required=True)
        stage.add_argument("--training-uri", required=True)
        stage.add_argument("--output-uri", required=True)
        stage.add_argument("--port", type=int, default=8848)
        stage.add_argument("--suite", required=True)
        stage.add_argument("--task-id", type=int, required=True)
        stage.add_argument("--trial-start", type=int, required=True)
        stage.add_argument("--num-trials", type=int, required=True)
        if name == "evaluate":
            stage.add_argument("--rollout-uri", required=True)
    visual = _add_stage_parser(subparsers, "visualize")
    visual.add_argument("--training-uri", required=True)
    visual.add_argument("--rollout-uri", required=True)
    visual.add_argument("--evaluation-uri", required=True)
    visual.add_argument("--rrd-uri", required=True)
    visual.add_argument("--output-uri", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Dispatch one substantive OpenWAM workflow stage."""

    args = build_parser().parse_args(argv)
    operations = {
        "prepare": prepare_assets,
        "fine-tune": fine_tune,
        "rollout": rollout,
        "evaluate": evaluate,
        "visualize": visualize,
    }
    operations[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
