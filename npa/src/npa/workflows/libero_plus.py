"""Run the upstream LIBERO-Plus perturbation benchmark with durable evidence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import urlparse
import zipfile


UPSTREAM_REPOSITORY = "https://github.com/sylvestf/LIBERO-plus.git"
UPSTREAM_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
ASSET_REPOSITORY = "Sylvest/LIBERO-plus"
ASSET_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
ASSET_FILE = "assets.zip"
ASSET_SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
DIMENSIONS = (
    "Objects Layout",
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
)


class LiberoPlusError(RuntimeError):
    """Raise when an upstream task, policy, or evidence contract is invalid."""


def _json_bytes(value: object) -> bytes:
    """Encode finite, deterministic JSON.

    Args:
        value: JSON-compatible value.

    Returns:
        UTF-8 JSON bytes ending in a newline.

    Raises:
        ValueError: The value contains a non-finite number.
    """
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _read_uri(uri: str) -> bytes:
    """Read a local file or exact S3 object URI.

    Args:
        uri: Local path, file URI, or S3 URI.

    Returns:
        Artifact bytes.

    Raises:
        LiberoPlusError: The URI is malformed.
    """
    if not uri.startswith("s3://"):
        return Path(uri.removeprefix("file://")).read_bytes()
    import boto3

    parsed = urlparse(uri)
    if not parsed.netloc or not parsed.path.strip("/"):
        raise LiberoPlusError(f"expected exact S3 object URI, got {uri!r}")
    client = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
    return client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))[
        "Body"
    ].read()


def _write_uri(uri: str, payload: bytes) -> None:
    """Write an artifact to a local file or exact S3 object URI.

    Args:
        uri: Destination file or S3 URI.
        payload: Immutable artifact bytes.

    Returns:
        None.

    Raises:
        LiberoPlusError: The URI is malformed.
    """
    if not uri.startswith("s3://"):
        target = Path(uri.removeprefix("file://"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return
    import boto3

    parsed = urlparse(uri)
    if not parsed.netloc or not parsed.path.strip("/"):
        raise LiberoPlusError(f"expected exact S3 object URI, got {uri!r}")
    client = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
    client.put_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Body=payload)
    observed = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))[
        "Body"
    ].read()
    if hashlib.sha256(observed).digest() != hashlib.sha256(payload).digest():
        raise LiberoPlusError("S3 read-after-write hash mismatch")


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for a potentially multi-gigabyte artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_root() -> Path:
    return (
        Path(os.environ.get("NPA_MODEL_CACHE_DIR", "/tmp/npa-model-cache"))
        / "libero-plus"
        / UPSTREAM_REVISION
    )


def _ensure_upstream() -> Path:
    """Fetch the exact upstream source and MIT asset archive into a locked cache.

    Args:
        None.

    Returns:
        Path to the checked-out upstream source tree.

    Raises:
        LiberoPlusError: An immutable identity or safe extraction check fails.
    """
    root = _cache_root()
    ready = root / "READY.json"
    if ready.exists():
        return root / "source"
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{UPSTREAM_REVISION}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready.exists():
            return root / "source"
        if root.exists():
            raise LiberoPlusError(f"incomplete LIBERO-Plus cache exists at {root}")
        temporary = Path(tempfile.mkdtemp(prefix="libero-plus-", dir=root.parent))
        try:
            source = temporary / "source"
            subprocess.run(
                ["git", "clone", "--no-checkout", UPSTREAM_REPOSITORY, str(source)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(source), "checkout", "--detach", UPSTREAM_REVISION],
                check=True,
            )
            resolved = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            if resolved != UPSTREAM_REVISION:
                raise LiberoPlusError(f"upstream revision drift: {resolved}")
            archive = temporary / ASSET_FILE
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:
                raise LiberoPlusError(
                    "runtime image requires huggingface_hub for the pinned asset fetch"
                ) from error
            downloaded = Path(
                hf_hub_download(
                    repo_id=ASSET_REPOSITORY,
                    filename=ASSET_FILE,
                    repo_type="dataset",
                    revision=ASSET_REVISION,
                    local_dir=temporary / "hf",
                )
            )
            shutil.move(downloaded, archive)
            if _sha256_file(archive) != ASSET_SHA256:
                raise LiberoPlusError("LIBERO-Plus asset archive SHA-256 mismatch")
            assets = (source / "libero" / "libero" / "assets").resolve()
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    target = (assets / member.filename).resolve()
                    is_link = (member.external_attr >> 16) & 0o170000 == 0o120000
                    if not target.is_relative_to(assets) or member.is_dir() or is_link:
                        raise LiberoPlusError("unsafe LIBERO-Plus asset archive member")
                    bundle.extract(member, assets)
            _write_uri(
                str(temporary / "READY.json"),
                _json_bytes(
                    {
                        "source_repository": UPSTREAM_REPOSITORY,
                        "source_revision": UPSTREAM_REVISION,
                        "asset_repository": ASSET_REPOSITORY,
                        "asset_revision": ASSET_REVISION,
                        "asset_sha256": ASSET_SHA256,
                    }
                ),
            )
            os.replace(temporary, root)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return root / "source"


def _classification(source: Path) -> list[dict[str, Any]]:
    path = source / "libero" / "libero" / "benchmark" / "task_classification.json"
    records = json.loads(path.read_text())
    tasks = [
        dict(item, suite=suite) for suite, values in records.items() for item in values
    ]
    categories = {str(item["category"]) for item in tasks}
    if len(tasks) != 10030 or categories != set(DIMENSIONS):
        raise LiberoPlusError(
            "upstream task classification no longer matches seven-dimension benchmark"
        )
    return tasks


def prepare(output_uri: str, mode: str, seed: int) -> None:
    """Materialize a pinned, matched task protocol.

    Args:
        output_uri: Exact manifest destination.
        mode: ``smoke`` selects one task per dimension; ``benchmark`` selects all tasks.
        seed: Stable selection and rollout seed.

    Returns:
        None.

    Raises:
        LiberoPlusError: Upstream content does not satisfy the contract.
    """
    source = _ensure_upstream()
    tasks = _classification(source)
    if mode not in {"smoke", "benchmark"}:
        raise LiberoPlusError(f"unsupported protocol mode {mode!r}")
    selected = tasks
    if mode == "smoke":
        selected = []
        for dimension in DIMENSIONS:
            category_tasks = sorted(
                (item for item in tasks if item["category"] == dimension),
                key=lambda item: str(item["id"]),
            )
            selected.append(category_tasks[seed % len(category_tasks)])
    payload = {
        "schema": "npa.libero-plus.task-protocol.v1",
        "mode": mode,
        "seed": seed,
        "task_count": len(selected),
        "tasks": selected,
        "provenance": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "asset_revision": ASSET_REVISION,
            "asset_sha256": ASSET_SHA256,
        },
    }
    _write_uri(output_uri, _json_bytes(payload))


def _policy(adapter: str):
    if adapter == "smoke-zero":
        return lambda _observation, _task: [0.0] * 7
    if ":" not in adapter:
        raise LiberoPlusError(
            "policy adapter must be module:function; smoke-zero is smoke-only"
        )
    module_name, function_name = adapter.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    return factory()


def _write_episode_video(frames: list[Any], target_uri: str) -> dict[str, Any]:
    """Encode observed simulator frames, rather than a synthetic proxy, as MP4."""
    if not frames:
        raise LiberoPlusError("upstream rollout yielded no camera frames")
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise LiberoPlusError(
            "runtime image requires imageio[ffmpeg] for episode MP4 evidence"
        ) from error
    with tempfile.TemporaryDirectory(prefix="libero-plus-video-") as tmp:
        video = Path(tmp) / "episode.mp4"
        imageio.mimsave(video, frames, fps=10, macro_block_size=1)
        payload = video.read_bytes()
    _write_uri(target_uri, payload)
    return {
        "uri": target_uri,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _camera_frame(observation: dict[str, Any]) -> Any | None:
    """Select the native agent-view RGB observation without inventing pixels."""
    for key in ("agentview_image", "agentview_rgb"):
        if key in observation:
            return observation[key]
    return None


def rollout(
    input_uri: str,
    output_uri: str,
    media_uri: str,
    role: str,
    adapter: str,
    max_steps: int,
    matched_baseline_uri: str | None = None,
) -> None:
    """Execute real upstream off-screen rollouts for one matched policy protocol.

    Args:
        input_uri: Prepared task protocol URI.
        output_uri: Exact per-role result URI.
        media_uri: S3-style prefix for actual per-episode MP4 objects.
        role: ``baseline`` or ``candidate`` provenance label.
        adapter: Callable policy adapter or smoke-only zero policy.
        max_steps: Upstream simulator action horizon.
        matched_baseline_uri: Existing baseline result required for a candidate run.

    Returns:
        None.

    Raises:
        LiberoPlusError: The policy or upstream simulation cannot run.
    """
    protocol = json.loads(_read_uri(input_uri))
    if protocol["mode"] == "benchmark" and adapter == "smoke-zero":
        raise LiberoPlusError("smoke-zero cannot produce a complete benchmark result")
    protocol_payload = _read_uri(input_uri)
    protocol_sha256 = hashlib.sha256(protocol_payload).hexdigest()
    matched_baseline_sha256: str | None = None
    if role == "candidate":
        if not matched_baseline_uri:
            raise LiberoPlusError(
                "candidate rollout requires the exact baseline result"
            )
        baseline_payload = _read_uri(matched_baseline_uri)
        baseline = json.loads(baseline_payload)
        if baseline.get("protocol_sha256") != protocol_sha256:
            raise LiberoPlusError(
                "candidate baseline did not consume the exact prepared protocol"
            )
        matched_baseline_sha256 = hashlib.sha256(baseline_payload).hexdigest()
    source = _ensure_upstream()
    os.environ["LIBERO_CONFIG_PATH"] = str(source / ".npa-libero-config")
    config = source / ".npa-libero-config" / "config.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        "benchmark_root: %s\nbddl_files: %s\ninit_states: %s\nassets: %s\ndatasets: %s\n"
        % tuple(
            str(source / "libero" / "libero" / part)
            for part in (".", "bddl_files", "init_files", "assets", "datasets")
        )
    )
    sys.path.insert(0, str(source / "libero"))
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import numpy as np

    act = _policy(adapter)
    results = []
    for record in protocol["tasks"]:
        benchmark = get_benchmark(record["suite"])()
        index = next(
            index
            for index, task in enumerate(benchmark.tasks)
            if task.name == record["name"]
        )
        task = benchmark.get_task(index)
        bddl = (
            source
            / "libero"
            / "libero"
            / "bddl_files"
            / task.problem_folder
            / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=128, camera_widths=128
        )
        try:
            env.seed(protocol["seed"])
            observation = env.set_init_state(benchmark.get_task_init_states(index)[0])
            frames = (
                [frame] if (frame := _camera_frame(observation)) is not None else []
            )
            done = False
            steps = 0
            for steps in range(1, max_steps + 1):
                observation, _reward, done, _info = env.step(
                    np.asarray(act(observation, record), dtype=float)
                )
                if (frame := _camera_frame(observation)) is not None:
                    frames.append(frame)
                if done:
                    break
        finally:
            env.close()
        task_key = f"{int(record['id']):05d}.mp4"
        evidence = _write_episode_video(frames, f"{media_uri.rstrip('/')}/{task_key}")
        results.append(
            {
                "task_id": record["id"],
                "name": record["name"],
                "category": record["category"],
                "success": bool(done),
                "steps": steps,
                "episode_media": evidence,
            }
        )
    _write_uri(
        output_uri,
        _json_bytes(
            {
                "schema": "npa.libero-plus.rollouts.v1",
                "role": role,
                "protocol_sha256": protocol_sha256,
                "adapter": adapter,
                "results": results,
                "matched_baseline_sha256": matched_baseline_sha256,
                "smoke_only": adapter == "smoke-zero",
            }
        ),
    )


def compare(
    protocol_uri: str, baseline_uri: str, candidate_uri: str, output_uri: str
) -> None:
    """Calculate per-dimension candidate-minus-baseline robustness deltas."""
    protocol = _read_uri(protocol_uri)
    baseline_payload = _read_uri(baseline_uri)
    candidate_payload = _read_uri(candidate_uri)
    baseline = json.loads(baseline_payload)
    candidate = json.loads(candidate_payload)
    digest = hashlib.sha256(protocol).hexdigest()
    if baseline["protocol_sha256"] != digest or candidate["protocol_sha256"] != digest:
        raise LiberoPlusError("rollouts did not consume the exact prepared protocol")
    values: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
    for left, right in zip(baseline["results"], candidate["results"], strict=True):
        if left["task_id"] != right["task_id"] or left["category"] != right["category"]:
            raise LiberoPlusError("baseline and candidate task protocols differ")
        values[left["category"]].append(
            float(right["success"]) - float(left["success"])
        )
    if any(not value for value in values.values()):
        raise LiberoPlusError(
            "matched rollouts did not cover every LIBERO-Plus dimension"
        )
    deltas = {key: sum(value) / len(value) for key, value in values.items()}
    _write_uri(
        output_uri,
        _json_bytes(
            {
                "schema": "npa.libero-plus.robustness-deltas.v1",
                "protocol_sha256": digest,
                "deltas": deltas,
                "mean_delta": sum(deltas.values()) / len(deltas),
                "smoke_only": baseline["smoke_only"] or candidate["smoke_only"],
            }
        ),
    )


def _emit_rrd(delta: dict[str, Any], target_uri: str) -> dict[str, Any]:
    """Record actual robustness metrics into a self-contained Rerun recording."""
    try:
        import rerun as rr
    except ImportError as error:
        raise LiberoPlusError(
            "runtime image requires rerun-sdk for RRD evidence"
        ) from error
    with tempfile.TemporaryDirectory(prefix="libero-plus-rrd-") as tmp:
        target = Path(tmp) / "task-level-results.rrd"
        recording = rr.RecordingStream("npa_libero_plus_robustness")
        recording.save(str(target))
        for category, value in delta["deltas"].items():
            entity = category.lower().replace(" ", "_")
            rr.log(
                f"robustness/{entity}", rr.Scalars(float(value)), recording=recording
            )
        rr.log(
            "robustness/mean",
            rr.Scalars(float(delta["mean_delta"])),
            recording=recording,
        )
        recording.flush()
        recording.disconnect()
        payload = target.read_bytes()
    _write_uri(target_uri, payload)
    return {
        "uri": target_uri,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def report(
    delta_uri: str, baseline_uri: str, candidate_uri: str, output_uri: str, rrd_uri: str
) -> None:
    """Emit task-level JSON plus a decoded RRD of measured robustness deltas."""
    delta = json.loads(_read_uri(delta_uri))
    baseline_payload = _read_uri(baseline_uri)
    candidate_payload = _read_uri(candidate_uri)
    baseline = json.loads(baseline_payload)
    candidate = json.loads(candidate_payload)
    rrd = _emit_rrd(delta, rrd_uri)
    _write_uri(
        output_uri,
        _json_bytes(
            {
                "schema": "npa.libero-plus.report.v1",
                "delta": delta,
                "baseline_results_sha256": hashlib.sha256(baseline_payload).hexdigest(),
                "candidate_results_sha256": hashlib.sha256(
                    candidate_payload
                ).hexdigest(),
                "rrd": rrd,
                "episode_media": {
                    "baseline": [row["episode_media"] for row in baseline["results"]],
                    "candidate": [row["episode_media"] for row in candidate["results"]],
                },
                "limitations": "smoke evidence is not a complete benchmark, convergence result, or physical-robot result",
            }
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """Dispatch one workflow stage from the installed runtime image."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output-uri", required=True)
    prepare_parser.add_argument("--mode", choices=("smoke", "benchmark"), required=True)
    prepare_parser.add_argument("--seed", type=int, required=True)
    rollout_parser = commands.add_parser("rollout")
    rollout_parser.add_argument("--input-uri", required=True)
    rollout_parser.add_argument("--output-uri", required=True)
    rollout_parser.add_argument("--media-uri", required=True)
    rollout_parser.add_argument(
        "--role", choices=("baseline", "candidate"), required=True
    )
    rollout_parser.add_argument("--policy-adapter", required=True)
    rollout_parser.add_argument("--max-steps", type=int, required=True)
    rollout_parser.add_argument("--matched-baseline-uri")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--protocol-uri", required=True)
    compare_parser.add_argument("--baseline-uri", required=True)
    compare_parser.add_argument("--candidate-uri", required=True)
    compare_parser.add_argument("--output-uri", required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--delta-uri", required=True)
    report_parser.add_argument("--baseline-uri", required=True)
    report_parser.add_argument("--candidate-uri", required=True)
    report_parser.add_argument("--output-uri", required=True)
    report_parser.add_argument("--rrd-uri", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.output_uri, args.mode, args.seed)
    elif args.command == "rollout":
        rollout(
            args.input_uri,
            args.output_uri,
            args.media_uri,
            args.role,
            args.policy_adapter,
            args.max_steps,
            args.matched_baseline_uri,
        )
    elif args.command == "compare":
        compare(
            args.protocol_uri, args.baseline_uri, args.candidate_uri, args.output_uri
        )
    else:
        report(
            args.delta_uri,
            args.baseline_uri,
            args.candidate_uri,
            args.output_uri,
            args.rrd_uri,
        )


if __name__ == "__main__":
    main()
