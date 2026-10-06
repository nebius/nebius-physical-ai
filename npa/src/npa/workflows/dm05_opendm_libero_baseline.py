"""Evaluate the released OpenDM DM05 LIBERO checkpoint for a matched comparison.

This adapter invokes Dexmal's pinned OpenDM inference service and Dexbotic's
native LIBERO evaluator.  It only owns NPA's durable artifact hand-off and
normalization of evaluator output into the comparison's existing rollout
schema.  The model emits absolute actions; LIBERO's OSC pose controller remains
relative, so neither adapter invents an action conversion.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from npa.workflows.lerobot_transfer_data import (
    file_sha256,
    materialize,
    publish,
    write_json,
)


BASELINE_CHECKPOINT = {
    "repository": "Dexmal/DM05-libero",
    "revision": "25a8e0d38a8eaeaae41a44d7b4a2378fd8ce1088",
    "license_label": "gemma",
}
OPENDM_IMPLEMENTATION = {
    "schema": "npa.dm05_opendm_libero_baseline.runtime.v1",
    "opendm": {
        "repository": "https://github.com/dexmal/opendm",
        "revision": "7d52f1591437332cb0157be3303c1c46da811344",
        "license": "Apache-2.0",
        "entrypoint": "playground/dm05_libero.py",
    },
    "dexbotic": {
        "repository": "https://github.com/dexmal/dexbotic-benchmark",
        "revision": "789b87f50d9fadc7663d2e8bac057941221aab81",
        "license": "MIT",
    },
    "libero": {
        "repository": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "revision": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "license": "MIT",
    },
    "checkpoint": BASELINE_CHECKPOINT,
}
RUNTIME_MANIFEST_ENV = "NPA_DM05_OPENDM_RUNTIME_MANIFEST"
RUNTIME_MANIFEST = Path("/opt/opendm/dm05-libero-baseline-runtime.json")
OPENDM_ROOT_ENV = "NPA_DM05_OPENDM_ROOT"
DEXBOTIC_ROOT_ENV = "NPA_DM05_DEXBOTIC_ROOT"
OPENDM_PYTHON_ENV = "NPA_DM05_OPENDM_PYTHON"
DEXBOTIC_PYTHON_ENV = "NPA_DM05_DEXBOTIC_PYTHON"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STATE_DIMENSION = 8
ACTION_DIMENSION = 7
CHUNK_SIZE = 10
LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_PORT = 7891


class OpenDMBaselineError(RuntimeError):
    """Report an invalid native OpenDM checkpoint, runtime, or evaluator artifact."""


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI contract used by the OpenDM baseline workflow toolRef.

    Returns:
        The parser for the single native rollout stage.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    rollout = commands.add_parser("rollout")
    rollout.add_argument("--input-path", required=True)
    rollout.add_argument("--output-path", required=True)
    rollout.add_argument("--server-device", default="0")
    rollout.add_argument("--evaluator-device", default="1")
    return parser


def _runtime_path(value: str | None, default: Path) -> Path:
    """Resolve one required image-owned runtime path.

    Args:
        value: Optional environment override for a path.
        default: Image-owned default path.

    Returns:
        A resolved existing path.

    Raises:
        OpenDMBaselineError: If the selected path is absent.
    """
    path = Path(value) if value else default
    if not path.is_dir():
        raise OpenDMBaselineError(f"required OpenDM runtime path is missing: {path}")
    return path


def _runtime_python(value: str | None, default: Path) -> Path:
    """Resolve one executable Python interpreter supplied by the image.

    Args:
        value: Optional environment override for an interpreter.
        default: Image-owned interpreter path.

    Returns:
        The selected executable path.

    Raises:
        OpenDMBaselineError: If the interpreter is absent or not executable.
    """
    path = Path(value) if value else default
    if not path.is_file() or not os.access(path, os.X_OK):
        raise OpenDMBaselineError(f"required OpenDM Python is missing: {path}")
    return path


def _runtime_paths() -> tuple[Path, Path, Path, Path]:
    """Return the exact source and interpreter paths supplied by the private image.

    Returns:
        OpenDM root, Dexbotic root, OpenDM Python, and evaluator Python.
    """
    opendm = _runtime_path(os.environ.get(OPENDM_ROOT_ENV), Path("/opt/opendm"))
    dexbotic = _runtime_path(
        os.environ.get(DEXBOTIC_ROOT_ENV), Path("/opt/dexbotic-benchmark")
    )
    opendm_python = _runtime_python(
        os.environ.get(OPENDM_PYTHON_ENV), Path("/opt/opendm-venv/bin/python")
    )
    evaluator_python = _runtime_python(
        os.environ.get(DEXBOTIC_PYTHON_ENV), Path("/opt/libero-venv/bin/python")
    )
    return opendm, dexbotic, opendm_python, evaluator_python


def _read_runtime_manifest() -> dict[str, Any]:
    """Require the image's exact OpenDM/Dexbotic source manifest.

    Returns:
        The verified manifest.

    Raises:
        OpenDMBaselineError: If the manifest is missing or differs from the review pin.
    """
    source = Path(os.environ.get(RUNTIME_MANIFEST_ENV, RUNTIME_MANIFEST))
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OpenDMBaselineError(
            "OpenDM baseline runtime manifest is unreadable"
        ) from error
    if manifest != OPENDM_IMPLEMENTATION:
        raise OpenDMBaselineError(
            "OpenDM baseline runtime manifest does not match the reviewed source pin"
        )
    return manifest


def _download_checkpoint(workspace: Path) -> Path:
    """Fetch the released baseline checkpoint at its immutable Hub revision.

    Args:
        workspace: Run-local directory that receives runtime-only checkpoint bytes.

    Returns:
        The downloaded checkpoint directory.
    """
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=BASELINE_CHECKPOINT["repository"],
            revision=BASELINE_CHECKPOINT["revision"],
            local_dir=str(workspace / "checkpoint"),
            local_dir_use_symlinks=False,
        )
    )


def _vector_dimension(payload: dict[str, Any], key: str) -> int:
    """Read one normalization vector's exact dimension.

    Args:
        payload: Checkpoint normalization document.
        key: ``state`` or ``action`` normalization key.

    Returns:
        The shared mean/std vector dimension.

    Raises:
        OpenDMBaselineError: If the normalization payload is malformed.
    """
    vector = payload.get("norm_stats", {}).get(key, {})
    mean, standard_deviation = vector.get("mean"), vector.get("std")
    if not isinstance(mean, list) or not isinstance(standard_deviation, list):
        raise OpenDMBaselineError(f"checkpoint norm_stats lacks {key} mean/std vectors")
    if len(mean) != len(standard_deviation):
        raise OpenDMBaselineError(f"checkpoint {key} normalization vectors disagree")
    return len(mean)


def _require_checkpoint_contract(checkpoint: Path) -> dict[str, Any]:
    """Verify the released baseline's raw config and LIBERO normalization contract.

    Args:
        checkpoint: Downloaded baseline checkpoint root.

    Returns:
        A compact record of raw and effective action dimensions.

    Raises:
        OpenDMBaselineError: If the checkpoint cannot support the sealed protocol.
    """
    try:
        config = json.loads((checkpoint / "config.json").read_text())
        normalization = json.loads((checkpoint / "norm_stats.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OpenDMBaselineError(
            "baseline checkpoint lacks readable config or norm_stats"
        ) from error
    if config.get("model_type") != "dm05":
        raise OpenDMBaselineError(
            "baseline checkpoint is not an OpenDM DM05 checkpoint"
        )
    if config.get("chunk_size") != 50 or config.get("action_dim") != 32:
        raise OpenDMBaselineError(
            "baseline checkpoint raw config is not the released DM05-libero"
        )
    if _vector_dimension(normalization, "state") != STATE_DIMENSION:
        raise OpenDMBaselineError(
            "baseline checkpoint does not normalize an 8-value state"
        )
    if _vector_dimension(normalization, "action") != ACTION_DIMENSION:
        raise OpenDMBaselineError(
            "baseline checkpoint does not normalize a 7-value action"
        )
    return {
        "raw_config_chunk_size": config["chunk_size"],
        "raw_config_action_dimension": config["action_dim"],
        "effective_chunk_size": CHUNK_SIZE,
        "effective_action_dimension": ACTION_DIMENSION,
    }


def _require_protocol(protocol: dict[str, Any]) -> None:
    """Reject a sealed protocol that changes the published representation boundary.

    Args:
        protocol: Materialized comparison protocol.

    Raises:
        OpenDMBaselineError: If protocol shape, suites, seed, or controller differs.
    """
    action = protocol.get("action", {})
    observation = protocol.get("observation", {})
    if tuple(protocol.get("suites", ())) != SUITES:
        raise OpenDMBaselineError(
            "OpenDM baseline requires the four published LIBERO suites"
        )
    if protocol.get("episodes_per_task") != 5 or protocol.get("tasks_per_suite") != 10:
        raise OpenDMBaselineError(
            "OpenDM baseline requires five trials for every LIBERO task"
        )
    seed = protocol.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise OpenDMBaselineError(
            "OpenDM baseline requires a nonnegative integer sealed protocol seed"
        )
    if observation.get("state_dimension") != STATE_DIMENSION:
        raise OpenDMBaselineError("OpenDM baseline requires an 8-value LIBERO state")
    if (
        action.get("dimension") != ACTION_DIMENSION
        or action.get("chunk_size") != CHUNK_SIZE
    ):
        raise OpenDMBaselineError(
            "OpenDM baseline requires 7D actions in chunks of ten"
        )
    if action.get("model_representation") != "absolute":
        raise OpenDMBaselineError("OpenDM baseline model actions must remain absolute")
    if action.get("environment_controller") != "relative":
        raise OpenDMBaselineError(
            "OpenDM baseline requires the relative LIBERO controller"
        )


def _native_environment(python: Path, root: Path) -> dict[str, str]:
    """Construct the environment required by the upstream shell launcher.

    Args:
        python: Image-owned interpreter for the selected upstream source.
        root: Exact upstream source root.

    Returns:
        The isolated environment for one native child process.
    """
    environment = os.environ.copy()
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(root / "libero"), environment.get("PYTHONPATH", ""))
    )
    environment["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    return environment


def _server_command(checkpoint: Path) -> list[str]:
    """Build the model-card-compatible OpenDM LIBERO inference command.

    Args:
        checkpoint: Exact runtime-fetched checkpoint path.

    Returns:
        The unredacted local command passed only to the owned worker.
    """
    return [
        "bash",
        "script/dm05_launcher.sh",
        "--exp",
        "playground/dm05_libero.py",
        "--task",
        "inference",
        "--nproc_per_node",
        "1",
        "--model-config.model-name-or-path",
        str(checkpoint),
        "--model-config.chunk-size",
        str(CHUNK_SIZE),
        "--inference-config.output-action-dim",
        str(ACTION_DIMENSION),
        "--inference-config.port",
        str(LOOPBACK_PORT),
    ]


def _start_server(
    root: Path, python: Path, checkpoint: Path, device: str
) -> subprocess.Popen[bytes]:
    """Start one owned inference server pinned to one worker-local GPU.

    Args:
        root: OpenDM source root.
        python: OpenDM interpreter.
        checkpoint: Runtime-fetched checkpoint directory.
        device: Worker-local CUDA device visible to the server.

    Returns:
        The owned inference process.
    """
    environment = _native_environment(python, root)
    environment["PYTHON"] = str(python)
    environment["CUDA_VISIBLE_DEVICES"] = device
    return subprocess.Popen(_server_command(checkpoint), cwd=root, env=environment)


def _wait_for_server(process: subprocess.Popen[bytes]) -> None:
    """Wait until the owned OpenDM service listens on its process-local endpoint.

    Args:
        process: Owned inference server.

    Raises:
        OpenDMBaselineError: If the child exits before opening the endpoint.
    """
    while True:
        if process.poll() is not None:
            raise OpenDMBaselineError(
                f"OpenDM inference exited before readiness, rc={process.returncode}"
            )
        try:
            connection = http.client.HTTPConnection(
                LOOPBACK_HOST, LOOPBACK_PORT, timeout=5
            )
            connection.request("GET", "/v1/infer")
            response = connection.getresponse()
            response.read()
            connection.close()
            if 300 <= response.status < 400:
                raise OpenDMBaselineError(
                    "OpenDM readiness endpoint returned a redirect"
                )
            return
        except (OSError, http.client.HTTPException):
            time.sleep(2)


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    """Terminate only the inference server started by this adapter.

    Args:
        process: Owned inference server.

    Returns:
        None.
    """
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _evaluator_command(suite: str, output: Path, trials: int, seed: int) -> list[str]:
    """Build one official Dexbotic evaluation command for a LIBERO suite.

    Args:
        suite: LIBERO suite to evaluate.
        output: Native result directory for that suite.
        trials: Trials required for each task.
        seed: Sealed protocol seed forwarded through Dexbotic's ``--set`` merge.

    Returns:
        The native evaluator argv.
    """
    return [
        "evaluation/run_libero_evaluation.py",
        "--config",
        "evaluation/configs/libero/example_dm05_libero.yaml",
        "--set",
        "benchmark",
        suite,
        "--set",
        "base_url",
        f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}",
        "--set",
        "output_dir",
        str(output),
        "--set",
        "num_trails_per_task",
        str(trials),
        "--set",
        "seed",
        str(seed),
    ]


def _run_suite(
    root: Path,
    python: Path,
    suite: str,
    output: Path,
    trials: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Run and parse one native Dexbotic suite without replacing its evaluator.

    Args:
        root: Dexbotic source root.
        python: Evaluator interpreter.
        suite: LIBERO suite name.
        output: Native evaluator result directory.
        trials: Trials per task.
        seed: Sealed protocol seed passed to Dexbotic's configuration merge.
        device: Worker-local CUDA device visible to the evaluator.

    Returns:
        Parsed upstream ``results.json``.
    """
    environment = _native_environment(python, root)
    environment.update({"EGL_PLATFORM": "device", "PYOPENGL_PLATFORM": "egl"})
    environment["CUDA_VISIBLE_DEVICES"] = device
    subprocess.run(
        [str(python), *_evaluator_command(suite, output, trials, seed)],
        cwd=root,
        env=environment,
        check=True,
    )
    results = output / "results.json"
    if not results.is_file():
        raise OpenDMBaselineError(f"Dexbotic did not write {suite} results.json")
    try:
        return json.loads(results.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OpenDMBaselineError(
            f"Dexbotic {suite} results are invalid JSON"
        ) from error


def _suite_records(
    suite: str, result: dict[str, Any], output: Path, trials: int
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Validate and normalize one suite's native results and videos.

    Args:
        suite: Evaluated LIBERO suite.
        result: Parsed upstream results document.
        output: Native result directory containing evaluator videos.
        trials: Required trials per task.

    Returns:
        Canonical task records, relative video paths, and success count.

    Raises:
        OpenDMBaselineError: If a task grid, success count, or video set is incomplete.
    """
    tasks = result.get("task_results")
    if (
        result.get("total_tasks") != 10
        or not isinstance(tasks, list)
        or len(tasks) != 10
    ):
        raise OpenDMBaselineError(f"Dexbotic {suite} did not evaluate ten tasks")
    records: list[dict[str, Any]] = []
    successes = 0
    for task_id, task in enumerate(tasks):
        episodes = task.get("episode_results") if isinstance(task, dict) else None
        if not isinstance(episodes, list) or len(episodes) != trials:
            raise OpenDMBaselineError(
                f"Dexbotic {suite} task {task_id} has wrong trial count"
            )
        values = [
            episode.get("success") for episode in episodes if isinstance(episode, dict)
        ]
        if len(values) != trials or any(
            not isinstance(value, bool) for value in values
        ):
            raise OpenDMBaselineError(
                f"Dexbotic {suite} task {task_id} has non-boolean success"
            )
        successes += sum(values)
        records.append(
            {"task_group": suite, "task_id": task_id, "metrics": {"successes": values}}
        )
    if (
        result.get("total_episodes") != 10 * trials
        or result.get("successful_episodes") != successes
    ):
        raise OpenDMBaselineError(
            f"Dexbotic {suite} summary disagrees with task results"
        )
    videos = sorted(
        path.relative_to(output).as_posix()
        for path in (output / "videos").glob("*.mp4")
    )
    if len(videos) != 10 * trials:
        raise OpenDMBaselineError(f"Dexbotic {suite} did not retain every rollout MP4")
    return records, videos, successes


def _copy_suite_outputs(native: Path, suite: str, suite_root: Path) -> Path:
    """Retain one evaluator directory under the durable canonical artifact tree.

    Args:
        native: Canonical artifact's native root.
        suite: Evaluated suite name.
        suite_root: Temporary upstream evaluator output.

    Returns:
        The copied suite directory.
    """
    target = native / suite
    shutil.copytree(suite_root, target)
    return target


def _write_canonical_eval_info(
    native: Path,
    protocol: dict[str, Any],
    suite_results: list[tuple[str, dict[str, Any]]],
) -> tuple[Path, list[str], dict[str, int]]:
    """Write the LeRobot-compatible evaluator record from Dexbotic's raw outputs.

    Args:
        native: Canonical native evidence root.
        protocol: Sealed comparison protocol.
        suite_results: Every suite's copied path and parsed native result.

    Returns:
        Eval-info path, all relative videos, and successes by suite.
    """
    tasks: list[dict[str, Any]] = []
    videos: list[str] = []
    successes: dict[str, int] = {}
    trials = int(protocol["episodes_per_task"])
    for suite, result in suite_results:
        records, suite_videos, suite_successes = _suite_records(
            suite, result, native / suite, trials
        )
        tasks.extend(records)
        videos.extend(f"native/{suite}/{path}" for path in suite_videos)
        successes[suite] = suite_successes
    info = native / "eval_info.json"
    write_json(
        info,
        {
            "per_task": tasks,
            "per_group": successes,
            "overall": {"successes": sum(successes.values())},
        },
    )
    return info, videos, successes


def run_rollout(
    protocol_root: Path, output: Path, *, server_device: str, evaluator_device: str
) -> None:
    """Run the released OpenDM checkpoint over all sealed LIBERO suites.

    Args:
        protocol_root: Materialized protocol artifact directory.
        output: Local output directory later published by NPA.
        server_device: Worker-local GPU selection for inference.
        evaluator_device: Worker-local GPU selection for renderer/evaluator.

    Returns:
        None.

    Raises:
        OpenDMBaselineError: If native source, checkpoint, or evaluator output diverges.
    """
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    _require_protocol(protocol)
    manifest = _read_runtime_manifest()
    opendm, dexbotic, opendm_python, evaluator_python = _runtime_paths()
    with tempfile.TemporaryDirectory(prefix="dm05-opendm-baseline-") as temporary:
        workspace = Path(temporary)
        checkpoint = _download_checkpoint(workspace)
        checkpoint_contract = _require_checkpoint_contract(checkpoint)
        server = _start_server(opendm, opendm_python, checkpoint, server_device)
        try:
            _wait_for_server(server)
            native, suite_outputs = _evaluate_suites(
                dexbotic, evaluator_python, workspace, protocol, evaluator_device
            )
        finally:
            _stop_server(server)
        info, videos, successes = _write_canonical_eval_info(
            native, protocol, suite_outputs
        )
        _write_rollout(
            output,
            protocol_root,
            protocol,
            manifest,
            checkpoint_contract,
            info,
            videos,
            successes,
        )


def _evaluate_suites(
    root: Path, python: Path, workspace: Path, protocol: dict[str, Any], device: str
) -> tuple[Path, list[tuple[str, dict[str, Any]]]]:
    """Evaluate all protocol suites and retain native result directories.

    Args:
        root: Dexbotic source root.
        python: Evaluator interpreter.
        workspace: Temporary working directory.
        protocol: Sealed comparison protocol.
        device: Worker-local evaluator GPU.

    Returns:
        Native evidence root and copied suite result names with parsed results.
    """
    native = workspace / "canonical-native"
    results: list[tuple[str, dict[str, Any]]] = []
    seed = protocol["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise OpenDMBaselineError(
            "sealed protocol seed was not validated as an integer"
        )
    for suite in SUITES:
        suite_root = workspace / "dexbotic" / suite
        result = _run_suite(
            root, python, suite, suite_root, protocol["episodes_per_task"], seed, device
        )
        _copy_suite_outputs(native, suite, suite_root)
        results.append((suite, result))
    return native, results


def _write_rollout(
    output: Path,
    protocol_root: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    checkpoint_contract: dict[str, Any],
    info: Path,
    videos: list[str],
    successes: dict[str, int],
) -> None:
    """Write the exact rollout artifact consumed by the common metrics stage.

    Args:
        output: Local rollout output tree.
        protocol_root: Materialized protocol directory.
        protocol: Sealed comparison protocol.
        manifest: Verified source/runtime manifest.
        checkpoint_contract: Verified checkpoint dimension record.
        info: Canonical native eval-info file.
        videos: Canonical relative MP4 paths.
        successes: Measured success totals by suite.

    Returns:
        None.
    """
    source_native = info.parent
    target_native = output / "native"
    shutil.copytree(source_native, target_native)
    copied_info = target_native / info.name
    write_json(
        output / "rollout.json",
        {
            "schema": "npa.dm05_lerobot_libero.rollout.v1",
            "checkpoint_role": "baseline",
            "checkpoint": BASELINE_CHECKPOINT,
            "protocol_sha256": file_sha256(protocol_root / "protocol.json"),
            "command": [
                "script/dm05_launcher.sh",
                "evaluation/run_libero_evaluation.py",
            ],
            "successes_by_suite": successes,
            "successes": sum(successes.values()),
            "episodes": protocol["total_episodes"],
            "native_eval_info_sha256": file_sha256(copied_info),
            "videos": videos,
            "model_action_representation": "absolute",
            "environment_controller": "relative",
            "opendm_runtime": manifest,
            "checkpoint_contract": checkpoint_contract,
            "native_checkpoint_load_verified": True,
            "native_evaluator": "Dexbotic LIBERO legacy /process_frame",
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute and publish the native OpenDM baseline rollout stage.

    Args:
        argv: Optional arguments excluding the executable name.

    Returns:
        Zero after a successful durable publish.
    """
    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="dm05-opendm-baseline-stage-") as temporary:
        workspace = Path(temporary)
        protocol = materialize(args.input_path, workspace / "protocol")
        output = workspace / "output"
        run_rollout(
            protocol,
            output,
            server_device=args.server_device,
            evaluator_device=args.evaluator_device,
        )
        publish(output, args.output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
