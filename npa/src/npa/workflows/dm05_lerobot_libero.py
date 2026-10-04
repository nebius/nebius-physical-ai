"""Run the published DM05 LeRobot LIBERO comparison without inventing a score.

The model card publishes a 40-task, five-episode-per-task protocol.  This
adapter seals that protocol before either policy runs, delegates closed-loop
rollouts to upstream ``lerobot-eval``, and only derives metrics from the two
returned ``eval_info.json`` files.  In particular, the model's absolute action
processor and LIBERO's relative environment controller are distinct contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from npa.workflows.lerobot_transfer_data import (
    file_sha256,
    materialize,
    publish,
    write_json,
)

MODEL_REPOSITORIES = {
    "baseline": ("Dexmal/DM05-Lerobot", "716afe317bfd01fa4d7ad7cfb84e3b19b7bd934d"),
    "candidate": (
        "Dexmal/DM05-Lerobot-LIBERO",
        "c22df98af5a69e7b9f6bfc1086d1a6982e647b26",
    ),
}
# `DM05-Lerobot` is the documented predecessor, but its published checkpoint is
# a 14-state/14-action, 50-step general policy.  It must be identified as such
# rather than silently validated against the 8-state/7-action LIBERO contract
# of the enhanced checkpoint.  A rollout rejects that representation mismatch
# before it can invent an action adapter or call the evaluation comparable.
CHECKPOINT_CONTRACTS = {
    "baseline": {
        "type": "dm05",
        "use_relative_actions": False,
        "add_state": True,
        "chunk_size": 50,
        "n_action_steps": 50,
        "state_dimension": 14,
        "action_dimension": 14,
    },
    "candidate": {
        "type": "dm05",
        "use_relative_actions": False,
        "add_state": False,
        "chunk_size": 10,
        "n_action_steps": 10,
        "state_dimension": 8,
        "action_dimension": 7,
    },
}
# The released checkpoint was prepared with the unmerged/superseded upstream
# implementation below.  It is intentionally not replaced with a similarly
# named current policy: the checkpoint's processor files are the original
# format.  The derivative image writes this exact manifest at build time and
# every GPU rollout rejects a different runtime before it fetches weights.
DM05_IMPLEMENTATION = {
    "schema": "npa.dm05_lerobot.runtime.v1",
    "repository": "https://github.com/hbzfeng/lerobot",
    "revision": "6eede4f7d2efe6b4f6a58ddb7b13ed55e2346b9c",
    "license": "Apache-2.0",
    "upstream_pull_request": "https://github.com/huggingface/lerobot/pull/4051",
    "checkpoint_processor_format": "dm05-pr-4051-original",
}
DM05_RUNTIME_MANIFEST_ENV = "NPA_DM05_RUNTIME_MANIFEST"
DM05_RUNTIME_MANIFEST = Path("/opt/lerobot/dm05-runtime.json")
LEROBOT_RELEASE = DM05_IMPLEMENTATION["revision"]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAMERA_MAPPING = {
    "agentview_image": "front",
    "robot0_eye_in_hand_image": "wrist",
}


def build_parser() -> argparse.ArgumentParser:
    """Return the explicit CLI contract used by workflow tool references."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--seed", type=int, default=7)
    prepare.add_argument("--episodes-per-task", type=int, default=5)
    rollout = commands.add_parser("rollout")
    rollout.add_argument(
        "--checkpoint-role", choices=tuple(MODEL_REPOSITORIES), required=True
    )
    rollout.add_argument("--policy-device", default="cuda")
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--baseline-path", required=True)
    metrics.add_argument("--candidate-path", required=True)
    report = commands.add_parser("report")
    report.add_argument("--baseline-path", required=True)
    report.add_argument("--candidate-path", required=True)
    report.add_argument("--metrics-path", required=True)
    report.add_argument("--run-id", required=True)
    for command in (prepare, rollout, metrics, report):
        command.add_argument("--output-path", required=True)
    for command in (rollout, metrics, report):
        command.add_argument("--input-path", required=True)
    return parser


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _protocol(seed: int, episodes_per_task: int) -> dict[str, Any]:
    if seed < 0 or episodes_per_task < 1:
        raise ValueError(
            "seed must be nonnegative and episodes-per-task must be positive"
        )
    return {
        "schema": "npa.dm05_lerobot_libero.protocol.v1",
        "protocol": "published-200-episode-comparison",
        "suites": list(SUITES),
        "tasks_per_suite": 10,
        "episodes_per_task": episodes_per_task,
        "total_episodes": len(SUITES) * 10 * episodes_per_task,
        "seed": seed,
        "camera_name_mapping": CAMERA_MAPPING,
        "observation": {"height": 256, "width": 256, "state_dimension": 8},
        "action": {
            "model_representation": "absolute",
            "model_processor_use_relative_actions": False,
            "environment_controller": "relative",
            "dimension": 7,
            "chunk_size": 10,
            "n_action_steps": 10,
        },
        "checkpoint_lineage": {
            role: {"repository": repo, "revision": revision}
            for role, (repo, revision) in MODEL_REPOSITORIES.items()
        },
        "lerobot_release": LEROBOT_RELEASE,
        "dm05_implementation": DM05_IMPLEMENTATION,
        "published_candidate_result": {
            "successes_by_suite": {
                "libero_spatial": 49,
                "libero_object": 50,
                "libero_goal": 50,
                "libero_10": 48,
            },
            "successes": 197,
            "episodes": 200,
            "is_live_result": False,
            "note": "Upstream model-card claim; never substituted for this run's metrics.",
        },
        "limitations": [
            "This is a 200-episode protocol, not the standard 2,000-episode complete LIBERO evaluation.",
            "Simulation results do not establish physical-robot performance.",
        ],
    }


def prepare(output: Path, *, seed: int, episodes_per_task: int) -> None:
    """Seal the camera/action protocol consumed by both real policy rollouts."""
    protocol = _protocol(seed, episodes_per_task)
    write_json(output / "protocol.json", protocol)
    write_json(output / "provenance.json", {"protocol_sha256": _sha256_json(protocol)})


def _download_checkpoint(role: str, workspace: Path) -> Path:
    """Fetch the named public checkpoint at its exact Hub commit at runtime only."""
    from huggingface_hub import snapshot_download

    repository, revision = MODEL_REPOSITORIES[role]
    checkpoint = Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=str(workspace / role),
            local_dir_use_symlinks=False,
        )
    )
    config = json.loads((checkpoint / "config.json").read_text())
    contract = CHECKPOINT_CONTRACTS[role]
    if (
        config.get("type") != contract["type"]
        or config.get("use_relative_actions") is not contract["use_relative_actions"]
        or config.get("add_state") is not contract["add_state"]
        or config.get("chunk_size") != contract["chunk_size"]
        or config.get("n_action_steps") != contract["n_action_steps"]
        or config.get("input_features", {}).get("observation.state", {}).get("shape")
        != [contract["state_dimension"]]
        or config.get("output_features", {}).get("action", {}).get("shape")
        != [contract["action_dimension"]]
    ):
        raise ValueError(
            f"Downloaded {role} DM05 checkpoint does not match its published policy contract"
        )
    return checkpoint


def _require_libero_checkpoint_contract(role: str, checkpoint: Path) -> None:
    """Reject a published checkpoint whose native representation cannot run LIBERO.

    This is intentionally separate from source identity.  The predecessor is
    real and provenance-verified, but it is not a valid matched LIBERO baseline
    without a documented, released representation conversion.  NPA does not
    manufacture that conversion or relabel an OpenDM checkpoint as LeRobot.
    """

    config = json.loads((checkpoint / "config.json").read_text())
    contract = CHECKPOINT_CONTRACTS[role]
    if (
        contract["state_dimension"] != 8
        or contract["action_dimension"] != 7
        or contract["chunk_size"] != 10
        or contract["n_action_steps"] != 10
    ):
        raise ValueError(
            f"Published {role} checkpoint is not LIBERO-compatible: it has "
            f"state/action dimensions {contract['state_dimension']}/"
            f"{contract['action_dimension']} and chunk/action steps "
            f"{contract['chunk_size']}/{contract['n_action_steps']}; the sealed "
            "LIBERO protocol requires 8/7 and 10/10. No action or state adapter "
            "is defined by the checkpoint release."
        )
    if config.get("use_relative_actions") is not False:
        raise ValueError("LIBERO candidate checkpoint must preserve absolute actions")


def _write_noninteractive_libero_config(workspace: Path) -> Path:
    """Point LIBERO at installed benchmark files without an interactive prompt.

    The upstream LIBERO package creates ``~/.libero/config.yaml`` by asking an
    interactive question on first import. A SkyPilot stage has no stdin, so use
    a run-local config that points to already-installed upstream benchmark
    files. This neither accepts terms nor downloads or redistributes datasets;
    a missing installed closure remains an explicit runtime error.
    """

    spec = importlib.util.find_spec("libero")
    roots = getattr(spec, "submodule_search_locations", None) if spec else None
    if not roots:
        raise RuntimeError("LIBERO package is not installed in the evaluation runtime")
    libero_root = Path(next(iter(roots))) / "libero"
    required = {
        "bddl_files": libero_root / "bddl_files",
        "init_states": libero_root / "init_files",
        "assets": libero_root / "assets",
    }
    missing = [name for name, path in required.items() if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "Installed LIBERO benchmark closure is incomplete; missing "
            + ", ".join(sorted(missing))
        )
    config_root = workspace / "libero-config"
    config_root.mkdir(parents=True)
    datasets = workspace / "libero-datasets"
    datasets.mkdir()
    write_json(
        config_root / "config.yaml",
        {
            "benchmark_root": str(libero_root),
            "bddl_files": str(required["bddl_files"]),
            "init_states": str(required["init_states"]),
            "datasets": str(datasets),
            "assets": str(required["assets"]),
        },
    )
    return config_root


def _read_dm05_runtime_manifest() -> dict[str, Any]:
    """Require the exact source provenance emitted by the private image recipe."""
    manifest_path = Path(
        os.environ.get(DM05_RUNTIME_MANIFEST_ENV, str(DM05_RUNTIME_MANIFEST))
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "DM05 evaluation requires the reviewed exact-policy runtime manifest "
            f"at {manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or manifest != DM05_IMPLEMENTATION:
        raise RuntimeError(
            "DM05 runtime provenance does not match the released-checkpoint "
            "implementation"
        )
    return manifest


def _require_dm05_policy_runtime() -> dict[str, Any]:
    """Verify registration and native class resolution before upstream evaluation.

    ``lerobot-eval`` performs the actual checkpoint deserialization in the same
    stage.  This preflight prevents a registry that merely accepts a string
    called ``dm05`` from silently selecting an unrelated implementation.
    """
    manifest = _read_dm05_runtime_manifest()
    try:
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.dm05.configuration_dm05 import DM05Config
        from lerobot.policies.dm05.modeling_dm05 import DM05Policy
        from lerobot.policies.factory import get_policy_class
    except ImportError as error:
        raise RuntimeError(
            "DM05 evaluation image does not contain the reviewed native policy "
            "implementation"
        ) from error
    try:
        registered = PreTrainedConfig.get_choice_class("dm05")
        policy_class = get_policy_class("dm05")
    except Exception as error:
        raise RuntimeError(
            "DM05 policy type is not registered in this LeRobot runtime"
        ) from error
    if registered is not DM05Config or policy_class is not DM05Policy:
        raise RuntimeError(
            "DM05 policy registration resolves to a class other than the reviewed "
            "hbzfeng/lerobot implementation"
        )
    return {
        "source": manifest,
        "config_class": f"{DM05Config.__module__}.{DM05Config.__name__}",
        "policy_class": f"{DM05Policy.__module__}.{DM05Policy.__name__}",
    }


def rollout_command(
    protocol: dict[str, Any], checkpoint: Path, output: Path, device: str
) -> list[str]:
    """Build the upstream-native evaluator invocation from the sealed protocol."""
    if protocol["action"]["model_processor_use_relative_actions"] is not False:
        raise ValueError("DM05 model actions must remain absolute")
    if protocol["action"]["environment_controller"] != "relative":
        raise ValueError("Published LIBERO evaluation requires a relative controller")
    return [
        "lerobot-eval",
        f"--policy.path={checkpoint}",
        "--env.type=libero",
        f"--env.task={','.join(protocol['suites'])}",
        f"--env.camera_name_mapping={json.dumps(protocol['camera_name_mapping'], sort_keys=True)}",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.control_mode=relative",
        f"--eval.n_episodes={protocol['episodes_per_task']}",
        "--eval.batch_size=1",
        f"--seed={protocol['seed']}",
        f"--policy.device={device}",
        f"--output_dir={output / 'native'}",
    ]


def _validate_eval_info(
    info: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, int]:
    if set(info) - {"per_task", "per_group", "overall"}:
        raise ValueError("Unexpected upstream eval_info fields")
    per_task = info.get("per_task")
    if not isinstance(per_task, list) or len(per_task) != len(SUITES) * 10:
        raise ValueError("Upstream evaluation did not return exactly 40 task records")
    successes: dict[str, int] = {suite: 0 for suite in SUITES}
    expected = protocol["episodes_per_task"]
    seen: set[tuple[str, int]] = set()
    for record in per_task:
        suite, task_id, metrics = (
            record.get("task_group"),
            record.get("task_id"),
            record.get("metrics"),
        )
        if (
            suite not in successes
            or not isinstance(task_id, int)
            or (suite, task_id) in seen
        ):
            raise ValueError(
                "Upstream evaluation task grid is incomplete or duplicated"
            )
        seen.add((suite, task_id))
        values = metrics.get("successes") if isinstance(metrics, dict) else None
        if (
            not isinstance(values, list)
            or len(values) != expected
            or any(not isinstance(v, bool) for v in values)
        ):
            raise ValueError(
                "Upstream evaluation did not return boolean success for every episode"
            )
        successes[suite] += sum(values)
    if seen != {(suite, task_id) for suite in SUITES for task_id in range(10)}:
        raise ValueError(
            "Upstream evaluation task identifiers do not match the complete suite"
        )
    return successes


def run_rollout(protocol_root: Path, output: Path, *, role: str, device: str) -> None:
    """Run one closed-loop checkpoint evaluation and retain upstream evidence verbatim."""
    protocol = json.loads((protocol_root / "protocol.json").read_text())
    protocol_sha256 = file_sha256(protocol_root / "protocol.json")
    with tempfile.TemporaryDirectory(prefix=f"dm05-{role}-") as temporary:
        checkpoint = _download_checkpoint(role, Path(temporary))
        _require_libero_checkpoint_contract(role, checkpoint)
        runtime = _require_dm05_policy_runtime()
        command = rollout_command(protocol, checkpoint, output, device)
        environment = os.environ.copy()
        environment["LIBERO_CONFIG_PATH"] = str(
            _write_noninteractive_libero_config(Path(temporary))
        )
        subprocess.run(command, check=True, env=environment)
    info_path = output / "native" / "eval_info.json"
    if not info_path.is_file():
        raise RuntimeError("lerobot-eval did not write its native eval_info.json")
    info = json.loads(info_path.read_text())
    successes = _validate_eval_info(info, protocol)
    video_paths = sorted(
        path.relative_to(output).as_posix()
        for path in (output / "native").rglob("*.mp4")
    )
    if not video_paths:
        raise RuntimeError("lerobot-eval did not emit a rollout MP4")
    repository, revision = MODEL_REPOSITORIES[role]
    write_json(
        output / "rollout.json",
        {
            "schema": "npa.dm05_lerobot_libero.rollout.v1",
            "checkpoint_role": role,
            "checkpoint": {"repository": repository, "revision": revision},
            "protocol_sha256": protocol_sha256,
            "command": command[:1]
            + [arg for arg in command[1:] if not arg.startswith("--policy.path=")],
            "successes_by_suite": successes,
            "successes": sum(successes.values()),
            "episodes": protocol["total_episodes"],
            "native_eval_info_sha256": file_sha256(info_path),
            "videos": video_paths,
            "model_action_representation": "absolute",
            "environment_controller": "relative",
            "dm05_runtime": runtime,
            "native_checkpoint_load_verified": True,
        },
    )


def calculate_metrics(
    protocol_root: Path, baseline_root: Path, candidate_root: Path, output: Path
) -> None:
    """Calculate matched suite deltas only from two protocol-bound rollout artifacts."""
    protocol_hash = file_sha256(protocol_root / "protocol.json")
    records = {
        role: json.loads((root / "rollout.json").read_text())
        for role, root in (("baseline", baseline_root), ("candidate", candidate_root))
    }
    if any(
        record.get("protocol_sha256") != protocol_hash for record in records.values()
    ):
        raise ValueError("Rollouts do not consume the exact same sealed protocol")
    if any(record.get("checkpoint_role") != role for role, record in records.items()):
        raise ValueError("Rollout checkpoint roles are swapped or invalid")
    episodes = records["baseline"].get("episodes")
    if (
        episodes != records["candidate"].get("episodes")
        or not isinstance(episodes, int)
        or episodes < 1
    ):
        raise ValueError("Matched rollouts have different episode counts")
    suites = []
    for suite in SUITES:
        baseline, candidate = (
            records[role]["successes_by_suite"].get(suite) for role in records
        )
        if not all(
            isinstance(value, int) and 0 <= value <= 50
            for value in (baseline, candidate)
        ):
            raise ValueError("Suite success counts are invalid")
        suites.append(
            {
                "suite": suite,
                "baseline_successes": baseline,
                "candidate_successes": candidate,
                "delta": candidate - baseline,
            }
        )
    write_json(
        output / "metrics.json",
        {
            "schema": "npa.dm05_lerobot_libero.metrics.v1",
            "protocol_sha256": protocol_hash,
            "episodes_per_policy": episodes,
            "suites": suites,
            "baseline_successes": records["baseline"]["successes"],
            "candidate_successes": records["candidate"]["successes"],
            "candidate_minus_baseline": records["candidate"]["successes"]
            - records["baseline"]["successes"],
            "published_197_of_200_reproduced": records["candidate"]["successes"] == 197
            and episodes == 200,
            "full_2000_episode_result": False,
            "physical_robot_tested": False,
        },
    )


def _comparison_mp4(baseline: Path, candidate: Path, output: Path) -> None:
    """Encode a side-by-side factual video from native rollout MP4s."""
    import av

    baseline_frames = list(av.open(str(baseline)).decode(video=0))
    candidate_frames = list(av.open(str(candidate)).decode(video=0))
    if not baseline_frames or not candidate_frames:
        raise ValueError("Native rollout MP4 has no decodable video frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        for left, right in zip(baseline_frames, candidate_frames, strict=False):
            left_pixels, right_pixels = (
                left.to_ndarray(format="rgb24"),
                right.to_ndarray(format="rgb24"),
            )
            height = min(left_pixels.shape[0], right_pixels.shape[0])
            pixels = np.concatenate(
                (left_pixels[:height], right_pixels[:height]), axis=1
            )
            stream.width, stream.height, stream.pix_fmt = (
                pixels.shape[1],
                pixels.shape[0],
                "yuv420p",
            )
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded = list(av.open(str(output)).decode(video=0))
    if not decoded:
        raise RuntimeError("Comparison MP4 is not independently decodable")


def _inspect_recording(path: Path, run_id: str) -> None:
    """Decode the RRD after writing it, rather than trusting file existence."""
    binary = Path(sys.executable).with_name("rerun")
    verified = subprocess.run(
        [str(binary), "rrd", "verify", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    decoded = subprocess.run(
        [str(binary), "rrd", "print", "-vv", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    for expected in ("npa_dm05_lerobot_libero", run_id, "metrics/libero_spatial/delta"):
        if expected not in decoded.stdout:
            raise ValueError(f"Rerun recording is missing {expected!r}")
    path.with_suffix(".inspection.txt").write_text(
        verified.stdout + decoded.stdout, encoding="utf-8"
    )


def emit_report(
    protocol_root: Path,
    baseline_root: Path,
    candidate_root: Path,
    metrics_root: Path,
    output: Path,
    run_id: str,
) -> None:
    """Emit independent MP4/RRD artifacts bound to raw metrics and rollouts."""
    import rerun as rr

    protocol_hash = file_sha256(protocol_root / "protocol.json")
    metrics = json.loads((metrics_root / "metrics.json").read_text())
    if metrics.get("protocol_sha256") != protocol_hash:
        raise ValueError("Metrics provenance does not match the protocol")
    sources = []
    for root in (baseline_root, candidate_root):
        rollout = json.loads((root / "rollout.json").read_text())
        if rollout.get("protocol_sha256") != protocol_hash or not rollout.get("videos"):
            raise ValueError("Report requires protocol-bound native rollout videos")
        sources.append(root / rollout["videos"][0])
    comparison = output / "comparison.mp4"
    _comparison_mp4(sources[0], sources[1], comparison)
    recording = output / "comparison.rrd"
    stream = rr.RecordingStream("npa_dm05_lerobot_libero", recording_id=run_id)
    stream.save(str(recording))
    stream.set_time("protocol", sequence=0)
    for suite in metrics["suites"]:
        stream.log(
            f"metrics/{suite['suite']}/baseline_successes",
            rr.Scalars(suite["baseline_successes"]),
        )
        stream.log(
            f"metrics/{suite['suite']}/candidate_successes",
            rr.Scalars(suite["candidate_successes"]),
        )
        stream.log(f"metrics/{suite['suite']}/delta", rr.Scalars(suite["delta"]))
    stream.log(
        "provenance/protocol_sha256",
        rr.TextDocument(protocol_hash, media_type="text/plain"),
        static=True,
    )
    stream.log(
        "provenance/run_id",
        rr.TextDocument(run_id, media_type="text/plain"),
        static=True,
    )
    stream.flush()
    stream.disconnect()
    if not recording.is_file() or recording.stat().st_size == 0:
        raise RuntimeError("Rerun recording was not written")
    _inspect_recording(recording, run_id)
    write_json(
        output / "report.json",
        {
            "schema": "npa.dm05_lerobot_libero.report.v1",
            "protocol_sha256": protocol_hash,
            "metrics_sha256": file_sha256(metrics_root / "metrics.json"),
            "comparison_mp4": comparison.name,
            "comparison_mp4_sha256": file_sha256(comparison),
            "comparison_rrd": recording.name,
            "comparison_rrd_sha256": file_sha256(recording),
            "reported_result": metrics,
            "physical_robot_tested": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Execute exactly one workflow stage and publish a readback-verified artifact tree."""
    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="dm05-lerobot-libero-") as temporary:
        workspace, output = Path(temporary), Path(temporary) / "output"
        if args.stage == "prepare":
            prepare(output, seed=args.seed, episodes_per_task=args.episodes_per_task)
        else:
            protocol = materialize(args.input_path, workspace / "protocol")
            if args.stage == "rollout":
                run_rollout(
                    protocol,
                    output,
                    role=args.checkpoint_role,
                    device=args.policy_device,
                )
            elif args.stage == "metrics":
                calculate_metrics(
                    protocol,
                    materialize(args.baseline_path, workspace / "baseline"),
                    materialize(args.candidate_path, workspace / "candidate"),
                    output,
                )
            else:
                emit_report(
                    protocol,
                    materialize(args.baseline_path, workspace / "baseline"),
                    materialize(args.candidate_path, workspace / "candidate"),
                    materialize(args.metrics_path, workspace / "metrics"),
                    output,
                    args.run_id,
                )
        publish(output, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
