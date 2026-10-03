"""Pinned LingBot-VA LIBERO-Long workflow stages.

This adapter intentionally keeps the Robbyant LIBERO-Long data out of images and
does not download it from Hugging Face.  The published dataset is CC BY-NC-SA
4.0, while this adapter cannot infer an operator's use classification.  A
caller supplies an already-authorized, run-scoped object-store prefix instead.

The executable model path is upstream-native: ``wan_va.train`` (Flex Attention)
and ``wan_va_server.VA_Server`` (Torch SDPA).  The narrow configuration overlay
only replaces upstream placeholder paths and disables the example W&B telemetry
configuration; it does not replace training, inference, or LIBERO rollout code.
"""

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
import tempfile
import time
from typing import Any


SOURCE_REPO = "https://github.com/Robbyant/lingbot-va"
SOURCE_REF = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
BASE_MODEL_ID = "robbyant/lingbot-va-base"
BASE_MODEL_REF = "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"
POSTTRAIN_MODEL_ID = "robbyant/lingbot-va-posttrain-libero-long"
POSTTRAIN_MODEL_REF = "0e89d1e753019988aba484e8da2dc0810e264d9f"
DATASET_ID = "robbyant/libero-long-lerobot"
DATASET_REF = "8c0313b1c7cd9fa3798798479cbf59b11af8979d"
DATASET_LICENSE = "CC BY-NC-SA-4.0"
LIBERO_REF = "8f1084e3132a39270c3a13ebe37270a43ece2a01"

TRAINING_ATTENTION = "flex"
INFERENCE_ATTENTION = "torch"
LIBERO_BENCHMARK = "libero_90"
UPSTREAM_TRAIN_STEPS = 5000
UPSTREAM_TRAIN_GPUS = 8

ACTION_CONTRACT: dict[str, Any] = {
    "action_dim": 30,
    "used_action_channel_ids": list(range(7)),
    "action_per_frame": 4,
    "action_norm_method": "quantiles",
    "snr_shift": 5.0,
    "action_snr_shift": 0.05,
    "q01": [
        -0.6589285731315613,
        -0.84375,
        -0.9375,
        -0.12107142806053162,
        -0.15964286029338837,
        -0.26571428775787354,
        -1.0,
    ]
    + [0.0] * 23,
    "q99": [
        0.8999999761581421,
        0.8544642925262451,
        0.9375,
        0.17142857611179352,
        0.1842857152223587,
        0.34392857551574707,
        1.0,
    ]
    + [0.0] * 23,
}
CAMERAS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _local_uri_path(uri: str) -> Path:
    return Path(uri.removeprefix("file://"))


def _read_json(uri: str, destination: Path) -> dict[str, Any]:
    _download_file(uri, destination)
    return json.loads(destination.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _download_file(uri: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_s3(uri):
        from npa.clients.storage import StorageClient

        StorageClient.from_environment().download_file(uri, str(destination))
    else:
        source = _local_uri_path(uri)
        if not source.is_file():
            raise FileNotFoundError(f"Expected file input at {uri}")
        shutil.copy2(source, destination)
    return destination


def _download_tree(uri: str, destination: Path) -> Path:
    if _is_s3(uri):
        from npa.clients.storage import StorageClient

        StorageClient.from_environment().download_directory(uri, str(destination))
    else:
        source = _local_uri_path(uri)
        if not source.is_dir():
            raise FileNotFoundError(f"Expected directory input at {uri}")
        shutil.copytree(source, destination, dirs_exist_ok=False)
    return destination


def _upload_file(path: Path, uri: str) -> str:
    if _is_s3(uri):
        from npa.clients.storage import StorageClient

        return StorageClient.from_environment().upload_file(str(path), uri)
    destination = _local_uri_path(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return f"file://{destination}"


def _upload_tree(path: Path, uri: str) -> str:
    if _is_s3(uri):
        from npa.clients.storage import StorageClient

        return StorageClient.from_environment().upload_directory(str(path), uri)
    destination = _local_uri_path(uri)
    shutil.copytree(path, destination, dirs_exist_ok=False)
    return f"file://{destination}"


def _tree_inventory_digest(root: Path) -> str:
    """Hash a stable filename/size inventory without reading data or weights twice."""
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _assert_action_contract(contract: dict[str, Any]) -> None:
    if contract != ACTION_CONTRACT:
        raise ValueError(
            "LingBot-VA action/SNR/statistics contract does not match va_libero_cfg.py"
        )
    if len(contract["q01"]) != 30 or len(contract["q99"]) != 30:
        raise ValueError("LingBot-VA action quantiles must have all 30 padded channels")


def _episode_records(root: Path) -> list[dict[str, Any]]:
    episodes = root / "meta" / "episodes.jsonl"
    info = root / "meta" / "info.json"
    if not episodes.is_file() or not info.is_file():
        raise ValueError("Expected LeRobot meta/info.json and meta/episodes.jsonl")
    if not (root / "empty_emb.pt").is_file():
        raise ValueError("LingBot-VA training requires dataset-local empty_emb.pt")
    records = [
        json.loads(line)
        for line in episodes.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not records:
        raise ValueError("LeRobot episodes.jsonl is empty")
    for record in records:
        index = record.get("episode_index")
        length = record.get("length")
        configs = record.get("action_config")
        if type(index) is not int or type(length) is not int or length <= 0:
            raise ValueError(
                "Every episode needs integer episode_index and positive length"
            )
        if not isinstance(configs, list) or not configs:
            raise ValueError(f"Episode {index} lacks required action_config")
        for config in configs:
            start, end, text = (
                config.get("start_frame"),
                config.get("end_frame"),
                config.get("action_text"),
            )
            if (
                type(start) is not int
                or type(end) is not int
                or not (0 <= start < end <= length)
            ):
                raise ValueError(
                    f"Episode {index} has invalid action_config frame bounds"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Episode {index} has action_config without action_text"
                )
            for camera in CAMERAS:
                pattern = (
                    f"latents/chunk-*/{camera}/episode_{index:06d}_{start}_{end}.pth"
                )
                if not list(root.glob(pattern)):
                    raise ValueError(
                        f"Episode {index} is missing expected latent {pattern}"
                    )
    return records


def _provenance() -> dict[str, Any]:
    return {
        "lingbot_va": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REF,
            "license": "Apache-2.0",
        },
        "base_checkpoint": {
            "id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REF,
            "license": "Apache-2.0",
        },
        "official_libero_long_checkpoint": {
            "id": POSTTRAIN_MODEL_ID,
            "revision": POSTTRAIN_MODEL_REF,
            "license": "Apache-2.0",
        },
        "libero": {
            "repo": "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "revision": LIBERO_REF,
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REF,
            "license": DATASET_LICENSE,
            "delivery": "operator-staged only; this workflow does not download or package it",
        },
        "attribution": ["Robbyant Team", "Wan-Video", "Mixture-of-Transformers (MoT)"],
    }


def prepare(
    source_uri: str, output_uri: str, *, heldout_fraction: float = 0.2
) -> dict[str, Any]:
    """Validate actual LeRobot/latent inputs and publish a deterministic split manifest."""
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be strictly between zero and one")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-prepare-") as temporary:
        work = Path(temporary)
        source = _download_tree(source_uri, work / "dataset")
        records = _episode_records(source)
        indices = sorted(record["episode_index"] for record in records)
        heldout_count = max(1, round(len(indices) * heldout_fraction))
        heldout = indices[-heldout_count:]
        train = indices[:-heldout_count]
        if not train:
            raise ValueError(
                "Dataset needs at least two episodes for a held-out LIBERO-Long split"
            )
        manifest = {
            "schema": "npa.lingbot_va.prepared.v1",
            "stage": "prepare_lerobot_libero_long",
            "source_dataset_uri": source_uri,
            "source_dataset_inventory_sha256": _tree_inventory_digest(source),
            "episode_count": len(records),
            "train_episode_indices": train,
            "heldout_episode_indices": heldout,
            "attention": {
                "training": TRAINING_ATTENTION,
                "inference": INFERENCE_ATTENTION,
            },
            "action_contract": ACTION_CONTRACT,
            "provenance": _provenance(),
            "data_authorization": "not asserted by NPA; input was operator-staged",
        }
        local_manifest = work / "prepared.json"
        _write_json(local_manifest, manifest)
        manifest["prepared_manifest_uri"] = _upload_file(local_manifest, output_uri)
        return manifest


def _snapshot_checkpoint(destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            POSTTRAIN_MODEL_ID, revision=POSTTRAIN_MODEL_REF, local_dir=str(destination)
        )
    )


def _replace_training_episodes(dataset: Path, train_indices: list[int]) -> None:
    episode_file = dataset / "meta" / "episodes.jsonl"
    records = [
        json.loads(line)
        for line in episode_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [
        record for record in records if record["episode_index"] in set(train_indices)
    ]
    if not selected:
        raise ValueError("Prepared manifest selects no training episodes")
    episode_file.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected),
        encoding="utf-8",
    )


def _native_train(dataset_dir: Path, model_dir: Path, save_root: Path) -> None:
    """Invoke the upstream FSDP trainer with only documented path/telemetry overrides."""
    import torch

    if torch.cuda.device_count() != UPSTREAM_TRAIN_GPUS:
        raise RuntimeError(
            f"Upstream LIBERO recipe requires exactly {UPSTREAM_TRAIN_GPUS} GPUs"
        )
    _enable_upstream()
    from wan_va.configs import VA_CONFIGS
    import wan_va.train as upstream_train

    config = VA_CONFIGS["libero_train"]
    config.dataset_path = str(dataset_dir.parent)
    config.empty_emb_path = str(dataset_dir / "empty_emb.pt")
    config.wan22_pretrained_model_name_or_path = str(model_dir)
    config.resume_from = str(model_dir)
    config.enable_wandb = False
    config.save_root = str(save_root)
    _assert_upstream_config(config, attention=TRAINING_ATTENTION)
    os.environ["WANDB_DISABLED"] = "true"
    upstream_train.run(
        argparse.Namespace(config_name="libero_train", save_root=str(save_root))
    )


def _assert_upstream_config(config: Any, *, attention: str) -> None:
    _assert_action_contract(
        {
            "action_dim": config.action_dim,
            "used_action_channel_ids": list(config.used_action_channel_ids),
            "action_per_frame": config.action_per_frame,
            "action_norm_method": config.action_norm_method,
            "snr_shift": config.snr_shift,
            "action_snr_shift": config.action_snr_shift,
            "q01": list(config.norm_stat["q01"]),
            "q99": list(config.norm_stat["q99"]),
        }
    )
    if attention == TRAINING_ATTENTION and config.num_steps != UPSTREAM_TRAIN_STEPS:
        raise ValueError(
            "The LingBot-VA LIBERO recipe must retain upstream 5000 training steps"
        )


def _enable_upstream() -> None:
    source = os.environ.get("NPA_LINGBOT_VA_SOURCE", "/opt/lingbot-va")
    if source not in sys.path:
        sys.path.insert(0, source)


def train(prepared_uri: str, checkpoint_uri: str, training_uri: str) -> dict[str, Any]:
    """Continue the official checkpoint using the upstream eight-GPU LIBERO recipe."""
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-train-") as temporary:
        work = Path(temporary)
        prepared = _read_json(prepared_uri, work / "prepared.json")
        _assert_action_contract(prepared["action_contract"])
        dataset = _download_tree(prepared["source_dataset_uri"], work / "dataset")
        _episode_records(dataset)
        _replace_training_episodes(dataset, list(prepared["train_episode_indices"]))
        # Upstream LeRobot 0.3.3 resolves repo ids below HF_LEROBOT_HOME and
        # separately resolves latent paths relative to cwd.  The symlink keeps
        # one data copy while satisfying both native path contracts.
        hf_root = work / "hf-home"
        repo_id = "robbyant/libero-long-lerobot"
        repo_root = hf_root / repo_id
        repo_root.parent.mkdir(parents=True)
        repo_root.symlink_to(dataset, target_is_directory=True)
        model = _snapshot_checkpoint(work / "official-posttrain")
        output = work / "output"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(UPSTREAM_TRAIN_GPUS),
            "-m",
            "npa.solutions.lingbot_va",
            "native-train",
            "--dataset-dir",
            str(repo_root),
            "--model-dir",
            str(model),
            "--save-root",
            str(output),
        ]
        environment = dict(
            os.environ,
            HF_LEROBOT_HOME=str(hf_root),
            TOKENIZERS_PARALLELISM="false",
            WANDB_DISABLED="true",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        subprocess.run(command, check=True, cwd=hf_root, env=environment)
        checkpoints = output / "checkpoints"
        if not list(checkpoints.glob("checkpoint_step_*/transformer/config.json")):
            raise RuntimeError(
                "Upstream trainer completed without a usable transformer checkpoint"
            )
        _upload_tree(checkpoints, checkpoint_uri)
        summary = {
            "schema": "npa.lingbot_va.training.v1",
            "stage": "upstream_libero_long_posttrain",
            "prepared_manifest_uri": prepared_uri,
            "checkpoint_uri": checkpoint_uri,
            "training_attention": TRAINING_ATTENTION,
            "world_size": UPSTREAM_TRAIN_GPUS,
            "num_steps": UPSTREAM_TRAIN_STEPS,
            "telemetry": "disabled; no W&B API key, team, project, or consent is configured",
            "provenance": _provenance(),
        }
        local = work / "training.json"
        _write_json(local, summary)
        summary["training_manifest_uri"] = _upload_file(local, training_uri)
        return summary


def _overlay_latest_transformer(model_root: Path, checkpoints: Path) -> Path:
    candidates = sorted(checkpoints.glob("checkpoint_step_*/transformer"))
    if not candidates:
        raise ValueError(
            "Training checkpoint artifact has no upstream transformer checkpoint"
        )
    selected = candidates[-1]
    shutil.copytree(selected, model_root / "transformer", dirs_exist_ok=True)
    return selected


def _native_server(model_dir: Path, output: Path, port: int) -> None:
    _enable_upstream()
    from wan_va.configs import VA_CONFIGS
    import wan_va.wan_va_server as upstream_server

    config = VA_CONFIGS["libero"]
    config.wan22_pretrained_model_name_or_path = str(model_dir)
    config.save_root = str(output)
    config.port = port
    _assert_upstream_config(config, attention=INFERENCE_ATTENTION)
    # The authoritative upstream server itself calls load_transformer(...,
    # attn_mode="torch").  Do not offer Flex Attention as an eval fallback.
    upstream_server.run(
        argparse.Namespace(config_name="libero", port=port, save_root=str(output))
    )


def _wait_for_port(
    port: int, process: subprocess.Popen[str], timeout: float = 180.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"LingBot-VA server stopped before becoming ready ({process.returncode})"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("LingBot-VA websocket server did not become ready")


def rollout(
    prepared_uri: str,
    checkpoint_uri: str,
    output_uri: str,
    rollout_root_uri: str,
    *,
    test_num: int = 50,
    task_start: int = 0,
    task_end: int = 10,
) -> dict[str, Any]:
    """Run the actual upstream websocket policy and LIBERO-90 closed-loop client."""
    if test_num < 1 or task_start < 0 or task_end <= task_start:
        raise ValueError("Invalid LIBERO rollout range")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-rollout-") as temporary:
        work = Path(temporary)
        prepared = _read_json(prepared_uri, work / "prepared.json")
        _assert_action_contract(prepared["action_contract"])
        checkpoints = _download_tree(checkpoint_uri, work / "checkpoints")
        model = _snapshot_checkpoint(work / "official-posttrain")
        selected = _overlay_latest_transformer(model, checkpoints)
        output = work / "rollout"
        output.mkdir()
        port = 23908
        server_command = [
            sys.executable,
            "-m",
            "npa.solutions.lingbot_va",
            "native-server",
            "--model-dir",
            str(model),
            "--output-dir",
            str(output),
            "--port",
            str(port),
        ]
        environment = dict(os.environ, TOKENIZERS_PARALLELISM="false")
        with (output / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                server_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            try:
                _wait_for_port(port, server)
                _enable_upstream()
                from evaluation.libero.client import run as upstream_client_run

                # The upstream CLI omits libero_90 from choices, but its native
                # run function accepts every benchmark registered by LIBERO.
                upstream_client_run(
                    LIBERO_BENCHMARK,
                    port,
                    str(output / "libero"),
                    test_num,
                    [task_start, task_end],
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        result_files = sorted(output.glob(f"libero/{LIBERO_BENCHMARK}_*.json"))
        if len(result_files) != task_end - task_start:
            raise RuntimeError(
                "LIBERO client did not publish a success result for every requested long-horizon task"
            )
        mp4s = sorted(output.glob(f"libero/{LIBERO_BENCHMARK}/**/*.mp4"))
        action_tensors = sorted(output.glob("real/**/actions_*.pt"))
        latent_tensors = sorted(output.glob("real/**/latents_*.pt"))
        if not mp4s or not action_tensors or not latent_tensors:
            raise RuntimeError(
                "Closed-loop rollout lacks real video, action, or predicted-latent artifacts"
            )
        _upload_tree(output, rollout_root_uri)
        manifest = {
            "schema": "npa.lingbot_va.rollout.v1",
            "stage": "libero_90_closed_loop_rollout",
            "prepared_manifest_uri": prepared_uri,
            "checkpoint_uri": checkpoint_uri,
            "checkpoint_transformer": selected.name,
            "rollout_root_uri": rollout_root_uri,
            "benchmark": LIBERO_BENCHMARK,
            "test_episodes_per_task": test_num,
            "task_range": [task_start, task_end],
            "inference_attention": INFERENCE_ATTENTION,
            "result_json_count": len(result_files),
            "rollout_mp4_count": len(mp4s),
            "predicted_action_chunk_count": len(action_tensors),
            "predicted_latent_chunk_count": len(latent_tensors),
            "provenance": _provenance(),
        }
        local = work / "rollout.json"
        _write_json(local, manifest)
        manifest["rollout_manifest_uri"] = _upload_file(local, output_uri)
        return manifest


def evaluate(
    rollout_uri: str, rollout_root_uri: str, output_uri: str
) -> dict[str, Any]:
    """Aggregate actual LIBERO success and action-prediction validity evidence."""
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-evaluate-") as temporary:
        work = Path(temporary)
        rollout_manifest = _read_json(rollout_uri, work / "rollout.json")
        if rollout_manifest.get("benchmark") != LIBERO_BENCHMARK:
            raise ValueError("Evaluation refuses a rollout from another benchmark")
        rollout = _download_tree(rollout_root_uri, work / "rollout")
        rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(rollout.glob(f"libero/{LIBERO_BENCHMARK}_*.json"))
        ]
        if not rows:
            raise RuntimeError(
                "No upstream LIBERO long-horizon success JSON artifacts found"
            )
        success_count = sum(float(row["succ_num"]) for row in rows)
        trial_count = sum(float(row["total_num"]) for row in rows)
        if trial_count <= 0:
            raise RuntimeError("LIBERO success artifacts reported no episodes")
        import torch

        action_tensors = sorted(rollout.glob("real/**/actions_*.pt"))
        if not action_tensors:
            raise RuntimeError("No LingBot-VA action-prediction tensors found")
        values = [
            torch.load(path, map_location="cpu", weights_only=True).float()[:, :7]
            for path in action_tensors
        ]
        actions = torch.cat([value.flatten() for value in values])
        finite = torch.isfinite(actions)
        finite_values = actions[finite]
        if not len(finite_values):
            raise RuntimeError("All predicted action values are non-finite")
        # This is a declared validity metric of emitted normalized predictions,
        # not an accuracy claim against unavailable aligned ground truth.
        prediction_validity = {
            "scope": "normalized predicted-action validity; not ground-truth action accuracy or calibrated video quality",
            "finite_fraction": float(finite.float().mean().item()),
            "within_normalized_range_fraction": float(
                ((finite_values >= -1.0) & (finite_values <= 1.0)).float().mean().item()
            ),
            "mean_absolute_normalized_action": float(finite_values.abs().mean().item()),
            "predicted_action_chunk_count": len(action_tensors),
        }
        metrics = {
            "schema": "npa.lingbot_va.evaluation.v1",
            "stage": "libero_90_long_horizon_evaluation",
            "rollout_manifest_uri": rollout_uri,
            "benchmark": LIBERO_BENCHMARK,
            "task_count": len(rows),
            "success_count": success_count,
            "trial_count": trial_count,
            "long_horizon_success_rate": success_count / trial_count,
            "action_prediction_validity": prediction_validity,
            "prediction_quality_limit": "No aligned held-out action/video target is fabricated. Add an explicitly aligned comparison artifact before making an accuracy or visual-quality claim.",
            "provenance": _provenance(),
        }
        local = work / "evaluation.json"
        _write_json(local, metrics)
        metrics["evaluation_uri"] = _upload_file(local, output_uri)
        return metrics


def visualize(
    rollout_uri: str,
    rollout_root_uri: str,
    evaluation_uri: str,
    rrd_uri: str,
    mp4_uri: str,
    output_uri: str,
) -> dict[str, Any]:
    """Emit a workflow-bound Rerun recording and MP4 from a completed rollout.

    Args:
        rollout_uri: Run-scoped URI of the native closed-loop rollout manifest.
        evaluation_uri: Run-scoped URI of the numerical evaluation manifest.
        output_uri: Run-scoped URI where visualization artifacts are written.

    Returns:
        Manifest identifying the generated Rerun recording, MP4, and provenance.

    Raises:
        RuntimeError: If the workflow did not supply its scoped run identifier.
    """
    run_id = os.environ.get("NPA_WORKFLOW_RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError("Visualization requires the workflow-scoped NPA_WORKFLOW_RUN_ID")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-viz-") as temporary:
        work = Path(temporary)
        rollout_manifest = _read_json(rollout_uri, work / "rollout.json")
        metrics = _read_json(evaluation_uri, work / "evaluation.json")
        rollout = _download_tree(rollout_root_uri, work / "rollout")
        videos = sorted(rollout.glob(f"libero/{LIBERO_BENCHMARK}/**/*.mp4"))
        if not videos:
            raise RuntimeError("No real LIBERO rollout MP4 available for visualization")
        preview = work / "libero-rollout-preview.mp4"
        shutil.copy2(videos[0], preview)
        import imageio.v3 as iio
        import rerun as rr

        rrd = work / "libero-long.rrd"
        rr.init("npa.lingbot_va", recording_id=run_id, spawn=False)
        rr.save(str(rrd))
        rr.log(
            "provenance/run",
            rr.TextLog(
                json.dumps(
                    {
                        "run_id": run_id,
                        "producer": "npa.solutions.lingbot_va",
                        "provenance": _provenance(),
                        "limitation": (
                            "Observed rollout camera frames and model-emitted action "
                            "validity are recorded; this run does not claim aligned "
                            "ground-truth video or action prediction accuracy."
                        ),
                    },
                    sort_keys=True,
                )
            ),
        )
        frame_count = 0
        for frame in iio.imiter(preview, plugin="pyav"):
            rr.set_time_sequence("rollout_frame", frame_count)
            rr.log("rollout/cameras", rr.Image(frame))
            frame_count += 1
        rr.log(
            "metrics/long_horizon_success_rate",
            rr.Scalar(metrics["long_horizon_success_rate"]),
        )
        rr.log(
            "metrics/action_finite_fraction",
            rr.Scalar(metrics["action_prediction_validity"]["finite_fraction"]),
        )
        rr.disconnect()
        if not rrd.is_file() or rrd.stat().st_size == 0 or frame_count == 0:
            raise RuntimeError("Rerun recording or copied real MP4 is empty")
        manifest = {
            "schema": "npa.lingbot_va.visualization.v1",
            "stage": "synchronized_rrd_mp4_provenance",
            "rollout_manifest_uri": rollout_uri,
            "evaluation_uri": evaluation_uri,
            "rrd_uri": _upload_file(rrd, rrd_uri),
            "mp4_uri": _upload_file(preview, mp4_uri),
            "source_rollout_mp4": str(videos[0].relative_to(rollout)),
            "decoded_frame_count": frame_count,
            "rollout_benchmark": rollout_manifest["benchmark"],
            "recording_id": run_id,
            "provenance": _provenance(),
        }
        local = work / "visualization.json"
        _write_json(local, manifest)
        manifest["visualization_manifest_uri"] = _upload_file(local, output_uri)
        return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPA LingBot-VA LIBERO-Long stages")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("prepare")
    command.add_argument("--source-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command.add_argument("--heldout-fraction", type=float, default=0.2)
    command = subcommands.add_parser("train")
    command.add_argument("--prepared-uri", required=True)
    command.add_argument("--checkpoint-uri", required=True)
    command.add_argument("--training-uri", required=True)
    command = subcommands.add_parser("rollout")
    command.add_argument("--prepared-uri", required=True)
    command.add_argument("--checkpoint-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command.add_argument("--rollout-root-uri", required=True)
    command.add_argument("--test-num", type=int, default=50)
    command.add_argument("--task-start", type=int, default=0)
    command.add_argument("--task-end", type=int, default=10)
    command = subcommands.add_parser("evaluate")
    command.add_argument("--rollout-uri", required=True)
    command.add_argument("--rollout-root-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command = subcommands.add_parser("visualize")
    command.add_argument("--rollout-uri", required=True)
    command.add_argument("--rollout-root-uri", required=True)
    command.add_argument("--evaluation-uri", required=True)
    command.add_argument("--rrd-uri", required=True)
    command.add_argument("--mp4-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command = subcommands.add_parser("native-train")
    command.add_argument("--dataset-dir", type=Path, required=True)
    command.add_argument("--model-dir", type=Path, required=True)
    command.add_argument("--save-root", type=Path, required=True)
    command = subcommands.add_parser("native-server")
    command.add_argument("--model-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--port", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.source_uri, args.output_uri, heldout_fraction=args.heldout_fraction
        )
    elif args.command == "train":
        result = train(args.prepared_uri, args.checkpoint_uri, args.training_uri)
    elif args.command == "rollout":
        result = rollout(
            args.prepared_uri,
            args.checkpoint_uri,
            args.output_uri,
            args.rollout_root_uri,
            test_num=args.test_num,
            task_start=args.task_start,
            task_end=args.task_end,
        )
    elif args.command == "evaluate":
        result = evaluate(args.rollout_uri, args.rollout_root_uri, args.output_uri)
    elif args.command == "visualize":
        result = visualize(
            args.rollout_uri,
            args.rollout_root_uri,
            args.evaluation_uri,
            args.rrd_uri,
            args.mp4_uri,
            args.output_uri,
        )
    elif args.command == "native-train":
        _native_train(args.dataset_dir, args.model_dir, args.save_root)
        return
    elif args.command == "native-server":
        _native_server(args.model_dir, args.output_dir, args.port)
        return
    else:  # pragma: no cover - argparse already constrains this
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
