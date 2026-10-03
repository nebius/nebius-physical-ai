"""Pinned FLUX 3 Action SO-101 task LoRA on the public PickOrange demonstrations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from npa.clients.storage import StorageClient

from .calibration import ID as CALIBRATION_ID, check_model_overlap
from .prepare_dataset import prepare
from .publish import publish
from .resume import restore

RECIPE = json.loads(Path(__file__).with_name("recipe.json").read_text())
LOSS = re.compile(r"\bloss[:=]\s*([^\s,]+)")


def training_command(
    work: Path, policy: Path, encoders: Path, steps: int, checkpoint: Path | None = None
) -> list[str]:
    source = [
        "--config_path=/opt/lerobot/examples/flux3/lora.json",
        f"--policy.path={policy}",
    ]
    if checkpoint is not None:
        source = [
            f"--config_path={checkpoint / 'pretrained_model/train_config.json'}",
            "--resume=true",
        ]
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        *source,
        "--policy.device=cuda",
        f"--policy.video_vae_id={encoders / 'video_vae.safetensors'}",
        f"--policy.text_encoder_id={encoders / 'text_encoder'}",
        f"--dataset.repo_id={RECIPE['dataset']['repo_id']}",
        f"--dataset.root={work / 'dataset'}",
        f"--dataset.revision={RECIPE['dataset']['revision']}",
        "--dataset.video_backend=pyav",
        '--rename_map={"observation.images.front":"observation.images.scene"}',
        f"--steps={steps}",
        f"--save_freq={5000 if steps >= 10000 else steps}",
        f"--log_freq={min(steps, 20)}",
        "--eval_steps=0",
        f"--output_dir={work / 'trainer'}",
    ]


def _download_models(work: Path) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download
    from lerobot.policies.flux3.configuration_flux3 import Flux3Config

    policy = work / "models/so101"
    encoders = work / "models/base"
    snapshot_download(
        RECIPE["policy"]["repo_id"],
        revision=RECIPE["policy"]["revision"],
        local_dir=policy,
        ignore_patterns=["variants/*"],
    )
    snapshot_download(
        RECIPE["encoders"]["repo_id"],
        revision=RECIPE["encoders"]["revision"],
        local_dir=encoders,
        allow_patterns=["video_vae.safetensors", "text_encoder/*"],
    )
    config = Flux3Config.from_pretrained(policy)
    if (config.action_dim, config.fps, config.camera_order) != (
        6,
        30,
        ["observation.images.scene", "observation.images.wrist"],
    ):
        raise ValueError("downloaded checkpoint does not match the SO-101 contract")
    for file in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (policy / file).is_file():
            raise FileNotFoundError(f"SO-101 {file} is missing")
    if not (encoders / "video_vae.safetensors").is_file():
        raise FileNotFoundError("shared video VAE is missing")
    return policy, encoders


def _losses(log_path: Path) -> dict[str, float | int]:
    values = [
        float(value) for value in LOSS.findall(log_path.read_text(errors="replace"))
    ]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("trainer did not report finite losses")
    return {"count": len(values), "first": values[0], "last": values[-1]}


def _verify_adapter(
    checkpoint: Path, base: Path, encoders: Path, dataset_root: Path
) -> list[float]:
    """Reload the exported raw adapter and run one finite action through saved processors."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.flux3.configuration_flux3 import Flux3Config
    from lerobot.policies.flux3.modeling_flux3 import Flux3Policy
    from peft import PeftConfig, PeftModel

    config = Flux3Config.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    config.device = "cuda"
    config.video_vae_id = str(encoders / "video_vae.safetensors")
    config.text_encoder_id = str(encoders / "text_encoder")
    config.use_peft = True
    peft_config = PeftConfig.from_pretrained(str(checkpoint))
    if peft_config.base_model_name_or_path != RECIPE["policy"]["repo_id"]:
        raise ValueError("adapter does not reference the pinned SO-101 base")
    policy = Flux3Policy.from_pretrained(base, config=config)
    policy = (
        PeftModel.from_pretrained(
            policy, str(checkpoint), config=peft_config, is_trainable=False
        )
        .to("cuda")
        .eval()
    )
    pre, post = make_pre_post_processors(config, pretrained_path=checkpoint)
    dataset = LeRobotDataset(
        RECIPE["dataset"]["repo_id"], root=dataset_root, video_backend="pyav"
    )
    frame = dataset[0]
    observation = {
        "observation.images.scene": frame["observation.images.front"].unsqueeze(0),
        "observation.images.wrist": frame["observation.images.wrist"].unsqueeze(0),
        "observation.state": frame["observation.state"].unsqueeze(0),
        "task": [frame["task"]],
    }
    with torch.inference_mode():
        action = post(policy.select_action(pre(observation))).detach().cpu().reshape(-1)
    if action.numel() != 6 or not torch.isfinite(action).all():
        raise ValueError("reloaded adapter produced a non-finite SO-101 action")
    return [float(value) for value in action]


def _upload_checkpoint(
    storage: StorageClient, result: Path, record: dict, output_path: str
) -> None:
    """Publish a complete checkpoint marker only after both adapter readbacks."""
    target = result / "checkpoints" / record["directory"]
    marker = target / "COMPLETE.json"
    for path in sorted(target.rglob("*")):
        if path.is_file() and path != marker:
            uri = output_path + path.relative_to(result).as_posix()
            storage.upload_file(str(path), uri)
    for variant, digest_key in (
        ("pretrained_model", "adapter_sha256"),
        ("pretrained_model_ema", "ema_adapter_sha256"),
    ):
        remote = (
            output_path
            + f"checkpoints/{record['directory']}/{variant}/adapter_model.safetensors"
        )
        readback = storage.read_bytes_with_etag(remote)
        if (
            readback is None
            or hashlib.sha256(readback[0]).hexdigest() != record[digest_key]
        ):
            raise RuntimeError(f"uploaded {variant} adapter failed SHA-256 readback")
    storage.upload_file(
        str(marker), output_path + marker.relative_to(result).as_posix()
    )


def _upload_diagnostics(storage: StorageClient, result: Path, output_path: str) -> None:
    for name in ("train.log", "command.json", "calibration-report.json"):
        path = result / name
        if path.is_file():
            storage.upload_file(str(path), output_path + name)


def run(*, output_path: str, run_id: str, steps: int, input_path: str = "") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id must contain letters, digits, dash, or underscore")
    if steps < 4 or steps % 4:
        raise ValueError(
            "steps must be a positive multiple of the preset's four-step accumulation"
        )
    parsed = urlparse(output_path)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("output_path must be a run-scoped s3:// URI")
    output_path = output_path.rstrip("/") + "/"
    storage = StorageClient.from_environment()
    prefix = parsed.path.strip("/") + "/"
    existing = storage.s3.list_objects_v2(
        Bucket=parsed.netloc, Prefix=prefix, MaxKeys=1
    )
    if existing.get("Contents"):
        raise FileExistsError(f"run prefix already contains objects: {output_path}")

    with tempfile.TemporaryDirectory(prefix="npa-flux3-") as temp:
        work = Path(temp)
        result = work / "result"
        result.mkdir()
        dataset = prepare(
            work / "dataset", RECIPE["dataset"], result, RECIPE["instruction"]
        )
        policy, encoders = _download_models(work)
        calibration = check_model_overlap(
            work / "dataset", policy, result / "calibration-report.json"
        )
        if calibration["frames"] != RECIPE["dataset"]["frames"]:
            raise ValueError("calibrated dataset frame count mismatch")
        checkpoint = work / "resume" if input_path else None
        resumed = (
            restore(storage, input_path, checkpoint, policy, RECIPE, steps)
            if checkpoint
            else None
        )
        command = training_command(work, policy, encoders, steps, checkpoint)
        (result / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        _upload_diagnostics(storage, result, output_path)
        log_path = result / "train.log"
        uploaded: set[str] = set()
        failures: list[Exception] = []

        def upload_ready() -> list[dict]:
            records = publish(
                work / "trainer", result, CALIBRATION_ID, RECIPE["policy"]["repo_id"]
            )
            for item in records:
                if item["directory"] not in uploaded:
                    _upload_checkpoint(storage, result, item, output_path)
                    uploaded.add(item["directory"])
                    _upload_diagnostics(storage, result, output_path)
            return records

        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            stop = threading.Event()

            def watch() -> None:
                while not stop.wait(30):
                    try:
                        upload_ready()
                    except Exception as error:
                        failures.append(error)
                        process.terminate()
                        return

            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                    log.flush()
                status = process.wait()
            finally:
                stop.set()
                watcher.join()
                _upload_diagnostics(storage, result, output_path)
        if failures:
            raise RuntimeError(
                f"checkpoint publication failed: {failures[0]}"
            ) from failures[0]
        published = upload_ready()
        if status:
            raise RuntimeError(f"LeRobot trainer exited {status}")
        losses = _losses(log_path)
        if not published or published[-1]["step"] != steps:
            raise RuntimeError(
                f"trainer exited without complete step-{steps} checkpoint"
            )
        final = result / "checkpoints" / published[-1]["directory"]
        for variant in ("pretrained_model", "pretrained_model_ema"):
            if not (final / variant / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(f"missing {variant} adapter")
        action = _verify_adapter(
            final / "pretrained_model", policy, encoders, work / "dataset"
        )
        record = {
            "schema": "npa.flux3-so101.training.v1",
            "status": "trained",
            "run_id": run_id,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "recipe": RECIPE,
            "calibration_id": CALIBRATION_ID,
            "microsteps": steps,
            "resume": resumed,
            "optimizer_updates": steps // 4,
            "dataset": dataset,
            "losses": losses,
            "adapter_reload_action": action,
            "checkpoints": published,
            "selected_checkpoint": f"checkpoints/{published[-1]['directory']}/pretrained_model",
            "quality_evaluation": "pending",
        }
        (result / "status.json").write_text(json.dumps(record, indent=2) + "\n")
        for path in sorted(result.iterdir()):
            if path.is_file():
                storage.upload_file(str(path), output_path + path.name)
        marker = result / "COMPLETE.json"
        marker.write_text(json.dumps(record, indent=2) + "\n")
        storage.upload_file(str(marker), output_path + "COMPLETE.json")
        echoed = storage.read_bytes_with_etag(output_path + "COMPLETE.json")
        if echoed is None or echoed[0] != marker.read_bytes():
            raise RuntimeError("completion marker failed Object Storage readback")
        return record
