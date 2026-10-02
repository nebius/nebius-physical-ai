"""Copy only complete LeRobot checkpoints into the local publication spool."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def publish(
    source: Path,
    destination: Path,
    calibration_id: str | None = None,
    base_repo_id: str | None = None,
) -> list[dict]:
    checkpoints = source / "checkpoints"
    last = checkpoints / "last"
    if not last.is_symlink():
        return []
    latest = os.readlink(last)
    if not latest.isdigit():
        raise ValueError(f"unexpected last checkpoint target: {latest}")
    published = []
    for checkpoint in sorted(
        checkpoints.iterdir(),
        key=lambda path: int(path.name) if path.name.isdigit() else -1,
    ):
        if (
            not checkpoint.is_dir()
            or not checkpoint.name.isdigit()
            or int(checkpoint.name) > int(latest)
        ):
            continue
        target = destination / "checkpoints" / checkpoint.name
        marker = target / "COMPLETE.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            if record.get("calibration_id") != calibration_id:
                raise ValueError(
                    f"checkpoint {checkpoint.name} calibration differs from this run"
                )
            published.append(record)
            continue
        for path in checkpoint.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"checkpoint contains symlink: {path}")
            relative = path.relative_to(checkpoint)
            output = target / relative
            if path.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
        adapter = target / "pretrained_model/adapter_model.safetensors"
        config = target / "pretrained_model/adapter_config.json"
        ema_adapter = target / "pretrained_model_ema/adapter_model.safetensors"
        ema_config = target / "pretrained_model_ema/adapter_config.json"
        if not all(
            path.is_file() for path in (adapter, config, ema_adapter, ema_config)
        ):
            raise ValueError(f"incomplete checkpoint {checkpoint.name}")
        if base_repo_id:
            for path in (config, ema_config):
                payload = json.loads(path.read_text())
                payload["base_model_name_or_path"] = base_repo_id
                path.write_text(json.dumps(payload, indent=2) + "\n")
        with adapter.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with ema_adapter.open("rb") as stream:
            ema_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        record = {
            "step": int(checkpoint.name),
            "directory": checkpoint.name,
            "adapter_sha256": digest,
            "ema_adapter_sha256": ema_digest,
        }
        if calibration_id is not None:
            record["calibration_id"] = calibration_id
        marker.write_text(json.dumps(record, indent=2) + "\n")
        published.append(record)
        print(f"published checkpoint {checkpoint.name}", flush=True)
    return published
