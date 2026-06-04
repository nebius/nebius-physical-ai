"""SkyPilot workflow helpers for Cosmos augmentation and reasoning."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_COSMOS3_NANO_MODEL_ID = "nvidia/Cosmos3-Nano"
DEFAULT_COSMOS3_SUPER_MODEL_ID = "nvidia/Cosmos3-Super"
DEFAULT_COSMOS3_SOURCE_REPO = "https://github.com/NVIDIA/cosmos-framework.git"
DEFAULT_TRANSFER25_MODEL_ID = "nvidia/Cosmos-Transfer2.5-2B"
DEFAULT_TRANSFER25_SOURCE_REPO = "https://github.com/nvidia-cosmos/cosmos-transfer2.5.git"
DEFAULT_COSMOS_IMAGE = "example.invalid/npa-cosmos:3.0.0"
DEFAULT_S3_ENDPOINT = ""
COSMOS_ATTRIBUTION = "Built on NVIDIA Cosmos"

REPO_ROOT = Path(__file__).resolve().parents[5]
SKYPILOT_ROOT = REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot"
COSMOS_AUGMENT_YAML = SKYPILOT_ROOT / "cosmos3-augment.yaml"
COSMOS_REASON_YAML = SKYPILOT_ROOT / "cosmos3-reason.yaml"

CONTROL_ALIASES = {
    "blur": "vis",
    "visual": "vis",
    "vis": "vis",
    "rgb": "vis",
    "edge": "edge",
    "depth": "depth",
    "seg": "seg",
    "segmentation": "seg",
}

COSMOS3_MODEL_SIZE_TO_CHECKPOINT = {
    "nano": "Cosmos3-Nano",
    "8b": "Cosmos3-Nano",
    "16b": "Cosmos3-Nano",
    "super": "Cosmos3-Super",
    "32b": "Cosmos3-Super",
    "64b": "Cosmos3-Super",
}


@dataclass(frozen=True)
class CosmosSkyLaunchResult:
    """Result of a Cosmos SkyPilot launch request."""

    status: str
    command: tuple[str, ...]
    env: dict[str, str]
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "command": list(self.command),
            "env": dict(self.env),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "attribution": COSMOS_ATTRIBUTION,
        }


def normalize_control_modality(value: str) -> str:
    """Return the Transfer 2.5 control key for a customer-facing modality."""

    key = value.strip().lower()
    try:
        return CONTROL_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(CONTROL_ALIASES))
        raise ValueError(f"Unsupported control modality {value!r}; expected one of: {valid}") from exc


def cosmos3_checkpoint_for_size(model_size: str) -> str:
    """Return the Cosmos 3 framework checkpoint name for a model-size selector."""

    key = model_size.strip().lower()
    try:
        return COSMOS3_MODEL_SIZE_TO_CHECKPOINT[key]
    except KeyError as exc:
        valid = ", ".join(sorted(COSMOS3_MODEL_SIZE_TO_CHECKPOINT))
        raise ValueError(f"Unsupported Cosmos 3 model size {model_size!r}; expected one of: {valid}") from exc


def build_cosmos_augment_env(
    *,
    source: str,
    output_path: str,
    prompt: str,
    control: str = "edge",
    control_config: str = "",
    model_size: str = "transfer2.5-2b",
    variants: int = 1,
    replicas: int = 1,
    image: str = "",
    s3_endpoint: str = DEFAULT_S3_ENDPOINT,
    source_repo_url: str = DEFAULT_TRANSFER25_SOURCE_REPO,
    hf_model_id: str = DEFAULT_TRANSFER25_MODEL_ID,
    hf_token_env: str = "HF_TOKEN",
    aws_profile: str = "",
) -> dict[str, str]:
    """Build env vars consumed by the raw Cosmos augmentation SkyPilot YAML."""

    if variants < 1:
        raise ValueError("variants must be >= 1")
    if replicas < 1:
        raise ValueError("replicas must be >= 1")
    return {
        "NPA_COSMOS_IMAGE": image or DEFAULT_COSMOS_IMAGE,
        "NPA_COSMOS_AUGMENT_SOURCE": source,
        "NPA_COSMOS_AUGMENT_OUTPUT": output_path,
        "NPA_COSMOS_AUGMENT_PROMPT": prompt,
        "NPA_COSMOS_AUGMENT_CONTROL": normalize_control_modality(control),
        "NPA_COSMOS_AUGMENT_CONTROL_CONFIG": control_config,
        "NPA_COSMOS_AUGMENT_MODEL_SIZE": model_size,
        "NPA_COSMOS_AUGMENT_VARIANTS": str(variants),
        "NPA_COSMOS_REPLICAS": str(replicas),
        "NPA_COSMOS_TRANSFER_SOURCE_REPO": source_repo_url,
        "NPA_COSMOS_TRANSFER_MODEL_ID": hf_model_id,
        "NPA_COSMOS_HF_TOKEN_ENV": hf_token_env,
        "AWS_ENDPOINT_URL": s3_endpoint,
        "AWS_PROFILE": aws_profile,
        "NPA_COSMOS_ATTRIBUTION": COSMOS_ATTRIBUTION,
    }


def build_cosmos_reason_env(
    *,
    input_path: str,
    output_path: str,
    criteria_prompt: str,
    model_size: str = "nano",
    replicas: int = 1,
    image: str = "",
    s3_endpoint: str = DEFAULT_S3_ENDPOINT,
    source_repo_url: str = DEFAULT_COSMOS3_SOURCE_REPO,
    hf_token_env: str = "HF_TOKEN",
    aws_profile: str = "",
) -> dict[str, str]:
    """Build env vars consumed by the raw Cosmos reasoning SkyPilot YAML."""

    if replicas < 1:
        raise ValueError("replicas must be >= 1")
    checkpoint = cosmos3_checkpoint_for_size(model_size)
    model_id = (
        DEFAULT_COSMOS3_SUPER_MODEL_ID
        if checkpoint == "Cosmos3-Super"
        else DEFAULT_COSMOS3_NANO_MODEL_ID
    )
    return {
        "NPA_COSMOS_IMAGE": image or DEFAULT_COSMOS_IMAGE,
        "NPA_COSMOS_REASON_INPUT": input_path,
        "NPA_COSMOS_REASON_OUTPUT": output_path,
        "NPA_COSMOS_REASON_CRITERIA": criteria_prompt,
        "NPA_COSMOS_REASON_MODEL_SIZE": model_size,
        "NPA_COSMOS_REASON_CHECKPOINT": checkpoint,
        "NPA_COSMOS_REASON_MODEL_ID": model_id,
        "NPA_COSMOS_REPLICAS": str(replicas),
        "NPA_COSMOS3_SOURCE_REPO": source_repo_url,
        "NPA_COSMOS_HF_TOKEN_ENV": hf_token_env,
        "AWS_ENDPOINT_URL": s3_endpoint,
        "AWS_PROFILE": aws_profile,
        "NPA_COSMOS_ATTRIBUTION": COSMOS_ATTRIBUTION,
    }


def launch_cosmos_sky_workflow(
    *,
    yaml_path: Path,
    env: Mapping[str, str],
    cluster: str = "",
    name: str = "",
    infra: str = "kubernetes",
    accelerator: str = "",
    num_nodes: int = 0,
    workdir: str = "",
    skypilot_bin: str = "",
    secrets: Sequence[str] = ("HF_TOKEN",),
    dry_run: bool = False,
    timeout: float | None = None,
) -> CosmosSkyLaunchResult:
    """Launch or render a Cosmos SkyPilot workflow."""

    sky_bin = skypilot_bin or os.environ.get("NPA_SKYPILOT_BIN", "") or shutil.which("sky") or "sky"
    command = [sky_bin, "launch", "--yes", "--infra", infra]
    if cluster:
        command.extend(["--cluster", cluster])
    if name:
        command.extend(["--name", name])
    if workdir:
        command.extend(["--workdir", workdir])
    if accelerator:
        command.extend(["--gpus", accelerator])
    if num_nodes > 0:
        command.extend(["--num-nodes", str(num_nodes)])
    for key, value in sorted(env.items()):
        if value != "":
            command.extend(["--env", f"{key}={value}"])
    for secret in secrets:
        if secret and os.environ.get(secret):
            command.extend(["--secret", secret])
    command.append(str(yaml_path))

    if dry_run:
        return CosmosSkyLaunchResult(
            status="dry_run",
            command=tuple(command),
            env=dict(env),
        )

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    status = "submitted" if result.returncode == 0 else "failed"
    return CosmosSkyLaunchResult(
        status=status,
        command=tuple(command),
        env=dict(env),
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def shell_join(command: Sequence[str]) -> str:
    """Return a shell-escaped command string for display."""

    return " ".join(shlex.quote(part) for part in command)
