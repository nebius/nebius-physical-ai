"""SkyPilot workflow helpers for Cosmos augmentation and reasoning."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_COSMOS3_NANO_MODEL_ID = "nvidia/Cosmos3-Nano"
DEFAULT_COSMOS3_SUPER_MODEL_ID = "nvidia/Cosmos3-Super"
DEFAULT_COSMOS3_SOURCE_REPO = "https://github.com/NVIDIA/cosmos-framework.git"
DEFAULT_TRANSFER25_MODEL_ID = "nvidia/Cosmos-Transfer2.5-2B"
DEFAULT_TRANSFER25_SOURCE_REPO = "https://github.com/nvidia-cosmos/cosmos-transfer2.5.git"
DEFAULT_COSMOS_IMAGE = "example.invalid/npa-cosmos:3.0.0"
DEFAULT_COSMOS2_TRANSFER_IMAGE = "example.invalid/npa-cosmos2-transfer:2.5.0"
DEFAULT_S3_ENDPOINT = ""
COSMOS_ATTRIBUTION = "Built on NVIDIA Cosmos"
SKYPILOT_DOCKER_LOGIN_ENV = {
    "SKYPILOT_DOCKER_USERNAME": "NPA_DOCKER_LOGIN_USERNAME",
    "SKYPILOT_DOCKER_PASSWORD": "NPA_DOCKER_LOGIN_PASSWORD",
    "SKYPILOT_DOCKER_SERVER": "NPA_DOCKER_LOGIN_SERVER",
}

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
    rendered_yaml: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "command": list(self.command),
            "env": dict(self.env),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "rendered_yaml": self.rendered_yaml,
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
        "NPA_COSMOS2_TRANSFER_IMAGE": image or DEFAULT_COSMOS2_TRANSFER_IMAGE,
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
        "NPA_COSMOS_TRANSFER_CUDA_EXTRA": "cu128",
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
    rendered_yaml = _render_materialized_workflow(
        yaml_path,
        env=env,
        infra=infra,
        accelerator=accelerator,
    )
    keep_rendered_yaml = dry_run
    command = [sky_bin, "launch", "--yes", "--infra", infra]
    docker_secret_names, docker_env_updates = _prepare_docker_login_env()
    previous_docker_env = {key: os.environ.get(key) for key in docker_env_updates}
    os.environ.update(docker_env_updates)
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
    for secret in secrets:
        if secret and os.environ.get(secret):
            command.extend(["--secret", secret])
    for secret in docker_secret_names:
        command.extend(["--secret", secret])
    command.append(str(rendered_yaml))

    if dry_run:
        try:
            return CosmosSkyLaunchResult(
                status="dry_run",
                command=tuple(command),
                env=dict(env),
                rendered_yaml=str(rendered_yaml),
            )
        finally:
            _restore_env(previous_docker_env)

    try:
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
            rendered_yaml=str(rendered_yaml),
        )
    finally:
        _restore_env(previous_docker_env)
        if not keep_rendered_yaml:
            rendered_yaml.unlink(missing_ok=True)


def shell_join(command: Sequence[str]) -> str:
    """Return a shell-escaped command string for display."""

    return " ".join(shlex.quote(part) for part in command)


def _render_materialized_workflow(
    yaml_path: Path,
    *,
    env: Mapping[str, str],
    infra: str,
    accelerator: str,
) -> Path:
    docs = [doc for doc in yaml.safe_load_all(yaml_path.read_text(encoding="utf-8")) if doc]
    if not docs:
        raise ValueError(f"workflow YAML has no documents: {yaml_path}")
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        envs = doc.setdefault("envs", {})
        if isinstance(envs, dict):
            for key, value in sorted(env.items()):
                if value != "":
                    envs[key] = value
        resources = doc.setdefault("resources", {})
        if isinstance(resources, dict):
            resources.pop("image_id", None)
            if infra == "nebius":
                resources["cloud"] = "nebius"
                resources.setdefault("region", "eu-north1")
            if accelerator:
                resources["accelerators"] = accelerator
        _assert_no_unrendered_submit_values(doc)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="npa-cosmos-",
        suffix=".yaml",
        delete=False,
    )
    with handle:
        yaml.safe_dump_all(docs, handle, sort_keys=False)
    return Path(handle.name)


def _assert_no_unrendered_submit_values(doc: Mapping[str, Any]) -> None:
    resources = doc.get("resources", {})
    if isinstance(resources, Mapping):
        image_id = resources.get("image_id")
        if isinstance(image_id, str) and "${" in image_id:
            raise ValueError("workflow resources.image_id contains an unrendered variable")

    envs = doc.get("envs", {})
    if not isinstance(envs, Mapping):
        return
    for key, value in envs.items():
        if not isinstance(value, str):
            continue
        if "${" in value:
            raise ValueError(f"workflow env {key} contains an unrendered variable")
        if value.startswith("s3://") and "${" in value:
            raise ValueError(f"workflow S3 env {key} contains an unrendered variable")


def _prepare_docker_login_env() -> tuple[tuple[str, ...], dict[str, str]]:
    sky_values = {
        sky_key: os.environ.get(sky_key, "")
        for sky_key in SKYPILOT_DOCKER_LOGIN_ENV
    }
    npa_values = {
        sky_key: os.environ.get(npa_key, "")
        for sky_key, npa_key in SKYPILOT_DOCKER_LOGIN_ENV.items()
    }
    sky_present = any(sky_values.values())
    npa_present = any(npa_values.values())
    if not sky_present and not npa_present:
        return (), {}

    values = sky_values if sky_present else npa_values
    missing = [key for key, value in values.items() if not value]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"incomplete Docker registry login configuration: {names}")
    return tuple(SKYPILOT_DOCKER_LOGIN_ENV), ({} if sky_present else values)


def _restore_env(previous: Mapping[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
