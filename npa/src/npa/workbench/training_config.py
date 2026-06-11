"""Shared training configuration helpers for workbench tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse


OverrideStyle = Literal["hydra", "cli"]


class TrainingConfigError(ValueError):
    """Raised when a shared training option is invalid."""


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases settings shared across training entrypoints."""

    enabled: bool = False
    project: str = ""
    run_name: str = ""
    mode: str = "offline"

    def env(self) -> dict[str, str]:
        mode = self.mode.strip() or ("offline" if self.enabled else "disabled")
        values = {
            "NPA_TRAINING_WANDB_ENABLED": "1" if self.enabled else "0",
            "WANDB_MODE": mode if self.enabled else "disabled",
        }
        if self.project:
            values["NPA_TRAINING_WANDB_PROJECT"] = self.project
            values["WANDB_PROJECT"] = self.project
        if self.run_name:
            values["NPA_TRAINING_WANDB_RUN_NAME"] = self.run_name
            values["WANDB_NAME"] = self.run_name
        return values


@dataclass(frozen=True)
class CheckpointS3Config:
    """Bring-your-own S3-compatible checkpoint destination."""

    uri: str = ""
    endpoint_url: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    def env(self) -> dict[str, str]:
        values: dict[str, str] = {}
        if self.uri:
            values["NPA_CHECKPOINT_S3_URI"] = self.uri
        if self.endpoint_url:
            values["NPA_CHECKPOINT_S3_ENDPOINT_URL"] = self.endpoint_url
            values["AWS_ENDPOINT_URL"] = self.endpoint_url
            values["NEBIUS_S3_ENDPOINT"] = self.endpoint_url
            values["S3_ENDPOINT_URL"] = self.endpoint_url
        if self.aws_access_key_id:
            values["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            values["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        return values

    def public_dict(self) -> dict[str, str]:
        return {
            "uri": self.uri,
            "endpoint_url": self.endpoint_url,
            "aws_access_key_id": "set" if self.aws_access_key_id else "",
            "aws_secret_access_key": "set" if self.aws_secret_access_key else "",
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Canonical general-purpose training interface."""

    data_path: str = ""
    overrides: tuple[str, ...] = field(default_factory=tuple)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint_s3: CheckpointS3Config = field(default_factory=CheckpointS3Config)

    def env(self) -> dict[str, str]:
        values: dict[str, str] = {}
        if self.data_path:
            values["NPA_TRAINING_DATA_PATH"] = self.data_path
        if self.overrides:
            values["NPA_TRAINING_OVERRIDES"] = " ".join(self.overrides)
            values["NPA_TRAINING_OVERRIDES_JSON"] = json.dumps(list(self.overrides))
        values.update(self.wandb.env())
        values.update(self.checkpoint_s3.env())
        return values

    def public_dict(self) -> dict[str, Any]:
        return {
            "data_path": self.data_path,
            "overrides": list(self.overrides),
            "wandb": asdict(self.wandb),
            "checkpoint_s3": self.checkpoint_s3.public_dict(),
        }


def build_training_config(
    *,
    data_path: str = "",
    overrides: Iterable[str] | None = None,
    overrides_json: str = "",
    wandb_enabled: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_mode: str = "offline",
    checkpoint_s3_uri: str = "",
    checkpoint_s3_endpoint_url: str = "",
    checkpoint_s3_access_key_id: str = "",
    checkpoint_s3_secret_access_key: str = "",
) -> TrainingConfig:
    """Build and validate the shared training config from CLI/SDK fields."""

    checkpoint = CheckpointS3Config(
        uri=checkpoint_s3_uri.strip(),
        endpoint_url=checkpoint_s3_endpoint_url.strip(),
        aws_access_key_id=checkpoint_s3_access_key_id.strip(),
        aws_secret_access_key=checkpoint_s3_secret_access_key.strip(),
    )
    if checkpoint.uri:
        _validate_s3_uri(checkpoint.uri, field_name="checkpoint_s3_uri")
    if checkpoint.endpoint_url:
        _validate_endpoint_url(checkpoint.endpoint_url)
    return TrainingConfig(
        data_path=data_path.strip(),
        overrides=parse_overrides(overrides or (), overrides_json=overrides_json),
        wandb=WandbConfig(
            enabled=bool(wandb_enabled),
            project=wandb_project.strip(),
            run_name=wandb_run_name.strip(),
            mode=(wandb_mode or "offline").strip(),
        ),
        checkpoint_s3=checkpoint,
    )


def training_config_from_mapping(values: Mapping[str, Any] | None) -> TrainingConfig:
    """Build a TrainingConfig from a YAML/API/SDK mapping."""

    payload = dict(values or {})
    checkpoint = dict(payload.get("checkpoint_s3") or {})
    wandb = dict(payload.get("wandb") or {})
    return build_training_config(
        data_path=str(payload.get("data_path") or ""),
        overrides=_overrides_from_value(payload.get("overrides")),
        wandb_enabled=bool(wandb.get("enabled", False)),
        wandb_project=str(wandb.get("project") or ""),
        wandb_run_name=str(wandb.get("run_name") or ""),
        wandb_mode=str(wandb.get("mode") or "offline"),
        checkpoint_s3_uri=str(checkpoint.get("uri") or payload.get("checkpoint_s3_uri") or ""),
        checkpoint_s3_endpoint_url=str(
            checkpoint.get("endpoint_url") or payload.get("checkpoint_s3_endpoint_url") or ""
        ),
        checkpoint_s3_access_key_id=str(
            checkpoint.get("aws_access_key_id") or payload.get("checkpoint_s3_access_key_id") or ""
        ),
        checkpoint_s3_secret_access_key=str(
            checkpoint.get("aws_secret_access_key") or payload.get("checkpoint_s3_secret_access_key") or ""
        ),
    )


def parse_overrides(values: Iterable[str], *, overrides_json: str = "") -> tuple[str, ...]:
    """Validate repeated Hydra-style key=value overrides."""

    parsed = list(values)
    if overrides_json.strip():
        try:
            raw_json = json.loads(overrides_json)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"overrides_json is not valid JSON: {exc}") from exc
        parsed.extend(_overrides_from_value(raw_json))

    result: list[str] = []
    for raw in parsed:
        value = str(raw).strip()
        if not value:
            continue
        if "=" not in value:
            raise TrainingConfigError(f"override must be KEY=VALUE, got: {value}")
        key = value.split("=", 1)[0].lstrip("+")
        if key.startswith("-"):
            key = key.lstrip("-")
        if not key:
            raise TrainingConfigError(f"override key must not be empty: {value}")
        result.append(value)
    return tuple(result)


def _overrides_from_value(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, Mapping):
        return [f"{key}={_json_scalar(val)}" for key, val in value.items()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return _overrides_from_value(loaded)
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    raise TrainingConfigError("overrides must be a mapping, list, JSON string, or repeated KEY=VALUE values")


def render_overrides(overrides: Iterable[str], *, style: OverrideStyle) -> str:
    """Render overrides for a shell command."""

    values = [format_override(value, style=style) for value in overrides]
    return shlex.join(values)


def overrides_to_mapping(overrides: Iterable[str]) -> dict[str, Any]:
    """Parse generic KEY=VALUE overrides into typed values for non-Hydra trainers."""

    result: dict[str, Any] = {}
    for raw in parse_overrides(overrides):
        key, value = raw.split("=", 1)
        result[_normalize_override_key(key)] = _parse_override_value(value)
    return result


def _normalize_override_key(key: str) -> str:
    normalized = key.strip()
    while normalized.startswith("+"):
        normalized = normalized[1:]
    while normalized.startswith("-"):
        normalized = normalized[1:]
    return normalized.strip()


def _parse_override_value(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() == "null":
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    if stripped.startswith(("[", "{", '"')):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return stripped


def format_override(value: str, *, style: OverrideStyle) -> str:
    override = str(value).strip()
    if style == "hydra":
        return override
    if override.startswith(("-", "+")):
        return override
    return f"--{override}"


def wandb_overrides(wandb: WandbConfig, *, style: OverrideStyle, prefix: str = "wandb") -> list[str]:
    """Return common Hydra/CLI W&B overrides for trainers with W&B config keys."""

    enabled = "true" if wandb.enabled else "false"
    overrides = [f"{prefix}.enable={enabled}"]
    if wandb.project:
        overrides.append(f"{prefix}.project={wandb.project}")
    if wandb.run_name:
        overrides.append(f"{prefix}.name={wandb.run_name}")
    return [format_override(value, style=style) for value in overrides]


def shell_env_exports(values: Mapping[str, str]) -> str:
    """Render shell exports for non-secret env values."""

    if not values:
        return ""
    return " ".join(f"export {key}={shlex.quote(value)};" for key, value in values.items())


def checkpoint_s3_uri(config: TrainingConfig, fallback: str = "") -> str:
    """Return the explicit checkpoint S3 URI, falling back to an older output path."""

    return config.checkpoint_s3.uri or fallback


def upload_checkpoint_path(local_path: str | Path, config: TrainingConfig) -> str:
    """Upload a local file or directory to the configured checkpoint S3 URI."""

    if not config.checkpoint_s3.uri:
        return ""
    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment(
        endpoint_url=config.checkpoint_s3.endpoint_url,
        aws_access_key_id=config.checkpoint_s3.aws_access_key_id,
        aws_secret_access_key=config.checkpoint_s3.aws_secret_access_key,
    )
    return client.upload_path(str(local_path), config.checkpoint_s3.uri)


def checkpoint_upload_python(local_dir_expr: str, uri_expr: str = 'os.environ["NPA_CHECKPOINT_S3_URI"]') -> str:
    """Return Python code that uploads a directory using checkpoint S3 env vars."""

    return f"""
import os
import pathlib
from urllib.parse import urlparse

import boto3

uri = {uri_expr}
parsed = urlparse(uri)
if parsed.scheme != "s3" or not parsed.netloc:
    raise SystemExit(f"invalid checkpoint S3 URI: {{uri}}")
prefix = parsed.path.lstrip("/")
if prefix and not prefix.endswith("/"):
    prefix += "/"
base = pathlib.Path({local_dir_expr})
s3 = boto3.client(
    "s3",
    endpoint_url=(
        os.environ.get("NPA_CHECKPOINT_S3_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT")
        or None
    ),
)
for file_path in base.rglob("*"):
    if file_path.is_file():
        key = prefix + str(file_path.relative_to(base))
        s3.upload_file(str(file_path), parsed.netloc, key)
        print(f"uploaded s3://{{parsed.netloc}}/{{key}}", flush=True)
"""


def _json_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _validate_s3_uri(uri: str, *, field_name: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise TrainingConfigError(f"{field_name} must be an s3://bucket/prefix URI")


def _validate_endpoint_url(endpoint_url: str) -> None:
    parsed = urlparse(endpoint_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise TrainingConfigError("checkpoint_s3_endpoint_url must be an http(s) URL")
