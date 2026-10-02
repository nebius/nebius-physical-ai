"""Validate the private configuration for a BEHAVIOR development evaluation."""

from pathlib import PurePosixPath
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from npa.workflows.behavior_challenge.evaluator_versions import (
    UPSTREAM_COMMIT,
    require_supported_upstream,
)

RequiredText = Annotated[str, Field(min_length=1)]


class SetupModel(BaseModel):
    """Reject unrecognized fields and implicit type conversions in setup files.

    Args:
        **data: Fields declared by the concrete setup model.
    Returns:
        Validated setup fields.
    Raises:
        ValueError: Fields contain unsupported values or template expressions.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*", mode="before")
    @classmethod
    def _literal(cls, value):
        if isinstance(value, str) and (
            any(ord(character) < 32 for character in value)
            or any(token in value for token in ("{{", "}}", "${", "\\"))
        ):
            raise ValueError(
                "Use literal values without control characters or templates"
            )
        return value


class SourceSetup(SetupModel):
    """Name the operator's local public evaluator checkout and pinned revision.

    Args:
        **data: Checkout path and supported revision.
    Returns:
        Validated source declaration.
    Raises:
        ValueError: Source fields are invalid.
    """

    checkout: RequiredText
    revision: str = UPSTREAM_COMMIT

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        return require_supported_upstream(value)


class RuntimeSetup(SetupModel):
    """Describe an operator-prepared simulator image, asset volume and GPU.

    Args:
        **data: Image digest, PVC, accelerator and worker paths.
    Returns:
        Validated runtime declaration; live readiness remains unchecked.
    Raises:
        ValueError: Runtime fields are invalid or placeholders.
    """

    image: RequiredText
    assets_claim: RequiredText
    accelerator: RequiredText
    upstream_root: str = "/opt/BEHAVIOR-1K"
    evaluator_python: str = "/opt/conda/envs/behavior/bin/python"
    data_root: str = "/data/behavior"

    @field_validator("image")
    @classmethod
    def _image(cls, value: str) -> str:
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value):
            raise ValueError("Use a complete image reference pinned by SHA-256")
        if value.endswith("0" * 64) or ".invalid/" in value:
            raise ValueError("Replace the placeholder runtime image")
        return value

    @field_validator("assets_claim")
    @classmethod
    def _claim(cls, value: str) -> str:
        if len(value) > 253 or not re.fullmatch(
            r"[a-z0-9][a-z0-9.-]*[a-z0-9]|[a-z0-9]", value
        ):
            raise ValueError("Use an existing Kubernetes PVC name")
        return value

    @field_validator("accelerator")
    @classmethod
    def _accelerator(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+:1", value):
            raise ValueError("Use one simulator GPU in NAME:1 form")
        if value.split(":")[0].upper() in {"H100", "H200", "B200", "B300", "A100"}:
            raise ValueError("Select a rendering GPU supported by the simulator")
        return value

    @field_validator("upstream_root", "evaluator_python", "data_root")
    @classmethod
    def _worker_path(cls, value: str) -> str:
        if not PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
            raise ValueError("Use an absolute container path without parent traversal")
        if any(character in value for character in ("\n", "\r", "\0", "{{", "${")):
            raise ValueError("Use a literal container path")
        return value


class PolicySetup(SetupModel):
    """Bind the declared checkpoint to an existing task-configured policy service.

    Args:
        **data: Host, port, checkpoint SHA-256 and local runbook path.
    Returns:
        Operator-declared policy identity and endpoint.
    Raises:
        ValueError: Policy fields are invalid or placeholders.
    """

    host: RequiredText
    port: int = Field(default=8000, ge=1, le=65535)
    checkpoint_sha256: RequiredText
    runbook: RequiredText

    @field_validator("host")
    @classmethod
    def _host(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", value) or value.endswith(
            ".invalid"
        ):
            raise ValueError("Use a reachable policy hostname or IPv4 address")
        return value

    @field_validator("checkpoint_sha256")
    @classmethod
    def _checkpoint(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value) or value == "0" * 64:
            raise ValueError(
                "Provide the SHA-256 of the exact served checkpoint archive"
            )
        return value


class ChallengeSetup(SetupModel):
    """Configure one prescribed BEHAVIOR DEV task through standard workflows.

    Args:
        **data: Versioned setup with project, source, runtime and policy fields.
    Returns:
        Validated local evaluation configuration.
    Raises:
        ValueError: A field is invalid or a required prerequisite is missing.
    """

    schema_version: Literal["npa.challenge-setup/v1"]
    challenge: Literal["behavior"]
    project: RequiredText
    infra: RequiredText
    artifact_root: RequiredText
    task: RequiredText
    split: Literal["development"]
    source: SourceSetup
    runtime: RuntimeSetup
    policy: PolicySetup

    @field_validator("project", "task")
    @classmethod
    def _name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("Use a literal project alias or official task name")
        return value

    @field_validator("infra")
    @classmethod
    def _infra(cls, value: str) -> str:
        if not re.fullmatch(r"k8s/[^\s\x00]+", value):
            raise ValueError("Use k8s/<your-kubernetes-context>")
        return value

    @field_validator("artifact_root")
    @classmethod
    def _artifact_root(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path.strip("/")
            or parsed.username
            or parsed.port
            or parsed.query
            or parsed.fragment
            or any(
                part in {".", "..", ""} for part in parsed.path.strip("/").split("/")
            )
            or any(
                character in value for character in (" ", "\n", "\r", "\0", "{{", "${")
            )
        ):
            raise ValueError("Use an S3 bucket and private nonempty prefix")
        return value.rstrip("/")


def starter_config() -> dict:
    """Return editable setup values without inventing operator resources.

    Args:
        None.
    Returns:
        A configuration whose empty prerequisite fields fail validation.
    Raises:
        None.
    """
    return {
        "schema_version": "npa.challenge-setup/v1",
        "challenge": "behavior",
        "project": "",
        "infra": "",
        "artifact_root": "",
        "task": "turning_on_radio",
        "split": "development",
        "source": {"checkout": "", "revision": UPSTREAM_COMMIT},
        "runtime": {
            "image": "",
            "assets_claim": "",
            "accelerator": "",
            "upstream_root": "/opt/BEHAVIOR-1K",
            "evaluator_python": "/opt/conda/envs/behavior/bin/python",
            "data_root": "/data/behavior",
        },
        "policy": {"host": "", "port": 8000, "checkpoint_sha256": "", "runbook": ""},
    }
