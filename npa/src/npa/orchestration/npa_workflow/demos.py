"""Select complete public reference workflows and retrieve their compact reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from uuid import uuid4

from npa.clients.config import (
    ConfigError,
    default_project_name,
    resolve_environment,
    resolve_project_storage,
)
from npa.clients.storage import StorageClient
from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec


@dataclass(frozen=True)
class WorkflowDemo:
    """A public sample and its authoritative declarative workflow."""

    name: str
    workflow: str
    sample: str
    result: str
    secret_env: tuple[str, ...] = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


DEMOS = (
    WorkflowDemo(
        "real-to-sim",
        "rgbd-scan-to-policy-demo.yaml",
        "Public calibrated TUM office RGB-D capture",
        "Measured collision scene, navigation checkpoint, and scored evaluation",
    ),
    WorkflowDemo(
        "synthetic-data",
        "multicamera-rgbd-warehouse.yaml",
        "Public NVIDIA warehouse; downloaded at runtime",
        "265 poses, four synchronized RGB/depth cameras, and colored point clouds",
    ),
    WorkflowDemo(
        "rl-improvement",
        "field-failure-reference-demo.yaml",
        "Public reconstructed office and warehouse navigation cases",
        "Baseline, continued checkpoint, and independently held-out comparison",
    ),
    WorkflowDemo(
        "nurec",
        "nurec-reconstruct.yaml",
        "Pinned public NVIDIA PPISP NCore capture",
        "Native Gaussian reconstruction, novel views, and measured image quality",
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "NGC_API_KEY"),
    ),
)


def list_demos() -> list[dict[str, object]]:
    """Describe the shipped public sample workflows without contacting providers.

    Args:
        None.
    Returns:
        Sample, result, credential-name, and workflow metadata for each demo.
    Raises:
        None.
    """
    return [asdict(demo) for demo in DEMOS]


def select_demo(name: str) -> WorkflowDemo:
    """Resolve a stable demo name without accepting arbitrary workflow paths.

    Args:
        name: One of the names returned by list_demos.
    Returns:
        The selected reference definition.
    Raises:
        ValueError: The demo name is unknown.
    """
    for demo in DEMOS:
        if demo.name == name:
            return demo
    raise ValueError(
        f"Unknown demo {name!r}; choose {', '.join(d.name for d in DEMOS)}."
    )


def validate_demo_run_id(run_id: str) -> str:
    """Validate one path-safe workflow identity used for execution and reports.

    Args:
        run_id: The run identity to validate.
    Returns:
        The unchanged validated identity.
    Raises:
        ValueError: The identity is empty or contains unsafe path characters.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run_id):
        raise ValueError(
            "Run ID must contain 1–63 lowercase letters, digits, or hyphens."
        )
    return run_id


def _storage(project: str):
    selected = project or default_project_name()
    if not selected:
        raise ConfigError("Configure a Workbench project first, then pass --project.")
    environment = resolve_environment(project=selected)
    if environment is None or not environment.project_id:
        raise ConfigError("The selected Workbench project is not configured.")
    storage = resolve_project_storage(
        project=selected,
        include_shared_credentials=False,
        include_environment=False,
    )
    uri = storage.checkpoint_bucket
    if "://" in uri and not uri.startswith("s3://"):
        raise ConfigError("The selected project needs an S3 storage location.")
    parsed = urlsplit(uri if uri.startswith("s3://") else f"s3://{uri}")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or not storage.endpoint_url
    ):
        raise ConfigError(
            "The selected project needs configured S3 storage and an endpoint."
        )
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        raise ConfigError("The selected project's storage prefix is invalid.")
    return selected, storage, parsed.netloc, parsed.path.strip("/")


def _prefix(base: str, name: str, run_id: str) -> str:
    return "/".join(part for part in (base, "demos", name, run_id) if part)


def prepare_demo(
    name: str, *, project: str = "", run_id: str = ""
) -> dict[str, object]:
    """Resolve public inputs and run paths for the standard workflow submitter.

    Args:
        name: Shipped demo name.
        project: Explicit configured project, or its configured default.
        run_id: Existing or desired identity; empty selects a fresh identity.
    Returns:
        Standard submit arguments and the resulting report location.
    Raises:
        ValueError: The name, run ID, or installed workflow is invalid.
        ConfigError: The selected project's storage is not configured.
    """
    demo = select_demo(name)
    spec = resolve_npa_workflow_spec(demo.workflow)
    if spec is None:
        raise ValueError(
            f"Workflow {demo.workflow} is missing; install the complete demo revision."
        )
    identity = (
        run_id or f"{name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    )
    validate_demo_run_id(identity)
    selected, storage, bucket, base = _storage(project)
    prefix = _prefix(base, name, identity)
    return {
        "name": name,
        "yaml_path": spec,
        "project": selected,
        "run_id": identity,
        "s3_bucket": bucket,
        "s3_endpoint": storage.endpoint_url,
        "s3_prefix": prefix,
        "secret_env": list(demo.secret_env),
        "var": [f"bucket={bucket}", f"prefix={prefix}", "source_overlay=true"],
        "report_uri": f"s3://{bucket}/{prefix}/reports/index.html",
    }


def demo_storage_environment(selection: dict) -> dict[str, str]:
    """Bind demo submission to the same saved project storage used for viewing.

    Args:
        selection: Non-secret prepare_demo result for this execution.
    Returns:
        Private environment values for a restoring submit context; never log them.
    Raises:
        ConfigError: Saved credentials are incomplete or routing changed.
    """
    from npa.orchestration.npa_workflow.submit_credentials import (
        STORAGE_ENDPOINT_ENV_NAMES,
    )

    _, storage, bucket, _ = _storage(selection["project"])
    if not storage.aws_access_key_id or not storage.aws_secret_access_key:
        raise ConfigError("The selected project needs a complete S3 credential pair.")
    if (
        bucket != selection["s3_bucket"]
        or storage.endpoint_url != selection["s3_endpoint"]
    ):
        raise ConfigError("The selected project's storage changed during preparation.")
    values = dict.fromkeys(STORAGE_ENDPOINT_ENV_NAMES, storage.endpoint_url)
    values.update(
        AWS_ACCESS_KEY_ID=storage.aws_access_key_id,
        AWS_SECRET_ACCESS_KEY=storage.aws_secret_access_key,
        AWS_SESSION_TOKEN="",
        AWS_SECURITY_TOKEN="",
        NPA_SKYPILOT_PROJECT=selection["project"],
        NPA_S3_BUCKET=bucket,
        NEBIUS_S3_BUCKET=bucket,
        NPA_CHECKPOINT_BUCKET=f"s3://{bucket}",
        NPA_S3_PREFIX=selection["s3_prefix"],
        NPA_SRC_S3_URI="",
        NPA_E2E_NPA_SRC_S3_URI="",
    )
    return values


def download_demo_report(name: str, run_id: str, *, project: str = "") -> Path:
    """Download a completed compact HTML report using selected-project credentials.

    Args:
        name: Shipped demo name.
        run_id: Exact workflow identity printed by the demo launcher.
        project: Explicit configured project, or its configured default.
    Returns:
        Local HTML file ready to open in a browser without serving S3 credentials.
    Raises:
        ValueError: The demo or run identity is invalid.
        ConfigError: Project storage is not configured.
        OSError: Local report storage is unavailable.
        botocore.exceptions.ClientError: The report is absent or access is denied.
    """
    select_demo(name)
    validate_demo_run_id(run_id)
    selected, storage, bucket, base = _storage(project)
    if not storage.aws_access_key_id or not storage.aws_secret_access_key:
        raise ConfigError("The selected project needs a complete S3 credential pair.")
    client = StorageClient(
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.aws_access_key_id,
        aws_secret_access_key=storage.aws_secret_access_key,
    )
    root = Path(os.environ.get("NPA_CONFIG_DIR") or Path.home() / ".npa")
    identity = "\n".join((selected, storage.endpoint_url, bucket, base))
    scope = hashlib.sha256(identity.encode()).hexdigest()
    directory = root / "demo-reports" / scope / name / run_id
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    uri = f"s3://{bucket}/{_prefix(base, name, run_id)}/reports/index.html"
    destination = directory / "index.html"
    with tempfile.TemporaryDirectory(prefix=".download-", dir=directory) as scratch:
        staged = Path(scratch) / "index.html"
        client.download_file(uri, str(staged))
        staged.chmod(0o600)
        staged.replace(destination)
    return destination
