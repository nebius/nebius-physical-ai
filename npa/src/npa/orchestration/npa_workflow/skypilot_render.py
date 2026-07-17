"""Render an ``npa.workflow`` execution plan as a SkyPilot multi-doc YAML."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import yaml

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import ExecutionPlan, PlanStep
from npa.orchestration.npa_workflow.scheduler import build_scheduler_task
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec

# Map toolRef prefixes / exact names onto CONTAINER_IMAGE_NAMES keys.
# Token Factory is a hosted HTTP API client. Do not pin the heavy cosmos image:
# SkyPilot's k8s apt-ssh runtime setup fails inside npa-cosmos. Use the default
# SkyPilot image and stage npa via NPA_SRC_S3_URI (or an image override).
TOOL_REF_IMAGE_TOOL: dict[str, str] = {
    "workbench.vlm_eval": "cosmos",
    "workbench.cosmos2": "cosmos2-transfer",
    "workbench.cosmos3": "cosmos3-reason",
    "workbench.lancedb": "lancedb",
    "workbench.detection_training": "detection-training",
    "workbench.fiftyone": "fiftyone",
    "workbench.rl": "isaac-lab",
    "workbench.isaac_lab": "isaac-lab",
    "workbench.lerobot": "lerobot",
    "workbench.sonic": "sonic",
    "workbench.mjlab": "sonic",
    "workbench.retargeting": "retargeting",
    "workbench.sim2real": "lerobot-vlm-rl",
    "workbench.sim2real_envgen": "envgen",
    "workbench.byof": "isaac-lab",
    "workbench.genesis": "genesis",
    "workbench.groot": "groot",
}

SECRET_ENV_HINTS: dict[str, tuple[str, ...]] = {
    "workbench.token_factory": ("NEBIUS_TOKEN_FACTORY_KEY",),
    "workbench.vlm_eval": (),
    "workbench.cosmos3": ("HF_TOKEN",),
    "workbench.sonic": ("HF_TOKEN", "NGC_API_KEY"),
    "workbench.groot": ("HF_TOKEN", "NGC_API_KEY"),
}


class NpaWorkflowRenderError(NpaWorkflowError):
    """Raised when an npa.workflow plan cannot be rendered to SkyPilot YAML."""


def _default_aws_endpoint_url() -> str:
    """Prefer the operator's configured endpoint over a hard-coded region."""

    import os

    return (
        os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT")
        or "https://storage.eu-north1.nebius.cloud"
    )


@dataclass(frozen=True)
class SkypilotRenderOptions:
    """Controls how planned steps become SkyPilot task documents."""

    registry: str = ""
    image_overrides: Mapping[str, str] = field(default_factory=dict)
    default_setup: bool = True
    execution: str = "serial"
    aws_endpoint_url: str = field(default_factory=_default_aws_endpoint_url)
    include_aws_endpoint: bool = True
    gpu_target: str = ""
    image_variant: str = ""
    # When False (``--plan-only``), embed placeholders instead of minting live
    # Nebius registry tokens into rendered YAML that may be printed to stdout.
    materialize_registry_secrets: bool = True


def normalize_resources(resources: Mapping[str, Any]) -> dict[str, Any]:
    """Map an npa.workflow resource profile onto a SkyPilot ``resources`` block.

    On Kubernetes, exact ``cpus`` / ``memory`` often fail prechecks when no node
    has that precise free shape. Append ``+`` so SkyPilot can schedule on larger
    nodes (including GPU nodes with spare CPU).
    """

    out: dict[str, Any] = {}
    for key in ("cloud", "accelerators", "cpus", "memory", "use_spot", "region"):
        if key not in resources or resources[key] in (None, ""):
            continue
        value = resources[key]
        if key == "memory" and isinstance(value, str):
            stripped = value.strip()
            if stripped.lower().endswith("gi"):
                value = stripped[:-2]
            elif stripped.lower().endswith("g"):
                value = stripped[:-1]
        out[key] = value

    cloud = str(out.get("cloud") or "").strip().lower()
    if cloud in {"kubernetes", "k8s"}:
        for key in ("cpus", "memory"):
            if key not in out:
                continue
            raw = str(out[key]).strip()
            if raw and not raw.endswith("+"):
                out[key] = f"{raw}+"
    return out


def tool_image_key(tool_ref: str) -> str | None:
    """Return the CONTAINER_IMAGE_NAMES key for a toolRef, if known."""

    if not tool_ref:
        return None
    if tool_ref in TOOL_REF_IMAGE_TOOL:
        return TOOL_REF_IMAGE_TOOL[tool_ref]
    # Longest-prefix match.
    best = ""
    for prefix, tool in TOOL_REF_IMAGE_TOOL.items():
        if tool_ref == prefix or tool_ref.startswith(prefix + "."):
            if len(prefix) > len(best):
                best = prefix
    return TOOL_REF_IMAGE_TOOL.get(best)


def resolve_task_image(
    tool_ref: str,
    resources: Mapping[str, Any],
    *,
    options: SkypilotRenderOptions,
) -> str:
    """Resolve a fully-qualified image ref for one planned step."""

    if tool_ref in options.image_overrides:
        return str(options.image_overrides[tool_ref] or "").strip()
    if "*" in options.image_overrides:
        return str(options.image_overrides["*"] or "").strip()

    explicit = str(resources.get("image") or "").strip()
    if explicit:
        return explicit

    tool = tool_image_key(tool_ref)
    if not tool:
        return ""

    from npa.deploy.images import container_image_for_tool

    kwargs: dict[str, Any] = {}
    if options.registry:
        kwargs["registry"] = options.registry
    if tool == "sonic":
        if options.gpu_target:
            kwargs["gpu_target"] = options.gpu_target
        if options.image_variant:
            kwargs["image_variant"] = options.image_variant
    return container_image_for_tool(tool, **kwargs)


def render_task_run_script(command: Sequence[str]) -> str:
    """Turn an argv list into a SkyPilot ``run:`` shell script."""

    if not command:
        raise NpaWorkflowRenderError("cannot render empty command for SkyPilot task")
    quoted = " ".join(shlex.quote(str(part)) for part in command)
    return (
        "set -euo pipefail\n"
        # Use unbraced $HOME/$PATH so SkyPilot placeholder lint stays clean.
        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
        f"{quoted}\n"
    )


def default_npa_setup() -> str:
    """Ensure the ``npa`` CLI is available on the SkyPilot worker.

    Workbench images bake npa at ``/opt/nebius-physical-ai/npa``. When a task
    uses SkyPilot's default image (e.g. Token Factory API twins), setup can:

    1. install from a mounted ``/tmp/npa-src`` (S3 URI via ``NPA_SRC_S3_URI``), or
    2. sync from ``NPA_SRC_S3_URI`` with the AWS CLI / boto3, then install.
    """

    return (
        "set -e\n"
        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
        "if ! command -v npa >/dev/null 2>&1; then\n"
        "  if [ -d /opt/nebius-physical-ai/npa ]; then\n"
        "    python3 -m pip install -q -e /opt/nebius-physical-ai/npa\n"
        "  else\n"
        "    if [ ! -d /tmp/npa-src ] && [ -n \"$NPA_SRC_S3_URI\" ]; then\n"
        "      python3 -m pip install -q boto3\n"
        "      python3 - <<'PY'\n"
        "import os, pathlib\n"
        "from urllib.parse import urlparse\n"
        "import boto3\n"
        "from botocore.client import Config\n"
        "uri = os.environ['NPA_SRC_S3_URI'].rstrip('/')\n"
        "parsed = urlparse(uri if '://' in uri else f's3://{uri}')\n"
        "bucket, prefix = parsed.netloc, parsed.path.lstrip('/')\n"
        "dest = pathlib.Path('/tmp/npa-src')\n"
        "dest.mkdir(parents=True, exist_ok=True)\n"
        "print('syncing', uri, '->', dest, flush=True)\n"
        "kwargs = {'config': Config(signature_version='s3v4')}\n"
        "if os.environ.get('AWS_ENDPOINT_URL'):\n"
        "    kwargs['endpoint_url'] = os.environ['AWS_ENDPOINT_URL']\n"
        "s3 = boto3.client('s3', **kwargs)\n"
        "token = None\n"
        "while True:\n"
        "    kw = {'Bucket': bucket, 'Prefix': prefix}\n"
        "    if token:\n"
        "        kw['ContinuationToken'] = token\n"
        "    resp = s3.list_objects_v2(**kw)\n"
        "    for obj in resp.get('Contents') or ():\n"
        "        key = obj['Key']\n"
        "        rel = key[len(prefix):].lstrip('/') if prefix else key\n"
        "        if not rel or key.endswith('/'):\n"
        "            continue\n"
        "        out = dest / rel\n"
        "        out.parent.mkdir(parents=True, exist_ok=True)\n"
        "        s3.download_file(bucket, key, str(out))\n"
        "    if not resp.get('IsTruncated'):\n"
        "        break\n"
        "    token = resp.get('NextContinuationToken')\n"
        "PY\n"
        "    fi\n"
        "    if [ -d /tmp/npa-src ]; then\n"
        "      python3 -m pip install -q -e /tmp/npa-src\n"
        "    else\n"
        "      echo 'npa CLI not found; set NPA_SRC_S3_URI or use a workbench image' >&2\n"
        "      exit 1\n"
        "    fi\n"
        "  fi\n"
        "fi\n"
        "command -v npa >/dev/null 2>&1 || "
        "{ echo 'npa still missing after setup' >&2; exit 1; }\n"
    )


def render_setup_for_tool(
    tool_ref: str,
    *,
    config: Mapping[str, Any],
    options: SkypilotRenderOptions,
) -> str:
    """Return a SkyPilot ``setup:`` block for a toolRef."""

    if not options.default_setup:
        return ""
    parts = [default_npa_setup()]
    backend = str(config.get("vlm_backend") or "").strip().lower()
    if tool_ref.startswith("workbench.vlm_eval") and backend in {"self-hosted", "self_hosted"}:
        parts.append(
            "python3 - <<'PY'\n"
            "import importlib.util\n"
            "import subprocess\n"
            "import sys\n"
            "\n"
            "if importlib.util.find_spec('vllm') is None:\n"
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'vllm>=0.8.5'])\n"
            "PY\n"
        )
    if tool_ref.startswith("workbench.token_factory"):
        # Avoid ${VAR:-} bash forms so SkyPilot placeholder lint stays clean.
        parts.append(
            "if [[ -z \"$NEBIUS_TOKEN_FACTORY_KEY\" ]]; then\n"
            "  echo 'NEBIUS_TOKEN_FACTORY_KEY is required. Pass it with --secret-env "
            "NEBIUS_TOKEN_FACTORY_KEY' >&2\n"
            "  exit 1\n"
            "fi\n"
        )
    return "".join(parts)


def secret_env_hints_for_plan(steps: Sequence[PlanStep]) -> tuple[str, ...]:
    """Collect recommended ``--secret-env`` names for a planned workflow."""

    hints: list[str] = []
    seen: set[str] = set()
    for step in steps:
        tool_ref = step.tool_ref or ""
        for prefix, names in SECRET_ENV_HINTS.items():
            if tool_ref == prefix or tool_ref.startswith(prefix + "."):
                for name in names:
                    if name not in seen:
                        seen.add(name)
                        hints.append(name)
    return tuple(hints)


def build_skypilot_task_doc(
    spec: NpaWorkflowSpec,
    step: PlanStep,
    *,
    run_id: str,
    options: SkypilotRenderOptions,
) -> dict[str, Any]:
    """Build one SkyPilot task document from a planned step."""

    scheduler_task = build_scheduler_task(spec, step, run_id=run_id)
    resources = normalize_resources(scheduler_task.get("resources") or {})
    image = resolve_task_image(
        str(scheduler_task.get("tool_ref") or ""),
        scheduler_task.get("resources") or {},
        options=options,
    )
    if image:
        resources["image_id"] = f"docker:{image}" if not image.startswith("docker:") else image

    command = list(scheduler_task.get("command") or [])
    if not command:
        raise NpaWorkflowRenderError(
            f"planned step {scheduler_task['name']!r} has no command to render"
        )

    envs: dict[str, str] = {
        "NPA_WORKFLOW_NAME": spec.name,
        "NPA_WORKFLOW_RUN_ID": run_id,
        "NPA_WORKFLOW_STATE": str(scheduler_task["name"]),
    }
    if options.include_aws_endpoint and options.aws_endpoint_url:
        envs["AWS_ENDPOINT_URL"] = options.aws_endpoint_url
    if image:
        envs["NPA_TASK_IMAGE"] = image.removeprefix("docker:")

    doc: dict[str, Any] = {
        "name": scheduler_task["name"],
        "resources": resources,
        "envs": envs,
        "run": render_task_run_script(command),
    }
    setup = render_setup_for_tool(
        str(scheduler_task.get("tool_ref") or ""),
        config=spec.config,
        options=options,
    )
    if setup.strip():
        doc["setup"] = setup
    # When no workbench image is pinned, point setup at an existing S3 copy of
    # the npa package (SkyPilot local file_mounts create new buckets and fail
    # on Nebius). Operators set NPA_SRC_S3_URI=s3://bucket/prefix/npa.
    if not image:
        import os

        src_uri = (
            os.environ.get("NPA_SRC_S3_URI")
            or os.environ.get("NPA_E2E_NPA_SRC_S3_URI")
            or ""
        ).strip()
        if not src_uri:
            raise NpaWorkflowRenderError(
                f"planned step {scheduler_task['name']!r} has no workbench image "
                "and NPA_SRC_S3_URI is unset; set NPA_SRC_S3_URI=s3://bucket/prefix/npa "
                "or pass --image <registry>/npa-<tool>:<tag>"
            )
        envs["NPA_SRC_S3_URI"] = src_uri
        doc["envs"] = envs
    _inject_nebius_registry_docker_secrets(
        doc,
        materialize=options.materialize_registry_secrets,
    )
    return doc


def _is_nebius_registry_image(image_id: str) -> bool:
    value = image_id.removeprefix("docker:").strip()
    host = value.split("/", 1)[0] if "/" in value else ""
    return host.startswith("cr.") and host.endswith(".nebius.cloud")


def _inject_nebius_registry_docker_secrets(
    doc: dict[str, Any],
    *,
    materialize: bool = True,
) -> None:
    """Embed SkyPilot Docker login secrets for private Nebius registry images.

    Matches the burst submit path: ``resources.image_id`` is pulled before YAML
    ``setup`` runs, so registry auth must live in task ``secrets``.

    When ``materialize`` is False (plan-only), embed a placeholder password so
    rendered YAML can be printed without minting or leaking live IAM tokens.
    """

    import os

    resources = doc.get("resources") or {}
    if not isinstance(resources, dict):
        return
    cloud = str(resources.get("cloud") or "").strip().lower()
    image_id = str(resources.get("image_id") or "").strip()
    # Nebius VMs need SKYPILOT_DOCKER_* for private pulls; k8s uses imagePullSecrets
    # but still benefits from secrets when the controller falls back to docker login.
    if cloud not in {"nebius", "kubernetes", "k8s"} or not _is_nebius_registry_image(image_id):
        return

    server = image_id.removeprefix("docker:").split("/", 1)[0]
    username = (
        os.environ.get("SKYPILOT_DOCKER_USERNAME")
        or os.environ.get("NPA_REGISTRY_USERNAME")
        or "iam"
    )
    if materialize:
        password = (
            os.environ.get("SKYPILOT_DOCKER_PASSWORD")
            or os.environ.get("NPA_REGISTRY_PASSWORD")
            or ""
        )
        if not password:
            try:
                from npa.workflows.sim2real.registry_auth import mint_nebius_registry_token

                password = mint_nebius_registry_token()
            except Exception as exc:  # noqa: BLE001
                raise NpaWorkflowRenderError(
                    "Nebius registry image requires SKYPILOT_DOCKER_PASSWORD "
                    f"(or mintable IAM token); failed to mint: {exc}"
                ) from exc
    else:
        password = "<SKYPILOT_DOCKER_PASSWORD>"

    secrets = doc.setdefault("secrets", {})
    if not isinstance(secrets, dict):
        raise NpaWorkflowRenderError("SkyPilot task secrets must be a mapping")
    secrets.setdefault("SKYPILOT_DOCKER_SERVER", server)
    secrets.setdefault("SKYPILOT_DOCKER_USERNAME", username)
    secrets.setdefault("SKYPILOT_DOCKER_PASSWORD", password)


def render_skypilot_yaml(
    spec: NpaWorkflowSpec,
    plan: ExecutionPlan,
    *,
    run_id: str,
    options: SkypilotRenderOptions | None = None,
) -> str:
    """Return multi-document SkyPilot YAML text for a planned npa.workflow."""

    opts = options or SkypilotRenderOptions()
    if opts.execution != "serial":
        raise NpaWorkflowRenderError(
            f"npa.workflow/v0.0.1 renderer only supports execution=serial, got {opts.execution!r}"
        )
    if not plan.steps:
        raise NpaWorkflowRenderError(f"workflow {spec.name!r} planned zero steps")

    header = {
        "name": spec.name,
        "execution": "serial",
    }
    docs: list[dict[str, Any]] = [header]
    for step in plan.steps:
        docs.append(build_skypilot_task_doc(spec, step, run_id=run_id, options=opts))

    chunks: list[str] = []
    for doc in docs:
        chunks.append(
            yaml.safe_dump(
                doc,
                sort_keys=False,
                default_flow_style=False,
            ).rstrip()
        )
    return "\n---\n".join(chunks) + "\n"


_SKYPILOT_PLACEHOLDER_RE = __import__("re").compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def assert_no_unresolved_placeholders(yaml_text: str) -> None:
    """Fail if rendered YAML still contains SkyPilot-style ``${NAME}`` placeholders.

    Allows bash parameter expansions such as ``$NAME`` (no braces) used in setup
    scripts. Flags only bare ``${NAME}`` forms that SkyPilot would leave literal.
    """

    unresolved = sorted(set(_SKYPILOT_PLACEHOLDER_RE.findall(yaml_text)))
    if unresolved:
        joined = ", ".join(f"${{{name}}}" for name in unresolved)
        raise NpaWorkflowRenderError(
            "rendered SkyPilot YAML still contains unresolved placeholders: "
            f"{joined}; resolve images and config before submit"
        )
