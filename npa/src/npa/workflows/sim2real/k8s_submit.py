"""Submit the canonical Sim2Real runbook as a direct Kubernetes Job."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from npa.deploy.images import container_image_for_tool
from npa.workflows.sim2real.constants import DEFAULT_LEROBOT_DATASET_ID, DEFAULT_PREFIX
from npa.workflows.sim2real.materialize import (
    controller_spec_digest,
    default_runbook_path,
    materialize_k8s_job,
)
from npa.workflows.sim2real.monitor import (
    load_operator_config,
    normalize_staged_run_id,
    resolve_kubeconfig,
)


@dataclass(frozen=True)
class Sim2RealSubmitResult:
    run_id: str
    job_name: str
    k8s_context: str
    run_prefix_uri: str
    status: str = "submitted"
    log_path: str = ""
    manifest_sha256: str = ""


_REQUIRED_REAL_IMAGE_ENVS = (
    "AUGMENT_IMAGE",
    "ENVGEN_IMAGE",
    "POLICY_IMAGE",
    "TRAINER_IMAGE",
    "VLM_IMAGE",
    "VLM_REASON2_IMAGE",
    "VLM_REASON3_IMAGE",
    "EVAL_IMAGE",
    "ISAAC_IMAGE",
)
_RERUN_VIEWER_IMAGE_ENV = "NPA_RERUN_VIEWER_IMAGE"
_PLACEHOLDER_IMAGE_MARKERS = ("example.invalid", "<your-registry-id>", "${")

_REQUIRED_REAL_COMMAND_ENVS = (
    "BYO_POLICY_COMMAND",
    "BYO_TRAINER_COMMAND",
    "BYO_EVAL_COMMAND",
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "npa" / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate repo root (npa/pyproject.toml)")


def _registry_qualified(image: str) -> bool:
    ref = image.removeprefix("docker:").strip()
    if not ref or any(marker in ref.lower() for marker in _PLACEHOLDER_IMAGE_MARKERS):
        return False
    if "/" not in ref:
        return False
    host, leaf = ref.split("/", 1)
    return bool(
        ("." in host or ":" in host or host == "localhost")
        and (":" in leaf or "@" in leaf)
    )


def _immutable_image(image: str) -> bool:
    ref = image.removeprefix("docker:").strip()
    digest = ref.rsplit("@sha256:", 1)[-1] if "@sha256:" in ref else ""
    return (
        _registry_qualified(ref)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest.lower())
    )


def _enabled(value: str, *, default: bool = False) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _required_real_image_envs(values: dict[str, str]) -> tuple[str, ...]:
    required = list(_REQUIRED_REAL_IMAGE_ENVS)
    if _enabled(values.get("NPA_SIM2REAL_RERUN", ""), default=True) and _enabled(
        values.get("NPA_SIM2REAL_RERUN_SERVE", ""), default=True
    ):
        required.append(_RERUN_VIEWER_IMAGE_ENV)
    return tuple(required)


def _resolve_image_overrides(overrides: dict[str, str]) -> dict[str, str]:
    """Resolve image aliases before qualification and pull-secret collection."""

    resolved = dict(overrides)
    if "VLM_IMAGE" in overrides:
        resolved.setdefault("VLM_REASON2_IMAGE", overrides["VLM_IMAGE"])
        resolved.setdefault("VLM_REASON3_IMAGE", overrides["VLM_IMAGE"])
    if (
        "NPA_SIM2REAL_RERUN_IMAGE" in overrides
        and _RERUN_VIEWER_IMAGE_ENV not in overrides
    ):
        resolved[_RERUN_VIEWER_IMAGE_ENV] = overrides["NPA_SIM2REAL_RERUN_IMAGE"]
    return resolved


@contextmanager
def _secure_temporary_manifest(
    *, run_id: str, job_name: str, manifest_yaml: str
) -> Iterator[Path]:
    """Write a mode-0600 run-scoped manifest and remove it on every exit path."""

    run_slug = re.sub(r"[^a-zA-Z0-9-]+", "-", run_id).strip("-")[:32] or "run"
    with tempfile.TemporaryDirectory(prefix=f"npa-sim2real-{run_slug}-") as temporary:
        manifest_root = Path(temporary)
        manifest_root.chmod(0o700)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f"{job_name}-",
            suffix=".yaml",
            dir=manifest_root,
            delete=False,
        ) as handle:
            handle.write(manifest_yaml)
            manifest_path = Path(handle.name)
        manifest_path.chmod(0o600)
        yield manifest_path


def _validate_real_runtime_env(values: dict[str, str]) -> None:
    """Fail before apply when a canonical real-tier knob is invalid or inert."""

    from npa.workflows.sim2real.camera_views import camera_view_names
    from npa.workflows.sim2real.capture import capture_settings, ppo_settings

    capture_settings(values)
    ppo_settings(values)
    camera_view_names(values.get("NPA_SIM2REAL_CAMERA_VIEWS", ""))

    def integer(name: str, *, minimum: int = 1) -> int:
        raw = values.get(name, "")
        try:
            parsed = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
        if parsed < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
        return parsed

    for name in (
        "INNER_ITERATIONS",
        "OUTER_ITERATIONS",
        "LOOP_OF_LOOPS_ITERATIONS",
        "ROLLOUT_COUNT",
        "STEPS_PER_ROLLOUT",
        "VALIDATION_ENV_COUNT",
        "HELDOUT_ENV_COUNT",
        "NPA_ENV_COUNT",
    ):
        integer(name)
    integer("NPA_SIM2REAL_HELDOUT_EVAL_LIMIT", minimum=0)
    integer("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", minimum=0)
    integer("NPA_BYO_ISAAC_JOB_TIMEOUT_S", minimum=0)
    integer("NPA_BYO_ISAAC_VALIDATION_INTERVAL")
    rollout_horizon = integer("NPA_SIM2REAL_ROLLOUT_HORIZON_STEPS")
    sampled_points = integer("STEPS_PER_ROLLOUT")
    if rollout_horizon < sampled_points:
        raise ValueError(
            "NPA_SIM2REAL_ROLLOUT_HORIZON_STEPS must be >= STEPS_PER_ROLLOUT"
        )
    reason_frames = integer("NPA_COSMOS_REASON_MAX_FRAMES")
    if reason_frames < sampled_points:
        raise ValueError(
            "NPA_COSMOS_REASON_MAX_FRAMES must be >= STEPS_PER_ROLLOUT so "
            "every decision/event has visual evidence"
        )
    reason_tokens = integer("NPA_COSMOS_REASON_MAX_NEW_TOKENS")
    minimum_reason_tokens = sampled_points * 64
    if reason_tokens < minimum_reason_tokens:
        raise ValueError(
            "NPA_COSMOS_REASON_MAX_NEW_TOKENS must be >= 64 * "
            f"STEPS_PER_ROLLOUT ({minimum_reason_tokens}), got {reason_tokens}"
        )

    threshold = float(values.get("SUCCESS_THRESHOLD", "0.50"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"SUCCESS_THRESHOLD must be in [0, 1], got {threshold}")
    distance = float(values.get("NPA_BYO_ISAAC_SUCCESS_DIST_M", "0.05"))
    if not 0.001 <= distance <= 10.0:
        raise ValueError(
            "NPA_BYO_ISAAC_SUCCESS_DIST_M must be in [0.001, 10] metres, "
            f"got {distance}"
        )
    boolean_values = {"", "0", "1", "false", "true", "no", "yes", "off", "on"}
    for name in (
        "NPA_SIM2REAL_EARLY_EXIT",
        "NPA_SIM2REAL_RERUN",
        "NPA_SIM2REAL_RERUN_SERVE",
    ):
        if values.get(name, "").strip().lower() not in boolean_values:
            raise ValueError(f"{name} must be boolean")
    for name in _REQUIRED_REAL_COMMAND_ENVS:
        if not values.get(name, "").strip():
            raise ValueError(
                f"{name} must invoke a real external component in the canonical real tier"
            )
    if values.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() != "1":
        raise ValueError(
            "NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS=1 is mandatory for the canonical real tier"
        )
    for name in ("OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA"):
        if values.get(name, "").strip().upper() != "YES":
            raise ValueError(
                f"{name}=YES is required after the operator accepts the NVIDIA Isaac licence"
            )


def _default_image_env(
    registry: str, *, orchestrator_image: str = ""
) -> tuple[dict[str, str], str]:
    trainer = container_image_for_tool("lerobot-vlm-rl", registry=registry)
    vlm = container_image_for_tool("cosmos3-reason", registry=registry)
    images = {
        "NPA_REGISTRY": registry,
        "AUGMENT_IMAGE": container_image_for_tool(
            "cosmos2-transfer", registry=registry
        ),
        "ENVGEN_IMAGE": container_image_for_tool("envgen", registry=registry),
        # Stage 7's real Isaac command and stage 9's PPO command both use the
        # trainer image as their NPA-capable parent before dispatching Isaac jobs.
        "POLICY_IMAGE": trainer,
        "TRAINER_IMAGE": trainer,
        "VLM_IMAGE": vlm,
        "VLM_REASON2_IMAGE": vlm,
        "VLM_REASON3_IMAGE": vlm,
        "EVAL_IMAGE": container_image_for_tool("loop-eval", registry=registry),
        "ISAAC_IMAGE": container_image_for_tool("isaac-lab", registry=registry),
        "NPA_RERUN_VIEWER_IMAGE": container_image_for_tool(
            "rerun-viewer", registry=registry
        ),
    }
    return images, orchestrator_image.strip() or trainer


def submit_sim2real_staged_job(
    *,
    run_id: str = "",
    trigger_dataset_uri: str = "",
    trigger_dataset_id: str = DEFAULT_LEROBOT_DATASET_ID,
    s3_bucket: str = "",
    s3_prefix: str = DEFAULT_PREFIX,
    s3_endpoint: str = "",
    k8s_context: str = "",
    registry: str = "",
    orchestrator_image: str = "",
    inner_iterations: int | None = None,
    outer_iterations: int | None = None,
    env_count: int | None = None,
    env_overrides: dict[str, str] | None = None,
    launch_monitor: bool = False,
    plan_only: bool = False,
) -> Sim2RealSubmitResult:
    """Materialize and optionally apply the canonical direct-K8s runbook."""

    del launch_monitor  # The CLI prints the canonical status command after submit.
    operator = load_operator_config()
    root = _repo_root()

    bucket = s3_bucket or operator.bucket
    endpoint = s3_endpoint or operator.endpoint_url
    context = k8s_context or operator.k8s_context
    resolved_registry = (registry or operator.registry).rstrip("/")
    if not resolved_registry:
        raise ValueError("storage.registry is not set in ~/.npa/config.yaml")
    if not _registry_qualified(f"{resolved_registry}/probe:tag"):
        raise ValueError(
            f"Sim2Real registry must be qualified, got {resolved_registry!r}"
        )

    resolved_run_id = normalize_staged_run_id(run_id or os.environ.get("RUN_ID") or "")
    if not resolved_run_id:
        raise ValueError("a non-empty Sim2Real run id is required")

    image_env, resolved_orchestrator = _default_image_env(
        resolved_registry,
        orchestrator_image=orchestrator_image,
    )
    overrides = _resolve_image_overrides(
        {str(key): str(value) for key, value in (env_overrides or {}).items()}
    )
    image_env.update(overrides)
    image_env.update(
        {
            "NPA_SIM2REAL_RUN_ID": resolved_run_id,
            "NPA_SIM2REAL_BUCKET": bucket,
            "NPA_SIM2REAL_PREFIX": s3_prefix,
            "AWS_ENDPOINT_URL": endpoint,
            "S3_ENDPOINT_URL": endpoint,
            "NPA_SIM2REAL_TRIGGER_DATASET_URI": trigger_dataset_uri,
            "NPA_SIM2REAL_TRIGGER_DATASET_ID": trigger_dataset_id,
        }
    )
    if inner_iterations is not None:
        image_env["INNER_ITERATIONS"] = str(inner_iterations)
    if outer_iterations is not None:
        image_env["OUTER_ITERATIONS"] = str(outer_iterations)
    if env_count is not None:
        image_env["NPA_ENV_COUNT"] = str(env_count)

    from npa.workflows.sim2real.materialize import load_runbook_task

    runbook_env = {
        str(key): str(value)
        for key, value in (
            load_runbook_task(default_runbook_path()).get("envs") or {}
        ).items()
    }
    effective_env = {**runbook_env, **image_env}
    _validate_real_runtime_env(effective_env)

    required_image_envs = _required_real_image_envs(effective_env)
    for key in required_image_envs:
        if not _immutable_image(image_env.get(key, "")):
            raise ValueError(
                f"{key} must be a registry-qualified image@sha256 for the real Kubernetes tier; "
                f"got {image_env.get(key, '')!r}"
            )
    if not _immutable_image(resolved_orchestrator):
        raise ValueError(
            "the Sim2Real orchestrator image must be registry-qualified and digest-pinned; "
            f"got {resolved_orchestrator!r}"
        )

    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("could not resolve the exact 40-character repository SHA")
    supplied_sha = image_env.get("NPA_SIM2REAL_SOURCE_SHA", "").strip()
    if supplied_sha and supplied_sha != source_sha:
        raise ValueError(
            "NPA_SIM2REAL_SOURCE_SHA differs from the checked-out exact head"
        )
    image_env["NPA_SIM2REAL_SOURCE_SHA"] = source_sha
    image_env["NPA_SIM2REAL_RUNTIME_IMAGE"] = resolved_orchestrator.removeprefix(
        "docker:"
    )
    image_env["NPA_SIM2REAL_CONTROLLER_SPEC_DIGEST"] = controller_spec_digest(
        default_runbook_path()
    )
    kubeconfig = resolve_kubeconfig(context)

    namespace = image_env.get("NPA_SIM2REAL_K8S_NAMESPACE", "default") or "default"
    materialized = materialize_k8s_job(
        default_runbook_path(),
        run_id=resolved_run_id,
        image=resolved_orchestrator,
        env_overrides=image_env,
        namespace=namespace,
    )
    if not plan_only:
        from npa.workflows.sim2real.registry_auth import (
            ensure_registry_pull_secret_for_images,
        )

        ensure_registry_pull_secret_for_images(
            resolved_orchestrator,
            *(image_env[key] for key in required_image_envs),
            namespace=namespace,
            kubeconfig=str(kubeconfig),
            k8s_context=context,
        )
        from npa.workflows.sim2real.k8s_client import KubernetesJobClient

        client = KubernetesJobClient.from_environment(
            namespace=namespace,
            kubeconfig=str(kubeconfig),
            context=context,
        )
        client.create_or_adopt(
            materialized.manifest,
            run_id=resolved_run_id,
            source_sha=source_sha,
            runtime_image=resolved_orchestrator,
        )
        selected_job_name = materialized.job_name
        manifest_yaml = materialized.to_yaml()
    else:
        selected_job_name = materialized.job_name
        manifest_yaml = materialized.to_yaml()

    manifest_sha256 = hashlib.sha256(manifest_yaml.encode("utf-8")).hexdigest()
    # The manifest can contain private registry/S3 coordinates. Materialize it
    # only long enough to exercise the secure-file contract, then retain its
    # digest as durable evidence. Never return the now-deleted temporary path.
    with _secure_temporary_manifest(
        run_id=resolved_run_id,
        job_name=selected_job_name,
        manifest_yaml=manifest_yaml,
    ):
        pass

    prefix_uri = f"s3://{bucket}/{s3_prefix.rstrip('/')}/{resolved_run_id}/"
    return Sim2RealSubmitResult(
        run_id=resolved_run_id,
        job_name=materialized.job_name if plan_only else selected_job_name,
        k8s_context=context,
        run_prefix_uri=prefix_uri,
        status="planned" if plan_only else "submitted",
        log_path="",
        manifest_sha256=manifest_sha256,
    )


def is_sim2real_runbook(yaml_path: Path) -> bool:
    """True only for the single committed 14-stage Sim2Real runbook."""

    try:
        return yaml_path.resolve() == default_runbook_path().resolve()
    except (OSError, RuntimeError):
        return False


def status_monitor_command(run_id: str) -> str:
    return f"npa workbench workflow status {run_id} --watch"


def submit_sim2real_from_workflow_vars(
    *,
    run_id: str,
    substitutions: dict[str, str],
    s3_bucket: str = "",
    s3_prefix: str = DEFAULT_PREFIX,
    s3_endpoint: str = "",
    registry: str = "",
    orchestrator_image: str = "",
    plan_only: bool = False,
) -> Sim2RealSubmitResult:
    """Submit the runbook using workflow ``--var KEY=VALUE`` substitutions."""

    aliases = {
        "bucket": "NPA_SIM2REAL_BUCKET",
        "prefix": "NPA_SIM2REAL_PREFIX",
        "trigger_uri": "NPA_SIM2REAL_TRIGGER_DATASET_URI",
        "trigger_dataset_id": "NPA_SIM2REAL_TRIGGER_DATASET_ID",
    }
    normalized = dict(substitutions)
    for old, new in aliases.items():
        if old in normalized and new not in normalized:
            normalized[new] = normalized[old]
    bucket = normalized.get("NPA_SIM2REAL_BUCKET") or s3_bucket
    prefix = normalized.get("NPA_SIM2REAL_PREFIX") or s3_prefix or DEFAULT_PREFIX
    endpoint = (
        normalized.get("AWS_ENDPOINT_URL")
        or normalized.get("S3_ENDPOINT_URL")
        or s3_endpoint
    )
    trigger_uri = (
        normalized.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
        or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
        or os.environ.get("TRIGGER_DATASET_URI", "")
    )
    trigger_id = (
        normalized.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
        or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
        or os.environ.get("TRIGGER_DATASET_ID")
        or DEFAULT_LEROBOT_DATASET_ID
    )
    inner = normalized.get("INNER_ITERATIONS")
    outer = normalized.get("OUTER_ITERATIONS")
    env_count = normalized.get("NPA_ENV_COUNT")
    return submit_sim2real_staged_job(
        run_id=run_id,
        trigger_dataset_uri=trigger_uri,
        trigger_dataset_id=trigger_id,
        s3_bucket=bucket,
        s3_prefix=prefix,
        s3_endpoint=endpoint,
        registry=registry,
        orchestrator_image=orchestrator_image,
        inner_iterations=int(inner) if inner else None,
        outer_iterations=int(outer) if outer else None,
        env_count=int(env_count) if env_count else None,
        env_overrides=normalized,
        launch_monitor=False,
        plan_only=plan_only,
    )
