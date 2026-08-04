"""Submit the canonical Sim2Real runbook as a direct Kubernetes Job."""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from npa.deploy.images import container_image_for_tool
from npa.workflows.sim2real.constants import DEFAULT_PREFIX
from npa.workflows.sim2real.materialize import default_runbook_path, materialize_k8s_job
from npa.workflows.sim2real.monitor import (
    load_operator_config,
    normalize_staged_run_id,
    resolve_kubeconfig,
)
from npa.workflows.sim2real.gpu_fallback import run_gpu_job_with_fallback


@dataclass(frozen=True)
class Sim2RealSubmitResult:
    run_id: str
    job_name: str
    k8s_context: str
    run_prefix_uri: str
    status: str = "submitted"
    log_path: str = ""
    manifest_path: str = ""


_REQUIRED_REAL_IMAGE_ENVS = (
    "AUGMENT_IMAGE",
    "ENVGEN_IMAGE",
    "POLICY_IMAGE",
    "TRAINER_IMAGE",
    "VLM_IMAGE",
    "EVAL_IMAGE",
    "ISAAC_IMAGE",
)

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
    if "/" not in ref:
        return False
    host, leaf = ref.split("/", 1)
    return bool(("." in host or ":" in host or host == "localhost") and (":" in leaf or "@" in leaf))


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
        "HELDOUT_ENV_COUNT",
        "NPA_ENV_COUNT",
    ):
        integer(name)
    integer("NPA_SIM2REAL_HELDOUT_EVAL_LIMIT", minimum=0)
    integer("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", minimum=0)
    integer("NPA_BYO_ISAAC_JOB_TIMEOUT_S", minimum=0)

    threshold = float(values.get("SUCCESS_THRESHOLD", "0.50"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"SUCCESS_THRESHOLD must be in [0, 1], got {threshold}")
    distance = float(values.get("NPA_BYO_ISAAC_SUCCESS_DIST_M", "0.05"))
    if not 0.001 <= distance <= 10.0:
        raise ValueError(
            "NPA_BYO_ISAAC_SUCCESS_DIST_M must be in [0.001, 10] metres, "
            f"got {distance}"
        )
    early_exit = values.get("NPA_SIM2REAL_EARLY_EXIT", "0").strip().lower()
    if early_exit not in {"", "0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError("NPA_SIM2REAL_EARLY_EXIT must be boolean")
    for name in _REQUIRED_REAL_COMMAND_ENVS:
        if not values.get(name, "").strip():
            raise ValueError(
                f"{name} must invoke a real external component in the canonical real tier"
            )
    for name in ("OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA"):
        if values.get(name, "").strip().upper() != "YES":
            raise ValueError(
                f"{name}=YES is required after the operator accepts the NVIDIA Isaac licence"
            )


def _default_image_env(registry: str, *, orchestrator_image: str = "") -> tuple[dict[str, str], str]:
    trainer = container_image_for_tool("lerobot-vlm-rl", registry=registry)
    vlm = container_image_for_tool("cosmos3-reason", registry=registry)
    images = {
        "NPA_REGISTRY": registry,
        "AUGMENT_IMAGE": container_image_for_tool("cosmos2-transfer", registry=registry),
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
        "NPA_RERUN_VIEWER_IMAGE": container_image_for_tool("rerun-viewer", registry=registry),
    }
    return images, orchestrator_image.strip() or trainer


def _source_tarball_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in tarinfo.name or tarinfo.name.endswith(".pyc"):
        return None
    return tarinfo


def _stage_orchestrator_source(
    *,
    root: Path,
    run_id: str,
    bucket: str,
    prefix: str,
    endpoint: str,
) -> str:
    """Upload current checkout code so the pre-push Job runs this exact branch."""

    from npa.clients.credentials import load_credentials
    from npa.clients.storage import StorageClient

    credentials = load_credentials()
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or credentials.s3_access_key_id
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or credentials.s3_secret_access_key
    if not access_key or not secret_key:
        raise ValueError("S3 HMAC credentials are required to stage the Sim2Real source")

    with tempfile.TemporaryDirectory(prefix="npa-sim2real-submit-") as temporary:
        tarball = Path(temporary) / "npa-source.tgz"
        with tarfile.open(tarball, "w:gz") as archive:
            archive.add(
                root / "npa" / "src",
                arcname="npa/src",
                filter=_source_tarball_filter,
            )
            archive.add(
                root / "npa" / "pyproject.toml",
                arcname="npa/pyproject.toml",
                filter=_source_tarball_filter,
            )
        destination = (
            f"s3://{bucket}/{prefix.strip('/')}/{run_id}/source/"
            f"orchestrator-{run_id}.tgz"
        )
        client = StorageClient(
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        return client.upload_file(str(tarball), destination)


def _apply_manifest(
    manifest_path: Path,
    *,
    kubeconfig: Path,
    context: str,
    namespace: str,
) -> None:
    from npa.clients.nebius_auth import strip_ambient_token_env

    command = [
        "kubectl",
        "--kubeconfig",
        str(kubeconfig),
        "--context",
        context,
        "-n",
        namespace,
        "apply",
        "-f",
        str(manifest_path),
    ]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=strip_ambient_token_env(os.environ),
        check=False,
    )
    if proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"sim2real K8s submit failed:\n{output}")


def _direct_kubectl(
    args: list[str],
    *,
    kubeconfig: Path,
    context: str,
    namespace: str,
    stdin: str | None = None,
    timeout_s: int = 300,
) -> subprocess.CompletedProcess[str]:
    from npa.clients.nebius_auth import strip_ambient_token_env

    command = [
        "kubectl",
        "--kubeconfig",
        str(kubeconfig),
        "--context",
        context,
        "-n",
        namespace,
        *args,
    ]
    return subprocess.run(
        command,
        input=stdin,
        capture_output=True,
        text=True,
        env=strip_ambient_token_env(os.environ),
        timeout=timeout_s,
        check=False,
    )


def submit_sim2real_staged_job(
    *,
    run_id: str = "",
    trigger_dataset_uri: str = "",
    trigger_dataset_id: str = "lerobot/pusht",
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
        raise ValueError(f"Sim2Real registry must be qualified, got {resolved_registry!r}")

    resolved_run_id = normalize_staged_run_id(run_id or os.environ.get("RUN_ID") or "")
    if not resolved_run_id:
        raise ValueError("a non-empty Sim2Real run id is required")

    image_env, resolved_orchestrator = _default_image_env(
        resolved_registry,
        orchestrator_image=orchestrator_image,
    )
    overrides = {str(key): str(value) for key, value in (env_overrides or {}).items()}
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
        for key, value in (load_runbook_task(default_runbook_path()).get("envs") or {}).items()
    }
    _validate_real_runtime_env({**runbook_env, **image_env})

    for key in _REQUIRED_REAL_IMAGE_ENVS:
        if not _registry_qualified(image_env.get(key, "")):
            raise ValueError(
                f"{key} must be a registry-qualified image for the real Kubernetes tier; "
                f"got {image_env.get(key, '')!r}"
            )
    if not _registry_qualified(resolved_orchestrator):
        raise ValueError(
            "the Sim2Real orchestrator image must be registry-qualified; "
            f"got {resolved_orchestrator!r}"
        )

    kubeconfig = resolve_kubeconfig(context)
    if plan_only:
        image_env.setdefault(
            "NPA_SIM2REAL_SOURCE_TARBALL_URI",
            f"s3://{bucket}/{s3_prefix.strip('/')}/{resolved_run_id}/source/"
            f"orchestrator-{resolved_run_id}.tgz",
        )
    else:
        image_env["NPA_SIM2REAL_SOURCE_TARBALL_URI"] = _stage_orchestrator_source(
            root=root,
            run_id=resolved_run_id,
            bucket=bucket,
            prefix=s3_prefix,
            endpoint=endpoint,
        )

    namespace = image_env.get("NPA_SIM2REAL_K8S_NAMESPACE", "default") or "default"
    materialized = materialize_k8s_job(
        default_runbook_path(),
        run_id=resolved_run_id,
        image=resolved_orchestrator,
        env_overrides=image_env,
        namespace=namespace,
    )
    manifest_root = Path("/tmp/sim2real-cluster")
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{materialized.job_name}.yaml"
    manifest_path.write_text(materialized.to_yaml(), encoding="utf-8")
    manifest_path.chmod(0o600)

    if not plan_only:
        from npa.workflows.sim2real.registry_auth import ensure_registry_pull_secret_for_images

        ensure_registry_pull_secret_for_images(
            resolved_orchestrator,
            *(image_env[key] for key in _REQUIRED_REAL_IMAGE_ENVS),
            image_env["NPA_RERUN_VIEWER_IMAGE"],
            namespace=namespace,
            kubeconfig=str(kubeconfig),
            k8s_context=context,
        )
        import yaml

        def manifest_factory(product: str, candidate_job_name: str) -> dict[str, object]:
            candidate_env = dict(image_env)
            candidate_env["NPA_SIM2REAL_K8S_GPU_PRODUCT"] = product
            candidate = materialize_k8s_job(
                default_runbook_path(),
                run_id=resolved_run_id,
                image=resolved_orchestrator,
                env_overrides=candidate_env,
                namespace=namespace,
            )
            payload = yaml.safe_load(candidate.to_yaml())
            payload["metadata"]["name"] = candidate_job_name
            return payload

        def kubectl(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            return _direct_kubectl(
                args,
                kubeconfig=kubeconfig,
                context=context,
                namespace=namespace,
                **kwargs,
            )

        placement = run_gpu_job_with_fallback(
            kubectl=kubectl,
            manifest_factory=manifest_factory,
            base_job_name=materialized.job_name,
            namespace=namespace,
            image=resolved_orchestrator,
            preferred_product=image_env.get(
                "NPA_SIM2REAL_K8S_GPU_PRODUCT",
                "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            ),
            explicit_candidates=image_env.get("NPA_SIM2REAL_K8S_GPU_CANDIDATES", ""),
            workload="isaac",
            gpu_resource=image_env.get("NPA_SIM2REAL_K8S_GPU_RESOURCE", "nvidia.com/gpu"),
            gpu_count=1,
            timeout_s=60,
            wait_for_completion=False,
        )
        image_env["NPA_SIM2REAL_K8S_GPU_PRODUCT"] = str(placement["selected_product"])
        materialized = materialize_k8s_job(
            default_runbook_path(),
            run_id=resolved_run_id,
            image=resolved_orchestrator,
            env_overrides=image_env,
            namespace=namespace,
        )
        selected_payload = yaml.safe_load(materialized.to_yaml())
        selected_job_name = str(placement["job_name"])
        selected_payload["metadata"]["name"] = selected_job_name
        manifest_path = manifest_root / f"{selected_job_name}.yaml"
        manifest_path.write_text(yaml.safe_dump(selected_payload, sort_keys=False), encoding="utf-8")
        manifest_path.chmod(0o600)

    prefix_uri = f"s3://{bucket}/{s3_prefix.rstrip('/')}/{resolved_run_id}/"
    return Sim2RealSubmitResult(
        run_id=resolved_run_id,
        job_name=materialized.job_name if plan_only else selected_job_name,
        k8s_context=context,
        run_prefix_uri=prefix_uri,
        status="planned" if plan_only else "submitted",
        log_path=f"/tmp/sim2real-cluster/{resolved_run_id}.log",
        manifest_path=str(manifest_path),
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
    endpoint = normalized.get("AWS_ENDPOINT_URL") or normalized.get("S3_ENDPOINT_URL") or s3_endpoint
    trigger_uri = (
        normalized.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
        or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
        or os.environ.get("TRIGGER_DATASET_URI", "")
    )
    trigger_id = (
        normalized.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
        or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
        or os.environ.get("TRIGGER_DATASET_ID")
        or "lerobot/pusht"
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
