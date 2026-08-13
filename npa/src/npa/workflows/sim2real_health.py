"""Preflight diagnostics for the Sim2Real VLM-to-RL workflow.

This module is the single source of truth for ``npa workbench health sim2real``.
It turns the recurring cold-start blockers into explicit PASS/WARN/FAIL/SKIP
checks that a customer can run before launching anything, instead of discovering
them mid-run.

Every check is a pure function that takes the resolved config plus an injectable
probe. The CLI wires real probes (S3, registry, kubectl); unit tests inject
fakes. Nothing here imports GPU-heavy packages or touches infrastructure at
import time.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from npa.guardrails.skypilot import unresolved_image_placeholders
from npa.workflows.sim2real.models import Sim2RealLoopConfig

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

CLI_MODULE = "npa.cli.workbench.sim2real"
CLI_CALLBACK = "run_command"
RUNBOOK_YAML = Path("npa/workflows/workbench/npa-workflows/sim2real.yaml")


@dataclass(frozen=True)
class Seam:
    """One bring-your-own plug point expressed across all three tiers.

    ``config_field`` doubles as the SDK keyword argument: ``sim2real.run`` and
    ``build_config_from_env`` accept overrides keyed by config field name.
    """

    name: str
    config_field: str
    cli_flag: str
    yaml_env: str
    required: bool = False


# Canonical BYO seams for the headline sim2real workflow. The order is the order
# a customer should reason about them: where data lands, what runs, how big.
SIM2REAL_SEAMS: tuple[Seam, ...] = (
    Seam("s3_endpoint", "s3_endpoint", "--s3-endpoint", "AWS_ENDPOINT_URL"),
    Seam("s3_bucket", "s3_bucket", "--s3-bucket", "NPA_SIM2REAL_BUCKET", required=True),
    Seam("s3_prefix", "s3_prefix", "--s3-prefix", "NPA_SIM2REAL_PREFIX"),
    Seam(
        "trigger_dataset_uri",
        "trigger_dataset_uri",
        "--trigger-dataset-uri",
        "NPA_SIM2REAL_TRIGGER_DATASET_URI",
    ),
    Seam(
        "trigger_dataset_id",
        "trigger_dataset_id",
        "--trigger-dataset-id",
        "NPA_SIM2REAL_TRIGGER_DATASET_ID",
    ),
    Seam("assets_uri", "assets_uri", "--assets-uri", "ASSETS_URI"),
    Seam("scene_spec_uri", "scene_spec_uri", "--scene-spec-uri", "SCENE_SPEC_URI"),
    Seam("augment_image", "augment_image", "--augment-image", "AUGMENT_IMAGE"),
    Seam("policy_image", "policy_image", "--policy-image", "POLICY_IMAGE"),
    Seam("trainer_image", "trainer_image", "--trainer-image", "TRAINER_IMAGE"),
    Seam("vlm_image", "vlm_image", "--vlm-image", "VLM_IMAGE"),
    Seam("eval_image", "eval_image", "--eval-image", "EVAL_IMAGE"),
    Seam(
        "k8s_isaac_cache_pvc",
        "k8s_isaac_cache_pvc",
        "--k8s-isaac-cache-pvc",
        "NPA_SIM2REAL_ISAAC_CACHE_PVC",
        required=True,
    ),
    Seam("vlm_model", "vlm_model", "--vlm-model", "VLM_MODEL"),
    Seam("threshold", "threshold", "--threshold", "SUCCESS_THRESHOLD"),
    Seam(
        "inner_iterations", "inner_iterations", "--inner-iterations", "INNER_ITERATIONS"
    ),
    Seam(
        "outer_iterations", "outer_iterations", "--outer-iterations", "OUTER_ITERATIONS"
    ),
    Seam(
        "loop_of_loops_iterations",
        "loop_of_loops_iterations",
        "--loop-of-loops-iterations",
        "LOOP_OF_LOOPS_ITERATIONS",
    ),
    Seam("rollout_count", "rollout_count", "--rollout-count", "ROLLOUT_COUNT"),
    Seam(
        "steps_per_rollout",
        "steps_per_rollout",
        "--steps-per-rollout",
        "STEPS_PER_ROLLOUT",
    ),
    Seam(
        "heldout_env_count",
        "heldout_env_count",
        "--heldout-env-count",
        "HELDOUT_ENV_COUNT",
    ),
)

# Container images a real run must be able to pull.
IMAGE_FIELDS: tuple[str, ...] = (
    "augment_image",
    "policy_image",
    "trainer_image",
    "vlm_image",
    "eval_image",
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single preflight check."""

    name: str
    status: str
    summary: str
    remedy: str = ""
    details: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "remedy": self.remedy,
            "details": list(self.details),
        }


@dataclass
class DoctorProbes:
    """Injectable side-effecting dependencies for infrastructure checks.

    Defaults are ``None`` so the engine stays pure and import-safe. The CLI fills
    these with real implementations; tests pass fakes.
    """

    s3_client_factory: Callable[[], Any] | None = None
    image_inspector: Callable[[str], bool | None] | None = None
    credentials: Any | None = None
    kube_runner: Callable[[list[str]], "KubeResult"] | None = None


@dataclass(frozen=True)
class KubeResult:
    """Result of one kubectl-style invocation."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


# Coherence -----------------------------------------------------------------


def coherence_failures(repo_root: Path) -> list[str]:
    """Return canonical compositional-workflow contract failures."""

    path = repo_root / RUNBOOK_YAML
    if not path.is_file():
        return [f"canonical workflow missing: {RUNBOOK_YAML}"]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if (
        payload.get("apiVersion") != "npa.workflow/v0.0.1"
        or payload.get("kind") != "Workflow"
    ):
        failures.append("canonical YAML is not an npa.workflow/v0.0.1 Workflow")
    config = dict(payload.get("config") or {})
    required_config = {
        "source_sha",
        "bucket",
        "root_uri",
        "trigger_uri",
        "controller_image",
        "transfer_image",
        "envgen_image",
        "reason_image",
        "isaac_image",
        "viewer_image",
        "isaac_cache_pvc",
        "gpu_queue",
        "gpu_priority_class",
    }
    missing_config = sorted(required_config - set(config))
    if missing_config:
        failures.append(f"canonical config inputs missing: {missing_config}")
    stages: set[int] = set()
    commands: list[str] = []
    for state in (payload.get("states") or {}).values():
        argv = list((state.get("run") or {}).get("argv") or [])
        commands.extend(str(item) for item in argv)
        if "--stage" in argv:
            stages.add(int(argv[argv.index("--stage") + 1]))
    if stages != set(range(1, 15)):
        failures.append(f"canonical stage adapters are incomplete: {sorted(stages)}")
    joined = " ".join(commands)
    for forbidden in (
        "run_preamble",
        "run_inner_loop",
        "run_single_outer_iteration",
        "run_finalize",
        "k8s_submit",
    ):
        if forbidden in joined:
            failures.append(
                f"canonical workflow calls retired orchestrator: {forbidden}"
            )
    return failures


def check_coherence(repo_root: Path) -> CheckResult:
    failures = coherence_failures(repo_root)
    if failures:
        return CheckResult(
            name="compositional-workflow-coherence",
            status=FAIL,
            summary=f"{len(failures)} canonical workflow contract failure(s).",
            remedy=(
                "Keep all 14 stages and immutable runtime inputs on the standard "
                "npa.workflow surface. See npa/workflows/workbench/sim2real/README.md."
            ),
            details=tuple(failures),
        )
    return CheckResult(
        name="compositional-workflow-coherence",
        status=PASS,
        summary="Canonical npa.workflow exposes all 14 stateless stage adapters and runtime inputs.",
    )


# Config --------------------------------------------------------------------


def check_config(config: Sim2RealLoopConfig) -> CheckResult:
    """Validate the schema and confirm the required BYO seams resolve."""

    try:
        config.validate()
    except Exception as exc:  # noqa: BLE001 - surfaced as a FAIL line, not a crash
        return CheckResult(
            name="config",
            status=FAIL,
            summary="Config failed schema validation.",
            remedy="Fix the reported value and rerun.",
            details=(str(exc),),
        )

    field_values = {f.name: getattr(config, f.name) for f in fields(Sim2RealLoopConfig)}
    missing_required = [
        seam.name
        for seam in SIM2REAL_SEAMS
        if seam.required and not str(field_values.get(seam.config_field, "")).strip()
    ]
    if missing_required:
        return CheckResult(
            name="config",
            status=FAIL,
            summary="Required BYO seam(s) are not set.",
            remedy=(
                "Set --s3-bucket / NPA_SIM2REAL_BUCKET so artifacts and sibling-Job "
                "I/O have a destination."
            ),
            details=tuple(f"missing: {name}" for name in missing_required),
        )

    soft_missing: list[str] = []
    if not config.trigger_dataset_uri.strip():
        soft_missing.append(
            "trigger_dataset_uri (derived from bucket+run-id; set it to pin the trigger path)"
        )
    if not config.assets_uri.strip() and not config.scene_spec_uri.strip():
        soft_missing.append(
            "assets_uri / scene_spec_uri (Stage 2 runs as a documented external stub without them)"
        )
    if soft_missing:
        return CheckResult(
            name="config",
            status=WARN,
            summary="Schema valid; some optional seams use derived defaults.",
            remedy="Set these explicitly for a reproducible BYO run.",
            details=tuple(soft_missing),
        )
    return CheckResult(
        name="config",
        status=PASS,
        summary="Schema valid and required BYO seams resolve.",
    )


# S3 ------------------------------------------------------------------------


def check_s3(config: Sim2RealLoopConfig, *, probes: DoctorProbes) -> CheckResult:
    """Confirm the configured S3 endpoint and bucket are reachable with creds."""

    creds = probes.credentials
    have_keys = bool(getattr(creds, "s3_access_key_id", "")) and bool(
        getattr(creds, "s3_secret_access_key", "")
    )
    if not config.s3_endpoint.strip():
        return CheckResult(
            name="s3",
            status=SKIP,
            summary="No S3 endpoint configured.",
            remedy="Set --s3-endpoint / AWS_ENDPOINT_URL to a non-default S3-compatible endpoint.",
        )
    if not config.s3_bucket.strip():
        return CheckResult(
            name="s3",
            status=SKIP,
            summary="No S3 bucket configured.",
            remedy="Set --s3-bucket / NPA_SIM2REAL_BUCKET.",
        )
    if not have_keys:
        return CheckResult(
            name="s3",
            status=SKIP,
            summary="No S3 credentials available to probe reachability.",
            remedy="Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or storage.* in ~/.npa/credentials.yaml.",
        )
    if probes.s3_client_factory is None:
        return CheckResult(
            name="s3",
            status=SKIP,
            summary="No S3 client available.",
        )
    bucket = config.s3_bucket.strip().removeprefix("s3://").strip("/")
    try:
        client = probes.s3_client_factory()
        client.list_checkpoints(f"s3://{bucket}/")
    except Exception as exc:  # noqa: BLE001 - reachability failure is the signal
        return CheckResult(
            name="s3",
            status=FAIL,
            summary=f"Cannot reach bucket {bucket!r} at the configured endpoint.",
            remedy=(
                "Verify the endpoint host, bucket name, and credentials. A wrong "
                "endpoint region is the most common cause."
            ),
            details=(_short(str(exc)),),
        )
    return CheckResult(
        name="s3",
        status=PASS,
        summary=f"Bucket {bucket!r} is reachable at the configured endpoint.",
    )


# Registry / image pull -----------------------------------------------------


def _looks_registry_qualified(image: str) -> bool:
    """Return true when an image reference carries an explicit registry host."""

    first = image.split("/", 1)[0]
    return "/" in image and ("." in first or ":" in first)


def check_registry(config: Sim2RealLoopConfig, *, probes: DoctorProbes) -> CheckResult:
    """Confirm the configured workflow images are pullable."""

    images = [getattr(config, name) for name in IMAGE_FIELDS]
    not_actionable = [
        img
        for img in images
        if unresolved_image_placeholders(img) or not _looks_registry_qualified(img)
    ]
    if not_actionable:
        return CheckResult(
            name="registry",
            status=WARN,
            summary="Some images are not fully qualified for a pull check.",
            remedy=(
                "Set NPA_REGISTRY (or NPA_REGISTRY_ID) or pass fully-qualified "
                "<registry>/<image>:<tag> values so the agent-sa pull path can be verified."
            ),
            details=tuple(sorted(set(not_actionable))),
        )
    if probes.image_inspector is None:
        return CheckResult(
            name="registry",
            status=SKIP,
            summary="No registry inspection tool available (install crane, skopeo, or docker).",
            remedy="Confirm the agent-sa pull path can reach these images before launch.",
        )

    unreachable: list[str] = []
    skipped: list[str] = []
    for image in images:
        result = probes.image_inspector(image)
        if result is None:
            skipped.append(image)
        elif result is False:
            unreachable.append(image)
    if unreachable:
        return CheckResult(
            name="registry",
            status=FAIL,
            summary=f"{len(unreachable)} image(s) are not pullable.",
            remedy=(
                "Push the image or refresh the registry pull secret (the "
                "npa-nebius-registry / agent-sa pull path expires silently)."
            ),
            details=tuple(unreachable),
        )
    if skipped and len(skipped) == len(images):
        return CheckResult(
            name="registry",
            status=SKIP,
            summary="No registry inspection tool available.",
            remedy="Confirm the agent-sa pull path can reach these images before launch.",
        )
    return CheckResult(
        name="registry",
        status=PASS,
        summary=f"All {len(images) - len(skipped)} configured image(s) are pullable.",
    )


# Tokens --------------------------------------------------------------------


def check_tokens(config: Sim2RealLoopConfig, *, probes: DoctorProbes) -> CheckResult:
    """Confirm gated-repo tokens are present for the configured VLM path."""

    creds = probes.credentials
    have_hf = bool(getattr(creds, "hf_token", ""))
    have_ngc = bool(getattr(creds, "ngc_api_key", ""))
    missing: list[str] = []
    if not have_hf:
        missing.append(
            "HF_TOKEN (gated VLM / model repos used by the eval and VLM images)"
        )
    if not have_ngc:
        missing.append("NGC_API_KEY (NGC-hosted base images and weights)")
    if missing:
        return CheckResult(
            name="tokens",
            status=WARN,
            summary="Gated-repo tokens are missing.",
            remedy="Add tokens to ~/.npa/credentials.yaml or export them before launch.",
            details=tuple(missing),
        )
    return CheckResult(
        name="tokens",
        status=PASS,
        summary="HF and NGC tokens are present.",
    )


# Cluster / GPU -------------------------------------------------------------


def check_cluster(config: Sim2RealLoopConfig, *, probes: DoctorProbes) -> CheckResult:
    """Confirm the active cluster is reachable and has schedulable GPUs.

    Surfaces the two recurring traps up front: an unpinned kube context pointing
    at the wrong cluster, and a schedulable-GPU count of zero (capacity that is
    on-demand-zero, or GPU-count-vs-node-count confusion).
    """

    runner = probes.kube_runner
    if runner is None:
        return CheckResult(
            name="cluster",
            status=SKIP,
            summary="kubectl is not available to probe the cluster.",
            remedy="Install kubectl and select the cluster context to enable this check.",
        )

    context = ""
    ctx_result = runner(["config", "current-context"])
    if ctx_result.returncode == 0:
        context = ctx_result.stdout.strip()
    if not context:
        return CheckResult(
            name="cluster",
            status=FAIL,
            summary="No active kube context is selected.",
            remedy=(
                "Select the target cluster context explicitly (an unpinned context "
                "routes the run to the wrong cluster). Set --k8s-context to pin it."
            ),
        )

    namespace = config.k8s_namespace or "default"
    can_i = runner(["auth", "can-i", "create", "pods", "-n", namespace])
    if can_i.returncode != 0 or can_i.stdout.strip().lower() != "yes":
        return CheckResult(
            name="cluster",
            status=FAIL,
            summary=f"Cannot create pods in namespace {namespace!r} on context {context!r}.",
            remedy=(
                "Refresh the managed-Kubernetes credentials and confirm the context "
                "and namespace. A stale NEBIUS_IAM_TOKEN in the environment shadows "
                "the kubeconfig exec plugin (npa retries once without it); clear it "
                "with 'unset NEBIUS_IAM_TOKEN'. 'sky check' HTTP 403 anonymous has "
                "the same root cause."
            ),
            details=(_short(can_i.stderr or can_i.stdout),),
        )

    service_account = config.k8s_service_account or "agent-sa"
    service_account_user = f"system:serviceaccount:{namespace}:{service_account}"
    controller_patch = runner(
        [
            "auth",
            "can-i",
            "patch",
            "jobs.batch",
            f"--as={service_account_user}",
            "-n",
            namespace,
        ]
    )
    if (
        controller_patch.returncode != 0
        or controller_patch.stdout.strip().lower() != "yes"
    ):
        return CheckResult(
            name="cluster",
            status=FAIL,
            summary=(
                f"Service account {service_account!r} cannot patch Jobs in "
                f"namespace {namespace!r}."
            ),
            remedy=(
                "Grant the Sim2Real controller Role the 'patch' verb on "
                "batch/jobs. Durable reconciliation records structured heartbeats "
                "and adopts exact-identity Jobs through the Kubernetes API."
            ),
            details=(_short(controller_patch.stderr or controller_patch.stdout),),
        )

    kueue_observe = runner(
        [
            "auth",
            "can-i",
            "list",
            "workloads.kueue.x-k8s.io",
            f"--as={service_account_user}",
            "-n",
            namespace,
        ]
    )
    if kueue_observe.returncode != 0 or kueue_observe.stdout.strip().lower() != "yes":
        return CheckResult(
            name="cluster",
            status=FAIL,
            summary=(
                f"Service account {service_account!r} cannot list Kueue Workloads "
                f"in namespace {namespace!r}."
            ),
            remedy=(
                "Grant the Sim2Real controller Role the 'list' verb on "
                "kueue.x-k8s.io/workloads. Durable reconciliation must observe "
                "the generated Workload admission, flavor, and terminal state."
            ),
            details=(_short(kueue_observe.stderr or kueue_observe.stdout),),
        )

    cache_pvc = config.k8s_isaac_cache_pvc.strip()
    pvc_result = runner(["get", "pvc", cache_pvc, "-n", namespace, "-o", "json"])
    if pvc_result.returncode != 0:
        return CheckResult(
            name="cluster",
            status=FAIL,
            summary=(
                f"Isaac runtime-cache PVC {cache_pvc!r} is not readable in "
                f"namespace {namespace!r}."
            ),
            remedy=(
                "Create the shared cache PVC, warm it on a CPU node with the exact "
                "Isaac image, and rerun preflight. GPU Jobs never fetch dependencies."
            ),
            details=(_short(pvc_result.stderr or pvc_result.stdout),),
        )
    try:
        import json

        pvc = json.loads(pvc_result.stdout)
    except (TypeError, ValueError):
        pvc = {}
    spec = pvc.get("spec") or {}
    status = pvc.get("status") or {}
    access_modes = {str(item) for item in spec.get("accessModes") or []}
    if (
        status.get("phase") != "Bound"
        or not str(spec.get("volumeName") or "").strip()
        or "ReadWriteMany" not in access_modes
    ):
        return CheckResult(
            name="cluster",
            status=FAIL,
            summary=(
                f"Isaac runtime-cache PVC {cache_pvc!r} is not a Bound RWX claim."
            ),
            remedy=(
                "Use a Bound ReadWriteMany volume so one CPU warm Job can publish "
                "the content-addressed closure for read-only GPU consumers."
            ),
            details=(
                f"phase={status.get('phase')!r}",
                f"volumeName={spec.get('volumeName')!r}",
                f"accessModes={sorted(access_modes)!r}",
            ),
        )

    gpu_resource = config.k8s_gpu_resource or "nvidia.com/gpu"
    nodes = runner(["get", "nodes", "-o", "json"])
    if nodes.returncode != 0:
        return CheckResult(
            name="cluster",
            status=WARN,
            summary=f"Reachable on context {context!r}, but node listing failed.",
            remedy="Confirm RBAC allows listing nodes to verify schedulable GPU capacity.",
            details=(_short(nodes.stderr or nodes.stdout),),
        )
    node_count, gpu_total = _count_schedulable_gpus(nodes.stdout, gpu_resource)
    if gpu_total <= 0:
        return CheckResult(
            name="cluster",
            status=FAIL,
            summary=(
                f"Context {context!r} has {node_count} node(s) but 0 schedulable "
                f"{gpu_resource}."
            ),
            remedy=(
                "Ask the operator to provision a GPU node group. On-demand capacity "
                "for some accelerators is zero by default."
            ),
        )
    return CheckResult(
        name="cluster",
        status=PASS,
        summary=(
            f"Context {context!r}: {gpu_total} schedulable {gpu_resource} across "
            f"{node_count} node(s); Isaac cache PVC {cache_pvc!r} is Bound RWX."
        ),
    )


def _count_schedulable_gpus(nodes_json: str, gpu_resource: str) -> tuple[int, int]:
    import json

    try:
        payload = json.loads(nodes_json)
    except (json.JSONDecodeError, TypeError):
        return (0, 0)
    items = payload.get("items") or []
    total = 0
    for node in items:
        allocatable = (node.get("status") or {}).get("allocatable") or {}
        raw = allocatable.get(gpu_resource)
        if raw is None:
            continue
        try:
            total += int(raw)
        except (TypeError, ValueError):
            continue
    return (len(items), total)


# Orchestration -------------------------------------------------------------

ALL_CHECKS: tuple[str, ...] = (
    "config",
    "coherence",
    "s3",
    "registry",
    "tokens",
    "cluster",
)


def run_preflight(
    config: Sim2RealLoopConfig,
    *,
    repo_root: Path,
    probes: DoctorProbes,
    checks: Iterable[str] | None = None,
) -> list[CheckResult]:
    """Run the selected checks and return their results in display order."""

    selected = tuple(checks) if checks is not None else ALL_CHECKS
    results: list[CheckResult] = []
    if "config" in selected:
        results.append(check_config(config))
    if "coherence" in selected:
        results.append(check_coherence(repo_root))
    if "s3" in selected:
        results.append(check_s3(config, probes=probes))
    if "registry" in selected:
        results.append(check_registry(config, probes=probes))
    if "tokens" in selected:
        results.append(check_tokens(config, probes=probes))
    if "cluster" in selected:
        results.append(check_cluster(config, probes=probes))
    return results


def has_failure(results: list[CheckResult]) -> bool:
    return any(result.status == FAIL for result in results)


_STATUS_ICON = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}


def format_check_report(results: list[CheckResult], *, output_json: bool) -> str:
    """Render a list of CheckResults as the shared preflight report.

    Single source of truth for how ``npa`` preflight surfaces (workbench health
    and agent) print their PASS/WARN/FAIL/SKIP output, so they stay aligned.
    """

    import json

    if output_json:
        return json.dumps(
            {
                "checks": [result.as_dict() for result in results],
                "ok": not has_failure(results),
            },
            indent=2,
            sort_keys=True,
        )
    lines: list[str] = []
    for result in results:
        lines.append(
            f"[{_STATUS_ICON.get(result.status, result.status)}] "
            f"{result.name}: {result.summary}"
        )
        for detail in result.details:
            lines.append(f"        - {detail}")
        if result.remedy and result.status in {FAIL, WARN, SKIP}:
            lines.append(f"        fix: {result.remedy}")
    counts = {status: 0 for status in (PASS, WARN, FAIL, SKIP)}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    lines.append(
        f"summary: {counts[PASS]} pass, {counts[WARN]} warn, "
        f"{counts[FAIL]} fail, {counts[SKIP]} skip"
    )
    return "\n".join(lines)


def _short(text: str, limit: int = 240) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "…"
    return collapsed


__all__ = [
    "ALL_CHECKS",
    "CheckResult",
    "DoctorProbes",
    "IMAGE_FIELDS",
    "KubeResult",
    "SIM2REAL_SEAMS",
    "Seam",
    "check_cluster",
    "check_coherence",
    "check_config",
    "check_registry",
    "check_s3",
    "check_tokens",
    "coherence_failures",
    "format_check_report",
    "has_failure",
    "run_preflight",
]
