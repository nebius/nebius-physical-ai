"""Fail-fast prerequisites for the canonical compositional Sim2Real workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
import re
from typing import Any

from npa.orchestration.npa_workflow.kubernetes_prerequisites import (
    format_cpu_memory_requirement,
    ready_schedulable_cpu_nodes,
)
from npa.orchestration.skypilot.controller import (
    DEFAULT_K8S_CONTROLLER_CPUS,
    DEFAULT_K8S_CONTROLLER_MEMORY_GB,
)
from npa.workflows.sim2real.constants import DEFAULT_TOKEN_FACTORY_MODEL


Issue = tuple[str, str]
_IMAGE_KEYS = (
    "controller_image",
    "cosmos3_image",
    "envgen_image",
    "isaac_image",
    "viewer_image",
)
_REQUIRED_SECRET_ENVS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "NEBIUS_TOKEN_FACTORY_KEY",
)
_DIGEST_IMAGE = re.compile(r"^[^/\s]+/.+@sha256:[0-9a-fA-F]{64}$")
_SIM2REAL_CPU_MILLICORES = (8 + DEFAULT_K8S_CONTROLLER_CPUS) * 1000
_SIM2REAL_MEMORY_BYTES = (32 + DEFAULT_K8S_CONTROLLER_MEMORY_GB) * 1024**3
_WAIT_FOR_FIRST_CONSUMER = "WaitForFirstConsumer"


def cpu_placement_requirement() -> str:
    return format_cpu_memory_requirement(
        _SIM2REAL_CPU_MILLICORES, _SIM2REAL_MEMORY_BYTES
    )


def _config_is_enabled(value: Any) -> bool:
    """Return whether a config flag uses an enabled truthy string form."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_source_sha(value: str) -> bool:
    """Return whether a value is an exact lowercase 40-character hexadecimal SHA."""
    return len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def _evaluation_issues(config: Mapping[str, Any]) -> list[Issue]:
    try:
        concurrency = int(str(config.get("evaluation_concurrency", "8")))
        max_frames = int(str(config.get("evaluation_max_frames", "0")))
        if concurrency > 0 and max_frames >= 0:
            return []
    except (TypeError, ValueError):
        pass
    return [
        (
            "Sim2Real hosted evaluation settings are invalid",
            "set evaluation_concurrency to a positive integer and evaluation_max_frames "
            "to a nonnegative integer; zero sends every declared primary frame",
        )
    ]


def _token_factory_endpoint_issues(config: Mapping[str, Any]) -> list[Issue]:
    """Reject an unsafe or stale configured hosted-inference endpoint."""

    from npa.clients.token_factory import DEFAULT_BASE_URL
    from npa.workflows.sim2real.hosted_evaluation import validate_token_factory_endpoint

    try:
        validate_token_factory_endpoint(
            str(config.get("token_factory_base_url") or DEFAULT_BASE_URL)
        )
    except (TypeError, ValueError) as exc:
        return [
            (
                f"Sim2Real Token Factory endpoint is invalid: {exc}",
                "set token_factory_base_url to the credential-free HTTPS endpoint "
                "that Stage 8 and Stage 9 will bind exactly",
            )
        ]
    return []


def _source_sha_issues(config: Mapping[str, Any]) -> list[Issue]:
    """Mirror the renderer's two independent ``config.source_sha`` gates.

    ``skypilot_render.build_skypilot_task_doc`` normalizes the value with
    ``strip().lower()`` and then (1) rejects any non-empty value that is not an
    exact 40-character hexadecimal SHA regardless of baked mode, and (2) requires
    a value to be present when ``require_baked_npa`` is enabled. Reproduce both so
    a missing or malformed attestation is reported by preflight rather than as a
    per-step render error.
    """
    issues: list[Issue] = []
    require_baked = _config_is_enabled(config.get("require_baked_npa"))
    source_sha_value = str(config.get("source_sha") or "").strip().lower()
    if source_sha_value and not _is_source_sha(source_sha_value):
        issues.append(
            (
                "config.source_sha is not an exact 40-character hexadecimal source "
                "attestation",
                "set --var source_sha=<40-hex> to the exact source commit the "
                "attested immutable task images were built from",
            )
        )
    elif require_baked and not source_sha_value:
        issues.append(
            (
                "config.source_sha is missing while config.require_baked_npa is "
                "enabled",
                "set --var source_sha=<40-hex> to the exact source commit the "
                "attested immutable task images were built from",
            )
        )
    return issues


def image_prerequisites(config: Mapping[str, Any]) -> list[Issue]:
    """Validate the complete immutable Sim2Real image bundle before pull probes.

    Args:
        config: Resolved workflow configuration.
    Returns:
        Missing or invalid image and source-attestation inputs.
    Raises:
        None.
    """
    issues: list[Issue] = []
    invalid_images = [
        key
        for key in _IMAGE_KEYS
        if not _DIGEST_IMAGE.fullmatch(str(config.get(key) or "").strip())
    ]
    if invalid_images:
        issues.append(
            (
                "Sim2Real image inputs are missing or not registry-qualified immutable "
                f"digests: {', '.join(invalid_images)}",
                "set every listed --var to <registry>/<repository>@sha256:<64-hex>; "
                "run `npa workbench workflow preflight-images <spec> --var ...` to "
                "verify the exact bytes",
            )
        )

    issues.extend(_source_sha_issues(config))
    return issues


def static_prerequisites(
    config: Mapping[str, Any],
    *,
    requested_secret_envs: Sequence[str],
    secret_values: Mapping[str, str],
    hf_validator: Callable[[str, str], Any],
    token_factory_validator: Callable[[str, str], Any],
) -> list[Issue]:
    """Validate immutable inputs, consent, secret forwarding, and gated access."""

    issues = image_prerequisites(config)
    issues.extend(_evaluation_issues(config))
    issues.extend(_token_factory_endpoint_issues(config))

    pvc = str(config.get("isaac_cache_pvc") or "").strip()
    if not pvc:
        issues.append(
            (
                "config.isaac_cache_pvc is empty",
                "run `npa workbench sim2real cache provision --name <rwx-pvc> "
                "--context <context> --namespace <namespace>`, then pass "
                "--var isaac_cache_pvc=<rwx-pvc>; canonical submission verifies "
                "that it is Bound or, for WaitForFirstConsumer, that the guarded "
                "cache writer is initial, then warms and attests it in the first "
                "SkyPilot state",
            )
        )

    # Rollout, validation, and gold evaluation use three distinct sealed splits.
    # Catch undersized reduced-proof profiles before Transfer/EnvGen spend GPU time.
    try:
        env_count = int(str(config["env_count"]))
        train_fraction = float(str(config["train_fraction"]))
        rollout_count = int(str(config["rollout_count"]))
        validation_count = int(str(config["validation_count"]))
        gold_count = int(str(config["gold_count"]))
    except (KeyError, TypeError, ValueError):
        issues.append(
            (
                "Sim2Real train/validation/gold split inputs are missing or not numeric",
                "set integer env_count, rollout_count, validation_count, and gold_count "
                "with a numeric train_fraction",
            )
        )
    else:
        requested_train = (
            int(round(env_count * train_fraction))
            if math.isfinite(train_fraction) and 0.0 < train_fraction < 1.0
            else -1
        )
        # EnvGen retains at least one train, validation, and gold row in each of
        # its three difficulty strata, then divides the heldout remainder evenly.
        train_count = min(env_count - 6, max(3, requested_train))
        validation_rows = (env_count - train_count) // 2
        gold_rows = env_count - train_count - validation_rows
        if (
            env_count < 9
            or not math.isfinite(train_fraction)
            or not 0.0 < train_fraction < 1.0
            or rollout_count <= 0
            or validation_count <= 0
            or gold_count <= 0
            or rollout_count > train_count
            or validation_count > validation_rows
            or gold_count > gold_rows
        ):
            issues.append(
                (
                    "Sim2Real evaluation counts exceed the sealed train/validation/gold "
                    "splits or their split inputs are invalid "
                    f"(env_count={env_count}, train_fraction={train_fraction}, "
                    f"available={train_count}/{validation_rows}/{gold_rows}, "
                    f"requested={rollout_count}/{validation_count}/{gold_count})",
                    "choose positive counts with 0 < train_fraction < 1 and keep each "
                    "request within its computed split; for example, env_count=640 and "
                    "train_fraction=0.8 support 64 rollout, validation, and gold rows",
                )
            )

    requested = {str(name).strip() for name in requested_secret_envs}
    not_forwarded = [name for name in _REQUIRED_SECRET_ENVS if name not in requested]
    if not_forwarded:
        issues.append(
            (
                "required runtime credentials are not forwarded with --secret-env: "
                + ", ".join(not_forwarded),
                "add `--secret-env AWS_ACCESS_KEY_ID --secret-env "
                "AWS_SECRET_ACCESS_KEY --secret-env HF_TOKEN --secret-env "
                "NEBIUS_TOKEN_FACTORY_KEY`; values resolve from the "
                "environment or the selected project's NPA credential store",
            )
        )

    hf_token = str(secret_values.get("HF_TOKEN") or "").strip()
    if hf_token:
        # Derive the gated repos from the sim2real capability's single source of
        # truth so this pre-launch gate matches the canonical Cosmos3 Stage 3.
        # Nano and Guardrail1 fetch at runtime; an old Transfer2.5 approval is
        # neither substituted nor treated as acceptance of the current model.
        from npa.workbench.model_access import HF, assets_for

        repos = list(
            dict.fromkeys(
                asset.repo
                for asset in assets_for(("sim2real",))
                if asset.provider == HF
            )
        )
        if not repos:
            repos = ["nvidia/Cosmos3-Nano", "nvidia/Cosmos-Guardrail1"]
        denied: list[str] = []
        for repo in (item for item in repos if item):
            result = hf_validator(hf_token, repo)
            if not getattr(result, "ok", False):
                detail = str(getattr(result, "error", "") or "access not verified")
                denied.append(f"{repo} ({detail})")
        if denied:
            issues.append(
                (
                    "Hugging Face access failed for required runtime model(s): "
                    + "; ".join(denied),
                    "accept each model's terms while signed in to the account that owns "
                    "HF_TOKEN, then run `npa workbench health access --capability "
                    "sim2real` until the repository passes",
                )
            )
    token_factory_key = str(secret_values.get("NEBIUS_TOKEN_FACTORY_KEY") or "").strip()
    if token_factory_key:
        from npa.workbench.cosmos.reason import (
            CosmosReasonError,
            hosted_rollout_model_family,
        )

        model = str(
            config.get("token_factory_model") or DEFAULT_TOKEN_FACTORY_MODEL
        ).strip()
        try:
            hosted_rollout_model_family(model)
        except CosmosReasonError as exc:
            issues.append(
                (
                    str(exc),
                    "select a supported hosted rollout evaluator with token_factory_model "
                    "before provisioning",
                )
            )
            return issues
        from npa.clients.token_factory import DEFAULT_BASE_URL
        from npa.workflows.sim2real.hosted_evaluation import (
            validate_token_factory_endpoint,
        )

        endpoint = validate_token_factory_endpoint(
            str(config.get("token_factory_base_url") or DEFAULT_BASE_URL)
        )
        result = token_factory_validator(token_factory_key, model, base_url=endpoint)
        if not getattr(result, "ok", False):
            detail = str(getattr(result, "error", "") or "access not verified")
            issues.append(
                (
                    f"Token Factory access failed for {model}: {detail}",
                    "verify the selected key, project-specific model availability, and "
                    "balance with `npa workbench token-factory models` and a minimal "
                    "inference before workflow submission",
                )
            )
    elif "NEBIUS_TOKEN_FACTORY_KEY" in requested:
        issues.append(
            (
                "NEBIUS_TOKEN_FACTORY_KEY could not be resolved for Sim2Real",
                "store the key in the selected project's private NPA credential store "
                "or runtime environment, then rerun Token Factory model preflight",
            )
        )
    return issues


def _ready_schedulable_cpu_nodes(nodes_json: str) -> list[str]:
    """Return nodes able to host one CPU stage and the SkyPilot controller."""
    return ready_schedulable_cpu_nodes(
        nodes_json,
        minimum_cpu_millicores=_SIM2REAL_CPU_MILLICORES,
        minimum_memory_bytes=_SIM2REAL_MEMORY_BYTES,
    )


def _is_rtx_pro_6000(labels: Mapping[str, Any]) -> bool:
    """Use the same reviewed GPU equivalences as workflow placement."""
    from npa.orchestration.skypilot.k8s_gpu_catalog import (
        KubernetesGpuCatalog,
        UnsatisfiableAcceleratorError,
        resolve_kubernetes_accelerator,
    )

    for key in (
        "nvidia.com/gpu.product",
        "nebius.com/gpu-name",
        "skypilot.co/accelerator",
    ):
        product = str(labels.get(key) or "")
        if not product:
            continue
        catalog = KubernetesGpuCatalog(
            context="driver-preflight",
            quantities_by_accelerator={product: frozenset({1})},
        )
        try:
            resolve_kubernetes_accelerator("RTXPRO6000:1", catalog=catalog)
        except UnsatisfiableAcceleratorError:
            continue
        return True
    return False


def _managed_driver_isaac_nodes(nodes_json: str, *, placements=None) -> list[str]:
    """Return RTX PRO 6000 nodes serving managed drivers rather than operator drivers.

    Isaac Sim's camera-bearing rollouts are validated on Kubernetes only against the
    GPU-Operator mounted RTX driver stack that `--gpu-workload-profile rtx-rendering`
    provisions. The Nebius managed-driver image still satisfies pure-compute CUDA, so
    Cosmos3 structural transfer and EnvGen succeed on it and the mismatch stays invisible until
    Stage 7 renders, where it surfaces as an opaque Warp illegal-memory-access after
    the earlier GPU stages have already been paid for.
    """
    payload = json.loads(nodes_json or "")
    from npa.orchestration.npa_workflow.sim2real_driver_preflight import (
        node_can_host_isaac,
    )

    if not isinstance(payload, Mapping) or not isinstance(payload.get("items"), list):
        raise ValueError("Isaac node listing must contain an items list")
    flagged: list[str] = []
    for node in payload["items"]:
        if not isinstance(node, Mapping):
            raise ValueError("Isaac node listing contains an invalid record")
        metadata = node.get("metadata")
        if not isinstance(metadata, Mapping) or not metadata.get("name"):
            raise ValueError("Isaac node metadata evidence is invalid")
        labels = metadata.get("labels", {})
        if not isinstance(labels, Mapping):
            raise ValueError("Isaac node label evidence is invalid")
        if not _is_rtx_pro_6000(labels):
            continue
        managed = str(labels.get("nebius.com/driverful") or "").lower() == "true"
        operator = (
            str(labels.get("nvidia.com/gpu.deploy.operands") or "").lower() == "true"
        )
        if managed and not operator and node_can_host_isaac(node, placements):
            flagged.append(str(metadata.get("name") or ""))
    return sorted(name for name in flagged if name)


def _verified_cpu_nodes(nodes_json):
    """Keep malformed node records inside the aggregated prerequisite boundary."""
    try:
        return _ready_schedulable_cpu_nodes(nodes_json)
    except (AttributeError, TypeError, ValueError):
        return []


def _cpu_placement_issues(nodes) -> list[Issue]:
    issues: list[Issue] = []
    if getattr(nodes, "returncode", 1) != 0:
        issues.append(
            (
                "Kubernetes nodes cannot be listed for Sim2Real CPU/GPU placement",
                "refresh the selected kube context and grant read access to nodes; run "
                "`kubectl get nodes -o wide`",
            )
        )
    elif not _verified_cpu_nodes(str(getattr(nodes, "stdout", ""))):
        issues.append(
            (
                "no Ready, schedulable, appropriately untainted node can fit the "
                "Sim2Real CPU stage plus SkyPilot controller "
                f"({cpu_placement_requirement()})",
                "provide a node with sufficient allocatable CPU and memory, such as "
                "cpu-e2/16vcpu-64gb (the "
                "8vcpu-32gb nominal preset loses capacity to Kubernetes reserve), wait for "
                "Ready, then rerun `kubectl get nodes -o json` on the selected context",
            )
        )

    return issues


def _driver_placement_issues(nodes, placements) -> list[Issue]:
    from npa.orchestration.skypilot.k8s_gpu_catalog import KubernetesGpuCatalogError

    if getattr(nodes, "returncode", 1) != 0:
        return []
    try:
        names = _managed_driver_isaac_nodes(
            str(getattr(nodes, "stdout", "")), placements=placements
        )
    except (ValueError, KubernetesGpuCatalogError):
        return [
            (
                "Isaac render node placement could not be verified",
                "verify the selected node listing and the effective Isaac resource "
                "profile, selectors, affinity, and tolerations",
            )
        ]
    if not names:
        return []
    return [
        (
            "Isaac render stages could run on eligible RTX PRO 6000 node(s) serving the "
            "Nebius managed-driver image instead of the validated GPU-Operator "
            "mounted RTX drivers: " + ", ".join(names),
            "reprovision the Isaac GPU pool with `npa cluster up "
            "--gpu-workload-profile rtx-rendering`, which selects the "
            "operator-mounted RTX driver path and its GLX/EGL/Vulkan readiness "
            "gate, then rerun `kubectl get nodes -o json` on the selected context",
        )
    ]


def _is_delayed_binding_isaac_cache(pvc: Mapping[str, Any], runner) -> bool:
    """Return whether an RWX claim can bind at the canonical cache-warm stage."""

    spec = pvc.get("spec") if isinstance(pvc.get("spec"), Mapping) else {}
    storage_class = str(spec.get("storageClassName") or "").strip()
    if not storage_class:
        return False
    result = runner(["get", "storageclass", storage_class, "-o", "json"])
    if getattr(result, "returncode", 1) != 0:
        return False
    try:
        payload = json.loads(str(getattr(result, "stdout", ""))) or {}
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, Mapping):
        return False
    return (
        str(payload.get("volumeBindingMode") or "").strip() == _WAIT_FOR_FIRST_CONSUMER
    )


def _isaac_cache_issues(
    config, runner, namespace, *, allow_wait_for_first_consumer: bool = False
) -> list[Issue]:
    if namespace is None:
        return [
            (
                "Sim2Real cache namespace could not be verified",
                "verify the selected Kubernetes context and its namespace; "
                "do not substitute another namespace's cache",
            )
        ]
    issues: list[Issue] = []
    pvc_name = str(config.get("isaac_cache_pvc") or "").strip()
    if pvc_name:
        pvc_result = runner(["get", "pvc", pvc_name, "-n", namespace, "-o", "json"])
        pvc: dict[str, Any] = {}
        if getattr(pvc_result, "returncode", 1) == 0:
            try:
                pvc = json.loads(str(getattr(pvc_result, "stdout", ""))) or {}
            except json.JSONDecodeError:
                pvc = {}
        modes = set((pvc.get("spec") or {}).get("accessModes") or [])
        phase = str((pvc.get("status") or {}).get("phase") or "").strip()
        bound_rwx = phase == "Bound" and "ReadWriteMany" in modes
        delayed_binding = (
            allow_wait_for_first_consumer
            and phase == "Pending"
            and "ReadWriteMany" in modes
            and _is_delayed_binding_isaac_cache(pvc, runner)
        )
        if not bound_rwx and not delayed_binding:
            issues.append(
                (
                    f"Isaac cache PVC {pvc_name!r} is missing or is not Bound ReadWriteMany "
                    f"in namespace {namespace!r}",
                    "provision or inspect this exact claim with `npa workbench sim2real "
                    "cache provision --name <pvc> --context <context> --namespace "
                    f"{namespace}`; a Pending RWX claim is admissible only when its exact "
                    "StorageClass uses WaitForFirstConsumer and the canonical stage-00 "
                    "cache-warm state is the initial consumer; otherwise wait for Bound",
                )
            )

    return issues


def kubernetes_prerequisites(
    config: Mapping[str, Any],
    *,
    runner: Callable[[list[str]], Any],
    namespace: str | None = "default",
    isaac_placements=None,
    allow_wait_for_first_consumer: bool = False,
) -> list[Issue]:
    """Validate cluster objects the real Sim2Real/SkyPilot path consumes.

    Args:
        config: Resolved workflow configuration.
        runner: Exact-context Kubernetes reader.
        namespace: Selected workflow namespace; None refuses cache lookup.
        isaac_placements: Effective render stage placement constraints.
        allow_wait_for_first_consumer: Permit an RWX delayed-binding claim only
            for the canonical initial cache-warm consumer.
    Returns:
        Missing prerequisites and their remediation guidance.
    Raises:
        None.
    """
    nodes = runner(["get", "nodes", "-o", "json"])
    return (
        _cpu_placement_issues(nodes)
        + _driver_placement_issues(nodes, isaac_placements)
        + _isaac_cache_issues(
            config,
            runner,
            namespace,
            allow_wait_for_first_consumer=allow_wait_for_first_consumer,
        )
    )
