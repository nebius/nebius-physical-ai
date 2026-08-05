"""Preview a managed-cluster drain without weakening workload safety.

Destroying a managed Kubernetes cluster drains every schedulable pod, and every
eviction respects its PodDisruptionBudget (PDB).  The preview therefore takes one
shared inventory of nodes, pods, controllers, and PDBs and applies the same
eviction-relevant selector/placement rules to every namespace.  It never patches
a PDB or force-deletes a protected pod.

The preview is best-effort and deliberately non-interactive. A temporary copy
of the selected kubeconfig marks exec plugins non-interactive and adds Nebius'
``--no-browser`` flag, so teardown can never unexpectedly launch a login page.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import yaml

DEFAULT_TIMEOUT_SECONDS = 60

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class DrainPreviewIssue:
    """Why the best-effort Kubernetes safety preview was unavailable."""

    kind: str
    summary: str


@dataclass(frozen=True)
class DisruptionBlocker:
    """A PDB that will deny at least one eviction in this drain inventory."""

    namespace: str
    name: str
    desired_healthy: int = 0
    current_healthy: int = 0
    disruptions_allowed: int = 0
    min_available: object | None = None
    max_unavailable: object | None = None
    unhealthy_pod_eviction_policy: str = "IfHealthyBudget"
    matching_pods: tuple[str, ...] = ()
    workloads: tuple[str, ...] = ()
    nodes: tuple[str, ...] = ()
    one_node_pool: bool = False
    reason: str = ""

    def render(self) -> str:
        return (
            f"{self.namespace}/{self.name} "
            f"(allows {self.disruptions_allowed} disruption(s); "
            f"{self.current_healthy}/{self.desired_healthy} healthy; "
            f"{len(self.matching_pods)} drain pod(s) on "
            f"{', '.join(self.nodes) or 'unknown nodes'})"
        )


@dataclass(frozen=True)
class _DrainPod:
    namespace: str
    name: str
    node: str
    labels: dict[str, str]
    workload: str
    ready: bool


@dataclass(frozen=True)
class DrainInventory:
    """One snapshot used for preview reporting and teardown expectations."""

    nodes: tuple[str, ...]
    pod_count: int
    pdb_count: int
    blockers: tuple[DisruptionBlocker, ...]


def classify_drain_preview_failure(message: str) -> DrainPreviewIssue:
    """Classify kubectl failures without making raw implementation noise primary."""

    lowered = " ".join(str(message or "").lower().split())
    if any(
        marker in lowered
        for marker in (
            "forbidden",
            "cannot list resource",
            "cannot get resource",
            "rbac",
        )
    ):
        return DrainPreviewIssue(
            "authorization",
            "Kubernetes RBAC denied listing nodes, pods, or policy/v1 PodDisruptionBudgets",
        )
    kubeconfig_marker = any(
        marker in lowered
        for marker in (
            "error loading config",
            "no configuration has been provided",
            "current-context is not set",
            "context was not found",
            "context does not exist",
            "kubeconfig",
        )
    ) or re.search(r"context .* (?:was )?not found", lowered) is not None
    if kubeconfig_marker and any(
        marker in lowered for marker in ("not found", "config", "context")
    ):
        return DrainPreviewIssue(
            "kubeconfig",
            "the selected kubeconfig or context could not be loaded",
        )
    if any(
        marker in lowered
        for marker in (
            "unauthorized",
            "unauthenticated",
            "invalid token",
            "must be logged in",
            "authentication required",
            "exec plugin",
            "credential plugin",
        )
    ):
        return DrainPreviewIssue(
            "authentication",
            "the selected kubeconfig could not authenticate non-interactively",
        )
    if any(
        marker in lowered
        for marker in (
            "connection refused",
            "unable to connect to the server",
            "dial tcp",
            "no such host",
            "i/o timeout",
            "tls handshake timeout",
            "service unavailable",
        )
    ):
        return DrainPreviewIssue(
            "api",
            "the Kubernetes API endpoint could not be reached",
        )
    return DrainPreviewIssue(
        "kubectl",
        "kubectl could not inspect the drain inventory",
    )


def describe_preview_unavailable(issue: DrainPreviewIssue) -> str:
    """Explain a preview-only failure and the relevant operator correction."""

    actions = {
        "authentication": (
            "Regenerate the cluster kubeconfig with `npa cluster kubeconfig` under a "
            "non-interactive service-account profile, or authenticate that profile before "
            "teardown; the preview will not open a browser by design."
        ),
        "authorization": (
            "Pass --kubeconfig for an existing NPA operator identity that already has "
            "read access to nodes, pods, and policy/v1 PodDisruptionBudgets; NPA will "
            "not broaden RBAC during teardown."
        ),
        "kubeconfig": (
            "Pass the cluster's kubeconfig with --kubeconfig, or regenerate the saved one "
            "with `npa cluster kubeconfig`."
        ),
        "api": (
            "Check the cluster API endpoint and operator-machine network path, then retry "
            "with the cluster's saved --kubeconfig if preview diagnostics are needed."
        ),
        "kubectl": (
            "Install/configure kubectl and pass the cluster's saved --kubeconfig to restore "
            "the preview."
        ),
    }
    action = actions.get(issue.kind, actions["kubectl"])
    return (
        f"best-effort drain preview unavailable: {issue.summary}; teardown will continue. "
        "The reported failure affects only the preview; Terraform destroy will still be "
        "attempted with its own credentials. PDB drain safety was not verified, so NPA "
        "could not check which workloads may delay eviction. "
        f"Corrective action: {action}"
    )


def blocking_pod_disruption_budgets(
    *,
    kubeconfig: str = "",
    context: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> tuple[list[DisruptionBlocker], DrainPreviewIssue | None]:
    """Return PDBs that will block the shared full-drain inventory."""

    inventory, issue = drain_inventory(
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
        runner=runner,
    )
    return (list(inventory.blockers) if inventory is not None else []), issue


def drain_inventory(
    *,
    kubeconfig: str = "",
    context: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> tuple[DrainInventory | None, DrainPreviewIssue | None]:
    """Collect and evaluate the exact cluster-wide drain inventory once."""

    cmd = [
        "kubectl",
        "get",
        "nodes,pods,poddisruptionbudgets",
        "--all-namespaces",
        "-o",
        "json",
    ]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    execute = runner or subprocess.run
    with _noninteractive_kubeconfig_env(kubeconfig) as (env, config_issue):
        if config_issue is not None:
            return None, config_issue
        try:
            result = execute(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, classify_drain_preview_failure(f"could not run kubectl: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        return None, classify_drain_preview_failure(detail)
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None, DrainPreviewIssue(
            "kubectl", "kubectl returned unreadable drain inventory data"
        )
    if not isinstance(payload, dict):
        return None, DrainPreviewIssue(
            "kubectl", "kubectl returned an unexpected drain inventory payload"
        )

    items = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
    nodes: dict[str, dict[str, str]] = {}
    pods: list[_DrainPod] = []
    pdbs: list[dict[str, Any]] = []
    for item in items:
        kind = str(item.get("kind") or "").strip().lower()
        metadata = _as_dict(item.get("metadata"))
        spec = _as_dict(item.get("spec"))
        status = _as_dict(item.get("status"))
        if kind == "node":
            name = str(metadata.get("name") or "").strip()
            if name:
                nodes[name] = _string_dict(metadata.get("labels"))
        elif kind == "pod":
            pod = _drain_pod(metadata, spec, status)
            if pod is not None:
                pods.append(pod)
        elif (
            kind in {"poddisruptionbudget", "pdb"}
            or ("disruptionsAllowed" in status and "selector" in spec)
            or (not kind and "disruptionsAllowed" in status)
        ):
            pdbs.append(item)

    pool_sizes: dict[str, int] = {}
    for node_name, labels in nodes.items():
        pool = _node_pool(node_name, labels)
        pool_sizes[pool] = pool_sizes.get(pool, 0) + 1
    blockers: list[DisruptionBlocker] = []
    for item in pdbs:
        metadata = _as_dict(item.get("metadata"))
        spec = _as_dict(item.get("spec"))
        status = _as_dict(item.get("status"))
        allowed = status.get("disruptionsAllowed")
        if allowed is None:
            continue
        try:
            allowed_count = int(allowed)
        except (TypeError, ValueError):
            continue
        namespace = str(metadata.get("namespace") or "").strip()
        name = str(metadata.get("name") or "").strip()
        if not namespace or not name:
            continue
        selector = _as_dict(spec.get("selector"))
        matching = [
            pod
            for pod in pods
            if pod.namespace == namespace and _selector_matches(selector, pod.labels)
        ]
        unhealthy_policy = str(
            spec.get("unhealthyPodEvictionPolicy") or "IfHealthyBudget"
        ).strip()
        desired_healthy = _as_int(status.get("desiredHealthy"))
        current_healthy = _as_int(status.get("currentHealthy"))
        ready_pods = [pod for pod in matching if pod.ready]
        unhealthy_pods = [pod for pod in matching if not pod.ready]
        # AlwaysAllow never charges an unhealthy running pod to the budget.
        # IfHealthyBudget likewise permits it once the guarded workload already
        # satisfies desiredHealthy; otherwise that unhealthy eviction remains
        # blocked even though it cannot reduce currentHealthy further.
        blocked_unhealthy = (
            unhealthy_pods
            if unhealthy_policy != "AlwaysAllow" and current_healthy < desired_healthy
            else []
        )
        eviction_relevant = [*ready_pods, *blocked_unhealthy]
        # Backward-compatible fallback for older kubectl/test payloads that
        # contain only PDBs. With pod inventory present, a selector that matches
        # no drain candidates cannot block this teardown.
        should_block = (
            allowed_count <= 0
            if not pods
            else len(ready_pods) > max(0, allowed_count) or bool(blocked_unhealthy)
        )
        if not should_block:
            continue
        blocker_nodes = tuple(
            sorted({pod.node for pod in eviction_relevant if pod.node})
        )
        matching_pods = tuple(
            sorted(f"{pod.namespace}/{pod.name}" for pod in eviction_relevant)
        )
        workloads = tuple(
            sorted({pod.workload for pod in eviction_relevant if pod.workload})
        )
        one_node_pool = any(
            pool_sizes.get(_node_pool(node, nodes.get(node, {})), 0) == 1
            for node in blocker_nodes
        )
        reasons: list[str] = []
        if len(ready_pods) > max(0, allowed_count):
            reasons.append(
                f"draining {len(ready_pods)} healthy matching pod(s) requires more "
                f"than the {max(0, allowed_count)} disruption(s) currently permitted"
            )
        if blocked_unhealthy:
            reasons.append(
                f"{len(blocked_unhealthy)} unhealthy pod(s) also remain protected "
                f"by IfHealthyBudget because currentHealthy {current_healthy} is below "
                f"desiredHealthy {desired_healthy}"
            )
        reason = "; ".join(reasons)
        if one_node_pool:
            reason += (
                "; at least one affected CPU/node pool has one node, so a controller "
                "cannot place a healthy replacement before that node is removed"
            )
        blockers.append(
            DisruptionBlocker(
                namespace=namespace,
                name=name,
                desired_healthy=desired_healthy,
                current_healthy=current_healthy,
                disruptions_allowed=allowed_count,
                min_available=spec.get("minAvailable"),
                max_unavailable=spec.get("maxUnavailable"),
                unhealthy_pod_eviction_policy=unhealthy_policy,
                matching_pods=matching_pods,
                workloads=workloads,
                nodes=blocker_nodes,
                one_node_pool=one_node_pool,
                reason=reason,
            )
        )
    blockers.sort(key=lambda blocker: (blocker.namespace, blocker.name))
    return (
        DrainInventory(
            nodes=tuple(sorted(nodes)),
            pod_count=len(pods),
            pdb_count=len(pdbs),
            blockers=tuple(blockers),
        ),
        None,
    )


def _drain_pod(
    metadata: dict[str, Any], spec: dict[str, Any], status: dict[str, Any]
) -> _DrainPod | None:
    namespace = str(metadata.get("namespace") or "").strip()
    name = str(metadata.get("name") or "").strip()
    node = str(spec.get("nodeName") or "").strip()
    phase = str(status.get("phase") or "").strip().lower()
    if (
        not namespace
        or not name
        or not node
        or metadata.get("deletionTimestamp")
        or phase in {"succeeded", "failed"}
    ):
        return None
    owners = [
        item
        for item in (metadata.get("ownerReferences") or [])
        if isinstance(item, dict)
    ]
    controller = next((item for item in owners if item.get("controller") is True), None)
    if controller is None and owners:
        controller = owners[0]
    controller_kind = str((controller or {}).get("kind") or "Pod").strip()
    controller_name = str((controller or {}).get("name") or name).strip()
    # DaemonSet and mirror/static pods are skipped by normal node drains.
    annotations = _string_dict(metadata.get("annotations"))
    if (
        controller_kind.lower() == "daemonset"
        or "kubernetes.io/config.mirror" in annotations
    ):
        return None
    ready = any(
        isinstance(condition, dict)
        and condition.get("type") == "Ready"
        and str(condition.get("status") or "").lower() == "true"
        for condition in (status.get("conditions") or [])
    )
    return _DrainPod(
        namespace=namespace,
        name=name,
        node=node,
        labels=_string_dict(metadata.get("labels")),
        workload=f"{namespace}/{controller_kind}/{controller_name}",
        ready=ready,
    )


def _selector_matches(selector: dict[str, Any], labels: dict[str, str]) -> bool:
    for key, expected in _string_dict(selector.get("matchLabels")).items():
        if labels.get(key) != expected:
            return False
    for expression in selector.get("matchExpressions") or []:
        if not isinstance(expression, dict):
            return False
        key = str(expression.get("key") or "").strip()
        operator = str(expression.get("operator") or "").strip()
        values = {str(value) for value in (expression.get("values") or [])}
        present = key in labels
        if operator == "In" and (not present or labels[key] not in values):
            return False
        if operator == "NotIn" and present and labels[key] in values:
            return False
        if operator == "Exists" and not present:
            return False
        if operator == "DoesNotExist" and present:
            return False
        if operator not in {"In", "NotIn", "Exists", "DoesNotExist"}:
            return False
    return True


def _node_pool(_node_name: str, labels: dict[str, str]) -> str:
    for key in (
        "nebius.com/node-group-id",
        "mk8s.nebius.ai/node-group-id",
        "nebius.ai/node-group-id",
        "node.kubernetes.io/instance-group",
        "cloud.google.com/gke-nodepool",
        "eks.amazonaws.com/nodegroup",
    ):
        if labels.get(key):
            return f"{key}={labels[key]}"
    gpu_product = labels.get("nvidia.com/gpu.product", "")
    gpu_present = labels.get("nvidia.com/gpu.present", "").lower() == "true"
    return f"accelerator={gpu_product or ('gpu' if gpu_present else 'cpu')}"


def _string_dict(value: object) -> dict[str, str]:
    return {
        str(key): str(item)
        for key, item in _as_dict(value).items()
        if key is not None and item is not None
    }


@contextmanager
def _noninteractive_kubeconfig_env(
    kubeconfig: str,
) -> Iterator[tuple[dict[str, str], DrainPreviewIssue | None]]:
    """Yield an env whose kubeconfig exec plugins cannot authenticate interactively."""

    env = os.environ.copy()
    configured = str(kubeconfig or "").strip() or str(env.get("KUBECONFIG") or "").strip()
    if configured:
        paths = [Path(value).expanduser() for value in configured.split(os.pathsep) if value]
    else:
        default = Path.home() / ".kube" / "config"
        paths = [default] if default.exists() else []
    if not paths:
        yield env, None
        return

    with tempfile.TemporaryDirectory(prefix="npa-drain-preview-") as temporary:
        rendered_paths: list[str] = []
        for index, path in enumerate(paths):
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                yield (
                    env,
                    DrainPreviewIssue(
                        "kubeconfig", "the selected kubeconfig could not be read safely"
                    ),
                )
                return
            if not isinstance(payload, dict):
                yield (
                    env,
                    DrainPreviewIssue(
                        "kubeconfig", "the selected kubeconfig is not a Kubernetes config object"
                    ),
                )
                return
            _disable_interactive_exec_auth(payload)
            rendered = Path(temporary) / f"kubeconfig-{index}.yaml"
            rendered.write_text(
                yaml.safe_dump(payload, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            rendered.chmod(0o600)
            rendered_paths.append(str(rendered))
        env["KUBECONFIG"] = os.pathsep.join(rendered_paths)
        yield env, None


def _disable_interactive_exec_auth(payload: dict[str, Any]) -> None:
    for user in payload.get("users") or []:
        if not isinstance(user, dict):
            continue
        config = _as_dict(user.get("user"))
        exec_config = config.get("exec")
        if not isinstance(exec_config, dict):
            continue
        exec_config["interactiveMode"] = "Never"
        command = Path(str(exec_config.get("command") or "")).name.lower()
        if command not in {"nebius", "nebius.exe"}:
            continue
        args = [str(value) for value in (exec_config.get("args") or [])]
        if "--no-browser" not in args:
            args.insert(0, "--no-browser")
        exec_config["args"] = args


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def describe_drain_expectation(blockers: list[DisruptionBlocker]) -> str:
    """Return operator guidance for protected PDBs that may delay deletion."""

    if not blockers:
        return ""
    details: list[str] = []
    for blocker in blockers:
        workloads = ", ".join(blocker.workloads) or "matching workload(s)"
        shape = (
            " This is a one-node pool: no spare CPU node exists for a healthy "
            "replacement before eviction."
            if blocker.one_node_pool
            else ""
        )
        details.append(
            f"{blocker.render()} protects {workloads}: {blocker.reason}.{shape}"
        )
    return (
        f"{len(blockers)} PodDisruptionBudget(s) will deny at least one eviction: "
        + " ".join(details)
        + " NPA will not patch budgets or force-delete protected pods. The provider "
        "drain will retry/wait and cluster deletion can still safely continue; expected "
        "PDB backoff can look stalled until controller/node-group teardown converges."
    )
