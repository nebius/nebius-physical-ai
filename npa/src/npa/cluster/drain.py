"""Report the PodDisruptionBudgets that will hold up a cluster teardown.

Destroying a managed-Kubernetes cluster drains its nodes, and eviction respects
PodDisruptionBudgets. The platform add-ons that ship with every cluster
(``coredns``, ``cilium-operator``, ``metrics-server``) declare budgets that allow
zero disruptions while they are running a single replica, so a drain sits and
retries -- around six minutes on a one-node CPU pool -- printing nothing an
operator can act on. It is normal, but indistinguishable from a hang.

Naming the budgets up front turns "terraform destroy has been quiet for five
minutes" into an expected, bounded wait.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import subprocess
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 60

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class DisruptionBlocker:
    """A PodDisruptionBudget that currently permits no evictions."""

    namespace: str
    name: str
    desired_healthy: int = 0
    current_healthy: int = 0

    def render(self) -> str:
        return (
            f"{self.namespace}/{self.name} "
            f"(allows 0 disruptions; {self.current_healthy}/{self.desired_healthy} healthy)"
        )


def blocking_pod_disruption_budgets(
    *,
    kubeconfig: str = "",
    context: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> tuple[list[DisruptionBlocker], str]:
    """Return the budgets that currently allow zero disruptions.

    The second element is an error string; an unreachable cluster is reported
    rather than treated as "nothing will block", so teardown guidance never
    claims a clean drain it did not verify.
    """

    cmd = ["kubectl", "get", "poddisruptionbudgets", "--all-namespaces", "-o", "json"]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    execute = runner or subprocess.run
    env: dict[str, str] | None = None
    if kubeconfig.strip():
        import os

        env = os.environ.copy()
        env["KUBECONFIG"] = kubeconfig.strip()
    try:
        result = execute(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"could not run kubectl: {exc}"
    if result.returncode != 0:
        return [], (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return [], "kubectl returned non-json output"

    blockers: list[DisruptionBlocker] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        metadata = _as_dict(item.get("metadata"))
        status = _as_dict(item.get("status"))
        allowed = status.get("disruptionsAllowed")
        if allowed is None:
            continue
        try:
            allowed_count = int(allowed)
        except (TypeError, ValueError):
            continue
        if allowed_count > 0:
            continue
        blockers.append(
            DisruptionBlocker(
                namespace=str(metadata.get("namespace") or ""),
                name=str(metadata.get("name") or ""),
                desired_healthy=_as_int(status.get("desiredHealthy")),
                current_healthy=_as_int(status.get("currentHealthy")),
            )
        )
    blockers.sort(key=lambda blocker: (blocker.namespace, blocker.name))
    return blockers, ""


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def describe_drain_expectation(blockers: list[DisruptionBlocker]) -> str:
    """Return operator-facing guidance for a drain that will be held up."""

    if not blockers:
        return ""
    names = ", ".join(blocker.render() for blocker in blockers)
    return (
        f"{len(blockers)} PodDisruptionBudget(s) currently allow no evictions: {names}. "
        "Node drain will retry until their pods reschedule or the node pool is removed, "
        "so teardown can look stalled for several minutes with no output. This is "
        "expected for single-replica platform add-ons (CoreDNS, Cilium, metrics-server) "
        "on a small node pool; let it run rather than interrupting it."
    )
