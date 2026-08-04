"""Prepare a managed-cluster drain without weakening user workload safety.

Destroying a managed Kubernetes cluster drains its nodes, and eviction respects
PodDisruptionBudgets (PDBs). On the one-node NPA default, single-replica system
add-ons can allow zero disruptions and make deletion appear stuck. A full
cluster deletion can safely relax those exact platform add-on budgets because
the add-ons are being removed with the cluster; arbitrary user or unknown PDBs
must remain untouched.

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

# Exact Managed Kubernetes add-ons observed on the NPA default cluster. Do not
# broaden this to all of kube-system: operators can install their own workloads
# there, and their disruption policy is still theirs.
_MANAGED_SYSTEM_PDB_NAMES = frozenset(
    {
        "cilium-operator",
        "coredns",
        "metrics-server",
    }
)


@dataclass(frozen=True)
class DrainPreviewIssue:
    """Why the best-effort Kubernetes safety preview was unavailable."""

    kind: str
    summary: str


@dataclass(frozen=True)
class DisruptionBlocker:
    """A PodDisruptionBudget that currently permits no evictions."""

    namespace: str
    name: str
    desired_healthy: int = 0
    current_healthy: int = 0
    min_available: object | None = None
    max_unavailable: object | None = None

    def render(self) -> str:
        return (
            f"{self.namespace}/{self.name} "
            f"(allows 0 disruptions; {self.current_healthy}/{self.desired_healthy} healthy)"
        )


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
            "Kubernetes RBAC denied listing policy/v1 PodDisruptionBudgets",
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
        "kubectl could not inspect policy/v1 PodDisruptionBudgets",
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
            "Grant the selected kubeconfig identity list access to policy/v1 "
            "PodDisruptionBudgets, or pass --kubeconfig for an identity with that read access."
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
    """Return PDBs that allow no evictions, or a classified preview issue."""

    cmd = ["kubectl", "get", "poddisruptionbudgets", "--all-namespaces", "-o", "json"]
    if context.strip():
        cmd[1:1] = ["--context", context.strip()]
    execute = runner or subprocess.run
    with _noninteractive_kubeconfig_env(kubeconfig) as (env, config_issue):
        if config_issue is not None:
            return [], config_issue
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
            return [], classify_drain_preview_failure(f"could not run kubectl: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        return [], classify_drain_preview_failure(detail)
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return [], DrainPreviewIssue(
            "kubectl", "kubectl returned unreadable PodDisruptionBudget data"
        )
    if not isinstance(payload, dict):
        return [], DrainPreviewIssue(
            "kubectl", "kubectl returned an unexpected PodDisruptionBudget payload"
        )

    blockers: list[DisruptionBlocker] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
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
        if allowed_count > 0:
            continue
        namespace = str(metadata.get("namespace") or "").strip()
        name = str(metadata.get("name") or "").strip()
        if not namespace or not name:
            continue
        blockers.append(
            DisruptionBlocker(
                namespace=namespace,
                name=name,
                desired_healthy=_as_int(status.get("desiredHealthy")),
                current_healthy=_as_int(status.get("currentHealthy")),
                min_available=spec.get("minAvailable"),
                max_unavailable=spec.get("maxUnavailable"),
            )
        )
    blockers.sort(key=lambda blocker: (blocker.namespace, blocker.name))
    return blockers, None


def is_managed_system_pdb(blocker: DisruptionBlocker) -> bool:
    """Whether *blocker* is an exact NPA Managed Kubernetes system add-on."""

    name = blocker.name.strip().lower()
    return blocker.namespace.strip() == "kube-system" and name in _MANAGED_SYSTEM_PDB_NAMES


def relax_managed_system_pdbs(
    blockers: list[DisruptionBlocker],
    *,
    kubeconfig: str = "",
    context: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> tuple[
    list[DisruptionBlocker],
    list[tuple[DisruptionBlocker, DrainPreviewIssue]],
]:
    """Relax exact system add-on PDBs for a full managed-cluster deletion.

    User and unrecognized PDBs are never patched. A failed patch is diagnostic
    only: Terraform teardown still proceeds and its normal reconciliation remains
    the authority.
    """

    execute = runner or subprocess.run
    relaxed: list[DisruptionBlocker] = []
    failures: list[tuple[DisruptionBlocker, DrainPreviewIssue]] = []
    for blocker in blockers:
        if not is_managed_system_pdb(blocker):
            continue
        patch: dict[str, dict[str, object]]
        if blocker.min_available is not None:
            patch = {"spec": {"minAvailable": 0}}
        elif blocker.max_unavailable is not None:
            patch = {"spec": {"maxUnavailable": "100%"}}
        else:
            failures.append(
                (
                    blocker,
                    DrainPreviewIssue(
                        "kubectl",
                        "the system add-on PDB has no supported minAvailable/maxUnavailable field",
                    ),
                )
            )
            continue
        cmd = [
            "kubectl",
            "patch",
            "poddisruptionbudget",
            blocker.name,
            "-n",
            blocker.namespace,
            "--type=merge",
            "--patch",
            json.dumps(patch),
        ]
        if context.strip():
            cmd[1:1] = ["--context", context.strip()]
        with _noninteractive_kubeconfig_env(kubeconfig) as (env, config_issue):
            if config_issue is not None:
                failures.append((blocker, config_issue))
                continue
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
                failures.append(
                    (blocker, classify_drain_preview_failure(f"could not run kubectl: {exc}"))
                )
                continue
        if result.returncode == 0:
            relaxed.append(blocker)
            continue
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        failures.append((blocker, classify_drain_preview_failure(detail)))
    return relaxed, failures


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
    names = ", ".join(blocker.render() for blocker in blockers)
    return (
        f"{len(blockers)} protected PodDisruptionBudget(s) currently allow no evictions: "
        f"{names}. NPA does not override user-workload or unrecognized budgets. Node drain "
        "will retry until their pods reschedule or the node pool is removed, so teardown "
        "can look stalled for several minutes with no output; this wait is expected while "
        "those protections remain."
    )
