"""Project-scoped teardown planning and orchestration through existing NPA guards."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
from typing import Any, Callable

from npa.clients.json_output import parse_single_json_document


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class DestroyPhase:
    name: str
    commands: tuple[tuple[str, ...], ...]
    detail: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.name,
            "commands": [list(command) for command in self.commands],
            "detail": self.detail,
            "requires": list(self.requires),
        }


def build_project_destroy_plan(
    project: str, *, delete_project: bool = False
) -> list[DestroyPhase]:
    """Build a read-only, exact-identity plan from project-local NPA state."""

    from npa.cli.agent import resolve_project_agents
    from npa.clients.config import resolve_environment, resolve_terraform_state
    from npa.cluster.state import list_local_clusters
    from npa.controller_ownership import controller_owner
    from npa.provisioning_journal import list_operations

    environment = resolve_environment(project)
    if environment is None or not environment.project_id:
        raise RuntimeError(
            f"Project {project!r} has no immutable project identity; refusing teardown."
        )
    project_id = str(environment.project_id)
    agents = resolve_project_agents(project)
    agent_names = set(agents)
    for operation in list_operations(
        project_alias=project, project_id=project_id, resource_type="agent"
    ):
        requested_name = str(operation.read().get("requested_name") or "").strip()
        if requested_name:
            agent_names.add(requested_name)
    agent_commands = tuple(
        (
            "npa",
            "agent",
            "destroy",
            "--project",
            project,
            "--name",
            str(name),
            "--yes",
            "--json",
        )
        for name in sorted(agent_names)
    )
    owner = controller_owner(project)
    controller_commands: tuple[tuple[str, ...], ...] = ()
    if owner is not None:
        controller_commands = (
            (
                "npa",
                "skypilot",
                "cleanup-controller",
                "--project",
                project,
                "--project-id",
                owner.project_id,
                "--context",
                owner.context,
                "--cluster-id",
                owner.cluster_id,
                "--cluster-name",
                owner.cluster_name,
                "--yes",
                "--json",
            ),
        )
    cluster_targets: dict[tuple[str, str], str] = {
        (cluster.name, cluster.cluster_id): ""
        for cluster in list_local_clusters()
        if cluster.project_id == project_id and cluster.cluster_id
    }
    # Retries intentionally produce a new operation and may receive a different
    # immutable cluster ID for the same context.  Inventory every attempt; a
    # destroyed historical ID is harmless audit evidence, while collapsing by
    # context would either forget it or deadlock it against the newer ID.
    for operation in list_operations(
        project_alias=project, project_id=project_id, resource_type="cluster"
    ):
        payload = operation.read()
        context = str(payload.get("requested_name") or "").strip()
        for resource in payload.get("resources") or []:
            if not isinstance(resource, dict):
                continue
            cluster_id = str(resource.get("provider_id") or "").strip()
            resource_project = str(resource.get("project_id") or project_id).strip()
            if (
                resource.get("resource_type") == "managed_kubernetes_cluster"
                and context
                and cluster_id
                and resource_project == project_id
            ):
                cluster_targets.setdefault(
                    (context, cluster_id), operation.operation_id
                )
    cluster_commands = tuple(
        (
            "npa",
            "cluster",
            "down",
            "--project",
            project,
            "--project-id",
            project_id,
            "--cluster-id",
            cluster_id,
            "--context",
            context,
            *(
                ("--operation-id", operation_id)
                if operation_id
                else ()
            ),
            "--force",
            "--json",
        )
        for (context, cluster_id), operation_id in sorted(cluster_targets.items())
    )
    state = resolve_terraform_state(project)
    bucket_commands: tuple[tuple[str, ...], ...] = ()
    if state.bucket:
        bucket_commands = (
            (
                "npa",
                "storage",
                "bucket",
                "delete",
                "--project",
                project,
                "--project-id",
                project_id,
                "--name",
                state.bucket,
                "--yes",
                "--wait",
                "--json",
            ),
        )
    phases = [
        DestroyPhase(
            "workflows",
            (
                (
                    "npa",
                    "workbench",
                    "workflow",
                    "list",
                    "--project",
                    project,
                    "--json",
                ),
            ),
            "Inventory durable runs, then cancel each exact run before controller teardown.",
        ),
        DestroyPhase(
            "agents", agent_commands, "Destroy every configured project agent."
        ),
        DestroyPhase(
            "controller",
            controller_commands,
            "Remove the exact bound shared controller.",
            ("workflows",),
        ),
        DestroyPhase(
            "clusters",
            cluster_commands,
            "Destroy exact project-matched cluster identities.",
            ("workflows", "controller"),
        ),
        DestroyPhase(
            "bucket",
            bucket_commands,
            "Delete and verify the exact state bucket.",
            ("agents", "clusters"),
        ),
        DestroyPhase(
            "storage_iam",
            (
                (
                    "npa",
                    "storage",
                    "service-account",
                    "delete",
                    "--project",
                    project,
                    "--yes",
                    "--json",
                ),
            ),
            "Delete only project-scoped NPA-owned storage IAM.",
            ("agents", "bucket"),
        ),
        DestroyPhase(
            "local_cleanup",
            (
                (
                    "npa",
                    "cleanup",
                    "--project",
                    project,
                    "--yes",
                    "--keep-sky",
                    "--json",
                ),
            ),
            "Remove only project-scoped local residue; preserve shared runtime state.",
        ),
        DestroyPhase(
            "forget_alias",
            (("npa", "configure", "--forget-project", project),),
            "Forget the alias only after cloud and IAM phases converge.",
            (
                "workflows",
                "agents",
                "controller",
                "clusters",
                "bucket",
                "storage_iam",
                "local_cleanup",
            ),
        ),
    ]
    if delete_project:
        phases.append(
            DestroyPhase(
                "delete_project",
                (),
                "BLOCKED: NPA has no provider-supported project deletion API; plan only.",
            )
        )
    return phases


def _run(command: tuple[str, ...], runner: Runner) -> subprocess.CompletedProcess[str]:
    argv = _internal_command_argv(command) if runner is subprocess.run else list(command)
    try:
        return runner(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            argv,
            127,
            stdout="",
            stderr=f"{type(exc).__name__}: NPA command could not be started",
        )


def _internal_command_argv(command: tuple[str, ...]) -> list[str]:
    """Invoke NPA with this process's interpreter, independent of ``PATH``.

    Plans and recovery receipts intentionally retain the operator-facing
    ``npa ...`` spelling.  Only in-process orchestration replaces that first
    token, without a shell, so editable installs and console-entrypoint installs
    execute the same imported package under the active environment.
    """

    if not command or command[0] != "npa":
        return list(command)
    return [sys.executable, "-m", "npa", *command[1:]]


def execute_project_destroy(
    project: str,
    phases: list[DestroyPhase],
    *,
    runner: Runner = subprocess.run,
    on_phase: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute every independent phase and report complete/partial convergence."""

    from npa.clients.config import resolve_environment

    environment = resolve_environment(project)
    if environment is None or not environment.project_id:
        raise RuntimeError(
            f"Project {project!r} has no immutable project identity; refusing teardown."
        )
    project_id = str(environment.project_id)
    results: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for phase in phases:
        if on_phase:
            on_phase(phase.name)
        commands = list(phase.commands)
        phase_errors: list[str] = []
        phase_warnings: list[str] = []
        executed: list[list[str]] = []
        recovery_commands: list[list[str]] = []
        blocked_by = [
            dependency
            for dependency in phase.requires
            if statuses.get(dependency) not in {"completed", "degraded"}
        ]
        if blocked_by:
            phase_errors.append("dependency not converged: " + ", ".join(blocked_by))
            recovery_commands.extend([list(command) for command in commands])
        elif phase.name == "delete_project":
            phase_errors.append(phase.detail)
        elif phase.name == "workflows" and commands:
            inventory = _run(commands[0], runner)
            executed.append(list(commands[0]))
            if inventory.returncode != 0:
                phase_errors.append("workflow inventory failed")
                recovery_commands.append(list(commands[0]))
            else:
                payload = parse_single_json_document(inventory.stdout or "")
                rows = payload.get("runs") if isinstance(payload, dict) else None
                if not isinstance(rows, list):
                    phase_errors.append("workflow inventory returned ambiguous JSON")
                    recovery_commands.append(list(commands[0]))
                else:
                    for row in rows:
                        run_id = (
                            str(row.get("run_id") or "")
                            if isinstance(row, dict)
                            else ""
                        )
                        if not run_id:
                            continue
                        submission_state = str(
                            row.get("submission_state")
                            or row.get("status")
                            or row.get("submission_status")
                            or ""
                        ).upper()
                        if submission_state in {"NOT_SUBMITTED", "PLAN_ONLY"}:
                            continue
                        cancel_command = (
                            "npa",
                            "workbench",
                            "workflow",
                            "cancel",
                            run_id,
                            "--project",
                            project,
                            "--json",
                        )
                        completed = _run(cancel_command, runner)
                        executed.append(list(cancel_command))
                        if completed.returncode != 0:
                            phase_errors.append(
                                f"workflow cancellation failed for {run_id}"
                            )
                            recovery_commands.append(list(cancel_command))
        else:
            for command in commands:
                completed = _run(command, runner)
                executed.append(list(command))
                parsed = parse_single_json_document(completed.stdout or "")
                remote_only_converged = bool(
                    phase.name == "controller"
                    and isinstance(parsed, dict)
                    and parsed.get("outcome") == "degraded_local_metadata"
                    and parsed.get("remote_absence_verified") is True
                )
                if remote_only_converged:
                    phase_warnings.append(
                        "exact remote controller absence verified; stale local "
                        "metadata remains for idempotent reconciliation"
                    )
                elif completed.returncode != 0:
                    phase_errors.append(f"command failed (exit {completed.returncode})")
                    recovery_commands.append(list(command))
        phase_status = (
            "skipped_dependency"
            if blocked_by
            else "degraded"
            if phase_warnings and not phase_errors
            else "completed"
            if not phase_errors
            else "partial"
        )
        statuses[phase.name] = phase_status
        results.append(
            {
                "phase": phase.name,
                "status": phase_status,
                "commands": executed,
                "errors": phase_errors,
                "warnings": phase_warnings,
                "recovery_commands": recovery_commands,
                "blocked_by": blocked_by,
            }
        )
        try:
            from npa.teardown_receipts import record_teardown_event

            record_teardown_event(
                phase=f"project_destroy_{phase.name}",
                resource=project,
                terminal_state=(
                    "degraded_local_metadata"
                    if phase_warnings and not phase_errors
                    else "completed"
                    if not phase_errors
                    else "partial"
                ),
                project_alias=project,
                project_id=project_id,
                precheck={"planned_command_count": len(commands)},
                action={"kind": "npa_guarded_phase", "executed_count": len(executed)},
                verification={
                    "converged": not phase_errors,
                    "remote_absence_only": bool(phase_warnings),
                },
                errors=phase_errors,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            phase_errors.append(f"receipt write failed: {type(exc).__name__}")
            statuses[phase.name] = "partial"
            results[-1]["status"] = "partial"
            results[-1]["errors"] = phase_errors
    complete = all(result["status"] == "completed" for result in results)
    return {
        "status": "success" if complete else "partial",
        "project": project,
        "phases": results,
    }
