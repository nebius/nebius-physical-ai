"""Exact local-state retirement for an absent NPA agent generation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class AgentLocalRetirementError(RuntimeError):
    """Exact agent cloud state is absent, but local retirement is incomplete."""


def cleanup_agent_local_files(
    project_alias: str,
    name: str,
    *,
    operation_ids: tuple[str, ...] = (),
) -> None:
    """Retire exact auth, Terraform, and operation state after cloud absence."""

    if (
        not project_alias
        or not name
        or Path(project_alias).name != project_alias
        or Path(name).name != name
        or project_alias in {".", ".."}
        or name in {".", ".."}
    ):
        raise AgentLocalRetirementError("agent local-state selectors are invalid")

    agent_root = Path.home() / ".npa" / "agents"
    agent_dir = agent_root / project_alias / name

    from npa.deploy import provisioner

    tf_dir = provisioner.working_dir_path(project_alias, name)
    targets = ((agent_dir, agent_root), (tf_dir, tf_dir.parent.parent))
    for target, root in targets:
        try:
            if not target.resolve(strict=False).is_relative_to(root.resolve()):
                raise AgentLocalRetirementError(
                    f"agent local-state path escapes its owned root: {target}"
                )
        except OSError as exc:
            raise AgentLocalRetirementError(
                f"could not verify agent local-state path {target}: {exc}"
            ) from exc
        if os.path.lexists(target):
            if target.is_symlink() or not target.is_dir():
                raise AgentLocalRetirementError(
                    f"agent local-state path is not a regular directory: {target}"
                )
            try:
                shutil.rmtree(target)
            except OSError as exc:
                raise AgentLocalRetirementError(
                    f"could not retire agent local-state path {target}: {exc}"
                ) from exc
        if os.path.lexists(target):
            raise AgentLocalRetirementError(
                f"agent local-state path remains after retirement: {target}"
            )

    if operation_ids:
        from npa.provisioning_journal import OperationJournalError, load_operation

        for operation_id in operation_ids:
            try:
                operation = load_operation(operation_id)
                payload = operation.read()
                if (
                    str(payload.get("project_alias") or "") != project_alias
                    or str(payload.get("resource_type") or "") != "agent"
                    or str(payload.get("requested_name") or "") != name
                ):
                    raise AgentLocalRetirementError(
                        "operation journal identity does not match the retired agent"
                    )
                operation.retire_state_copies()
            except (OperationJournalError, OSError, ValueError) as exc:
                raise AgentLocalRetirementError(
                    f"could not retire exact operation state {operation_id}: {exc}"
                ) from exc

    # Empty alias parents are residue too. A sibling keeps the parent non-empty.
    for parent in (agent_dir.parent, tf_dir.parent):
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError as exc:
            raise AgentLocalRetirementError(
                f"could not prune empty agent state parent {parent}: {exc}"
            ) from exc
