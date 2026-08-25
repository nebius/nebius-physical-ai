"""Exact local-state retirement for an absent NPA agent generation."""

from __future__ import annotations

import os
import shutil
import tempfile
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
    targets = ((tf_dir, tf_dir.parent.parent), (agent_dir, agent_root))
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
        if os.path.lexists(target) and (target.is_symlink() or not target.is_dir()):
            raise AgentLocalRetirementError(
                f"agent local-state path is not a regular directory: {target}"
            )

    def retire_directory(target: Path) -> None:
        if os.path.lexists(target):
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

    auth_path = agent_dir / "auth.env"
    auth_snapshot: tuple[bytes, int] | None = None
    if os.path.lexists(auth_path):
        if auth_path.is_symlink() or not auth_path.is_file():
            raise AgentLocalRetirementError(
                f"agent authentication path is not a regular file: {auth_path}"
            )
        try:
            auth_snapshot = (
                auth_path.read_bytes(),
                auth_path.stat().st_mode & 0o600 or 0o600,
            )
        except OSError as exc:
            raise AgentLocalRetirementError(
                f"could not preserve agent authentication recovery data: {exc}"
            ) from exc

    def restore_auth_after_failed_retirement() -> None:
        """Restore the exact credential if recursive deletion failed midway."""

        if auth_snapshot is None:
            return
        data, mode = auth_snapshot
        if os.path.lexists(agent_dir) and (
            agent_dir.is_symlink() or not agent_dir.is_dir()
        ):
            raise AgentLocalRetirementError(
                "agent authentication directory changed during retirement"
            )
        agent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = -1
        temporary = ""
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=".auth.env.recovery-", dir=agent_dir
            )
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                os.fchmod(handle.fileno(), mode)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, auth_path)
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
            raise AgentLocalRetirementError(
                "agent authentication recovery could not be restored after a "
                f"partial local deletion: {exc}"
            ) from exc

    # Terraform and durable operation evidence must converge before the auth
    # directory is removed. Any injected or real failure therefore preserves
    # credentials and the remaining recovery material.
    retire_directory(tf_dir)

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

    try:
        retire_directory(agent_dir)
    except AgentLocalRetirementError:
        restore_auth_after_failed_retirement()
        raise

    # Empty alias parents are residue too. A sibling keeps the parent non-empty.
    for parent in (agent_dir.parent, tf_dir.parent):
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError as exc:
            raise AgentLocalRetirementError(
                f"could not prune empty agent state parent {parent}: {exc}"
            ) from exc
