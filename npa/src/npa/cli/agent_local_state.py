"""Exact local-state retirement for an absent NPA agent generation."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


class AgentLocalRetirementError(RuntimeError):
    """Exact agent cloud state is absent, but local retirement is incomplete."""


@dataclass(frozen=True)
class _RecoveryDirectory:
    relative_path: Path
    mode: int


@dataclass(frozen=True)
class _RecoveryFile:
    relative_path: Path
    data: bytes
    mode: int


@dataclass(frozen=True)
class AgentLocalRecoverySnapshot:
    """In-memory rollback material for the small owner-only agent state tree."""

    agent_dir: Path
    existed: bool
    directories: tuple[_RecoveryDirectory, ...] = ()
    files: tuple[_RecoveryFile, ...] = ()

    def restore(self) -> None:
        """Restore exact credential/recovery files after incomplete retirement."""

        if not self.existed:
            return
        if os.path.lexists(self.agent_dir) and (
            self.agent_dir.is_symlink() or not self.agent_dir.is_dir()
        ):
            raise AgentLocalRetirementError(
                "agent recovery directory changed during retirement"
            )
        expected = {
            item.relative_path for item in (*self.directories, *self.files)
        }
        if self.agent_dir.is_dir():
            for candidate in self.agent_dir.rglob("*"):
                relative = candidate.relative_to(self.agent_dir)
                if relative not in expected:
                    raise AgentLocalRetirementError(
                        "agent recovery directory gained unexpected state during "
                        f"retirement: {relative}"
                    )
        try:
            self.agent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.agent_dir.chmod(0o700)
            for directory in sorted(
                self.directories,
                key=lambda item: len(item.relative_path.parts),
            ):
                target = self.agent_dir / directory.relative_path
                if os.path.lexists(target) and (
                    target.is_symlink() or not target.is_dir()
                ):
                    raise AgentLocalRetirementError(
                        f"agent recovery path changed during retirement: {target}"
                    )
                target.mkdir(parents=True, exist_ok=True, mode=directory.mode)
                target.chmod(directory.mode)
            for saved_file in self.files:
                target = self.agent_dir / saved_file.relative_path
                if os.path.lexists(target):
                    if target.is_symlink() or not target.is_file():
                        raise AgentLocalRetirementError(
                            f"agent recovery path changed during retirement: {target}"
                        )
                    if target.read_bytes() != saved_file.data:
                        raise AgentLocalRetirementError(
                            "agent recovery file changed during retirement: "
                            f"{target}"
                        )
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor, temporary_path = tempfile.mkstemp(
                    prefix=f".{target.name}.recovery-", dir=target.parent
                )
                temporary = Path(temporary_path)
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        os.fchmod(handle.fileno(), saved_file.mode)
                        handle.write(saved_file.data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        except (OSError, AgentLocalRetirementError) as exc:
            if isinstance(exc, AgentLocalRetirementError):
                raise
            raise AgentLocalRetirementError(
                f"agent recovery state could not be restored: {exc}"
            ) from exc


def _capture_agent_local_recovery(agent_dir: Path) -> AgentLocalRecoverySnapshot:
    """Capture regular owner-only files before an irreversible directory delete."""

    if not os.path.lexists(agent_dir):
        return AgentLocalRecoverySnapshot(agent_dir=agent_dir, existed=False)
    directories: list[_RecoveryDirectory] = []
    files: list[_RecoveryFile] = []
    try:
        for candidate in sorted(agent_dir.rglob("*"), key=lambda path: path.parts):
            relative = candidate.relative_to(agent_dir)
            if candidate.is_symlink():
                raise AgentLocalRetirementError(
                    f"agent recovery path is a symlink: {candidate}"
                )
            if candidate.is_dir():
                directories.append(
                    _RecoveryDirectory(
                        relative_path=relative,
                        mode=candidate.stat().st_mode & 0o700 or 0o700,
                    )
                )
            elif candidate.is_file():
                files.append(
                    _RecoveryFile(
                        relative_path=relative,
                        data=candidate.read_bytes(),
                        mode=candidate.stat().st_mode & 0o600 or 0o600,
                    )
                )
            else:
                raise AgentLocalRetirementError(
                    f"agent recovery path is not a regular file or directory: {candidate}"
                )
    except OSError as exc:
        raise AgentLocalRetirementError(
            f"could not preserve agent recovery data: {exc}"
        ) from exc
    return AgentLocalRecoverySnapshot(
        agent_dir=agent_dir,
        existed=True,
        directories=tuple(directories),
        files=tuple(files),
    )


def cleanup_agent_local_files(
    project_alias: str,
    name: str,
    *,
    operation_ids: tuple[str, ...] = (),
) -> AgentLocalRecoverySnapshot:
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

    recovery_snapshot = _capture_agent_local_recovery(agent_dir)

    # Terraform and durable operation evidence must converge before the auth
    # directory is removed. Any injected or real failure therefore preserves
    # credentials and the remaining recovery material.
    try:
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

        retire_directory(agent_dir)

        # Empty alias parents are residue too. A sibling keeps the parent non-empty.
        for parent in (agent_dir.parent, tf_dir.parent):
            try:
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError as exc:
                raise AgentLocalRetirementError(
                    f"could not prune empty agent state parent {parent}: {exc}"
                ) from exc
    except BaseException as exc:
        try:
            recovery_snapshot.restore()
        except AgentLocalRetirementError as restore_exc:
            raise AgentLocalRetirementError(
                "local retirement failed and agent recovery state could not be "
                f"restored: {restore_exc}"
            ) from exc
        raise
    return recovery_snapshot
