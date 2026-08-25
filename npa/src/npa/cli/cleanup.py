"""`npa cleanup` — report and remove local NPA/SkyPilot residue after teardown.

`agent destroy`, `cluster down`, `storage bucket delete` and
`configure --forget-project` clear the cloud resources and their secrets, but a
few large, secret-free local caches survive with no single command to see or
remove them: the isolated SkyPilot venv (~500 MB), the Terraform provider cache,
SkyPilot's own `~/.sky` state (~100 MB), and empty per-alias state directories.

`npa cleanup` lists them (with sizes); `--yes` removes them. Existing `--yes`
semantics stay local and secret-free. The explicitly broader `--full --yes`
scope also removes the locally saved HF / Token Factory / NGC credentials and
prunes an empty NPA-owned config/tree after project teardown. Cloud IAM remains
a separate, ownership-checked storage command so a shared identity is never
silently removed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import typer

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract


@dataclass
class _Residue:
    label: str
    path: Path
    size: int
    device: int
    inode: int


@dataclass(frozen=True)
class _PrivateYamlSnapshot:
    """Secret-bearing local recovery document held only in memory."""

    label: str
    path: Path
    existed: bool
    document: dict[str, Any]


@dataclass(frozen=True)
class CleanupPhase:
    """One ordered, machine-readable NPA-only cleanup recommendation."""

    phase: int
    resource: str
    observed_state: str
    recommended_npa_command: str
    safety_status: str
    ownership_status: str
    operator_action_required: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "resource": self.resource,
            "observed_state": self.observed_state,
            "recommended_npa_command": self.recommended_npa_command,
            "safety_status": self.safety_status,
            "ownership_status": self.ownership_status,
            "operator_action_required": self.operator_action_required,
            "operator_action_remains": self.operator_action_required,
        }


def _snapshot_cleanup_recovery_documents() -> tuple[_PrivateYamlSnapshot, ...]:
    """Capture config/credentials before the final retirement transaction."""

    import yaml

    from npa.clients import config, credentials

    snapshots: list[_PrivateYamlSnapshot] = []
    for label, path in (
        ("project configuration", config.CONFIG_PATH),
        ("credential store", credentials.CREDENTIALS_PATH),
    ):
        if path.is_symlink():
            raise RuntimeError(f"{label} is a symlink")
        if not path.exists():
            snapshots.append(_PrivateYamlSnapshot(label, path, False, {}))
            continue
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(f"{label} cannot be snapshotted safely") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise RuntimeError(f"{label} is schema-invalid")
        snapshots.append(_PrivateYamlSnapshot(label, path, True, loaded))
    return tuple(snapshots)


def _restore_cleanup_recovery_documents(
    snapshots: tuple[_PrivateYamlSnapshot, ...],
    *,
    project_alias: str,
    project_id: str,
) -> tuple[list[str], list[dict[str, str]], bool, str]:
    """Restore recovery documents and return their refreshed safe inventory."""

    from copy import deepcopy

    from npa.clients.credentials import update_private_yaml

    failures: list[str] = []
    for snapshot in snapshots:
        try:
            update_private_yaml(
                snapshot.path,
                lambda _current, saved=snapshot: (
                    deepcopy(saved.document) if saved.existed else None
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(f"{snapshot.label}: {type(exc).__name__}")
    if not project_id:
        return failures, [], False, "no exact project recovery inventory requested"

    from npa.clients.config import ConfigError, resolve_environment
    from npa.clients.project_credential_store import project_credential_residue

    try:
        residue = project_credential_residue(project_id)
        restored_environment = (
            resolve_environment(project_alias) if project_alias else None
        )
        if (
            restored_environment is not None
            and str(restored_environment.project_id or "") == project_id
        ):
            residue.append(
                {
                    "path": f"config.projects.{project_alias}",
                    "class": "project_alias_or_default",
                }
            )
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        failures.append(f"recovery inventory: {type(exc).__name__}")
        return failures, [], True, "restored recovery inventory is unresolved"
    try:
        present, detail = _agent_operational_state_present(project_alias, project_id)
    except (OSError, RuntimeError, ValueError) as exc:
        present = True
        detail = f"restored inventory unresolved: {exc}"
    return failures, residue, present, detail


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file() and not child.is_symlink():
                    total += child.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.0f}TB"


def _empty_alias_dirs(npa_dir: Path, project: str) -> list[Path]:
    """Return empty per-alias state dirs under ~/.npa/{agents,workbenches}/.

    `agent destroy` removes the managed `<alias>/<name>/` tree; the `<alias>/`
    parent (and the `agents`/`workbenches` base) can be left behind empty.
    """
    found: list[Path] = []
    for base_name in ("agents", "workbenches"):
        base = npa_dir / base_name
        if not base.is_dir():
            continue
        for alias_dir in sorted(base.iterdir()):
            if not alias_dir.is_dir():
                continue
            if project and alias_dir.name != project:
                continue
            if not any(alias_dir.rglob("*")):
                found.append(alias_dir)
    return found


_FULL_TOKEN_KEYS = (
    "HF_TOKEN",
    "NEBIUS_TOKEN_FACTORY_KEY",
    "NGC_API_KEY",
    "NGC_ORG",
    "NGC_TEAM",
)
_SERVICE_CREDENTIAL_FIELDS = {
    "huggingface": ("token", "hf_token", "api_key", "key", "HF_TOKEN"),
    "token_factory": (
        "api_key",
        "apikey",
        "key",
        "token",
        "NEBIUS_TOKEN_FACTORY_KEY",
    ),
    "ngc": (
        "api_key",
        "apikey",
        "key",
        "token",
        "org",
        "organization",
        "team",
        "NGC_API_KEY",
        "NGC_ORG",
        "NGC_TEAM",
    ),
}


def _full_credential_labels() -> list[str]:
    """Return the full-cleanup credential groups present on disk."""

    import yaml

    from npa.clients.credentials import CREDENTIALS_PATH

    if not CREDENTIALS_PATH.exists():
        return []
    try:
        loaded = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError("local credential inventory is unreadable") from exc
    data = {} if loaded is None else loaded
    if not isinstance(data, dict):
        raise RuntimeError("local credential inventory root is schema-invalid")
    raw_tokens = data.get("tokens")
    if "tokens" in data and not isinstance(raw_tokens, dict):
        raise RuntimeError("local credential token inventory is schema-invalid")
    tokens = raw_tokens if isinstance(raw_tokens, dict) else {}
    for section_name in _SERVICE_CREDENTIAL_FIELDS:
        if section_name in data and not isinstance(data[section_name], dict):
            raise RuntimeError(
                f"local credential section {section_name!r} is schema-invalid"
            )
    labels: list[str] = []
    if (
        data.get("HF_TOKEN")
        or tokens.get("HF_TOKEN")
        or _section_has_any(data, "huggingface")
    ):
        labels.append("Hugging Face token")
    if (
        data.get("NEBIUS_TOKEN_FACTORY_KEY")
        or tokens.get("NEBIUS_TOKEN_FACTORY_KEY")
        or _section_has_any(data, "token_factory")
    ):
        labels.append("Token Factory key")
    if any(
        data.get(key) or tokens.get(key)
        for key in ("NGC_API_KEY", "NGC_ORG", "NGC_TEAM")
    ):
        labels.append("NGC credentials")
    elif _section_has_any(data, "ngc"):
        labels.append("NGC credentials")
    return labels


def _section_has_any(data: dict, section_name: str) -> bool:
    section = data.get(section_name)
    if not isinstance(section, dict):
        return False
    return any(
        section.get(key) not in (None, "")
        for key in _SERVICE_CREDENTIAL_FIELDS[section_name]
    )


def _clear_full_credentials() -> list[str]:
    """Remove only known shared-service credentials, preserving other data."""

    from copy import deepcopy

    from npa.clients.credentials import CREDENTIALS_PATH, update_private_yaml

    labels = _full_credential_labels()
    if not labels:
        return []

    def clear(existing: dict[str, Any]) -> dict[str, Any]:
        data = deepcopy(existing)
        for key in _FULL_TOKEN_KEYS:
            data.pop(key, None)
        tokens = data.get("tokens")
        if isinstance(tokens, dict):
            tokens = dict(tokens)
            for key in _FULL_TOKEN_KEYS:
                tokens.pop(key, None)
            if tokens:
                data["tokens"] = tokens
            else:
                data.pop("tokens", None)
        for section_name, fields in _SERVICE_CREDENTIAL_FIELDS.items():
            section = data.get(section_name)
            if not isinstance(section, dict):
                continue
            section = dict(section)
            for key in fields:
                section.pop(key, None)
            if section:
                data[section_name] = section
            else:
                data.pop(section_name, None)
        return data

    try:
        update_private_yaml(CREDENTIALS_PATH, clear)
    except (OSError, ValueError):
        return []
    return labels


def _yaml_file_is_empty(path: Path) -> bool:
    import yaml

    if not path.is_file():
        return False
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return _structure_is_empty(data)


def _structure_is_empty(value: object) -> bool:
    """Whether YAML data contains no meaningful scalar user state."""

    if value is None or value == "":
        return True
    if isinstance(value, dict):
        return all(_structure_is_empty(item) for item in value.values())
    if isinstance(value, list):
        return all(_structure_is_empty(item) for item in value)
    return False


def _npa_state_dir() -> Path:
    """Resolve the active local state root at operation time."""

    configured = os.environ.get("NPA_CONFIG_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".npa"


def _full_empty_state(npa_dir: Path) -> list[Path]:
    """Return empty, known NPA-owned state that full cleanup can prune."""

    found: list[Path] = []
    for path in (npa_dir / "config.yaml", npa_dir / "credentials.yaml"):
        if _yaml_file_is_empty(path):
            found.append(path)
    for base_name in ("agents", "workbenches", "clusters"):
        base = npa_dir / base_name
        if not base.is_dir():
            continue
        for path in sorted(
            (item for item in base.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                if not any(path.iterdir()):
                    found.append(path)
            except OSError:
                continue
        try:
            if not any(base.iterdir()):
                found.append(base)
        except OSError:
            continue
    try:
        if npa_dir.is_dir() and not any(npa_dir.iterdir()):
            found.append(npa_dir)
    except OSError:
        pass
    return list(dict.fromkeys(found))


def _prune_full_empty_state(npa_dir: Path) -> list[tuple[str, Path]]:
    """Prune only empty config/known dirs, then ~/.npa if truly empty."""

    removed: list[tuple[str, Path]] = []
    for label, path in (
        ("config file", npa_dir / "config.yaml"),
        ("credentials file", npa_dir / "credentials.yaml"),
    ):
        if not _yaml_file_is_empty(path):
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed.append((f"empty {label}", path))
    for base_name in ("agents", "workbenches", "clusters"):
        base = npa_dir / base_name
        if not base.is_dir():
            continue
        directories = sorted(
            [base, *(item for item in base.rglob("*") if item.is_dir())],
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for path in directories:
            try:
                if any(path.iterdir()):
                    continue
                path.rmdir()
            except OSError:
                continue
            removed.append(("empty state dir", path))
    try:
        if npa_dir.is_dir() and not any(npa_dir.iterdir()):
            npa_dir.rmdir()
            removed.append(("empty NPA home", npa_dir))
    except OSError:
        pass
    return removed


def _collect_residue(*, include_sky: bool) -> list[_Residue]:
    npa_dir = _npa_state_dir()
    residue: list[_Residue] = []
    for label, path in (
        ("SkyPilot venv", npa_dir / "skypilot-venv"),
        ("Terraform provider cache", npa_dir / "terraform-plugin-cache"),
    ):
        if path.exists() and not path.is_symlink():
            try:
                identity = path.stat(follow_symlinks=False)
            except OSError:
                continue
            residue.append(
                _Residue(label, path, _dir_size(path), identity.st_dev, identity.st_ino)
            )
    if include_sky:
        sky_home = Path(
            os.environ.get("NPA_SKY_STATE_DIR", "").strip() or (Path.home() / ".sky")
        )
        if sky_home.exists() and not sky_home.is_symlink():
            sky_identity: os.stat_result | None
            try:
                sky_identity = sky_home.stat(follow_symlinks=False)
            except OSError:
                sky_identity = None
            if sky_identity is not None:
                residue.append(
                    _Residue(
                        "SkyPilot state (~/.sky)",
                        sky_home,
                        _dir_size(sky_home),
                        sky_identity.st_dev,
                        sky_identity.st_ino,
                    )
                )
    return residue


def _shared_sky_preservation_reason(project_id: str = "") -> str:
    """Explain why controller ownership cannot authorize global-state deletion."""

    from npa.controller_ownership import controller_owner

    try:
        owner = controller_owner()
    except (OSError, RuntimeError, ValueError):
        owner = None
    owner_project = str(getattr(owner, "project_id", "") or "")
    selected = str(project_id or "")
    if not owner_project:
        scope = "no controller owner is recorded"
    elif selected and owner_project == selected:
        scope = "the selected project owns a controller"
    else:
        scope = "another or unselected project owns a controller"
    return (
        f"{scope}; managed-job audit and controller ownership do not prove exclusive "
        "ownership of machine-global SkyPilot state"
    )


def _remove_exact_residue(item: _Residue) -> str:
    """Remove only the inode inventoried by this cleanup run."""

    import shutil
    from uuid import uuid4

    try:
        current = item.path.stat(follow_symlinks=False)
        home = Path.home().resolve()
        parent = item.path.parent.resolve()
    except OSError as exc:
        return f"identity recheck failed: {exc}"
    if item.path.is_symlink() or not parent.is_relative_to(home):
        return "target is a symlink or escaped the operator home"
    if (current.st_dev, current.st_ino) != (item.device, item.inode):
        return "target inode changed after inventory"
    quarantine = item.path.parent / f".npa-cleanup-{item.inode}-{uuid4().hex}"
    try:
        item.path.rename(quarantine)
        moved = quarantine.stat(follow_symlinks=False)
        if (moved.st_dev, moved.st_ino) != (item.device, item.inode):
            if not os.path.lexists(item.path):
                quarantine.rename(item.path)
            return "target inode changed during atomic quarantine"
        if quarantine.is_dir():
            shutil.rmtree(quarantine)
        else:
            quarantine.unlink()
    except OSError as exc:
        return str(exc)
    if os.path.lexists(quarantine) or os.path.lexists(item.path):
        return "exact path was only partially removed or replaced during deletion"
    return ""


def _storage_iam_full_check(
    project: str,
    *,
    prune_verified_absence: bool,
) -> tuple[str, bool, str, str]:
    """Return a truthful read-only IAM outcome and whether cleanup is partial."""

    from npa.cli.storage import (
        _observe_storage_iam,
        _persist_storage_iam_observation,
        _remove_storage_service_account_record,
        _resolve_storage_iam_context,
        _storage_service_account_record,
        _untrusted_storage_account_ids,
    )
    from npa.clients.config import (
        ConfigError,
        storage_iam_residue,
        storage_iam_residues,
    )

    aliases = [project] if project else list(storage_iam_residues())
    if not aliases:
        record, _note = _storage_service_account_record()
        if record is None and not _untrusted_storage_account_ids():
            return (
                "Storage IAM: no saved identity evidence or unresolved project marker remains.",
                False,
                "fully_clean",
                "verified_terminal",
            )
        aliases = [""]

    messages: list[str] = []
    states: list[str] = []
    ownership_states: list[str] = []
    partial = False

    def retained_access_receipt_is_terminal(
        exact_project: str, marker: dict[str, Any]
    ) -> bool:
        from npa.teardown_receipts import (
            latest_resource_generation_events,
            teardown_event_authorizes_convergence,
        )

        account_id = str(marker.get("service_account_id") or "").strip()
        if not account_id:
            return False
        matching = [
            event
            for event in latest_resource_generation_events(
                project_id=exact_project, phase="storage_iam", strict=True
            )
            if isinstance(event.get("identity"), dict)
            and str(event["identity"].get("service_account_id") or "").strip()
            == account_id
        ]
        return bool(
            matching
            and all(
                isinstance(event.get("action"), dict)
                and event["action"].get("kind")
                == "preserve_unowned_account_remove_npa_access"
                and teardown_event_authorizes_convergence(event)
                for event in matching
            )
        )

    for alias in aliases:
        try:
            if alias.startswith("project-"):
                context = _resolve_storage_iam_context(project_id=alias)
            else:
                context = _resolve_storage_iam_context(alias)
            # Once exact storage lifecycle evidence is terminal and removed, a
            # project-scoped cleanup must not rediscover a pre-existing/shared
            # account merely because it has NPA's familiar default name.
            marker = storage_iam_residue(context.alias) if context.alias else {}
            record, _note = _storage_service_account_record(
                project_id=context.project_id
            )
            untrusted_ids = _untrusted_storage_account_ids(context.project_id)
            if (
                marker
                and record is None
                and not untrusted_ids
                and retained_access_receipt_is_terminal(context.project_id, marker)
            ):
                ownership_states.append("retained_shared")
                if prune_verified_absence and context.alias:
                    from npa.clients.config import clear_storage_iam_residue

                    clear_storage_iam_residue(
                        context.alias,
                        account_id=str(marker.get("service_account_id") or ""),
                    )
                    states.append("fully_cleaned")
                    messages.append(
                        "Storage IAM: removed a stale local marker after exact terminal "
                        "access-cleanup receipt verification; the shared account remains."
                    )
                else:
                    states.append("partial_local_cleanup")
                    partial = True
                    messages.append(
                        "Storage IAM: exact access cleanup is terminal and the shared "
                        "account remains; a stale local marker awaits full local cleanup."
                    )
                continue
            if not marker and record is None and not untrusted_ids:
                ownership_states.append("verified_terminal")
                messages.append(
                    "Storage IAM: no saved identity evidence or unresolved project "
                    "marker remains; unrelated provider identities were not searched by name."
                )
                states.append(
                    "fully_cleaned" if prune_verified_absence else "fully_clean"
                )
                continue
            observation = _observe_storage_iam(context)
            if prune_verified_absence or not observation.verified_absent:
                _persist_storage_iam_observation(observation)
        except ConfigError as exc:
            record, _note = _storage_service_account_record()
            ownership_states.append("owned" if record is not None else "unknown")
            messages.append(
                "Storage IAM: verification could not retain project evidence; "
                f"cleanup is partial: {exc}"
            )
            states.append("partial_verification_failure")
            partial = True
            continue
        if observation.outcome == "verification_failed":
            ownership_states.append(
                "owned" if observation.ownership == "npa" else "pending_verification"
            )
            messages.append(
                "Storage IAM: provider/auth verification failure for project "
                f"{context.alias or context.project_id}; cleanup is partial and "
                f"the residue marker was preserved: {observation.detail}"
            )
            states.append("partial_verification_failure")
            partial = True
            continue
        if observation.retained_account_access_delete_planned:
            ownership_states.append("retained_shared")
            messages.append(
                "Storage IAM: the pre-existing/shared account was preserved, but "
                "exact NPA-created access state still requires the guarded storage "
                "service-account cleanup command."
            )
            states.append("locally_clean_cloud_iam_unresolved")
            partial = True
            continue
        if observation.retained_account_access_resolved:
            ownership_states.append("retained_shared")
            if prune_verified_absence:
                from npa.clients.config import clear_storage_iam_residue

                if not _remove_storage_service_account_record(observation.account_id):
                    messages.append(
                        "Storage IAM: run-scoped access is absent, but the stale local "
                        "storage generation could not be retired; fix permissions and retry."
                    )
                    states.append("partial_local_cleanup")
                    partial = True
                    continue
                if context.alias:
                    clear_storage_iam_residue(
                        context.alias, account_id=observation.account_id
                    )
            messages.append(
                "Storage IAM: exact run-scoped access state is absent; the "
                "pre-existing/shared service account was preserved."
            )
            states.append("fully_cleaned" if prune_verified_absence else "fully_clean")
            continue
        if observation.present:
            ownership_states.append(
                "owned" if observation.ownership == "npa" else "pending_verification"
            )
            ownership = (
                "guarded NPA ownership provenance is present"
                if observation.ownership == "npa"
                else "ownership is unresolved"
            )
            messages.append(
                "Storage IAM: verified present — exact service account "
                f"{observation.account_name} ({observation.account_id}) remains in "
                f"{context.project_id}; {ownership}. It was left untouched and the "
                "project was preserved. The ordered phase model below supplies the "
                "supported NPA reconciliation/deletion command."
            )
            states.append("locally_clean_cloud_iam_unresolved")
            partial = True
            continue
        ownership_states.append("verified_terminal")
        record, _note = _storage_service_account_record()
        if (
            prune_verified_absence
            and record is not None
            and observation.account_id == record.account_id
            and not _remove_storage_service_account_record(record.account_id)
        ):
            messages.append(
                "Storage IAM: verified absence, but stale local ownership provenance "
                "could not be removed; fix ~/.npa permissions and retry."
            )
            states.append("partial_local_cleanup")
            partial = True
            continue
        messages.append(
            "Storage IAM: verified absence — the exact identity is not present; "
            "its unresolved project marker was cleared."
        )
        states.append("fully_cleaned" if prune_verified_absence else "fully_clean")

    status = (
        "partial_verification_failure"
        if "partial_verification_failure" in states
        else "locally_clean_cloud_iam_unresolved"
        if "locally_clean_cloud_iam_unresolved" in states
        else "partial_local_cleanup"
        if "partial_local_cleanup" in states
        else "fully_cleaned"
        if "fully_cleaned" in states
        else "fully_clean"
    )
    ownership_state = (
        "pending_verification"
        if "pending_verification" in ownership_states
        else "owned"
        if "owned" in ownership_states
        else "retained_shared"
        if "retained_shared" in ownership_states
        else "unknown"
        if "unknown" in ownership_states
        else "verified_terminal"
    )
    return "\n".join(messages), partial, status, ownership_state


_CLOUD_CLEANUP_TERMINAL_STATES = frozenset(
    {
        "already_absent",
        "cancelled",
        "completed",
        "deleted",
        "not_submitted",
        "operator_attested",
        "terminal",
        "verified_absent",
        "verified_deleted",
    }
)
_CLOUD_CLEANUP_REQUIRED_GROUPS = (
    ("workflow_audit", "workflow", "project_destroy_workflows"),
    ("agent", "project_destroy_agents"),
    ("bucket", "project_destroy_bucket"),
    ("storage_iam", "project_destroy_storage_iam"),
)
_CLOUD_CLEANUP_OPTIONAL_GROUPS = (
    ("controller", "project_destroy_controller"),
    ("cluster", "project_destroy_clusters"),
)
_CLOUD_CLEANUP_PHASE_GROUPS = (
    *_CLOUD_CLEANUP_REQUIRED_GROUPS,
    *_CLOUD_CLEANUP_OPTIONAL_GROUPS,
)
_AGENT_TERRAFORM_GRAPH = frozenset(
    {
        "compute_instance",
        "boot_disk",
        "network",
        "subnet",
        "security_group",
        "public_ip",
    }
)


def _agent_operational_state_present(
    project_alias: str, project_id: str
) -> tuple[bool, str]:
    """Inventory exact local agent state without treating audit journals as live."""

    from npa.cli.agent_records import resolve_project_agents
    from npa.deploy import provisioner
    from npa.provisioning_journal import TERMINAL_PHASES, list_operations

    alias = str(project_alias or "").strip()
    exact_project = str(project_id or "").strip()
    if alias and resolve_project_agents(alias):
        return True, "saved agent record(s) remain"
    roots: list[Path] = []
    auth_root = Path.home() / ".npa" / "agents"
    workbench_root = provisioner.working_dir_path(
        alias, ".cleanup-inventory"
    ).parent
    if alias:
        roots.extend((auth_root / alias, workbench_root))
    else:
        roots.extend((auth_root, workbench_root))
    for root in roots:
        if not os.path.lexists(root):
            continue
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"agent local-state root {root} is not a directory")
        try:
            if any(root.iterdir()):
                return True, f"agent local-state tree remains at {root}"
        except OSError as exc:
            raise RuntimeError(f"agent local-state inventory failed at {root}") from exc
    for operation in list_operations(
        project_alias=alias,
        project_id=exact_project,
        resource_type="agent",
        strict=True,
    ):
        payload = operation.read()
        audit_only = payload.get("audit_only") is True
        if operation.state_copies() or not (
            audit_only and str(payload.get("phase") or "") in TERMINAL_PHASES
        ):
            return True, f"operational agent journal {operation.operation_id} remains"
    return False, "no local operational agent state remains"


def _unscoped_project_state_present() -> tuple[bool, str]:
    """Fail-closed inventory when full cleanup has no immutable project scope."""

    import yaml

    from npa.clients.config import list_projects
    from npa.clients.credentials import CREDENTIALS_PATH

    projects = list_projects()
    if projects:
        return True, "configured project scope exists but no project was selected"
    present, detail = _agent_operational_state_present("", "")
    if present:
        return True, detail
    if not CREDENTIALS_PATH.exists():
        return False, "no project-scoped local state exists"
    try:
        loaded = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError("unscoped credential inventory is unreadable") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("unscoped credential inventory is schema-invalid")
    agent_iam = loaded.get("agent_iam")
    agent_projects = (
        agent_iam.get("projects") if isinstance(agent_iam, dict) else None
    )
    if isinstance(agent_projects, dict) and agent_projects:
        return True, "agent IAM recovery records exist without selected project scope"
    project_credentials = loaded.get("project_credentials")
    credential_projects = (
        project_credentials.get("projects")
        if isinstance(project_credentials, dict)
        else None
    )
    if isinstance(credential_projects, dict) and credential_projects:
        return True, "project credential records exist without selected project scope"
    storage_iam = loaded.get("storage_iam")
    if isinstance(storage_iam, dict) and storage_iam:
        return True, "storage IAM recovery records exist without selected project scope"
    return False, "no project-scoped local state exists"


def _phase_group_events(
    phase_states: dict[str, dict[str, object]], names: tuple[str, ...]
) -> list[dict[str, object]]:
    return [phase_states[name] for name in names if phase_states.get(name)]


def _newest_phase_group_is_terminal(
    phase_states: dict[str, dict[str, object]], names: tuple[str, ...]
) -> bool:
    events = _phase_group_events(phase_states, names)
    if not events:
        return False
    receipt_ids = {str(event.get("_receipt_id") or "") for event in events}
    if len(receipt_ids) == 1:
        # Alternative workflow evidence in one receipt has an honest local
        # clock. A later audit may supersede an earlier workflow probe.
        newest_sequence = max(int(event.get("sequence") or 0) for event in events)
        events = [
            event
            for event in events
            if int(event.get("sequence") or 0) == newest_sequence
        ]
    # Across receipt files there is no shared sequence clock: every current
    # generation must converge independently.
    return all(_event_authorizes_cloud_absence(event) for event in events)


def _event_authorizes_cloud_absence(event: Mapping[str, object]) -> bool:
    """Require phase-specific structured convergence, never a terminal word."""
    from npa.teardown_receipts import teardown_event_authorizes_convergence

    return teardown_event_authorizes_convergence(event)


def _cloud_cleanup_receipts_are_terminal(
    phase_states: dict[str, dict[str, object]],
) -> bool:
    """Accept the newest monolithic or equivalent exact cleanup evidence."""

    if not all(
        _newest_phase_group_is_terminal(phase_states, names)
        for names in _CLOUD_CLEANUP_REQUIRED_GROUPS
    ):
        return False
    # Optional controller/cluster phases are not applicable to an agent-only
    # fresh-config run. If NPA has any such evidence, however, uncertainty must
    # still block credential and alias retirement.
    return all(
        not _phase_group_events(phase_states, names)
        or _newest_phase_group_is_terminal(phase_states, names)
        for names in _CLOUD_CLEANUP_OPTIONAL_GROUPS
    )


def _agent_lifecycle_allows_project_retirement(
    project_alias: str, project_id: str, *, retire: bool = False
) -> tuple[bool, str]:
    """Reconcile surviving agent generations before forgetting their credentials.

    Receipts are historical evidence, not an inventory. A reused alias/name may
    already have a newer agent record, provisioning operation, or Terraform
    state. Only one immutable state identity that matches complete graph-absence
    evidence and an exact provider NotFound may be treated as stale.
    """

    from npa.cli import agent as agent_module
    from npa.cli.agent_terraform import AgentTerraformStateIdentityError
    from npa.clients.nebius import NebiusError, get_compute_instance_identity
    from npa.deploy import provisioner
    from npa.provisioning_journal import (
        ProvisioningOperation,
        list_operations,
        operation_context,
    )
    from npa.teardown_receipts import list_teardown_receipts

    alias = str(project_alias or "").strip()
    exact_project = str(project_id or "").strip()
    if not exact_project:
        return False, "an exact immutable project ID is required"

    if not alias:
        # Once an alias is gone, an unbound local Terraform tree cannot be
        # proven unrelated to the exact project. Preserve its credentials. A
        # completely absent tree, by contrast, is positive local evidence that
        # there is no surviving alias/name generation to reconcile.
        workbench_base = provisioner.working_dir_path("", ".cleanup-inventory").parent
        auth_base = Path.home() / ".npa" / "agents"
        try:
            for local_root in (workbench_base, auth_base):
                if not local_root.exists():
                    continue
                if local_root.is_symlink() or not local_root.is_dir():
                    return False, "an agent local-state root is not a regular directory"
                if any(local_root.iterdir()):
                    return False, (
                        "alias-free agent local state cannot be bound to the "
                        "exact project"
                    )
        except OSError as exc:
            return False, f"agent Terraform state inventory is unreadable: {exc}"

        try:
            for operation in list_operations(
                project_id=exact_project, resource_type="agent", strict=True
            ):
                payload = operation.read()
                if not (
                    payload.get("audit_only") is True
                    and str(payload.get("phase") or "")
                    in {"committed", "destroyed", "rolled-back"}
                    and not operation.state_copies()
                ):
                    return False, (
                        "alias-free agent provisioning state remains for the "
                        "exact project"
                    )
        except (OSError, RuntimeError, ValueError) as exc:
            return False, f"provisioning journal inventory is unreadable: {exc}"
        return True, "no alias-bound agent lifecycle evidence survives"

    try:
        records = agent_module.resolve_project_agents(alias)
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"saved agent records are unreadable: {exc}"
    if not isinstance(records, dict):
        return False, "saved agent records are schema-invalid"

    names: set[str] = set()
    for saved_name in records:
        if (
            not isinstance(saved_name, str)
            or not saved_name.strip()
            or saved_name != saved_name.strip()
        ):
            return False, "a saved agent record has an invalid deployment name"
        names.add(saved_name)
    local_names: set[str] = set()
    workbench_root = provisioner.working_dir_path(alias, ".cleanup-inventory").parent
    try:
        auth_root = Path.home() / ".npa" / "agents" / alias
        for local_root in (workbench_root, auth_root):
            if not local_root.exists():
                continue
            if local_root.is_symlink() or not local_root.is_dir():
                return False, "an agent local-state root is not a regular directory"
            for child in local_root.iterdir():
                if child.is_symlink():
                    return False, "an agent local-state path is a symlink"
                if not child.is_dir():
                    return False, "an agent local-state entry is not a directory"
                # Even an empty per-agent directory is current operational
                # presence.  It may only be retired after the name is bound to
                # one exact historical generation and provider absence.
                local_names.add(child.name)
    except OSError as exc:
        return False, f"agent Terraform state inventory is unreadable: {exc}"
    names.update(local_names)

    operations_by_name: dict[str, list[dict[str, object]]] = {}
    try:
        for operation in list_operations(resource_type="agent", strict=True):
            payload = operation.read()
            saved_alias = str(payload.get("project_alias") or "")
            saved_project = str(payload.get("project_id") or "")
            if saved_alias != alias and saved_project != exact_project:
                continue
            if saved_alias and saved_alias != alias:
                return False, "an agent journal conflicts with the selected alias"
            if saved_project and saved_project != exact_project:
                return False, "an agent journal conflicts with the selected project"
            name = str(payload.get("requested_name") or "").strip()
            if not name:
                return False, "an agent journal has no requested deployment name"
            names.add(name)
            operations_by_name.setdefault(name, []).append(
                {**dict(payload), "_operation_id": operation.operation_id}
            )
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"provisioning journal inventory is unreadable: {exc}"

    terminal_graphs: dict[tuple[str, str], dict[str, object]] = {}
    try:
        receipts = list_teardown_receipts(
            project_id=exact_project, legacy="exclude", strict=True
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"agent teardown receipts are unreadable: {exc}"
    for receipt in receipts:
        for event in receipt.get("events") or []:
            if not isinstance(event, Mapping) or event.get("phase") != "agent":
                continue
            identity = event.get("identity")
            identity = identity if isinstance(identity, Mapping) else {}
            verification = event.get("verification")
            verification = verification if isinstance(verification, Mapping) else {}
            action = event.get("action")
            action = action if isinstance(action, Mapping) else {}
            graph = verification.get("terraform_dependency_graph")
            errors = event.get("errors")
            name = str(identity.get("agent_name") or event.get("resource") or "").strip()
            instance_id = str(identity.get("instance_id") or "").strip()
            if not (
                name
                and instance_id
                and str(event.get("terminal_state") or "").lower()
                in {"verified_absent", "verified_deleted"}
                and action.get("kind") == "terraform_agent_destroy"
                and verification.get("exact_instance_absent") is True
                and verification.get("terraform_destroy_completed") is True
                and isinstance(graph, list)
                and _AGENT_TERRAFORM_GRAPH.issubset(graph)
                and isinstance(errors, list)
                and not errors
            ):
                continue
            key = (name, instance_id)
            prior = terminal_graphs.get(key, {})
            if int(event.get("sequence") or 0) > int(prior.get("sequence") or 0):
                terminal_graphs[key] = dict(event)

    retirements: list[tuple[str, str, bool, tuple[str, ...], str]] = []
    for name in sorted(names):
        record_present = name in records
        if record_present:
            from npa.cli.agent_records import AgentRecordState, decode_agent_record

            try:
                decoded = decode_agent_record(alias, name)
            except (OSError, RuntimeError, ValueError) as exc:
                return False, f"saved agent record {name!r} is unreadable: {exc}"
            if decoded.state is not AgentRecordState.COMPLETE:
                return False, (
                    f"saved agent record {name!r} is {decoded.state.value}: "
                    f"{decoded.detail}"
                )
            record = decoded.record
        else:
            record = {}
        saved_project = record.get("project_id")
        record_project = str(saved_project or "").strip()
        if record_project != exact_project and record_present:
            return False, f"saved agent record {name!r} conflicts with the project"
        saved_instance = record.get("instance_id")

        try:
            state_exists = agent_module._agent_terraform_state_exists(alias, name)
        except (OSError, RuntimeError, ValueError) as exc:
            return False, f"agent {name!r} Terraform inventory is unreadable: {exc}"
        instance_ids = {
            value
            for value in (str(saved_instance or "").strip(),)
            if value
        }
        if state_exists:
            try:
                instance_ids.add(
                    agent_module._agent_terraform_instance_id(alias, name)
                )
            except AgentTerraformStateIdentityError as exc:
                return False, f"agent {name!r} Terraform identity is ambiguous: {exc}"

        operations = operations_by_name.get(name, [])
        active_operations = [
            operation
            for operation in operations
            if str(operation.get("phase") or "") not in {"destroyed", "rolled-back"}
        ]
        if len(active_operations) > 1:
            return False, (
                f"agent {name!r} has multiple current same-name operation generations"
            )
        has_local_files = name in local_names
        if not (
            record_present or state_exists or active_operations or has_local_files
        ):
            continue
        if not instance_ids:
            receipt_instance_ids = {
                instance_id
                for candidate_name, instance_id in terminal_graphs
                if candidate_name == name
            }
            if len(receipt_instance_ids) == 1:
                instance_ids.update(receipt_instance_ids)
        if len(instance_ids) != 1:
            return False, f"agent {name!r} has no single immutable instance identity"
        instance_id = next(iter(instance_ids))
        terminal = terminal_graphs.get((name, instance_id))
        if not terminal:
            return False, (
                f"agent {name!r} has surviving lifecycle state without matching "
                "complete terminal graph evidence"
            )
        for operation in active_operations:
            compute_ids = {
                str(resource.get("provider_id") or "").strip()
                for resource in operation.get("resources") or []
                if isinstance(resource, Mapping)
                and resource.get("resource_type") == "compute_instance"
                and str(resource.get("provider_id") or "").strip()
            }
            if compute_ids != {instance_id}:
                return False, (
                    f"agent {name!r} has a current operation that is not bound "
                    "to the terminal receipt generation"
                )

        identity = terminal.get("identity")
        identity = identity if isinstance(identity, Mapping) else {}
        try:
            remote = get_compute_instance_identity(
                instance_id,
                project_id=exact_project,
                expected_name=f"agent-{alias}-{name}",
                profile=str(identity.get("profile") or "") or None,
            )
        except NebiusError as exc:
            return False, f"agent {name!r} provider absence is unresolved: {exc}"
        if remote is not None:
            return False, f"agent {name!r} is still present at the provider"

        retirements.append(
            (
                name,
                instance_id,
                record_present,
                tuple(
                    str(operation.get("_operation_id") or "")
                    for operation in operations
                    if str(operation.get("_operation_id") or "")
                ),
                str(identity.get("profile") or ""),
            )
        )

    if retire:
        for name, instance_id, expected_record, operation_ids, profile in retirements:
            transaction = ProvisioningOperation.prepare(
                command="npa cleanup agent-local-retirement",
                project_alias=alias,
                project_id=exact_project,
                resource_type="agent-teardown",
                requested_name=name,
                ownership_source="cleanup-agent-local-retirement",
                resume_command="",
                resume_argv=(
                    "npa",
                    "cleanup",
                    "--project",
                    alias,
                    "--full",
                    "--yes",
                ),
            )
            try:
                with operation_context(transaction):
                    if str(transaction.read().get("phase") or "") == "prepared":
                        transaction.transition("mutating")
                    from npa.cli.agent_records import AgentRecordState

                    current = agent_module.decode_agent_record(alias, name)
                    if current.present != expected_record:
                        raise RuntimeError(
                            "saved agent record presence changed during retirement"
                        )
                    if current.present and (
                        current.state is not AgentRecordState.COMPLETE
                        or current.record.get("project_id") != exact_project
                        or current.record.get("instance_id") != instance_id
                    ):
                        raise RuntimeError(
                            "saved agent generation changed during retirement"
                        )
                    current_operations = list_operations(
                        project_alias=alias,
                        project_id=exact_project,
                        resource_type="agent",
                        requested_name=name,
                        strict=True,
                    )
                    current_ids = tuple(
                        sorted(item.operation_id for item in current_operations)
                    )
                    if current_ids != tuple(sorted(operation_ids)):
                        raise RuntimeError(
                            "agent operation generation changed during retirement"
                        )
                    if agent_module._agent_terraform_state_exists(alias, name):
                        current_instance = agent_module._agent_terraform_instance_id(
                            alias, name
                        )
                        if current_instance != instance_id:
                            raise RuntimeError(
                                "Terraform state generation changed during retirement"
                            )
                    remote = get_compute_instance_identity(
                        instance_id,
                        project_id=exact_project,
                        expected_name=f"agent-{alias}-{name}",
                        profile=profile or None,
                    )
                    if remote is not None:
                        raise RuntimeError(
                            "exact provider instance reappeared during retirement"
                        )
                    agent_module._cleanup_agent_local_files(
                        alias, name, operation_ids=operation_ids
                    )
                    if expected_record:
                        agent_module._remove_agent_record(alias, name)
                        if agent_module.decode_agent_record(alias, name).present:
                            raise RuntimeError(f"saved agent record {name!r} remains")
                    transaction.transition("destroyed")
            except (OSError, RuntimeError, ValueError) as exc:
                phase = str(transaction.read().get("phase") or "")
                if phase not in {"committed", "destroyed", "rolled-back"}:
                    transaction.transition(
                        "recovery-required",
                        error="exact local agent retirement did not converge",
                    )
                return False, f"agent {name!r} local retirement failed: {exc}"
        try:
            present, detail = _agent_operational_state_present(alias, exact_project)
        except (OSError, RuntimeError, ValueError) as exc:
            return False, f"final agent local-state inventory failed: {exc}"
        if present:
            return False, detail

    return True, "surviving agent lifecycle evidence is absent or exactly stale"


def cleanup_phase_model(
    *,
    jobs: list[str],
    jobs_note: str,
    iam_state: str,
    iam_ownership_state: str,
    local_state: str,
    receipt_phases: dict[str, dict[str, object]] | None = None,
) -> list[CleanupPhase]:
    """Derive every text/JSON recommendation from one deterministic model."""

    if jobs:
        workflow_state = "active_managed_jobs:" + ",".join(jobs)
        workflow_action = True
        workflow_safety = "exact_run_resolution_required"
    elif jobs_note:
        workflow_state = "verification_unavailable:" + jobs_note
        workflow_action = True
        workflow_safety = "repeat_safe_but_verification_required"
    else:
        workflow_state = "verified_no_nonterminal_jobs"
        workflow_action = False
        workflow_safety = "verified_repeat_safe_noop"

    iam_unresolved = iam_state in {
        "locally_clean_cloud_iam_unresolved",
        "partial_verification_failure",
        "partial_local_cleanup",
    }
    iam_verified = iam_state in {"fully_clean", "fully_cleaned"}
    iam_owned = iam_ownership_state == "owned"
    iam_command = (
        "npa storage service-account delete --project <alias> --dry-run"
        if iam_owned or iam_verified
        else "npa storage service-account reconcile --project <alias> "
        "--id <exact-id> --dry-run"
    )
    iam_observed = (
        "verified_deleted_or_absent"
        if iam_verified
        else iam_state
        if iam_state != "not_checked"
        else "not_checked"
    )
    iam_ownership = (
        "verified_terminal"
        if iam_verified
        else "owned_pending_delete"
        if iam_owned
        else iam_ownership_state
    )
    local_clean = local_state in {"fully_clean", "fully_cleaned"}
    receipt_phases = dict(receipt_phases or {})

    def completed(phase: str) -> bool:
        from npa.teardown_receipts import TERMINAL_STATES

        event = receipt_phases.get(phase) or {}
        if phase in {
            name for group in _CLOUD_CLEANUP_PHASE_GROUPS for name in group
        }:
            return _event_authorizes_cloud_absence(event)
        state = str(event.get("terminal_state") or "")
        return state in TERMINAL_STATES

    def observed(phase: str, fallback: str) -> str:
        event = receipt_phases.get(phase) or {}
        state = str(event.get("terminal_state") or "")
        recorded = str(event.get("recorded_at") or "")
        return f"receipt:{state}@{recorded}" if state else fallback

    workflow_receipt_complete = completed("workflow_audit") or completed("workflow")
    if not jobs and workflow_receipt_complete:
        workflow_action = False
        workflow_safety = "durable_terminal_or_absent_evidence"
    workflow_complete = not workflow_action
    agent_complete = completed("agent")
    cluster_complete = completed("cluster")
    # Deleting the exact whole cluster is also conclusive absence for any
    # Kubernetes-hosted shared controller workload. Local cache removal remains
    # represented independently by the local_cleanup phase.
    controller_complete = completed("controller") or cluster_complete
    bucket_complete = completed("bucket")
    project_complete = completed("project_config")
    local_complete = completed("local_cleanup") or local_clean

    return [
        CleanupPhase(
            1,
            "workflow runs",
            observed("workflow_audit", workflow_state),
            "npa workflow cancel <run-id> --project <alias> --json",
            workflow_safety,
            "not_applicable",
            workflow_action or not workflow_complete,
        ),
        CleanupPhase(
            2,
            "agent VM",
            observed("agent", "not_checked_by_local_cleanup"),
            "npa agent destroy --project <alias> --name <name> --yes",
            "repeat_safe_provider_verification",
            "npa_managed_identity_required",
            not agent_complete,
        ),
        CleanupPhase(
            3,
            "SkyPilot controller",
            observed(
                "controller",
                "ready_after_workflows_terminal"
                if not workflow_action
                else "workflow_gate_pending",
            ),
            "npa skypilot cleanup-controller --yes",
            "requires_no_nonterminal_managed_jobs",
            "not_applicable",
            not controller_complete,
        ),
        CleanupPhase(
            4,
            "Kubernetes cluster",
            observed("cluster", "not_checked_by_local_cleanup"),
            "npa cluster down --force",
            "pdb_aware_best_effort_convergence",
            "npa_managed_cluster_required",
            not cluster_complete,
        ),
        CleanupPhase(
            5,
            "object-storage bucket",
            observed("bucket", "not_checked_by_local_cleanup"),
            "npa storage bucket delete --project <alias> --yes --wait",
            "preserves_non_secret_iam_tombstone",
            "bucket_identity_verified_by_npa",
            not bucket_complete,
        ),
        CleanupPhase(
            6,
            "storage IAM",
            iam_observed,
            iam_command,
            "guarded_exact_id_reconciliation",
            iam_ownership,
            not iam_verified,
        ),
        CleanupPhase(
            7,
            "storage IAM deletion",
            iam_observed,
            "npa storage service-account delete --project <alias> --yes",
            "exact_identity_and_ownership_guarded",
            iam_ownership,
            not iam_verified,
        ),
        CleanupPhase(
            8,
            "project configuration",
            observed(
                "project_config",
                "blocked_by_iam" if iam_unresolved else "eligible_after_cloud_cleanup",
            ),
            "npa configure --forget-project <alias>",
            "refuses_unresolved_storage_iam",
            iam_ownership,
            not project_complete,
        ),
        CleanupPhase(
            9,
            "local caches and known credentials",
            local_state,
            "npa cleanup --full --yes",
            "local_only_preserves_unrelated_state",
            iam_ownership,
            not local_complete,
        ),
    ]


def _nonterminal_jobs(sky_bin: str = "") -> tuple[list[str], str, str]:
    """Return non-terminal IDs, lookup detail, and the verified queue state."""

    from npa.orchestration.skypilot._bin import SkyPilotNotInstalledError
    from npa.orchestration.skypilot.cleanup import (
        NONTERMINAL_JOB_STATUSES,
        _all_jobs,
        _job_statuses,
    )

    try:
        snapshot = _all_jobs(
            isolated_config_dir=None, config_path=None, sky_bin=sky_bin or None
        )
        nonterminal = sorted(
            job_id
            for job_id, status in _job_statuses(snapshot.jobs).items()
            if status in NONTERMINAL_JOB_STATUSES
        )
        state = (
            "verified_empty"
            if snapshot.state == "verified_empty"
            else "verified_active_jobs"
            if nonterminal
            else "verified_terminal_only"
        )
        return nonterminal, "", state
    except SkyPilotNotInstalledError:
        return (
            [],
            "SkyPilot is not installed, so managed jobs were not checked",
            "unreadable_or_unverified",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return (
            [],
            f"could not read the managed-job queue: {exc}",
            "unreadable_or_unverified",
        )


def _report_managed_jobs(jobs: list[str], note: str) -> None:
    if note:
        typer.echo(f"Managed jobs: {note}")
        return
    if not jobs:
        typer.echo("Managed jobs: none non-terminal.")
        return
    typer.echo(f"Managed jobs still non-terminal: {', '.join(jobs)}")
    typer.echo(
        "  These block `npa skypilot cleanup-controller`. A job whose pod cannot "
        "start stays PENDING forever rather than failing, so check it before "
        "assuming it is still doing work."
    )


def _print_runbook(phases: list[CleanupPhase]) -> None:
    typer.echo("")
    typer.echo(
        "Full teardown order (printed only; cleanup never implies the preceding cloud steps):"
    )
    for item in phases:
        action = (
            "operator action remains"
            if item.operator_action_required
            else "verified no-op"
        )
        typer.echo(
            f"  {item.phase}. {item.recommended_npa_command}  "
            f"[{item.resource}: {item.observed_state}; "
            f"{item.safety_status.replace('_', '-')}; {action}]"
        )


def _iam_note() -> str:
    """A hint about cloud IAM leftovers npa deliberately does not delete."""
    generic = (
        "Cloud IAM (not removed): pre-existing service accounts are left in place; "
        "the ordered cleanup model below selects the guarded exact-ID NPA path."
    )
    try:
        import yaml

        from npa.clients.credentials import CREDENTIALS_PATH

        if CREDENTIALS_PATH.exists():
            data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
            nebius = (data or {}).get("nebius") or {}
            storage_iam = (data or {}).get("storage_iam") or {}
            ownership = (
                storage_iam
                if str(storage_iam.get("service_account_managed_by", "") or "") == "npa"
                else nebius
            )
            owned_sa_id = str(ownership.get("service_account_id", "") or "").strip()
            if (
                owned_sa_id
                and str(ownership.get("service_account_managed_by", "") or "") == "npa"
            ):
                return (
                    f"Cloud IAM (not removed here): NPA recorded creating storage "
                    f"principal {owned_sa_id}; its guarded cleanup remains visible below."
                )
            sa_id = str(nebius.get("service_account_id", "") or "").strip()
            if sa_id:
                return (
                    f"Cloud IAM (not removed): the storage principal {sa_id} and any "
                    "pre-existing service accounts remain — deleting a shared SA can "
                    "break other work. The exact-ID reconciliation phase remains visible below."
                )
    except Exception:  # noqa: BLE001 - the note is best-effort
        return generic
    return generic


@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def cleanup_cmd(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help=(
            "Remove the local caches (otherwise just report). Local only: this never "
            "deletes cloud resources -- see the printed runbook for those."
        ),
    ),
    include_sky: bool = typer.Option(
        True,
        "--include-sky/--keep-sky",
        help=(
            "Include machine-shared ~/.sky in the audit. It is always preserved; "
            "project teardown removes only separately isolated, affirmatively owned state."
        ),
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help=(
            "Broaden --yes to also remove locally saved HF, Token Factory, and NGC "
            "credentials, validated NPA Terraform residue, and empty config/known "
            "~/.npa state. Also read-only verifies recorded storage IAM; an "
            "unverified/present account makes cleanup partial (exit 2)."
        ),
    ),
    project: str = typer.Option(
        "",
        "--project",
        help=(
            "Scope per-alias state and the --full read-only storage-IAM check to "
            "this NPA project alias."
        ),
    ),
    skip_jobs: bool = typer.Option(
        False,
        "--skip-jobs",
        help="Do not query the SkyPilot managed-job queue.",
    ),
    attest_no_active_jobs: bool = typer.Option(
        False,
        "--attest-no-active-jobs",
        help=(
            "With --skip-jobs, explicitly attest no active jobs after exact project "
            "terminal/no-submission evidence is verified."
        ),
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution.",
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit a machine-readable final cleanup result."
    ),
    list_receipts: bool = typer.Option(
        False,
        "--list-receipts",
        help="List retained non-secret teardown audit receipts and exit.",
    ),
    prune_receipts: bool = typer.Option(
        False,
        "--prune-receipts",
        help=(
            "Explicitly prune only terminal teardown receipts older than "
            "--receipt-retention-days; requires --yes."
        ),
    ),
    receipt_retention_days: int = typer.Option(
        90,
        "--receipt-retention-days",
        min=0,
        help="Minimum age for explicit terminal-receipt pruning (default: 90 days).",
    ),
) -> None:
    """Report (or with --yes remove) local NPA/SkyPilot residue left after teardown.

    Cloud resources (agent VM, cluster, bucket, IAM) are removed only by the
    commands in the printed runbook. Existing `--yes` keeps credentials/config;
    `--full` removes known local credentials/state and performs a read-only
    storage-IAM verification. Neither scope deletes cloud resources. Full cleanup
    exits 2 when IAM is present/unverified or provider verification fails.
    """
    import json
    from npa.cli.cluster.terraform_runtime import (
        collect_terraform_residue,
        remove_terraform_residue,
    )
    from npa.clients.config import clear_skypilot_bin
    from npa.teardown_receipts import (
        latest_phase_states,
        list_teardown_receipts,
        prune_teardown_receipts,
        record_teardown_event,
    )

    receipt_alias = project
    receipt_project_id = ""
    if project:
        environment = None
        try:
            from npa.clients.config import resolve_environment

            environment = resolve_environment(project)
            receipt_project_id = (
                str(environment.project_id or "") if environment is not None else ""
            )
        except Exception:  # noqa: BLE001 - exact immutable ID is valid for receipt audit
            environment = None
        if not receipt_project_id and project.startswith("project-"):
            # After the alias is deliberately forgotten, the immutable project
            # ID remains a valid receipt scope. This is read-only recovery
            # evidence and must never recreate configuration for the project.
            receipt_alias = ""
            receipt_project_id = project

    if list_receipts or prune_receipts:
        if list_receipts and prune_receipts:
            raise typer.BadParameter(
                "Use either --list-receipts or --prune-receipts, not both."
            )
        payload: dict[str, Any]
        if prune_receipts:
            if not yes:
                raise typer.BadParameter("--prune-receipts requires --yes")
            removed, retained = prune_teardown_receipts(
                older_than_days=receipt_retention_days
            )
            payload = {
                "result": "receipts_pruned",
                "removed": [str(path) for path in removed],
                "retained": retained,
                "retention_days": receipt_retention_days,
            }
        else:
            receipts = list_teardown_receipts(
                project_alias=receipt_alias,
                project_id=receipt_project_id,
                legacy="exclude" if project else "include",
            )
            payload = {
                "result": "receipts_listed",
                "retention": (
                    "retained indefinitely until explicit `npa cleanup "
                    "--prune-receipts --receipt-retention-days <days> --yes`"
                ),
                "receipts": receipts,
            }
        if output_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(str(payload["result"]).replace("_", " ").capitalize() + ":")
            if list_receipts:
                receipts = payload["receipts"]
                if not receipts:
                    typer.echo("  none")
                for receipt in receipts:
                    typer.echo(
                        f"  {receipt.get('receipt_id')}: "
                        f"{len(receipt.get('events') or [])} event(s), "
                        f"updated {receipt.get('updated_at')}"
                    )
                typer.echo(f"Retention: {payload['retention']}")
            else:
                for path in payload["removed"]:
                    typer.echo(f"  removed {path}")
                for note in payload["retained"]:
                    typer.echo(f"  retained {note}")
        return

    def emit(message: str = "", *, err: bool = False) -> None:
        if not output_json:
            typer.echo(message, err=err)

    removed_total = 0
    shared_runtime_preserved = False
    project_credential_residue_items: list[dict[str, str]] = []

    def emit_json(
        result: str, local_state: str, *, cleanup_failed: bool = False
    ) -> None:
        receipt_phases = latest_phase_states(
            project_alias=receipt_alias,
            project_id=receipt_project_id,
            strict=yes,
        )
        phases = cleanup_phase_model(
            jobs=job_ids,
            jobs_note=job_note,
            iam_state=iam_status,
            iam_ownership_state=iam_ownership_state,
            local_state=local_state,
            receipt_phases=receipt_phases,
        )
        from npa.teardown_receipts import TERMINAL_STATES

        operational_residue = bool(
            local_state not in {"fully_clean", "fully_cleaned", "preserved_shared_sky"}
            or project_credential_residue_items
            or agent_operational_state_present
        )
        if receipt_project_id:
            cloud_phase_names = {
                name for group in _CLOUD_CLEANUP_PHASE_GROUPS for name in group
            }
            unresolved_receipts = bool(
                any(
                    _phase_group_events(receipt_phases, group)
                    and not _newest_phase_group_is_terminal(receipt_phases, group)
                    for group in _CLOUD_CLEANUP_PHASE_GROUPS
                )
                or any(
                    name not in cloud_phase_names
                    and str(event.get("terminal_state") or "")
                    not in TERMINAL_STATES
                    for name, event in receipt_phases.items()
                )
            )
        else:
            unresolved_receipts = any(
                str(event.get("terminal_state") or "") not in TERMINAL_STATES
                for event in receipt_phases.values()
            )
        verification_unresolved = bool(
            iam_partial
            or job_note
            or (job_queue_state == "SKIPPED_BY_OPERATOR" and not attestation_safe)
            or cleanup_failed
            or unresolved_receipts
            or (
                bool(receipt_project_id)
                and not _cloud_cleanup_receipts_are_terminal(receipt_phases)
            )
        )
        retained_receipts = len(
            list_teardown_receipts(
                project_alias=receipt_alias,
                project_id=receipt_project_id,
                legacy="exclude" if project else "include",
            )
        )
        typer.echo(
            json.dumps(
                {
                    "result": result,
                    "local_state": "fully_cleaned"
                    if not operational_residue
                    else local_state,
                    "operational_residue_present": operational_residue,
                    "residue_present": operational_residue,
                    "verification_unresolved": verification_unresolved,
                    "managed_job_queue_state": job_queue_state,
                    "managed_job_queue_detail": job_note,
                    "nonterminal_job_ids": job_ids,
                    "iam_state": iam_status,
                    "iam_verification_required": iam_partial,
                    "project_retained": iam_partial,
                    "cleanup_failed": cleanup_failed,
                    "removed_bytes": removed_total,
                    "iam_detail": iam_message,
                    "phases": [item.to_dict() for item in phases],
                    "retained_audit_receipts": retained_receipts,
                    "audit_receipts_retained": bool(retained_receipts),
                    "audit_receipts_are_operational_residue": False,
                    "preserved_shared_runtime": shared_runtime_preserved,
                    "retained_local_residue": project_credential_residue_items,
                    "agent_operational_state_present": agent_operational_state_present,
                    "agent_operational_state_detail": agent_operational_state_detail,
                },
                indent=2,
                sort_keys=True,
            )
        )

    npa_dir = _npa_state_dir()
    residue = _collect_residue(include_sky=include_sky)
    terraform_residue = collect_terraform_residue()
    empty_dirs = _empty_alias_dirs(npa_dir, project)
    credential_labels = _full_credential_labels() if full else []
    full_empty_state = _full_empty_state(npa_dir) if full else []
    iam_message = _iam_note()
    iam_partial = False
    iam_status = "not_checked"
    iam_ownership_state = (
        "owned" if "recorded creating storage principal" in iam_message else "unknown"
    )
    if full:
        (
            iam_message,
            iam_partial,
            iam_status,
            iam_ownership_state,
        ) = _storage_iam_full_check(project, prune_verified_absence=False)

    agent_operational_state_present = False
    agent_operational_state_detail = "not checked"
    unscoped_scope_verified_absent = False

    if full and receipt_project_id:
        from npa.clients.project_credential_store import project_credential_residue

        project_credential_residue_items = project_credential_residue(
            receipt_project_id
        )
        if receipt_alias:
            from npa.clients.config import resolve_environment

            environment = resolve_environment(receipt_alias)
            if (
                environment is not None
                and str(environment.project_id or "") == receipt_project_id
            ):
                project_credential_residue_items.append(
                    {
                        "path": f"config.projects.{receipt_alias}",
                        "class": "project_alias_or_default",
                    }
                )
        try:
            (
                agent_operational_state_present,
                agent_operational_state_detail,
            ) = _agent_operational_state_present(receipt_alias, receipt_project_id)
        except (OSError, RuntimeError, ValueError) as exc:
            agent_operational_state_present = True
            agent_operational_state_detail = f"inventory unresolved: {exc}"
    elif full:
        try:
            (
                agent_operational_state_present,
                agent_operational_state_detail,
            ) = _unscoped_project_state_present()
            unscoped_scope_verified_absent = not agent_operational_state_present
        except (OSError, RuntimeError, ValueError) as exc:
            agent_operational_state_present = True
            agent_operational_state_detail = f"unscoped inventory unresolved: {exc}"

    prior_phases = latest_phase_states(
        project_alias=receipt_alias,
        project_id=receipt_project_id,
        strict=yes,
    )
    prior_workflow_terminal = _newest_phase_group_is_terminal(
        prior_phases, _CLOUD_CLEANUP_REQUIRED_GROUPS[0]
    )
    sky_operational_state_present = any(
        item.label in {"SkyPilot venv", "SkyPilot state (~/.sky)"} for item in residue
    )
    project_submission_audit = None
    if project:
        from npa.orchestration.npa_workflow.submission_state import (
            audit_project_submissions,
        )

        project_submission_audit = audit_project_submissions(project)
    project_audit_skip = bool(
        project_submission_audit
        and project_submission_audit.outcome == "not_submitted"
        and (not yes or not include_sky)
    )
    attestation_safe = False
    if attest_no_active_jobs:
        if not skip_jobs or not project or not receipt_project_id:
            raise typer.BadParameter(
                "--attest-no-active-jobs requires --skip-jobs and an exact configured "
                "or durably receipted project."
            )
        if not prior_workflow_terminal:
            raise typer.BadParameter(
                "No terminal/not-submitted project workflow receipt supports this attestation."
            )
        from npa.controller_ownership import controller_owner

        owner = controller_owner()
        if owner is not None and owner.project_id != receipt_project_id:
            raise typer.BadParameter(
                "The shared SkyPilot controller is owned by a different immutable project."
            )
        attestation_safe = True
    if skip_jobs:
        job_ids: list[str] = []
        job_note = ""
        job_queue_state = "SKIPPED_BY_OPERATOR"
    elif project_audit_skip:
        job_ids = []
        job_note = ""
        job_queue_state = "PROJECT_NOT_SUBMITTED"
    else:
        job_audit = _nonterminal_jobs(sky_bin)
        # Keep compatibility with extensions/tests that wrapped the historical
        # two-field helper while production now reports the richer state.
        if len(job_audit) == 3:
            job_ids, job_note, job_queue_state = job_audit
        else:  # pragma: no cover - compatibility shim for external wrappers
            job_ids, job_note = job_audit
            job_queue_state = (
                "unreadable_or_unverified"
                if job_note
                else "verified_active_jobs"
                if job_ids
                else "verified_empty"
            )
    if (
        not job_ids
        and job_note
        and prior_workflow_terminal
        and not sky_operational_state_present
    ):
        job_note = ""
        if not output_json:
            typer.echo(
                "Managed jobs: using retained terminal/absent audit evidence because "
                "SkyPilot operational state was already removed."
            )
    try:
        environment = None
        if project:
            from npa.clients.config import resolve_environment

            environment = resolve_environment(project)
        record_teardown_event(
            phase="workflow_audit",
            resource="all-managed-jobs",
            terminal_state=(
                "not_submitted"
                if job_queue_state == "PROJECT_NOT_SUBMITTED"
                else "verified_absent"
                if not job_ids and not job_note and not skip_jobs
                else "operator_attested"
                if attestation_safe
                else "skipped_by_operator"
                if skip_jobs
                else "active"
                if job_ids
                else "verification_failed"
            ),
            project_alias=receipt_alias,
            project_id=receipt_project_id
            or str(getattr(environment, "project_id", "") or ""),
            precheck={"skip_requested": skip_jobs},
            action={"kind": "read_only_managed_job_audit"},
            verification={
                "queue_state": job_queue_state,
                "nonterminal_job_ids": job_ids,
                "detail": job_note,
                "operator_attestation": attestation_safe,
            },
            errors=[job_note] if job_note else [],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        job_note = (
            (job_note + "; ") if job_note else ""
        ) + f"durable managed-job audit receipt failed: {exc}"
    if not output_json:
        if skip_jobs:
            typer.echo(
                "Managed jobs: SKIPPED_BY_OPERATOR"
                + (
                    " (explicit terminal-evidence attestation accepted)"
                    if attestation_safe
                    else ""
                )
            )
        else:
            _report_managed_jobs(job_ids, job_note)

    terraform_sizes = {item.path: _dir_size(item.path) for item in terraform_residue}
    total = sum(item.size for item in residue) + sum(terraform_sizes.values())
    if (
        not residue
        and not terraform_residue
        and not empty_dirs
        and not credential_labels
        and not full_empty_state
        and not project_credential_residue_items
        and not agent_operational_state_present
    ):
        phase_states = latest_phase_states(
            project_alias=receipt_alias,
            project_id=receipt_project_id,
            strict=yes,
        )
        cloud_incomplete = bool(
            full
            and yes
            and receipt_project_id
            and not _cloud_cleanup_receipts_are_terminal(phase_states)
        )
        if output_json:
            emit_json(
                iam_status
                if iam_partial
                else "partial_cloud_cleanup"
                if cloud_incomplete
                else ("fully_cleaned" if yes else "fully_clean"),
                "partial_cloud_cleanup" if cloud_incomplete else "fully_clean",
                cleanup_failed=cloud_incomplete,
            )
        else:
            typer.echo("No local NPA/SkyPilot residue to clean up.")
            typer.echo(iam_message or _iam_note())
            _print_runbook(
                cleanup_phase_model(
                    jobs=job_ids,
                    jobs_note=job_note,
                    iam_state=iam_status,
                    iam_ownership_state=iam_ownership_state,
                    local_state="fully_clean",
                    receipt_phases=latest_phase_states(
                        project_alias=receipt_alias,
                        project_id=receipt_project_id,
                        strict=yes,
                    ),
                )
            )
            receipts = list_teardown_receipts(
                project_alias=receipt_alias,
                project_id=receipt_project_id,
                legacy="exclude" if project else "include",
            )
            if receipts:
                typer.echo(
                    f"Retained audit receipts: {len(receipts)} non-secret file(s); "
                    "these are audit evidence, not operational residue."
                )
        if iam_partial:
            emit(
                "Full cleanup is partial because storage IAM was not verified absent/deleted.",
                err=True,
            )
            raise typer.Exit(code=2)
        if cloud_incomplete:
            emit(
                "Full cleanup is partial because complete phase-specific cloud "
                "absence evidence is missing or unresolved.",
                err=True,
            )
            raise typer.Exit(code=1)
        return

    scope = (
        "full local scope; known shared credentials are included"
        if full
        else "secret-free; your tokens/config are untouched"
    )
    emit(f"Local residue after teardown ({scope}):")
    for residue_item in residue:
        emit(
            f"  {residue_item.label:<26} {_human(residue_item.size):>8}  "
            f"{residue_item.path}"
        )
    for tf_item in terraform_residue:
        suffix = (
            f" ({tf_item.reason}; will not remove)" if not tf_item.removable else ""
        )
        emit(
            f"  {tf_item.label:<26} {_human(terraform_sizes[tf_item.path]):>8}  "
            f"{tf_item.path}{suffix}"
        )
    for empty in empty_dirs:
        emit(f"  {'empty state dir':<26} {'-':>8}  {empty}")
    for label in credential_labels:
        emit(f"  {label:<26} {'saved':>8}")
    for path in full_empty_state:
        emit(f"  {'empty local state':<26} {'-':>8}  {path}")
    if agent_operational_state_present:
        emit(
            f"  {'agent operational state':<26} {'saved':>8}  "
            f"{agent_operational_state_detail}"
        )
    if residue or terraform_residue:
        emit(f"  {'total':<26} {_human(total):>8}")

    if not yes:
        emit("")
        rerun = "--full --yes" if full else "--yes"
        emit(
            f"Re-run with {rerun} to remove owned residue. Machine-shared ~/.sky "
            "is always preserved."
        )
        emit(iam_message or _iam_note())
        if output_json:
            emit_json(
                iam_status if iam_partial else "cleanup_planned", "residue_present"
            )
        else:
            _print_runbook(
                cleanup_phase_model(
                    jobs=job_ids,
                    jobs_note=job_note,
                    iam_state=iam_status,
                    iam_ownership_state=iam_ownership_state,
                    local_state="residue_present",
                    receipt_phases=latest_phase_states(
                        project_alias=receipt_alias,
                        project_id=receipt_project_id,
                        strict=yes,
                    ),
                )
            )
        if iam_partial:
            raise typer.Exit(code=2)
        return

    removed_bin = False
    cleanup_failed = False
    recovery_snapshots: tuple[_PrivateYamlSnapshot, ...] = ()
    recovery_state_mutated = False
    sky_audit_safe = (
        not skip_jobs and not job_ids and not job_note
    ) or attestation_safe
    sky_preserved_by_skip = False
    try:
        record_teardown_event(
            phase="local_cleanup",
            resource="npa-local-state",
            terminal_state="in_progress",
            project_alias=receipt_alias,
            project_id=receipt_project_id,
            precheck={"managed_jobs_verified_terminal_or_absent": sky_audit_safe},
            action={
                "kind": "local_cleanup",
                "full": full,
                "include_sky": include_sky,
            },
            verification={"local_state_removal": "pending"},
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cleanup_failed = True
        emit(
            "Preserved local state because the durable cleanup transaction "
            f"could not be started: {exc}",
            err=True,
        )
        if output_json:
            emit_json("partial_cleanup", "residue_present", cleanup_failed=True)
        raise typer.Exit(code=1) from exc
    if full:
        try:
            recovery_snapshots = _snapshot_cleanup_recovery_documents()
        except (OSError, RuntimeError, ValueError) as exc:
            cleanup_failed = True
            emit(
                "Preserved local state because recovery credentials/configuration "
                f"could not be snapshotted before retirement: {exc}",
                err=True,
            )
            if output_json:
                emit_json("partial_cleanup", "residue_present", cleanup_failed=True)
            raise typer.Exit(code=1) from exc
    for residue_item in residue:
        if project and residue_item.label in {
            "SkyPilot venv",
            "Terraform provider cache",
        }:
            shared_runtime_preserved = True
            emit(
                f"Preserved shared {residue_item.label} at {residue_item.path}: "
                "project-scoped cleanup does not own global runtime/cache state."
            )
            continue
        if residue_item.label == "SkyPilot state (~/.sky)":
            shared_runtime_preserved = True
            sky_preserved_by_skip = True
            emit(
                f"Preserved shared {residue_item.label} at {residue_item.path}: "
                f"{_shared_sky_preservation_reason(receipt_project_id)}."
            )
            continue
        if (
            residue_item.label in {"SkyPilot venv", "SkyPilot state (~/.sky)"}
            and not sky_audit_safe
        ):
            sky_preserved_by_skip = sky_preserved_by_skip or skip_jobs
            if not skip_jobs:
                cleanup_failed = True
            emit(
                f"Preserved {residue_item.label} at {residue_item.path}: managed jobs are active or "
                "their terminal state could not be durably verified before local "
                "SkyPilot state removal.",
                err=True,
            )
            if skip_jobs and not attest_no_active_jobs and project:
                emit(
                    "Follow-up after exact terminal evidence: "
                    f"npa cleanup --project {project} --include-sky --skip-jobs "
                    "--attest-no-active-jobs --yes",
                    err=True,
                )
            continue
        try:
            problem = _remove_exact_residue(residue_item)
            if problem:
                raise OSError(problem)
        except OSError as exc:
            cleanup_failed = True
            emit(
                f"Warning: could not remove {residue_item.label} at {residue_item.path}: {exc}",
                err=True,
            )
            continue
        removed_total += residue_item.size
        emit(f"Removed {residue_item.label}: {residue_item.path}")
        if residue_item.label == "SkyPilot venv":
            removed_bin = clear_skypilot_bin()
    for tf_item in terraform_residue:
        problem = remove_terraform_residue(tf_item)
        if not problem and os.path.lexists(tf_item.path):
            problem = "exact Terraform residue path still exists after removal"
        if problem:
            cleanup_failed = True
            emit(
                f"Warning: could not remove {tf_item.label} at {tf_item.path}: {problem}",
                err=True,
            )
        else:
            removed_total += terraform_sizes[tf_item.path]
            emit(f"Removed {tf_item.label}: {tf_item.path}")
    if removed_bin:
        emit("Cleared skypilot.sky_bin from ~/.npa/config.yaml.")
    for empty in empty_dirs:
        try:
            empty.rmdir()
            emit(f"Removed empty state dir: {empty}")
        except OSError:
            pass
    # Drop the now-empty agents/ and workbenches/ base dirs too in the narrow
    # scope. Full cleanup handles these plus clusters/ and ~/.npa below.
    for base_name in () if full else ("agents", "workbenches"):
        base = npa_dir / base_name
        try:
            if base.is_dir() and not any(base.iterdir()):
                base.rmdir()
        except OSError:
            pass
    cleared_credentials: list[str] = []
    pruned_state: list[tuple[str, Path]] = []
    remaining_terraform = collect_terraform_residue()
    if remaining_terraform:
        cleanup_failed = True
        emit(
            "Warning: Terraform residue remains after cleanup: "
            + ", ".join(str(item.path) for item in remaining_terraform),
            err=True,
        )
    if full:
        agent_retirement_safe = True
        agent_retirement_detail = "no exact-project credential retirement requested"
        if receipt_project_id and agent_operational_state_present:
            agent_retirement_safe, agent_retirement_detail = (
                _agent_lifecycle_allows_project_retirement(
                    receipt_alias, receipt_project_id, retire=False
                )
            )
        credentials_retirement_safe = bool(
            (
                unscoped_scope_verified_absent
                if not receipt_project_id
                else agent_retirement_safe
            )
            and not remaining_terraform
        )
        phase_states = latest_phase_states(
            project_alias=receipt_alias,
            project_id=receipt_project_id,
            strict=True,
        )
        cloud_absent = bool(
            unscoped_scope_verified_absent
            if not receipt_project_id
            else (
                _cloud_cleanup_receipts_are_terminal(phase_states)
                and agent_retirement_safe
            )
        )
        if receipt_project_id and not cloud_absent:
            cleanup_failed = True
            emit(
                "Warning: full cleanup remains partial because complete "
                "phase-specific cloud absence evidence is missing or unresolved.",
                err=True,
            )
        if (
            cloud_absent
            and credentials_retirement_safe
            and not cleanup_failed
            and not iam_partial
        ):
            (
                iam_message,
                iam_partial,
                iam_status,
                iam_ownership_state,
            ) = _storage_iam_full_check(project, prune_verified_absence=True)
            if iam_partial:
                cleanup_failed = True
                emit(
                    "Warning: final storage IAM evidence retirement did not converge.",
                    err=True,
                )
        convergence_safe = bool(
            credentials_retirement_safe
            and cloud_absent
            and not iam_partial
            and not cleanup_failed
        )
        if (
            convergence_safe
            and receipt_project_id
            and agent_operational_state_present
        ):
            recovery_state_mutated = True
            agent_retirement_safe, agent_retirement_detail = (
                _agent_lifecycle_allows_project_retirement(
                    receipt_alias, receipt_project_id, retire=True
                )
            )
            if agent_retirement_safe:
                try:
                    (
                        agent_operational_state_present,
                        agent_operational_state_detail,
                    ) = _agent_operational_state_present(
                        receipt_alias, receipt_project_id
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    agent_retirement_safe = False
                    agent_operational_state_present = True
                    agent_retirement_detail = (
                        f"final agent local-state inventory failed: {exc}"
                    )
            if not agent_retirement_safe or agent_operational_state_present:
                cleanup_failed = True
        convergence_safe = bool(
            convergence_safe
            and agent_retirement_safe
            and not agent_operational_state_present
            and not cleanup_failed
        )
        if project_credential_residue_items and convergence_safe:
            from npa.clients.config import forget_project, resolve_environment
            from npa.clients.project_credential_store import (
                forget_project_credentials,
                project_credential_residue,
            )

            try:
                recovery_state_mutated = True
                removed_credentials = forget_project_credentials(receipt_project_id)
                environment = (
                    resolve_environment(receipt_alias) if receipt_alias else None
                )
                removed_alias = False
                if (
                    environment is not None
                    and str(environment.project_id or "") == receipt_project_id
                ):
                    removed_alias = forget_project(receipt_alias)
            except (OSError, RuntimeError, ValueError) as exc:
                removed_credentials = False
                removed_alias = False
                cleanup_failed = True
                emit(
                    "Warning: exact-project credential/config retirement failed: "
                    f"{exc}",
                    err=True,
                )
            if removed_credentials or removed_alias:
                emit(
                    "Removed exact-project operational credentials/configuration after "
                    "complete cloud absence proof."
                )
            project_credential_residue_items = project_credential_residue(
                receipt_project_id
            )
            remaining_environment = (
                resolve_environment(receipt_alias) if receipt_alias else None
            )
            if (
                remaining_environment is not None
                and str(remaining_environment.project_id or "") == receipt_project_id
            ):
                project_credential_residue_items.append(
                    {
                        "path": f"config.projects.{receipt_alias}",
                        "class": "project_alias_or_default",
                    }
                )
        if agent_operational_state_present or not agent_retirement_safe:
            cleanup_failed = True
            emit(
                "Warning: exact-project credentials and alias were preserved because "
                f"agent lifecycle verification is unresolved: {agent_retirement_detail}.",
                err=True,
            )
        if project_credential_residue_items:
            cleanup_failed = True
            emit(
                "Warning: exact-project operational credential residue remains at "
                + ", ".join(item["path"] for item in project_credential_residue_items),
                err=True,
            )
        # Reinventory after all dependent local/cloud retirement. A concurrent
        # Terraform generation or failed deletion keeps every credential intact.
        remaining_terraform = collect_terraform_residue()
        if remaining_terraform:
            cleanup_failed = True
            emit(
                "Warning: final Terraform inventory found residue: "
                + ", ".join(str(item.path) for item in remaining_terraform),
                err=True,
            )
        final_credentials_safe = bool(
            credentials_retirement_safe
            and cloud_absent
            and not iam_partial
            and not cleanup_failed
            and not project_credential_residue_items
            and not remaining_terraform
        )
        if final_credentials_safe:
            recovery_state_mutated = True
            cleared_credentials = _clear_full_credentials()
            if cleared_credentials:
                emit(
                    "Removed locally stored "
                    + ", ".join(cleared_credentials)
                    + "."
                )
            if credential_labels and set(cleared_credentials) != set(
                credential_labels
            ):
                cleanup_failed = True
                missing_credentials = sorted(
                    set(credential_labels) - set(cleared_credentials)
                )
                emit(
                    "Warning: requested shared credential group(s) could not be removed: "
                    + ", ".join(missing_credentials)
                    + ". Unrelated credential data was preserved.",
                    err=True,
                )
        elif credential_labels:
            cleanup_failed = True
            emit(
                "Preserved shared credentials because local/cloud convergence is "
                "unresolved.",
                err=True,
            )
        if final_credentials_safe:
            recovery_state_mutated = True
        pruned_state = _prune_full_empty_state(npa_dir) if final_credentials_safe else []
        for label, path in pruned_state:
            if label == "empty NPA home":
                emit(f"Removed empty NPA home: {path}")
            else:
                emit(f"Removed {label}: {path}")
    if full and cleanup_failed and recovery_state_mutated:
        (
            restore_failures,
            project_credential_residue_items,
            agent_operational_state_present,
            agent_operational_state_detail,
        ) = _restore_cleanup_recovery_documents(
            recovery_snapshots,
            project_alias=receipt_alias,
            project_id=receipt_project_id,
        )
        if restore_failures:
            emit(
                "Warning: recovery credential/config restoration was incomplete: "
                + ", ".join(restore_failures),
                err=True,
            )
        else:
            emit(
                "Restored recovery credentials/configuration because local "
                "retirement did not converge."
            )
    # Terminal state is derived from the final inventory, never from mutation
    # attempts that preceded it.
    remaining_terraform = collect_terraform_residue()
    if remaining_terraform:
        cleanup_failed = True
    local_terminal = not cleanup_failed and not iam_partial and not remaining_terraform
    try:
        record_teardown_event(
            phase="local_cleanup",
            resource="npa-local-state",
            terminal_state="completed" if local_terminal else "partial",
            project_alias=receipt_alias,
            project_id=receipt_project_id,
            precheck={"managed_jobs_verified_terminal_or_absent": sky_audit_safe},
            action={
                "kind": "local_cleanup",
                "full": full,
                "include_sky": include_sky,
            },
            verification={
                "remaining_terraform_count": len(remaining_terraform),
                "sky_state_preserved": bool(
                    sky_preserved_by_skip or not sky_audit_safe
                ),
                "shared_runtime_preserved": shared_runtime_preserved,
                "managed_job_verification": job_queue_state,
            },
            errors=(["local cleanup incomplete"] if cleanup_failed else []),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cleanup_failed = True
        emit(f"Warning: local cleanup receipt could not be written: {exc}", err=True)
        if full and recovery_state_mutated:
            (
                restore_failures,
                project_credential_residue_items,
                agent_operational_state_present,
                agent_operational_state_detail,
            ) = _restore_cleanup_recovery_documents(
                recovery_snapshots,
                project_alias=receipt_alias,
                project_id=receipt_project_id,
            )
            if restore_failures:
                emit(
                    "Warning: recovery credential/config restoration was incomplete: "
                    + ", ".join(restore_failures),
                    err=True,
                )
            else:
                emit(
                    "Restored recovery credentials/configuration because the "
                    "terminal cleanup receipt was not durable."
                )
    emit("")
    if full and not cleanup_failed and not iam_partial:
        emit(
            f"Freed {_human(removed_total)} of local caches. Known shared credentials and "
            "empty NPA-owned state were removed; non-empty/unrelated data was kept."
        )
    elif full:
        emit(
            f"Freed {_human(removed_total)} of local caches, but full local cleanup was incomplete. "
            "Non-empty/unrelated data was kept; fix the warning above and retry."
        )
    else:
        emit(
            f"Freed {_human(removed_total)} of local caches. Tokens and project config were kept."
        )
    emit(iam_message or _iam_note())
    if output_json:
        result = (
            iam_status
            if iam_partial
            else "partial_local_cleanup"
            if cleanup_failed
            else "completed_with_preserved_sky"
            if sky_preserved_by_skip
            else "fully_cleaned"
        )
        emit_json(
            result,
            "partial_local_cleanup"
            if cleanup_failed
            else "preserved_shared_sky"
            if sky_preserved_by_skip
            else "fully_cleaned",
            cleanup_failed=cleanup_failed,
        )
    else:
        _print_runbook(
            cleanup_phase_model(
                jobs=job_ids,
                jobs_note=job_note,
                iam_state=iam_status,
                iam_ownership_state=iam_ownership_state,
                local_state=(
                    "partial_local_cleanup" if cleanup_failed else "fully_cleaned"
                ),
                receipt_phases=latest_phase_states(
                    project_alias=receipt_alias,
                    project_id=receipt_project_id,
                    strict=yes,
                ),
            )
        )
        receipts = list_teardown_receipts(
            project_alias=receipt_alias,
            project_id=receipt_project_id,
            legacy="exclude" if project else "include",
        )
        if receipts:
            typer.echo(
                f"Retained audit receipts: {len(receipts)} non-secret file(s); "
                "these are audit evidence, not operational residue."
            )
    if iam_partial:
        emit(
            "Full cleanup is partial because storage IAM was not verified absent/deleted.",
            err=True,
        )
        raise typer.Exit(code=2)
    if cleanup_failed:
        raise typer.Exit(code=1)
