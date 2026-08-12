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

    try:
        data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(data, dict):
        return []
    raw_tokens = data.get("tokens")
    tokens = raw_tokens if isinstance(raw_tokens, dict) else {}
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
    from npa.clients.config import ConfigError, storage_iam_residues

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
    for alias in aliases:
        try:
            if alias.startswith("project-"):
                context = _resolve_storage_iam_context(project_id=alias)
            else:
                context = _resolve_storage_iam_context(alias)
            observation = _observe_storage_iam(context)
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
        else "unknown"
        if "unknown" in ownership_states
        else "verified_terminal"
    )
    return "\n".join(messages), partial, status, ownership_state


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

        state = str((receipt_phases.get(phase) or {}).get("terminal_state") or "")
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
            project_alias=receipt_alias, project_id=receipt_project_id
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
            local_state
            not in {"fully_clean", "fully_cleaned", "preserved_shared_sky"}
            or project_credential_residue_items
        )
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
        ) = _storage_iam_full_check(project, prune_verified_absence=yes)

    if full and receipt_project_id:
        from npa.clients.project_credential_store import project_credential_residue

        project_credential_residue_items = project_credential_residue(
            receipt_project_id
        )
        if receipt_alias:
            project_credential_residue_items.append(
                {
                    "path": f"config.projects.{receipt_alias}",
                    "class": "project_alias_or_default",
                }
            )

    prior_phases = latest_phase_states(
        project_alias=receipt_alias, project_id=receipt_project_id
    )
    prior_workflow = (
        prior_phases.get("workflow_audit")
        or prior_phases.get("workflow")
        or prior_phases.get("project_destroy_workflows")
    )
    prior_workflow_state = str((prior_workflow or {}).get("terminal_state") or "")
    prior_workflow_terminal = prior_workflow_state in {
        "already_absent",
        "cancelled",
        "completed",
        "deleted",
        "terminal",
        "verified_absent",
        "verified_deleted",
    }
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
            project_alias=project,
            project_id=str(getattr(environment, "project_id", "") or ""),
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
    ):
        if output_json:
            emit_json(
                iam_status
                if iam_partial
                else ("fully_cleaned" if yes else "fully_clean"),
                "fully_clean",
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
                        project_alias=receipt_alias, project_id=receipt_project_id
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
                        project_alias=receipt_alias, project_id=receipt_project_id
                    ),
                )
            )
        if iam_partial:
            raise typer.Exit(code=2)
        return

    removed_bin = False
    cleanup_failed = False
    sky_audit_safe = (
        not skip_jobs and not job_ids and not job_note
    ) or attestation_safe
    sky_preserved_by_skip = False
    try:
        record_teardown_event(
            phase="local_cleanup",
            resource="npa-local-state",
            terminal_state="in_progress",
            project_alias=project,
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
    for residue_item in residue:
        if (
            project
            and residue_item.label in {"SkyPilot venv", "Terraform provider cache"}
        ):
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
    if full:
        cleared_credentials = _clear_full_credentials()
        if cleared_credentials:
            emit("Removed locally stored " + ", ".join(cleared_credentials) + ".")
        if credential_labels and set(cleared_credentials) != set(credential_labels):
            cleanup_failed = True
            missing_credentials = sorted(
                set(credential_labels) - set(cleared_credentials)
            )
            emit(
                "Warning: requested shared credential group(s) could not be removed: "
                + ", ".join(missing_credentials)
                + ". Any groups reported removed above remain removed; unrelated "
                "credential data was preserved.",
                err=True,
            )
        cloud_terminal_states = {
            "already_absent",
            "cancelled",
            "completed",
            "deleted",
            "not_submitted",
            "terminal",
            "verified_absent",
            "verified_deleted",
        }
        cloud_phase_names = (
            "project_destroy_workflows",
            "project_destroy_agents",
            "project_destroy_controller",
            "project_destroy_clusters",
            "project_destroy_bucket",
            "project_destroy_storage_iam",
        )
        phase_states = latest_phase_states(
            project_alias=receipt_alias, project_id=receipt_project_id
        )
        cloud_absent = bool(
            receipt_project_id
            and all(
                str((phase_states.get(name) or {}).get("terminal_state") or "")
                in cloud_terminal_states
                for name in cloud_phase_names
            )
        )
        if project_credential_residue_items and not iam_partial and cloud_absent:
            from npa.clients.project_credential_store import (
                forget_project_credentials,
                project_credential_residue,
            )

            if forget_project_credentials(receipt_project_id):
                project_credential_residue_items = project_credential_residue(
                    receipt_project_id
                )
                emit(
                    "Removed exact-project operational credentials after complete cloud absence proof."
                )
        if project_credential_residue_items:
            cleanup_failed = True
            emit(
                "Warning: exact-project operational credential residue remains at "
                + ", ".join(item["path"] for item in project_credential_residue_items),
                err=True,
            )
        pruned_state = _prune_full_empty_state(npa_dir)
        for label, path in pruned_state:
            if label == "empty NPA home":
                emit(f"Removed empty NPA home: {path}")
            else:
                emit(f"Removed {label}: {path}")
    local_terminal = not cleanup_failed and not iam_partial
    try:
        record_teardown_event(
            phase="local_cleanup",
            resource="npa-local-state",
            terminal_state="completed" if local_terminal else "partial",
            project_alias=project,
            precheck={"managed_jobs_verified_terminal_or_absent": sky_audit_safe},
            action={
                "kind": "local_cleanup",
                "full": full,
                "include_sky": include_sky,
            },
            verification={
                "remaining_terraform_count": len(collect_terraform_residue()),
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
    remaining_terraform = collect_terraform_residue()
    if remaining_terraform:
        cleanup_failed = True
        emit(
            "Warning: Terraform residue remains after cleanup: "
            + ", ".join(str(item.path) for item in remaining_terraform),
            err=True,
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
                    project_alias=receipt_alias, project_id=receipt_project_id
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
