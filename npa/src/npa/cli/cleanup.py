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
from pathlib import Path

import typer


@dataclass
class _Residue:
    label: str
    path: Path
    size: int


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
    if data.get("HF_TOKEN") or tokens.get("HF_TOKEN") or _section_has_any(
        data, "huggingface"
    ):
        labels.append("Hugging Face token")
    if (
        data.get("NEBIUS_TOKEN_FACTORY_KEY")
        or tokens.get("NEBIUS_TOKEN_FACTORY_KEY")
        or _section_has_any(data, "token_factory")
    ):
        labels.append("Token Factory key")
    if any(data.get(key) or tokens.get(key) for key in ("NGC_API_KEY", "NGC_ORG", "NGC_TEAM")):
        labels.append("NGC credentials")
    elif _section_has_any(data, "ngc"):
        labels.append("NGC credentials")
    return labels


def _section_has_any(data: dict, section_name: str) -> bool:
    section = data.get(section_name)
    if not isinstance(section, dict):
        return False
    return any(section.get(key) not in (None, "") for key in _SERVICE_CREDENTIAL_FIELDS[section_name])


def _clear_full_credentials() -> list[str]:
    """Remove only known shared-service credentials, preserving other data."""

    import yaml

    from npa.clients.credentials import CREDENTIALS_PATH

    labels = _full_credential_labels()
    if not labels:
        return []
    try:
        data = yaml.safe_load(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(data, dict):
        return []

    for key in _FULL_TOKEN_KEYS:
        data.pop(key, None)
    tokens = data.get("tokens")
    if isinstance(tokens, dict):
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
        for key in fields:
            section.pop(key, None)
        if section:
            data[section_name] = section
        else:
            data.pop(section_name, None)

    if not data:
        try:
            CREDENTIALS_PATH.unlink()
        except OSError:
            return []
        return labels
    try:
        CREDENTIALS_PATH.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        CREDENTIALS_PATH.chmod(0o600)
    except OSError:
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


def _full_empty_state(npa_dir: Path) -> list[Path]:
    """Return empty, known NPA-owned state that full cleanup can prune."""

    from npa.clients.config import CONFIG_PATH
    from npa.clients.credentials import CREDENTIALS_PATH

    found: list[Path] = []
    for path in (CONFIG_PATH, CREDENTIALS_PATH):
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

    from npa.clients.config import CONFIG_PATH
    from npa.clients.credentials import CREDENTIALS_PATH

    removed: list[tuple[str, Path]] = []
    for label, path in (("config file", CONFIG_PATH), ("credentials file", CREDENTIALS_PATH)):
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
    home = Path.home()
    npa_dir = home / ".npa"
    residue: list[_Residue] = []
    for label, path in (
        ("SkyPilot venv", npa_dir / "skypilot-venv"),
        ("Terraform provider cache", npa_dir / "terraform-plugin-cache"),
    ):
        if path.exists():
            residue.append(_Residue(label, path, _dir_size(path)))
    if include_sky:
        sky_home = home / ".sky"
        if sky_home.exists():
            residue.append(_Residue("SkyPilot state (~/.sky)", sky_home, _dir_size(sky_home)))
    return residue


# Teardown is an ordered sequence and nothing checks the order. The step most
# often missed is the first one: a managed job left non-terminal keeps the jobs
# controller alive and makes `sky down` refuse.
TEARDOWN_RUNBOOK = (
    "sky jobs cancel -a  (then wait for `sky jobs queue --all` to show them terminal; "
    "with no controller it errors 'No in-progress managed jobs' -- that is success)",
    "npa agent destroy --project <alias> --name <name> --yes",
    "npa cluster down --force",
    "npa storage bucket delete --project <alias> --yes --wait",
    "npa storage service-account delete --project <alias> --yes "
    "(only when configure recorded that NPA created it)",
    "npa configure --forget-project <alias>",
    # Cleanup checks managed jobs before removing the isolated SkyPilot venv;
    # running `skypilot uninstall` afterwards would only be a dead no-op.
    "npa cleanup --full --yes     (known shared tokens + caches + empty local state)",
)


def _nonterminal_jobs(sky_bin: str = "") -> tuple[list[str], str]:
    """Return managed jobs that are still non-terminal, and any lookup problem."""

    from npa.orchestration.skypilot._bin import SkyPilotNotInstalledError
    from npa.orchestration.skypilot.cleanup import _nonterminal_job_ids

    try:
        return (
            _nonterminal_job_ids(
                isolated_config_dir=None, config_path=None, sky_bin=sky_bin or None
            ),
            "",
        )
    except SkyPilotNotInstalledError:
        return [], "SkyPilot is not installed, so managed jobs were not checked"
    except (OSError, ValueError) as exc:
        return [], f"could not read the managed-job queue: {exc}"


def _report_managed_jobs(sky_bin: str) -> None:
    jobs, note = _nonterminal_jobs(sky_bin)
    if note:
        typer.echo(f"Managed jobs: {note}")
        return
    if not jobs:
        typer.echo("Managed jobs: none non-terminal.")
        return
    typer.echo(f"Managed jobs still non-terminal: {', '.join(jobs)}")
    typer.echo(
        "  These block `sky down` of the jobs controller. A job whose pod cannot "
        "start stays PENDING forever rather than failing, so check it before "
        "assuming it is still doing work."
    )


def _print_runbook() -> None:
    typer.echo("")
    typer.echo(
        "Full teardown order (printed only; cleanup never implies the preceding cloud steps):"
    )
    for index, step in enumerate(TEARDOWN_RUNBOOK, start=1):
        typer.echo(f"  {index}. {step}")


def _iam_note() -> str:
    """A hint about cloud IAM leftovers npa deliberately does not delete."""
    generic = (
        "Cloud IAM (not removed): pre-existing service accounts are left in place; "
        "remove one deliberately with `nebius iam service-account delete --id <id>`."
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
                    f"principal {owned_sa_id}. After deleting its bucket, remove it safely with "
                    "`npa storage service-account delete --project <alias> --yes`."
                )
            sa_id = str(nebius.get("service_account_id", "") or "").strip()
            if sa_id:
                return (
                    f"Cloud IAM (not removed): the storage principal {sa_id} and any "
                    "pre-existing service accounts remain — deleting a shared SA can "
                    "break other work. Remove one deliberately with "
                    f"`nebius iam service-account delete --id {sa_id}`."
                )
    except Exception:  # noqa: BLE001 - the note is best-effort
        return generic
    return generic


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
        help="Also remove SkyPilot's own ~/.sky state cache (safe once no clusters/jobs run).",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help=(
            "Broaden --yes to also remove locally saved HF, Token Factory, and NGC "
            "credentials, then prune empty config/known ~/.npa state. Non-empty and "
            "unrelated data is preserved."
        ),
    ),
    project: str = typer.Option(
        "", "--project", help="Scope the empty per-alias state-dir report to this alias."
    ),
    skip_jobs: bool = typer.Option(
        False,
        "--skip-jobs",
        help="Do not query the SkyPilot managed-job queue.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution.",
    ),
) -> None:
    """Report (or with --yes remove) local NPA/SkyPilot residue left after teardown.

    Local only. Cloud resources (agent VM, cluster, bucket, IAM) are removed by
    the commands in the printed runbook. Existing `--yes` keeps credentials and
    config; add `--full` to remove known shared-service credentials and empty
    NPA-owned local state. Neither scope deletes anything in the cloud.
    """
    import shutil

    from npa.clients.config import clear_skypilot_bin

    npa_dir = Path.home() / ".npa"
    residue = _collect_residue(include_sky=include_sky)
    empty_dirs = _empty_alias_dirs(npa_dir, project)
    credential_labels = _full_credential_labels() if full else []
    full_empty_state = _full_empty_state(npa_dir) if full else []

    if not skip_jobs:
        _report_managed_jobs(sky_bin)

    total = sum(item.size for item in residue)
    if not residue and not empty_dirs and not credential_labels and not full_empty_state:
        typer.echo("No local NPA/SkyPilot residue to clean up.")
        typer.echo(_iam_note())
        if not yes:
            _print_runbook()
        return

    scope = (
        "full local scope; known shared credentials are included"
        if full
        else "secret-free; your tokens/config are untouched"
    )
    typer.echo(f"Local residue after teardown ({scope}):")
    for item in residue:
        typer.echo(f"  {item.label:<26} {_human(item.size):>8}  {item.path}")
    for empty in empty_dirs:
        typer.echo(f"  {'empty state dir':<26} {'-':>8}  {empty}")
    for label in credential_labels:
        typer.echo(f"  {label:<26} {'saved':>8}")
    for path in full_empty_state:
        typer.echo(f"  {'empty local state':<26} {'-':>8}  {path}")
    if residue:
        typer.echo(f"  {'total':<26} {_human(total):>8}")

    if not yes:
        typer.echo("")
        rerun = "--full --yes" if full else "--yes"
        typer.echo(f"Re-run with {rerun} to remove them (or --keep-sky to leave ~/.sky).")
        typer.echo(_iam_note())
        _print_runbook()
        return

    removed_bin = False
    for item in residue:
        shutil.rmtree(item.path, ignore_errors=True)
        typer.echo(f"Removed {item.label}: {item.path}")
        if item.label == "SkyPilot venv":
            removed_bin = clear_skypilot_bin()
    if removed_bin:
        typer.echo("Cleared skypilot.sky_bin from ~/.npa/config.yaml.")
    for empty in empty_dirs:
        try:
            empty.rmdir()
            typer.echo(f"Removed empty state dir: {empty}")
        except OSError:
            pass
    # Drop the now-empty agents/ and workbenches/ base dirs too in the narrow
    # scope. Full cleanup handles these plus clusters/ and ~/.npa below.
    for base_name in (() if full else ("agents", "workbenches")):
        base = npa_dir / base_name
        try:
            if base.is_dir() and not any(base.iterdir()):
                base.rmdir()
        except OSError:
            pass
    cleanup_failed = False
    cleared_credentials: list[str] = []
    pruned_state: list[tuple[str, Path]] = []
    if full:
        cleared_credentials = _clear_full_credentials()
        if cleared_credentials:
            typer.echo(
                "Removed locally stored " + ", ".join(cleared_credentials) + "."
            )
        if credential_labels and set(cleared_credentials) != set(credential_labels):
            cleanup_failed = True
            typer.echo(
                "Warning: one or more requested shared credentials could not be removed; "
                "the credentials file was preserved for a safe retry.",
                err=True,
            )
        pruned_state = _prune_full_empty_state(npa_dir)
        for label, path in pruned_state:
            if label == "empty NPA home":
                typer.echo(f"Removed empty NPA home: {path}")
            else:
                typer.echo(f"Removed {label}: {path}")
    typer.echo("")
    if full and not cleanup_failed:
        typer.echo(
            f"Freed ~{_human(total)} of local caches. Known shared credentials and "
            "empty NPA-owned state were removed; non-empty/unrelated data was kept."
        )
    elif full:
        typer.echo(
            f"Freed ~{_human(total)} of local caches, but full local cleanup was incomplete. "
            "Non-empty/unrelated data was kept; fix the warning above and retry."
        )
    else:
        typer.echo(f"Freed ~{_human(total)} of local caches. Tokens and project config were kept.")
    typer.echo(_iam_note())
    if cleanup_failed:
        raise typer.Exit(code=1)
