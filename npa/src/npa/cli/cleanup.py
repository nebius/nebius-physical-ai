"""`npa cleanup` — report and remove local NPA/SkyPilot residue after teardown.

`agent destroy`, `cluster down`, `storage bucket delete` and
`configure --forget-project` clear the cloud resources and their secrets, but a
few large, secret-free local caches survive with no single command to see or
remove them: the isolated SkyPilot venv (~500 MB), the Terraform provider cache,
SkyPilot's own `~/.sky` state (~100 MB), and empty per-alias state directories.

`npa cleanup` lists them (with sizes); `--yes` removes them. It never touches
credentials (your HF / Token Factory / NGC tokens) or config project stanzas —
use `npa configure --forget-project` for a project — and it cannot delete cloud
IAM: pre-existing service accounts (e.g. a storage principal) are reported with
the raw `nebius iam …` command, not removed, since deleting a shared SA can break
other work.
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
    "npa configure --forget-project <alias>",
    # `npa cleanup` reads the managed-job queue through SkyPilot, so removing
    # SkyPilot first silently drops that safety check.
    "npa cleanup --yes            (local caches; keep this before the uninstall)",
    "npa skypilot uninstall --yes (last: cleanup needs it to see managed jobs)",
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
    typer.echo("Full teardown order (npa cleanup does NOT run these; they touch cloud resources):")
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
            sa_id = str(((data or {}).get("nebius") or {}).get("service_account_id", "") or "").strip()
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
    the commands in the printed runbook -- `--yes` never deletes anything in the
    cloud, which the report says explicitly so it is not mistaken for a teardown.
    """
    import shutil

    from npa.clients.config import clear_skypilot_bin

    npa_dir = Path.home() / ".npa"
    residue = _collect_residue(include_sky=include_sky)
    empty_dirs = _empty_alias_dirs(npa_dir, project)

    if not skip_jobs:
        _report_managed_jobs(sky_bin)

    total = sum(item.size for item in residue)
    if not residue and not empty_dirs:
        typer.echo("No local NPA/SkyPilot residue to clean up.")
        typer.echo(_iam_note())
        if not yes:
            _print_runbook()
        return

    typer.echo("Local residue after teardown (secret-free; your tokens/config are untouched):")
    for item in residue:
        typer.echo(f"  {item.label:<26} {_human(item.size):>8}  {item.path}")
    for empty in empty_dirs:
        typer.echo(f"  {'empty state dir':<26} {'-':>8}  {empty}")
    if residue:
        typer.echo(f"  {'total':<26} {_human(total):>8}")

    if not yes:
        typer.echo("")
        typer.echo("Re-run with --yes to remove them (or --keep-sky to leave ~/.sky).")
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
    # Drop the now-empty agents/ and workbenches/ base dirs too.
    for base_name in ("agents", "workbenches"):
        base = npa_dir / base_name
        try:
            if base.is_dir() and not any(base.iterdir()):
                base.rmdir()
        except OSError:
            pass
    typer.echo("")
    typer.echo(f"Freed ~{_human(total)} of local caches. Tokens and project config were kept.")
    typer.echo(_iam_note())
