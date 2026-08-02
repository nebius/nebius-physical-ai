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


def _iam_note() -> str:
    """A hint about cloud IAM leftovers npa deliberately does not delete."""
    try:
        from npa.clients.credentials import CREDENTIALS_PATH
        import yaml

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
        pass
    return (
        "Cloud IAM (not removed): pre-existing service accounts are left in place; "
        "remove one deliberately with `nebius iam service-account delete --id <id>`."
    )


def cleanup_cmd(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Remove the local caches (otherwise just report)."
    ),
    include_sky: bool = typer.Option(
        True,
        "--include-sky/--keep-sky",
        help="Also remove SkyPilot's own ~/.sky state cache (safe once no clusters/jobs run).",
    ),
    project: str = typer.Option(
        "", "--project", help="Scope the empty per-alias state-dir report to this alias."
    ),
) -> None:
    """Report (or with --yes remove) local NPA/SkyPilot residue left after teardown."""
    import shutil

    from npa.clients.config import clear_skypilot_bin

    npa_dir = Path.home() / ".npa"
    residue = _collect_residue(include_sky=include_sky)
    empty_dirs = _empty_alias_dirs(npa_dir, project)

    total = sum(item.size for item in residue)
    if not residue and not empty_dirs:
        typer.echo("No local NPA/SkyPilot residue to clean up.")
        typer.echo(_iam_note())
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
