"""Report and remove what a full NPA teardown leaves behind.

Tearing an environment down takes six commands in a specific order (cancel the
managed jobs, destroy the agent, destroy the cluster, delete the bucket, forget
the project, remove the SkyPilot venv). Nothing checks the order, and two things
are easy to miss: a managed job still holding the jobs controller, and the IAM
service account `npa configure` creates, which no destroy path removes.

This command is the missing overview. It reports residual state, prints the
ordered runbook, and can wipe the purely-local caches. It never deletes cloud
resources -- particularly not service accounts, which are frequently shared with
work that has nothing to do with this environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import shutil
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="cleanup", help="Report and remove NPA teardown leftovers.")
console = Console(stderr=True)

# Service accounts NPA creates. They are reported, never deleted: `lerobot-training`
# in particular is shared with other work in the same project.
REPORTED_SERVICE_ACCOUNTS = ("lerobot-training", "npa-agent")

TEARDOWN_RUNBOOK = (
    "sky jobs cancel -a  (and wait for `sky jobs queue --all` to show them terminal)",
    "npa agent destroy --project <alias> --name <name> --yes",
    "npa cluster down --force",
    "delete the object-storage bucket",
    "remove the project entry from ~/.npa/config.yaml",
    "npa cleanup --yes  (local caches)",
)


@dataclass
class Residue:
    """Local and cloud state left over after a teardown."""

    local_paths: list[Path] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    live_jobs: list[str] = field(default_factory=list)
    service_accounts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.local_paths or self.projects or self.live_jobs)

    def to_dict(self) -> dict[str, object]:
        return {
            "local_paths": [str(path) for path in self.local_paths],
            "projects": list(self.projects),
            "live_jobs": list(self.live_jobs),
            "service_accounts": list(self.service_accounts),
            "notes": list(self.notes),
            "clean": self.clean,
        }


def _npa_home() -> Path:
    return Path.home() / ".npa"


def _local_cache_paths() -> list[Path]:
    home = _npa_home()
    return [
        home / "skypilot-venv",
        home / "terraform-plugin-cache",
        Path.home() / ".sky",
    ]


def _empty_dirs(parent: Path) -> list[Path]:
    """Return alias directories under ``parent`` that hold nothing."""

    if not parent.is_dir():
        return []
    empty: list[Path] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        if not any(child.iterdir()):
            empty.append(child)
    return empty


def _configured_projects() -> tuple[list[str], str]:
    config_path = _npa_home() / "config.yaml"
    if not config_path.is_file():
        return [], ""
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        return [], f"could not read {config_path}: {exc}"
    projects = data.get("projects") if isinstance(data, dict) else None
    if not isinstance(projects, dict):
        return [], ""
    return sorted(str(name) for name in projects), ""


def _nonterminal_jobs(sky_bin: str) -> tuple[list[str], str]:
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


def collect_residue(*, sky_bin: str = "", include_jobs: bool = True) -> Residue:
    """Gather what a teardown has left behind, without changing anything."""

    residue = Residue()
    residue.local_paths = [path for path in _local_cache_paths() if path.exists()]
    home = _npa_home()
    for parent in (home / "agents", home / "clusters", home / "workbenches"):
        residue.local_paths.extend(_empty_dirs(parent))

    projects, project_note = _configured_projects()
    residue.projects = projects
    if project_note:
        residue.notes.append(project_note)

    if include_jobs:
        jobs, job_note = _nonterminal_jobs(sky_bin)
        residue.live_jobs = jobs
        if job_note:
            residue.notes.append(job_note)
        if jobs:
            residue.notes.append(
                "A non-terminal managed job blocks `sky down` of the jobs controller. "
                "A job whose pod cannot start stays PENDING forever rather than failing, "
                "so check it before assuming it is still doing work."
            )

    residue.service_accounts = list(REPORTED_SERVICE_ACCOUNTS)
    residue.notes.append(
        "Service accounts are reported, never deleted: "
        f"{', '.join(REPORTED_SERVICE_ACCOUNTS)} may be shared with other work in the "
        "same project. Remove them yourself with the Nebius CLI if they are genuinely unused."
    )
    return residue


def remove_local_caches(residue: Residue, *, keep_sky: bool = False) -> list[Path]:
    """Delete the purely-local caches in ``residue``. No cloud state is touched."""

    removed: list[Path] = []
    sky_home = Path.home() / ".sky"
    for path in residue.local_paths:
        if keep_sky and path == sky_home:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            console.print(f"[yellow]warning:[/yellow] could not remove {path}: {exc}")
            continue
        removed.append(path)
    return removed


@app.callback(invoke_without_command=True)
def cleanup_cmd(
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Remove the local caches listed in the report. Never touches cloud resources.",
    ),
    keep_sky: bool = typer.Option(
        False,
        "--keep-sky",
        help="With --yes, keep ~/.sky (SkyPilot's own state) in place.",
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
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON report."),
) -> None:
    """Report NPA teardown leftovers, and optionally wipe the local caches."""

    residue = collect_residue(sky_bin=sky_bin, include_jobs=not skip_jobs)
    removed: list[Path] = []
    if yes:
        removed = remove_local_caches(residue, keep_sky=keep_sky)

    if json_output:
        payload = residue.to_dict()
        payload["removed"] = [str(path) for path in removed]
        payload["runbook"] = list(TEARDOWN_RUNBOOK)
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if residue.live_jobs:
        typer.echo(f"managed jobs still non-terminal: {', '.join(residue.live_jobs)}")
    if residue.projects:
        typer.echo(f"configured projects: {', '.join(residue.projects)}")
    if residue.local_paths:
        typer.echo("local leftovers:")
        for path in residue.local_paths:
            marker = "removed" if path in removed else "present"
            typer.echo(f"  {path} ({marker})")
    if residue.clean and not removed:
        typer.echo("no local leftovers found")
    typer.echo(f"service accounts (reported, not deleted): {', '.join(residue.service_accounts)}")
    for note in residue.notes:
        typer.echo(f"note: {note}")
    if not yes:
        typer.echo("")
        typer.echo("Full teardown order:")
        for index, step in enumerate(TEARDOWN_RUNBOOK, start=1):
            typer.echo(f"  {index}. {step}")
        typer.echo("Rerun with --yes to remove the local caches above.")
