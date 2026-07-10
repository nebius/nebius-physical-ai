"""npa workbench byof — bring-your-own-fork OSS onboarding."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    name="byof",
    help="Onboard an OSS repo as a BYOF container (Tier 0 of the OSS ladder).",
    no_args_is_help=True,
)
console = Console(stderr=True)

_LADDER_DOC = "docs/architecture/oss-onboarding-ladder.md"
_SKILL_PATH = "skills/workflows/byof-onboard/SKILL.md"


def _repo_root() -> Path:
    override = os.environ.get("NPA_REPO_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "npa" / "scripts" / "run_byof_repo.py").is_file():
            return parent
    return Path.cwd()


def _script_path() -> Path:
    return _repo_root() / "npa" / "scripts" / "run_byof_repo.py"


class BaseProfile(str, Enum):
    ubuntu = "ubuntu"
    isaac_lab = "isaac-lab"


class Workload(str, Enum):
    container_verify = "container-verify"
    rl_train = "rl-train"
    datagen = "datagen"
    solution_smoke = "solution-smoke"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _load_runner():
    script = _script_path()
    spec = importlib.util.spec_from_file_location("npa_run_byof_repo", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load BYOF runner at {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_byof_argv(
    *,
    repo_url: str,
    repo_ref: str = "main",
    base_profile: str = "ubuntu",
    base_image: str = "",
    workload: str = "container-verify",
    build_command: str = "",
    smoke_command: str = "",
    solution_name: str = "",
    capability_name: str = "",
    smoke_artifact_name: str = "",
    project: str = "",
    registry: str = "",
    image: str = "",
    run_id: str = "",
    task: str = "Isaac-Cartpole-v0",
    iterations: int = 1,
    num_envs: int = 4,
    num_demos: int = 4,
    yaml_path: str = "",
    output_root: str = "",
    wait_timeout: int = 21600,
    poll_interval: int = 60,
    sky_bin: str = "",
    config_path: str = "",
    cleanup: bool = True,
    skip_build: bool = False,
    skip_push: bool = False,
    skip_run: bool = False,
) -> list[str]:
    """Build argv for ``run_byof_repo.py`` (shared by CLI and workflow catalog)."""

    argv = [
        "--repo-url",
        repo_url,
        "--repo-ref",
        repo_ref,
        "--base-profile",
        base_profile,
        "--workload",
        workload,
        "--task",
        task,
        "--iterations",
        str(iterations),
        "--num-envs",
        str(num_envs),
        "--num-demos",
        str(num_demos),
        "--wait-timeout",
        str(wait_timeout),
        "--poll-interval",
        str(poll_interval),
    ]
    if base_image:
        argv.extend(["--base-image", base_image])
    if build_command:
        argv.extend(["--build-command", build_command])
    if smoke_command:
        argv.extend(["--smoke-command", smoke_command])
    if solution_name:
        argv.extend(["--solution-name", solution_name])
    if capability_name:
        argv.extend(["--capability-name", capability_name])
    if smoke_artifact_name:
        argv.extend(["--smoke-artifact-name", smoke_artifact_name])
    if project:
        argv.extend(["--project", project])
    if registry:
        argv.extend(["--registry", registry])
    if image:
        argv.extend(["--image", image])
    if run_id:
        argv.extend(["--run-id", run_id])
    if yaml_path:
        argv.extend(["--yaml", yaml_path])
    if output_root:
        argv.extend(["--output-root", output_root])
    if sky_bin:
        argv.extend(["--sky-bin", sky_bin])
    if config_path:
        argv.extend(["--config-path", config_path])
    argv.append("--cleanup" if cleanup else "--no-cleanup")
    if skip_build:
        argv.append("--skip-build")
    if skip_push:
        argv.append("--skip-push")
    if skip_run:
        argv.append("--skip-run")
    return argv


@app.command("run")
def run_cmd(
    repo_url: str = typer.Option(..., "--repo-url", help="Public GitHub/GitLab repo URL."),
    repo_ref: str = typer.Option("main", "--repo-ref", help="Git ref to clone into the image."),
    base_profile: BaseProfile = typer.Option(
        BaseProfile.ubuntu,
        "--base-profile",
        help="ubuntu (generic OSS) or isaac-lab (sim workloads).",
    ),
    base_image: str = typer.Option("", "--base-image", help="Explicit base image override."),
    workload: Workload = typer.Option(
        Workload.container_verify,
        "--workload",
        help="container-verify, rl-train, datagen, or solution-smoke.",
    ),
    build_command: str = typer.Option(
        "",
        "--build-command",
        help="Optional shell command run at image build time from /opt/byof.",
    ),
    smoke_command: str = typer.Option(
        "",
        "--smoke-command",
        help="Optional documented shell command for solution-smoke from /opt/byof.",
    ),
    solution_name: str = typer.Option("", "--solution-name", help="Registry solution name."),
    capability_name: str = typer.Option("", "--capability-name", help="Registry capability name."),
    smoke_artifact_name: str = typer.Option(
        "",
        "--smoke-artifact-name",
        help="Expected JSON artifact filename for solution-smoke.",
    ),
    project: str = typer.Option("", "--project", help="Project alias for registry resolution."),
    registry: str = typer.Option("", "--registry", help="Override registry host/path."),
    image: str = typer.Option("", "--image", help="Fully-qualified image ref to build/push."),
    run_id: str = typer.Option("", "--run-id", help="Run identifier (default byof-<stamp>)."),
    task: str = typer.Option("Isaac-Cartpole-v0", "--task", help="Isaac task for RL/datagen."),
    iterations: int = typer.Option(1, "--iterations", help="RL training iterations."),
    num_envs: int = typer.Option(4, "--num-envs", help="Parallel sim envs (datagen)."),
    num_demos: int = typer.Option(4, "--num-demos", help="Demonstrations to record (datagen)."),
    yaml_path: str = typer.Option("", "--yaml", help="Optional SkyPilot YAML override."),
    output_root: str = typer.Option("", "--output-root", help="Override workload output root."),
    wait_timeout: int = typer.Option(21600, "--wait-timeout", help="Workload wait timeout seconds."),
    poll_interval: int = typer.Option(60, "--poll-interval", help="Poll interval seconds."),
    sky_bin: str = typer.Option("", "--sky-bin", help="SkyPilot binary override."),
    config_path: str = typer.Option("", "--config-path", help="SkyPilot global config YAML."),
    cleanup: bool = typer.Option(True, "--cleanup/--no-cleanup", help="Cleanup SkyPilot resources."),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip docker build."),
    skip_push: bool = typer.Option(False, "--skip-push", help="Skip docker push."),
    skip_run: bool = typer.Option(False, "--skip-run", help="Build/push only; skip live workload."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print argv JSON and exit without running."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format for dry-run."),
) -> None:
    """Build/push a BYOF image and optionally run a live workload."""

    argv = build_byof_argv(
        repo_url=repo_url,
        repo_ref=repo_ref,
        base_profile=base_profile.value,
        base_image=base_image,
        workload=workload.value,
        build_command=build_command,
        smoke_command=smoke_command,
        solution_name=solution_name,
        capability_name=capability_name,
        smoke_artifact_name=smoke_artifact_name,
        project=project,
        registry=registry,
        image=image,
        run_id=run_id,
        task=task,
        iterations=iterations,
        num_envs=num_envs,
        num_demos=num_demos,
        yaml_path=yaml_path,
        output_root=output_root,
        wait_timeout=wait_timeout,
        poll_interval=poll_interval,
        sky_bin=sky_bin,
        config_path=config_path,
        cleanup=cleanup,
        skip_build=skip_build,
        skip_push=skip_push,
        skip_run=skip_run,
    )
    if dry_run:
        payload: dict[str, Any] = {
            "script": "npa/scripts/run_byof_repo.py",
            "argv": argv,
            "ladder": _LADDER_DOC,
            "skill": _SKILL_PATH,
        }
        if output == OutputFormat.json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(" ".join(["npa", "workbench", "byof", "run", *argv]))
        return

    runner = _load_runner()
    code = int(runner.main(argv))
    raise SystemExit(code)


@app.command("ladder")
def ladder_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show the OSS onboarding ladder (Tier 0 → Tier 2)."""

    payload = {
        "doc": _LADDER_DOC,
        "skill": _SKILL_PATH,
        "tiers": [
            {"tier": 0, "name": "BYOF container", "cli": "npa workbench byof run"},
            {"tier": 1, "name": "Solution workflow", "cli": "npa workbench workflow validate-spec"},
            {"tier": 2, "name": "First-class tool", "cli": "npa workbench <tool>"},
        ],
    }
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"OSS onboarding ladder: {_LADDER_DOC}")
    typer.echo(f"Operator skill: {_SKILL_PATH}")
    for tier in payload["tiers"]:
        typer.echo(f"  Tier {tier['tier']}: {tier['name']} — {tier['cli']}")


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Report BYOF packaging surfaces (CLI / SDK / YAML)."""

    payload = {
        "cli": "npa workbench byof",
        "sdk": "npa.sdk.workbench.byof",
        "tool_refs": ["workbench.byof.repo", "workbench.isaac_lab.byof_repo"],
        "workflow": "npa/workflows/workbench/npa-workflows/byof.yaml",
        "script": "npa/scripts/run_byof_repo.py",
        "packaging": "docs/workbench/container-packaging.md",
        "ladder": _LADDER_DOC,
    }
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, list):
            typer.echo(f"{key}: {', '.join(value)}")
        else:
            typer.echo(f"{key}: {value}")
