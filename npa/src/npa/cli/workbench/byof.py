"""npa workbench byof — bring-your-own-fork OSS onboarding."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

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


def _script_path() -> Path:
    """Locate the BYOF runner in a checkout or staged NPA source package."""

    override = os.environ.get("NPA_REPO_ROOT", "").strip()
    roots = (
        [Path(override)]
        if override
        else [*Path(__file__).resolve().parents, Path.cwd()]
    )
    for root in roots:
        for relative in (
            Path("npa/scripts/run_byof_repo.py"),
            Path("scripts/run_byof_repo.py"),
        ):
            candidate = root / relative
            if candidate.is_file():
                return candidate
    preferred = roots[0] if roots else Path.cwd()
    return preferred / "npa" / "scripts" / "run_byof_repo.py"


class BaseProfile(str, Enum):
    ubuntu = "ubuntu"
    isaac_lab = "isaac-lab"
    prebuilt = "prebuilt"


class Workload(str, Enum):
    container_verify = "container-verify"
    rl_train = "rl-train"
    datagen = "datagen"
    solution_smoke = "solution-smoke"


class RepoAuth(str, Enum):
    none = "none"
    github = "github"


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


@contextmanager
def _robotwin_runtime_materialization(
    runner: Any, argv: list[str]
) -> Iterator[Any | None]:
    """Materialize a validated worker transport for the internal workflow bridge."""

    from npa.orchestration.npa_workflow.robotwin_preflight import (
        CONTEXT_ENV_NAMES,
        MATERIALIZED_KUBECONFIG_ENV,
        MATERIALIZED_SKYPILOT_CONFIG_ENV,
        PUBLIC_CONTEXT_ENV,
        TRANSPORT_CONTEXT_ENV,
        is_robotwin_request,
        load_runtime_authorization,
        materialize_transport,
        validate_invocation,
    )

    parsed = runner._parse_args(argv)
    transport = os.environ.get(TRANSPORT_CONTEXT_ENV, "")
    if not is_robotwin_request(
        solution_name=parsed.solution_name,
        repo_url=parsed.repo_url,
        base_image=parsed.base_image,
        image=parsed.image,
        smoke_command=parsed.smoke_command,
        capability_name=parsed.capability_name,
        yaml_path=parsed.yaml,
    ):
        yield None
        return
    if not transport:
        yield None
        return
    validate_invocation(parsed)
    if os.environ.get(PUBLIC_CONTEXT_ENV, "").strip():
        raise ValueError("RoboTwin worker received conflicting context channels")
    previous = {name: os.environ.get(name) for name in CONTEXT_ENV_NAMES}
    with tempfile.TemporaryDirectory(prefix="npa-robotwin-runtime-") as raw_dir:
        materialized = materialize_transport(transport, Path(raw_dir))
        try:
            os.environ.pop(TRANSPORT_CONTEXT_ENV, None)
            os.environ[PUBLIC_CONTEXT_ENV] = str(materialized.context_path)
            os.environ[MATERIALIZED_KUBECONFIG_ENV] = str(
                materialized.kubeconfig_path
            )
            os.environ[MATERIALIZED_SKYPILOT_CONFIG_ENV] = str(
                materialized.skypilot_config_path
            )
            yield load_runtime_authorization()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def build_byof_argv(
    *,
    repo_url: str,
    repo_ref: str = "main",
    repo_auth: str = "none",
    repo_token_env: str = "",
    base_profile: str = "ubuntu",
    base_image: str = "",
    workload: str = "container-verify",
    build_command: str = "",
    smoke_command: str = "",
    solution_name: str = "",
    capability_name: str = "",
    smoke_artifact_name: str = "",
    runtime_context_env: str = "",
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
        "--repo-auth",
        repo_auth,
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
    if repo_token_env:
        argv.extend(["--repo-token-env", repo_token_env])
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
    if runtime_context_env:
        argv.extend(["--runtime-context-env", runtime_context_env])
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
    repo_url: str = typer.Option(
        ..., "--repo-url", help="GitHub/GitLab repository URL without credentials."
    ),
    repo_ref: str = typer.Option(
        "main", "--repo-ref", help="Git ref to clone into the image."
    ),
    repo_auth: RepoAuth = typer.Option(
        RepoAuth.none,
        "--repo-auth",
        help="none for public source; github for secret-mounted private access.",
    ),
    repo_token_env: str = typer.Option(
        "",
        "--repo-token-env",
        help=(
            "Environment variable holding a fine-grained GitHub token; otherwise "
            "use GH_TOKEN, GITHUB_TOKEN, or the existing gh login."
        ),
    ),
    base_profile: BaseProfile = typer.Option(
        BaseProfile.ubuntu,
        "--base-profile",
        help="ubuntu, isaac-lab, or prebuilt with --base-image tool://<registered-tool>.",
    ),
    base_image: str = typer.Option(
        "", "--base-image", help="Explicit base image override."
    ),
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
    solution_name: str = typer.Option(
        "", "--solution-name", help="Registry solution name."
    ),
    capability_name: str = typer.Option(
        "", "--capability-name", help="Registry capability name."
    ),
    smoke_artifact_name: str = typer.Option(
        "",
        "--smoke-artifact-name",
        help="Expected JSON artifact filename for solution-smoke.",
    ),
    runtime_context_env: str = typer.Option(
        "",
        "--runtime-context-env",
        help=(
            "Environment-variable name containing owner-only runtime authorization; "
            "the secret value is never placed in argv."
        ),
    ),
    project: str = typer.Option(
        "", "--project", help="Project alias for registry resolution."
    ),
    registry: str = typer.Option("", "--registry", help="Override registry host/path."),
    image: str = typer.Option(
        "", "--image", help="Fully-qualified image ref to build/push."
    ),
    run_id: str = typer.Option(
        "", "--run-id", help="Run identifier (default byof-<stamp>)."
    ),
    task: str = typer.Option(
        "Isaac-Cartpole-v0", "--task", help="Isaac task for RL/datagen."
    ),
    iterations: int = typer.Option(1, "--iterations", help="RL training iterations."),
    num_envs: int = typer.Option(4, "--num-envs", help="Parallel sim envs (datagen)."),
    num_demos: int = typer.Option(
        4, "--num-demos", help="Demonstrations to record (datagen)."
    ),
    yaml_path: str = typer.Option(
        "", "--yaml", help="Optional SkyPilot YAML override."
    ),
    output_root: str = typer.Option(
        "", "--output-root", help="Override workload output root."
    ),
    wait_timeout: int = typer.Option(
        21600, "--wait-timeout", help="Workload wait timeout seconds."
    ),
    poll_interval: int = typer.Option(
        60, "--poll-interval", help="Poll interval seconds."
    ),
    sky_bin: str = typer.Option("", "--sky-bin", help="SkyPilot binary override."),
    config_path: str = typer.Option(
        "", "--config-path", help="SkyPilot global config YAML."
    ),
    cleanup: bool = typer.Option(
        True, "--cleanup/--no-cleanup", help="Cleanup SkyPilot resources."
    ),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip docker build."),
    skip_push: bool = typer.Option(False, "--skip-push", help="Skip docker push."),
    skip_run: bool = typer.Option(
        False, "--skip-run", help="Build/push only; skip live workload."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print argv JSON and exit without running."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format for dry-run."
    ),
) -> None:
    """Build/push a BYOF image and optionally run a live workload."""

    argv = build_byof_argv(
        repo_url=repo_url,
        repo_ref=repo_ref,
        repo_auth=repo_auth.value,
        repo_token_env=repo_token_env,
        base_profile=base_profile.value,
        base_image=base_image,
        workload=workload.value,
        build_command=build_command,
        smoke_command=smoke_command,
        solution_name=solution_name,
        capability_name=capability_name,
        smoke_artifact_name=smoke_artifact_name,
        runtime_context_env=runtime_context_env,
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

    from npa.orchestration.npa_workflow.robotwin_preflight import (
        TRANSPORT_CONTEXT_ENV,
        is_robotwin_request,
    )

    if is_robotwin_request(
        solution_name=solution_name,
        repo_url=repo_url,
        base_image=base_image,
        image=image,
        smoke_command=smoke_command,
        capability_name=capability_name,
        yaml_path=yaml_path,
    ):
        if not os.environ.get(TRANSPORT_CONTEXT_ENV, ""):
            raise typer.BadParameter(
                "RoboTwin runs only through normal npa workbench workflow submit"
            )

    runner = _load_runner()
    with _robotwin_runtime_materialization(runner, argv) as authorization:
        code = int(
            runner._run_authorized_robotwin(argv, authorization=authorization)
            if authorization is not None
            else runner.main(argv)
        )
    raise SystemExit(code)


@app.command("ladder")
def ladder_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Show the OSS onboarding ladder (Tier 0 → Tier 2)."""

    payload: dict[str, Any] = {
        "doc": _LADDER_DOC,
        "skill": _SKILL_PATH,
        "tiers": [
            {"tier": 0, "name": "BYOF container", "cli": "npa workbench byof run"},
            {
                "tier": 1,
                "name": "Solution workflow",
                "cli": "npa workbench workflow validate-spec",
            },
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
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Report BYOF packaging surfaces (CLI / SDK / YAML)."""

    payload = {
        "cli": "npa workbench byof",
        "sdk": "npa.sdk.workbench.byof",
        "tool_refs": ["workbench.byof.repo", "workbench.isaac_lab.byof_repo"],
        "workflow": "workflows/testing/byof.yaml",
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
