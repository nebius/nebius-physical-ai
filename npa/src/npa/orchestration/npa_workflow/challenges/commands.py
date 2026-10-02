"""Describe the standard Workbench operations following challenge preparation."""

from pathlib import Path
import shlex

from .config import ChallengeSetup

_RUN_GUIDE = """# Run the prepared BEHAVIOR DEV evaluation

This kit fixes one task's ten prescribed DEV cases. GPU readiness is not yet verified.
Start the task-configured policy service described in policy.md before submitting.
The simulator image and asset PVC must already exist on the selected cluster.
Validate GPU rendering, asset access and policy behavior on that runtime.
Publish recipe.json and policy.md with the prepare command's --publish option.
Credentials stay in NPA's private project configuration.

## Preflight

```bash
{preflight}
```

## Submit and observe

```bash
{observe}
```

## Results

Results: {output_uri}
summary.json records completed/planned cases, evaluated_mean_q and per-case success.
Q is task progress; inspect success counts and original videos alongside it.
A partial panel's mean is not a complete-panel result. This DEV kit creates no submission ZIP.

## Failure and cleanup

Preserve the original output prefix and worker evidence after a started rollout fails.
Do not repeat it under a fresh run ID. The basic evaluator does not resume partial panels.
The [campaign guide](https://github.com/nebius/nebius-physical-ai/blob/main/docs/workbench/behavior-campaign.md)
describes the separate durable campaign path; it does not migrate this kit's results.
Cancel before removing an owned controller or worker; retain assets and checkpoints.

```bash
{cancel}
```
"""


def _preflight_commands(setup, workflow, base, project):
    infra = ["--infra", setup.infra]
    placement = ["--context", setup.infra.removeprefix("k8s/"), "--spec", workflow]
    credentials = ["npa", "workbench", "health", "preflight", *project]
    return {
        "credentials": [*credentials, "--checks", "nebius,s3", "--json"],
        "images": [*base, "preflight-images", workflow, *project, *infra, "--json"],
        "placement": [*base, "gpus", *project, *placement, "--json"],
    }


def next_commands(setup: ChallengeSetup, directory: Path, run_id: str) -> dict:
    """Return argument vectors for preflight, submission and observation.

    Args:
        setup: Validated private setup.
        directory: Prepared local evaluation kit.
        run_id: Frozen workflow identity.
    Returns:
        Named argv lists; nothing is executed.
    Raises:
        None.
    """
    workflow = str(directory / "workflow.yaml")
    base = ["npa", "workbench", "workflow"]
    project = ["--project", setup.project]
    state_uri = f"{setup.artifact_root}/{run_id}/npa-workflow"
    scope = [*project, "--workflow-s3-uri", state_uri]
    identity = ["--infra", setup.infra, "--run-id", run_id, "--runtime"]
    submit = [*base, "submit", workflow, *project, *identity]
    return {
        **_preflight_commands(setup, workflow, base, project),
        "submit_preflight": [*submit, "--plan-only"],
        "submit": submit,
        "status": [*base, "status", run_id, *scope, "--json"],
        "logs": [*base, "logs", run_id, *scope, "--stage", "evaluate"],
        "artifacts": [*base, "artifacts", run_id, *scope, "--json"],
        "cancel": [*base, "cancel", run_id, *scope, "--json"],
    }


def run_instructions(commands: dict, manifest: dict) -> str:
    """Explain remaining readiness gates and the prepared evaluation's outputs.

    Args:
        commands: Standard Workbench argument vectors.
        manifest: Kit identity and artifact locations.
    Returns:
        Operator Markdown with shell-quoted commands.
    Raises:
        None.
    """
    preflight = ("credentials", "images", "placement", "submit_preflight")
    observe = ("submit", "status", "logs", "artifacts")
    return _RUN_GUIDE.format(
        preflight="\n".join(shlex.join(commands[key]) for key in preflight),
        observe="\n".join(shlex.join(commands[key]) for key in observe),
        output_uri=manifest["output_uri"],
        cancel=shlex.join(commands["cancel"]),
    )
