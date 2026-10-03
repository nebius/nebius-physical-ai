"""Check local challenge prerequisites without contacting cloud infrastructure."""

from pathlib import Path

from pydantic import ValidationError
import yaml

from .behavior import inspect_source, local_path
from .config import ChallengeSetup


def load_inputs(config_path: Path) -> tuple[ChallengeSetup, dict, bytes]:
    """Validate the setup, public evaluator source and operator policy runbook.

    Args:
        config_path: Private setup YAML, with paths relative to this file.
    Returns:
        Validated setup, prescribed case plan and policy runbook bytes.
    Raises:
        ValueError: A local prerequisite is missing or invalid.
    """
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("config: provide a readable UTF-8 YAML setup file") from exc
    try:
        setup = ChallengeSetup.model_validate(document)
    except ValidationError as exc:
        issues = [
            f"{'.'.join(map(str, issue['loc'])) or 'config'}: {issue['msg']}"
            for issue in exc.errors(include_input=False, include_url=False)
        ]
        raise ValueError("\n".join(issues)) from exc
    plan = inspect_source(setup, config_path)
    runbook = _read_runbook(config_path, setup.policy.runbook)
    return setup, plan, runbook


def _read_runbook(config_path: Path, value: str) -> bytes:
    try:
        runbook = local_path(config_path, value).read_bytes()
        if not runbook.decode("utf-8").strip():
            raise ValueError(
                "policy.runbook: describe how to reproduce the policy service"
            )
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            "policy.runbook: provide a readable UTF-8 policy runbook"
        ) from exc
    return runbook


def check_setup(config_path: Path) -> dict:
    """Report local preparation readiness separately from GPU execution readiness.

    Args:
        config_path: Private configuration YAML.
    Returns:
        Machine-readable checks with actionable local failures.
    Raises:
        None.
    """
    result = {
        "schema_version": "npa.challenge-check/v1",
        "gpu_readiness": "not-checked",
    }
    try:
        setup, plan, _ = load_inputs(config_path)
    except ValueError as exc:
        return {**result, "status": "blocked", "issues": str(exc).splitlines()}
    return {
        **result,
        "status": "ready-to-prepare",
        "issues": [],
        "task": setup.task,
        "split": setup.split,
        "planned_cases": len(plan["cases"]),
        "upstream_commit": setup.source.revision,
        "checkpoint_identity": "operator-declared",
    }
