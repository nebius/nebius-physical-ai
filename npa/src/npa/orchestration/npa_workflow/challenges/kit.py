"""Create private, reviewable evaluation kits for the standard workflow runtime."""

import hashlib
import json
from pathlib import Path
import re
import tempfile

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec

from .behavior import materialize_workflow, recipe_for
from .checks import check_setup, load_inputs
from .commands import next_commands, run_instructions
from .config import starter_config
from .publication import publish_inputs

__all__ = ["check_setup", "initialize", "prepare"]


def _json_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_private(directory: Path, payloads: dict[str, bytes]) -> None:
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    for filename, payload in payloads.items():
        path = directory / filename
        with path.open("xb") as stream:
            path.chmod(0o600)
            stream.write(payload)


def initialize(directory: Path) -> dict:
    """Create a private BEHAVIOR setup file with explicit missing prerequisites.

    Args:
        directory: New local directory whose parent already exists.
    Returns:
        Setup path and next action; nothing is provisioned or downloaded.
    Raises:
        OSError: The directory exists or cannot be created.
    """
    directory = directory.expanduser().absolute()
    _write_private(
        directory,
        {"setup.yaml": yaml.safe_dump(starter_config(), sort_keys=False).encode()},
    )
    return {"status": "needs-configuration", "config": str(directory / "setup.yaml")}


def _workflow_payload(workflow: dict, run_id: str) -> tuple[bytes, bytes]:
    raw = yaml.safe_dump(workflow, sort_keys=False).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="npa-challenge-plan-") as temporary:
        path = Path(temporary) / "workflow.yaml"
        path.write_bytes(raw)
        plan = build_plan(load_spec(path), run_id=run_id)
    return raw, _json_bytes(plan.to_dict())


def _manifest(setup, plan, run_id, workflow, template_digest, payloads) -> dict:
    return {
        "schema_version": "npa.challenge-kit/v1",
        "challenge": setup.challenge,
        "task": setup.task,
        "split": setup.split,
        "run_id": run_id,
        "planned_cases": len(plan["cases"]),
        "upstream_commit": setup.source.revision,
        "template_sha256": template_digest,
        "runtime_image": setup.runtime.image,
        "policy_checkpoint_sha256": setup.policy.checkpoint_sha256,
        "checkpoint_identity": "operator-declared",
        "gpu_readiness": "not-checked",
        "output_uri": workflow["config"]["output_uri"],
        "workflow_state_uri": f"{setup.artifact_root}/{run_id}/npa-workflow",
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
        },
    }


def _kit_payloads(setup, plan, runbook, directory, run_id) -> tuple[dict, dict]:
    workflow, template_digest = materialize_workflow(setup, run_id)
    raw, execution_plan = _workflow_payload(workflow, run_id)
    payloads = {
        "recipe.json": _json_bytes(recipe_for(setup)),
        "policy.md": runbook,
        "plan.json": _json_bytes(plan),
        "workflow.yaml": raw,
        "execution-plan.json": execution_plan,
    }
    manifest = _manifest(setup, plan, run_id, workflow, template_digest, payloads)
    commands = next_commands(setup, directory, run_id)
    payloads.update(
        {
            "manifest.json": _json_bytes(manifest),
            "commands.json": _json_bytes(commands),
            "RUN.md": run_instructions(commands, manifest).encode("utf-8"),
        }
    )
    return payloads, manifest


def prepare(
    config_path: Path, directory: Path, run_id: str, *, publish: bool = False
) -> dict:
    """Freeze one DEV task and optionally publish its immutable evaluation inputs.

    Args:
        config_path: Operator setup YAML.
        directory: New private kit directory whose parent exists.
        run_id: Explicit unique identity for the standard workflow run.
        publish: Verify project storage scope and upload recipe and runbook.
    Returns:
        Kit identity, artifact locations and input publication status.
    Raises:
        ValueError: Configuration, source, plan or publication verification fails.
        OSError: Local files cannot be read or the output directory already exists.
        RuntimeError: Remote input publication fails; the local kit is retained.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run_id):
        raise ValueError("run_id: use 1–63 lowercase letters, digits or hyphens")
    directory = directory.expanduser().absolute()
    setup, plan, runbook = load_inputs(config_path.expanduser())
    payloads, manifest = _kit_payloads(setup, plan, runbook, directory, run_id)
    _write_private(directory, payloads)
    if publish:
        prefix = f"{setup.artifact_root}/{run_id}/inputs/"
        publish_inputs(setup, prefix, payloads)
    return {**manifest, "directory": str(directory), "inputs_published": publish}
