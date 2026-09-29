"""Adapt a validated BEHAVIOR setup to the shipped evaluation workflow."""

import hashlib
from importlib.resources import files
from pathlib import Path
import subprocess

import npa
import yaml

from npa.workflow_build import catalog_files
from npa.workflows.behavior_challenge.protocol import make_plan, verify_upstream

from .config import ChallengeSetup


def local_path(config_path: Path, value: str) -> Path:
    """Resolve an operator file relative to its setup file.

    Args:
        config_path: Setup YAML location.
        value: Absolute or setup-relative local path.
    Returns:
        Resolved local path.
    Raises:
        OSError: A path cannot be resolved.
    """
    return (config_path.parent / Path(value).expanduser()).resolve()


def recipe_for(setup: ChallengeSetup) -> dict:
    """Create the existing evaluator's recipe without case-selection overrides.

    Args:
        setup: Validated operator configuration.
    Returns:
        One task's fixed development recipe.
    Raises:
        None.
    """
    return {
        "schema": "npa.behavior.recipe.v1",
        "tasks": [setup.task],
        "split": setup.split,
        "policy_checkpoint_sha256": setup.policy.checkpoint_sha256,
        "upstream_commit": setup.source.revision,
    }


def inspect_source(setup: ChallengeSetup, config_path: Path) -> dict:
    """Check public source and freeze the evaluator's prescribed case plan.

    Args:
        setup: Validated setup.
        config_path: Configuration location for relative paths.
    Returns:
        The official plan from the shared BEHAVIOR protocol implementation.
    Raises:
        ValueError: Source verification or official task selection fails.
    """
    root = local_path(config_path, setup.source.checkout)
    try:
        verify_upstream(root, setup.source.revision)
        return make_plan(recipe_for(setup), root)
    except (
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        raise ValueError(
            "Use an unmodified checkout at source.revision and an official task ID"
        ) from exc


def workflow_template() -> tuple[dict, str]:
    """Load the canonical workflow from a checkout or an installed distribution.

    Args:
        None.
    Returns:
        Parsed workflow and its original SHA-256.
    Raises:
        OSError: The shipped workflow is unavailable.
        ValueError: The catalog does not contain the expected workflow.
    """
    relative = Path("testing/behavior-challenge-eval.yaml")
    package_root = Path(npa.__file__).resolve().parents[2]
    source = catalog_files(package_root).get(relative)
    raw = (
        source.read_bytes()
        if source
        else files("npa.workflows").joinpath(str(relative)).read_bytes()
    )
    document = yaml.safe_load(raw)
    if document.get("metadata", {}).get("name") != "behavior-challenge-eval":
        raise ValueError("The shipped BEHAVIOR workflow is missing or incompatible")
    return document, hashlib.sha256(raw).hexdigest()


def _workflow_config(setup: ChallengeSetup, run_id: str) -> dict:
    prefix = f"{setup.artifact_root}/{run_id}"
    return {
        "bucket": setup.artifact_root.split("/")[2],
        "prefix": f"{setup.artifact_root.split('/', 3)[3]}/{run_id}",
        "input_uri": f"{prefix}/inputs/recipe.json",
        "policy_readme_uri": f"{prefix}/inputs/policy.md",
        "output_uri": f"{prefix}/results/",
        "runtime_image": setup.runtime.image,
        "assets_claim": setup.runtime.assets_claim,
        "data_root": setup.runtime.data_root,
        "upstream_root": setup.runtime.upstream_root,
        "evaluator_python": setup.runtime.evaluator_python,
        "policy_host": setup.policy.host,
        "policy_port": setup.policy.port,
    }


def materialize_workflow(setup: ChallengeSetup, run_id: str) -> tuple[dict, str]:
    """Apply setup values to the existing workflow and its asset mount.

    Args:
        setup: Validated operator configuration.
        run_id: Explicit workflow invocation identity.
    Returns:
        Materialized workflow and source-template digest.
    Raises:
        OSError: The template cannot be read.
        ValueError: The template is incompatible.
    """
    workflow, digest = workflow_template()
    workflow["config"].update(_workflow_config(setup, run_id))
    simulator = workflow["resources"]["simulator"]
    simulator["accelerators"] = setup.runtime.accelerator
    mounts = simulator["kubernetes"]["pod_config"]["spec"]["containers"][0][
        "volumeMounts"
    ]
    for mount in mounts:
        if mount["name"] == "behavior-assets":
            mount["mountPath"] = setup.runtime.data_root
    return workflow, digest
