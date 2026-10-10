"""Expand dataset selections into isolated, validated workflow run plans."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr
import yaml

from npa.orchestration.npa_workflow.interpreter import build_reachability_plan
from npa.orchestration.npa_workflow.submit import load_spec_for_submit
from npa.orchestration.npa_workflow.src_staging import (
    find_npa_package_root,
    source_fingerprint,
)
from npa.workflows.data_factory_input import (
    select_paidf_input,
    validate_lerobot_selector,
)

Name = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,39}$")]
Scalar = StrictStr | StrictInt | StrictBool
ConfigKey = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _EpisodeRange(_Contract):
    """An explicit half-open episode range."""

    start: Annotated[int, Field(ge=0)]
    stop: Annotated[int, Field(gt=0)]


class _BatchEntry(_Contract):
    """One ordinary workflow run or a selection of LeRobot episodes."""

    id: Name
    vars: dict[ConfigKey, Scalar] = Field(default_factory=dict)
    input_uri: str = ""
    lerobot_uri: str = ""
    lerobot_camera: str = ""
    episodes: list[StrictInt] | _EpisodeRange | None = None


class _BatchManifest(_Contract):
    """Private submission coordinates and entries for one immutable batch."""

    apiVersion: Literal["npa.workflow.batch/v0.0.1"]
    batch_id: Name
    workflow: str
    project: Annotated[str, Field(min_length=1)]
    infra: str = ""
    config_path: str = ""
    vars: dict[ConfigKey, Scalar] = Field(default_factory=dict)
    secret_env: list[Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]] = (
        Field(default_factory=list)
    )
    entries: Annotated[list[_BatchEntry], Field(min_length=1)]


def _episodes(entry: _BatchEntry) -> list[int | None]:
    if not entry.lerobot_uri:
        if entry.episodes is not None or entry.lerobot_camera:
            raise ValueError(f"{entry.id}: episodes and camera require lerobot_uri")
        return [None]
    episodes = entry.episodes
    if isinstance(episodes, _EpisodeRange):
        episodes = list(range(episodes.start, episodes.stop))
    if not episodes or any(value < 0 for value in episodes):
        raise ValueError(f"{entry.id}: supply nonempty, non-negative episodes")
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"{entry.id}: duplicate episodes")
    return list(episodes)


def _input_args(entry: _BatchEntry, episode: int | None) -> list[str]:
    selection = select_paidf_input(
        input_uri=entry.input_uri, lerobot_uri=entry.lerobot_uri
    )
    validate_lerobot_selector(
        selection=selection,
        camera=entry.lerobot_camera,
        episode=episode or 0,
        require_explicit_selection=bool(entry.lerobot_uri),
        episode_was_explicit=episode is not None,
    )
    if entry.input_uri:
        return ["--input-uri", entry.input_uri]
    if entry.lerobot_uri:
        return [
            "--lerobot-uri",
            entry.lerobot_uri,
            "--lerobot-camera",
            entry.lerobot_camera,
            "--lerobot-episode",
            str(episode),
            "--require-explicit-lerobot-selection",
        ]
    return []


def _value(value: Scalar) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _run_plan(
    manifest: _BatchManifest, workflow: Path, entry: _BatchEntry, episode: int | None
) -> dict:
    suffix = "" if episode is None else f"-e{episode:06d}"
    run_id = f"{manifest.batch_id}-{entry.id}{suffix}"
    variables = {
        key: _value(value) for key, value in (manifest.vars | entry.vars).items()
    }
    spec = load_spec_for_submit(workflow, config_overrides=variables)
    variables.update(
        {
            key: str(spec.config[key])
            for key in ("bucket", "prefix")
            if key in spec.config
        }
    )
    from npa.orchestration.npa_workflow.run_state import is_paidf_input_workflow_name

    input_args = _input_args(entry, episode)
    paidf = is_paidf_input_workflow_name(spec.name)
    if paidf and not input_args:
        raise ValueError(f"{entry.id}: PAIDF batches require an explicit input")
    if input_args and not paidf:
        raise ValueError(f"{entry.id}: input selectors require a PAIDF workflow")
    if "{{run.id}}" not in str(spec.config.get("prefix", "")):
        raise ValueError("batch workflow config.prefix must include {{run.id}}")
    if spec.name == "paidf-cosmos3" and input_args:
        spec.config.update(_selected_input_config(entry, episode))
    plan = build_reachability_plan(spec, run_id=run_id)
    return {
        "run_id": run_id,
        "entry_id": entry.id,
        "episode": episode,
        "input_args": input_args,
        "vars": variables,
        "outputs": sorted(
            {item["uri"] for step in plan.steps for item in step.outputs}
        ),
    }


def _selected_input_config(entry, episode):
    if entry.lerobot_uri:
        return {
            "input_kind": "lerobot",
            "lerobot_dataset_uri": entry.lerobot_uri,
            "input_episode": str(episode),
            "input_camera": entry.lerobot_camera,
        }
    return {"input_kind": "video", "input_video_uri": entry.input_uri}


def _check_output_isolation(runs: list[dict]) -> None:
    owners: dict[str, str] = {}
    for run in runs:
        for uri in run["outputs"]:
            normalized = uri.rstrip("/")
            owner = owners.setdefault(normalized, run["run_id"])
            if owner != run["run_id"]:
                raise ValueError(
                    f"declared output collision between {owner} and {run['run_id']}"
                )
    ancestors: list[str] = []
    for uri in sorted(owners):
        while ancestors and not uri.startswith(ancestors[-1] + "/"):
            ancestors.pop()
        if any(owners[ancestor] != owners[uri] for ancestor in ancestors):
            raise ValueError("declared output prefixes overlap across batch runs")
        ancestors.append(uri)


def _read_manifest(path: Path) -> _BatchManifest:
    loader = _BatchLoader(path.read_text())
    try:
        payload = loader.get_single_data()
    finally:
        loader.dispose()
    return _BatchManifest.model_validate(payload)


class _BatchLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("batch YAML mapping keys must be strings")
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate YAML mapping key in batch manifest")
        return super().construct_mapping(node, deep=deep)


def plan_batch(manifest_path: Path) -> dict:
    """Validate and expand a private batch without storage or cluster access.

    Args:
        manifest_path: Local YAML batch manifest; workflow paths are relative to it.
    Returns:
        Fingerprinted plan with unique run IDs and declared output locations.
    Raises:
        ValueError: Invalid entries, duplicate IDs, or overlapping declared outputs.
        OSError: The manifest or workflow cannot be read.
        NpaWorkflowError: The workflow cannot be validated or planned.
    """
    manifest = _read_manifest(manifest_path)
    workflow = (manifest_path.parent / manifest.workflow).resolve()
    workflow_digest = hashlib.sha256(workflow.read_bytes()).hexdigest()
    ids = [entry.id for entry in manifest.entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate batch entry IDs")
    runs = [
        _run_plan(manifest, workflow, entry, episode)
        for entry in manifest.entries
        for episode in _episodes(entry)
    ]
    if len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("expanded batch run IDs collide; rename the entries")
    _check_output_isolation(runs)
    if hashlib.sha256(workflow.read_bytes()).hexdigest() != workflow_digest:
        raise ValueError("workflow changed during batch planning")
    return _plan_payload(manifest, manifest_path, workflow, runs)


def _plan_payload(manifest, manifest_path, workflow, runs):
    payload = {
        "schema": "npa.workflow.batch-plan.v1",
        "batch_id": manifest.batch_id,
        "workflow": str(workflow),
        "workflow_sha256": hashlib.sha256(workflow.read_bytes()).hexdigest(),
        "project": manifest.project,
        "infra": manifest.infra,
        "config_path": str((manifest_path.parent / manifest.config_path).resolve())
        if manifest.config_path
        else "",
        "secret_env": manifest.secret_env,
        "runs": runs,
    }
    payload["source_sha256"] = source_fingerprint(find_npa_package_root())
    explicit_source = os.environ.get("NPA_SRC_S3_URI") or os.environ.get(
        "NPA_E2E_NPA_SRC_S3_URI", ""
    )
    payload["source_selection_sha256"] = hashlib.sha256(
        explicit_source.encode()
    ).hexdigest()
    payload["config_sha256"] = (
        hashlib.sha256(Path(payload["config_path"]).read_bytes()).hexdigest()
        if payload["config_path"]
        else ""
    )
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload
