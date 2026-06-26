"""Workflow YAML generation and validation for the NPA agent UI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

API_VERSION = "npa.workflow/v0.0.1"


def generate_sim2real_two_step_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "sim2real-two-step",
) -> str:
    """Return a minimal 2-step Sim2Real npa.workflow spec (augment → envgen)."""
    return f"""apiVersion: {API_VERSION}
kind: Workflow

metadata:
  name: {name}
  description: >
    Two-step Sim2Real pipeline: Cosmos Transfer augment, then raw env generation.

config:
  bucket: {bucket}
  prefix: "sim2real/{{{{run.id}}}}"
  env_count: "1000"

  trigger_uri: "s3://{{{{config.bucket}}}}/sim2real-triggers/{{{{run.id}}}}/lerobot-pusht/"
  augment_uri: "s3://{{{{config.bucket}}}}/{{{{config.prefix}}}}/augment/"
  raw_envs_uri: "s3://{{{{config.bucket}}}}/{{{{config.prefix}}}}/envs/raw/"

resources:
  gpu:
    cloud: kubernetes
    accelerators: RTXPRO6000:1

initial: augment

states:
  augment:
    description: Cosmos Transfer augment of LeRobot trigger data.
    toolRef: workbench.cosmos2.transfer
    resources: gpu
    outputs:
      - uri: "{{{{config.augment_uri}}}}manifest.json"
        schema: npa.sim2real.augment.v1
    next: envgen

  envgen:
    description: Generate raw env shard catalog on object storage.
    needs: [augment]
    toolRef: workbench.sim2real_envgen.raw_shard
    resources: gpu
    outputs:
      - uri: "{{{{config.raw_envs_uri}}}}manifest.json"
        schema: npa.sim2real.split_manifest.v1
    terminal: true
"""


def validate_workflow_yaml_text(
    yaml_text: str,
    *,
    tool_refs: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Validate workflow YAML; prefers npa.orchestration when available."""
    text = str(yaml_text or "").strip()
    if not text:
        return {"ok": False, "status": "invalid", "error": "empty workflow YAML"}
    try:
        return _validate_with_npa(text)
    except ImportError:
        return _validate_lightweight(text, tool_refs=tool_refs)


def plan_workflow_yaml_text(
    yaml_text: str,
    *,
    run_id: str = "",
    assume_decision: str = "",
    tool_refs: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Expand workflow YAML into a dry-run plan."""
    text = str(yaml_text or "").strip()
    if not text:
        return {"ok": False, "error": "empty workflow YAML"}
    try:
        return _plan_with_npa(text, run_id=run_id, assume_decision=assume_decision)
    except ImportError:
        return _plan_lightweight(text, run_id=run_id, tool_refs=tool_refs)


def format_workflow_chat_reply(yaml_text: str, validation: dict[str, Any]) -> str:
    """Markdown reply for chat when a workflow YAML is generated."""
    name = str(validation.get("name") or "unnamed")
    status = str(validation.get("status") or ("valid" if validation.get("ok") else "invalid"))
    states = validation.get("states") or []
    state_label = ", ".join(str(s) for s in states) if isinstance(states, list) else str(states)
    lines = [
        "**Generated npa.workflow/v0.0.1 spec** (2-step Sim2Real pipeline):",
        f"- **name**: `{name}`",
        f"- **validation**: `{status}`",
        f"- **states**: `{state_label or 'n/a'}`",
        "",
        "Edit in the **Workflow YAML** panel, then **Validate**, **Plan**, or **Submit**.",
        "",
        "```yaml",
        yaml_text.rstrip(),
        "```",
    ]
    if not validation.get("ok"):
        err = str(validation.get("error") or "validation failed")
        lines.insert(4, f"- **error**: `{err}`")
    return "\n".join(lines)


def _validate_with_npa(yaml_text: str) -> dict[str, Any]:
    from npa.orchestration.npa_workflow import NpaWorkflowError, load_spec

    path = _write_temp_yaml(yaml_text)
    try:
        spec = load_spec(path)
    except NpaWorkflowError as exc:
        return {"ok": False, "status": "invalid", "error": str(exc)}
    return {
        "ok": True,
        "status": "valid",
        "apiVersion": spec.api_version,
        "name": spec.name,
        "states": sorted(spec.states),
        "initial": spec.initial,
    }


def _plan_with_npa(yaml_text: str, *, run_id: str, assume_decision: str) -> dict[str, Any]:
    from npa.orchestration.npa_workflow import NpaWorkflowError, build_plan, load_spec

    path = _write_temp_yaml(yaml_text)
    try:
        spec = load_spec(path)
        resolved_run_id = run_id or f"{spec.name}-plan"
        plan = build_plan(spec, run_id=resolved_run_id, assume_decision=assume_decision)
    except NpaWorkflowError as exc:
        return {"ok": False, "error": str(exc)}
    payload = plan.to_dict()
    payload["ok"] = True
    payload["run_id"] = resolved_run_id
    return payload


def _validate_lightweight(yaml_text: str, *, tool_refs: frozenset[str] | None) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        return {"ok": False, "status": "invalid", "error": f"invalid YAML: {exc}"}
    if not isinstance(data, dict):
        return {"ok": False, "status": "invalid", "error": "workflow spec must be a mapping"}

    api_version = str(data.get("apiVersion") or "")
    if api_version != API_VERSION:
        return {
            "ok": False,
            "status": "invalid",
            "error": f"unsupported apiVersion {api_version!r} (expected {API_VERSION})",
        }

    metadata = data.get("metadata") or {}
    name = str(metadata.get("name") or "unnamed") if isinstance(metadata, dict) else "unnamed"
    states_raw = data.get("states") or {}
    if not isinstance(states_raw, dict) or not states_raw:
        return {"ok": False, "status": "invalid", "error": "states must be a non-empty mapping"}

    initial = str(data.get("initial") or next(iter(states_raw)))
    if initial not in states_raw:
        return {"ok": False, "status": "invalid", "error": f"initial state {initial!r} not found"}

    catalog = tool_refs or frozenset()
    visited: set[str] = set()
    current = initial
    while current:
        if current in visited:
            return {"ok": False, "status": "invalid", "error": f"cycle detected at state {current!r}"}
        visited.add(current)
        entry = states_raw.get(current)
        if not isinstance(entry, dict):
            return {"ok": False, "status": "invalid", "error": f"state {current!r} must be a mapping"}
        tool_ref = str(entry.get("toolRef") or "").strip()
        if tool_ref and catalog and tool_ref not in catalog:
            return {"ok": False, "status": "invalid", "error": f"unknown toolRef {tool_ref!r}"}
        if entry.get("terminal"):
            break
        nxt = str(entry.get("next") or "").strip()
        if not nxt:
            break
        if nxt not in states_raw:
            return {"ok": False, "status": "invalid", "error": f"next state {nxt!r} not found"}
        current = nxt

    return {
        "ok": True,
        "status": "valid",
        "apiVersion": api_version,
        "name": name,
        "states": sorted(str(k) for k in states_raw),
        "initial": initial,
        "mode": "lightweight",
    }


def _plan_lightweight(yaml_text: str, *, run_id: str, tool_refs: frozenset[str] | None) -> dict[str, Any]:
    validation = _validate_lightweight(yaml_text, tool_refs=tool_refs)
    if not validation.get("ok"):
        return {"ok": False, "error": str(validation.get("error") or "validation failed")}

    import yaml

    data = yaml.safe_load(yaml_text) or {}
    states_raw = data.get("states") or {}
    metadata = data.get("metadata") or {}
    name = str(metadata.get("name") or "unnamed") if isinstance(metadata, dict) else "unnamed"
    initial = str(data.get("initial") or next(iter(states_raw)))
    resolved_run_id = run_id or f"{name}-plan"

    steps: list[dict[str, Any]] = []
    visited: set[str] = set()
    current = initial
    while current and current not in visited:
        visited.add(current)
        entry = states_raw[current]
        steps.append(
            {
                "state": current,
                "iteration": None,
                "tool_ref": str(entry.get("toolRef") or ""),
                "resources": str(entry.get("resources") or "default"),
            }
        )
        if entry.get("terminal"):
            break
        current = str(entry.get("next") or "").strip()

    return {
        "ok": True,
        "workflow": name,
        "api_version": API_VERSION,
        "initial": initial,
        "run_id": resolved_run_id,
        "steps": steps,
        "mode": "lightweight",
    }


def _write_temp_yaml(yaml_text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False)
    try:
        handle.write(yaml_text)
        handle.flush()
        return Path(handle.name)
    finally:
        handle.close()
