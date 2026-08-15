"""A solution whose smoke hits a vendor gate must have an operator-env entry.

`OPERATOR_RUNTIME_ENVS_BY_SOLUTION` is keyed by the `solution_name` a workflow
spec declares. For solutions whose resource profile does not export the vendor's
acceptance variable itself, that map is the *only* path by which the acceptance
reaches the pod — SkyPilot's redacted secret channel.

Keying the map by solution fixed a real leak (one shared tuple forwarded
`HF_TOKEN` into every BYOF image), but it introduced a failure mode with no unit
coverage: a spec whose name is missing from the map silently forwards nothing,
and the run dies inside the container at the vendor gate. That is exactly what
happened to `wan2.2-multigpu`, whose spec declares a different `solution_name`
than `wan2.2` — invisible to every existing test, and only reachable on a
4×B200 live run.

So the rule is checked against the specs rather than trusted: if a spec's smoke
invokes a runtime that refuses without an acceptance variable, that spec's
solution name must be in the map.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_container_verify.py"

#: Bootstraps that refuse with EX_CONFIG until the operator has accepted a
#: vendor's terms. A smoke that calls one of these needs its acceptance
#: forwarded, or it fails inside the pod.
GATED_RUNTIMES = ("wan-runtime", "ltx-runtime")


def _runner_module():
    spec = importlib.util.spec_from_file_location("byof_container_verify", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _specs_invoking_a_gated_runtime() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(SPEC_DIR.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        config = (payload or {}).get("config")
        if not isinstance(config, dict):
            continue
        name = str(config.get("solution_name") or "").strip()
        smoke = str(config.get("smoke_command") or "")
        if not name or not smoke:
            continue
        if any(re.search(rf"\b{runtime}\b", smoke) for runtime in GATED_RUNTIMES):
            found[name] = path
    return found


def test_the_scan_finds_the_specs_it_is_supposed_to_guard() -> None:
    """Without this the assertion below would pass on an empty set."""

    gated = _specs_invoking_a_gated_runtime()

    assert gated, "no spec appears to invoke a gated runtime; has the shape changed?"
    assert "wan2.2-multigpu" in gated, (
        "the distributed Wan spec is the case this guardrail exists for"
    )


@pytest.mark.parametrize(
    "solution_name", sorted(_specs_invoking_a_gated_runtime()), ids=lambda name: name
)
def test_every_gated_solution_can_receive_its_acceptance(solution_name: str) -> None:
    module = _runner_module()
    mapping = module.OPERATOR_RUNTIME_ENVS_BY_SOLUTION

    assert solution_name in mapping, (
        f"{solution_name} runs a gated runtime but has no entry in "
        "OPERATOR_RUNTIME_ENVS_BY_SOLUTION, so the operator's acceptance is "
        "never forwarded and the run dies at the vendor gate inside the pod"
    )
    assert mapping[solution_name], f"{solution_name} maps to an empty tuple"


def test_no_entry_names_a_solution_no_spec_declares() -> None:
    """The other direction, so the map does not accumulate dead keys."""

    module = _runner_module()
    declared = {
        str(
            (yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            .get("config", {})
            .get("solution_name")
            or ""
        ).strip()
        for path in SPEC_DIR.glob("*.yaml")
    }

    stale = sorted(set(module.OPERATOR_RUNTIME_ENVS_BY_SOLUTION) - declared)

    assert stale == [], f"{stale} are keyed but no shipped spec declares them"
