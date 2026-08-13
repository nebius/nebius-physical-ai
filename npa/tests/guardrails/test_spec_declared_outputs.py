"""Guardrail: a spec's declared artifact must be the file its tool really writes.

Three separate live runs were needed to learn this lesson three times:

* `sonic eval` was handed ``--output json`` (a format word to a path option), so the
  result landed in a relative ``json/`` directory inside the pod (EVIDENCE §R5);
* `vlm-eval run` writes ``<prefix>/vlm_eval_stub.json`` while four specs declared
  ``<prefix>/report.json`` (run ``npa-wf-cpu-vlm-eval-token-factory-736df0b1``);
* `mjlab eval` writes ``<prefix>/mjlab_eval.json`` while two specs declared
  ``<prefix>/report.json``.
* Cosmos Transfer live job 339 reported SUCCEEDED for historical stages,
  but those stages promised ``manifest.json`` while the tool wrote ``index.json``
  with a different schema. The extracted implementation now publishes a canonical
  transfer manifest and retains the reference frame index as a separate contract.

In every case the stage reported SUCCEEDED. `outputs:` is a contract with whoever reads
the run prefix next, so it has to be checked offline rather than discovered live.

Each tool below exposes a ``result_uri_for(prefix)`` helper — the single source of truth
for where its result lands — so this test resolves the spec's argv, asks the tool where
it would write, and compares.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.blueprints import iter_npa_workflow_specs
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec


ROOT = Path(__file__).resolve().parents[3]
DIAGRAM_EXAMPLE_SPECS = tuple(
    sorted(
        (ROOT / "skills" / "workflows" / "diagram-to-npa-workflow" / "examples").glob(
            "*.yaml"
        )
    )
)


def _checked_specs() -> tuple[Path, ...]:
    """Shipped specs plus skill examples that users are instructed to copy."""

    return (*iter_npa_workflow_specs(), *DIAGRAM_EXAMPLE_SPECS)


#: toolRef prefix -> (argv flag naming the output prefix, dotted `result_uri_for`).
RESULT_URI_TOOLS: dict[str, tuple[str, str]] = {
    "workbench.vlm_eval.run": (
        "--output-path",
        "npa.workbench.vlm_eval:result_uri_for",
    ),
    "workbench.vlm_eval.loop": (
        "--output-path",
        "npa.workbench.vlm_eval:loop_report_uri_for",
    ),
    "workbench.vlm_eval.benchmark": (
        "--output",
        "npa.workbench.vlm_eval:benchmark_result_uri_for",
    ),
    # Do not remove these entries as redundant now that the producer is fixed.
    # Job 339 is why declared Cosmos outputs must stay bound to the helper the
    # tool actually implements, even when every current spec happens to agree.
    "workbench.cosmos2.transfer": (
        "--output-uri",
        "npa.workbench.cosmos.transfer:transfer_manifest_uri_for",
    ),
    "workbench.cosmos2.transfer_execute": (
        "--output-uri",
        "npa.workbench.cosmos.transfer:transfer_manifest_uri_for",
    ),
    "workbench.cosmos2.transfer_conditioned_execute": (
        "--output-uri",
        "npa.workbench.cosmos.transfer:transfer_manifest_uri_for",
    ),
    "workbench.mjlab.eval": ("--output-path", "npa.workbench.mjlab:result_uri_for"),
    "workbench.sonic.eval": ("--output", "npa.workbench.sonic.eval:result_uri_for"),
    "workbench.token_factory.caption": (
        "--output-path",
        "npa.workbench.token_factory:caption_result_uri_for",
    ),
    "workbench.token_factory.generate": (
        "--output-path",
        "npa.workbench.token_factory:generate_result_uri_for",
    ),
    "workbench.token_factory.reason": (
        "--output-path",
        "npa.workbench.token_factory:reason_result_uri_for",
    ),
    # All three eval stages publish the same canonical metrics object under their own
    # output prefix. Before `--write-canonical-metrics` existed, nothing wrote it and the
    # BDD100K spec declared it anyway.
    **{
        f"workbench.detection_training.eval_{view}": (
            "--output-uri",
            "npa.workbench.detection_training.artifacts:eval_result_uri_for",
        )
        for view in ("rider", "nighttime", "distant")
    },
    # Retargeting's real artifact is a directory of motions; the JSON a downstream
    # stage reads is the metadata sidecar, so that is what `outputs:` must name.
    "workbench.retargeting.run": (
        "--output-path",
        "npa.workbench.retargeting:metadata_uri_for",
    ),
}

# Schemas whose producer exposes the canonical value next to its URI helper.
# Keeping this separate lets the general URI guard cover older tools while the
# Cosmos transfer contract fails if either its filename or schema drifts.
RESULT_SCHEMA_TOOLS: dict[str, str] = {
    "workbench.cosmos2.transfer": (
        "npa.workbench.cosmos.transfer:TRANSFER_MANIFEST_SCHEMA"
    ),
    "workbench.cosmos2.transfer_execute": (
        "npa.workbench.cosmos.transfer:TRANSFER_MANIFEST_SCHEMA"
    ),
    "workbench.cosmos2.transfer_conditioned_execute": (
        "npa.workbench.cosmos.transfer:TRANSFER_MANIFEST_SCHEMA"
    ),
}

TRANSFER_TOOLREFS = frozenset(RESULT_SCHEMA_TOOLS)
GENERIC_TRANSFER_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()
JOB_339_COSMOS_OUTPUT_LOCATIONS = frozenset(
    {
        ("cosmos-synth-fanout-curation.yaml", "synth-shard-a"),
        ("cosmos-synth-fanout-curation.yaml", "synth-shard-b"),
        ("cosmos2-transfer.yaml", "transfer"),
        ("tokenfactory-cosmos-gate.yaml", "augment-scene"),
    }
)


def test_cosmos_transfer_toolrefs_match_declared_input_contracts() -> None:
    """Shipped GPU workflows use a fail-closed execute toolRef with a trigger."""

    offenders: list[str] = []
    for path in _checked_specs():
        spec = load_spec(path)
        for name, state in spec.states.items():
            if state.tool_ref == "workbench.cosmos2.transfer":
                location = (path.name, name)
                if location not in GENERIC_TRANSFER_ALLOWLIST:
                    offenders.append(
                        f"{path.name}:{name}: generic transfer is not a fail-closed "
                        "real-GPU execution path"
                    )
            if state.tool_ref in {
                "workbench.cosmos2.transfer_execute",
                "workbench.cosmos2.transfer_conditioned_execute",
            }:
                input_uris = {artifact.uri for artifact in state.inputs}
                if "{{config.trigger_uri}}" not in input_uris:
                    offenders.append(
                        f"{path.name}:{name}: execute transfer lacks trigger input"
                    )

    assert not offenders, (
        "Cosmos transfer toolRefs must match their declared input contracts: "
        f"{offenders}"
    )

    observed_generic = {
        (path.name, name)
        for path in _checked_specs()
        for name, state in load_spec(path).states.items()
        if state.tool_ref == "workbench.cosmos2.transfer"
    }
    assert observed_generic == GENERIC_TRANSFER_ALLOWLIST


def _resolve(dotted: str):
    from importlib import import_module

    module_name, _, attr = dotted.partition(":")
    return getattr(import_module(module_name), attr)


def _cases() -> list[tuple[str, str, str, str, str]]:
    """Return spec/state/toolRef plus declared URI/schema for checkable stages."""

    out: list[tuple[str, str, str, str, str]] = []
    for path in _checked_specs():
        spec = load_spec(path)
        assume = (
            "promote_checkpoint"
            if any(state.transitions for state in spec.states.values())
            else None
        )
        plan = build_plan(spec, run_id="declared-outputs", assume_decision=assume)
        for step in plan.steps:
            if step.tool_ref not in RESULT_URI_TOOLS:
                continue
            for artifact in step.outputs:
                if artifact["uri"]:
                    out.append(
                        (
                            path.name,
                            step.state,
                            step.tool_ref,
                            artifact["uri"],
                            artifact["schema"],
                        )
                    )
    return out


CASES = _cases()


def test_live_job_339_cosmos_output_regressions_remain_guarded() -> None:
    """Keep every still-shipped historical false-success location guarded."""

    guarded = {
        (spec_name, state)
        for spec_name, state, tool_ref, _, _ in CASES
        if tool_ref in TRANSFER_TOOLREFS
    }

    assert len(JOB_339_COSMOS_OUTPUT_LOCATIONS) == 4
    assert len({spec for spec, _ in JOB_339_COSMOS_OUTPUT_LOCATIONS}) == 3
    assert JOB_339_COSMOS_OUTPUT_LOCATIONS <= guarded


@pytest.mark.parametrize(
    "path",
    tuple(
        path
        for path in _checked_specs()
        if "augment_uri" in load_spec(path).config
        and any(
            state.tool_ref in TRANSFER_TOOLREFS
            for state in load_spec(path).states.values()
        )
    ),
    ids=lambda path: path.name,
)
def test_cosmos_declared_manifest_does_not_depend_on_trailing_prefix_slash(
    path: Path,
) -> None:
    """Producer and declaration stay aligned if ``augment_uri`` loses ``/``."""

    spec = load_spec(path)
    assert "augment_manifest_uri" in spec.config
    spec.config["augment_uri"] = str(spec.config["augment_uri"]).rstrip("/")
    assume = (
        "promote_checkpoint"
        if any(state.transitions for state in spec.states.values())
        else None
    )
    plan = build_plan(spec, run_id="no-trailing-slash", assume_decision=assume)
    for step in plan.steps:
        if step.tool_ref not in TRANSFER_TOOLREFS:
            continue
        output_prefix = step.argv[step.argv.index("--output-uri") + 1]
        manifests = [
            output
            for output in step.outputs
            if output["schema"] == "npa.cosmos2.transfer.v1"
        ]
        assert len(manifests) == 1
        assert manifests[0]["uri"] == _resolve(
            "npa.workbench.cosmos.transfer:transfer_manifest_uri_for"
        )(output_prefix)


def test_there_are_result_uri_stages_to_check() -> None:
    assert len(CASES) >= 5, f"expected several checkable stages, found {len(CASES)}"


@pytest.mark.parametrize(
    ("spec_name", "state", "tool_ref", "declared", "declared_schema"),
    CASES,
    ids=[f"{name}:{state}" for name, state, _, _, _ in CASES],
)
def test_declared_output_is_where_the_tool_writes(
    spec_name: str,
    state: str,
    tool_ref: str,
    declared: str,
    declared_schema: str,
) -> None:
    del declared_schema
    flag, dotted = RESULT_URI_TOOLS[tool_ref]
    spec = load_spec(_spec_path(spec_name))
    assume = (
        "promote_checkpoint"
        if any(candidate.transitions for candidate in spec.states.values())
        else None
    )
    plan = build_plan(spec, run_id="declared-outputs", assume_decision=assume)
    step = next(s for s in plan.steps if s.state == state)
    assert flag in step.argv, f"{spec_name}:{state} argv is missing {flag}"
    prefix = step.argv[step.argv.index(flag) + 1]

    expected = _resolve(dotted)(prefix)

    assert declared == expected, (
        f"{spec_name}:{state} declares {declared!r} but `{tool_ref}` writes "
        f"{expected!r}. The stage will SUCCEED and the declared artifact will not exist."
    )


@pytest.mark.parametrize(
    ("spec_name", "state", "tool_ref", "declared", "declared_schema"),
    [case for case in CASES if case[2] in RESULT_SCHEMA_TOOLS],
    ids=[
        f"{name}:{state}"
        for name, state, tool_ref, _, _ in CASES
        if tool_ref in RESULT_SCHEMA_TOOLS
    ],
)
def test_declared_output_schema_matches_the_tool_contract(
    spec_name: str,
    state: str,
    tool_ref: str,
    declared: str,
    declared_schema: str,
) -> None:
    del declared
    expected = _resolve(RESULT_SCHEMA_TOOLS[tool_ref])

    assert declared_schema == expected, (
        f"{spec_name}:{state} declares schema {declared_schema!r}, but `{tool_ref}` "
        f"publishes {expected!r}."
    )


COSMOS_TO_ENVGEN_SPECS = tuple(
    path
    for path in _checked_specs()
    if (
        any(
            state.tool_ref in TRANSFER_TOOLREFS
            for state in load_spec(path).states.values()
        )
        and any(
            state.tool_ref == "workbench.sim2real_envgen.raw_shard"
            for state in load_spec(path).states.values()
        )
    )
)


def test_there_are_cosmos_to_envgen_contracts_to_check() -> None:
    assert len(COSMOS_TO_ENVGEN_SPECS) >= 2


@pytest.mark.parametrize(
    "path", COSMOS_TO_ENVGEN_SPECS, ids=[path.name for path in COSMOS_TO_ENVGEN_SPECS]
)
def test_cosmos_transfer_manifest_flows_into_envgen(path: Path) -> None:
    """A real transfer is useless to envgen unless its exact frame list is consumed."""

    spec = load_spec(path)
    assume = (
        "promote_checkpoint"
        if any(state.transitions for state in spec.states.values())
        else None
    )
    plan = build_plan(spec, run_id="manifest-flow", assume_decision=assume)
    transfer_steps = [step for step in plan.steps if step.tool_ref in TRANSFER_TOOLREFS]
    envgen_steps = [
        step
        for step in plan.steps
        if step.tool_ref == "workbench.sim2real_envgen.raw_shard"
    ]

    assert len(transfer_steps) == 1, f"{path.name} must have one unambiguous transfer"
    assert len(envgen_steps) == 1, f"{path.name} must have one unambiguous envgen"
    transfer = transfer_steps[0]
    envgen = envgen_steps[0]
    manifests = [
        artifact
        for artifact in transfer.outputs
        if artifact["schema"] == "npa.cosmos2.transfer.v1"
    ]
    assert len(manifests) == 1, f"{path.name} must declare one transfer manifest"
    manifest_uri = manifests[0]["uri"]

    assert "--augmented-frames-uri" in envgen.argv
    consumed_uri = envgen.argv[envgen.argv.index("--augmented-frames-uri") + 1]
    assert consumed_uri == manifest_uri
    assert {
        "uri": manifest_uri,
        "schema": "npa.cosmos2.transfer.v1",
    } in envgen.inputs


def _spec_path(name: str) -> Path:
    for path in _checked_specs():
        if path.name == name:
            return path
    raise AssertionError(f"spec not found: {name}")
