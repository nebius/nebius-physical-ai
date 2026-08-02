"""Guardrail: a spec's declared artifact must be the file its tool really writes.

Three separate live runs were needed to learn this lesson three times:

* `sonic eval` was handed ``--output json`` (a format word to a path option), so the
  result landed in a relative ``json/`` directory inside the pod (EVIDENCE §R5);
* `vlm-eval run` writes ``<prefix>/vlm_eval_stub.json`` while four specs declared
  ``<prefix>/report.json`` (run ``npa-wf-cpu-vlm-eval-token-factory-736df0b1``);
* `mjlab eval` writes ``<prefix>/mjlab_eval.json`` while two specs declared
  ``<prefix>/report.json``.

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

#: toolRef prefix -> (argv flag naming the output prefix, dotted `result_uri_for`).
RESULT_URI_TOOLS: dict[str, tuple[str, str]] = {
    "workbench.vlm_eval.run": ("--output-path", "npa.workbench.vlm_eval:result_uri_for"),
    "workbench.vlm_eval.loop": (
        "--output-path",
        "npa.workbench.vlm_eval:loop_report_uri_for",
    ),
    "workbench.vlm_eval.benchmark": (
        "--output",
        "npa.workbench.vlm_eval:benchmark_result_uri_for",
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


def _resolve(dotted: str):
    from importlib import import_module

    module_name, _, attr = dotted.partition(":")
    return getattr(import_module(module_name), attr)


def _cases() -> list[tuple[str, str, str, str]]:
    """Return (spec name, state, toolRef, declared uri) for every checkable stage."""

    out: list[tuple[str, str, str, str]] = []
    for path in iter_npa_workflow_specs():
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
                    out.append((path.name, step.state, step.tool_ref, artifact["uri"]))
    return out


CASES = _cases()


def test_there_are_result_uri_stages_to_check() -> None:
    assert len(CASES) >= 5, f"expected several checkable stages, found {len(CASES)}"


@pytest.mark.parametrize(
    ("spec_name", "state", "tool_ref", "declared"),
    CASES,
    ids=[f"{name}:{state}" for name, state, _, _ in CASES],
)
def test_declared_output_is_where_the_tool_writes(
    spec_name: str, state: str, tool_ref: str, declared: str
) -> None:
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


def _spec_path(name: str) -> Path:
    for path in iter_npa_workflow_specs():
        if path.name == name:
            return path
    raise AssertionError(f"spec not found: {name}")
