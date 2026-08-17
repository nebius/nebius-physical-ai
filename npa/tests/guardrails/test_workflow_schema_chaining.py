"""A state that depends on another must read a schema that one produces.

An `npa.workflow` spec declares, per state, which artifacts it consumes and
which it emits. Nothing checked that those line up: `validate-spec` accepts any
schema string, so a state could declare an input schema no upstream state ever
writes and the spec would still validate, plan, and submit.

That is not a typo-catcher. It found a real defect in `byof-ltx2.yaml`, where the
FiftyOne curation state declared `npa.sim2real.augmented_frames.v1` over the run
prefix while the only upstream producer emitted `video/mp4`. The graph looked
connected and described a data flow that did not exist.

States with no `needs` are exempt: a source state legitimately reads data from
outside the workflow, which is why the rule is scoped to states that declare an
upstream dependency rather than applied to every input.
"""

from __future__ import annotations

from npa.orchestration.npa_workflow import load_spec
from npa.orchestration.npa_workflow.blueprints import iter_npa_workflow_specs

#: Violations that already existed when this lint was written: 16 state/schema
#: pairs across ten specs. Recorded so the
#: gate can be turned on now rather than after someone finds time to fix ten
#: other people's specs. Shrink it by making the producer declare the schema its
#: consumer names, or by correcting the consumer; never grow it to admit a new
#: one — that is what the assertion below is for.
KNOWN_GAPS: frozenset[tuple[str, str, str]] = frozenset(
    {
        (
            "adversarial-scenario-hardening.yaml",
            "publish",
            "npa.rl.policy_checkpoint.v1",
        ),
        (
            "adversarial-scenario-hardening.yaml",
            "publish",
            "npa.scenario_gen.hardening_decision.v1",
        ),
        (
            "hardening-with-insights.yaml",
            "aggregate",
            "npa.scenario_gen.hardening_decision.v1",
        ),
        ("hardening-with-insights.yaml", "publish", "npa.rl.policy_checkpoint.v1"),
        (
            "hardening-with-insights.yaml",
            "publish",
            "npa.scenario_gen.hardening_decision.v1",
        ),
        ("isaac-franka-capture-reason.yaml", "reason", "npa.workbench.scene_frames.v1"),
        ("isaac-lab-rl-sweep.yaml", "select-best", "npa.rl_sweep.variant_metrics.v1"),
        (
            "physical-ai-data-factory.yaml",
            "annotate-original",
            "npa.data_factory.frames.v1",
        ),
        ("physical-ai-data-factory.yaml", "augment", "npa.data_factory.frames.v1"),
        ("physical-ai-data-factory.yaml", "cosmos-curate", "npa.cosmos2.transfer.v1"),
        ("physical-ai-data-factory.yaml", "curate", "npa.cosmos2.transfer.v1"),
        ("sim2real-envgen-shards.yaml", "actions", "npa.sim2real.env_catalog.v1"),
        ("token-factory-gate-loop.yaml", "score-batch", "npa.workbench.rollout_set.v1"),
        (
            "token-factory-parallel-fanout.yaml",
            "aggregate",
            "npa.token_factory.captions.v1",
        ),
        (
            "tokenfactory-cosmos-gate.yaml",
            "augment-scene",
            "npa.token_factory.scene.v1",
        ),
        (
            "tokenfactory-rollout-judge.yaml",
            "judge-rollouts",
            "npa.workbench.rollout_set.v1",
        ),
    }
)


def _transitive_predecessors(spec, name: str, seen: set[str] | None = None) -> set[str]:
    seen = seen if seen is not None else set()
    for dependency in spec.states[name].needs or []:
        if dependency in seen or dependency not in spec.states:
            continue
        seen.add(dependency)
        _transitive_predecessors(spec, dependency, seen)
    return seen


def _unproduced_input_schemas() -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for path in iter_npa_workflow_specs():
        try:
            spec = load_spec(path)
        except Exception:  # noqa: BLE001 - other suites own spec-loading failures
            continue
        for name, state in spec.states.items():
            if not (state.needs or []):
                continue
            produced = {
                output.schema
                for upstream in _transitive_predecessors(spec, name)
                for output in (spec.states[upstream].outputs or [])
            }
            for item in state.inputs or []:
                if item.schema and item.schema not in produced:
                    found.add((path.name, name, item.schema))
    return found


def test_no_new_state_reads_a_schema_nothing_upstream_produces() -> None:
    new = sorted(_unproduced_input_schemas() - KNOWN_GAPS)

    assert new == [], (
        "these states declare an input schema no upstream state produces, so the "
        "spec describes a data flow that does not exist:\n  "
        + "\n  ".join(f"{spec}: {state} reads {schema}" for spec, state, schema in new)
    )


def test_the_baseline_is_the_size_its_comment_claims() -> None:
    """A count in prose drifts; this keeps the comment honest as the list shrinks."""

    assert len(KNOWN_GAPS) == 16
    assert len({row[0] for row in KNOWN_GAPS}) == 10


def test_the_known_gap_list_does_not_go_stale() -> None:
    """A fixed spec must leave the list, or the list stops meaning anything."""

    fixed = sorted(KNOWN_GAPS - _unproduced_input_schemas())

    assert fixed == [], (
        f"{fixed} no longer violate the rule; remove them from KNOWN_GAPS so the "
        "list keeps naming only real, outstanding gaps"
    )


def test_the_ltx2_spec_is_clean() -> None:
    """The spec this lint was written for must not be in the baseline."""

    assert not [row for row in KNOWN_GAPS if row[0] == "byof-ltx2.yaml"]
    assert not [
        row for row in _unproduced_input_schemas() if row[0] == "byof-ltx2.yaml"
    ]
