"""Daily E2E image-coverage accounting for ``npa.workflow`` specs.

The daily dev-VM test workflow (``scripts/dev-vm-daily-tests.sh``) must, every
day, exercise at least one *comprehensive* (>= 4-step) workflow E2E and keep
every workflow-reachable container image covered by such a workflow. This
module is the single source of truth for:

- how many executable steps each spec has (``spec_step_summary``),
- which specs qualify as comprehensive (``comprehensive_specs``),
- which images a comprehensive workflow references (via the renderer's
  authoritative ``tool_image_key`` mapping), and
- a regression guard (``assert_coverage``) that fails when a currently covered
  image drops out of every comprehensive workflow.

It is import-only (no side effects) so it is unit-testable in hosted CI and
reusable by the ``daily_workflow_e2e.py`` operator script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from npa.deploy.images import CONTAINER_IMAGE_NAMES
from npa.orchestration.npa_workflow import load_spec
from npa.orchestration.npa_workflow.blueprints import iter_npa_workflow_specs
from npa.orchestration.npa_workflow.skypilot_render import (
    TOOL_REF_IMAGE_TOOL,
    tool_image_key,
)

#: The minimum number of executable steps for a workflow to count as a
#: "comprehensive" daily E2E. The operator asked for "at least one 4-step
#: minimum workflow E2E" every day.
MIN_COMPREHENSIVE_STEPS = 4

#: Image tools a workflow *can* reference through a ``toolRef`` (the values of
#: the renderer's toolRef -> image map). This is the universe the >= 4-step
#: coverage requirement applies to.
WORKFLOW_IMAGE_TOOLS: frozenset[str] = frozenset(TOOL_REF_IMAGE_TOOL.values())

#: Workflow-reachable images that are NOT yet exercised by a >= 4-step spec, or
#: are not referenced by any spec toolRef today. Tracked explicitly so the gate
#: stays green while making the gap visible; shrink this set by extending or
#: authoring comprehensive workflows, never grow it to hide a regression.
#:
#:   sonic / retargeting : only appear in the 3-step SONIC locomotion chain.
#:   cosmos3-reason      : single-step reason spec only.
#:   alpamayo2-super     : dedicated single-step inference spec; covered by its
#:                         own B200 and RTX PRO 6000 workflow validation.
#:   cosmos3-ray-serve   : one-step CPU submission client for a separately
#:                         deployed persistent GPU service; its exact image has
#:                         dedicated model-backed B200/RTX validation.
#:   lerobot / genesis : component/tool images with no comprehensive workflow
#:                       toolRef chain yet (covered by their own tool + serverless
#:                       E2Es and by the daily registry-reachability check).
#:
#: ``groot`` left this set with the GR00T 1.7 multi-GPU training workflow.
#: ``lerobot`` briefly left it too, when byof-ltx2.yaml ended in a policy-training
#: state — but that state trained on a hub dataset and consumed no LTX output, so
#: it was removed rather than left standing as coverage it did not provide.
EXEMPT_IMAGE_TOOLS: frozenset[str] = frozenset(
    {
        "sonic",
        "retargeting",
        "cosmos3-reason",
        "alpamayo2-super",
        "cosmos3-ray-serve",
        "genesis",
    }
)


@dataclass(frozen=True)
class SpecSummary:
    """Executable-step and image accounting for one spec."""

    name: str
    path: Path
    total_states: int
    exec_steps: int
    image_tools: frozenset[str]

    @property
    def is_comprehensive(self) -> bool:
        return self.exec_steps >= MIN_COMPREHENSIVE_STEPS


def _summarize(path: Path) -> SpecSummary:
    spec = load_spec(path)
    images: set[str] = set()
    exec_steps = 0
    for state in spec.states.values():
        if state.tool_ref:
            exec_steps += 1
            key = tool_image_key(state.tool_ref)
            if key:
                images.add(key)
        elif state.run is not None and not state.run.is_empty():
            exec_steps += 1
    return SpecSummary(
        name=path.name,
        path=path,
        total_states=len(spec.states),
        exec_steps=exec_steps,
        image_tools=frozenset(images),
    )


def spec_step_summary() -> list[SpecSummary]:
    """Summarize every discoverable npa.workflow spec, sorted by name."""

    return sorted((_summarize(p) for p in iter_npa_workflow_specs()), key=lambda s: s.name)


def comprehensive_specs(summaries: list[SpecSummary] | None = None) -> list[SpecSummary]:
    """Specs with at least ``MIN_COMPREHENSIVE_STEPS`` executable steps."""

    summaries = summaries if summaries is not None else spec_step_summary()
    return [s for s in summaries if s.is_comprehensive]


@dataclass(frozen=True)
class CoverageReport:
    """Which workflow-reachable images a >= 4-step workflow covers today."""

    covered: frozenset[str]
    reachable: frozenset[str]
    required: frozenset[str]
    missing: frozenset[str]
    covering_workflows: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing


def image_coverage(summaries: list[SpecSummary] | None = None) -> CoverageReport:
    """Compute >= 4-step workflow coverage over workflow-reachable images."""

    summaries = summaries if summaries is not None else spec_step_summary()
    covered: set[str] = set()
    reachable: set[str] = set()
    covering: dict[str, list[str]] = {}
    for summary in summaries:
        reachable |= summary.image_tools
        if summary.is_comprehensive:
            covered |= summary.image_tools
            for image in sorted(summary.image_tools):
                covering.setdefault(image, []).append(summary.name)

    required = frozenset(WORKFLOW_IMAGE_TOOLS - EXEMPT_IMAGE_TOOLS)
    missing = frozenset(required - covered)
    return CoverageReport(
        covered=frozenset(covered),
        reachable=frozenset(reachable),
        required=required,
        missing=missing,
        covering_workflows=covering,
    )


def assert_coverage(report: CoverageReport | None = None) -> None:
    """Raise ``AssertionError`` if a required image lost >= 4-step coverage.

    A required image dropping out of every comprehensive workflow is a
    regression: extend/author a workflow to re-cover it (or, only with an
    explicit rationale, move it to ``EXEMPT_IMAGE_TOOLS``).
    """

    report = report if report is not None else image_coverage()
    if report.missing:
        raise AssertionError(
            "Required workflow images are not covered by any >= "
            f"{MIN_COMPREHENSIVE_STEPS}-step workflow E2E: "
            f"{sorted(report.missing)}. Extend a comprehensive workflow to "
            "cover them, or justify moving them to EXEMPT_IMAGE_TOOLS."
        )


def minimal_cover(summaries: list[SpecSummary] | None = None) -> list[SpecSummary]:
    """A small set of comprehensive specs whose union covers every covered image.

    Greedy set cover over the currently covered images, so the daily run can
    plan a compact set of >= 4-step workflows that together touch every
    workflow-reachable (non-exempt) image every day. Deterministic: ties break
    on spec name.
    """

    summaries = summaries if summaries is not None else spec_step_summary()
    report = image_coverage(summaries)
    remaining = set(report.covered)
    pool = sorted(
        comprehensive_specs(summaries),
        key=lambda s: (-len(s.image_tools & report.covered), s.name),
    )
    chosen: list[SpecSummary] = []
    for summary in pool:
        if not remaining:
            break
        useful = summary.image_tools & remaining
        if useful:
            chosen.append(summary)
            remaining -= useful
    return sorted(chosen, key=lambda s: s.name)


def rotating_spec(day_index: int, summaries: list[SpecSummary] | None = None) -> SpecSummary | None:
    """Pick one comprehensive spec for ``day_index`` (round-robins over days)."""

    pool = comprehensive_specs(summaries)
    if not pool:
        return None
    pool = sorted(pool, key=lambda s: s.name)
    return pool[day_index % len(pool)]


def daily_plan_set(day_index: int, summaries: list[SpecSummary] | None = None) -> list[SpecSummary]:
    """The >= 4-step workflows to plan today: the image-covering set + a rotating extra."""

    summaries = summaries if summaries is not None else spec_step_summary()
    chosen = {s.name: s for s in minimal_cover(summaries)}
    extra = rotating_spec(day_index, summaries)
    if extra is not None:
        chosen.setdefault(extra.name, extra)
    return sorted(chosen.values(), key=lambda s: s.name)


def all_container_image_names() -> frozenset[str]:
    """Every workbench image key (superset of workflow-reachable images)."""

    return frozenset(CONTAINER_IMAGE_NAMES)
