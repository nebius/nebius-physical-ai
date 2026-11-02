"""The multiway optimization was a step nobody could check had run.

`run_multiway` builds a full pairwise pose graph and calls `global_optimization`,
which is the entire difference between multiway registration and chaining pairwise
transforms. It mutates the graph in place: it moves node poses, and it prunes loop
closures whose confidence falls below `edge_prune_threshold`. The record serialized
the graph afterwards, so both effects were unrecoverable -- the pruned edges were
simply absent from the list, and nothing stated the correction.

Measured, in `docs/evidence/open3d/pose-graph-optimization.json`:

| case                                      | correction        | edges            |
| ----------------------------------------- | ----------------- | ---------------- |
| control: a node induced 0.5 off           | 0.5, moved back   | 1 of 1 survived  |
| upstream `DemoICPPointClouds`             | 4.7e-06           | 3 of 3 survived  |
| `run_multiway` on the smoke's fragments   | exactly 0.0       | 3 of 3 survived  |
| four overlapping views closing a loop     | 6.5e-07           | 5 of 6 survived  |

So on both sets of fragments the pipeline runs on, the optimizer corrects nothing
worth acting on -- correctly, because the pairwise estimates already agree. The
defect is not that it does nothing. It is that the capability was only ever exercised
by cases in which it cannot fail, and the record could not distinguish those from a
failure. The loop case is the one that shows the other half: an edge was pruned on
every repeat, and a reader of the serialized graph would just see five edges.

The control matters more than it looks. Without it, a measured zero is
indistinguishable from a probe that cannot read poses, which is what a zero on four
of four cases first looked like.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from npa.workbench.open3d.artifacts import Open3dError, validate_pose_graph

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "evidence"
    / "open3d"
    / "pose-graph-optimization.json"
)


def _record() -> dict:
    return json.loads(EVIDENCE.read_text())


def _identity() -> list[list[float]]:
    return [[1.0 if row == col else 0.0 for col in range(4)] for row in range(4)]


def _graph(**optimization: Any) -> dict:
    """A two-node graph the base validator accepts, so only the new block is at issue."""

    graph = {
        "nodes": [{"pose": _identity()}, {"pose": _identity()}],
        "edges": [
            {
                "source_node_id": 0,
                "target_node_id": 1,
                "transformation": _identity(),
                "uncertain": False,
            }
        ],
    }
    if optimization:
        graph["optimization"] = optimization.get("optimization", optimization)
    return graph


def _summary(**overrides: Any) -> dict:
    summary = {
        "edges_submitted": 1,
        "loop_closures_submitted": 0,
        "edges_that_survived_pruning": 1,
        "edges_pruned": 0,
        "max_pose_correction": 0.0,
        "max_pose_correction_per_node": [0.0, 0.0],
    }
    summary.update(overrides)
    return summary


def test_a_graph_reporting_what_the_optimization_did_validates() -> None:
    validate_pose_graph(_graph(optimization=_summary()), fragment_count=2)


def test_a_graph_predating_the_optimization_block_still_validates() -> None:
    """Reports written before this block existed are still valid pose graphs."""

    graph = _graph()
    assert "optimization" not in graph
    validate_pose_graph(graph, fragment_count=2)


def test_a_graph_that_publishes_the_block_as_nothing_is_rejected() -> None:
    """Absent is fine; present and hollow is not.

    An empty mapping satisfies `"optimization" in graph` while asserting nothing,
    which is the one way to get the appearance of the measurement without it.
    """

    with pytest.raises(Open3dError, match="non-empty mapping"):
        validate_pose_graph({**_graph(), "optimization": {}}, fragment_count=2)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param(
            {"edges_that_survived_pruning": 2},
            "different surviving edge count",
            id="survivors_disagree_with_the_serialized_edges",
        ),
        pytest.param(
            {"edges_submitted": 3, "edges_pruned": 0},
            "do not add up",
            id="submitted_minus_pruned_is_not_the_survivors",
        ),
        pytest.param(
            {"edges_submitted": 2, "loop_closures_submitted": 3, "edges_pruned": 1},
            "more loop closures than edges",
            id="more_closures_than_edges",
        ),
        pytest.param(
            {"edges_submitted": 2, "loop_closures_submitted": 0, "edges_pruned": 1},
            "more pruned edges than it submitted",
            id="pruned_something_that_was_not_prunable",
        ),
        pytest.param(
            {"max_pose_correction_per_node": [0.0]},
            "one pose correction per node",
            id="one_correction_short",
        ),
        pytest.param(
            {"max_pose_correction_per_node": [0.0, -1.0]},
            "finite, non-negative",
            id="a_negative_correction",
        ),
        pytest.param(
            {"max_pose_correction_per_node": [0.0, math.nan]},
            "finite, non-negative",
            id="a_correction_that_is_not_a_number",
        ),
        pytest.param(
            {"max_pose_correction_per_node": [0.0, None]},
            "finite, non-negative",
            id="a_correction_that_was_not_taken",
        ),
        pytest.param(
            {"max_pose_correction": 0.5},
            "not the largest of the per-node",
            id="headline_larger_than_every_node",
        ),
        pytest.param(
            {"max_pose_correction": 0.0, "max_pose_correction_per_node": [0.0, 0.25]},
            "not the largest of the per-node",
            id="headline_hides_a_node_that_moved",
        ),
        pytest.param(
            {"edges_submitted": True},
            "must be a count",
            id="a_boolean_edge_count",
        ),
        pytest.param(
            {"edges_submitted": -1, "edges_pruned": -2},
            "must be a count",
            id="a_negative_edge_count",
        ),
    ],
)
def test_a_summary_that_contradicts_its_graph_is_rejected(
    overrides: dict, expected: str
) -> None:
    with pytest.raises(Open3dError, match=expected):
        validate_pose_graph(
            {**_graph(), "optimization": _summary(**overrides)}, fragment_count=2
        )


@pytest.mark.parametrize(
    "missing",
    [
        "edges_submitted",
        "loop_closures_submitted",
        "edges_that_survived_pruning",
        "edges_pruned",
        "max_pose_correction",
        "max_pose_correction_per_node",
    ],
)
def test_every_required_figure_is_required_by_name(missing: str) -> None:
    summary = _summary()
    del summary[missing]
    with pytest.raises(Open3dError, match=missing):
        validate_pose_graph({**_graph(), "optimization": summary}, fragment_count=2)


def test_the_headline_hiding_a_moved_node_is_the_case_worth_naming() -> None:
    """The one that would matter in production, stated as its own claim.

    A consumer reads `max_pose_correction` to decide whether the optimization did
    anything. If that figure can be smaller than a node's actual movement, a run
    that moved a fragment by a quarter of a metre can publish a correction of zero,
    which is exactly the reading the block exists to make impossible.
    """

    summary = _summary(
        max_pose_correction=0.0, max_pose_correction_per_node=[0.0, 0.25]
    )
    with pytest.raises(Open3dError, match="not the largest of the per-node"):
        validate_pose_graph({**_graph(), "optimization": summary}, fragment_count=2)


def test_the_record_keeps_the_control_that_makes_a_zero_readable() -> None:
    control = _record()["probe_control"]
    assert control["node_offset_before"] == 0.5
    assert control["node_offset_after"] == pytest.approx(0.0, abs=1e-6)
    assert control["max_pose_correction"] == pytest.approx(0.5, abs=1e-6), control
    assert "probe that cannot read poses" in control["so"]


def test_the_record_measures_both_sets_of_fragments_the_pipeline_runs_on() -> None:
    """Two inputs, two answers, not one figure standing in for both.

    They were nearly conflated: the first measurement went through `run_multiway` on
    the smoke's synthetic clouds and returned exactly zero, and the harness then
    described the upstream demo fragments with it. Those are different data at a
    different voxel size, and they give 0.0 and 4.7e-06.
    """

    record = _record()
    upstream = record["demo_fragments"]
    product = record["through_the_product_pipeline"]

    assert product["entry_point"].endswith("run_multiway")
    assert product["correction_is_exactly_zero"] is True
    assert product["max_pose_correction"] == 0.0

    low, high = upstream["max_pose_correction_over_repeats"]
    assert high > 0.0, "the upstream figure is not exactly zero and should not say so"
    assert low <= high
    # Small relative to the scene, which is the only sense in which it is small.
    assert upstream["largest_correction_as_a_share_of_scene_extent"] < 1e-5, upstream
    assert upstream["scene_extent"] > 1.0, upstream


def test_the_record_shows_pruning_leaving_no_trace() -> None:
    loop = _record()["loop_with_overlap"]
    assert loop["at_least_one_edge_was_pruned"] is True
    low, high = loop["edges_pruned_over_repeats"]
    assert low >= 1, loop
    assert loop["edges_submitted"] > high, loop


def test_the_record_declines_to_claim_the_optimizer_is_broken() -> None:
    """Zero correction on consistent input is the right answer, and it says so."""

    limits = _record()["what_this_does_not_claim"]
    assert "right answer" in limits["not_that_optimization_is_broken"]
    assert "mechanism working" in limits["not_that_pruning_is_wrong"]
    assert set(limits) == {
        "not_that_optimization_is_broken",
        "not_that_pruning_is_wrong",
        "not_a_drift_correction_benchmark",
    }


def test_the_record_keeps_the_case_where_the_two_mutations_are_observable() -> None:
    """Both product-path mutations are invisible on the fragments the smoke used.

    Reading the poses after optimization rather than before, and publishing the
    surviving edge count as the submitted one, produce byte-identical output on three
    consistent fragments: the correction is 0.0 and nothing is prunable either way.
    This case is the one that can fail, and the smoke drives the real pipeline over
    it for exactly that reason.
    """

    loop = _record()["through_the_product_on_a_loop"]
    assert loop["entry_point"].endswith("run_multiway")
    assert loop["edges_submitted"] == 6, loop
    assert loop["edges_pruned"] >= 1, loop
    assert loop["correction_is_nonzero"] is True, loop
    assert loop["max_pose_correction"] > 0.0, loop
    assert loop["published_block_matches_the_snapshot"] is True, loop


def test_the_two_loop_measurements_are_not_reconciled_into_one_figure() -> None:
    """They disagree, and the record says so instead of averaging them.

    The standalone loop at the harness's own constants prunes one edge; the same
    geometry through `run_multiway` at the manifest's voxel size and ICP distance
    factor prunes three. A single published figure would have been wrong for whichever
    reader used the other entry point.
    """

    record = _record()
    standalone = record["loop_with_overlap"]
    product = record["through_the_product_on_a_loop"]
    assert standalone["edges_submitted"] == product["edges_submitted"]
    assert standalone["edges_pruned_over_repeats"][1] != product["edges_pruned"], (
        "the two loop measurements have stopped disagreeing, so the reason this "
        "record keeps both no longer holds; re-check which one a test may assert on"
    )


def test_the_record_binds_the_harness_that_produced_it() -> None:
    """A digest nothing recomputes is decoration."""

    provenance = _record()["provenance"]
    assert "this evidence directory" in provenance["path_is_relative_to"]
    harness = EVIDENCE.parent / provenance["harness"]
    assert harness.exists(), harness
    import hashlib

    assert (
        hashlib.sha256(harness.read_bytes()).hexdigest() == provenance["harness_sha256"]
    ), "the harness changed without the record being regenerated"


def test_the_false_positive_that_made_the_audit_look_complete_is_recorded() -> None:
    """Scanning keys for 'before' matched `fused_points_before_downsample`.

    Retained because it briefly produced the answer that the submitted edge count was
    already published, which would have closed this scope as already handled.
    """

    omitted = _record()["what_the_record_omitted"]
    assert omitted["edges_submitted_was_published"] is False
    assert omitted["pose_correction_was_published"] is False
    assert (
        "fused_points_before_downsample"
        in (omitted["a_keyword_scan_said_otherwise_and_was_wrong"])
    )
