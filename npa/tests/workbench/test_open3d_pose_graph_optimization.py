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
| upstream `DemoICPPointClouds`             | 4.7e-06           | 2 of 3 survived  |
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
import tempfile
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
        # The coordinate-consistent half of the block, present by default because the
        # producer emits it. The matrix-entry figures above are kept under their own
        # names; these are the lengths.
        "max_point_displacement": 0.0,
        "max_point_displacement_per_node": [0.0, 0.0],
        "scene_extent": 4.0,
        "max_point_displacement_as_share_of_extent": 0.0,
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
    assert upstream["largest_matrix_entry_change_over_scene_extent"] < 1e-5, upstream
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
    geometry through `run_multiway` prunes three. A single published figure would have
    been wrong for whichever reader used the other entry point.

    Not the voxel size: both are 0.05. The correspondence distance differs (the
    harness's `voxel * 1.5 = 0.075` against the product's
    `voxel_size * icp_distance_factor = 0.02`), and so does the estimation path -- the
    product initialises with RANSAC/FPFH and refines at full resolution where the
    harness runs identity-initialised ICP on the downsampled clouds.
    """

    record = _record()
    standalone = record["loop_with_overlap"]
    product = record["through_the_product_on_a_loop"]
    assert standalone["edges_submitted"] == product["edges_submitted"]
    assert standalone["edges_pruned_over_repeats"][1] != product["edges_pruned"], (
        "the two loop measurements have stopped disagreeing, so the reason this "
        "record keeps both no longer holds; re-check which one a test may assert on"
    )


def _cases_reporting_edges(node: Any, path: str = "") -> list[tuple[str, dict]]:
    """Every mapping anywhere in the record that reports a submitted edge count.

    Found by shape rather than by listing case names, so a case added later is
    covered without anyone remembering to add it here. Naming the cases is how the
    gap below went unnoticed: four of five carried the arithmetic and the fifth,
    which no test named, is the one whose figure the PR body got wrong.
    """

    found: list[tuple[str, dict]] = []
    if isinstance(node, dict):
        if "edges_submitted" in node:
            found.append((path or "<root>", node))
        for key, value in node.items():
            found.extend(
                _cases_reporting_edges(value, f"{path}.{key}" if path else key)
            )
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_cases_reporting_edges(value, f"{path}[{index}]"))
    return found


def test_every_case_conserves_its_edges() -> None:
    """Submitted minus pruned must equal survived, in every case, with all three said.

    Review set `edges_that_survived_pruning` to 99 in two cases, `edges_pruned` to 99
    in a third, and rewrote the fourth to the body's own wrong figure, and the suite
    stayed at 30 passed. The pose-correction half of this record was guarded and the
    edge-survival half was not.
    """

    cases = _cases_reporting_edges(_record())
    assert len(cases) >= 5, [name for name, _ in cases]
    for name, case in cases:
        missing = sorted(
            key
            for key in ("edges_that_survived_pruning", "edges_pruned")
            if key not in case
        )
        assert not missing, f"{name} reports a submitted count without {missing}"
        assert (
            case["edges_submitted"] - case["edges_pruned"]
            == case["edges_that_survived_pruning"]
        ), f"{name} does not conserve its edges: {case}"
        assert case["edges_pruned"] >= 0, f"{name}: {case}"


def test_the_upstream_case_pruned_an_edge_and_says_so() -> None:
    """The specific cell the PR body got wrong, pinned as its own claim.

    The body read "3 of 3 survived" where 3 were submitted and 2 survived. It was the
    one case whose record omitted `edges_pruned`, so there was no number to contradict
    the sentence -- which is the mechanism this record is about, arriving in the
    record's own prose.
    """

    upstream = _record()["demo_fragments"]
    assert upstream["edges_submitted"] == 3, upstream
    assert upstream["edges_that_survived_pruning"] == 2, upstream
    assert upstream["edges_pruned"] == 1, (
        "the upstream demo fragments prune exactly one edge; if that has changed, the "
        "PR body and this test both need the new figure"
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


#: Which record case each row of this module's summary table reports, in the table's
#: own order. The table is prose until something reads it, and the row it got wrong was
#: the one claiming an edge count the record contradicts.
SUMMARY_TABLE_ROWS = (
    "probe_control",
    "demo_fragments",
    "through_the_product_pipeline",
    "loop_with_overlap",
)


def test_the_summary_table_in_this_docstring_agrees_with_the_record() -> None:
    """The table said "3 of 3 survived" for a case the record records as 2 of 3.

    That is verbatim the error this PR exists to prevent -- publishing the surviving
    edge count as the submitted one -- in the module whose tests prove the record does
    not make it. It was corrected in the PR description and left standing here, and
    none of the tests read docstrings, so nothing could catch it.
    """

    table = [
        line
        for line in (__doc__ or "").splitlines()
        if line.startswith("|") and "survived" in line and "---" not in line
    ]
    assert len(table) == len(SUMMARY_TABLE_ROWS), table

    record = _record()
    for line, case in zip(table, SUMMARY_TABLE_ROWS, strict=True):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        claimed = cells[-1]
        measured = record[case]
        expected = (
            f"{measured['edges_that_survived_pruning']} of "
            f"{measured['edges_submitted']} survived"
        )
        assert claimed == expected, (
            f"the table's row for {case} claims {claimed!r} where the record measures "
            f"{expected!r}"
        )


def test_the_record_explains_why_its_own_ratio_is_not_a_share_of_the_scene() -> None:
    """The ratio divides a pose matrix-entry delta by a length.

    A 4x4 pose mixes unitless rotation coefficients with a translation in scene units,
    so the largest changed entry can be either and the quotient is not scale-free. It
    was published as `largest_correction_as_a_share_of_scene_extent`, which it could
    not be. The measurement is kept -- it is what was taken -- under a name that says
    what it is, with the reason beside it.
    """

    case = _record()["demo_fragments"]
    assert "largest_correction_as_a_share_of_scene_extent" not in case
    ratio = case["largest_matrix_entry_change_over_scene_extent"]
    # Loose, because the record publishes the correction rounded to ten places and
    # the ratio to twelve, so recomputing from the published numerator differs in the
    # sixth significant figure. The point is that the ratio is that quotient, not that
    # the rounding is reversible.
    assert ratio == pytest.approx(
        max(case["max_pose_correction_over_repeats"]) / case["scene_extent"], rel=1e-4
    )
    why = case["why_that_is_not_a_share_of_the_scene"]
    assert "millimetres" in why and "rotation" in why, why


def _displacement_and_entry_share(
    scale: float,
) -> tuple[float, float]:
    """Both ratios for one geometric change, with the scene expressed at `scale`.

    The pose change is a 90-degree rotation about z: the case root used as a
    counterexample, where the largest changed matrix entry is 1 regardless of units.
    """

    import numpy as np

    points = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [5.0, 5.0, 2.0],
            ]
        )
        * scale
    )
    before = np.identity(4)
    after = np.identity(4)
    after[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # The producer's own function, not a copy of its formula. A counterexample that
    # decides which quantity to publish has to run against the code that publishes it,
    # or it proves a property of the test.
    from npa.workbench.open3d.runner import _point_displacement

    displacement = _point_displacement(points, before, after)
    entry_change = float(np.abs(after - before).max())
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return displacement / extent, entry_change / extent


def test_the_published_share_is_the_same_number_for_the_same_geometry_at_any_scale() -> (
    None
):
    """The counterexample that decided which quantity to publish.

    `max_pose_correction` over the scene extent was published as a scene-relative
    share. It is not one: the numerator is the largest changed entry of a 4x4 pose,
    mixing unitless rotation coefficients with a translation in scene units, and the
    denominator is a length. Express the same scene in millimetres instead of metres
    and that quotient changes by a thousand for geometry that did not change.

    The displacement the pose change applies to the cloud's own points is a length in
    the same units as the extent, so the ratio is invariant -- which is the property a
    reader comparing two runs is relying on.
    """

    metres_displacement, metres_entry = _displacement_and_entry_share(1.0)
    millimetres_displacement, millimetres_entry = _displacement_and_entry_share(1000.0)

    assert millimetres_displacement == pytest.approx(metres_displacement, rel=1e-12), (
        "the displacement share has to describe the geometry, not the units"
    )
    # And the quantity that was published instead does not, by three orders of
    # magnitude. Asserted rather than described, so this stays a measurement.
    assert millimetres_entry == pytest.approx(metres_entry / 1000.0, rel=1e-9)
    assert metres_entry / millimetres_entry == pytest.approx(1000.0, rel=1e-9)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"max_point_displacement_per_node": [0.1, 0.2, 0.3]},
            "one point displacement per node",
        ),
        ({"max_point_displacement": 0.5}, "not the largest of the per-node"),
        ({"max_point_displacement_per_node": [0.1, float("nan")]}, "finite, non-negat"),
        ({"scene_extent": 0.0}, "finite, positive length"),
        ({"scene_extent": -1.0}, "finite, positive length"),
        (
            {"max_point_displacement_as_share_of_extent": None},
            "no share of the extent",
        ),
        (
            {"max_point_displacement_as_share_of_extent": 0.5},
            "is not the displacement",
        ),
    ],
)
def test_the_displacement_figures_are_held_to_each_other(
    overrides: dict, message: str
) -> None:
    """These are the fields the record argues a reader should act on.

    They were published outside the validator, so a graph could carry a real
    displacement under a zero share, or a share computed against a different extent,
    and pass the gate every real run goes through. That is the failure this block
    exists to prevent, displaced one field over.
    """

    summary = _summary(**overrides)
    with pytest.raises(Open3dError, match=message):
        validate_pose_graph(_graph(optimization=summary), fragment_count=2)


def test_a_record_without_the_displacement_fields_is_still_valid() -> None:
    """Graphs written before these fields existed are still valid pose graphs.

    The optimization block is optional for that reason, and the displacement group has
    to be optional within it for the same one, or this change invalidates every
    artifact already on disk.
    """

    summary = _summary()
    for key in (
        "max_point_displacement",
        "max_point_displacement_per_node",
        "max_point_displacement_as_share_of_extent",
        "scene_extent",
    ):
        summary.pop(key)
    validate_pose_graph(_graph(optimization=summary), fragment_count=2)


def _optimization_keys_the_producer_emits() -> set[str]:
    """The keys of the `optimization` dict literal in `run_multiway`, read statically.

    By AST rather than by running it, so this holds in the shared test shards where
    Open3D is not installed. A producer key is either a leaf in the published table or
    a container whose own leaves are, and this returns the top level of that.
    """

    import ast

    source = (
        Path(__file__).resolve().parents[3]
        / "npa"
        / "src"
        / "npa"
        / "workbench"
        / "open3d"
        / "runner.py"
    )
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "optimization"
                and isinstance(value, ast.Dict)
            ):
                return {
                    inner.value
                    for inner in value.keys
                    if isinstance(inner, ast.Constant)
                }
    raise AssertionError("no optimization dict literal found in runner.py")


def test_every_field_the_producer_publishes_is_in_the_public_report_allowlist() -> None:
    """A new field in the published block stops the next public run report.

    `derive_public_run_report.py` carries a leaf through only if its table names it,
    and `_classify` raises `SystemExit` otherwise. The whole `optimization` block was
    added to the producer without an entry, so the next derivation would have aborted
    on `pose_graph.optimization.edges_submitted`. It fails closed, which is the right
    direction and still a break, and nothing bound the two together.
    """

    import importlib.util

    harness = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "evidence"
        / "open3d"
        / "harness"
        / "derive_public_run_report.py"
    )
    spec = importlib.util.spec_from_file_location("_derive_public_run_report", harness)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    emitted = _optimization_keys_the_producer_emits()
    assert emitted, "the producer's optimization block should not be empty"

    for prefix in ("graph/pose_graph.json", "graph/result.json"):
        allowed = module.ALLOWED[prefix]
        under = {
            key.split("optimization.", 1)[1].split("[")[0].split(".")[0]
            for key in allowed
            if "optimization." in key
        }
        unlisted = sorted(emitted - under)
        assert not unlisted, (
            f"{prefix} does not classify {unlisted}, so deriving a public run report "
            "from a multiway run would exit on the first one"
        )


def test_the_validators_required_fields_are_all_publishable() -> None:
    """The gate and the publication table have to agree on the same block.

    One rejecting a record the other cannot carry would make a valid run
    unpublishable, which is the same desynchronisation in the other direction.
    """

    import importlib.util

    harness = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "evidence"
        / "open3d"
        / "harness"
        / "derive_public_run_report.py"
    )
    spec = importlib.util.spec_from_file_location("_derive_public_run_report", harness)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    allowed = module.ALLOWED["graph/pose_graph.json"]
    under = {
        key.split("optimization.", 1)[1].split("[")[0].split(".")[0]
        for key in allowed
        if "optimization." in key
    }
    # Every key `_summary()` carries is one the validator accepts and the producer
    # emits, so the table has to name all of them.
    assert not sorted(set(_summary()) - under)


def test_the_producer_divides_the_displacement_and_not_the_entry_delta() -> None:
    """Which variable reaches the published share, measured by running the producer.

    The tests above hold a *given* block to itself, and `_point_displacement` is
    covered directly. Neither sees which of the two quantities `run_multiway` actually
    puts over the extent, and swapping them back is a one-word edit that reinstates
    exactly the defect this change exists to remove.

    Skipped where Open3D is absent, which is the shared shards -- a skip is not
    coverage. The always-on guards for this are the container smoke, which asserts the
    same identity on its own run, and `validate_pose_graph`, which every real run
    passes through and which now rejects a share that is not the displacement over the
    extent.
    """

    o3d = pytest.importorskip("open3d")
    import numpy as np

    from npa.workbench.open3d.runner import run_multiway
    from npa.workbench.open3d.schemas import RegistrationManifest

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix="npa-posegraph-share-") as directory:
        root = Path(directory)
        paths = {}
        for index in range(3):
            points = rng.uniform(-1.0, 1.0, size=(4000, 3))
            points[:, 2] = rng.uniform(-0.02, 0.02, 4000)
            points[:, 0] += index * 0.35
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            path = root / f"fragment_{index}.ply"
            assert o3d.io.write_point_cloud(str(path), cloud)
            paths[f"fragment_{index}"] = str(path)

        output = root / "multiway"
        output.mkdir()
        manifest = RegistrationManifest(
            fragments=[
                {
                    "id": name,
                    "uri": paths[name],
                    "sha256": "0" * 64,
                    "bytes": Path(paths[name]).stat().st_size,
                }
                for name in sorted(paths)
            ],
            voxel_size=0.05,
        ).model_dump(mode="json")
        report = run_multiway({"manifest": manifest, "fragments": paths}, output, "t")

    block = report["pose_graph"]["optimization"]
    extent = block["scene_extent"]
    assert extent > 0.0, block
    assert block["max_point_displacement_as_share_of_extent"] == pytest.approx(
        block["max_point_displacement"] / extent, rel=1e-12
    ), block

    # And where the two quantities differ, the published share is not the other one.
    # On fragments that already agree both are zero and this says nothing, which is
    # why the assertion is conditional rather than absent.
    if block["max_pose_correction"] != block["max_point_displacement"]:
        assert block["max_point_displacement_as_share_of_extent"] != pytest.approx(
            block["max_pose_correction"] / extent, rel=1e-9
        ), block
