"""The registration numbers are not accuracy, and the measure that shows it.

`icp.fitness` and `icp.inlier_rmse` come from ICP's own result, computed against the
correspondences ICP chose. A consumer reading `inlier_rmse 0.0124` at `voxel_size 0.05`
concludes the alignment is good to a quarter of a voxel. On a symmetric pair that
reading is available while the alignment is tens of voxels wrong.

Everything here runs with `open3d` absent, which is the state of every CI shard: the
definitions are plain arithmetic and are separated from the Open3D call that supplies
their inputs. `docs/evidence/open3d/registration-ambiguity.json` carries the three
measured cases, including the one this measure misses.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from npa.workbench.open3d.artifacts import Open3dError, validate_pair
from npa.workbench.open3d.runner import (
    AGREEMENT_COUNTEREXAMPLE,
    _agreement,
    _transform_displacement,
)
from npa.workbench.open3d.schemas import RegistrationManifest

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs" / "evidence" / "open3d"
RECORD = EVIDENCE / "registration-ambiguity.json"
SRC = REPO / "npa" / "src" / "npa" / "workbench" / "open3d"

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def shifted(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, dx],
        [0.0, 1.0, 0.0, dy],
        [0.0, 0.0, 1.0, dz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def turned(degrees: float) -> list[list[float]]:
    """A rotation about z, so a displacement depends on how far out the points are."""

    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return [
        [cos, -sin, 0.0, 0.0],
        [sin, cos, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def record() -> dict:
    return json.loads(RECORD.read_text())


def test_a_pure_translation_displaces_every_point_by_its_length() -> None:
    points = [[0.0, 0.0, 0.0], [5.0, -3.0, 2.0]]
    moved = _transform_displacement(IDENTITY, shifted(dx=3.0, dy=4.0), points)
    assert moved == pytest.approx(5.0)


def test_the_same_transform_twice_displaces_nothing() -> None:
    points = [[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0]]
    assert _transform_displacement(turned(37.0), turned(37.0), points) == 0.0


def test_a_rotation_is_measured_through_the_points_and_not_as_an_angle() -> None:
    """The reason transforms are compared through the cloud rather than by matrix.

    The same ten degrees about z moves a point ten units out by ten times as far as a
    point one unit out. An angle cannot tell a consumer how far their data moves, and
    the distance it does move is the thing they can act on.
    """

    near = _transform_displacement(IDENTITY, turned(10.0), [[1.0, 0.0, 0.0]])
    far = _transform_displacement(IDENTITY, turned(10.0), [[10.0, 0.0, 0.0]])
    assert far == pytest.approx(near * 10.0)
    # A chord, not an arc: the straight-line distance between where the two
    # transforms put the point.
    assert near == pytest.approx(2.0 * math.sin(math.radians(5.0)))


def test_displacement_is_an_rms_and_not_the_worst_point() -> None:
    """One point moves, one does not. The mean of squares, so the answer is 1/sqrt(2)."""

    moved = _transform_displacement(
        IDENTITY, shifted(dx=1.0), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert moved == pytest.approx(1.0)
    partial = _transform_displacement(
        IDENTITY, turned(90.0), [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # (1,0,0) goes to (0,1,0), a distance of sqrt(2); (0,0,1) is on the axis and
    # does not move. RMS of {sqrt(2), 0} is 1.
    assert partial == pytest.approx(1.0)


def attempt(
    seed: int, transform: list[list[float]], fitness: float, rmse: float
) -> dict:
    return {
        "seed": seed,
        "transformation": transform,
        "icp": {"fitness": fitness, "inlier_rmse": rmse, "correspondence_count": 10},
    }


def test_the_worst_disagreement_is_over_every_pair_and_not_just_the_first() -> None:
    """Three seeds where the first is between the other two.

    Comparing each attempt against attempt zero -- the obvious implementation, and the
    one my own probe used before this was shipped -- reports 1.0 here. The two seeds
    that actually disagree are 1 and 2, by 2.0.
    """

    attempts = [
        attempt(0, IDENTITY, 0.9, 0.01),
        attempt(1, shifted(dx=-1.0), 0.9, 0.01),
        attempt(2, shifted(dx=1.0), 0.9, 0.01),
    ]
    block = _agreement(attempts, [[0.0, 0.0, 0.0]], voxel=0.5)
    assert block["worst_pairwise_displacement"] == pytest.approx(2.0)
    assert block["worst_disagreeing_seeds"] == [1, 2]
    assert block["worst_pairwise_displacement_voxels"] == pytest.approx(4.0)


def test_agreement_is_zero_when_every_seed_lands_in_the_same_place() -> None:
    attempts = [attempt(seed, turned(12.0), 0.9, 0.01) for seed in (4, 5, 6)]
    block = _agreement(attempts, [[2.0, 1.0, 0.0]], voxel=0.1)
    assert block["worst_pairwise_displacement"] == 0.0
    assert block["worst_disagreeing_seeds"] == []


def test_the_spreads_are_the_full_range_and_not_the_ends_of_the_list() -> None:
    """The largest fitness is in the middle, so first-to-last would understate it."""

    attempts = [
        attempt(0, IDENTITY, 0.50, 0.010),
        attempt(1, IDENTITY, 0.90, 0.030),
        attempt(2, IDENTITY, 0.60, 0.012),
    ]
    block = _agreement(attempts, [[1.0, 0.0, 0.0]], voxel=1.0)
    assert block["fitness_spread"] == pytest.approx(0.40)
    assert block["inlier_rmse_spread"] == pytest.approx(0.020)


def test_the_published_block_carries_the_case_where_agreement_was_wrong() -> None:
    """Because a consumer who reads agreement as correctness is the invited failure.

    As figures rather than as prose. The first version of this was a sentence and a
    guard that looked for the phrase "wrong branch" in it, which let the sentence be
    turned into its own opposite -- "can never agree on the same wrong branch" -- with
    everything still passing. A relation between two numbers cannot be inverted that
    way without the numbers moving.
    """

    block = _agreement(
        [attempt(0, IDENTITY, 0.9, 0.01), attempt(1, IDENTITY, 0.9, 0.01)],
        [[1.0, 0.0, 0.0]],
        voxel=1.0,
    )
    case = block["agreement_is_not_correctness"]["measured_counterexample"]
    assert (
        case["every_seed_was_wrong_by_voxels"]
        > 10.0 * case["seeds_agreed_within_voxels"]
    )


def test_the_counterexample_the_code_publishes_is_the_one_that_was_measured() -> None:
    """The figures in the source have to be the record's, or they are decoration."""

    case = AGREEMENT_COUNTEREXAMPLE
    missed = record()["cases"]["discrete_symmetry"]
    measured_agreement = missed["agreement_between_seeds"][
        "worst_pairwise_displacement_voxels"
    ]
    measured_error = missed["distance_from_the_known_truth"]["best_voxels"]
    assert case["seeds_agreed_within_voxels"] == pytest.approx(
        measured_agreement, abs=0.005
    ), (
        f"the code publishes {case['seeds_agreed_within_voxels']} voxels of agreement "
        f"against a measured {measured_agreement:.4f}"
    )
    assert case["every_seed_was_wrong_by_voxels"] == pytest.approx(
        measured_error, abs=0.05
    ), (
        f"the code publishes an error of {case['every_seed_was_wrong_by_voxels']} "
        f"voxels against a measured {measured_error:.4f}"
    )
    assert case["seeds"] == len(missed["seeds"])
    assert (REPO / case["evidence"]).exists()


def test_a_zero_voxel_size_reports_no_voxel_figure_rather_than_dividing() -> None:
    block = _agreement(
        [attempt(0, IDENTITY, 0.9, 0.01), attempt(1, shifted(dx=1.0), 0.9, 0.01)],
        [[0.0, 0.0, 0.0]],
        voxel=0.0,
    )
    assert block["worst_pairwise_displacement"] == pytest.approx(1.0)
    assert block["worst_pairwise_displacement_voxels"] is None


def good_row(**overrides) -> dict:
    block = _agreement(
        [
            attempt(0, IDENTITY, 0.90, 0.010),
            attempt(1, shifted(dx=0.2), 0.88, 0.011),
        ],
        [[0.0, 0.0, 0.0]],
        voxel=0.05,
    )
    row = {
        "pair_id": "a__b",
        "source_id": "a",
        "target_id": "b",
        "source_points": 100,
        "target_points": 120,
        "source_downsampled_points": 40,
        "target_downsampled_points": 50,
        "fpfh_dimension": 33,
        "global_registration": {
            "fitness": 0.5,
            "inlier_rmse": 0.02,
            "correspondence_count": 30,
        },
        "icp": {"fitness": 0.9, "inlier_rmse": 0.01, "correspondence_count": 60},
        "transformation": IDENTITY,
        "aligned_points": 100,
        "aligned_sha256": "a" * 64,
        "wall_seconds": 1.5,
        "agreement": block,
    }
    row.update(overrides)
    return row


def test_a_consistent_row_with_agreement_validates() -> None:
    validate_pair(good_row())


def test_a_row_from_a_manifest_that_did_not_ask_for_agreement_validates() -> None:
    """Opt-in. Every row already written in this repository's evidence lacks it."""

    row = good_row()
    del row["agreement"]
    validate_pair(row)


@pytest.mark.parametrize(
    "hollow", [None, {}, [], 0], ids=["none", "empty", "list", "zero"]
)
def test_carrying_the_key_with_nothing_in_it_is_not_the_same_as_not_asking(
    hollow,
) -> None:
    """Absent is exempt; present and hollow is a writer declining to measure.

    Keyed on absence rather than on `None`, because those read the same through
    `row.get` and deserve opposite answers.
    """

    with pytest.raises(Open3dError, match="missing measurements"):
        validate_pair(good_row(agreement=hollow))


@pytest.mark.parametrize(
    ("what", "mutate"),
    [
        ("one seed is not a comparison", lambda b: b.__setitem__("seeds", [0])),
        (
            "the same seed twice is one run",
            lambda b: b.__setitem__("seeds", [3, 3]),
        ),
        (
            "fewer fitness values than seeds",
            lambda b: b.__setitem__("fitness_by_seed", [0.9]),
        ),
        (
            "a fitness spread that is not the range",
            lambda b: b.__setitem__("fitness_spread", 0.5),
        ),
        (
            "an rmse spread that is not the range",
            lambda b: b.__setitem__("inlier_rmse_spread", 0.5),
        ),
        (
            "a negative displacement",
            lambda b: b.__setitem__("worst_pairwise_displacement", -1.0),
        ),
        (
            "a disagreement that names no seeds",
            lambda b: b.__setitem__("worst_disagreeing_seeds", []),
        ),
        (
            "a disagreement between seeds that were not run",
            lambda b: b.__setitem__("worst_disagreeing_seeds", [11, 12]),
        ),
        (
            "an infinite fitness",
            lambda b: b.__setitem__("fitness_by_seed", [float("inf"), 0.9]),
        ),
        (
            "the limitation quietly dropped",
            lambda b: b.__setitem__("agreement_is_not_correctness", "looks good"),
        ),
        (
            "a counterexample that is not one",
            lambda b: b["agreement_is_not_correctness"][
                "measured_counterexample"
            ].update({"every_seed_was_wrong_by_voxels": 0.2}),
        ),
    ],
)
def test_an_inconsistent_agreement_block_is_rejected(what: str, mutate) -> None:
    row = good_row()
    mutate(row["agreement"])
    with pytest.raises(Open3dError):
        validate_pair(row)


def test_every_rejection_says_something_different() -> None:
    """Ten identical messages would be one check wearing ten hats."""

    messages = set()
    mutations = [
        lambda b: b.__setitem__("seeds", [0]),
        lambda b: b.__setitem__("seeds", [3, 3]),
        lambda b: b.__setitem__("fitness_by_seed", [0.9]),
        lambda b: b.__setitem__("fitness_spread", 0.5),
        lambda b: b.__setitem__("worst_pairwise_displacement", -1.0),
        lambda b: b.__setitem__("worst_disagreeing_seeds", []),
        lambda b: b.__setitem__("worst_disagreeing_seeds", [11, 12]),
        lambda b: b.__setitem__("fitness_by_seed", [float("inf"), 0.9]),
        lambda b: b.__setitem__("agreement_is_not_correctness", "looks good"),
        lambda b: b["agreement_is_not_correctness"]["measured_counterexample"].update(
            {"every_seed_was_wrong_by_voxels": 0.2}
        ),
    ]
    for mutate in mutations:
        row = good_row()
        mutate(row["agreement"])
        with pytest.raises(Open3dError) as caught:
            validate_pair(row)
        messages.add(str(caught.value))
    assert len(messages) == len(mutations), sorted(messages)


def test_the_measurement_is_off_unless_a_manifest_asks_for_it() -> None:
    """A default that changed the cost of every existing run would not be opt-in."""

    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                {"id": "a", "uri": "file:///a.ply", "sha256": "0" * 64, "bytes": 1},
                {"id": "b", "uri": "file:///b.ply", "sha256": "1" * 64, "bytes": 1},
            ],
            "voxel_size": 0.05,
        }
    )
    assert manifest.agreement_seeds == 0


def test_the_register_path_still_publishes_what_the_validator_reads() -> None:
    """The join between writer and validator, where Open3D is absent.

    `validate_pair` skips the block when a row does not carry it, so that rows already
    written validate. Deleting the emission would therefore switch the whole check off
    with nothing going red.
    """

    tree = ast.parse((SRC / "runner.py").read_text())
    writer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_register"
    )
    assigned = {
        node.targets[0].slice.value
        for node in ast.walk(writer)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].slice, ast.Constant)
    }
    assert "agreement" in assigned, (
        "run_register no longer attaches the agreement block, and validate_pair skips "
        "it on a row that lacks it, so the measurement would be gone silently"
    )


def test_the_record_publishes_the_case_this_measure_misses() -> None:
    """A capability record that showed only the case it catches would be a brochure."""

    cases = record()["cases"]
    assert set(cases) == {
        "continuous_symmetry",
        "discrete_symmetry",
        "the_real_demo_scans",
    }
    assert cases["continuous_symmetry"]["the_reading"]["so_this_case_is"] == "detected"
    assert cases["discrete_symmetry"]["the_reading"]["so_this_case_is"] == "missed"


def test_the_missed_case_really_is_agreement_on_a_wrong_answer() -> None:
    """Otherwise 'missed' could mean the construction simply failed to be symmetric."""

    missed = record()["cases"]["discrete_symmetry"]
    agreement = missed["agreement_between_seeds"]["worst_pairwise_displacement_voxels"]
    truth = missed["distance_from_the_known_truth"]
    assert agreement < 1.0, "the seeds did not agree, so this is not the missed case"
    assert truth["every_seed_is_wrong"] is True
    assert truth["best_voxels"] > 1.0
    # Not equality: the six wrong answers differ from each other by about 0.001
    # voxels, which is the thread-order residue rather than a different branch. The
    # claim is that the variation among them is negligible beside how wrong they are.
    spread = truth["worst_voxels"] - truth["best_voxels"]
    assert spread < 0.01 * truth["best_voxels"], (
        f"the seeds' errors span {spread:.4f} voxels against an error of "
        f"{truth['best_voxels']:.1f}, so they landed in materially different wrong "
        "places and this is the detected case rather than the missed one"
    )


def test_the_detected_case_is_detected_without_knowing_the_truth() -> None:
    """The disagreement has to be the thing that fires, not the truth comparison."""

    detected = record()["cases"]["continuous_symmetry"]
    reading = detected["the_reading"]
    assert reading["the_disagreement_would_have_warned_them"] is True
    assert reading["published_numbers_would_have_reassured_a_consumer"] is True
    assert reading["the_alignment_is_wrong"] is True
    published = detected["published_numbers_a_consumer_reads"]
    assert published["worst_inlier_rmse_in_voxels"] < 1.0, (
        "the published rmse already looked bad, so this case does not show that the "
        "published numbers mislead"
    )


def test_the_control_is_quiet_on_the_data_the_records_were_built_from() -> None:
    """A measure that fires on the working pair would be unusable, not conservative."""

    control = record()["cases"]["the_real_demo_scans"]
    agreement = control["agreement_between_seeds"]
    assert agreement["worst_pairwise_displacement_voxels"] < 0.01
    assert "no_truth_is_available" in control
    assert "the_reading" not in control, (
        "a truth comparison on the real pair would be invented; there is none"
    )


def test_the_record_declines_to_set_a_threshold() -> None:
    """Three cases, one of each kind. A cutoff drawn from that would be unsafe."""

    claims = " ".join(record()["what_this_does_not_claim"])
    assert "A threshold" in claims
    assert "n=1 of each" in claims


def test_the_record_states_the_cost_it_adds() -> None:
    cost = record()["cost"]
    assert "one more full registration" in cost
    assert "0" in cost and "agreement_seeds" in cost


def test_the_harness_measures_through_the_shipped_registration() -> None:
    """A record produced by a reimplementation would describe the reimplementation."""

    harness = (EVIDENCE / "harness" / "registration_ambiguity.py").read_text()
    assert "from npa.workbench.open3d.runner import" in harness
    assert "_register_pair" in harness
    assert record()["environment"]["registration"] == (
        "the shipped _register_pair, unmodified"
    )
