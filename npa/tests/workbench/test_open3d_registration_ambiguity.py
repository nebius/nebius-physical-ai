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
from unittest import mock

import pytest

import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workbench.open3d import runner as mod
from npa.workbench.open3d import runtime
from npa.workbench.open3d.artifacts import (
    Open3dError,
    canonical,
    sha256_bytes,
    validate_pair,
)
from npa.workbench.open3d.runner import (
    AGREEMENT_COUNTEREXAMPLE,
    _agreement,
    _transform_displacement,
)
from npa.workbench.open3d.schemas import (
    Fragment,
    RegistrationManifest,
    StageDemoRequest,
)

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


def test_editing_one_report_does_not_change_what_the_next_one_claims() -> None:
    """The counterexample is copied into each report, not shared with it.

    Published by reference, one `report[...].update(...)` anywhere changed what every
    later report in the process claimed. A test in this file did exactly that -- it
    mutates the figure to check the validator rejects a non-counterexample -- and the
    corruption only surfaced when xdist ran the tests in a different order, which is
    the worst way to find out.
    """

    first = _agreement(
        [attempt(0, IDENTITY, 0.9, 0.01), attempt(1, IDENTITY, 0.9, 0.01)],
        [[1.0, 0.0, 0.0]],
        voxel=1.0,
    )
    first["agreement_is_not_correctness"]["measured_counterexample"].update(
        {"every_seed_was_wrong_by_voxels": 0.2, "construction": "nonsense"}
    )
    second = _agreement(
        [attempt(0, IDENTITY, 0.9, 0.01), attempt(1, IDENTITY, 0.9, 0.01)],
        [[1.0, 0.0, 0.0]],
        voxel=1.0,
    )
    case = second["agreement_is_not_correctness"]["measured_counterexample"]
    assert case == dict(AGREEMENT_COUNTEREXAMPLE)
    with pytest.raises(TypeError):
        AGREEMENT_COUNTEREXAMPLE["seeds"] = 99  # type: ignore[index]


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


@pytest.mark.parametrize(
    "value",
    [None, [], "0.13", "many", {}, True],
    ids=["null", "list", "numeric_string", "word", "dict", "bool"],
)
def test_a_counterexample_measurement_that_is_not_a_number_is_rejected_as_one(
    value,
) -> None:
    """As `Open3dError`, not as whatever `float()` happens to raise.

    A reviewer found the guard for this ran *after* the `float()` unpack, so `None`
    escaped as `TypeError` -- which the CLI's `(Open3dError, ValueError)` does not
    catch -- and `null` is the likeliest real shape for a measurement that was not
    taken. The correct check was already written; it ran second.
    """

    row = good_row()
    case = row["agreement"]["agreement_is_not_correctness"]["measured_counterexample"]
    case["every_seed_was_wrong_by_voxels"] = value
    with pytest.raises(Open3dError, match="finite numbers"):
        validate_pair(row)


def test_a_counterexample_exactly_on_the_boundary_is_rejected() -> None:
    """The guard makes the claim hard to satisfy, so the boundary is the strict side."""

    row = good_row()
    case = row["agreement"]["agreement_is_not_correctness"]["measured_counterexample"]
    case["seeds_agreed_within_voxels"] = 1.0
    case["every_seed_was_wrong_by_voxels"] = 10.0
    with pytest.raises(Open3dError, match="not one"):
        validate_pair(row)
    case["every_seed_was_wrong_by_voxels"] = 10.000001
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
    declaration = cases["hashes_that_name_no_file_here"]
    assert isinstance(declaration, str) and declaration.strip()
    assert set(cases) - {"hashes_that_name_no_file_here"} == {
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


# --- F1: the measurement is reachable from the surfaces an agent actually drives


def test_the_measurement_can_be_asked_for_without_editing_object_storage() -> None:
    """The field existed and nothing could set it.

    `RegistrationManifest.agreement_seeds` was declared, but `_publish_manifest` built
    the manifest from four other fields, no request model carried it, and no CLI flag
    or catalog template mentioned it. The only way in was to hand-edit a digest-bound
    `manifest.json` between `prepare` and `register` -- which is the one thing that
    digest exists to prevent.
    """

    request = StageDemoRequest(output_path="s3://b/p/", agreement_seeds=3)
    published: dict[str, bytes] = {}

    def capture(uri: str, body: bytes) -> None:
        published[uri] = body

    with mock.patch.object(runtime, "_publish", capture):
        result = runtime._publish_manifest(
            request,
            [
                Fragment(id="a", uri="file:///a.ply", sha256="0" * 64, bytes=11),
                Fragment(id="b", uri="file:///b.ply", sha256="1" * 64, bytes=22),
            ],
            demo_release=None,
        )

    assert result["agreement_seeds"] == 3
    body = json.loads(next(iter(published.values())))
    assert body["agreement_seeds"] == 3, (
        "prepare published a manifest that does not carry the seeds it was asked for, "
        "so register cannot read them"
    )


def test_both_staging_commands_expose_the_flag() -> None:
    """A flag that only one of the two entry points has is reachable from one."""

    runner = CliRunner()
    for command in ("prepare", "stage-demo"):
        helped = runner.invoke(app, ["workbench", "open3d", command, "--help"])
        assert helped.exit_code == 0, helped.output
        assert "--agreement-seeds" in helped.output, command


def test_the_catalog_passes_the_flag_and_the_shipped_spec_defines_it() -> None:
    """A workflow stage renders cleanly and then fails in the pod, or it is wired.

    An argv_template naming a config key no spec defines renders as an empty string
    and the stage dies on argument parsing; a spec key no template reads is inert.
    Both halves have to be there.
    """

    spec = yaml.safe_load(
        (REPO / "workflows" / "testing" / "open3d-registration.yaml").read_text()
    )
    assert "open3d_agreement_seeds" in spec["config"]
    for name in ("workbench.open3d.prepare", "workbench.open3d.stage_demo"):
        argv = TOOL_CATALOG[name].argv_template
        assert "--agreement-seeds" in argv, name
        value = argv[argv.index("--agreement-seeds") + 1]
        assert value == "{{config.open3d_agreement_seeds}}", (name, value)


def test_the_agent_facing_contract_mentions_the_measurement() -> None:
    """An agent reading only the skill has to be able to find the input, and the
    one-directional reading of its output, or it will read agreement as accuracy."""

    skill = (REPO / "skills" / "tools" / "open3d" / "SKILL.md").read_text()
    assert "--agreement-seeds" in skill
    assert "agreement" in skill and "proves nothing" in skill


# --- F2: turning the diagnostic on does not change what it reports on


class _RngDouble:
    """A generator whose draws depend on where in the stream they are taken.

    Root's point: asserting byte equality out of real multi-threaded Open3D would be
    asserting something #1033 has already measured to be false, because the reductions
    are thread-ordered. What is being tested here is narrower and is exactly the defect
    -- whether the primary registrations see the same RNG stream in both modes -- so the
    generator is a double that makes any difference in stream position visible.
    """

    def __init__(self) -> None:
        self.state = 0
        self.draws: list[int] = []

    def seed(self, value: int) -> None:
        self.state = value * 1000

    def draw(self) -> int:
        self.state += 1
        self.draws.append(self.state)
        return self.state


def _fragment_files(root: Path) -> dict[str, str]:
    """Fragment input paths under the test's own temp directory.

    `_read_cloud` is stubbed in these tests, so nothing opens them; they are created
    anyway so the payload names files that exist, and they live under `tmp_path`
    rather than a hard-coded /tmp literal, which the security gate rejects.
    """

    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("a", "b", "c", "d"):
        path = root / f"{name}.ply"
        path.write_bytes(b"")
        paths[name] = str(path)
    return paths


def _register_against(rng: _RngDouble):
    def register(o3d, source, target, manifest) -> dict:
        drawn = rng.draw()
        return {
            "source_points": 10,
            "target_points": 10,
            "source_downsampled_points": 5,
            "target_downsampled_points": 5,
            "fpfh_dimension": 33,
            "global_registration": {
                "fitness": 0.5,
                "inlier_rmse": 0.02,
                "correspondence_count": 3,
            },
            "icp": {
                "fitness": 0.9 + drawn * 1e-6,
                "inlier_rmse": 0.01,
                "correspondence_count": 6,
            },
            # The alignment depends on the draw, which is what makes a changed RNG
            # stream position observable in the published transform.
            "transformation": shifted(dx=drawn * 1e-3),
        }

    return register


def _primary_alignments(monkeypatch, tmp_path: Path, seeds: int) -> list[list]:
    """Run `run_register` with the native library replaced, and return the published
    primary transforms in order."""

    rng = _RngDouble()
    fake = mock.MagicMock()
    fake.utility.random.seed.side_effect = rng.seed
    fake.io.write_point_cloud.return_value = True

    class _Cloud:
        points = [[0.0, 0.0, 0.0]]

        def transform(self, matrix) -> None:
            return None

    fake.geometry.PointCloud.side_effect = lambda source: _Cloud()

    monkeypatch.setattr(mod, "_open3d", lambda: fake)
    monkeypatch.setattr(mod, "_read_cloud", lambda o3d, path: _Cloud())
    monkeypatch.setattr(mod, "_register_pair", _register_against(rng))
    monkeypatch.setattr(mod, "_digest", lambda path: "d" * 64)
    monkeypatch.setattr(
        mod.Path, "stat", lambda self: mock.Mock(st_size=1), raising=False
    )

    out = tmp_path / f"seeds-{seeds}"
    out.mkdir()
    fragments = _fragment_files(tmp_path / f"fragments-{seeds}")
    payload = {
        "manifest": {
            "fragments": [
                {
                    "id": name,
                    "uri": f"file:///{name}.ply",
                    "sha256": "0" * 64,
                    "bytes": 1,
                }
                for name in ("a", "b", "c", "d")
            ],
            "voxel_size": 0.05,
            "agreement_seeds": seeds,
        },
        "fragments": fragments,
    }
    mod.run_register(payload, out, "run-1")
    rows = [
        json.loads(line)
        for line in (out / mod.PAIRS_JOURNAL).read_text().splitlines()
        if line
    ]
    return [row["transformation"] for row in rows]


def test_turning_the_diagnostic_on_does_not_move_the_alignment_it_reports(
    monkeypatch, tmp_path: Path
) -> None:
    """The defect, directly.

    `_measure_agreement` seeds the global generator per attempt and used to run inside
    the pair loop, leaving a re-seeded generator in front of pair 2. So pair 2 onwards
    published a different `transformation` and a different `aligned_sha256` between two
    runs of the same manifest that differed only in whether the diagnostic was asked
    for -- and the attempt labelled with the manifest's own seed was an alignment that
    existed only because the measurement was switched on.

    Three pairs, because the first pair was never affected and a two-fragment manifest
    cannot see this.
    """

    off = _primary_alignments(monkeypatch, tmp_path, seeds=0)
    on = _primary_alignments(monkeypatch, tmp_path, seeds=2)
    assert len(off) == 3
    assert on == off, (
        "the published primary alignments differ between agreement_seeds=0 and "
        "agreement_seeds=2, so the diagnostic is changing the thing it measures"
    )


def test_the_default_path_is_not_reseeded_per_pair_either(
    monkeypatch, tmp_path: Path
) -> None:
    """The other way to get this wrong.

    Seeding every pair would have made the two modes agree by changing the alignments
    the default path publishes -- every manifest already registered without this
    feature would move. The fix has to leave `agreement_seeds=0` alone, so this pins
    the seeding of the default path to one call, before the pairs.
    """

    rng = _RngDouble()
    fake = mock.MagicMock()
    fake.utility.random.seed.side_effect = rng.seed
    fake.io.write_point_cloud.return_value = True
    seeded: list[int] = []
    monkeypatch.setattr(
        mod, "_seeded", lambda o3d, manifest: seeded.append(manifest.random_seed)
    )
    monkeypatch.setattr(mod, "_open3d", lambda: fake)

    class _Cloud:
        points = [[0.0, 0.0, 0.0]]

        def transform(self, matrix) -> None:
            return None

    fake.geometry.PointCloud.side_effect = lambda source: _Cloud()
    monkeypatch.setattr(mod, "_read_cloud", lambda o3d, path: _Cloud())
    monkeypatch.setattr(mod, "_register_pair", _register_against(rng))
    monkeypatch.setattr(mod, "_digest", lambda path: "d" * 64)
    monkeypatch.setattr(
        mod.Path, "stat", lambda self: mock.Mock(st_size=1), raising=False
    )

    out = tmp_path / "default-seeding"
    out.mkdir()
    fragments = _fragment_files(tmp_path / "default-seeding-fragments")
    mod.run_register(
        {
            "manifest": {
                "fragments": [
                    {
                        "id": name,
                        "uri": f"file:///{name}.ply",
                        "sha256": "0" * 64,
                        "bytes": 1,
                    }
                    for name in ("a", "b", "c", "d")
                ],
                "voxel_size": 0.05,
            },
            "fragments": fragments,
        },
        out,
        "run-1",
    )
    assert seeded == [0], (
        "the default path is seeded once before the pairs, as it always was; seeding "
        "per pair would change every registration already published without this "
        f"feature (saw {len(seeded)} seedings)"
    )


# --- F3: a failed diagnostic re-run is a fact about the pair, not a dead stage


def test_one_failing_extra_seed_does_not_destroy_the_stage() -> None:
    """`_register_pair` raises when RANSAC or ICP finds no correspondences, and the
    extra attempts ran unguarded. A marginal-overlap pair -- the population this
    feature exists to flag -- could clear the primary seed and fail on seed k, taking
    `pairs.jsonl`, `result.json` and every already-computed pair with it, and
    reporting a diagnostic's RANSAC failure as though the requested registration had
    hit it."""

    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                {"id": "a", "uri": "file:///a.ply", "sha256": "0" * 64, "bytes": 1},
                {"id": "b", "uri": "file:///b.ply", "sha256": "1" * 64, "bytes": 1},
            ],
            "voxel_size": 0.05,
            "agreement_seeds": 2,
        }
    )
    calls = {"n": 0}

    def flaky(o3d, source, target, manifest):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Open3dError("RANSAC found no correspondences at this seed")
        measured = attempt(0, shifted(dx=0.01), 0.9, 0.01)
        measured.pop("seed")  # the real _register_pair does not label attempts
        return measured

    source = mock.Mock(points=[[0.0, 0.0, 0.0]])
    with mock.patch.object(mod, "_register_pair", flaky):
        block = mod._measure_agreement(
            mock.MagicMock(),
            source,
            mock.Mock(),
            manifest,
            attempt(0, IDENTITY, 0.9, 0.01),
        )

    assert block is not None
    assert block["failed_seeds"] == [
        {"seed": 1, "error": "RANSAC found no correspondences at this seed"}
    ]
    assert block["seeds"] == [0, 2], "the seeds that did register are the ones compared"
    validate_pair(good_row(agreement=block))


def test_too_few_surviving_seeds_says_so_rather_than_vanishing() -> None:
    """Dropping the key would be indistinguishable from never having asked for the
    measurement, which is the distinction the opt-in shape depends on."""

    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                {"id": "a", "uri": "file:///a.ply", "sha256": "0" * 64, "bytes": 1},
                {"id": "b", "uri": "file:///b.ply", "sha256": "1" * 64, "bytes": 1},
            ],
            "voxel_size": 0.05,
            "agreement_seeds": 1,
        }
    )

    def always_fails(o3d, source, target, manifest):
        raise Open3dError("ICP found no correspondences")

    with mock.patch.object(mod, "_register_pair", always_fails):
        block = mod._measure_agreement(
            mock.MagicMock(),
            mock.Mock(points=[[0.0, 0.0, 0.0]]),
            mock.Mock(),
            manifest,
            attempt(0, IDENTITY, 0.9, 0.01),
        )

    assert block["measured"] is False
    assert "fewer than the two" in block["why_not"]
    assert block["failed_seeds"][0]["seed"] == 1
    validate_pair(good_row(agreement=block))


@pytest.mark.parametrize(
    "broken",
    [
        {"measured": False},
        {"measured": False, "why_not": ""},
        {"measured": False, "why_not": "x", "failed_seeds": []},
        {"measured": False, "why_not": "x", "failed_seeds": [{"seed": "one"}]},
        {"measured": False, "why_not": "x", "failed_seeds": [{"seed": 1}]},
    ],
)
def test_an_unmeasured_agreement_still_has_to_account_for_itself(broken: dict) -> None:
    with pytest.raises(Open3dError):
        validate_pair(good_row(agreement=broken))


# --- F4: the manifest digest of an untouched prefix did not move


#: `sha256(canonical(...))` of a two-fragment manifest at 0.05 voxels, computed by
#: running the *base commit's* `RegistrationManifest.model_dump(mode="json")` -- the
#: schema as it was before this PR added a field to it. Hard-coded on purpose: two
#: hashes taken from the current model would agree with each other no matter what the
#: current model does, and the thing being checked is agreement with history.
PRE_CHANGE_MANIFEST_SHA256 = (
    "fb4b0dad0b1f6c76e49b471189c33a795bf3c90e668277a1efcaad16c12a9675"
)


def _two_fragment_manifest(**extra) -> RegistrationManifest:
    return RegistrationManifest.model_validate(
        {
            "fragments": [
                {"id": "a", "uri": "file:///a.ply", "sha256": "0" * 64, "bytes": 11},
                {"id": "b", "uri": "file:///b.ply", "sha256": "1" * 64, "bytes": 22},
            ],
            "voxel_size": 0.05,
            **extra,
        }
    )


def test_a_manifest_that_never_asked_hashes_as_it_did_before_the_field_existed() -> (
    None
):
    """`manifest_sha256` binds the published manifest bytes and `visualize` enforces
    that binding. Adding a defaulted field to `model_dump` changed the canonical bytes
    of every manifest that never used the feature, so a prefix published before this
    PR stopped matching its own recorded digest."""

    body = canonical(_two_fragment_manifest().canonical_document())
    assert sha256_bytes(body) == PRE_CHANGE_MANIFEST_SHA256


def test_asking_for_the_measurement_is_a_different_manifest_and_hashes_differently() -> (
    None
):
    """The compatibility is for prefixes that never asked, not a blanket exclusion."""

    body = canonical(_two_fragment_manifest(agreement_seeds=2).canonical_document())
    assert sha256_bytes(body) != PRE_CHANGE_MANIFEST_SHA256
    assert json.loads(body)["agreement_seeds"] == 2


def test_the_other_defaulted_fields_are_still_in_the_canonical_bytes() -> None:
    """`exclude_defaults=True` would also have made the digest compatible -- by
    dropping `random_seed`, `distance_factor`, `icp_estimation` and six more from the
    canonical form of every manifest, which changes far more than this PR touches and
    breaks the binding in the other direction."""

    document = _two_fragment_manifest().canonical_document()
    for field in (
        "random_seed",
        "distance_factor",
        "feature_radius_factor",
        "normal_radius_factor",
        "icp_distance_factor",
        "icp_estimation",
        "ransac_max_iteration",
        "ransac_confidence",
        "demo_data_release",
        "schema_version",
    ):
        assert field in document, field
    assert "agreement_seeds" not in document


def test_every_place_that_hashes_a_manifest_hashes_the_same_document() -> None:
    """Three call sites bind this digest: `prepare` publishes it, `register` records
    it, and `lineage` verifies it. One of them hashing a different document is the
    failure this guards -- it would read as a tampered manifest."""

    for path in ("runtime.py", "lineage.py"):
        text = (SRC / path).read_text()
        hashes = [
            line
            for line in text.splitlines()
            if "canonical(manifest." in line or "canonical(\n" in line
        ]
        assert hashes, f"{path} no longer hashes a manifest at all"
        for line in hashes:
            assert "canonical_document()" in line, (path, line.strip())


# --- F5 / F6: the evidence says what the record says, and says where it came from


def test_the_readme_quotes_the_control_figure_the_record_measured() -> None:
    """The README read the control's agreement as `0.00001 voxels`; the record says
    `9.74e-05`, ten times larger. The figure is the one quoted to argue the measure is
    quiet on working data."""

    control = record()["cases"]["the_real_demo_scans"]["agreement_between_seeds"]
    voxels = control["worst_pairwise_displacement_voxels"]
    readme = (REPO / "docs" / "evidence" / "open3d" / "README.md").read_text()
    assert f"{voxels:.3g}" in readme, (
        f"the README does not quote the measured {voxels:.3g} voxels"
    )
    assert "0.00001 voxels" not in readme


def test_the_record_does_not_claim_an_environment_it_did_not_run_in() -> None:
    """The directory's blanket rule said every JSON here came from the `npa-open3d`
    container. This one did not -- it is a host run on 3.14.7 against the image's
    3.11. Disclosed rather than relabelled: the figures are of the same Open3D and
    the same unmodified `_register_pair`, and regenerating them inside the image is a
    separate act from describing them correctly."""

    environment = record()["environment"]
    assert environment["python"] == "3.14.7"
    assert "not the npa-open3d image" in environment["interpreter"]
    assert environment["why_that_is_recorded"]
    readme = (REPO / "docs" / "evidence" / "open3d" / "README.md").read_text()
    assert "Every JSON here was written by a script in `harness/`, run inside" not in (
        readme
    ), "the blanket container claim is back"


def test_the_harness_can_run_the_way_the_readme_says_to_run_it() -> None:
    """The documented way is `docker run -v .../docs/evidence/open3d:/data ... python
    /data/harness/<name>.py`. Under that mount there is no repository above the file,
    so a `parents[4]`-only import and an output path beside the script both fail --
    the one documented route to regenerate this record could not execute."""

    harness = (
        REPO / "docs" / "evidence" / "open3d" / "harness" / "registration_ambiguity.py"
    ).read_text()
    assert "except ImportError" in harness, (
        "the import has no path that works when npa is installed and there is no "
        "repository above the file"
    )
    assert 'Path("/data")' in harness


def test_the_harness_import_fallback_actually_recovers(tmp_path: Path) -> None:
    """A fallback that cannot run is not a fallback.

    The first attempt leaves its half-imported modules in `sys.modules`, so adding a
    path and retrying returns the same failure -- which is what happens whenever
    *some other* npa is importable, as it is in any environment with a second checkout
    installed. The cache has to be dropped for the retry to reach this tree.
    """

    harness = (
        REPO / "docs" / "evidence" / "open3d" / "harness" / "registration_ambiguity.py"
    ).read_text()
    fallback = harness[harness.index("except ImportError") :]
    fallback = fallback[: fallback.index("\n\n")]
    assert "del sys.modules[" in fallback, (
        "the retry reuses the modules the failed import cached, so it fails the same "
        "way"
    )
    assert "sys.path.insert" in fallback
    readme = (REPO / "docs" / "evidence" / "open3d" / "README.md").read_text()
    build = next(line for line in readme.splitlines() if "docker build -f" in line)
    dockerfile = build.split("-f ")[1].split()[0]
    assert (REPO / dockerfile).is_file(), (
        f"the documented build names {dockerfile}, which is not a file"
    )


# --- F7: a hostile published journal is a clean error, not a traceback


@pytest.mark.parametrize(
    "unhashable",
    [
        {"seeds": [[1], [2]]},
        {"seeds": [{}, {}]},
        {"worst_disagreeing_seeds": [{}, {}]},
        {"worst_disagreeing_seeds": [[0], [1]]},
        {"seeds": [0, True]},
    ],
)
def test_a_journal_whose_seeds_are_not_numbers_is_rejected_not_raised_through(
    unhashable: dict,
) -> None:
    """`len(set(seeds))` and `set(pair) <= set(seeds)` called `set()` on list contents
    with no type guard, so a published row carrying `"seeds": [[1], [2]]` raised
    `TypeError: unhashable type` out of `read_journal`. The CLI catches
    `(Open3dError, ValueError)`, so that surfaced as an unhandled traceback and a
    non-contract exit -- the same hole this PR closed for `float(None)` one guard
    further down."""

    block = dict(good_row()["agreement"])
    block.update(unhashable)
    with pytest.raises(Open3dError):
        validate_pair(good_row(agreement=block))


def test_the_rejection_is_not_a_type_error_in_disguise() -> None:
    """A guard that converts the traceback into `Open3dError` only by catching it
    somewhere would still have lost the message."""

    block = dict(good_row()["agreement"])
    block["seeds"] = [[1], [2]]
    with pytest.raises(Open3dError, match="integers"):
        validate_pair(good_row(agreement=block))


# --- F8: the voxel-normalised figure is checkable from the row


def test_a_wrong_voxel_normalisation_is_caught_by_the_published_validator() -> None:
    """`worst_pairwise_displacement_voxels` is the figure the README, the record and
    the counterexample all quote, and the only one in the block the validator could
    not check -- the row did not carry the voxel size, so a stale or wrong
    normalisation passed `validate` silently."""

    block = dict(good_row()["agreement"])
    block["worst_pairwise_displacement_voxels"] = (
        block["worst_pairwise_displacement_voxels"] * 10
    )
    with pytest.raises(Open3dError, match="divided by voxel_size"):
        validate_pair(good_row(agreement=block))


def test_the_normalised_figure_has_to_say_what_it_was_divided_by() -> None:
    block = dict(good_row()["agreement"])
    block.pop("voxel_size")
    with pytest.raises(Open3dError, match="voxel size"):
        validate_pair(good_row(agreement=block))


def test_a_block_with_no_voxel_figure_needs_no_divisor() -> None:
    """A zero voxel size reports no normalised figure rather than dividing, and that
    shape stays legal."""

    block = _agreement(
        [attempt(0, IDENTITY, 0.9, 0.01), attempt(1, shifted(dx=0.2), 0.88, 0.011)],
        [[0.0, 0.0, 0.0]],
        voxel=0.0,
    )
    assert block["worst_pairwise_displacement_voxels"] is None
    assert block["voxel_size"] is None
    validate_pair(good_row(agreement=block))
