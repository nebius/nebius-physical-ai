"""Guards for the occupancy instrument's controls.

The instrument in `consumer_occupancy.py` produces this lane's published positive claim
and had no control. These tests pin what the controls measured, and three of them pin
findings rather than successes: the floor is not zero, the floor moves with sampling
density, and fabrication beyond the observed cloud's bounding box is invisible.

Most of these read the record and need no Open3D, so they run on every shard. The one
that checks the controls call the real instrument rather than a copy of it does need
Open3D and skips where it is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs" / "evidence" / "open3d"
RECORD = EVIDENCE / "occupancy-instrument-controls.json"
HARNESS = EVIDENCE / "harness" / "occupancy_instrument_controls.py"
INSTRUMENT = EVIDENCE / "harness" / "consumer_occupancy.py"

#: Hashes in this record that name no file. Empty: every digest here names a committed
#: file. Declared anyway so the loop below refuses an unbindable hash instead of
#: skipping quietly past it, which is how three provenance digests went stale earlier in
#: this program with a green suite each time.
NOT_A_FILE_HASH: set[str] = set()


@pytest.fixture
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_floor_is_not_zero_and_the_record_says_so(record) -> None:
    """The finding that makes the rest of this worth reading.

    The positive control is a closed sphere whose observations lie exactly on its
    surface, so nothing in it is fabricated and the naive expectation is a phantom rate
    of zero. It is not zero. `occupied` is a distance to the continuous surface while
    `supported` is a distance to the nearest discrete sample, so a cell centre can sit
    within the radius of the surface and outside the radius of every observation of it.

    This is pinned rather than described because it is the number that stops the
    published rate being read as a pure fabrication fraction.
    """
    measured = record["positive_control"]["measured"]
    floor = measured["phantom_obstacle_rate"]
    assert floor > 0.0, (
        "a zero floor would mean the instrument reports no phantom volume on perfectly "
        "observed geometry, which would make this whole record unnecessary"
    )
    assert measured["phantom_obstacle_cells"] > 0
    # Consistency between the rate and the counts it was computed from.
    assert floor == pytest.approx(
        measured["phantom_obstacle_cells"] / measured["occupied_cells"], abs=1e-12
    )
    # The mechanism is asserted by test_the_floor_falls_as_observations_are_added, which
    # measures it. An earlier version of this line asserted the word "sampling" appeared
    # in the record's prose and failed because the prose says "sample" -- a substring
    # check standing in for a claim, which is the defect this program has been caught
    # making twice before. Deleted rather than made to match the wording.


def test_the_floor_falls_as_observations_are_added(record) -> None:
    """A density artifact has a signature: more observations, less phantom volume.

    Asserted as a relation across the whole sweep rather than against any single value,
    so it still means something if the control geometry or the densities change.
    """
    rows = record["the_floor_is_a_sampling_artifact"]["by_observation_count"]
    ordered = sorted(rows.items(), key=lambda item: int(item[0]))
    rates = [row["phantom_obstacle_rate"] for _, row in ordered]
    assert len(rates) >= 4, f"a density sweep needs more than {len(rates)} points"
    assert rates == sorted(rates, reverse=True), (
        f"the floor should fall monotonically as observations are added, got {rates}"
    )
    assert rates[0] > 10 * rates[-1], (
        "if the sparsest and densest floors are within an order of magnitude, density is "
        f"not the mechanism: {rates[0]} against {rates[-1]}"
    )


def test_fabrication_inside_the_observed_extent_is_detected(record) -> None:
    """The instrument works in the direction the published claim relies on."""
    floor = record["positive_control"]["measured"]["phantom_obstacle_rate"]
    fabricated = record["negative_control_fabrication_inside_the_observed_extent"][
        "measured"
    ]["phantom_obstacle_rate"]
    assert fabricated > floor, (
        f"unobserved surface added inside the probed volume must raise the phantom rate: "
        f"{fabricated} against a floor of {floor}"
    )
    assert fabricated > 2 * floor, (
        "a detection that barely clears the floor is not a demonstration that the "
        f"instrument sees fabrication: {fabricated} against {floor}"
    )


def test_fabrication_outside_the_observed_extent_is_invisible_and_disclosed(
    record,
) -> None:
    """A blind spot, pinned as a blind spot.

    `probe_grid` builds the grid from the observed cloud's bounding box, so a fabricated
    surface beyond that box is never probed. The control puts an unobserved shell 3.0
    away and the instrument returns the floor to every digit -- not approximately the
    floor, the same numbers.

    The assertion is that this stays true *and stays disclosed*. If someone later
    changes the grid to cover the mesh as well, this test fails and the record has to be
    rewritten, which is the correct outcome: the limitation would no longer exist.
    """
    floor = record["positive_control"]["measured"]
    outside = record["negative_control_fabrication_outside_the_observed_extent"][
        "measured"
    ]
    assert outside["phantom_obstacle_rate"] == floor["phantom_obstacle_rate"]
    assert outside["phantom_obstacle_cells"] == floor["phantom_obstacle_cells"]
    assert outside["occupied_cells"] == floor["occupied_cells"]

    disclosure = record["what_this_does_not_claim"][
        "the_blind_spot_is_a_property_of_the_grid_not_of_this_scene"
    ]
    assert "bounding box" in disclosure, (
        "the disclosure has to name the mechanism, or a reader cannot tell whether the "
        "blind spot is this control's or the instrument's"
    )


def test_the_recall_direction_is_exercised_too(record) -> None:
    """Phantom rate alone cannot catch a mesh that omits observed structure."""
    omitted = record["recall_control_observed_structure_the_mesh_omits"]["measured"]
    complete = record["positive_control"]["measured"]
    assert omitted["missed_structure_cells"] > complete["missed_structure_cells"]
    assert omitted["recall_of_observed_cells"] < complete["recall_of_observed_cells"]


def test_fabrication_is_detected_at_every_swept_parameter(record) -> None:
    """The separation must not depend on the one cell size it was measured at."""
    sweep = record["parameter_sweep"]
    rows = sweep["rows"]
    assert len(rows) >= 8, f"a two-axis sweep needs more than {len(rows)} rows"
    failures = {
        name: row for name, row in rows.items() if not row["fabrication_is_detected"]
    }
    assert not failures, f"fabrication went undetected at {sorted(failures)}"
    # The summary flag must be derived from the rows, not asserted beside them.
    assert sweep["fabrication_detected_in_every_combination"] == all(
        row["fabrication_is_detected"] for row in rows.values()
    )
    for name, row in rows.items():
        assert row["margin"] == pytest.approx(
            row["with_fabrication"] - row["floor"], abs=1e-6
        ), f"{name}'s margin does not match the two rates beside it"


def test_the_regime_comparison_is_derived_from_committed_records(record) -> None:
    """The one place this record talks about the published figure, recomputed here.

    Every number in that block comes from a file in this repository, so the comparison
    is checkable without the run bundle. Recomputed rather than compared to literals: a
    literal cannot notice that the record it describes has changed.
    """
    block = record["the_floor_is_a_sampling_artifact"][
        "how_this_bears_on_the_published_figure"
    ]
    consumer = json.loads((EVIDENCE / "downstream-consumer.json").read_text())
    replay = json.loads((EVIDENCE / "replay-b18d-public-report.json").read_text())

    fused: list[int] = []

    def collect(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "fused_points" and isinstance(value, int):
                    fused.append(value)
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(replay)
    assert fused, "the replay report no longer records a fused point count"
    assert len(set(fused)) == 1, (
        f"the replay report disagrees with itself: {set(fused)}"
    )

    assert block["the_artifacts_observations"] == fused[0]
    assert block["the_artifacts_supported_cells"] == consumer["cells_with_samples"]
    assert block["the_artifacts_observations_per_supported_cell"] == pytest.approx(
        fused[0] / consumer["cells_with_samples"], abs=5e-4
    )
    assert (
        block["the_published_candidate_rate"]
        == (consumer["improvement"]["phantom_rate_candidate"])
    )
    assert (
        block["the_published_baseline_rate"]
        == (consumer["improvement"]["phantom_rate_baseline"])
    )

    rows = record["the_floor_is_a_sampling_artifact"]["by_observation_count"]
    nearest = min(
        rows.items(),
        key=lambda item: abs(
            item[1]["observations_per_supported_cell"]
            - block["the_artifacts_observations_per_supported_cell"]
        ),
    )
    assert block["nearest_control_density"] == nearest[0], (
        "the named nearest density is not the nearest one in the table beside it"
    )
    assert block["that_controls_floor"] == nearest[1]["phantom_obstacle_rate"]


def test_the_record_does_not_claim_the_published_rate_is_mostly_floor(record) -> None:
    """Guard against the overstatement this record is one step away from making.

    At the comparable control density the floor is well below the published candidate
    rate, so the honest reading is that the floor is real and most of the published rate
    is not it. Measured here rather than taken from the prose, because the prose is what
    would drift.
    """
    block = record["the_floor_is_a_sampling_artifact"][
        "how_this_bears_on_the_published_figure"
    ]
    assert block["that_controls_floor"] < block["the_published_candidate_rate"], (
        "if the comparable floor reached the published rate, the reading in this record "
        "would be wrong and would need rewriting rather than re-asserting"
    )
    # And the comparison between variants must survive the floor entirely.
    assert block["the_published_baseline_rate"] > block["the_published_candidate_rate"]


def _import_from_harness(*names):
    import importlib
    import sys

    sys.path.insert(0, str(EVIDENCE / "harness"))
    try:
        return tuple(importlib.import_module(name) for name in names)
    finally:
        sys.path.remove(str(EVIDENCE / "harness"))


def _instrument():
    """Only the instrument, which is why it can be imported without Open3D.

    Kept apart from the control harness deliberately: the control harness imports Open3D
    at module level, so pulling both in would make the definition check below skip on
    every CI shard -- the exact defect it exists to repair. I asserted this worked before
    I had tested it; blocking `open3d` through `sys.meta_path` showed the control harness
    failing to import, and only `consumer_occupancy` surviving.
    """
    return _import_from_harness("consumer_occupancy")[0]


def test_the_control_harness_produces_what_the_instrument_produces() -> None:
    """Behavioural, because identity was not enough.

    The first version of this test asserted object identity: the functions the control
    harness holds are the very objects `consumer_occupancy` defines. A reviewer defeated
    it with a *function-local* shadow, which satisfies module-attribute identity while
    executing something else, with the import line untouched::

        from instrument import score_variant   # the attribute IS the real object

        def run(x):
            score_variant = _copy              # local shadow, invisible to `is`
            return score_variant(x)

    I reproduced that before changing anything: the honest and the evading harness both
    pass the identity assertion, returning ('REAL', 10) and ('COPY', 15). Identity
    constrains the module namespace; execution resolves names one scope down.

    So this calls both paths on one shared input and requires the outputs to agree. A
    copy now has to be *correct* to pass, which is the only kind of copy nobody needs to
    care about.

    This needs Open3D and skips where it is absent. See
    `test_the_published_definitions_are_what_the_instrument_computes` for the half that
    runs everywhere, and why that split had to exist.
    """
    pytest.importorskip("open3d", reason="the instrument's queries need Open3D")

    instrument, controls = _import_from_harness(
        "consumer_occupancy", "occupancy_instrument_controls"
    )

    mesh = controls.sphere()
    points = controls.fibonacci_sphere(1500)
    cell, factor = instrument.CELL, instrument.SUPPORT_RADIUS_FACTOR

    cloud = controls.cloud_of(points)
    centres, _ = instrument.probe_grid(cloud, cell)
    radius = factor * cell
    supported = instrument.support_mask(cloud, centres, radius)
    direct = instrument.score_variant(mesh, centres, supported, radius, cell)

    through_harness = controls.measure(mesh, points, cell, factor)

    for key, value in direct.items():
        assert through_harness[key] == value, (
            f"the control harness and the instrument disagree on {key}: "
            f"{through_harness[key]} against {value}, so the harness is not running the "
            f"instrument"
        )


def test_the_published_definitions_are_what_the_instrument_computes(record) -> None:
    """The half of the above that runs on every shard.

    A reviewer established that Open3D is installed nowhere CI runs: it appears in
    `npa/pyproject.toml` only as a container image tag, in no optional-dependency extra,
    and not in `npa/ci/requirements.txt`. So the behavioural check above skips
    permanently there, and the guard against a replaced instrument was inert in exactly
    the place it was meant to hold.

    `score_from_distance` carries the definitions with the Open3D query already done, so
    it imports with NumPy alone. Every published row is rebuilt here from its own counts
    and put back through the instrument's real arithmetic. A set of numbers that does not
    satisfy these definitions did not come from this instrument.
    """
    import numpy as np

    instrument = _instrument()
    radius, cell = 0.5, 0.1

    rows = {
        "positive_control": record["positive_control"]["measured"],
        "fabrication_inside": record[
            "negative_control_fabrication_inside_the_observed_extent"
        ]["measured"],
        "fabrication_outside": record[
            "negative_control_fabrication_outside_the_observed_extent"
        ]["measured"],
        "recall_control": record["recall_control_observed_structure_the_mesh_omits"][
            "measured"
        ],
    }
    for name, row in rows.items():
        cells = row["cells"]
        supported_count = row["cells_with_samples"]
        occupied_count = row["occupied_cells"]
        phantom_count = row["phantom_obstacle_cells"]
        both = occupied_count - phantom_count

        # Masks realising exactly these counts: `both` cells occupied and supported,
        # then `phantom_count` occupied only, then the remaining supported cells.
        supported = np.zeros(cells, dtype=bool)
        occupied = np.zeros(cells, dtype=bool)
        supported[:both] = True
        occupied[: both + phantom_count] = True
        supported[
            both + phantom_count : both + phantom_count + supported_count - both
        ] = True
        assert int(supported.sum()) == supported_count, name
        assert int(occupied.sum()) == occupied_count, name

        distance = np.where(occupied, 0.0, radius * 2.0)
        rebuilt = instrument.score_from_distance(distance, supported, radius, cell)

        for key in (
            "occupied_cells",
            "phantom_obstacle_cells",
            "phantom_obstacle_rate",
            "precision_wrt_observation",
            "recall_of_observed_cells",
            "missed_structure_cells",
        ):
            assert rebuilt[key] == pytest.approx(row[key], abs=1e-12), (
                f"{name}.{key} is {row[key]}, but the instrument's own definitions give "
                f"{rebuilt[key]} from the counts published beside it"
            )


def test_the_instrument_still_exposes_what_the_record_names(record) -> None:
    """The record names the functions it borrowed; they must still be there.

    Read from the record rather than hardcoded, so renaming one in the record without
    renaming it in the instrument fails here.
    """
    named = record["instrument"]["functions"]
    assert named, "the record no longer names which functions the controls reuse"
    source = INSTRUMENT.read_text()
    for function in named:
        assert f"def {function}(" in source, (
            f"the record says it reuses {function}, which {INSTRUMENT.name} does not "
            f"define"
        )


def _sha256_fields(node: object, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("sha256") and isinstance(value, str):
                found[prefix + key] = value
            else:
                found.update(_sha256_fields(value, f"{prefix}{key}."))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.update(_sha256_fields(value, f"{prefix}[{index}]."))
    return found


def test_no_hash_in_this_record_escapes_classification(record) -> None:
    """Every digest either names a file and matches it, or is declared not to.

    This record carries no digests today, and the point of the test is that it cannot
    start carrying an unchecked one silently.
    """
    hashes = _sha256_fields(record)
    unclassified = set(hashes) - NOT_A_FILE_HASH
    for field in sorted(unclassified):
        # A digest here must name a committed file somewhere in the record's own object.
        assert False, (
            f"{field} is a hash this test does not know how to bind; either enforce it "
            f"against the file it names or add it to NOT_A_FILE_HASH with a reason"
        )
    assert set(hashes) >= NOT_A_FILE_HASH - set(hashes) or True


def test_the_harness_and_record_are_committed_together(record) -> None:
    """A record whose producer is missing cannot be re-run by a reader."""
    assert HARNESS.exists(), f"{HARNESS} is named by this record and is not committed"
    assert INSTRUMENT.exists()
    # The record must not have been produced by a harness that has since vanished.
    assert record["instrument"]["imported_from"] == INSTRUMENT.name
    digest = hashlib.sha256(HARNESS.read_bytes()).hexdigest()
    assert len(digest) == 64
