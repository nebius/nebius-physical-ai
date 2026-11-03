"""The seed fixes RANSAC's draw and does not make a rerun byte-identical.

`_seeded` claimed it made RANSAC repeatable and the skill told readers a rerun
repeats. Neither held, and the consequence is in the published output: `run_register`
writes an `aligned_sha256` per pair, which reads as a digest two runs can be compared
on, and it differs on every run.

These tests hold the record to its own measurements and hold the product's prose to
the record, because the defect being fixed here was prose that no measurement
supported. The Open3D-dependent test is skipped where Open3D is absent; everything
else reads the committed record, so the claims stay gated in CI either way.
"""

from __future__ import annotations

import json
import os
import pathlib
from pathlib import Path

import pytest
from _pytest.outcomes import Skipped

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
RECORD = EVIDENCE / "register-seed-determinism.json"
SKILL = Path(__file__).resolve().parents[3] / "skills" / "tools" / "open3d" / "SKILL.md"


def _load_harness():
    """Execute the harness module without leaving its `sys.path` edit behind.

    The harness does `sys.path.insert(0, .../npa/src)` at module scope, which loading
    it runs. Left in place that outlives the test and silently shadows an installed
    `npa` for the rest of the session -- the kind of thing that surfaces as a
    different test failing for no visible reason.
    """

    import importlib.util
    import sys

    path = EVIDENCE / "harness" / "register_seed_determinism.py"
    spec = importlib.util.spec_from_file_location("register_seed_determinism", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    return module


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_record_shows_the_pipeline_is_not_byte_reproducible() -> None:
    """The finding itself, in both constructions.

    Asserted as "every repeat differed" rather than a count, so a future run that
    happens to collide twice does not fail a test over it -- but a run where the
    pipeline became reproducible would, and that is a result worth being told about.
    """

    record = _record()
    assert record["seed_held_fixed_at"] is not None
    for label, case in record["cases"].items():
        composed = case["the_composed_pipeline"]
        assert composed["repeats"] >= 8, composed
        assert not composed["bit_identical_every_repeat"], (
            f"{label} reproduced byte-for-byte, which contradicts what this record "
            "and the code's docstring now say; that is a finding, not a pass"
        )


def test_the_answer_is_stable_even_though_the_bytes_are_not() -> None:
    """Both halves, because either alone misleads.

    Without this, the record reads as an accuracy problem. The well-conditioned pair
    is the one that carries the point: its repeats disagree by floating-point noise.
    """

    cases = _record()["cases"]
    well_conditioned = cases["non_periodic_bowl"]["the_composed_pipeline"]
    assert well_conditioned["worst_translation_disagreement_in_voxels"] < 1e-6, (
        "the well-conditioned pair's repeats disagree by more than float noise, "
        "which would make this a different and more serious finding"
    )
    for label, case in cases.items():
        composed = case["the_composed_pipeline"]
        assert composed["worst_translation_disagreement_in_voxels"] < 0.01, (
            f"{label} disagrees by a hundredth of a voxel or more between repeats; "
            "the claim that the answer is stable no longer holds"
        )
        assert composed["worst_rotation_disagreement_degrees"] < 0.01, composed


def test_the_published_metrics_look_stable_while_the_answer_moves() -> None:
    """Why watching `fitness` would not have caught this.

    `icp.fitness` is identical across every repeat in both cases while the
    transformation differs in all of them. A consumer monitoring the reported
    agreement measures sees a perfectly steady pipeline.
    """

    for label, case in _record()["cases"].items():
        composed = case["the_composed_pipeline"]
        assert composed["distinct_icp_fitness_values"] == 1, (
            f"{label} now reports more than one fitness across repeats, so the "
            "contrast this record draws no longer holds"
        )
        assert composed["distinct_results"] > 1, label


def test_every_stage_that_varies_was_handed_identical_input() -> None:
    """The control without which the finding is just a harness feeding bad data."""

    for label, case in _record()["cases"].items():
        for name, cell in case["each_stage_alone"]["icp_from_one_held_pose"].items():
            assert cell["every_repeat_was_handed_identical_input"], (
                f"{label}:{name} cannot attribute its result to ICP, because the "
                "input digest moved between repeats"
            )
            if not cell["bit_identical_every_repeat"]:
                assert cell["the_input_digest"], cell


def test_the_input_digest_cannot_be_fooled_by_regrouping() -> None:
    """A reviewer broke the first version of that control. Pin the fix.

    It concatenated every array into one flat buffer, so points of shape (2, 3)
    beside normals of (2, 3) hashed the same as (1, 3) beside (3, 3). Unreachable
    through the harness, because Open3D returns one normal per point -- but the
    thing that made it unreachable was a third-party library's behaviour rather
    than anything the control asserted.
    """

    # The harness imports Open3D at module scope, so loading it needs Open3D even
    # though the function under test only hashes arrays.
    pytest.importorskip("open3d")
    import numpy as np

    harness = _load_harness()

    same = np.arange(6.0).reshape(2, 3)
    assert harness._digest_all(same, same) != harness._digest_all(
        same[:1], np.arange(9.0).reshape(3, 3)
    ), "the control still collides when the same bytes are regrouped"
    assert harness._digest_all(same, same) == harness._digest_all(same, same)
    assert harness._digest_all(same) != harness._digest_all(same.T), (
        "a transpose carries the same bytes in a different shape and must differ"
    )


def test_the_shipped_estimation_is_the_one_that_varies() -> None:
    """Named specifically, because it decides what a reader should do about it.

    Point-to-point is reproducible; point-to-plane, which is the default, is not.
    An assertion over "some stage varies" would pass if that reversed, and the
    reversal would change the advice.

    Both affected pairs, named individually. The upstream demo scan is the one that
    matters to a reader -- a reviewer found the effect there after I had claimed it
    needed a near-degenerate surface -- and a loop over whichever cases happen to
    vary would let that case drop out silently.
    """

    record = _record()
    for label in ("periodic_ripple", "upstream_demo_scan"):
        stages = record["cases"][label]["each_stage_alone"]
        assert stages["the_shipped_icp_estimation"] == "point_to_plane"
        icp = stages["icp_from_one_held_pose"]
        assert icp["point_to_point"]["bit_identical_every_repeat"], (
            f"{label}: point-to-point varies too, which would make this a property "
            "of ICP rather than of the estimation the product ships"
        )
        assert not icp["point_to_plane"]["bit_identical_every_repeat"], icp[
            "point_to_plane"
        ]
        assert (
            f"{label}:icp_point_to_plane_from_a_held_pose"
            in (record["stages_that_are_not_bit_reproducible_alone"])
        )

    # And the bowl, which this test used to read the other way round. The held-pose
    # cell there is reproducible, and that was taken as the well-conditioned contrast
    # that made the effect a conditioning effect. It is not: that cell ran at fitness
    # 0.0 with zero correspondences, so the call found nothing within the
    # correspondence distance and returned the pose it was handed. A call that does no
    # work is reproducible for reasons that say nothing about the surface.
    bowl = record["cases"]["non_periodic_bowl"]["each_stage_alone"]
    held = bowl["icp_from_one_held_pose"]["point_to_plane"]
    assert held["bit_identical_every_repeat"], held
    assert held["found_no_correspondences_in_some_repeat"], (
        "the bowl's held-pose cell now finds correspondences, so its reproducibility "
        "means something different than it did and this reasoning needs rereading"
    )
    assert held["fitness"]["max"] == 0.0, held["fitness"]
    from_coarse = bowl["icp_from_the_pipelines_own_coarse_pose"]["point_to_plane"]
    assert from_coarse["correspondences"]["min"] > 0, from_coarse
    assert not from_coarse["bit_identical_every_repeat"], (
        "started where the pipeline starts it, the shipped call is reproducible on "
        "the bowl -- which would make conditioning the explanation after all"
    )
    assert (
        "non_periodic_bowl:icp_point_to_plane_from_the_pipelines_coarse_pose"
        in record["stages_that_are_not_bit_reproducible_alone"]
    )


def test_one_thread_is_measured_as_the_remedy_and_the_env_var_is_not() -> None:
    """The actionable half, including the lever that does not work.

    A reader who knows the thread-count finding in `capability-record-support-crop`
    will reach for `OMP_NUM_THREADS` first. It does not reach this call, and the
    record has to say so or the advice is worse than none.
    """

    cells = _record()["how_the_thread_count_participates"]
    assert set(cells) == {"periodic_ripple", "upstream_demo_scan"}, (
        "the remedy is the one actionable instruction the skill gives, and the "
        "capture the product ships against has to be one of the cases it was "
        "measured on"
    )
    for label, threads in cells.items():
        assert threads["the_default_thread_count_here"] > 1, threads

        one = threads["at_1_threads"]
        assert one["point_to_plane_icp_from_a_held_pose"]["bit_identical_every_repeat"]
        assert one["the_whole_register_pipeline"]["bit_identical_every_repeat"], (
            f"{label}: one thread no longer makes the pipeline reproducible, so the "
            "remedy this record offers is no longer true"
        )
        assert not threads["at_2_threads"]["the_whole_register_pipeline"][
            "bit_identical_every_repeat"
        ], f"{label}: two threads reproduced, which would move where the boundary is"

        # The environment variable, measured rather than asserted. This cell was a
        # hard-coded sentence interpolating the parent's own thread count: the one
        # claim in the thread section not produced by the code that printed it, in a
        # record whose subject is prose no measurement supports. Run under the
        # variable on a host where OpenMP honoured it, it would have printed "leaves
        # get_max_threads at the host's 1" as a measured fact.
        env = threads["the_environment_variable_is_not_the_lever"]
        assert env["measured"] is True, env
        assert env["the_childs_get_max_threads"] > 1, env
        assert env["the_variable_reached_the_thread_count"] is False, env
        assert not env["the_whole_register_pipeline"]["bit_identical_every_repeat"], (
            f"{label}: the child run under OMP_NUM_THREADS=1 was reproducible, so the "
            "variable does reach the thread count and the advice is now backwards"
        )


def test_the_prose_a_user_reads_matches_the_record() -> None:
    """The actual defect was prose no measurement supported. Bind them.

    Both the skill and `_seeded` told a reader a rerun repeats. A corrected sentence
    that drifts from the record is the same defect again, so the claim a user reads
    has to name the record and the remedy.
    """

    import npa.workbench.open3d.runner as runner

    skill = SKILL.read_text()
    assert "register-seed-determinism.json" in skill, (
        "the skill states a reproducibility claim without pointing at what measured it"
    )
    assert "set_max_threads(1)" in skill, skill[skill.find("RANSAC is randomized") :][
        :400
    ]
    assert "does **not** make a rerun byte-identical" in skill

    doc = runner._seeded.__doc__ or ""
    assert "never matches the last" in doc, doc
    assert "register-seed-determinism.json" in doc, doc
    assert "repeatable" not in doc.split("This said")[0], (
        "the docstring still leads with the claim this change is correcting"
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_the_measurement_itself_reproduces(periodic: bool) -> None:
    """Re-run the live call, so the record is not the only thing asserting this.

    Four repeats rather than the record's sixteen: the effect is all-or-nothing in
    every cell measured, so this needs only to distinguish one answer from several.
    """

    pytest.importorskip("open3d")
    import numpy as np
    import open3d as o3d

    from npa.workbench.open3d.runner import _register_pair, _seeded
    from npa.workbench.open3d.schemas import RegistrationManifest

    # The variation is parallel accumulation, so on a host that resolves to one thread
    # the repeats *do* agree -- the record measures exactly that, in
    # `how_the_thread_count_participates`. Asserting non-determinism unconditionally
    # turns a correct single-vCPU runner, single-core container or single-threaded
    # Open3D build into a failure. The boundary comes from the record rather than from
    # a number chosen here: it is reproducible at 1 thread and not at 2.
    threads = o3d.utility.get_max_threads()
    if threads <= 1:
        thread_cell = _record()["how_the_thread_count_participates"]["periodic_ripple"]
        assert thread_cell["at_1_threads"]["the_whole_register_pipeline"][
            "bit_identical_every_repeat"
        ], "the record no longer says one thread is reproducible; reread this skip"
        pytest.skip(
            f"Open3D reports {threads} thread(s) here, and the record measures the "
            "pipeline as bit-reproducible at one thread -- there is no "
            "non-determinism to observe on this host"
        )

    rng = np.random.default_rng(7)

    def patch(x0: float) -> np.ndarray:
        x = rng.uniform(x0, x0 + 1.6, 4000)
        y = rng.uniform(0, 1.6, 4000)
        height = (
            0.12 * np.sin(3 * x) * np.cos(3 * y)
            if periodic
            else 0.3 * x * x + 0.2 * y * np.sqrt(x + 1)
        )
        return np.column_stack([x, y, height + rng.normal(0, 0.002, 4000)])

    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(patch(0.0)))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(patch(0.8)))
    manifest = RegistrationManifest.model_validate(
        {
            "schema_version": "npa.open3d.registration.v1",
            "voxel_size": 0.05,
            "random_seed": _record()["seed_held_fixed_at"],
            "fragments": [
                {
                    "id": f"f{index}",
                    "uri": f"s3://constructed/f{index}.ply",
                    "sha256": str(index) * 64,
                    "bytes": index + 1,
                }
                for index in range(2)
            ],
        }
    )

    poses = []
    for _ in range(4):
        _seeded(o3d, manifest)
        measured = _register_pair(
            o3d,
            o3d.geometry.PointCloud(source),
            o3d.geometry.PointCloud(target),
            manifest,
        )
        poses.append(np.array(measured["transformation"]))

    assert len({one.tobytes() for one in poses}) > 1, (
        f"four seeded repeats agreed byte-for-byte at {threads} threads, which "
        "contradicts the record"
    )
    translations = np.array([one[:3, 3] for one in poses])
    worst = max(
        float(np.linalg.norm(a - b)) for a in translations for b in translations
    )
    assert worst / 0.05 < 0.01, (
        f"the repeats disagree by {worst / 0.05} voxels, so the answer is not as "
        "stable as the record claims"
    )


def test_ransac_is_named_everywhere_the_stable_stages_are_listed() -> None:
    """The record measured RANSAC varying and then left it out of every summary.

    `stages_that_are_not_bit_reproducible_alone` listed it; the prose a reader
    actually reads enumerated "normal estimation, FPFH and point-to-point ICP" as the
    stable stages and stopped. A consumer reading only the summary concluded a fixed
    seed makes the RANSAC repeat, which is the sentence this whole change exists to
    remove -- and which survived in two more places.
    """

    record = _record()
    for label in ("periodic_ripple", "upstream_demo_scan"):
        cell = record["cases"][label]["each_stage_alone"]["ransac_global_registration"]
        assert not cell["bit_identical_every_repeat"], cell
        assert (
            f"{label}:ransac_global_registration"
            in (record["stages_that_are_not_bit_reproducible_alone"])
        )

    establishes = record["what_the_cases_establish"]
    assert "RANSAC" in establishes, establishes

    readme = (EVIDENCE / "README.md").read_text()
    section = readme[readme.index("The seed is not what is loose") :][:3000]
    assert "RANSAC" in section, section[:400]


def test_the_two_other_places_that_claimed_a_seeded_rerun_repeats() -> None:
    """Four locations held the sentence; the change corrected two of them.

    `schemas.py` is the input contract a caller reads to find out what `random_seed`
    buys, and the golden-eval docstring is what a reader consults when the smoke
    tolerances are questioned. Both still said RANSAC repeats bit-for-bit because the
    generator is seeded, which this record measures as false.
    """

    from npa.workbench.open3d import schemas

    contract = pathlib.Path(schemas.__file__).read_text()
    assert "so a rerun repeats the same RANSAC" not in " ".join(contract.split())
    assert "register-seed-determinism.json" in contract, (
        "the corrected sentence has to name the measurement, or it is one more claim "
        "a reader cannot check"
    )

    # Read as source: importing the golden-eval module pulls in its Open3D-dependent
    # fixtures, and the claim is in a docstring either way.
    smoke = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "npa"
        / "smoke"
        / "test_open3d_functional.py"
    ).read_text()
    flat = " ".join(smoke.split())
    assert "repeat bit-for-bit because the generator is seeded" not in flat, (
        "the claim wraps across lines in the source, so this has to be checked "
        "against whitespace-flattened text -- the first version of this assertion "
        "could not have matched the sentence it was written to catch"
    )
    assert "register-seed-determinism.json" in smoke


def test_the_record_says_which_published_figures_move() -> None:
    """`fitness` holding still is the point; the README generalised it to all of them.

    It read "a consumer watching the reported agreement measures sees a perfectly
    steady pipeline". `inlier_rmse` is published per pair by the same `_result_summary`
    and takes a different value on almost every repeat, so the sentence overstated how
    blind a consumer is -- in the artefact whose purpose is to stop claims drifting
    from measurements.
    """

    record = _record()
    moving = {
        label: case["the_composed_pipeline"]["distinct_icp_inlier_rmse_values"]
        for label, case in record["cases"].items()
    }
    assert all(count > 1 for count in moving.values()), moving
    steady = {
        label: case["the_composed_pipeline"]["distinct_icp_fitness_values"]
        for label, case in record["cases"].items()
    }
    assert all(count == 1 for count in steady.values()), steady

    readme = (EVIDENCE / "README.md").read_text()
    assert "icp.inlier_rmse" in readme, (
        "the README names fitness as the steady figure and has to name the moving one"
    )


def test_the_record_is_bound_to_the_harness_that_produced_it() -> None:
    """Named in prose, the audit could confirm the file exists and nothing more.

    This directory's guard binds a `sha256` sitting beside a `harness` key, and the
    directory's README documents unbound-reference drift as a recurring failure here.
    What follows from binding is that editing the harness without regenerating is a
    finding rather than something a reader has to notice.
    """

    import hashlib

    record = _record()
    produced = record["produced_by"]
    assert produced["harness"] == "harness/register_seed_determinism.py"
    harness = EVIDENCE / "harness" / "register_seed_determinism.py"
    assert produced["sha256"] == hashlib.sha256(harness.read_bytes()).hexdigest(), (
        "the record was not regenerated after the harness changed, which is exactly "
        "what this binding exists to catch"
    )


def test_every_icp_cell_publishes_how_much_work_it_did() -> None:
    """Without this a reproducible cell cannot be told from a call that did nothing.

    Which is not hypothetical: the bowl's held-pose cell was read as the
    well-conditioned contrast for a month, and it was a call with zero
    correspondences returning its own start pose.
    """

    for case in _record()["cases"].values():
        stages = case["each_stage_alone"]
        for group in (
            "icp_from_one_held_pose",
            "icp_from_the_pipelines_own_coarse_pose",
        ):
            for estimation, cell in stages[group].items():
                if not isinstance(cell, dict):
                    continue
                for field in ("fitness", "correspondences"):
                    assert set(cell[field]) == {"min", "max", "distinct_values"}, (
                        f"{group}.{estimation}.{field}"
                    )
                assert cell["the_start_pose"], f"{group}.{estimation}"
                assert isinstance(cell["found_no_correspondences_in_some_repeat"], bool)


def test_loading_the_harness_leaves_the_import_path_alone() -> None:
    """The side effect, asserted rather than described.

    Removing the restore used to change nothing any test could see, which is the same
    shape as the defects this record is about: a correction with nothing holding it.
    """

    import sys

    pytest.importorskip("open3d")
    before = list(sys.path)
    harness = _load_harness()
    assert harness._digest_all is not None
    assert sys.path == before, (
        "the harness's module-level sys.path.insert outlived the load, so `npa` now "
        "resolves from the repo tree ahead of anything installed for the rest of "
        "this session"
    )


def _assert_the_gate_skips(o3d) -> None:
    """Fake one thread, require a skip, and prove the fake was undone.

    Shared with the child process below so that the same assertions run against a
    real one-thread Open3D and against a monkeypatch on a many-thread host.
    """

    original_getter = o3d.utility.get_max_threads
    original_count = original_getter()

    monkeypatched = pytest.MonkeyPatch()
    try:
        monkeypatched.setattr(o3d.utility, "get_max_threads", lambda: 1)
        with pytest.raises(Skipped) as skipped:
            test_the_measurement_itself_reproduces(True)
    finally:
        monkeypatched.undo()

    assert "no non-determinism to observe" in str(skipped.value), skipped.value

    # Against the state captured above, not against a thread count this host is
    # assumed to have. `assert get_max_threads() > 1` was the first version, and it
    # asserted the host rather than the restoration -- the same defect this gate
    # exists to prevent, reintroduced in the gate itself. It fails on a legitimate
    # one-thread host, where `original_count` is 1 and nothing is wrong.
    assert o3d.utility.get_max_threads is original_getter, (
        "the monkeypatch outlived the test: the getter was not restored"
    )
    assert o3d.utility.get_max_threads() == original_count, (
        "the getter was restored but reports a different count than before the patch"
    )


def test_the_non_determinism_test_skips_rather_than_fails_at_one_thread() -> None:
    """The gate, exercised. This host has many threads, so removing it is invisible.

    The record measures the pipeline as bit-reproducible at one thread, so asserting
    non-determinism unconditionally turns a correct single-vCPU runner or
    single-threaded Open3D build into a red pipeline. That cannot be proved by
    running on a many-thread host, so the thread count is faked and the outcome
    checked: a skip, not a failure.
    """

    o3d = pytest.importorskip("open3d")
    _assert_the_gate_skips(o3d)


def test_the_gate_holds_on_a_real_one_thread_open3d() -> None:
    """The control the monkeypatch cannot give: `set_max_threads(1)` for real.

    `o3d.utility.set_max_threads` is process-global, so it runs in a child rather
    than inside this session. The child is the whole point of the control -- a
    restoration assertion written against an assumed many-thread host passes here
    and fails on a single-vCPU runner, which is where this gate matters.
    """

    pytest.importorskip("open3d")

    import subprocess
    import sys

    child = subprocess.run(  # noqa: S603 - this interpreter, a literal argv
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--one-thread-child"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )

    assert child.returncode == 0, (
        f"the gate does not hold at one real thread:\n{child.stdout}\n{child.stderr}"
    )
    assert "skipped at 1 real thread(s)" in child.stdout, child.stdout


def _one_thread_child() -> int:
    """Run the gate's assertions under a real one-thread Open3D."""

    import open3d as o3d

    o3d.utility.set_max_threads(1)
    threads = o3d.utility.get_max_threads()
    if threads != 1:
        print(f"this build ignores set_max_threads: reports {threads}")
        return 0 if threads > 0 else 1

    _assert_the_gate_skips(o3d)
    print(f"skipped at {threads} real thread(s), and the getter was restored to it")
    return 0


def test_the_other_direction_runs_the_measurement_rather_than_skipping() -> None:
    """The normal-thread control: with the real getter, the gate must not skip.

    Both halves are needed. A gate that always skipped would satisfy the test above
    and measure nothing, which is the failure mode this PR's own record documents
    for the bowl's ICP cell.
    """

    o3d = pytest.importorskip("open3d")

    if o3d.utility.get_max_threads() <= 1:
        pytest.skip("this host really has one thread; the skip above is the behaviour")

    test_the_measurement_itself_reproduces(True)


def test_the_skill_scopes_the_remedy_to_the_cases_it_was_measured_on() -> None:
    """The one actionable instruction the skill gives, held to what was measured.

    It read "`o3d.utility.set_max_threads(1)` makes the whole pipeline
    bit-reproducible" while the cell ran only on a constructed ripple -- and the
    record's own argument is that the demo capture is the one to read first, because
    it is what every accepted record in that directory is built from. The same bullet
    told a reader `OMP_NUM_THREADS` does not reach it, from a hard-coded sentence.
    """

    # Flattened, because the bullet wraps: the same trap as the golden-eval docstring
    # above, where a flat substring check could not match the sentence it was written
    # to catch.
    skill = SKILL.read_text()
    bullet = " ".join(
        next(
            block for block in skill.split("\n- ") if "set_max_threads(1)" in block
        ).split()
    )
    assert "both cases measured" in bullet, bullet
    assert "demo capture" in bullet, bullet
    assert "child process" in bullet, (
        "the environment-variable claim is a measurement now, and the advice should "
        "say so rather than reading as received wisdom"
    )
    assert "RANSAC" in bullet, (
        "a fixed seed does not make the RANSAC repeat either, which is the sentence "
        "this change set out to remove"
    )
    assert "not a recommendation" in bullet or "trade" in bullet, bullet


if __name__ == "__main__":
    import sys

    if "--one-thread-child" in sys.argv:
        raise SystemExit(_one_thread_child())
    raise SystemExit(
        "this module is a pytest file; --one-thread-child is its only mode"
    )
