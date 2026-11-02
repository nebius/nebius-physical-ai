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
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
RECORD = EVIDENCE / "register-seed-determinism.json"
SKILL = Path(__file__).resolve().parents[3] / "skills" / "tools" / "open3d" / "SKILL.md"


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


def test_the_shipped_estimation_is_the_one_that_varies() -> None:
    """Named specifically, because it decides what a reader should do about it.

    Point-to-point is reproducible; point-to-plane, which is the default, is not.
    An assertion over "some stage varies" would pass if that reversed, and the
    reversal would change the advice.
    """

    record = _record()
    ambiguous = record["cases"]["periodic_ripple"]["each_stage_alone"]
    assert ambiguous["the_shipped_icp_estimation"] == "point_to_plane"
    icp = ambiguous["icp_from_one_held_pose"]
    assert icp["point_to_point"]["bit_identical_every_repeat"], icp["point_to_point"]
    assert not icp["point_to_plane"]["bit_identical_every_repeat"], icp[
        "point_to_plane"
    ]
    assert (
        "periodic_ripple:icp_point_to_plane"
        in (record["stages_that_are_not_bit_reproducible_alone"])
    )

    # And on the well-conditioned pair the same call is exactly reproducible, which
    # is what makes this a conditioning effect rather than a broken solver.
    bowl = record["cases"]["non_periodic_bowl"]["each_stage_alone"]
    assert bowl["icp_from_one_held_pose"]["point_to_plane"][
        "bit_identical_every_repeat"
    ], bowl["icp_from_one_held_pose"]["point_to_plane"]


def test_one_thread_is_measured_as_the_remedy_and_the_env_var_is_not() -> None:
    """The actionable half, including the lever that does not work.

    A reader who knows the thread-count finding in `capability-record-support-crop`
    will reach for `OMP_NUM_THREADS` first. It does not reach this call, and the
    record has to say so or the advice is worse than none.
    """

    threads = _record()["how_the_thread_count_participates"]
    assert threads["the_default_thread_count_here"] > 1, threads
    assert "OMP_NUM_THREADS" in threads["the_environment_variable_is_not_the_lever"]

    one = threads["at_1_threads"]
    assert one["point_to_plane_icp_from_a_held_pose"]["bit_identical_every_repeat"]
    assert one["the_whole_register_pipeline"]["bit_identical_every_repeat"], (
        "one thread no longer makes the pipeline reproducible, so the remedy this "
        "record offers is no longer true"
    )
    assert not threads["at_2_threads"]["the_whole_register_pipeline"][
        "bit_identical_every_repeat"
    ], "two threads reproduced, which would move where the boundary is"


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
        "four seeded repeats agreed byte-for-byte, which contradicts the record"
    )
    translations = np.array([one[:3, 3] for one in poses])
    worst = max(
        float(np.linalg.norm(a - b)) for a in translations for b in translations
    )
    assert worst / 0.05 < 0.01, (
        f"the repeats disagree by {worst / 0.05} voxels, so the answer is not as "
        "stable as the record claims"
    )
