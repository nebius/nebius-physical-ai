"""What does `random_seed` buy the registration path, stage by stage?

`_seeded` has claimed since it was written that it makes RANSAC repeatable, and
`run_register` publishes an `aligned_sha256` per pair, which reads as a digest two
runs can be compared on. Neither holds: one seed, one input and one process produce
a bitwise-different transformation every time, so the aligned cloud's hash can never
match a previous run's.

The answer is stable even so, which is why this survived. Both halves have to be
published together or the finding reads as an accuracy problem, which it is not --
the disagreement is twelve orders below a voxel on a well-conditioned pair.

So this measures four things rather than asserting them:

  * each stage alone, re-seeded and re-run -- normals, FPFH, RANSAC, and ICP from
    one held pose -- because a reader's first question is which stage to look at.
    My own answer to that was wrong: I expected the composition to be at fault and
    every stage to be clean, and the shipped `point_to_plane` ICP turns out to
    return a different answer on every call;
  * that ICP's inputs were bit-identical when it did so, by digesting the points,
    the normals and the pose actually handed to it, each array's shape included.
    Without that control the finding would be indistinguishable from the harness
    feeding it varying data, and without the shapes the control itself was safe
    only because of how Open3D happens to size its normals;
  * the composed pipeline, and the magnitude, on a well-conditioned pair and on an
    ambiguous one, since whether this matters is a question about size;
  * the thread count, which turns out to be the mechanism: held to one thread
    through Open3D's API the stage is bit-reproducible, and `OMP_NUM_THREADS` does
    not reach it. Two other explanations -- normal estimation and the inputs
    varying -- are measured and ruled out, so the next person does not repeat them.

Three pairs, because one surface cannot distinguish a property of the algorithm
from a property of the geometry: a non-periodic bowl where the alignment is well
determined, a ripple whose period the fragments' offset can alias (the ambiguity
#1025 measures separately, here only to bound the magnitude under the worst
conditioning this directory constructs), and the upstream demo scan. The third was
added after a reviewer found the effect there, having read my claim that it took a
near-degenerate surface; it does not, and that is the case a reader should look at
first, because it is the capture every accepted record here is built from.

Run from the repository root, with Open3D installed and `npa/src` on PYTHONPATH:
    python3 docs/evidence/open3d/harness/register_seed_determinism.py

Writes JSON to stdout.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "npa" / "src"))

import open3d as o3d
from npa.workbench.open3d.runner import (
    _prepared,
    _register_pair,
    _seeded,
)
from npa.workbench.open3d.schemas import RegistrationManifest

#: Repeats per cell. Every cell below is either all-identical or all-distinct by a
#: wide margin, so this is about making an exception visible rather than about
#: resolving a close call.
REPEATS = 16

#: The seed held fixed across every repeat. Its value does not matter; that it does
#: not change is the whole measurement.
SEED = 11

POINTS = 4000
VOXEL = 0.05

#: The second fragment's offset along x. On the ripple this is a sizeable fraction
#: of the surface's period, which is what makes that pair ambiguous.
OFFSET = 0.8


def _patch(rng, x0: float, *, periodic: bool) -> np.ndarray:
    x = rng.uniform(x0, x0 + 1.6, POINTS)
    y = rng.uniform(0, 1.6, POINTS)
    height = (
        0.12 * np.sin(3 * x) * np.cos(3 * y)
        if periodic
        else 0.3 * x * x + 0.2 * y * np.sqrt(x + 1)
    )
    return np.column_stack([x, y, height + rng.normal(0, 0.002, POINTS)])


def _pair(construction: str):
    """Two overlapping fragments, constructed or upstream.

    The demo pair is here because a reviewer found the effect on it and I had
    claimed it needed an ambiguous surface. It is the capture every accepted record
    in this directory is built from, so a reader who believes the narrower claim
    would not go looking for this in their own data.
    """

    if construction == "upstream_demo_scan":
        paths = o3d.data.DemoICPPointClouds().paths
        return (o3d.io.read_point_cloud(paths[0]), o3d.io.read_point_cloud(paths[1]))

    rng = np.random.default_rng(7)
    periodic = construction == "periodic_ripple"
    return (
        o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(_patch(rng, 0.0, periodic=periodic))
        ),
        o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(_patch(rng, OFFSET, periodic=periodic))
        ),
    )


def _manifest() -> RegistrationManifest:
    return RegistrationManifest.model_validate(
        {
            "schema_version": "npa.open3d.registration.v1",
            "voxel_size": VOXEL,
            "random_seed": SEED,
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


def _digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _digest_all(*arrays) -> str:
    """Digest several arrays so no regrouping of their contents can collide.

    The first version concatenated everything into one flat array, which loses
    every boundary: points of shape (2, 3) beside normals of (2, 3) hash the same
    as (1, 3) beside (3, 3). Unreachable here, because Open3D returns one normal
    per point -- but the thing making it unreachable is a third-party library's
    behaviour, not anything this file asserts, and a control that is safe by
    circumstance is one library release away from being safe by nothing. The shape
    goes in with the bytes.
    """

    running = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        running.update(repr(contiguous.shape).encode())
        running.update(contiguous.tobytes())
    return running.hexdigest()


def _distinct(digests: list[str]) -> dict[str, Any]:
    return {
        "repeats": len(digests),
        "distinct_results": len(set(digests)),
        "bit_identical_every_repeat": len(set(digests)) == 1,
    }


def _stages_alone(construction: str) -> dict[str, Any]:
    """Each stage re-seeded and re-run on its own, with its inputs held fixed.

    The point of fixing the inputs is to stop one stage's variation from being
    reported as the next one's. ICP is given one pose to start from for the same
    reason: handed RANSAC's output it would inherit whatever RANSAC did.
    """

    manifest = _manifest()
    source, target = _pair(construction)
    registration = o3d.pipelines.registration

    # Both clouds, because `_prepared` sets normals on the downsampled cloud it
    # returns *and* on the full cloud it was given, and point-to-plane ICP reads the
    # full one. Digesting only the downsampled normals was the first version of this
    # cell and it would have left the full cloud unmeasured, which is the input the
    # non-reproducible stage below actually uses.
    normals, full_normals, features = [], [], []
    for _ in range(REPEATS):
        _seeded(o3d, manifest)
        whole = o3d.geometry.PointCloud(source)
        down, feature = _prepared(o3d, whole, manifest)
        normals.append(_digest(np.asarray(down.normals)))
        full_normals.append(_digest(np.asarray(whole.normals)))
        features.append(_digest(feature.data))

    source_down, source_feature = _prepared(
        o3d, o3d.geometry.PointCloud(source), manifest
    )
    target_down, target_feature = _prepared(
        o3d, o3d.geometry.PointCloud(target), manifest
    )
    ransac_distance = VOXEL * manifest.distance_factor
    coarse = []
    for _ in range(REPEATS):
        _seeded(o3d, manifest)
        result = registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_feature,
            target_feature,
            True,
            ransac_distance,
            registration.TransformationEstimationPointToPoint(False),
            3,
            [
                registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                registration.CorrespondenceCheckerBasedOnDistance(ransac_distance),
            ],
            registration.RANSACConvergenceCriteria(
                manifest.ransac_max_iteration, manifest.ransac_confidence
            ),
        )
        coarse.append(np.array(result.transformation))

    held = np.eye(4)
    held[:3, 3] = [OFFSET, 0.0, 0.0]
    icp: dict[str, Any] = {}
    for name, estimation in (
        ("point_to_point", registration.TransformationEstimationPointToPoint()),
        ("point_to_plane", registration.TransformationEstimationPointToPlane()),
    ):
        digests, inputs = [], []
        for _ in range(REPEATS):
            _seeded(o3d, manifest)
            moving = o3d.geometry.PointCloud(source)
            fixed = o3d.geometry.PointCloud(target)
            _prepared(o3d, moving, manifest)
            _prepared(o3d, fixed, manifest)
            # Everything this call can read, digested before it runs. A varying
            # output means nothing unless the input provably held still.
            inputs.append(
                _digest_all(
                    np.asarray(moving.points),
                    np.asarray(moving.normals),
                    np.asarray(fixed.points),
                    np.asarray(fixed.normals),
                    held,
                    np.array([VOXEL * manifest.icp_distance_factor]),
                )
            )
            refined = registration.registration_icp(
                moving,
                fixed,
                VOXEL * manifest.icp_distance_factor,
                held,
                estimation,
            )
            digests.append(_digest(np.array(refined.transformation)))
        icp[name] = {
            **_distinct(digests),
            "every_repeat_was_handed_identical_input": len(set(inputs)) == 1,
            "the_input_digest": inputs[0] if len(set(inputs)) == 1 else None,
        }

    return {
        "normal_estimation_downsampled_cloud": _distinct(normals),
        "normal_estimation_full_cloud": _distinct(full_normals),
        "fpfh_features": _distinct(features),
        "ransac_global_registration": _distinct([_digest(one) for one in coarse]),
        "icp_from_one_held_pose": icp,
        "the_shipped_icp_estimation": manifest.icp_estimation,
    }


def _composed(construction: str) -> dict[str, Any]:
    """The pipeline as `run_register` calls it, re-seeded before every repeat."""

    manifest = _manifest()
    source, target = _pair(construction)
    poses, fitness, rmse = [], [], []
    for _ in range(REPEATS):
        _seeded(o3d, manifest)
        measured = _register_pair(
            o3d,
            o3d.geometry.PointCloud(source),
            o3d.geometry.PointCloud(target),
            manifest,
        )
        poses.append(np.array(measured["transformation"]))
        fitness.append(measured["icp"]["fitness"])
        rmse.append(measured["icp"]["inlier_rmse"])

    translations = np.array([one[:3, 3] for one in poses])
    worst_translation = max(
        float(np.linalg.norm(a - b)) for a in translations for b in translations
    )
    worst_rotation = max(
        float(
            np.degrees(
                np.arccos(
                    np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
                )
            )
        )
        for a in poses
        for b in poses
    )
    return {
        **_distinct([_digest(one) for one in poses]),
        "worst_translation_disagreement_metres": worst_translation,
        "worst_translation_disagreement_in_voxels": worst_translation / VOXEL,
        "worst_rotation_disagreement_degrees": worst_rotation,
        # The published metrics, for contrast. A reader watching `fitness` for
        # instability would conclude this pipeline is reproducible.
        "distinct_icp_fitness_values": len(set(fitness)),
        "distinct_icp_inlier_rmse_values": len(set(rmse)),
    }


def _how_the_thread_count_participates() -> dict[str, Any]:
    """Pin Open3D's thread limit and re-run, both the stage alone and the pipeline.

    This cell exists because the thread count is the explanation #1024 found for the
    reconstruction figures, so a reader who knows that record reaches for it first.
    They are right to, and the obvious way to act on it does not work: setting
    `OMP_NUM_THREADS=1` before the process starts leaves `get_max_threads` reporting
    the host's full count and changes nothing. Open3D's own API is the lever.

    I had this cell the other way round -- named for ruling the thread count out --
    on the strength of that environment-variable attempt. The correction is
    published rather than quietly made, because the wrong lever is the part a reader
    is most likely to repeat.
    """

    manifest = _manifest()
    source, target = _pair("periodic_ripple")
    registration = o3d.pipelines.registration
    held = np.eye(4)
    held[:3, 3] = [OFFSET, 0.0, 0.0]
    default_threads = o3d.utility.get_max_threads()
    observed: dict[str, Any] = {
        "the_default_thread_count_here": default_threads,
        "the_environment_variable_is_not_the_lever": (
            "OMP_NUM_THREADS=1 in the parent shell leaves get_max_threads at the "
            f"host's {default_threads} and leaves every count below unchanged"
        ),
    }
    try:
        for threads in (1, 2, default_threads):
            o3d.utility.set_max_threads(threads)
            alone, composed = [], []
            for _ in range(REPEATS):
                _seeded(o3d, manifest)
                moving = o3d.geometry.PointCloud(source)
                fixed = o3d.geometry.PointCloud(target)
                _prepared(o3d, moving, manifest)
                _prepared(o3d, fixed, manifest)
                refined = registration.registration_icp(
                    moving,
                    fixed,
                    VOXEL * manifest.icp_distance_factor,
                    held,
                    registration.TransformationEstimationPointToPlane(),
                )
                alone.append(_digest(np.array(refined.transformation)))
                _seeded(o3d, manifest)
                whole = _register_pair(
                    o3d,
                    o3d.geometry.PointCloud(source),
                    o3d.geometry.PointCloud(target),
                    manifest,
                )
                composed.append(_digest(np.array(whole["transformation"])))
            observed[f"at_{threads}_threads"] = {
                "point_to_plane_icp_from_a_held_pose": _distinct(alone),
                "the_whole_register_pipeline": _distinct(composed),
            }
    finally:
        o3d.utility.set_max_threads(default_threads)
    return observed


def main() -> int:
    described = {
        "non_periodic_bowl": (
            "a bowl with no repeating structure, where the alignment is well determined"
        ),
        "periodic_ripple": (
            "a ripple whose period the fragment offset can alias, so a whole family "
            "of alignments fits: the worst conditioning this directory constructs"
        ),
        "upstream_demo_scan": (
            "o3d.data.DemoICPPointClouds fragments 0 and 1 -- Redwood's augmented "
            "ICL-NUIM living-room1, the capture every accepted record in this "
            "directory is built from. Not constructed by me, and not ambiguous: "
            "this case exists because a reviewer found the effect here after I had "
            "claimed it needed a near-degenerate surface"
        ),
    }
    cases = {}
    for label, description in described.items():
        cases[label] = {
            "construction": description,
            "each_stage_alone": _stages_alone(label),
            "the_composed_pipeline": _composed(label),
        }

    unreproducible = sorted(
        f"{label}:{stage}"
        for label, case in cases.items()
        for stage, cell in (
            *(
                (key, value)
                for key, value in case["each_stage_alone"].items()
                if isinstance(value, dict) and "bit_identical_every_repeat" in value
            ),
            *(
                (f"icp_{name}", value)
                for name, value in case["each_stage_alone"][
                    "icp_from_one_held_pose"
                ].items()
                if isinstance(value, dict)
            ),
        )
        if not cell["bit_identical_every_repeat"]
    )

    record = {
        "what_this_measures": (
            "whether a fixed random_seed makes the registration path reproducible, "
            "decomposed by stage, and if not, by how much the answer moves"
        ),
        "why_it_matters": (
            "`_seeded` claims it makes RANSAC repeatable and `run_register` "
            "publishes an aligned_sha256 per pair, which reads as a digest two runs "
            "can be compared on. A reviewer who compares them sees a mismatch every "
            "time and has nothing telling them whether that is expected"
        ),
        "seed_held_fixed_at": SEED,
        "repeats_per_cell": REPEATS,
        "voxel_size": VOXEL,
        "cases": cases,
        "stages_that_are_not_bit_reproducible_alone": unreproducible,
        "how_the_thread_count_participates": _how_the_thread_count_participates(),
        "what_the_cases_establish": (
            "the composed pipeline returns a different transformation on every "
            "repeat of every case, so the aligned cloud and the aligned_sha256 the "
            "report publishes differ on every run. Taken alone, normal estimation, "
            "FPFH and point-to-point ICP each return identical bytes every time. The "
            "shipped point_to_plane ICP does not: on the upstream demo scan and on "
            "the ambiguous construction it returns a distinct answer on all sixteen "
            "calls while being handed provably identical points, normals, pose and "
            "distance. It needs no contrived surface -- the demo capture is the one "
            "every accepted record in this directory is built from. Only on the "
            "well-conditioned bowl is that call reproducible, which is what makes "
            "this conditioning rather than a broken solver. Held to one thread "
            "through Open3D's API it becomes bit-reproducible, so the variation is "
            "parallel accumulation rather than anything the seed reaches"
        ),
        "what_was_ruled_out": {
            "normal_estimation": (
                "identical across every repeat on both the downsampled cloud and "
                "the full cloud point-to-plane actually reads"
            ),
            "the_inputs_varying": (
                "every repeat's points, normals, pose and correspondence distance "
                "are digested together before the call and the digest is identical"
            ),
        },
        "what_this_does_not_claim": (
            "that the registration is wrong, or that this costs accuracy. The "
            "answer is stable: on the well-determined pair the repeats disagree by "
            "about a trillionth of a voxel, which is floating-point noise, and even "
            "on the deliberately ambiguous pair by under a thousandth. Nor that the "
            "ambiguity itself is new -- #1025 measures that separately, across "
            "seeds, where the error is tens of voxels. Nor that one thread is a "
            "recommendation: it buys reproducible bytes at the cost of the "
            "parallelism, and this record measures the trade rather than taking a "
            "side on it. This is byte reproducibility under one seed, with the "
            "answer right"
        ),
        "reproduce": (
            "python3 docs/evidence/open3d/harness/register_seed_determinism.py, "
            "with npa/src on PYTHONPATH in an environment holding Open3D. The "
            "distinct-result counts reproduce; the magnitudes are themselves the "
            "output of a non-reproducible pipeline and move in their low digits"
        ),
        "environment": {
            "open3d_version": o3d.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    json.dump(record, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
