"""Two accepted capability records explain run-to-run variation. Are those the causes?

Both statements are limitations, which is the honest place for them, and both name a
mechanism a reader can act on:

`capability-record-distance-bands.json` `limitations[3]`:

    The Armadillo figure moves slightly between runs (0.2009 here, 0.2055 in the earlier
    threshold sweep) because the 300k-point sampling is not seeded. [...] the fourth
    decimal place is not reproducible.

`capability-record-support-crop.json` `limitations[3]`:

    Floating-point results vary with container thread count; reproducibility is numeric
    to ~4e-06 relative on surface area, not byte-identical.

A reader acts on each of these the same way: seed the sampling, or pin the thread count,
and expect the numbers to settle. Neither works, and that was not knowable when these
were written -- the reconstruction's own non-determinism was measured later, in
`poisson-figure-reproducibility.json`.

What this measures, on the construction the first record names, is both figures at once:
the unsupported-area fraction that record is about, and the mesh surface area the second
record is about, from the same meshes. Three conditions, so the two causes can be
separated from the residue:

- sampling unseeded, which is what both records ran
- sampling seeded, with the cloud's bytes hashed to prove it really is the same input
- sampling seeded and `o3d.utility.set_max_threads(1)`, with `get_max_threads()` recorded
  before and during, because an earlier attempt of mine set `OMP_NUM_THREADS` in a build
  that links no OpenMP and measured nothing

Run (about ten minutes, three conditions of four reconstructions):
    python docs/evidence/open3d/harness/declared_reproducibility_causes.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPEATS = 4
SAMPLES = 300_000
SEED = 1729
POISSON_DEPTH = 9
DENSITY_QUANTILE = 0.02

#: Transcribed from the two records at the head this was written against.
DISTANCE_BANDS_CLAIM = (
    "The Armadillo figure moves slightly between runs (0.2009 here, 0.2055 in the "
    "earlier threshold sweep) because the 300k-point sampling is not seeded. The "
    "separation being measured is three orders of magnitude, so this does not affect "
    "the finding, but the fourth decimal place is not reproducible."
)
SUPPORT_CROP_CLAIM = (
    "Floating-point results vary with container thread count; reproducibility is "
    "numeric to ~4e-06 relative on surface area, not byte-identical."
)
RECORDED_FRACTIONS = (0.2009, 0.2055)
RECORDED_SURFACE_AREA_BOUND = 4e-06
#: The record naming the mechanism, which lands in a separate pull request. Both the
#: disclosure above and the guard in the tests key off whether this file is present.
CITED_RECORD = "poisson-figure-reproducibility.json"

#: Earlier runs of this same harness on this host. Recorded because a spread measured
#: over four repeats is itself unstable, and a sentence derived from one run of it can
#: be wrong in the next -- which happened here. The first draft said the unseeded spread
#: "already contains" the record's 0.0046 gap, true of the first two runs (5.605e-03 and
#: 5.298e-03) and false of the third (3.645e-03). Keeping the observations makes the
#: comparison an order-of-magnitude one, which is all four repeats can support.
EARLIER_RUNS = {
    "unseeded_unsupported_area_fraction_absolute_spread": [
        5.605e-03,
        5.298e-03,
        3.645e-03,
        0.002235571773337275,
        0.004882696105666856,
        0.0009468462263132194,
    ],
    "fixed_input_surface_area_relative_spread": [
        2.275e-06,
        2.424e-06,
        3.454e-06,
        2.9766904159109415e-06,
        3.555532551652766e-06,
        2.2903013427507294e-06,
    ],
    "one_thread_surface_area_relative_spread": [
        3.915e-06,
        4.455e-06,
        4.427e-06,
        2.0900294400293636e-06,
        3.469837523287435e-06,
        2.28332400462555e-06,
    ],
    #: Retained from the third run onwards. A reviewer went looking for 2.3e-05 -- a
    #: figure the first corrected sentence quoted -- and could not find it here, because
    #: this series was not being kept while the other three were. It was run two's
    #: value. Two runs' worth is therefore missing from this list and not reconstructed,
    #: rather than back-filled from memory into something that reads like a measurement.
    "seeded_unsupported_area_fraction_absolute_spread": [
        2.106e-05,
        2.341e-05,
        5.3443300723765574e-06,
        1.6251176341935958e-05,
    ],
}
#: The series above that were kept from the first run, against the one that was not.
SERIES_RETAINED_FROM_THE_FIRST_RUN = (
    "unseeded_unsupported_area_fraction_absolute_spread",
    "fixed_input_surface_area_relative_spread",
    "one_thread_surface_area_relative_spread",
)


def digest(points) -> str:
    return hashlib.sha256(points.tobytes()).hexdigest()


def spacing(o3d, cloud) -> float:
    """The voxel the tool runs at: twice the median nearest-neighbour gap."""

    import numpy as np

    return float(np.median(np.asarray(cloud.compute_nearest_neighbor_distance())) * 2.0)


def reconstruct(o3d, cloud, voxel: float) -> dict[str, Any]:
    """The band harness's own pipeline, to the two figures the records state."""

    import numpy as np

    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
    )
    down.orient_normals_consistent_tangent_plane(k=30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        down, depth=POISSON_DEPTH
    )
    density = np.asarray(densities)
    mesh.remove_vertices_by_mask(
        density < float(np.quantile(density, DENSITY_QUANTILE))
    )
    mesh.compute_vertex_normals()

    tree = o3d.geometry.KDTreeFlann(down)
    vertices = np.asarray(mesh.vertices)
    to_sample = np.array(
        [float(np.sqrt(tree.search_knn_vector_3d(v, 1)[2][0])) for v in vertices]
    )
    triangles = np.asarray(mesh.triangles)
    areas = (
        np.linalg.norm(
            np.cross(
                vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
            ),
            axis=1,
        )
        / 2.0
    )
    unsupported = to_sample[triangles].max(axis=1) > voxel
    return {
        "unsupported_area_fraction": float(areas[unsupported].sum() / areas.sum()),
        "surface_area": float(mesh.get_surface_area()),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "downsampled_points": len(down.points),
        "downsampled_digest": digest(np.asarray(down.points)),
        # The normals too, not only the points. Hashing the points alone proves the
        # cloud entering reconstruction is identical and leaves `estimate_normals` and
        # `orient_normals_consistent_tangent_plane` -- both of which run after that
        # hash -- as unexcluded sources of any residue. A tangent-plane orientation
        # walk has tie-breaks; an attribution that names the Poisson solver has to
        # exclude them at this site rather than inherit the exclusion from elsewhere.
        "normals_digest": digest(np.asarray(down.normals)),
    }


def spread(values: list[float]) -> dict[str, Any]:
    import numpy as np

    values = [float(v) for v in values]
    absolute = max(values) - min(values)
    mean = float(np.mean(values))
    return {
        "values": values,
        "distinct": len(set(values)),
        "identical": len(set(values)) == 1,
        "absolute_spread": absolute,
        "relative_spread": absolute / mean if mean else 0.0,
    }


def condition(o3d, reference, label: str, *, seeded: bool, one_thread: bool) -> dict:
    import numpy as np

    before_threads = o3d.utility.get_max_threads()
    observed_threads = before_threads
    if one_thread:
        o3d.utility.set_max_threads(1)
        observed_threads = o3d.utility.get_max_threads()
    try:
        clouds: list[str] = []
        rows = []
        for _ in range(REPEATS):
            if seeded:
                o3d.utility.random.seed(SEED)
            cloud = reference.sample_points_uniformly(number_of_points=SAMPLES)
            clouds.append(digest(np.asarray(cloud.points)))
            if seeded:
                o3d.utility.random.seed(SEED)
            rows.append(reconstruct(o3d, cloud, spacing(o3d, cloud)))
    finally:
        if one_thread:
            o3d.utility.set_max_threads(before_threads)

    return {
        "label": label,
        "sampling_seeded": seeded,
        "repeats": REPEATS,
        "input_cloud_is_the_same_bytes_every_repeat": len(set(clouds)) == 1,
        "input_cloud_digests_observed": len(set(clouds)),
        "downsampled_is_the_same_bytes_every_repeat": len(
            {r["downsampled_digest"] for r in rows}
        )
        == 1,
        # The attribution turns on this. If the points are identical and the normals are
        # too, then everything before the Poisson solver is excluded by measurement at
        # this site, and the residue is the solver and the density crop after it. If the
        # normals differ, normal estimation or the tangent-plane orientation walk is a
        # live alternative and no Poisson-specific claim is available from this harness.
        "normals_are_the_same_bytes_every_repeat": len(
            {r["normals_digest"] for r in rows}
        )
        == 1,
        "normals_digests_observed": len({r["normals_digest"] for r in rows}),
        "max_threads_before": before_threads,
        "max_threads_observed_during": observed_threads,
        "the_thread_knob_engaged": (observed_threads == 1 and before_threads != 1)
        if one_thread
        else None,
        "unsupported_area_fraction": spread(
            [r["unsupported_area_fraction"] for r in rows]
        ),
        "surface_area": spread([r["surface_area"] for r in rows]),
        "triangles": spread([float(r["triangles"]) for r in rows]),
        "vertices": spread([float(r["vertices"]) for r in rows]),
    }


def _git(*args: str) -> str | None:
    """stdout, or None when git could not answer. Never a silent empty string."""

    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else None


def provenance() -> dict[str, Any]:
    """The commit, and whether the tree it was built from was clean.

    A commit alone pins nothing if the tree was dirty: the harness that ran is not
    the harness at that commit. The sibling capability records carry
    `working_tree_clean_at_build` for this reason, and a record that omits it is
    weaker provenance than its neighbours. A git failure is recorded as a failure
    rather than as an empty commit string.
    """

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "repository_commit": commit,
        "working_tree_clean_at_build": (status == "") if status is not None else None,
        "git_answered": commit is not None and status is not None,
    }


def citation_disclosure(cited_present: bool) -> dict[str, str]:
    """The record says the file it cites is missing only while it is missing.

    A flat unconditional key makes the record fail its own citation guard the moment
    the cited record lands, and the only way to pass is to hand-edit a generated
    artifact -- which the next regeneration silently undoes, leaving the record on
    disk and the harness output permanently disagreeing.
    """

    if cited_present:
        return {}
    return {
        "the_record_it_cites_is_not_on_main_yet": (
            f"`{CITED_RECORD}` is in pull request 1021 and is not in "
            "`docs/evidence/open3d/` on main, so a reviewer reading this on main "
            "cannot open it. Stated rather than left to be discovered: a citation a "
            "reviewer cannot resolve is a citation that has to be followed by asking "
            "the author, and this lane has already had one provenance pin re-based "
            "for exactly that reason. The measurements above stand on their own -- "
            "each condition's values are in this file, and the isolation of the "
            "residue is measured here -- so the dependency is for the mechanism's "
            "name, not for the result."
        )
    }


def record_path() -> Path:
    return Path(
        os.environ.get(
            "NPA_REPRO_CAUSES_OUT",
            Path(__file__).resolve().parents[1]
            / "declared-reproducibility-causes.json",
        )
    )


def earlier_runs(out: Path) -> tuple[dict[str, list[float]], str]:
    """The series so far, read from the record being regenerated.

    `EARLIER_RUNS` is the seed for a host that has no record yet. Reading the series
    back instead means a second run of the harness extends them: with the literal
    alone, run eight replaced run seven's value while the record went on claiming
    seven runs, so "the values each run produced" held only as long as somebody
    remembered to hand-edit the literal first.
    """

    if not out.exists():
        return {k: list(v) for k, v in EARLIER_RUNS.items()}, "EARLIER_RUNS literal"
    try:
        prior = json.loads(out.read_text())[
            "observed_across_independent_runs_of_this_harness"
        ]
    except (json.JSONDecodeError, KeyError, OSError):
        return (
            {k: list(v) for k, v in EARLIER_RUNS.items()},
            "EARLIER_RUNS literal (the record on disk had no readable series)",
        )
    series = {}
    for key, seed in EARLIER_RUNS.items():
        values = prior.get(key, {}).get("values")
        series[key] = (
            [float(v) for v in values]
            if isinstance(values, list) and values
            else list(seed)
        )
    return series, f"the series in {out.name}, extended by this run"


def main() -> int:
    import open3d as o3d

    out = record_path()
    series, series_source = earlier_runs(out)
    reference = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    conditions = {
        "as_both_records_ran_it": condition(
            o3d, reference, "unseeded", seeded=False, one_thread=False
        ),
        "sampling_seeded": condition(
            o3d, reference, "seeded", seeded=True, one_thread=False
        ),
        "sampling_seeded_and_one_thread": condition(
            o3d, reference, "seeded+1thread", seeded=True, one_thread=True
        ),
    }

    unseeded = conditions["as_both_records_ran_it"]
    seeded = conditions["sampling_seeded"]
    limited = conditions["sampling_seeded_and_one_thread"]

    # Emitted only while the cited record really is absent. A flat unconditional key
    # would make the record fail its own citation guard the moment #1021 lands, and
    # the only way to pass would be hand-editing a generated artifact -- which the
    # next regeneration silently undoes.
    cited_record_present = (Path(__file__).resolve().parents[1] / CITED_RECORD).exists()

    recorded_gap = abs(RECORDED_FRACTIONS[1] - RECORDED_FRACTIONS[0])
    residue = seeded["unsupported_area_fraction"]["absolute_spread"]

    def observed(key: str, this_run: float) -> dict[str, Any]:
        """This run's value together with the earlier ones, so neither stands alone."""

        values = [*series[key], this_run]
        return {
            "values": values,
            "runs": len(values),
            "min": min(values),
            "max": max(values),
            "where_the_earlier_values_came_from": series_source,
        }

    unseeded_observed = observed(
        "unseeded_unsupported_area_fraction_absolute_spread",
        unseeded["unsupported_area_fraction"]["absolute_spread"],
    )
    fixed_observed = observed(
        "fixed_input_surface_area_relative_spread",
        seeded["surface_area"]["relative_spread"],
    )
    one_thread_observed = observed(
        "one_thread_surface_area_relative_spread",
        limited["surface_area"]["relative_spread"],
    )
    seeded_observed = observed(
        "seeded_unsupported_area_fraction_absolute_spread",
        seeded["unsupported_area_fraction"]["absolute_spread"],
    )

    def figure(value: float) -> str:
        """One spelling for a figure, so a record and a sentence can be compared."""

        return f"{value:.1e}"

    # Exactly the figures this lane's correction put into the two capability records.
    # A reviewer mutated one of them to 9.9e-01 and nothing failed, because those
    # documents are not pinned by anything -- which was true before this PR and is why
    # new derived figures must not be left in that space. Each is listed with the series
    # it came from so the sentence and the evidence can be checked against each other.
    def quoted(series: str, observed: dict[str, Any]) -> dict[str, Any]:
        return {
            "series": series,
            "range_quoted_in_the_sentence": [
                figure(observed["min"]),
                figure(observed["max"]),
            ],
        }

    introduced = {
        "capability-record-distance-bands.json": [
            quoted(
                "unseeded_unsupported_area_fraction_absolute_spread",
                unseeded_observed,
            ),
            quoted("seeded_unsupported_area_fraction_absolute_spread", seeded_observed),
        ],
        "capability-record-support-crop.json": [
            quoted("fixed_input_surface_area_relative_spread", fixed_observed),
        ],
    }

    record = {
        "schema": "npa.evidence.open3d.declared-reproducibility-causes/v0.0.1",
        "question": (
            "Two capability records tell a reader which knob to turn to stop a number "
            "moving: seed the sampling, or pin the thread count. Does either work?"
        ),
        "answer": (
            "Neither settles the figure. Seeding the sampling does make the input "
            "bit-identical, and it does remove most of the movement, so that record's "
            "attribution is right about the dominant cause and wrong only in what it "
            "implies about reproducibility. Limiting threads leaves the residue intact. "
            "The residue is downstream of the reconstruction's inputs: under seeding "
            "the sampled cloud, the downsampled cloud and the estimated and oriented "
            "normals are all byte-identical across the repeats, hashed here to prove "
            "it, and the figures still move. That leaves the Poisson solver and the "
            "density crop applied to what it returns -- which neither record could "
            "have known about, because it was measured afterwards."
        ),
        "claims_examined": {
            "distance_bands": {
                "source": "docs/evidence/open3d/capability-record-distance-bands.json",
                "field": "limitations[3]",
                "text": DISTANCE_BANDS_CLAIM,
                "recorded_values": list(RECORDED_FRACTIONS),
                "what_holds": (
                    "The attribution. Unseeded, the fraction's spread over "
                    f"{REPEATS} repeats came out {unseeded_observed['min']:.3e} to "
                    f"{unseeded_observed['max']:.3e} across "
                    f"{unseeded_observed['runs']} independent runs of this harness -- "
                    f"the same order as the record's own gap of {recorded_gap:.4f} "
                    "between its two values, straddling it rather than bounding it. So "
                    "unseeded sampling alone produces differences of that magnitude, "
                    "and the record needs no second explanation for the pair it saw. "
                    f"Seeding reduces the spread to {residue:.3e} in this run, so it "
                    "really is the dominant cause and really is seedable."
                ),
                "what_this_does_not_settle_about_that_pair": (
                    "That sampling is all that separated the record's two values. They "
                    "came from different runs of different harnesses, so other "
                    "differences were available; this shows only that none is needed to "
                    "explain a gap that size."
                ),
                "what_does_not_hold": (
                    "That seeding would make the figure reproduce. The cloud's bytes "
                    "are identical across the seeded repeats, hashed to prove it, and "
                    "the fraction still takes "
                    f"{seeded['unsupported_area_fraction']['distinct']} distinct values "
                    f"in {REPEATS} runs. 'The fourth decimal place is not reproducible' "
                    "is true as written; what a reader would take from the sentence -- "
                    "that the named cause is the whole cause -- is not."
                ),
            },
            "support_crop": {
                "source": "docs/evidence/open3d/capability-record-support-crop.json",
                "field": "limitations[3]",
                "text": SUPPORT_CROP_CLAIM,
                "declared_bound_relative_on_surface_area": RECORDED_SURFACE_AREA_BOUND,
                "what_holds": (
                    "The magnitude, under the condition that record's pipeline runs in, "
                    "and not by much. Surface area's relative spread with the input "
                    f"fixed came out {fixed_observed['min']:.3e} to "
                    f"{fixed_observed['max']:.3e} across {fixed_observed['runs']} "
                    "independent runs, all inside the declared ~4e-06 but close enough "
                    "to it that a further run could exceed it. The record is right that "
                    "the result is not byte-identical, and that pipeline reads a fused "
                    "cloud from a file, so its input is fixed and the figure is "
                    "measured where it applies."
                ),
                "the_figure_is_conditional_on_something_it_does_not_say": (
                    "~4e-06 relative is a fixed-input number. On this construction, "
                    "where the cloud is sampled rather than read, the same quantity "
                    f"spans {unseeded['surface_area']['relative_spread']:.3e} relative "
                    f"-- about {unseeded['surface_area']['relative_spread'] / RECORDED_SURFACE_AREA_BOUND:.0f}x "
                    "the declared figure, and about "
                    f"{unseeded['surface_area']['relative_spread'] / seeded['surface_area']['relative_spread']:.0f}x "
                    "the fixed-input measurement beside it. That is not the record "
                    "being wrong, because "
                    "that is not the record's pipeline. It is worth stating because a "
                    "reader carrying ~4e-06 to a pipeline that samples its own input "
                    "would be two orders of magnitude out."
                ),
                "what_does_not_hold": (
                    "The cause. The variation is attributed to container thread count, "
                    "and a reader pinning the thread count would expect it to stop. "
                    "Every repeat in the seeded condition ran in one process at one "
                    "thread count and the area still took "
                    f"{seeded['surface_area']['distinct']} distinct values in {REPEATS} "
                    "runs. Setting the count to 1 explicitly -- engaged and recorded, "
                    f"{limited['max_threads_before']} to "
                    f"{limited['max_threads_observed_during']} -- left "
                    f"{limited['surface_area']['distinct']} distinct values in "
                    f"{REPEATS}. Its relative spread at one thread came out "
                    f"{one_thread_observed['min']:.3e} to "
                    f"{one_thread_observed['max']:.3e} over "
                    f"{one_thread_observed['runs']} runs, against "
                    f"{fixed_observed['min']:.3e} to {fixed_observed['max']:.3e} at the "
                    f"default {seeded['max_threads_before']} threads -- "
                    + (
                        "larger in every run"
                        if one_thread_observed["min"] > fixed_observed["max"]
                        else "overlapping"
                    )
                    + ". Either way it does not settle, which is what matters: thread "
                    "count may contribute, but it is not what makes this "
                    "irreproducible."
                ),
            },
        },
        "the_corrections_applied": {
            "how_to_read_the_quotes_above": (
                "Each `text` is the sentence as it stood before this measurement, and "
                "it is no longer in the file it cites -- that is the point. The records "
                "are corrected in place rather than superseded, because a limitation "
                "that sends a reader to the wrong knob is worth fixing where they will "
                "read it, and a correction record they have to find first is not a fix."
            ),
            "what_changed": [
                {
                    "file": "docs/evidence/open3d/capability-record-distance-bands.json",
                    "field": "limitations[3]",
                    "now_also_states": (
                        "that seeding reduces the movement by a large factor but does "
                        "not remove it, with both measured spreads and the reason"
                    ),
                },
                {
                    "file": "docs/evidence/open3d/capability-record-support-crop.json",
                    "field": "limitations[3]",
                    "now_also_states": (
                        "that thread count is not the cause, that the ~4e-06 figure is "
                        "a fixed-input number, and what the cause is"
                    ),
                },
            ],
            "a_third_site_in_product_source": {
                "file": "npa/src/npa/smoke/test_open3d_functional.py",
                "where": "_assert_support_crop_kept_the_observations docstring",
                "why_it_matters_more_than_the_other_two": (
                    "This is where the ~4e-06 figure the support-crop record quotes "
                    "comes from, and it is advice to the next maintainer rather than a "
                    "limitation to a consumer: it told them not to tighten the "
                    "tolerances 'without pinning the thread count', which reads as a "
                    "knob that would let them. There is no such knob. Someone acting "
                    "on it would pin the count, watch the figure keep moving, and have "
                    "no account of why."
                ),
                "the_phrase_removed": (
                    "Do not tighten these to byte equality without pinning the thread "
                    "count"
                ),
                "what_it_says_now": (
                    "that pinning the count will not get you there, with the measured "
                    "behaviour and the Poisson reconstruction named as the residue"
                ),
            },
            "what_was_left_alone": (
                "Every figure in both records, every other field, and the bytes of "
                "every artifact they describe. One string changed in each file. Nothing "
                "here re-measures either record's own pipeline: the construction used "
                "is the one the distance-bands record names, and the support-crop "
                "record's inputs are not reachable from this host."
            ),
        },
        "conditions": conditions,
        "observed_across_independent_runs_of_this_harness": {
            "why_this_is_here": (
                "A spread measured over four repeats is itself unstable, so a sentence "
                "derived from one run of it can be false in the next. That happened: "
                "the first draft said the unseeded spread contains the record's 0.0046 "
                "gap, which was true of two runs and false of the third. These are the "
                "values each run produced, so the comparisons above are made against "
                "the range rather than against whichever run wrote the file."
            ),
            "unseeded_unsupported_area_fraction_absolute_spread": unseeded_observed,
            "seeded_unsupported_area_fraction_absolute_spread": seeded_observed,
            "fixed_input_surface_area_relative_spread": fixed_observed,
            "one_thread_surface_area_relative_spread": one_thread_observed,
            "the_recorded_gap_being_compared_against": recorded_gap,
            "the_declared_surface_area_figure": RECORDED_SURFACE_AREA_BOUND,
            "one_series_was_retained_late": {
                "which": "seeded_unsupported_area_fraction_absolute_spread",
                "why_it_is_shorter": (
                    "It was not kept while the other three were, and a reviewer "
                    "noticed by looking for a figure the corrected sentence quoted -- "
                    "2.3e-05, which was run two's value -- and not finding it here. "
                    "Two runs' worth is missing and is not reconstructed from memory, "
                    "because a back-filled number reads exactly like a measured one."
                ),
                "series_kept_from_the_first_run": list(
                    SERIES_RETAINED_FROM_THE_FIRST_RUN
                ),
            },
        },
        "figures_this_correction_put_into_the_capability_records": {
            "why_this_is_here": (
                "Those documents are pinned by nothing: a reviewer changed one of these "
                "figures to 9.9e-01 and no test failed, and the same is true of the "
                "figures that were already in them before this PR. That is not "
                "something this PR introduced, but two new derived figures in an "
                "unpinned document are, so each one is listed here with the series it "
                "came from and a test requires it to appear in both places. Each is a "
                "range over retained runs, not a single run's value, because the "
                "instability of these spreads is the subject of this record."
            ),
            "by_record": introduced,
        },
        "where_the_residue_comes_from": {
            "isolated_here": (
                "Under the seeded condition every step before the Poisson solver is "
                "byte-identical across the repeats, measured in this run: the sampled "
                "cloud "
                f"({seeded['input_cloud_digests_observed']} digest over "
                f"{REPEATS} repeats), the voxel-downsampled cloud, and the estimated "
                "and tangent-plane-oriented normals "
                f"({seeded['normals_digests_observed']} digest). Normals are hashed "
                "because `estimate_normals` and "
                "`orient_normals_consistent_tangent_plane` run after the point hash, "
                "and an orientation walk has tie-breaks; hashing only the points "
                "would leave them as unexcluded alternatives. What remains after that "
                "hash is `create_from_point_cloud_poisson` and the density-quantile "
                "crop applied to what it returns. The surface area still takes "
                f"{seeded['surface_area']['distinct']} distinct values across those "
                f"{REPEATS} repeats. So the residue is downstream of identical "
                "inputs and identical normals, on this host at this version."
            ),
            "what_that_does_not_establish": (
                "Which part of the solver, or that the same holds at another Open3D "
                "version, on another build, or on a scene with different topology. "
                "`create_from_point_cloud_poisson` returns both the mesh and the "
                "densities; this harness does not separate a varying mesh from a "
                "varying density field, and the crop is deterministic given the "
                "densities it is handed."
            ),
            "why_neither_record_is_at_fault": (
                "The reconstruction's own non-determinism had not been measured when "
                "either record was written, so the visible causes were the ones they "
                "named."
            ),
        },
        **citation_disclosure(cited_record_present),
        "what_this_does_not_claim": [
            (
                f"That {REPEATS} repeats bound anything. These are observed spreads over "
                "a sample, and a fifth run may exceed any of them."
            ),
            (
                "That the sampling does not matter. It matters most, by a large factor, "
                "and it is seedable. The correction is to what seeding achieves."
            ),
            (
                "That thread count has no effect on these figures. One thread count was "
                "compared against the default on one machine; that cannot separate a "
                "small thread-count contribution from the residue."
            ),
            (
                "That either record's published figures are wrong. No figure in either "
                "record is changed by this; the limitation sentences are."
            ),
        ],
        "generator": {
            "harness": "docs/evidence/open3d/harness/declared_reproducibility_causes.py",
            "scene": "open3d.data.ArmadilloMesh, 300k uniform surface samples",
            "open3d_version": o3d.__version__,
            "python": sys.version.split()[0],
            **provenance(),
            "poisson_depth": POISSON_DEPTH,
            "density_quantile": DENSITY_QUANTILE,
            "seed": SEED,
        },
    }

    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    for name, block in conditions.items():
        frac = block["unsupported_area_fraction"]
        area = block["surface_area"]
        print(
            f"\n{name}\n  input identical: "
            f"{block['input_cloud_is_the_same_bytes_every_repeat']}  "
            f"threads {block['max_threads_observed_during']}\n"
            f"  unsupported fraction: {frac['distinct']} distinct, spread "
            f"{frac['absolute_spread']:.3e} abs / {frac['relative_spread']:.3e} rel\n"
            f"  surface area:         {area['distinct']} distinct, spread "
            f"{area['absolute_spread']:.3e} abs / {area['relative_spread']:.3e} rel"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
