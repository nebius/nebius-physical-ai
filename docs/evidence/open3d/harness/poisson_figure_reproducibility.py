#!/usr/bin/env python3
"""Measure how reproducible the published ray-cast occlusion figures are.

`audit-view-occlusion.json` publishes figures to sixteen significant digits. Re-running
the harness that wrote them, on the same host and in the same venv against the same
committed code, does not return the same number. A reviewer had flagged the ray-cast path
as unverified and suspected it might share the one-pixel environment dependence they
found in the z-buffer footprint path. It does not share that cause.

The cause is upstream: `create_from_point_cloud_poisson` is not deterministic in Open3D
0.20.0. Identical input, one process, same parameters, and the reconstructed surface
differs. The topology is stable -- vertex and triangle counts repeat exactly -- so this is
not a different mesh, it is the same mesh with vertices in slightly different places,
which is what floating-point accumulation under thread scheduling looks like.

Everything here is measured in this file rather than cited, because the remedies are the
interesting part and both of the obvious ones fail:

- Seeding does not help. The sampling seed already in `audit_view_occlusion.py` is real
  and does bind the armadillo's *sampling*, but Poisson's variation is not drawn from
  Open3D's generator, so seeding before the call changes nothing.
- Limiting threads through the solver's own per-call argument did make four repeated
  surface-area results identical. Two things went wrong on the way to that
  sentence, and both are recorded rather than tidied away. A first two-run probe put
  single-threading two orders of magnitude ahead; at four repeats it is within a small
  factor, so two runs supported nothing. And the experiment behind it was measured by
  setting `OMP_NUM_THREADS=1`, which a reviewer showed is not a knob in this build at
  all - no OpenMP runtime linked, no OpenMP symbols. It ran and tested nothing. The
  measurement now passes `n_threads=1` to the Poisson call and records per-call wall
  time beside the result.

The default path remains non-exact, so the honest move is to state the precision the
published default-path figures hold while recording the bounded deterministic option.

Run:
    npa/.venv/bin/python docs/evidence/open3d/harness/poisson_figure_reproducibility.py

Needs Open3D. Writes `docs/evidence/open3d/poisson-figure-reproducibility.json`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
HARNESS = Path(__file__).resolve().parent
OUT = REPO / "docs" / "evidence" / "open3d" / "poisson-figure-reproducibility.json"

#: Poisson depth the audit harness uses.
DEPTH = 9

#: Repeats per condition. Two is enough to show non-identity; more is needed to say
#: anything about the size of the spread, which is the number the record has to publish.
REPEATS = 4

#: The published figures this is about, transcribed from `audit-view-occlusion.json` at
#: `04aacfcf89`. Not read from that file: this record has to stand on its own if that
#: record is regenerated, and a reader comparing the two should see both numbers.
PUBLISHED = {
    "visible_with_retained_surface_shown": 0.3521503113541809,
    "visible_with_only_its_own_self_occlusion": 0.7848810005123376,
}

#: The threshold the record's conclusion rests on. The margin against it is what decides
#: whether non-reproducibility matters here or is a curiosity.
MINIMUM_VISIBLE_AREA = 0.5


def surface_area(mesh) -> float:
    """Total triangle area, which is what the occlusion figures are fractions of."""
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float((0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)).sum())


def reconstruct(o3d, cloud, seed: int | None, *, n_threads: int = -1) -> dict:
    if seed is not None:
        o3d.utility.random.seed(seed)
    started = time.perf_counter()
    mesh, _density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=DEPTH, n_threads=n_threads
    )
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "surface_area": surface_area(mesh),
        "wall_seconds": time.perf_counter() - started,
    }


def spread(values: list[float]) -> dict:
    lo, hi = min(values), max(values)
    return {
        "values": values,
        "identical": lo == hi,
        "absolute_spread": hi - lo,
        "relative_spread": (hi - lo) / lo if lo else None,
    }


def condition(o3d, cloud, label: str, seed: int | None, *, n_threads: int = -1) -> dict:
    runs = [reconstruct(o3d, cloud, seed, n_threads=n_threads) for _ in range(REPEATS)]
    areas = [run["surface_area"] for run in runs]
    return {
        "repeats": REPEATS,
        "seeded_before_each_call": seed is not None,
        "vertex_and_triangle_counts_are_stable": (
            len({(r["vertices"], r["triangles"]) for r in runs}) == 1
        ),
        "vertices": runs[0]["vertices"],
        "triangles": runs[0]["triangles"],
        "surface_area": spread(areas),
        "wall_seconds": [run["wall_seconds"] for run in runs],
        "poisson_n_threads": n_threads,
        "label": label,
    }


def openmp_is_not_linked() -> dict:
    """Whether `OMP_NUM_THREADS` could have done anything in this build.

    An earlier version of this harness measured single-threading by setting
    `OMP_NUM_THREADS=1` in a subprocess, so the pool would be sized before Open3D
    loaded. A reviewer pointed out that the timing was careful and the premise was not:
    the variable only matters if an OpenMP runtime is linked, and here none is. The
    experiment ran, produced numbers, and tested nothing.

    So this records whether the knob could engage, rather than assuming it from the fact
    that setting it is conventional. Check that the knob turned before believing the
    result.
    """
    import open3d as o3d

    base = Path(o3d.__file__).parent
    binaries = [path for path in base.rglob("*.so") if "pybind" in path.name]
    linked, symbols = None, None
    ldd_returncode, nm_returncode = None, None
    if binaries:
        target = str(binaries[0])
        ldd = subprocess.run(
            ["ldd", target], capture_output=True, text=True, check=False
        )
        nm = subprocess.run(
            ["nm", "-D", target], capture_output=True, text=True, check=False
        )
        linked, symbols = ldd.stdout, nm.stdout
        ldd_returncode, nm_returncode = ldd.returncode, nm.returncode
    checks_succeeded = ldd_returncode == 0 and nm_returncode == 0
    return {
        "binary": binaries[0].name if binaries else None,
        "ldd_returncode": ldd_returncode,
        "nm_returncode": nm_returncode,
        "openmp_runtime_linked": None
        if not checks_succeeded
        else bool(
            linked and any(lib in linked for lib in ("libgomp", "libomp", "libiomp"))
        ),
        "openmp_symbols_present": None
        if not checks_succeeded
        else bool(symbols and ("GOMP_" in symbols or "omp_get_max_threads" in symbols)),
        "omp_num_threads_was_inert_in_this_build": (
            checks_succeeded
            and not any(lib in linked for lib in ("libgomp", "libomp", "libiomp"))
            and "GOMP_" not in symbols
            and "omp_get_max_threads" not in symbols
        ),
    }


def thread_limited_condition(o3d, cloud) -> dict:
    """Pass the Poisson solver's documented per-call thread argument."""

    return condition(o3d, cloud, "poisson_n_threads_1", None, n_threads=1)


def source_provenance() -> dict:
    """Bind the generated record to the exact committed producer bytes."""

    harness_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    clean = (
        subprocess.run(
            ["git", "-C", str(REPO), "diff", "--quiet"], check=False
        ).returncode
        == 0
    )
    return {
        "source_commit": head,
        "clean_tree_before_write": clean,
        "harness_sha256": harness_sha,
    }


def main() -> int:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(o3d.data.EaglePointCloud().path)
    points = len(cloud.points)

    default = condition(o3d, cloud, "default", None)
    seeded = condition(o3d, cloud, "seeded", 1729)
    single = thread_limited_condition(o3d, cloud)
    openmp = openmp_is_not_linked()

    # Derived rather than asserted, because a two-run probe of this had me about to
    # publish "two orders of magnitude" as a finding and four repeats put it at a
    # factor of two. The conditions are compared by the numbers that were actually
    # measured above, so this sentence cannot drift from them.
    default_spread = default["surface_area"]["relative_spread"]
    single_spread = single["surface_area"]["relative_spread"]
    improvement = default_spread / single_spread if single_spread else None
    distinct = {
        name: len(set(row["surface_area"]["values"]))
        for name, row in (
            ("default", default),
            ("seeded_before_each_call", seeded),
            ("poisson_n_threads_1", single),
        )
    }
    if single["surface_area"]["identical"]:
        single_threading_verdict = (
            f"Made surface area exact across {REPEATS} observations in this run: "
            f"the spread went from {default_spread:.2e} to 0 with the Poisson call's "
            "documented n_threads=1. This is an observed bounded result, not a general "
            "determinism guarantee; the per-call value and wall times are recorded."
        )
    else:
        direction = "smaller" if single_spread < default_spread else "larger"
        factor = f"a factor of {improvement:.1f}; " if improvement is not None else ""
        single_threading_verdict = (
            f"Did not make this exact. The spread went from {default_spread:.2e} to "
            f"{single_spread:.2e}, {factor}{direction} with n_threads=1; "
            f"{distinct['poisson_n_threads_1']} distinct areas remained."
        )
    published_spread = abs(
        PUBLISHED["visible_with_only_its_own_self_occlusion"] - 0.7848528993182663
    )
    margin = (
        PUBLISHED["visible_with_only_its_own_self_occlusion"] - MINIMUM_VISIBLE_AREA
    )

    report = {
        "schema": "npa.evidence.open3d.poisson-figure-reproducibility/v0.0.1",
        "provenance": source_provenance(),
        "what_this_measures": (
            "Whether the ray-cast occlusion figures in `audit-view-occlusion.json` are "
            "reproducible, and if not, to what precision they are."
        ),
        "why": (
            "A reviewer flagged the ray-cast path as unverified and suspected it might "
            "share the one-pixel environment dependence they measured in the z-buffer "
            "footprint path. It does not. The z-buffer path reproduced exactly on this "
            "host; this one does not reproduce against itself."
        ),
        "subject_availability": (
            "The audit-view record and harness are supplied by PR 1014 and are not in "
            "this merge target. Values are pinned to commit 04aacfcf89; the causal "
            "attribution here is inferred from a different fixed Eagle input."
        ),
        "environment": {
            "open3d": o3d.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "input": "Open3D's own EaglePointCloud, a fixed dataset with no sampling",
            "input_points": points,
            "poisson_depth": DEPTH,
        },
        "the_published_figures": {
            "transcribed_from": "docs/evidence/open3d/audit-view-occlusion.json @ 04aacfcf89",
            "values": PUBLISHED,
            "one_rerun_gave": {
                "visible_with_retained_surface_shown": 0.3521360294407621,
                "visible_with_only_its_own_self_occlusion": 0.7848528993182663,
            },
            "absolute_difference_on_the_headline": published_spread,
        },
        "cause": {
            "claim": (
                "`create_from_point_cloud_poisson` is not deterministic in Open3D "
                f"{o3d.__version__}."
            ),
            "how_it_was_established": (
                "Identical input, one process, same depth, repeated. Vertex and triangle "
                "counts repeat exactly while total surface area does not. Counts alone "
                "do not establish identical mesh connectivity."
            ),
            "why_an_earlier_experiment_here_was_void": openmp,
            "conditions": {
                "default": default,
                "seeded_before_each_call": seeded,
                "poisson_n_threads_1": single,
            },
        },
        "what_the_remedies_do": {
            "seeding": (
                "Does not help. The `SAMPLING_SEED` already in `audit_view_occlusion.py` "
                "is real and does bind the armadillo's sampling -- but Poisson's variation "
                "is not drawn from Open3D's generator, so seeding before the call leaves "
                "the spread where it was. Measured, not reasoned: see "
                "`cause.conditions.seeded_before_each_call`."
            ),
            "limiting_threads": single_threading_verdict,
            "so_the_record_should": (
                "State the precision the figures hold instead of publishing sixteen "
                "digits that do not survive a second run."
            ),
        },
        "does_this_change_any_conclusion": {
            "conclusion_changed": False,
            "explanation": (
                "The record's conclusion is that the removed surface is visible from the "
                "shipped camera well above the 50% criterion. The margin is "
                f"{margin:.4f} and the observed run-to-run difference is "
                f"{published_spread:.2e}, four orders of magnitude smaller."
            ),
            "margin_against_the_threshold": margin,
            "observed_difference": published_spread,
            "ratio": margin / published_spread if published_spread else None,
        },
        "guard_scope": {
            "files": "tracked Python files from git ls-files",
            "covered": [
                "exact comparisons and membership",
                "pytest.approx including its default tolerance",
                "math.isclose and assert_allclose",
            ],
            "known_limit": (
                "Values obtained through arbitrary function returns are not followed."
            ),
        },
        "what_this_does_not_claim": [
            (
                "That the figures are wrong. One rerun agreed to four significant "
                "figures; that single pair is not a general precision bound."
            ),
            (
                "That Poisson is wrong. A non-deterministic surface reconstruction is a "
                "normal consequence of parallel floating-point accumulation; the defect "
                "is publishing sixteen digits of it without saying so."
            ),
            (
                "That the z-buffer footprint path shares this. It reproduced exactly on "
                "this host, and the reviewer's one-pixel difference there came from a "
                "second environment, which is a different question this does not answer."
            ),
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"input: {points} points, poisson depth {DEPTH}, {REPEATS} repeats")
    for name, row in report["cause"]["conditions"].items():
        area = row["surface_area"]
        print(
            f"  {name:26} counts_stable="
            f"{row['vertex_and_triangle_counts_are_stable']} "
            f"identical={area['identical']} relative_spread={area['relative_spread']:.2e}"
        )
    print(f"margin against the 50% criterion: {margin:.4f}")
    print(f"observed run-to-run difference:   {published_spread:.2e}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
