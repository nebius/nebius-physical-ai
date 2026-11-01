"""Is the removed surface the audit view exists to show actually visible from its camera?

The blueprint ships a view called "Removed: unsupported surface". The source's own
comment on `VIEW_GEOMETRY` states why it matters: *"A clipped audit view is worse than
a missing one, because it looks like the audit happened."*

Clipping is solved. Every view now gets a camera fitted against its own geometry, and
`camera_framing_probe.py` reports the worst point of each view in units of the frame
half-extent. **In frame is not the same as visible.** That view draws `/world/mesh` and
`/world/fused` in the same pane, and the removed surface lies behind the retained mesh
wherever the crop cut away a bridge on the far side of real structure. A vertex can sit
at reach 0.3 -- comfortably inside the frame, reported clean by that probe -- and be
entirely hidden behind opaque retained geometry. Same failure the comment names, by a
mechanism its fix does not address.

So this casts one ray from the camera eye toward each sample of the removed surface
against the geometry drawn in front of it, and reports the fraction that survives.

Area-weighted, by sampling triangles rather than counting vertices. Poisson refines
where the data is dense, so a vertex count reports whichever region happens to have the
most vertices rather than the most surface, and the two are not the same region.

Real Open3D ray casting (`o3d.t.geometry.RaycastingScene`) and the product's own
`_camera`/`_view_cameras`/`_sample_distances`, imported rather than reimplemented: a
measurement of a camera this file fitted itself would prove nothing about the one that
ships. No renderer, no network beyond Open3D fetching its own published datasets, no
Rerun.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def repository_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"not inside a git checkout: {here}")


REPO = repository_root()
sys.path.insert(0, str(REPO / "npa/src"))

OUT = Path(
    os.environ.get(
        "NPA_OCCLUSION_OUT",
        Path(__file__).resolve().parent.parent / "audit-view-occlusion.json",
    )
)

#: Quantile of Poisson density discarded before anything else, matching the tool.
DENSITY_QUANTILE = 0.02

#: Barycentric samples per triangle. One centroid would miss a triangle that is half
#: behind an occluder, and the removed surface meets the retained one along an edge, so
#: partially occluded triangles are the common case rather than the exotic one.
SAMPLES_PER_TRIANGLE = 6

#: Rays start this many voxels off the surface toward the eye, and a hit nearer than
#: this to the target does not count. Without it every sample occludes itself: the
#: removed surface is in the occluder scene's neighbourhood along the seam where the
#: crop cut, and floating-point coincidence there would read as total occlusion.
SELF_HIT_VOXELS = 0.25


def barycentric_samples(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted sample points over a mesh, with the area each one stands for."""

    a, b, c = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    # A fixed lattice rather than random draws, so the number this reports does not
    # move between runs of the same geometry.
    lattice = [
        (1 / 3, 1 / 3),
        (2 / 3, 1 / 6),
        (1 / 6, 2 / 3),
        (1 / 6, 1 / 6),
        (0.5, 0.25),
        (0.25, 0.5),
    ][:SAMPLES_PER_TRIANGLE]

    points = []
    weights = []
    for u, v in lattice:
        points.append(a * (1.0 - u - v) + b * u + c * v)
        weights.append(areas / len(lattice))
    return np.concatenate(points), np.concatenate(weights)


def visible_fraction(
    o3d,
    target_vertices: np.ndarray,
    target_triangles: np.ndarray,
    occluders: list,
    camera: dict,
    voxel: float,
) -> dict:
    """Area fraction of the target surface with no occluder between it and the eye."""

    points, weights = barycentric_samples(target_vertices, target_triangles)
    eye = np.asarray(camera["eye"], float)

    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("target surface has no area")

    scene = o3d.t.geometry.RaycastingScene()
    occluder_triangles = 0
    for occluder in occluders:
        if occluder is None or len(np.asarray(occluder.triangles)) == 0:
            continue
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(occluder))
        occluder_triangles += len(np.asarray(occluder.triangles))

    direction = points - eye
    reach = np.linalg.norm(direction, axis=1)
    unit = direction / reach[:, None]

    if occluder_triangles == 0:
        hit = np.full(len(points), np.inf)
    else:
        rays = o3d.core.Tensor(
            np.concatenate([np.tile(eye, (len(points), 1)), unit], axis=1).astype(
                np.float32
            )
        )
        hit = scene.cast_rays(rays)["t_hit"].numpy()

    tolerance = voxel * SELF_HIT_VOXELS
    occluded = hit < (reach - tolerance)
    return {
        "samples": len(points),
        "occluder_triangles": int(occluder_triangles),
        "target_area": total,
        "visible_area_fraction": float(weights[~occluded].sum() / total),
        "occluded_area_fraction": float(weights[occluded].sum() / total),
        "self_hit_tolerance": float(tolerance),
        # Reported so a zero can be told from a view that frames nothing: a sample
        # behind the eye is a framing defect and `camera_framing_probe.py` owns it,
        # but it must not be silently folded into an occlusion number.
        "samples_behind_eye": int((((points - eye) @ _forward(camera)) <= 0.0).sum()),
    }


def _covers(hidden: str, entity: str) -> bool:
    """Does hiding ``hidden`` hide ``entity``? Entity paths, not string prefixes.

    `"/world/mesh_unsupported".startswith("/world/mesh")` is true and the answer is no.
    Written as a prefix test, this dropped the overlay -- the view's entire subject --
    out of the list of entities being shown, and would have put it in the occluder set
    had the table ever hidden a different entity. A path is covered by its ancestors,
    and `mesh_unsupported` is not a child of `mesh`.
    """

    entity = entity.removesuffix("/**").rstrip("/")
    hidden = hidden.removesuffix("/**").rstrip("/")
    return entity == hidden or entity.startswith(hidden + "/")


def _forward(camera: dict) -> np.ndarray:
    forward = np.asarray(camera["look_target"], float) - np.asarray(
        camera["eye"], float
    )
    return forward / np.linalg.norm(forward)


def build_eagle(o3d):
    """Open3D's own partial eagle scan: Poisson genuinely closes a shell over it."""

    cloud = o3d.io.read_point_cloud(o3d.data.EaglePointCloud().path)
    voxel = float(
        np.median(np.asarray(cloud.compute_nearest_neighbor_distance())) * 2.0
    )
    return cloud, voxel


#: Open3D seeds a global generator rather than a call, which is what `_seeded` in the
#: runner relies on too. Without it this scene resamples differently every run: the first
#: two runs of this harness reported the armadillo at 38.5% and then 38.0%, which is small
#: enough to look like noise in a threshold test and is really an unseeded draw. Verified
#: to bind rather than assumed -- the same seed reproduces the sample bytes and a
#: different seed does not.
SAMPLING_SEED = 1729


def build_armadillo(o3d):
    """A complete mesh resampled: the control where little should be unsupported."""

    reference = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    reference.compute_vertex_normals()
    o3d.utility.random.seed(SAMPLING_SEED)
    cloud = reference.sample_points_uniformly(number_of_points=300000)
    voxel = float(
        np.median(np.asarray(cloud.compute_nearest_neighbor_distance())) * 2.0
    )
    return cloud, voxel


SCENES = {"eagle_partial_scan": build_eagle, "armadillo_complete": build_armadillo}

#: Azimuths swept at the product's own elevation to answer whether a camera could have
#: fixed this. Every 15 degrees: fine enough that a favourable angle could not fall
#: between two samples, and the answer turned out not to be close.
SWEEP_AZIMUTHS = range(0, 360, 15)


def _quad(o3d, x: float, half: float):
    """One axis-aligned quad at ``x``, facing an eye on the +x axis."""

    vertices = np.array(
        [[x, -half, -half], [x, half, -half], [x, half, half], [x, -half, half]],
        dtype=float,
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]])),
    )
    mesh.compute_vertex_normals()
    return mesh


def controls(o3d) -> dict:
    """Both extremes, so a measured number is not a constant.

    Geometry fully behind an occluder must read nothing visible; the same geometry moved
    in front of the same occluder, from the same eye, must read everything. Both are in
    frame, which is what separates a zero here from a clipped view.
    """

    eye = {
        "eye": [10.0, 0.0, 0.0],
        "look_target": [0.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "fov_degrees": 60.0,
        "elevation_degrees": 35.0,
    }
    occluder = _quad(o3d, 2.0, 2.0)
    results = {}
    for label, target in (
        ("fully_behind_an_occluder", _quad(o3d, 0.0, 1.0)),
        ("fully_in_front_of_it", _quad(o3d, 4.0, 1.0)),
    ):
        results[label] = visible_fraction(
            o3d,
            np.asarray(target.vertices),
            np.asarray(target.triangles),
            [occluder],
            eye,
            0.05,
        )
    return results


def azimuth_sweep(o3d, removed, kept, samples, camera, voxel: float) -> dict:
    """Could any angle have cleared the gate, with the retained surface still shown?

    This is the question that decides whether the remedy is the view's contents or its
    camera. Self-occlusion is deliberately excluded here, so the sweep is scored on the
    most favourable reading available to a camera-only fix.
    """

    points = np.concatenate(
        [
            np.asarray(removed.vertices),
            np.asarray(kept.vertices),
            np.asarray(samples.points),
        ]
    )
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    elevation = np.radians(camera["elevation_degrees"])
    scored = []
    for degrees in SWEEP_AZIMUTHS:
        azimuth = np.radians(degrees)
        offset = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        candidate = dict(camera)
        candidate["eye"] = list(centre + offset * camera["distance"])
        candidate["look_target"] = list(centre)
        scored.append(
            (
                visible_fraction(
                    o3d,
                    np.asarray(removed.vertices),
                    np.asarray(removed.triangles),
                    [kept],
                    candidate,
                    voxel,
                )["visible_area_fraction"],
                int(degrees),
            )
        )
    best = max(scored)
    worst = min(scored)
    shipped = visible_fraction(
        o3d,
        np.asarray(removed.vertices),
        np.asarray(removed.triangles),
        [kept],
        camera,
        voxel,
    )["visible_area_fraction"]
    return {
        "azimuths_measured": len(scored),
        "elevation_degrees": camera["elevation_degrees"],
        "self_occlusion_excluded": True,
        "best_visible_area_fraction": best[0],
        "best_azimuth_degrees": best[1],
        "worst_visible_area_fraction": worst[0],
        "worst_azimuth_degrees": worst[1],
        "shipped_visible_area_fraction": shipped,
    }


def reconstruct_and_crop(o3d, cloud, voxel: float):
    """Poisson, density trim, then the support crop, exactly as the tool orders them."""

    from npa.workbench.open3d.runner import _sample_distances

    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
    )
    down.orient_normals_consistent_tangent_plane(30)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        down, depth=9
    )
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, DENSITY_QUANTILE))
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    distances = _sample_distances(o3d, down, vertices)
    # The tool crops at voxel * support_distance_factor and the overlay is drawn at the
    # same limit; `_log_removed_surface` exists because drawing it at any other value
    # contradicts the mesh beside it while every published number stays right.
    limit = voxel * 1.0
    per_triangle = distances[triangles].max(axis=1)

    def submesh(mask):
        if not mask.any():
            return None
        part = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles[mask]),
        )
        part.remove_unreferenced_vertices()
        part.compute_vertex_normals()
        return part

    return down, submesh(per_triangle <= limit), submesh(per_triangle > limit)


def main() -> int:
    import open3d as o3d
    from npa.workbench.open3d.runner import REMOVED_VIEW, VIEW_GEOMETRY, _view_cameras

    rows = []
    sweep: dict = {}
    for name, build in SCENES.items():
        cloud, voxel = build(o3d)
        samples, kept, removed = reconstruct_and_crop(o3d, cloud, voxel)
        if removed is None or kept is None:
            rows.append({"scene": name, "verdict": "no_removed_surface"})
            print(f"{name}: no removed surface", flush=True)
            continue

        removed_vertices = np.asarray(removed.vertices)
        # The camera the product would ship for this view, fitted by the product's own
        # code against the same geometry keys the blueprint gives it.
        cameras = _view_cameras(
            {
                "fused": np.asarray(samples.points),
                "mesh": np.asarray(kept.vertices),
                "unsupported": removed_vertices,
            },
            [0.0, 0.0, 1.0],
        )
        camera = cameras[REMOVED_VIEW]

        # Which meshes the view actually shows, read off the product's own table rather
        # than assumed here. A harness that hardcodes the occluder set measures the
        # assumption it was written with, and would keep reporting the fixed number
        # after someone re-enabled the entity that caused the defect.
        view = VIEW_GEOMETRY[REMOVED_VIEW]
        shown = [
            entity
            for entity in view.contents
            if not any(_covers(hidden, entity) for hidden in view.hidden_by_default)
        ]
        # The fused cloud is logged as points and occludes nothing a ray can hit, so
        # only meshes enter the occluder set. The removed surface is always in it: it is
        # opaque in the viewer, so a reviewer loses the part of it that is behind the
        # rest of it, and leaving that out would overstate what they can see.
        occluders = [removed] + ([kept] if "/world/mesh" in shown else [])

        result = visible_fraction(
            o3d,
            removed_vertices,
            np.asarray(removed.triangles),
            occluders,
            camera,
            voxel,
        )
        # Both counterfactuals, so the headline can be attributed rather than asserted.
        self_only = visible_fraction(
            o3d,
            removed_vertices,
            np.asarray(removed.triangles),
            [removed],
            camera,
            voxel,
        )
        with_retained = visible_fraction(
            o3d,
            removed_vertices,
            np.asarray(removed.triangles),
            [removed, kept],
            camera,
            voxel,
        )
        rows.append(
            {
                "scene": name,
                "voxel_size": voxel,
                "removed_triangles": len(np.asarray(removed.triangles)),
                "retained_triangles": len(np.asarray(kept.triangles)),
                "removed_area_fraction_of_surface": float(
                    result["target_area"]
                    / (
                        result["target_area"]
                        + _area(np.asarray(kept.vertices), np.asarray(kept.triangles))
                    )
                ),
                "camera_eye": camera["eye"],
                "camera_elevation_degrees": camera["elevation_degrees"],
                # The eye alone cannot be checked against anything: it is three floats
                # fitted per scene, so any three floats look plausible. With the target
                # and up it was fitted around, the elevation is recoverable from the
                # geometry -- which is what makes the recorded angle checkable rather
                # than merely present.
                "camera_look_target": camera["look_target"],
                "camera_up": camera["up"],
                "view_contents": list(view.contents),
                "view_hidden_by_default": list(view.hidden_by_default),
                "entities_shown": shown,
                "visible_with_retained_surface_shown": with_retained[
                    "visible_area_fraction"
                ],
                "visible_with_only_its_own_self_occlusion": self_only[
                    "visible_area_fraction"
                ],
                "area_the_retained_surface_hides": (
                    self_only["visible_area_fraction"]
                    - with_retained["visible_area_fraction"]
                ),
                **result,
            }
        )
        print(
            f"{name}: {result['visible_area_fraction']:.1%} of the removed surface "
            f"visible from the shipped audit camera "
            f"({with_retained['visible_area_fraction']:.1%} if the retained surface "
            f"is shown, {self_only['visible_area_fraction']:.1%} if it is not)",
            flush=True,
        )

        if not sweep:
            sweep = azimuth_sweep(o3d, removed, kept, samples, camera, voxel)
            print(
                f"  no camera fixes it: best of {sweep['azimuths_measured']} azimuths is "
                f"{sweep['best_visible_area_fraction']:.1%} at "
                f"{sweep['best_azimuth_degrees']} degrees, against "
                f"{sweep['shipped_visible_area_fraction']:.1%} shipped",
                flush=True,
            )

    OUT.write_text(
        json.dumps(
            {
                "what_this_measures": (
                    "Whether the removed surface the audit view exists to show is "
                    "actually visible from the camera that ships, rather than merely "
                    "inside the frame. One ray from the camera eye per area-weighted "
                    "sample of the removed surface, against the meshes the view draws."
                ),
                "why_in_frame_was_not_enough": (
                    "camera_framing_probe.py reports the worst point of each view in "
                    "units of the frame half-extent, and the audit view was clean by "
                    "that measure. It draws the retained surface in the same pane, so a "
                    "sample can sit at reach 0.3 and be entirely behind opaque geometry. "
                    "That is the failure the source's own VIEW_GEOMETRY comment names -- "
                    "an audit view that looks like the audit happened -- reached by a "
                    "mechanism the comment's fix does not address."
                ),
                "what_changed": (
                    "/world/mesh is hidden by default in this view, through the same "
                    "table that gives the view its contents and its camera geometry. It "
                    "is still carried and one click away. The fused cloud stays visible: "
                    "points occlude nothing a ray can hit, so the overlay is still shown "
                    "against the observations it is judged against."
                ),
                "why_the_camera_is_not_the_fix": sweep,
                "the_residual_this_does_not_fix": (
                    "The removed surface occludes itself. It is an extrapolated shell, so "
                    "from any single eye the far side of it sits behind the near side, "
                    "which is the whole remaining loss after this change and is not "
                    "something a visibility default or a camera can recover. An orbiting "
                    "reviewer sees the rest; a single static frame does not."
                ),
                "reproducibility": {
                    "sampling_seed": SAMPLING_SEED,
                    "why_seeded": (
                        "The armadillo scene resamples a complete mesh, and Open3D seeds "
                        "a global generator rather than a call. Unseeded, the first two "
                        "runs of this harness reported 38.5% and then 38.0% -- small "
                        "enough to read as noise and really an unseeded draw."
                    ),
                    "and_still_not_bit_exact": (
                        "Seeding does not make this reproduce exactly, and claiming it "
                        "would be wrong. Measured across two full runs: the visible "
                        "fractions agree to 2e-5, while triangle counts move by 1 of "
                        "1.09M on the eagle and 22 of 1.04M on the armadillo, with the "
                        "kept and removed counts moving by the same amount in opposite "
                        "directions. Those are triangles sitting within floating-point "
                        "reach of the crop limit changing which side they land on. "
                        "orient_normals_consistent_tangent_plane is multithreaded and its "
                        "tie-breaking is not seeded, so a seed cannot remove this. The "
                        "synthetic controls are exactly identical across runs, which "
                        "locates the variation in the reconstruction rather than in the "
                        "measurement."
                    ),
                    "why_it_does_not_affect_the_verdict": (
                        "The gate is 50% and the measured values are 78% and 82%, so a "
                        "2e-5 wobble is 4 orders of magnitude below the margin. It is "
                        "recorded because a figure republished to the last digit would "
                        "otherwise look falsified on the next run."
                    ),
                },
                "controls": controls(o3d),
                "scenes": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT}")
    return 0


def _area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    a, b, c = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    return float((0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)).sum())


if __name__ == "__main__":
    raise SystemExit(main())
