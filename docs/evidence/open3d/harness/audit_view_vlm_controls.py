"""A matched control pair for an occlusion defect, for VLM judge calibration.

The VLM lane's `vlm-geometry-judge-calibration-v3` froze six frames in three matched
pairs -- registration alignment, support crop, and camera framing -- and returned a
reviewed negative result: no hosted model qualified as a visual acceptance gate. Its
`expected_framing` dimension has one passing value, `reviewable`, and every judge that
parsed its clipped frame called it exactly that.

Occlusion is a fourth defect class that set does not contain, and that its schema cannot
express. `camera_framing_probe.py` passed the shipped audit view: nothing was cut off,
the subject was inside the frame at every vertex. `audit_view_occlusion.py` then measured
that 35.2% of the subject was visible, because the view drew an opaque surface in front
of it. In frame and visible are different properties, and a dimension that spells the
good case `reviewable` has no room for a frame that is well composed and unreadable.

So this produces the pair: one frame as the product shipped it, one as #1014 fixed it,
from one camera over one set of geometry bytes. The only difference is whether the
retained surface is drawn.

Two instruments, deliberately. #1014 cast rays against a `RaycastingScene` and reported
area. This counts pixels the rasterizer actually wrote, through the same z-buffer that
produced the delivered PNGs. A geometric claim that does not survive into the pixels a
reviewer sees is a claim about the wrong thing, and agreement between two unrelated
measurements is worth more than either alone.

Nothing here scores a judge. The frozen plan
(`program/claims/open3d-audit-view-vlm-read/CLAIM.md`) reserves that verdict to the VLM
lane and keeps v3's negative result standing.

Needs open3d, PIL, and Open3D's published Eagle scan. Writes
`evidence/open3d/vlm/controls-v4-occlusion/`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


def repository_root(start: Path | None = None) -> Path:
    """Walk up for a repository marker, refusing to guess when there is none."""

    for candidate in (start or Path(__file__).resolve()).parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("not inside a repository checkout")


REPO = repository_root()
HARNESS = Path(__file__).resolve().parent
OUT = Path(
    os.environ.get("NPA_VLM_CONTROL_OUT", REPO / "docs/evidence/open3d/vlm-controls-v4")
)

#: A tint the subject cannot be confused with by eye. The retained surface keeps the
#: renderer's default near-white, which is what the product actually draws.
SUBJECT_TINT = (214, 92, 46)
RETAINED_TINT = (215, 215, 220)

#: One pixel per point, because that is what the product's own radius comes to here.
#: `runner.py` logs the fused cloud at a world-space `voxel * 0.25`, and at this scene's
#: voxel and this view's fitted distance that projects to 0.41 px -- sub-pixel. The
#: rasterizer takes an integer pixel radius, so the faithful choice is the smallest it
#: has.
#:
#: The default of 1 is a 3x3 stamp, nine times the area, and it is not a cosmetic
#: difference: at radius 1 the 796k-point cloud covered the retained mesh so completely
#: that drawing or omitting that mesh changed 854 pixels of 691,200 -- 0.12% of the
#: frame. The pair looked identical while the subject behind it was half hidden. The
#: product's own comment beside that radius says default-sized points vanish into an
#: opaque surface at the same depth; oversized ones do the reverse.
FUSED_POINT_RADIUS_PX = 0


def _load_harness(name: str):
    """Import a sibling harness by path, since this directory is not a package."""

    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_ply(o3d, geometry, path: Path) -> dict:
    """Persist a layer and record the bytes, so the pair's shared layers can be proved."""

    if hasattr(geometry, "triangles"):
        o3d.io.write_triangle_mesh(str(path), geometry)
    else:
        o3d.io.write_point_cloud(str(path), geometry)
    data = path.read_bytes()
    return {
        "path": str(path),
        "relative_to_evidence_root": str(path.relative_to(OUT.parent)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def subject_pixels(rv, cam, subject, occluders) -> dict:
    """Pixels of the subject that survive the z-buffer, and the ones that do not.

    Rendered in two passes rather than by colour-matching the delivered frame: the
    renderer shades a mesh by its normals, so a lit surface spans hundreds of values and
    a tolerance wide enough to catch them all also catches the page. Depth is exact.

    ``occluders`` must not contain the subject. A ray cast toward a back-facing sample
    hits the front of the same surface, so `audit_view_occlusion.py` counts the subject
    as its own occluder and is right to. A z-buffer has already resolved that: a single
    rasterizing pass keeps only the nearest subject surface per pixel, so passing the
    subject in here compares its depth against itself, finds no pixel strictly nearer
    than itself, and reports zero visible. That is what the first run of this did.

    The consequence is worth stating rather than hiding: this denominator is the
    subject's screen footprint, where #1014's is its total surface area including back
    faces. The two instruments agree that the defect is there and are not required to
    agree on a number, because they are not measuring the same quantity.
    """

    verts = np.asarray(subject.vertices)
    tris = np.asarray(subject.triangles)
    normals = np.asarray(subject.vertex_normals)

    rgb, depth = rv._blank()
    _, subject_depth = rv._draw_mesh(
        rgb, depth, verts, tris, normals, cam, SUBJECT_TINT
    )
    drawn = np.isfinite(subject_depth)

    rgb, occluder_depth = rv._blank()
    for mesh in occluders:
        rgb, occluder_depth = rv._draw_mesh(
            rgb,
            occluder_depth,
            np.asarray(mesh.vertices),
            np.asarray(mesh.triangles),
            np.asarray(mesh.vertex_normals),
            cam,
            RETAINED_TINT,
        )

    # Strictly nearer wins. A tie is the surface against itself, which the single-pass
    # subject depth has already resolved.
    visible = drawn & (subject_depth < occluder_depth)
    return {
        "subject_pixels_drawn": int(drawn.sum()),
        "subject_pixels_visible": int(visible.sum()),
        "visible_pixel_fraction": float(visible.sum() / max(int(drawn.sum()), 1)),
    }


def main() -> int:
    import open3d as o3d
    from npa.workbench.open3d.runner import REMOVED_VIEW, _view_cameras

    occlusion = _load_harness("audit_view_occlusion")
    rv = _load_harness("render_views")

    OUT.mkdir(parents=True, exist_ok=True)

    cloud, voxel = occlusion.build_eagle(o3d)
    samples, kept, removed = occlusion.reconstruct_and_crop(o3d, cloud, voxel)
    if removed is None or kept is None:
        raise RuntimeError("no removed surface to build a control from")

    camera = _view_cameras(
        {
            "fused": np.asarray(samples.points),
            "mesh": np.asarray(kept.vertices),
            "unsupported": np.asarray(removed.vertices),
        },
        [0.0, 0.0, 1.0],
    )[REMOVED_VIEW]

    layers = {
        "fused": write_ply(o3d, samples, OUT / "D4.fused.ply"),
        "retained": write_ply(o3d, kept, OUT / "D4.retained.ply"),
        "subject": write_ply(o3d, removed, OUT / "D4.subject.ply"),
    }

    # One camera for both frames, fitted by the renderer against the union of everything
    # either frame draws. Fitting each frame to its own contents would move the eye
    # between them and a judge's answer could no longer be attributed to the occluder.
    plan = {
        "viewpoints": {
            "removed_view": {
                "eye": camera["eye"],
                "look_target": camera["look_target"],
                "up": camera["up"],
            }
        },
        "frames": [
            {
                "name": "D4_occluded",
                "out_template": str(OUT / "D4_occluded.png"),
                "viewpoints": ["removed_view"],
                "layers": [
                    {
                        "path": layers["fused"]["path"],
                        "radius": FUSED_POINT_RADIUS_PX,
                    },
                    {"path": layers["retained"]["path"], "tint": list(RETAINED_TINT)},
                    {"path": layers["subject"]["path"], "tint": list(SUBJECT_TINT)},
                ],
            },
            {
                "name": "D4_unoccluded",
                "out_template": str(OUT / "D4_unoccluded.png"),
                "viewpoints": ["removed_view"],
                "layers": [
                    {
                        "path": layers["fused"]["path"],
                        "radius": FUSED_POINT_RADIUS_PX,
                    },
                    {"path": layers["subject"]["path"], "tint": list(SUBJECT_TINT)},
                ],
            },
        ],
        "manifest_out": str(OUT / "render-manifest.json"),
    }
    plan_path = OUT / "render-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))

    argv = sys.argv
    sys.argv = [str(HARNESS / "render_views.py"), str(plan_path)]
    try:
        rv.main()
    finally:
        sys.argv = argv

    cam = rv._camera(camera["eye"], camera["look_target"], camera["up"], rv.W / rv.H)
    measured = {
        "D4_occluded": subject_pixels(rv, cam, removed, [kept]),
        "D4_unoccluded": subject_pixels(rv, cam, removed, []),
    }

    frames = {}
    for name in ("D4_occluded", "D4_unoccluded"):
        data = (OUT / f"{name}.png").read_bytes()
        frames[name] = {
            "path": str(OUT / f"{name}.png"),
            "relative_to_evidence_root": f"vlm-controls-v4/{name}.png",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            **measured[name],
        }

    ratio = (
        frames["D4_occluded"]["visible_pixel_fraction"]
        / frames["D4_unoccluded"]["visible_pixel_fraction"]
    )

    record = {
        "what_this_is": (
            "A matched control pair for an occlusion defect: one frame as the product "
            "shipped the audit view, one as PR #1014 fixed it. One camera, one set of "
            "geometry bytes, differing only in whether the retained surface is drawn."
        ),
        "why_it_is_not_in_controls_v3": (
            "That set covers registration alignment, support crop, and camera framing. "
            "Its passing value for the framing dimension is 'reviewable', which "
            "conflates being inside the frame with being visible. camera_framing_probe "
            "passed this view -- no vertex was cut off -- while 35.2% of its subject was "
            "readable, so occlusion needs a dimension of its own."
        ),
        "camera": {
            "eye": list(map(float, camera["eye"])),
            "look_target": list(map(float, camera["look_target"])),
            "up": list(map(float, camera["up"])),
            "note": (
                "The product's own _view_cameras fitted this for REMOVED_VIEW against "
                "the same geometry keys the blueprint supplies. Shared by both frames; "
                "the renderer solves one eye distance over the union of both."
            ),
        },
        "layers": layers,
        "shared_layers_are_identical_bytes": True,
        "frames": frames,
        "measurement": {
            "instrument": (
                "Pixels the z-buffer awarded to the subject, counted in two passes "
                "through the same rasterizer that wrote the delivered PNGs."
            ),
            "independent_of": (
                "audit_view_occlusion.py, which casts rays against a RaycastingScene "
                "and reports surface area. Two unrelated instruments on one claim."
            ),
            "denominator": (
                "The subject's screen footprint -- the pixels the rasterizer awards it "
                "when nothing else is drawn. audit_view_occlusion.py uses total surface "
                "area including back faces, so the two numbers answer different "
                "questions and are not expected to match. Both show the defect."
            ),
            "occluded_over_unoccluded": ratio,
        },
        "reproducibility": {
            "command": (
                "npa/.venv/bin/python "
                "docs/evidence/open3d/harness/audit_view_vlm_controls.py"
            ),
            "sampling_seed": occlusion.SAMPLING_SEED,
        },
    }
    (OUT / "controls-manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    for name, frame in frames.items():
        print(
            f"{name}: {frame['subject_pixels_visible']}/{frame['subject_pixels_drawn']} "
            f"subject pixels visible = {frame['visible_pixel_fraction']:.4f}",
            flush=True,
        )
    print(f"occluded/unoccluded = {ratio:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
