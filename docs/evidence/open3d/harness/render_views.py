"""Deterministic CPU renders of delivered Open3D artifacts, for review only.

This is evidence tooling, not shipped product code. Open3D's own offscreen
renderer needs Vulkan, which is unavailable headless and not worth adding to a
CPU geometry image, so this rasterizes the delivered PLY bytes directly with a
z-buffer. That has one property the review needs above all: the camera is exactly
the same for baseline and candidate, so a difference in the picture is a
difference in the geometry and not in the viewpoint.

Reads only the artifact bytes it is pointed at. Writes PNGs plus a manifest
recording the camera, the source hashes, and the pixel hash of every frame.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 960, 720
FOV_DEGREES = 55.0
#: Fraction of the half-frame the content is fitted into, leaving a visible margin.
#: Hand-placed eyes framed one angle correctly and clipped the other two; the
#: top-down ran off the bottom edge and cut the axis triad at the top.
FRAME_FILL = 0.92
#: A near-black page. Chosen by measurement, not taste: this scan's content is
#: overwhelmingly light -- median luminance 174, only ~1% below 34 -- so the old 250
#: page gave the typical pixel just 76 levels of contrast. Holding the content fixed
#: and varying only the page, darkening to this lifts the median to 158 and cuts the
#: share of content within 24 levels of the page from 4.74% to 1.76%.
#:
#: Pure black measures better still (median 174, 0.52%). It is not used because the
#: clipping and legibility audit defines content as "differs from the page", so a
#: page equal to a possible content value blinds the measurement to black geometry
#: at the border -- the exact defect being measured. The offset costs ~16 levels.
#: Figures: evidence/open3d/background-choice-sweep-refit.json.
BACKGROUND = (16, 17, 20)
#: Ink for the scale bar and its label, light enough to read on that page.
INK = (232, 232, 236)


def _read_ply(path: Path):
    """Minimal binary/ascii PLY reader for the vertex/face subset Open3D writes."""
    import plyfile  # noqa: F401  (import guarded by caller)


def _load(path: Path):
    import open3d as o3d

    text = path.read_bytes()[:2048].decode("latin-1")
    if "element face" in text and "\nelement face 0\n" not in text:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            return (
                np.asarray(mesh.vertices),
                np.asarray(mesh.triangles),
                np.asarray(mesh.vertex_normals),
                None,
            )
    cloud = o3d.io.read_point_cloud(str(path))
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    return np.asarray(cloud.points), None, None, colors


def _camera(eye, target, up, aspect):
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    focal = 1.0 / np.tan(np.radians(FOV_DEGREES) / 2.0)
    return eye, np.stack([right, true_up, forward]), focal, aspect


def _project(points, cam):
    eye, basis, focal, aspect = cam
    local = (points - eye) @ basis.T
    depth = local[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    x = (local[:, 0] / safe) * focal / aspect
    y = (local[:, 1] / safe) * focal
    px = (x + 1.0) * 0.5 * W
    py = (1.0 - (y + 1.0) * 0.5) * H
    return np.stack([px, py], axis=1), depth


def _blank():
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, :] = BACKGROUND
    return rgb, np.full((H, W), np.inf)


def _fit_eye(subject, framed, eye, look_target, up, aspect):
    """Solve the eye distance that frames every point, keeping the plan's direction.

    Closed form rather than iterative: moving the eye back along the view axis by
    ``d`` adds ``d`` to every point's depth and changes neither its lateral nor its
    vertical offset, so each point imposes a linear lower bound on ``d`` and the
    binding one is simply the largest.

    ``subject`` is centred and ``framed`` is kept inside the frame, which are not the
    same set. The reference triad has to be framed or it hangs off an edge, as it did
    in the delivered top-down view, but letting it drag the centring pushes the scan
    itself off to one side -- trading a clipped marker for the empty third of frame
    the review reported.

    Returns the fitted eye and the recentred target. The view direction is the plan's
    and is deliberately untouched, so this fixes framing without quietly choosing a
    more flattering angle.
    """

    subject = np.asarray(subject, float)
    framed = np.asarray(framed, float)
    target = np.asarray(look_target, float)
    forward = target - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    def in_view(points):
        relative = points - target
        return relative @ right, relative @ true_up, relative @ forward

    subject_lateral, subject_vertical, _ = in_view(subject)
    mid_lateral = float((subject_lateral.min() + subject_lateral.max()) / 2.0)
    mid_vertical = float((subject_vertical.min() + subject_vertical.max()) / 2.0)
    centre = target + mid_lateral * right + mid_vertical * true_up

    lateral, vertical, along = in_view(framed)
    lateral = lateral - mid_lateral
    vertical = vertical - mid_vertical

    half = np.tan(np.radians(FOV_DEGREES) / 2.0)
    distance = max(
        float((np.abs(lateral) / (FRAME_FILL * half * aspect) - along).max()),
        float((np.abs(vertical) / (FRAME_FILL * half) - along).max()),
    )
    # Every point must also stay in front of the eye, which the fit above does not
    # guarantee for a subject deep along the view axis.
    span = float(np.linalg.norm(framed.max(axis=0) - framed.min(axis=0)))
    distance = max(distance, -float(along.min()) + 0.05 * span)
    return centre - forward * distance, centre


def _draw_points(rgb, zbuf, points, colors, cam, radius=1):
    xy, depth = _project(points, cam)
    keep = depth > 1e-4
    xy, depth = xy[keep], depth[keep]
    cols = (
        (colors[keep] * 255).astype(np.uint8)
        if colors is not None
        else np.full((len(xy), 3), 70, np.uint8)
    )
    order = np.argsort(-depth)  # far to near
    xi = np.rint(xy[:, 0]).astype(int)[order]
    yi = np.rint(xy[:, 1]).astype(int)[order]
    dz, cc = depth[order], cols[order]
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = xi + dx, yi + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            ax, ay, ad, ac = x[ok], y[ok], dz[ok], cc[ok]
            nearer = ad < zbuf[ay, ax]
            zbuf[ay[nearer], ax[nearer]] = ad[nearer]
            rgb[ay[nearer], ax[nearer]] = ac[nearer]
    return rgb, zbuf


def _draw_mesh(rgb, zbuf, verts, tris, normals, cam, tint):
    """Flat-shaded z-buffered triangles. Slow and simple; correctness over speed."""
    xy, depth = _project(verts, cam)
    light = np.array([0.4, 0.6, 0.7])
    light /= np.linalg.norm(light)
    tri_n = normals[tris].mean(axis=1)
    norms = np.linalg.norm(tri_n, axis=1, keepdims=True)
    tri_n = tri_n / np.where(norms < 1e-9, 1.0, norms)
    shade = 0.35 + 0.65 * np.abs(tri_n @ light)
    tri_depth = depth[tris].mean(axis=1)
    for idx in np.argsort(-tri_depth):
        tri = tris[idx]
        if np.any(depth[tri] <= 1e-4):
            continue
        p = xy[tri]
        x0, x1 = int(np.floor(p[:, 0].min())), int(np.ceil(p[:, 0].max()))
        y0, y1 = int(np.floor(p[:, 1].min())), int(np.ceil(p[:, 1].max()))
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        ax, ay, bx, by, cx, cy = (
            p[0, 0],
            p[0, 1],
            p[1, 0],
            p[1, 1],
            p[2, 0],
            p[2, 1],
        )
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-12:
            continue
        w1 = ((gx - ax) * (cy - ay) - (cx - ax) * (gy - ay)) / det
        w2 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / det
        inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        if not inside.any():
            continue
        w0 = 1.0 - w1 - w2
        d = w0 * depth[tri[0]] + w1 * depth[tri[1]] + w2 * depth[tri[2]]
        sel = inside & (d < zbuf[gy, gx])
        if not sel.any():
            continue
        colour = np.clip(np.asarray(tint, float) * shade[idx], 0, 255).astype(np.uint8)
        zbuf[gy[sel], gx[sel]] = d[sel]
        rgb[gy[sel], gx[sel]] = colour
    return rgb, zbuf


def _draw_segment(rgb, zbuf, p0, p1, cam, colour, width=2):
    xy, depth = _project(np.stack([p0, p1]), cam)
    if np.any(depth <= 1e-4):
        return rgb, zbuf
    steps = int(max(abs(xy[1, 0] - xy[0, 0]), abs(xy[1, 1] - xy[0, 1]))) + 1
    for t in np.linspace(0.0, 1.0, max(steps, 2)):
        x, y = xy[0] + t * (xy[1] - xy[0])
        d = depth[0] + t * (depth[1] - depth[0])
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                xi, yi = int(round(x)) + dx, int(round(y)) + dy
                if 0 <= xi < W and 0 <= yi < H and d <= zbuf[yi, xi] + 1e-3:
                    rgb[yi, xi] = colour
    return rgb, zbuf


def _annotate(rgb, cam, origin, unit, label):
    """A metre scale bar and the unit, so extents are readable without the JSON."""
    from PIL import ImageDraw

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    # Project a one-unit segment at the look target's depth to get its pixel length.
    eye, basis, focal, aspect = cam
    target = np.asarray(cam[0], float)
    xy, _ = _project(
        np.stack([np.asarray(origin, float), np.asarray(origin, float) + unit]), cam
    )
    pixels = float(np.linalg.norm(xy[1] - xy[0]))
    bar = int(min(max(pixels, 20), W * 0.45))
    x0, y0 = 28, H - 42
    draw.rectangle([x0, y0, x0 + bar, y0 + 7], fill=INK)
    for tick in (x0, x0 + bar):
        draw.rectangle([tick - 1, y0 - 6, tick + 1, y0 + 13], fill=INK)
    draw.text((x0, y0 - 24), label, fill=INK)
    return np.asarray(img)


def render(spec: dict, cam, plan: dict) -> bytes:
    rgb, zbuf = _blank()
    for layer in spec["layers"]:
        pts, tris, normals, colors = _load(Path(layer["path"]))
        if tris is None:
            rgb, zbuf = _draw_points(
                rgb, zbuf, pts, colors, cam, radius=layer.get("radius", 1)
            )
        else:
            rgb, zbuf = _draw_mesh(
                rgb, zbuf, pts, tris, normals, cam, layer.get("tint", (215, 215, 220))
            )
    if plan.get("axes_origin") is not None:
        origin = np.asarray(plan["axes_origin"], float)
        length = float(plan.get("axes_length", 1.0))
        for direction, colour in (
            ((1, 0, 0), (200, 40, 40)),
            ((0, 1, 0), (40, 160, 60)),
            ((0, 0, 1), (50, 90, 210)),
        ):
            rgb, zbuf = _draw_segment(
                rgb,
                zbuf,
                origin,
                origin + np.asarray(direction, float) * length,
                cam,
                colour,
            )
        rgb = _annotate(
            rgb,
            cam,
            origin,
            np.asarray([length, 0.0, 0.0]),
            plan.get("scale_label", f"{length:g} unit"),
        )
    out = Path(spec["out"])
    Image.fromarray(rgb).save(out)
    return out.read_bytes()


#: Container mount point of the evidence tree, and the prefix a host consumer needs.
#: The renderer runs inside the image with the open3d evidence directory bind-mounted at
#: /data, so every path it is handed is container-absolute. Recording only those paths is
#: what made a delivered control set look like missing bytes: the VLM lane resolved
#: /data/vlm/controls-v3/*.png against the host, found nothing, and reported the frames
#: absent while they sat beside their own manifest. A manifest a consumer cannot resolve
#: is not a record, so every path is written twice.
MOUNT_PREFIX = "/data/"
EVIDENCE_PREFIX = "open3d/"


def _portable(path: str) -> str:
    """Path relative to the evidence root, for a consumer outside this container."""

    if path.startswith(MOUNT_PREFIX):
        return EVIDENCE_PREFIX + path[len(MOUNT_PREFIX) :]
    return path


def main() -> int:
    plan = json.loads(Path(sys.argv[1]).read_text())
    manifest = {
        "renderer": "npa evidence harness z-buffer rasterizer",
        "resolution": [W, H],
        "fov_degrees": FOV_DEGREES,
        "note": (
            "Offscreen CPU renders of the delivered PLY bytes, not screenshots of "
            "the Rerun web viewer. Same camera for every matched pair, so a "
            "difference in the picture is a difference in the geometry."
        ),
        "framing_note": (
            "The plan's eye positions supply the view direction only; the distance "
            "and the in-plane centre are solved against the union of every frame "
            "drawn from that viewpoint, so no frame is clipped and matched pairs "
            "still share one camera. Hand-placed distances framed one angle and "
            "clipped the others."
        ),
        "requested_viewpoints": plan["viewpoints"],
        "fitted_viewpoints": {},
        "frames": [],
        "path_note": (
            "Paths beginning /data/ are the container mount that produced these frames "
            "and are kept as the literal argv the renderer saw. They do not resolve on "
            "the host; use the *_relative_to_evidence_root keys."
        ),
    }
    fitted = manifest["fitted_viewpoints"]
    for view_name, view in plan["viewpoints"].items():
        members = [
            spec
            for spec in plan["frames"]
            if view_name in spec.get("viewpoints", [view_name])
        ]
        if not members:
            continue
        # Fit once per viewpoint, against the union of every frame drawn from it
        # plus the axis triad. Fitting each frame to its own contents would frame
        # them all perfectly and destroy the only property this renderer exists to
        # guarantee: baseline and candidate share a camera, so a difference in the
        # picture is a difference in the geometry. The union keeps that, and the
        # before frame's larger extent sets the scale for both -- which is the
        # honest direction for a before/after.
        subject = np.vstack(
            [
                _load(Path(layer["path"]))[0]
                for spec in members
                for layer in spec["layers"]
            ]
        )
        framed = [subject]
        if plan.get("axes_origin") is not None:
            origin = np.asarray(plan["axes_origin"], float)
            length = float(plan.get("axes_length", 1.0))
            framed.append(origin + np.eye(3) * length)
            framed.append(origin[None, :])
        eye, target = _fit_eye(
            subject,
            np.vstack(framed),
            view["eye"],
            view["look_target"],
            view["up"],
            W / H,
        )
        cam = _camera(eye, target, view["up"], W / H)
        fitted[view_name] = {
            "requested_eye": list(view["eye"]),
            "fitted_eye": [float(v) for v in eye],
            "recentred_look_target": [float(v) for v in target],
            "up": list(view["up"]),
            "frame_fill": FRAME_FILL,
            "fitted_against": sorted(
                {layer["path"] for spec in members for layer in spec["layers"]}
            ),
        }
        for spec in members:
            out = spec["out_template"].format(view=view_name)
            png = render({**spec, "out": out}, cam, plan)
            manifest["frames"].append(
                {
                    "name": f"{spec['name']}__{view_name}",
                    "viewpoint": view_name,
                    "out": out,
                    "out_relative_to_evidence_root": _portable(out),
                    "frame_sha256": hashlib.sha256(png).hexdigest(),
                    "layers": [
                        {
                            "path": layer["path"],
                            "path_relative_to_evidence_root": _portable(layer["path"]),
                            "source_sha256": hashlib.sha256(
                                Path(layer["path"]).read_bytes()
                            ).hexdigest(),
                        }
                        for layer in spec["layers"]
                    ],
                }
            )
            print(
                "rendered",
                manifest["frames"][-1]["name"],
                manifest["frames"][-1]["frame_sha256"][:16],
            )
    Path(plan["manifest_out"]).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
