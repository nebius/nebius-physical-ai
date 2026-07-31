"""Regenerate ``sim2real_sample.mcap``, the browser suite's MCAP fixture.

The mocked Cypress suite serves this file at ``/lichtblick/recordings/sim2real.mcap``
so it can assert the embedded viewer is fed substantive data. It must be produced by
the real emitter (``npa.workflows.sim2real_viz.emit_sim2real_mcap``) rather than
hand-rolled, otherwise the fixture silently drifts from the schemas the viewer
actually receives — e.g. a point cloud missing its ``alpha`` field renders fully
transparent, which a hand-written fixture would never surface.

Run from the repo root with an env that has ``npa`` importable::

    python npa/tests/browser/cypress/fixtures/generate_sim2real_sample_mcap.py

Output is deterministic, so a no-op regeneration produces no diff.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "sim2real_sample.mcap"
ENV_IDS = ("env-00001", "env-00002", "env-00003")
FRAMES_PER_ENV = 2
FRAME_SIZE = 48
POINTS_PER_CLOUD = 1000


def _render_like_frame(index: int) -> np.ndarray:
    """A deterministic 48x48 RGB frame with realistic mid-range brightness.

    The suite asserts decoded frames are neither a dark-noise regression (mean > 60)
    nor saturated (mean < 250), so this renders a smooth gradient with a bright
    highlight block, giving a mean near 130.
    """

    ramp = np.linspace(60, 200, FRAME_SIZE, dtype=np.float32)
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)
    frame[:, :, 0] = ramp[None, :]
    frame[:, :, 1] = ramp[:, None]
    frame[:, :, 2] = 140.0 + 10.0 * index
    # A bright "object" block so the frame is not a flat gradient.
    frame[12:28, 16:36, :] = 225.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _write_png(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="PNG")


def _build_run_tree(root: Path) -> tuple[dict, dict]:
    from npa.workflows.sim2real_viz import POINTCLOUD_SUBDIR

    renders = root / "eval" / "heldout" / "renders"
    episodes = []
    for env_id in ENV_IDS:
        names = []
        for index in range(FRAMES_PER_ENV):
            name = f"camera-{index:03d}.png"
            _write_png(renders / env_id / name, _render_like_frame(index))
            names.append(name)
        episodes.append({"env_id": env_id, "frames": names})

    # GPU-reconstructed point clouds for the 3D panel (primary env only, matching
    # the held-out eval's output layout).
    cloud_dir = renders / POINTCLOUD_SUBDIR / ENV_IDS[0]
    cloud_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260730)
    for index in range(FRAMES_PER_ENV):
        xyz = rng.uniform(-0.6, 0.6, size=(POINTS_PER_CLOUD, 3)).astype("float32")
        rgb = rng.integers(40, 240, size=(POINTS_PER_CLOUD, 3), dtype="uint8")
        np.savez_compressed(cloud_dir / f"cloud-{index:04d}.npz", xyz=xyz, rgb=rgb)

    heldout_report = {
        "success_rate": 0.667,
        "sim_backend": "isaac",
        "per_env": [
            {"env_id": ENV_IDS[0], "success": True, "score": 0.91},
            {"env_id": ENV_IDS[1], "success": True, "score": 0.78},
            {"env_id": ENV_IDS[2], "success": False, "score": 0.12},
        ],
        "render_manifest": {
            "schema": "npa.sim2real.heldout_renders.v1",
            "sim_backend": "isaac",
            "episodes": episodes,
        },
    }
    inner_evidence = {"iterations": [], "reward_trend": [0.1, 0.2]}
    return inner_evidence, heldout_report


def main() -> int:
    from npa.workflows.sim2real_viz import emit_sim2real_mcap

    with tempfile.TemporaryDirectory(prefix="npa-mcap-fixture-") as tmp:
        root = Path(tmp)
        inner_evidence, heldout_report = _build_run_tree(root)
        result = emit_sim2real_mcap(
            local_dir=root,
            inner_evidence=inner_evidence,
            heldout_report=heldout_report,
            output_mcap=FIXTURE,
        )
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
    print(
        "camera={camera} pointcloud={cloud} scalar={scalar} transform={tf}".format(
            camera=result.camera_message_count,
            cloud=result.pointcloud_message_count,
            scalar=result.scalar_message_count,
            tf=result.transform_message_count,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
