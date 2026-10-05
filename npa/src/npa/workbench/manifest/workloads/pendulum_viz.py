"""Newton XPBD double-pendulum on GPU, rendered to MP4.

A real pre-existing workload (the 2026-09-27 demo sim), onboarded onto the
manifest path by descriptor only. The container image stays the pinned
pytorch base; the workload's own dependencies (newton, matplotlib,
imageio-ffmpeg) come from the descriptor's `environment.pip`.
"""

import argparse
import json
import os
import time

import numpy as np


def run_sim(steps: int, dt: float) -> dict:
    import warp as wp

    wp.init()
    import newton

    device = "cuda:0"
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))

    link1 = builder.add_link(
        mass=1.0,
        label="link1",
        xform=wp.transform((0.0, -0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    j1 = builder.add_joint_revolute(
        parent=-1,
        child=link1,
        axis=(0.0, 0.0, 1.0),
        label="joint1",
        child_xform=wp.transform((0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    builder.add_shape_box(link1, hx=0.05, hy=0.5, hz=0.05, label="link1_box")

    link2 = builder.add_link(
        mass=1.0,
        label="link2",
        xform=wp.transform((0.0, -1.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    j2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        axis=(0.0, 0.0, 1.0),
        label="joint2",
        parent_xform=wp.transform((0.0, -0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
        child_xform=wp.transform((0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    builder.add_shape_box(link2, hx=0.05, hy=0.5, hz=0.05, label="link2_box")
    builder.add_articulation([j1, j2], label="double_pendulum")

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model)
    state0, state1 = model.state(), model.state()
    control = model.control()

    def quat_z(deg):
        a = np.radians(deg) / 2
        return (0.0, 0.0, np.sin(a), np.cos(a))

    q = state0.body_q.numpy()
    for b, ang in [(0, 60.0), (1, 60.0)]:
        p = q[b, :3]
        th = np.radians(ang)
        x, y = p[0], p[1]
        q[b, 0] = x * np.cos(th) - y * np.sin(th)
        q[b, 1] = x * np.sin(th) + y * np.cos(th)
        q[b, 3:7] = quat_z(ang)
    state0.body_q.assign(wp.array(q, dtype=wp.transformf, device=device))
    state1.body_q.assign(wp.array(q, dtype=wp.transformf, device=device))

    t0 = time.time()
    traj = [state0.body_q.numpy().copy()]
    for _ in range(steps):
        contacts = model.collide(state0)
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0
        traj.append(state0.body_q.numpy().copy())
    elapsed = time.time() - t0
    traj = np.stack(traj)
    return {
        "traj": traj,
        "elapsed_s": elapsed,
        "newton": newton.__version__,
        "warp": wp.__version__,
    }


def render_mp4(traj: np.ndarray, dt: float, out_path: str) -> None:
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    n_frames = traj.shape[0]
    p1, p2 = traj[:, 0, :2], traj[:, 1, :2]
    j0 = np.zeros(2)
    j1 = 2 * p1 - j0
    j2 = 2 * p2 - j1

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 1.0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.plot(0, 0, "ko", ms=9, zorder=5)

    (link1_line,) = ax.plot([], [], "-", lw=6, color="#1f77b4", solid_capstyle="round")
    (link2_line,) = ax.plot([], [], "-", lw=6, color="#ff7f0e", solid_capstyle="round")
    (joints,) = ax.plot([], [], "ko", ms=6, zorder=5)
    (trail1,) = ax.plot([], [], "-", lw=1, color="#1f77b4", alpha=0.45)
    (trail2,) = ax.plot([], [], "-", lw=1, color="#ff7f0e", alpha=0.45)
    title = ax.set_title("")

    stride = max(1, n_frames // 240)
    frames = list(range(0, n_frames, stride))
    if frames[-1] != n_frames - 1:
        frames.append(n_frames - 1)

    def draw(f):
        link1_line.set_data([j0[0], j1[f, 0]], [j0[1], j1[f, 1]])
        link2_line.set_data([j1[f, 0], j2[f, 0]], [j1[f, 1], j2[f, 1]])
        joints.set_data([j0[0], j1[f, 0], j2[f, 0]], [j0[1], j1[f, 1], j2[f, 1]])
        trail1.set_data(j1[: f + 1 : stride, 0], j1[: f + 1 : stride, 1])
        trail2.set_data(j2[: f + 1 : stride, 0], j2[: f + 1 : stride, 1])
        title.set_text(
            f"Newton XPBD double pendulum on cuda:0 — "
            f"step {f}/{n_frames - 1} (t={f * dt:.2f}s sim)"
        )
        return [link1_line, link2_line, joints, trail1, trail2, title]

    anim = FuncAnimation(fig, draw, frames=frames, blit=False, interval=50)
    anim.save(
        out_path,
        writer="ffmpeg",
        fps=24,
        dpi=110,
        metadata={
            "title": "Newton GPU double pendulum via manifest",
            "comment": "simulated on RTX PRO 6000, onboarded by descriptor",
        },
    )
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out-mp4", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    dt = 1.0 / 60.0

    sim = run_sim(args.steps, dt)
    traj = sim["traj"]
    render_mp4(traj, dt, args.out_mp4)
    mp4_bytes = os.path.getsize(args.out_mp4)

    import torch

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    summary = {
        "device_ok": torch.cuda.is_available(),
        "device": device_name,
        "steps": args.steps,
        "dt": dt,
        "frames": int(traj.shape[0]),
        "sim_elapsed_s": round(sim["elapsed_s"], 3),
        "mp4_bytes": mp4_bytes,
        "newton": sim["newton"],
        "warp": sim["warp"],
        "traj_checksum": float(traj.sum(dtype=np.float64)),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
