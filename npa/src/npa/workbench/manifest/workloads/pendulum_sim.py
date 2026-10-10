"""Shared Newton double-pendulum sim for the manifest workloads.

The 2026-09-27 GPU validation sim, factored for reuse: returns the
body_q trajectory plus metadata. Imported by payload scripts, not a
workload itself.
"""

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
    return {
        "traj": np.stack(traj),
        "elapsed_s": elapsed,
        "newton": newton.__version__,
        "warp": wp.__version__,
    }


def joint_positions(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World pivot j0 and per-frame joint positions j1, j2 (sim x,y plane)."""
    p1, p2 = traj[:, 0, :2], traj[:, 1, :2]
    j0 = np.zeros(2)
    return j0, 2 * p1 - j0, 2 * p2 - (2 * p1 - j0)
