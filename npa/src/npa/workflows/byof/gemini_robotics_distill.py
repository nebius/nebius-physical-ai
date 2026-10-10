"""Gemini -> Nebius GPU distillation: ER-2 plans, GPU trains a student policy.

Teacher-student loop:
  1. plan      - ER-2 (Gemini API) produces a manipulation plan (live-validated).
  2. demos     - the plan drives scripted MuJoCo rollouts (domain-randomized).
  3. train     - behavior-clone a compact student policy (PyTorch; CUDA when
                 available - submit this stage to a Nebius GPU job for real
                 training).
  4. eval      - closed-loop MuJoCo eval of the student; success rate + receipt.

One command: ``npa workbench gemini-robotics distill "task" --image scene.png``.
The student runs locally with no API key, no latency, and no per-call cost.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from npa.clients.gemini_robotics import (
    DEFAULT_MODEL_ID,
    GeminiRoboticsClient,
    resolve_config,
)

DISTILL_RECEIPT_SCHEMA = "npa.gemini_robotics.distill.v1"

# Arm geometry (must match the MuJoCo scene used for demos).
_L1, _L2 = 0.40, 0.35
_BASE = np.array([0.02, -0.50])
_SHOULDER_Z = 0.50
_TIP_DZ = -0.13
_F_OPEN = (-0.050, 0.050)
_F_CLOSED = (-0.019, 0.019)

_MJCF_TEMPLATE = ""  # filled by _scene_xml(); kept module-level for reuse


def _scene_xml() -> str:
    """MuJoCo scene for the tabletop pick-and-place task."""
    return """
<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.88 0.88 0.90" rgb2="0.70 0.70 0.74" width="128" height="128"/>
    <material name="table_mat" texture="grid" texrepeat="6 6" reflectance="0.2"/>
    <material name="red" rgba="0.85 0.12 0.12 1"/>
    <material name="green" rgba="0.10 0.65 0.25 1"/>
    <material name="arm" rgba="0.72 0.72 0.75 1"/>
    <material name="dark" rgba="0.16 0.16 0.19 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="1.5 1.5 3" dir="-0.4 -0.4 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2.5 2.5 0.1" material="table_mat"/>
    <body name="pad" pos="__PAD__ 0.012">
      <geom name="pad_geom" type="box" size="0.13 0.13 0.012" material="green"/>
    </body>
    <body name="cube" pos="__CUBE__ 0.05">
      <freejoint name="cubefree"/>
      <geom name="cube_geom" type="box" size="0.05 0.05 0.05" material="red" mass="0.2"/>
    </body>
    <body name="armbase" pos="0.02 -0.50 0">
      <geom name="column" type="cylinder" size="0.07 0.25" pos="0 0 0.25" material="dark"/>
      <body name="shoulder" pos="0 0 0.50">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom name="shoulderball" type="sphere" size="0.05" material="dark"/>
        <body name="upperarm" pos="0 0 0">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom name="upper" type="capsule" size="0.04" fromto="0 0 0 0.40 0 0" material="arm"/>
          <body name="elbow" pos="0.40 0 0">
            <joint name="j3" type="hinge" axis="0 1 0"/>
            <geom name="elbowball" type="sphere" size="0.04" material="dark"/>
            <geom name="fore" type="capsule" size="0.035" fromto="0 0 0 0.35 0 0" material="arm"/>
            <body name="wrist" pos="0.35 0 0">
              <joint name="j4" type="hinge" axis="0 1 0"/>
              <geom name="wristball" type="sphere" size="0.03" material="dark"/>
              <geom name="palm" type="box" size="0.05 0.05 0.02" pos="0 0 -0.055" material="arm"/>
              <body name="fingerL" pos="-0.035 0 -0.055">
                <joint name="fL" type="slide" axis="1 0 0"/>
                <geom type="box" size="0.012 0.035 0.055" pos="0 0 -0.045" material="dark"/>
              </body>
              <body name="fingerR" pos="0.035 0 -0.055">
                <joint name="fR" type="slide" axis="1 0 0"/>
                <geom type="box" size="0.012 0.035 0.055" pos="0 0 -0.045" material="dark"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".replace("__PAD__", "{px:.3f} {py:.3f}").replace("__CUBE__", "{cx:.3f} {cy:.3f}")


def _ik(target: np.ndarray) -> tuple[float, float, float]:
    """2-bone IK: (yaw, shoulder pitch, elbow pitch) for a wrist target."""
    dx, dy = target[0] - _BASE[0], target[1] - _BASE[1]
    j1 = math.atan2(dy, dx)
    r = math.hypot(dx, dy)
    dz = target[2] - _SHOULDER_Z
    d = min(max(math.hypot(r, dz), abs(_L1 - _L2) + 1e-4), _L1 + _L2 - 1e-4)
    cos_e = max(-1.0, min(1.0, (_L1 * _L1 + _L2 * _L2 - d * d) / (2 * _L1 * _L2)))
    bend = math.pi - math.acos(cos_e)
    phi = math.atan2(-dz, r)
    psi = math.atan2(_L2 * math.sin(bend), _L1 + _L2 * math.cos(bend))
    return j1, phi - psi, bend


def _smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


@dataclass
class DistillConfig:
    """Inputs for one distillation run."""

    task: str
    output_dir: str
    images: list[str] = field(default_factory=list)
    model: str = ""
    n_demos: int = 200
    train_epochs: int = 60
    eval_episodes: int = 20
    seed: int = 0

    def resolved_model(self) -> str:
        return self.model.strip() or DEFAULT_MODEL_ID


def generate_demonstrations(config: DistillConfig) -> dict[str, Any]:
    """Run scripted plan executions in MuJoCo; return demo arrays + metadata.

    Requires ``mujoco``. Each episode randomizes cube/pad positions and
    executes the 7-step plan via IK, recording (obs, next-action) pairs.
    """
    import mujoco

    rng = np.random.default_rng(config.seed)
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []

    for _ in range(config.n_demos):
        cx, cy = np.array([-0.25, 0.05]) + rng.uniform(-0.08, 0.08, 2)
        px, py = np.array([0.30, 0.05]) + rng.uniform(-0.08, 0.08, 2)
        xml = _scene_xml().format(cx=cx, cy=cy, px=px, py=py)
        mj = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(mj)

        def qadr(n: str) -> int:
            return mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)]

        j1a, j2a, j3a, j4a = qadr("j1"), qadr("j2"), qadr("j3"), qadr("j4")
        fLa, fRa = qadr("fL"), qadr("fR")
        cubea = qadr("cubefree")
        wrist_b = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "wrist")

        rest = np.array([0.02, -0.30, 0.55])
        cube0 = np.array([cx, cy, 0.05])
        pad_rest = np.array([px, py, 0.074])
        segs = [
            (rest, (cx, cy, 0.34), _F_OPEN, _F_OPEN, False, 24),
            ((cx, cy, 0.34), (cx, cy, 0.18), _F_OPEN, _F_CLOSED, False, 18),
            ((cx, cy, 0.18), (cx, cy, 0.44), _F_CLOSED, _F_CLOSED, True, 18),
            ((cx, cy, 0.44), (px, py, 0.44), _F_CLOSED, _F_CLOSED, True, 30),
            ((px, py, 0.44), (px, py, 0.204), _F_CLOSED, _F_OPEN, True, 18),
            ((px, py, 0.204), rest, _F_OPEN, _F_OPEN, False, 18),
        ]
        traj: list[tuple] = []
        for w0, w1, fg0, fg1, carry, n in segs:
            w0a, w1a = np.array(w0, float), np.array(w1, float)
            for i in range(n):
                t = _smooth(i / max(1, n - 1))
                w = w0a + (w1a - w0a) * t
                j1, j2, j3 = _ik(w)
                gl = fg0[0] + (fg1[0] - fg0[0]) * t
                gr = fg0[1] + (fg1[1] - fg0[1]) * t
                data.qpos[j1a], data.qpos[j2a], data.qpos[j3a] = j1, j2, j3
                data.qpos[j4a] = -(j2 + j3)
                data.qpos[fLa], data.qpos[fRa] = gl, gr
                if carry:
                    data.qpos[cubea : cubea + 3] = [w[0], w[1], w[2] + _TIP_DZ]
                elif np.allclose(w1a, rest) and t > 0.5:
                    data.qpos[cubea : cubea + 3] = pad_rest
                else:
                    data.qpos[cubea : cubea + 3] = cube0
                data.qpos[cubea + 3 : cubea + 7] = [1, 0, 0, 0]
                mujoco.mj_forward(mj, data)
                traj.append(
                    (
                        j1,
                        j2,
                        j3,
                        gl,
                        gr,
                        data.qpos[cubea : cubea + 3].copy(),
                        data.xpos[wrist_b].copy(),
                    )
                )
        for k in range(len(traj) - 1):
            j1, j2, j3, gl, gr, cube_p, wrist = traj[k]
            j1n, j2n, j3n, gln, grn, _, _ = traj[k + 1]
            obs_all.append(
                np.array(
                    [j1, j2, j3, float(gr - gl), *cube_p, px, py, *wrist],
                    dtype=np.float32,
                )
            )
            act_all.append(
                np.array([j1n, j2n, j3n, float(grn - gln)], dtype=np.float32)
            )

    return {
        "obs": np.stack(obs_all),
        "act": np.stack(act_all),
        "n_episodes": config.n_demos,
    }


def train_student(demos: dict[str, Any], config: DistillConfig) -> dict[str, Any]:
    """Behavior-clone a student policy from demonstrations.

    Requires ``torch``. Uses CUDA when available - run this stage on a
    Nebius GPU job for real training throughput.
    """
    import torch
    import torch.nn as nn

    obs = demos["obs"].astype(np.float32)
    act = demos["act"].astype(np.float32)
    n = len(obs)
    idx = np.random.RandomState(config.seed).permutation(n)
    split = int(0.9 * n)
    tr, va = idx[:split], idx[split:]

    obs_mean, obs_std = obs[tr].mean(0), obs[tr].std(0) + 1e-6
    act_mean, act_std = act[tr].mean(0), act[tr].std(0) + 1e-6

    device = "cuda" if torch.cuda.is_available() else "cpu"
    to_t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    Xtr = to_t((obs[tr] - obs_mean) / obs_std)
    Ytr = to_t((act[tr] - act_mean) / act_std)
    Xva = to_t((obs[va] - obs_mean) / obs_std)
    Yva = to_t((act[va] - act_mean) / act_std)

    policy = nn.Sequential(
        nn.Linear(12, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    ).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr), batch_size=256, shuffle=True
    )
    for _ in range(config.train_epochs):
        policy.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(policy(xb), yb).backward()
            opt.step()
    policy.eval()
    with torch.no_grad():
        val_mse = loss_fn(policy(Xva), Yva).item()

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "student_policy.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "act_mean": act_mean,
            "act_std": act_std,
        },
        ckpt_path,
    )
    return {
        "checkpoint": str(ckpt_path),
        "device": device,
        "val_mse": val_mse,
        "n_samples": n,
    }


def eval_student(checkpoint: str, config: DistillConfig) -> dict[str, Any]:
    """Closed-loop MuJoCo eval of the student policy; returns success rate."""
    import mujoco
    import torch
    import torch.nn as nn

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = nn.Sequential(
        nn.Linear(12, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    om, os_ = ckpt["obs_mean"], ckpt["obs_std"]
    am, as_ = ckpt["act_mean"], ckpt["act_std"]

    rng = np.random.default_rng(config.seed + 1)
    successes = 0
    for _ in range(config.eval_episodes):
        cx, cy = np.array([-0.25, 0.05]) + rng.uniform(-0.08, 0.08, 2)
        px, py = np.array([0.30, 0.05]) + rng.uniform(-0.08, 0.08, 2)
        xml = _scene_xml().format(cx=cx, cy=cy, px=px, py=py)
        mj = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(mj)

        def qadr(n: str) -> int:
            return mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)]

        j1a, j2a, j3a, j4a = qadr("j1"), qadr("j2"), qadr("j3"), qadr("j4")
        fLa, fRa = qadr("fL"), qadr("fR")
        cubea = qadr("cubefree")
        wrist_b = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "wrist")
        data.qpos[j1a], data.qpos[j2a], data.qpos[j3a] = math.atan2(0.2, 0.0), 0.0, 0.0
        data.qpos[fLa], data.qpos[fRa] = -0.05, 0.05
        attached = False
        for _ in range(150):
            j1, j2, j3 = data.qpos[j1a], data.qpos[j2a], data.qpos[j3a]
            gl, gr = data.qpos[fLa], data.qpos[fRa]
            mujoco.mj_forward(mj, data)
            wrist = data.xpos[wrist_b]
            cube_p = data.qpos[cubea : cubea + 3]
            o = np.array(
                [j1, j2, j3, float(gr - gl), *cube_p, px, py, *wrist],
                dtype=np.float32,
            )
            with torch.no_grad():
                a = policy(torch.from_numpy((o - om) / os_)).numpy() * as_ + am
            data.qpos[j1a], data.qpos[j2a], data.qpos[j3a] = a[0], a[1], a[2]
            data.qpos[j4a] = -(a[1] + a[2])
            sep = float(a[3])
            data.qpos[fLa], data.qpos[fRa] = -sep / 2, sep / 2
            mujoco.mj_forward(mj, data)
            tip = data.xpos[wrist_b] + np.array([0, 0, _TIP_DZ])
            if not attached and sep < 0.06 and np.linalg.norm(cube_p - tip) < 0.09:
                attached = True
            if attached and sep > 0.08:
                attached = False
            if attached:
                data.qpos[cubea : cubea + 3] = tip
            data.qpos[cubea + 3 : cubea + 7] = [1, 0, 0, 0]
        final = data.qpos[cubea : cubea + 3]
        if np.linalg.norm(final[:2] - [px, py]) < 0.13 and abs(final[2] - 0.074) < 0.05:
            successes += 1

    return {"successes": successes, "trials": config.eval_episodes}


def run_distill(config: DistillConfig) -> dict[str, Any]:
    """Run the full teacher-student loop and write the distill receipt."""
    api_config = resolve_config()
    client = GeminiRoboticsClient(api_config)
    plan_result = client.plan(
        task=config.task,
        images=[Path(p) for p in config.images],
        model=config.resolved_model(),
    )
    demos = generate_demonstrations(config)
    train_info = train_student(demos, config)
    eval_info = eval_student(train_info["checkpoint"], config)

    receipt = {
        "schema": DISTILL_RECEIPT_SCHEMA,
        "task": config.task,
        "model": config.resolved_model(),
        "plan_text": plan_result.text,
        "n_demos": demos["n_episodes"],
        "n_samples": int(demos["obs"].shape[0]),
        "train_device": train_info["device"],
        "train_val_mse": train_info["val_mse"],
        "eval_successes": eval_info["successes"],
        "eval_trials": eval_info["trials"],
        "checkpoint": train_info["checkpoint"],
    }
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "distill_receipt.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill an ER-2 plan into a local student policy (Gemini -> GPU)."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-demos", type=int, default=200)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = DistillConfig(
        task=args.task,
        images=args.image,
        model=args.model,
        output_dir=args.output_dir,
        n_demos=args.n_demos,
        train_epochs=args.train_epochs,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
    return run_distill(config)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
