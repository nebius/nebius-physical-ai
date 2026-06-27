"""Mock customer trainer container — a SEAM/CONTRACT proof, not a real LeRobot fork.

The sim2real loop lets a customer plug in THEIR OWN trainer container through the
``byo_trainer_command`` seam. The concrete real trainer today is
``byo_isaac_trainer`` (Isaac PPO). This module is a DISTINCT, deliberately small
trainer that is NOT Isaac: it stands in for a customer's LeRobot+VLM fork to prove
that an arbitrary container satisfying the contract plugs into the seam and that
the VLM-signal -> policy-update data flow works end to end.

Contract (identical to byo_isaac_trainer's, the generic byo_trainer_command seam):
  * read the parsed VLM RL-signal batch from ``NPA_SIM2REAL_SIGNAL_JSON``
    (``npa.sim2real.rl_signal.v1``: per_step reward/advantage + corrective target);
  * perform a REAL weight update on a policy (a compact ACT-style MLP action head);
  * write a real checkpoint;
  * write a ``VlmSignalUpdateResult`` JSON to ``NPA_SIM2REAL_OUTPUT_JSON``.

What this proves: the seam, the contract, and the VLM->reward->weight-update flow
a customer's own container relies on. What it does NOT do: faithfully reproduce a
real LeRobot/ACT training stack (out of scope) — the policy here is a tiny MLP and
the "observation" is a fixed canonical context, so the update is exercised against
the VLM signal, not a real robot rollout. The customer's container swaps this body
for their real trainer while keeping the same env-var contract.

The update is genuinely VLM-driven: it is advantage-weighted gradient descent of
the action head toward the VLM's corrective targets, so a more informative critique
(larger advantage spread) produces a larger, directed policy delta, and a degenerate
critique (uniform reward -> zero advantage) produces ~no update.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_RL_SIGNAL = "npa.sim2real.rl_signal.v1"
BACKEND = "mock_lerobot_vlm_act_head"
# Fixed canonical "observation" (proprio placeholder) the standalone head maps from.
OBS_DIM = 8
HIDDEN_DIM = 16
DEFAULT_ACTION_DIM = 3  # matches the VLM corrective-target action_delta dimension
DEFAULT_GRAD_STEPS = 5
WEIGHT_SEED = 20260627


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# --------------------------------------------------------------------------- #
# Signal parsing
# --------------------------------------------------------------------------- #
def summarize_signal_batch(payload: Any) -> dict[str, Any]:
    """Reduce a VLM RL-signal batch to the quantities that drive the update.

    Returns the advantage-weighted corrective target (the direction the VLM
    critique says to move the policy), the signal strength (mean |advantage| —
    how strongly the critique disagrees with the mean), and mean reward/advantage
    and counts for the result record.
    """

    signals = payload.get("signals") if isinstance(payload, dict) else payload
    rewards: list[float] = []
    advantages: list[float] = []
    action_dim = DEFAULT_ACTION_DIM
    weighted = np.zeros(DEFAULT_ACTION_DIM, dtype=np.float64)
    weight_sum = 0.0
    step_count = 0
    for signal in signals or []:
        for step in (signal or {}).get("per_step", []) or []:
            step_count += 1
            reward = float(step.get("reward", 0.0))
            advantage = step.get("advantage")
            advantage = float(advantage) if advantage is not None else reward
            rewards.append(reward)
            advantages.append(advantage)
            target = (step.get("target") or {}).get("action_delta") or []
            if target:
                action_dim = max(action_dim, len(target))
                vec = np.zeros(action_dim, dtype=np.float64)
                if weighted.shape[0] < action_dim:
                    grown = np.zeros(action_dim, dtype=np.float64)
                    grown[: weighted.shape[0]] = weighted
                    weighted = grown
                for i, value in enumerate(target):
                    vec[i] = float(value)
                # Advantage-weighted: steps the VLM rates above the rollout mean pull
                # the policy toward their corrective direction; below-mean steps push
                # away. Degenerate (uniform-reward) batches have advantage ~0 -> no pull.
                weighted[: vec.shape[0]] += advantage * vec
                weight_sum += abs(advantage)
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mean_advantage = sum(advantages) / len(advantages) if advantages else 0.0
    signal_strength = (
        sum(abs(a) for a in advantages) / len(advantages) if advantages else 0.0
    )
    if weight_sum > 1e-9:
        weighted = weighted / weight_sum
    else:
        weighted = np.zeros(action_dim, dtype=np.float64)
    return {
        "weighted_target": weighted.astype(np.float64),
        "signal_strength": float(signal_strength),
        "mean_reward": float(mean_reward),
        "mean_advantage": float(mean_advantage),
        "step_count": int(step_count),
        "signal_count": int(len(signals or [])),
        "action_dim": int(action_dim),
    }


# --------------------------------------------------------------------------- #
# Compact ACT-style MLP action head (numpy, CPU)
# --------------------------------------------------------------------------- #
class MlpPolicy:
    """obs(8) -> tanh hidden(16) -> action(A). A real (if tiny) parametric policy."""

    def __init__(self, action_dim: int, *, seed: int = WEIGHT_SEED,
                 weights: dict[str, np.ndarray] | None = None) -> None:
        self.action_dim = int(action_dim)
        if weights is not None:
            self.W1 = weights["W1"]
            self.b1 = weights["b1"]
            self.W2 = weights["W2"]
            self.b2 = weights["b2"]
        else:
            rng = np.random.default_rng(seed)
            self.W1 = rng.normal(0.0, 0.3, size=(OBS_DIM, HIDDEN_DIM))
            self.b1 = np.zeros(HIDDEN_DIM)
            self.W2 = rng.normal(0.0, 0.3, size=(HIDDEN_DIM, self.action_dim))
            self.b2 = np.zeros(self.action_dim)

    @staticmethod
    def canonical_obs() -> np.ndarray:
        # Deterministic proprio placeholder so the standalone head is a fixed map.
        return np.linspace(-1.0, 1.0, OBS_DIM)

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(obs @ self.W1 + self.b1)
        a = h @ self.W2 + self.b2
        return a, h

    def params_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()]
        )

    def to_weights(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}


def update_policy(
    policy: MlpPolicy,
    weighted_target: np.ndarray,
    signal_strength: float,
    *,
    learning_rate: float,
    grad_steps: int = DEFAULT_GRAD_STEPS,
) -> dict[str, Any]:
    """Advantage-weighted gradient descent of the action head toward the VLM target.

    A REAL weight update: backprop MSE(policy(obs), VLM_corrective_target) and step
    the parameters, with the effective step scaled by the VLM signal strength so a
    stronger critique moves the policy more. Returns before/after policy outputs,
    losses, and the L2 norm of the parameter change.
    """

    obs = policy.canonical_obs()
    before = policy.params_vector().copy()
    out_before, _ = policy.forward(obs)
    target = np.asarray(weighted_target, dtype=np.float64)[: policy.action_dim]
    if target.shape[0] < policy.action_dim:
        target = np.pad(target, (0, policy.action_dim - target.shape[0]))

    def mse(out: np.ndarray) -> float:
        return float(0.5 * np.mean((out - target) ** 2))

    loss_before = mse(out_before)
    eff_lr = float(learning_rate) * float(signal_strength)
    for _ in range(max(1, grad_steps)):
        a, h = policy.forward(obs)
        # dL/da = (a - target)/A
        d_a = (a - target) / float(policy.action_dim)
        # output layer grads
        g_W2 = np.outer(h, d_a)
        g_b2 = d_a
        # backprop into hidden (tanh')
        d_h = (policy.W2 @ d_a) * (1.0 - h ** 2)
        g_W1 = np.outer(obs, d_h)
        g_b1 = d_h
        policy.W2 -= eff_lr * g_W2
        policy.b2 -= eff_lr * g_b2
        policy.W1 -= eff_lr * g_W1
        policy.b1 -= eff_lr * g_b1
    out_after, _ = policy.forward(obs)
    after = policy.params_vector()
    policy_delta_l2 = float(np.linalg.norm(after - before))
    return {
        "policy_output_before": [round(float(x), 6) for x in out_before],
        "policy_output_after": [round(float(x), 6) for x in out_after],
        "loss_before": round(loss_before, 6),
        "loss_after": round(mse(out_after), 6),
        "policy_delta_l2": round(policy_delta_l2, 6),
        "effective_lr": round(eff_lr, 6),
    }


# --------------------------------------------------------------------------- #
# Checkpoint I/O (real file; local or S3)
# --------------------------------------------------------------------------- #
def save_checkpoint(policy: MlpPolicy, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, action_dim=np.array(policy.action_dim), **policy.to_weights())
    # np.savez appends .npz if missing; normalize to the actual written file.
    return path if path.exists() else path.with_suffix(".npz")


def load_checkpoint(path: Path) -> MlpPolicy:
    data = np.load(path)
    return MlpPolicy(
        int(data["action_dim"]),
        weights={k: data[k] for k in ("W1", "b1", "W2", "b2")},
    )


def _download_s3(uri: str, dst: Path) -> bool:
    try:
        import boto3
        from urllib.parse import urlparse

        u = urlparse(uri)
        s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(u.netloc, u.path.lstrip("/"), str(dst))
        return True
    except Exception as exc:  # best-effort; fresh init on failure
        print(f"mock_trainer: resume download failed ({exc}); starting fresh", flush=True)
        return False


def _upload_s3(local: Path, uri: str) -> bool:
    try:
        import boto3
        from urllib.parse import urlparse

        u = urlparse(uri)
        s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
        s3.upload_file(str(local), u.netloc, u.path.lstrip("/"))
        return True
    except Exception as exc:
        print(f"mock_trainer: checkpoint upload failed ({exc})", flush=True)
        return False


# --------------------------------------------------------------------------- #
# Result record (VlmSignalUpdateResult-shaped)
# --------------------------------------------------------------------------- #
def build_result(
    *,
    summary: dict[str, Any],
    update: dict[str, Any],
    initial_reward_head: float,
    checkpoint_path: str,
    duration_ms: float,
) -> dict[str, Any]:
    mean_reward = float(summary["mean_reward"])
    reward_target = max(0.0, min(1.0, (mean_reward + 1.0) / 2.0))
    reward_head_after = round(
        initial_reward_head + 0.5 * (reward_target - initial_reward_head), 6
    )
    return {
        "schema": "npa.lerobot.vlm_signal_adapter.v1",
        "status": "success",
        "backend": BACKEND,
        "steps": int(summary["step_count"]),
        "loss_before": update["loss_before"],
        "loss_after": update["loss_after"],
        "reward_head_before": round(float(initial_reward_head), 6),
        "reward_head_after": reward_head_after,
        "policy_output_before": update["policy_output_before"],
        "policy_output_after": update["policy_output_after"],
        "policy_delta_l2": update["policy_delta_l2"],
        "mean_reward": round(mean_reward, 6),
        "mean_advantage": round(float(summary["mean_advantage"]), 6),
        "checkpoint_path": checkpoint_path,
        "signal_count": int(summary["signal_count"]),
        "control": False,
        "loss_integration_point": (
            "Mock customer LeRobot+VLM container (seam proof): VLM critique -> "
            "advantage-weighted corrective target -> gradient step on a compact "
            "ACT-style MLP action head. Proves the byo_trainer_command contract + "
            f"VLM->update data flow (signal_strength={round(float(summary['signal_strength']), 6)})."
        ),
        "duration_ms": round(float(duration_ms), 3),
    }


def run_training(signal_json: str, *, run_id: str) -> dict[str, Any]:
    start = time.time()
    try:
        payload = json.loads(Path(signal_json).read_text(encoding="utf-8"))
    except Exception:
        payload = {"signals": []}
    summary = summarize_signal_batch(payload)

    # Optional resume (honors the same generic seam var as byo_isaac_trainer): a
    # customer container that keeps state across outer iterations continues its
    # policy instead of reinitializing — proves the resume seam is trainer-agnostic.
    resume_uri = _env("NPA_SIM2REAL_RESUME_CHECKPOINT_URI")
    policy: MlpPolicy | None = None
    if resume_uri:
        local = Path(resume_uri)
        if resume_uri.startswith("s3://"):
            tmp = Path("/tmp/mock_trainer_resume.npz")
            local = tmp if _download_s3(resume_uri, tmp) else None
        if local is not None and local.exists():
            try:
                policy = load_checkpoint(local)
                print(f"mock_trainer: resumed policy from {resume_uri}", flush=True)
            except Exception as exc:
                print(f"mock_trainer: resume load failed ({exc}); fresh init", flush=True)
    if policy is None:
        policy = MlpPolicy(summary["action_dim"])

    learning_rate = float(_env("NPA_SIM2REAL_LEARNING_RATE", "0.5") or 0.5)
    update = update_policy(
        policy, summary["weighted_target"], summary["signal_strength"],
        learning_rate=learning_rate,
    )

    out_dir = Path(_env("NPA_SIM2REAL_OUTPUT_JSON", "/tmp/mock-update.json")).parent
    ckpt_local = save_checkpoint(policy, out_dir / "mock_policy.npz")
    checkpoint_path = str(ckpt_local)
    bucket = _env("NPA_SIM2REAL_S3_BUCKET") or _env("NPA_SIM2REAL_BUCKET")
    tag = _env("NPA_SIM2REAL_TRAINER_TAG") or "mock"
    if bucket:
        uri = f"s3://{bucket}/sim2real-b/{run_id}/byo-trainer/mock/{tag}/mock_policy.npz"
        if _upload_s3(ckpt_local, uri):
            checkpoint_path = uri
            print(f"mock_trainer: uploaded checkpoint -> {uri}", flush=True)

    result = build_result(
        summary=summary,
        update=update,
        initial_reward_head=float(_env("NPA_SIM2REAL_INITIAL_REWARD_HEAD", "0.0") or 0.0),
        checkpoint_path=checkpoint_path,
        duration_ms=(time.time() - start) * 1000.0,
    )
    print(
        f"mock_trainer: signal_strength={summary['signal_strength']:.4f} "
        f"policy_delta_l2={update['policy_delta_l2']:.4f} "
        f"loss {update['loss_before']:.4f}->{update['loss_after']:.4f}",
        flush=True,
    )
    return result


def main() -> int:
    output_json = _env("NPA_SIM2REAL_OUTPUT_JSON")
    if not output_json:
        print("mock_lerobot_vlm_trainer: NPA_SIM2REAL_OUTPUT_JSON not set", file=sys.stderr)
        return 2
    signal_json = _env("NPA_SIM2REAL_SIGNAL_JSON")
    run_id = _env("NPA_SIM2REAL_RUN_ID") or _env("RUN_ID") or "mock-run"
    result = run_training(signal_json, run_id=run_id)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"mock_lerobot_vlm_trainer: wrote update -> {output_json} "
        f"(checkpoint={result['checkpoint_path']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
