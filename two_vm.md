# Genesis Distillation Pipeline — Session State

> **Legacy / unmaintained.** This root-level file captures a one-off two-VM
> session state and is not a current setup guide. For platform setup see
> [docs/quickstart.md](docs/quickstart.md) and
> [docs/workbench/getting-started.md](docs/workbench/getting-started.md). Kept
> for historical context only.

## VMs

| Alias | GPU | IP | SSH | Status |
|-------|-----|----|-----|--------|
| `l40s-distill-genesis` | L40S 46GB | `<GENESIS_VM_IP>` | `ssh -i ~/.ssh/id_ed25519 ubuntu@<GENESIS_VM_IP>` | **Up** — genesis-world 0.4.6, rsl-rl 2.2.4, npa 0.1.0 installed |
| `h100-distill-lerobot` | H100 80GB | `<LEROBOT_VM_IP>` | `ssh -i ~/.ssh/id_ed25519 ubuntu@<LEROBOT_VM_IP>` | **Up** — lerobot 0.5.1, torch 2.10. No genesis/npa. |

Both in `eu-north1` project. Config: `~/.npa/config.yaml` under `eu-north1` → `l40s-distill-genesis` / `h100-distill-lerobot`.

## What's on the L40S VM

```
/home/ubuntu/npa/                          # editable npa install (pip install -e .)
/home/ubuntu/logs/teacher/model_499.pt     # original teacher (500 iter, default rewards, no DR)
/home/ubuntu/logs/teacher/arch_config.json # created manually, default [256,256,128]
/home/ubuntu/checkpoints/tuned_v4/model.pt # best attempt: linear reward, DR, 3000 iter
/home/ubuntu/checkpoints/tuned_v3/         # linear reward, no DR, 1000 iter
/home/ubuntu/checkpoints/tune/             # 5-round auto-tune artifacts (round_01..round_05)
/opt/lerobot/venv/                         # shared venv (genesis, rsl-rl, lerobot, npa)
```

Activate with: `source /opt/lerobot/venv/bin/activate`

## Diagnosis Results

All teacher checkpoints produce **0% task success**. The bottleneck is consistently **approach** — the gripper never gets within 8cm of the cube.

Best checkpoint (`tuned_v4`) approach distance distribution across 64 episodes:
- Median: 0.29m, Min: 0.17m, Max: 0.42m
- 0% episodes reach < 15cm, 55% reach < 30cm
- The gripper moves toward the cube but gets stuck ~30cm away

## What Was Changed in Code

### `env_pick_place.py` (EnvConfig additions)
- `approach_scale: float = 5.0` — exponential distance scaling. 0 = linear `(1-d)*weight`.
- `place_scale: float = 5.0` — same for place reward.
- `action_scale: float = 0.0` — clamp delta joint actions to ±N radians. 0 = no clamping.
- `ee_pos` added to `get_privileged_obs()` return dict (NOT in `flat`, no checkpoint compat change).
- `gs.init()` guarded with `if not gs._initialized` to allow multiple envs per process.

### `diagnose.py` (new file)
- `diagnose_teacher()` — runs rollouts, tracks per-timestep `EpisodeTrace`, classifies failure phase.
- Phases: approach (>8cm), grasp (<2 contacts), lift (<3cm rise), place (>8cm from target), timeout.
- `SUGGESTIONS` dict maps each phase to config changes.

### `tune.py` (new file)
- `tune_teacher()` — loop: diagnose → apply suggestion → `_retrain_with_overrides()` → re-diagnose.
- Artifacts split into `round_NN/diagnosis/` and `round_NN/retrained/`.
- Early-exit rounds write `env_overrides.json`.

### `cli/genesis/__init__.py`
- Added `diagnose` and `tune` commands under `npa workbench genesis`.

## The Blocker: Approach Convergence

The Franka starts at ee_pos ≈ (0.31, 0, 0.59). The cube is at (0.5, 0, 0.04). Starting distance: 0.58m.

**Why the policy gets stuck at ~30cm:**
The arm lowers toward the table but can't reorient its wrist to reach forward to x=0.5. The MLP policy finds a kinematic local minimum. The exponential reward `exp(-5*d)` gives near-zero gradient at 0.58m; switching to linear `(1-d)*weight` helped (from stuck at 0.58m → median 0.29m) but didn't solve it.

**What to try next (in priority order):**
1. **Curriculum**: start cube at (0.35, 0, 0.30) near the gripper, gradually move to (0.5, 0, 0.04).
2. **IK action space**: output Cartesian delta (dx, dy, dz, gripper) instead of joint deltas. Genesis supports `robot.control_dofs_position()` with IK targets.
3. **Longer training**: 10k+ iterations (3000 iter = ~4 min on L40S with 4096 envs).
4. **Warm-start tune**: modify `_retrain_with_overrides` to load previous checkpoint weights instead of training from scratch.

## Commands Cheat Sheet

```bash
# Diagnose a checkpoint
npa workbench genesis diagnose --checkpoint /path/to/model.pt --n-envs 1024

# Auto-tune loop
npa workbench genesis tune --checkpoint /path/to/model.pt --max-rounds 5

# Train teacher (via Python, full control)
python3 -c "
from pathlib import Path
from npa.genesis.tune import _retrain_with_overrides
_retrain_with_overrides(
    n_envs=4096, max_iterations=3000,
    output_dir=Path('/home/ubuntu/checkpoints/next'),
    log_dir=Path('/home/ubuntu/logs/next'),
    device='cuda', seed=42,
    env_overrides={'approach_weight': 5.0, 'approach_scale': 0, 'domain_randomize': True},
)
"

# Eval teacher
npa workbench genesis eval-teacher --checkpoint /path/to/model.pt --n-envs 1024 --seed 7777

# Generate demos (requires working teacher)
npa workbench genesis generate-demos --checkpoint /path/to/model.pt --n-envs 4096

# Eval student (requires trained student)
npa workbench genesis eval-student --checkpoint /path/to/student/ --n-envs 1024
```
