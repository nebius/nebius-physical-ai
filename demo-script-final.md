# Nebius Physical AI Workbench Demo Script

## Pre-demo setup

### Option A: One command (default teacher)

```bash
npa workflow distill \
  --teacher-max-iterations 3000 \
  --student-policy act \
  --student-epochs 100 \
  --eval-n-episodes 64 \
  --action-space cartesian
```

Artifacts end up on VMs at `/opt/npa/runs/{run_id}/`. Config saved to `~/.npa/config.yaml`.

### Option B: Individual commands (tuned teacher, local GPU)

Use when you need `--env-override` for custom reward/friction settings.

```bash
npa workbench genesis train-teacher \
  --n-envs 1024 --max-iterations 3000 --action-space cartesian \
  --env-override approach_scale=0 --env-override approach_weight=5.0 \
  --env-override friction_min=0.6 --env-override domain_randomize=true \
  -o ./checkpoints/demo-teacher/

npa workbench genesis generate-demos \
  --checkpoint ./checkpoints/demo-teacher/model.pt \
  --action-space cartesian --n-envs 64 --domain-randomize \
  --fps 20 --seed 42 -o ./data/demos/

npa adapter convert \
  -i ./data/demos/ -o ./data/dataset/ \
  --fps 20 --robot franka_panda

npa workbench lerobot train-student \
  --dataset ./data/dataset/ --policy act \
  --epochs 100 --batch-size 64 --output-dir ./checkpoints/demo-student/

npa workbench genesis eval-teacher \
  --checkpoint ./checkpoints/demo-teacher/model.pt \
  --action-space cartesian --n-envs 64 --seed 7777

npa workbench genesis eval-student \
  --checkpoint ./checkpoints/demo-student/ \
  --action-space cartesian --n-envs 64 --n-episodes 64 \
  --seed 7777 -o ./eval/
```

Also have the side-by-side rollout video ready (teacher vs student).

---

## Live demo (~5 min)

### 1. One command (30s)

"Training a robot policy usually takes weeks of simulator setup, data pipeline wiring, and reward debugging. We did it in one command."

```bash
echo '$ npa workflow distill --teacher-max-iterations 3000 --student-policy act --action-space cartesian'
```

"This provisions an L40S for simulation and an H100 for training, installs everything, runs five stages, and hands off artifacts via S3."

Open the Nebius S3 console in the browser. Navigate to `distill/{run_id}/`.

```
distill/{run_id}/
├── teacher/    ← RL checkpoint from L40S
├── dataset/    ← LeRobot videos + parquet, converted on L40S
├── student/    ← ACT policy trained on H100
└── eval/       ← held-out eval metrics
```

"The L40S trained the teacher, generated camera demos, and converted the data. The H100 trained the student. Everything passed through S3 — no shared filesystem, no manual copies."

Point at file sizes and video files in `dataset/` to show it's real data. Move on quickly.

---

### 2. Diagnose + eval (2 min)

The live portion the audience will remember. Two commands, real output.

**Diagnose the teacher:**

"The teacher was trained with RL in Genesis. It cheats — it sees exact object positions, forces, goal coordinates. But does it work?"

```bash
npa workbench genesis diagnose \
  --checkpoint ./checkpoints/demo-teacher/model.pt \
  --action-space cartesian --n-envs 64
```

Walk through the phase breakdown as it prints: approach, grasp, lift, place. Point at the success rate and the bottleneck suggestion.

"30 seconds to find exactly where the policy fails and what to fix."

**Eval the student:**

"The student only sees cameras — no privileged state. Does it still work?"

```bash
npa workbench genesis eval-student \
  --checkpoint ./checkpoints/demo-student/ \
  --action-space cartesian --n-envs 64 --n-episodes 64 \
  --seed 7777 -o ./eval/
```

```bash
npa workbench genesis eval-teacher \
  --checkpoint ./checkpoints/demo-teacher/model.pt \
  --action-space cartesian --n-envs 64 --seed 7777
```

"Teacher: XX% with privileged state. Student: XX% with cameras only. That gap is the cost of deploying to reality."

**Play the video** (side-by-side teacher vs student rollouts).

"Left cheats. Right doesn't. Right is what goes on the real robot."

---

### 3. How we got here (1 min)

"Our first teacher got 0% success. Instead of manually debugging reward functions:"

```bash
echo '$ npa workbench genesis diagnose --checkpoint v1/model.pt --action-space joint'
echo 'Bottleneck: APPROACH (62/64). Suggestion: switch to --action-space cartesian'
echo ''
echo '$ npa workbench genesis diagnose --checkpoint v3/model.pt --action-space cartesian'
echo 'Bottleneck: LIFT (46/64). Suggestion: increase friction, raise grasp_weight'
echo ''
echo '$ npa workbench genesis tune --checkpoint v3/model.pt --max-rounds 5 --min-success-rate 0.20'
echo 'Round 1: 0.0% → Round 2: 1.6% → Round 3: ...'
```

"Diagnose finds the bottleneck. Tune fixes it automatically. Each iteration is 30 seconds of diagnosis plus a few minutes of retraining."

---

### 4. Agents drive it (1 min)

"Everything I showed you was built and run by AI agents. Claude Code drove the CLI, provisioned VMs, ran the pipeline, and debugged 10 failures. Codex reviewed the code and found 5 issues Claude Code fixed."

```python
cat << 'EOF'
import subprocess, json

def npa(*args):
    r = subprocess.run(["npa"] + list(args) + ["--output-format", "json"],
                       capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None

# One call
npa("workflow", "distill", "--teacher-max-iterations", "3000",
    "--student-policy", "act", "--action-space", "cartesian")

# Or step by step with auto-diagnosis
diag = npa("workbench", "genesis", "diagnose",
           "--checkpoint", "model.pt", "--action-space", "cartesian")
if diag and diag["success_rate"] == 0:
    npa("workbench", "genesis", "tune", "--checkpoint", "model.pt",
        "--max-rounds", "5", "--min-success-rate", "0.20")
EOF
```

"`pip install npa`, `--output-format json`, any LLM that can call subprocess runs the whole thing."

| Layer | How | Status |
|---|---|---|
| CLI | Engineer types `npa workflow distill` | Working today |
| Agent | Claude Code runs the CLI autonomously | Used to build this demo |
| Multi-agent | Claude Code executes, Codex reviews | Codex found 5 issues |

---

### 5. Close (30s)

"From zero infrastructure to a trained robot policy. One command, two GPUs, an afternoon. An AI agent already runs it end to end."

"This is the Nebius Physical AI Workbench."

---

## Reference

### Timing

| Section | Duration | Type |
|---|---|---|
| One command + S3 | 1 min | Echo + S3 console in browser |
| Diagnose + eval | 2 min | Live (~30s each) + video |
| How we got here | 1 min | Echo past output |
| Agents drive it | 1 min | Show code + table |
| Close | 30s | Talk |
| **Total** | **~5.5 min** | |

### What runs live

- S3 console — browse `distill/{run_id}/` artifacts in browser
- `npa workbench genesis diagnose` — ~30s, real phase breakdown
- `npa workbench genesis eval-student` — ~30s, real success rate
- `npa workbench genesis eval-teacher` — ~30s, comparison baseline
- Side-by-side rollout video (pre-recorded)

### 5-stage pipeline

```bash
# Individual commands (local GPU, supports --env-override)
npa workbench genesis train-teacher --n-envs 1024 --max-iterations 3000 --action-space cartesian
npa workbench genesis generate-demos --checkpoint ./checkpoints/teacher/model.pt --n-envs 64 --domain-randomize
npa adapter convert -i ./data/demos/ -o ./data/dataset/ --fps 20 --robot franka_panda
npa workbench lerobot train-student --dataset ./data/dataset/ --policy act --epochs 100
npa workbench genesis eval-student --checkpoint ./checkpoints/student/ --n-envs 64 --seed 7777

# Or one command (L40S sim + H100 train, S3 handoff)
npa workflow distill \
  --teacher-max-iterations 3000 --student-policy act \
  --student-epochs 100 --action-space cartesian --teardown
```

`distill` does not support `--env-override`. Use individual commands for tuned teachers.
