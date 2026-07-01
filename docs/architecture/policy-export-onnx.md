# Policy export to ONNX (deployable inference)

## What gap this closes

The sim2real BYO trainer produces an rsl_rl `OnPolicyRunner` checkpoint
(`model_*.pt`) and uploads it to S3. That checkpoint stores an `ActorCritic`
`model_state_dict`; running the trained policy from it requires **torch *and*
Isaac Lab** on the consumer. A robot controller, an edge box, or a lightweight
inference service should not have to install that stack.

`npa.workflows.sim2real.policy_export` re-materializes the actor MLP directly
from the checkpoint's `model_state_dict` (no Isaac Lab, no environment
construction) and exports it to ONNX, plus a `policy_contract.json` sidecar. The
exported `policy.onnx` then runs with nothing but `onnxruntime` (CPU) — the same
runtime convention SONIC eval already uses (`npa.workbench.sonic.eval`).

```
model_975.pt (rsl_rl ActorCritic)            policy.onnx        (single file)
  actor.0/2/4/6 = MLP 36→256→128→64→8   ──►  policy_contract.json
  critic.*, std (dropped for inference)      obs[1,36] → action[1,8]
```

## What you get

- **Portable inference.** `policy.onnx` is a single self-contained file (weights
  embedded, no external-data sidecar) consumable by any ONNX runtime, in any
  language, on CPU. No torch, no Isaac, no GPU required.
- **An explicit obs/action contract** (`policy_contract.json`) describing exactly
  what the consumer must feed and how to interpret the output.
- **Faithful export.** The exported graph reproduces the rsl_rl actor's
  deterministic (mean) action. A real-checkpoint round-trip matches the torch
  actor to within `~2.4e-6` (see `npa/tests/workflows/test_policy_export.py`).

### CLI

```bash
npa workbench isaac-lab export-onnx \
  --input-path s3://<bucket>/sim2real-b/<run>/byo-trainer/.../checkpoints/model_975.pt \
  --task Isaac-Lift-Cube-Franka-v0 \
  --output-path s3://<bucket>/<run>/onnx/      # or a local directory
```

`--input-path`/`--output-path` accept either an `s3://` URI or a local path
(this is an in-pod/local export step — torch lives in the pod, not the public
CLI host). `obs_dim`/`act_dim` are inferred from the actor's first/last layer; an
explicit `--obs-dim`/`--act-dim` is cross-checked and a mismatch is an error.

## The contract the real robot must satisfy

`policy_contract.json` (`format: npa_sim2real_onnx_export_v1`) declares:

| field | meaning |
|-------|---------|
| `obs.dim`, `obs.dtype`, `obs.shape` | input is `float32` `[batch, obs_dim]` on the `obs` tensor |
| `obs.layout` | ordered observation-term names + dims **when recoverable** from the task's observation manager; otherwise `kind: "opaque"` with the guarantee that it is a flat vector in the **same term order the sim used during training** |
| `action.dim`, `action.dtype` | output is `float32` `[batch, act_dim]` on the `action` tensor |
| `action.type`, `scaling`, `limits` | declared action-space semantics (e.g. `joint_position`); known-task hints are filled from a conservative registry, everything else is `opaque` and must be confirmed against the Isaac task `ActionManager` |
| `normalization` | `none` for a raw-obs actor; if the checkpoint carries an rsl_rl `EmpiricalNormalization`, it is **baked into the ONNX graph** so the consumer always feeds raw observations |
| `network` | `mlp`, activation (rsl_rl default `elu`), hidden dims |
| `checkpoint` | provenance: source URI/path, filename, size, sha256, train iter |

The consumer's job: build an observation vector of length `obs.dim` in the
declared order/units/frame, run the session, and apply the `action` vector with
the declared action semantics.

## ⚠️ Export is necessary, NOT sufficient, for sim-to-real

**ONNX export makes the policy portable. It does NOT make it correct on a real
robot.** Nothing in this step bridges the sim-to-real gap:

- **No domain randomization.** The policy is exported exactly as trained. If
  training had no randomization over physics, friction, mass, latency, sensor
  noise, etc., the exported policy is as brittle as the trained one.
- **No real-dynamics validation.** The export never touches a real robot or a
  validated real-dynamics model. A high sim reward says nothing about hardware
  behavior.
- **No observation alignment.** The contract *documents* the obs layout; it does
  not *guarantee* your real sensor pipeline produces the same vector. If the real
  robot's joint ordering, units, frames, or term order differ from the sim
  observation manager — even subtly — the policy will receive out-of-distribution
  input and act unpredictably. Aligning the real observation pipeline to the sim
  contract is the integrator's responsibility and is the most common failure
  mode.
- **No action-space verification.** `action.type`/`scaling`/`limits` are declared
  best-effort. The exact mapping (absolute vs. delta, scale, default-joint
  offset, gripper encoding, safety limits) comes from the Isaac task
  `ActionManager` and **must be re-derived and verified** before sending commands
  to hardware.

Treat the exported policy as **untrusted on hardware until validated there.**
Closing the actual sim-to-real gap (randomization during training, real-dynamics
or hardware-in-the-loop evaluation, and a verified observation/action bridge)
remains separate, unsolved work outside the scope of this export step.

## Implementation notes

- `build_policy_contract(...)` and `infer_mlp_dims(...)` are pure (no torch) and
  unit-tested; the heavy `torch`/`onnx` imports are guarded inside functions.
- The actor is rebuilt as a plain `nn.Sequential` from the `actor.<n>.{weight,
  bias}` entries; activations carry no parameters, so the activation must match
  training (default `elu`, overridable via `--activation`).
- The legacy TorchScript exporter is preferred so weights embed into one file;
  an external-data sidecar (`policy.onnx.data`) is rejected as a contract
  violation.
