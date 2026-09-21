# Stock-anchored BEHAVIOR training

This page records a prospective supervised fine-tuning ablation for the public
BEHAVIOR policy work. GPU training, checkpoint scoring, and simulator evaluation
are pending. The implementation does not establish an improvement.

## Objective

The student starts from released stock checkpoint 2. For each training example,
the frozen parent and trainable student receive the same observation
augmentation, stage-conditioned prompt, flow time, and flow noise. The parent
velocity is stopped before the loss is formed:

```text
x_t = t z + (1 - t) a
u_t = z - a
L_demo = mean((v_student(obs, x_t, t, stage) - u_t)²)
L_anchor = mean((v_student(obs, x_t, t, stage)
                 - stop_gradient(v_stock(obs, x_t, t, stage)))²)
L_total = L_demo + L_anchor
```

[`anchored_objective.py`](../../workflows/implementations/behavior-anchored-training/anchored_objective.py)
contains a publication-safe, backend-neutral form of the objective prepared for
this ablation. It accepts a
NumPy-compatible array module and an explicit `stop_gradient` operation, so the
same equation can be tested with NumPy and used by the JAX training runtime. It
removes the private flow-velocity tensor from the returned logging metrics.

The exact 23-leaf `nnx.Param` partition remains in
[`action_partition.py`](../../workflows/implementations/behavior-matched-training/action_partition.py).
That filter excludes non-parameter state, including the FP32 action-correlation
intermediate, and freezes task and stage parameters. The frozen-parent buffers
do not enter the optimizer or exponential moving average.

### Explicit canonical correlation state

The public adapter can consume a raw little-endian FP32 correlation matrix when
a run has already established canonical bytes for its pinned model and
normalization inputs. The file must contain exactly a `960 × 960` C-order
matrix. Callers provide both its path and SHA-256; the repository does not ship
or infer canonical matrix bytes.

For training, install the artifact immediately after native state
initialization and before copying the frozen teacher:

```python
from npa.workflows.behavior_challenge.rlc_correlation import (
    install_training_correlation,
)

installations = install_training_correlation(
    train_state,
    correlation_artifact,
    correlation_sha256,
)
# Capture the frozen teacher from train_state.params only after this call.
```

The helper requires the target leaf to carry the real `nnx.Intermediate` type,
installs identical FP32 bytes in `params` and `ema_params` when EMA exists, and
checks that all other state objects and values remain unchanged. It returns a
device-readback identity for each state. The default native initializer remains
unchanged when the helper is not called.

For stock serving, pass the same explicit pair to the internal evaluation
stage:

```text
--policy-kind rlc
--policy-stock-correlation-asset /path/to/action-correlation.float32.bin
--policy-stock-correlation-sha256 <64-lowercase-hex-digest>
```

Both flags are required together. The server installs the matrix during the
single `PiBehavior.load_correlation_matrix` call, verifies the actual
`nnx.Intermediate`, and checks the captured `Policy` model before inference.
Without the flags, stock serving follows the existing native numerical loading
path byte for byte. A matching digest proves artifact identity; it does not by
itself prove that an artifact is canonical for another checkpoint or runtime.

## Frozen recipe

[`recipe.json`](../../workflows/implementations/behavior-anchored-training/recipe.json)
records the machine-readable proposal:

- released demonstrations for tasks 0, 1, and 22 only;
- uniform task-balanced sampling, with late-stage oversampling disabled;
- 2,400 updates, batch size 16, four flow samples, and seed 0;
- 400-step warmup, peak learning rate `2e-6`, and decay learning rate `2e-7`;
- anchor weight 1.0;
- checkpoints 400, 800, 1,200, 1,600, 2,000, and 2,399.

This changes the schedule as well as adding the anchor, so a future result would
measure the complete recipe rather than the anchor in isolation. Evaluation
cases, rollout outcomes, videos, and report metrics are excluded from training.

## Admission and evaluation

Before a full run, the real GPU preflight must show that:

1. parent and student action states are byte-identical at initialization;
2. the initial anchor loss is exactly zero for shared stochastic inputs;
3. the parent remains unchanged after a real update;
4. changes remain inside the 23 action leaves and their corresponding moving
   averages;
5. the action-correlation intermediate remains its canonical FP32 state;
6. losses, gradients, parameters, and fixed-batch actions are finite.

Each candidate checkpoint must keep every task's uniform held-out flow loss at
or below `1.02 ×` stock and its non-late proxy at or below `1.005 ×` stock.
Eligible checkpoints rank by equal-task late-proxy improvement, with the earlier
step winning an exact tie. If no checkpoint passes, the policy remains stock.

Development evaluation reuses cases 311–320 for tasks 0, 1, and 22. The
candidate is selected only for a strict gain in complete-three-task equal-task
mean Q; a tie retains stock. Cases 301–310 remain reserved until that policy
choice is frozen. These small panels do not establish performance over all 100
challenge tasks.

## Validation status

The public tests cover zero loss at initialization, positive loss after a known
velocity drift, shape mismatch rejection, a real JAX gradient check showing no
parent gradient, and the frozen recipe. Full model compilation, memory use,
training, export parity, and simulator scores remain pending GPU evidence.
