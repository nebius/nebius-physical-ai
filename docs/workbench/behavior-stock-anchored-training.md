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
