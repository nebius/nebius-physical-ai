# Complete LIBERO-10 quality curve after eight-B200 training

The first scheduled checkpoint to meet the predeclared **90% overall success**
target was update **1,500**, scoring **464/500 (92.8%)**. Its model files were
saved after about **5 h 52 m 57 s of training**. The final update-2,000
checkpoint scored **475/500 (95.0%)**. Every checkpoint was evaluated on all
ten tasks, with fifty initial states per task and no infrastructure errors.

![Actual quality curve and every task's result](policy-quality.png)

| Update | Successes / trials | Success | Pooled 95% Wilson interval | Training time when saved, approximately | Evaluation process seconds |
| --- | --- | --- | --- | --- | --- |
| 500 | 227 / 500 | 45.4% | 41.09–49.78% | 1 h 58 m 23 s | 2,757.054 |
| 1,000 | 428 / 500 | 85.6% | 82.25–88.41% | 3 h 55 m 47 s | 2,353.390 |
| 1,500 | 464 / 500 | 92.8% | 90.19–94.75% | 5 h 52 m 57 s | 2,194.578 |
| 2,000 | 475 / 500 | 95.0% | 92.72–96.59% | 7 h 50 m 18 s | 2,240.083 |

The four evaluations contain **2,000 trials across four distinct policies**;
their successes must not be pooled into one policy score. The target applies
to the observed aggregate success rate, not the lower Wilson bound or every
individual task. The intervals summarize the pooled trial counts; they do not
measure variation across tasks or independently trained seeds. These are the
same ten task types used for training, not an unseen-task benchmark. No
separately evaluated control policy is used to claim an improvement over a
baseline.

## What time to quality means

[quality-curve.json](quality-curve.json) joins each complete evaluation to the
actual native checkpoint-completion event and verified model identity. Those
UTC log events have one-second resolution. The exact training-time interval
for the first passing checkpoint is **21,177.321–21,178.321 seconds**. Update
1,000 did not pass; the measurements do not establish an exact crossing
between those saved checkpoints.

This is a retrospective measurement. The original run continued through all
2,000 updates, finishing in 28,233.477 seconds; evaluation ran afterward. The
final checkpoint was saved about fifteen seconds before process exit. Neither
the checkpoint times nor the full training duration includes these later
evaluations or campaign queue delay.

Evaluation used eight B200 policy servers and eight simulator environments per
server. The four complete evaluation processes took **9,545.105 seconds
(2 h 39 m 5 s)** in total, or **21.211 evaluation-process GPU-hours**. Each
process duration includes checkpoint hashing, model loading, server setup and
parallel simulation. It is not steady inference latency or the complete
campaign's GPU usage. Slurm recorded successful `COMPLETED`, `0:0` exits and
allocation durations of 2,758, 2,354, 2,196 and 2,241 seconds, respectively.

## Evaluation contract and original evidence

All four runs used the same pinned framework and LIBERO revisions as training,
seed 42, fifty default initial states per task, eight policy servers, eight
simulator environments per server, and no video recording. The native servers
loaded EMA weights with thirty UniPC denoising steps and guidance 1.0. Their
actions use sixteen-step chunks, 20 Hz, front and wrist cameras, 10D rot6d,
`quantile_rot` normalization and the native `zero_one` gripper mapping. CPU
MuJoCo with OSMesa renders the simulator observations.

The pinned [native evaluator](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/simulation/libero/closed_loop_eval.py)
selects initial-state indices 0–49 for each task. Its LIBERO-10 limit is 520
simulator steps, including ten stabilization steps. In vectorized evaluation,
`episode_results[].elapsed_s` allocates a wave's elapsed time equally across
its trials. Those values are not isolated per-trial latencies. Use the
whole-process measurements above for evaluation duration.

- `quality-step-*.json` are the unmodified native wrapper outputs, including
  all per-task aggregates and every trial's outcome, simulator-step count and
  error field. All 2,000 trials have positive step counts and null errors.
- `model-hashes-step-*.json` are the unmodified nine-file model manifests. Every
  component matches the independently verified archive of the corresponding
  training checkpoint. The final manifest also exactly matches the
  [recorded final-policy videos](../cosmos3-wam-final-visual/README.md).
- `quality-curve.json` is the executed recipe reducer's output. The reducer
  rehashes all four model payloads, verifies trial counts and matching
  evaluation settings, and derives checkpoint times from the native training
  log. It does not infer success from training loss or a ten-trial visual pass.
- `evidence.json` links source files, actual Slurm accounting and private worker
  records by hash. Raw logs, configurations and infrastructure details remain
  in private operator evidence.
- `plot.py` verifies the original quality-receipt hashes, model-manifest hashes
  and per-trial counts before rendering. `render-manifest.json` records the
  numerical-library versions and generated image hash.

The [training report](../cosmos3-wam-full-8/README.md) remains unchanged. Its
`quality_measured: false` field describes that earlier training-only report;
these separate evaluation records establish policy quality. Source attribution
for the model and simulator is retained in the
[visual evidence notice](../cosmos3-wam-final-visual/NOTICE.md).

## Reproduce the measurements and figure

Follow the [native Slurm recipe](../../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md)
through training, simulator preparation, runtime checks and guardrail prefetch.
Keep its documented HF token-file environment in the Slurm job environment.
Set `WAM_FULL_RUN` to the completed training run and `WAM_EVAL_ROOT` to a fresh
private evaluation parent directory. On the reserved B200 Slurm cluster:

```bash
set -euo pipefail
for WAM_STEP in 500 1000 1500 2000; do
  srun --nodes=1 --ntasks=1 --gpus-per-task=8 --cpus-per-task=128 --exclusive \
    "$WAM_SHARED_ROOT/framework/.venv/bin/python" "$WAM_RECIPE/evaluate.py" \
    --shared-root "$WAM_SHARED_ROOT" --run-dir "$WAM_FULL_RUN" \
    --step "$WAM_STEP" --output-dir "$WAM_EVAL_ROOT/step-$WAM_STEP" \
    --workers 8 --envs 8 --trials 50 --seed 42
done
"$WAM_SHARED_ROOT/framework/.venv/bin/python" "$WAM_RECIPE/quality_curve.py" \
  --run-dir "$WAM_FULL_RUN" \
  --evaluations "$WAM_EVAL_ROOT/step-500" "$WAM_EVAL_ROOT/step-1000" \
                "$WAM_EVAL_ROOT/step-1500" "$WAM_EVAL_ROOT/step-2000" \
  --output-path "$WAM_EVAL_ROOT/quality-curve.json"
```

Keep all four model checkpoints and the original training log until reduction
finishes. Require successful Slurm completion and all eight worker completion
receipts for each evaluation. A repeated run need not reproduce identical
weights, trajectories or outcomes; preserve every new result.

To regenerate this figure from the committed evidence, with the Workbench
Python environment, NumPy and Matplotlib:

```bash
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-quality-8/plot.py
```

For a new measurement, use a separate directory containing its generated
`quality-curve.json`, `quality-step-STEP.json` and
`model-hashes-step-STEP.json` files for all four steps, plus a copy of `plot.py`.
Do not replace the committed observations with a new run's results.
