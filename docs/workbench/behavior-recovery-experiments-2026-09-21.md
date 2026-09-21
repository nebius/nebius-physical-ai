# BEHAVIOR recovery experiments — September 21, 2026

The previous complete six-task development comparison regressed from stock
**Q=0.324028** to fine-tuned **Q=0.298750**. Stock remains the reference policy.
See the [completed results](behavior-matched-results-2026-09-19.md) for per-task
scores, partial diagnostics, and the model-state dtype difference.

This document records experiment choices before their rollout results.
Packages have local validation; GPU training and simulator results are pending.
There is no demonstrated aggregate improvement from these experiments yet.

## Recent research reviewed

[Behavior-Skill](https://arxiv.org/abs/2608.30536), published August 31, evaluates
constituent skills and identifies persistent contact-manipulation bottlenecks.
This supports examining where execution fails, although our trajectory time bins
do not reproduce its semantic skill annotations.
[Evidence-Gated Regularization](https://arxiv.org/abs/2609.03142), published
September 2, targets dependence between sensors and robustness to corruption.
Our inference is that it deserves a separate camera-sensitivity experiment; the
current regression alone does not establish that sensor dependence is its cause.

## Fixed-weight execution comparison

Five arms run the same ten public development instances of picking up trash:

| Model | Execution rule | Question |
| --- | --- | --- |
| Stock checkpoint 2 | Native | What is the contemporaneous baseline? |
| Stock checkpoint 2 | Final-stage backward vote | Does allowing consistent earlier-stage predictions help? |
| Stock checkpoint 2 | Shorter late-stage action chunks | Does more frequent replanning help near completion? |
| Previous selected fine-tune | Native | Does the retained candidate reproduce its development behavior? |
| Previous selected fine-tune | Final-stage backward vote | Does the promising one-case recovery repeat across the panel? |

The backward-vote rule requires three unanimous predictions for an earlier stage
while the controller is in its final stage. It preserves native action queues.
The chunk variant changes the next chunk at prediction boundaries, using shorter
chunks for the last two predicted stages. It never truncates an active queue.
Both use policy predictions and permitted onboard observations.

[AAC](https://arxiv.org/abs/2604.04161) adapts chunk size using action entropy.
[DEHP](https://arxiv.org/abs/2606.11408), revised July 2026, learns a horizon
predictor while keeping the action policy frozen. These motivate testing execution
horizon independently of weights. Our deterministic stage rule is a simpler
hypothesis; it does not reproduce either paper's algorithm.

All five complete panels are required for selection. Rank by mean Q, then full
successes, then fewer steps. Prefer stock native on an exact tie; otherwise use
the predeclared arm order. Promote only a strict mean-Q gain over contemporaneous
stock native. The fine-tuned pair also provides a within-model comparison.

## Training with a stock-policy anchor

The [objective and frozen recipe](behavior-stock-anchored-training.md) document
the training math and its public CPU checks.

Start again from stock checkpoint 2. Train its 23 action-expert parameter leaves
on the original task-balanced training split for radio, trash, and shoes. Keep
the stage classifier frozen and preserve the action-correlation intermediate in
its canonical FP32 dtype. Add a teacher penalty to the demonstration flow loss:

```text
loss = demonstration_loss + mean((student_velocity - frozen_stock_velocity)^2)
```

Teacher and student receive identical observation augmentation, flow time, noise,
and stage conditioning. The penalty weight is 1. Use 2,400 updates, batch size 16,
four flow samples, and a learning rate decreasing from 2e-6 to 2e-7 after a
400-step warmup. This also changes the previous learning rate and schedule, so
any gain cannot be attributed to the anchor alone.

[RLT](https://www.pi.website/research/rlt) motivates preserving a capable base
policy while learning targeted changes. Our supervised flow distillation has
neither RLT's token architecture nor online actor–critic training.
[RECALL](https://arxiv.org/abs/2606.23617) documents the forgetting risk of
recovery-only adaptation. We use existing demonstrations and have not collected
the recovery supervision needed to reproduce that method.

Before training, a GPU check must verify matching initial teacher/student state,
zero initial anchor loss, an unchanged teacher, and updates confined to the 23
allowed parameter leaves and their moving averages. It records actual initialized
teacher dtypes and memory sizes.

## Checkpoint and reporting gates

Evaluate saved updates 400, 800, 1200, 1600, 2000, and 2399 against stock using
fixed held-out traces and random seeds. Each task must satisfy both checks:

- Uniform held-out flow loss is at most 1.02 times stock.
- The non-late action proxy is at most 1.005 times stock.

Among eligible checkpoints, select the greatest equal-task late-proxy improvement,
breaking ties toward the earlier update. If none qualify, retain stock. Time bins
identify trajectory portions; they are not semantic contact labels.

The selected export must reproduce the native checkpoint's typed model state and
fixed-seed actions after reloading. Simulator development evaluation then decides
whether it improves mean Q across all three tasks. Radio and shoes have historical
stock comparisons; trash also has a contemporaneous stock run.

Development indices 10–19 are reused and are not unseen data. Reporting indices
0–9 remain separate until the choice is fixed. Keep interrupted panels and failed
runs visible; do not select lucky reruns. The official evaluator supplies Q under
the [challenge rules](https://behavior.stanford.edu/challenge/evaluation.html).
Neither this small panel nor lower held-out loss establishes a competitive score
across all 100 challenge tasks.
