---
name: gpu-allocation-fallback
description: Track typed GPU placement failures in the NPA agent and offer a consent-gated on-demand-to-preemptible fallback after repeated failures or deterministic preflight. Use when quota, capacity, Unschedulable GPU, or compatible-product placement blocks an allocation.
---

# GPU Allocation Fallback

Send typed results to `POST /api/agent/gpu-allocation/attempt` with a stable
`logical_allocation`, the requested compatibility/execution/disk invariants,
typed failure evidence, and the identical compatible preemptible candidate.
The grounded route uses zero model tokens.

Only quota/capacity exhaustion, insufficient GPU or Unschedulable, and no
compatible product/affinity count. Auth, RBAC, network, image-pull, checkpoint,
application, runtime, cancellation, and timeout failures never count. The
default prompt occurs on the third qualifying failure, or immediately when
deterministic preflight proves on-demand cannot succeed and compatible
preemptible capacity is available.

Never switch automatically. Present the returned question and proposed action.
Accept through `POST /api/agent/gpu-allocation/consent` with its single-use,
action-digest-bound `confirm_token`; decline with `accept: false`. A decline
keeps on-demand and suppresses the same evidence until materially new evidence
arrives.

Preserve GPU family/product/count, image/digest, SM and RT-core requirements,
backend, model, workload tier, execution mode, and boot-disk count/bytes. The
state record contains only redacted digests, classification, selected pool, and
consent outcome. Success or a changed logical allocation resets attempt state.

Verify changes with:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/cli/test_agent_gpu_allocation_fallback.py \
  npa/tests/cli/test_agent_backend_render.py -q
```
