---
name: physical-prompt-comparison
description: Run or review the physical prompt comparison workflow with full Wan 2.1 14B generation, matched prompt arms, and blinded video evaluation.
---

# Physical prompt comparison

Use [the guide](../../../docs/workbench/physical-prompt-comparison.md) and
[`physical-prompt-comparison.yaml`](../../../workflows/testing/physical-prompt-comparison.yaml) for this
inference experiment. It compares baseline prompts, physical descriptions, and
negative guidance at matched seeds. Treat its sampled-frame scores as exploratory
measurements, not a physics benchmark, training method, or robot-policy result.

Run through `npa workbench workflow submit` on the supported Linux isolated
SkyPilot path. Select the operator's project/context and storage explicitly.
Follow `health-preflight` and `submit-workflow`; verify the exact pinned Wan
weight payload and hosted text/vision model access before GPU execution.

Keep the six scenarios, sealed assertions, paired seeds, and three conditioning
arms intact for the shipped comparison. Full generation means 81 frames,
1280×720, 50 steps, and the pinned native Wan 2.1 14B model. Reduced settings
would be a different experiment and cannot qualify this workflow. Generation
uses one resident model per GPU shard, with original media and native receipts.

Inspect the final report and complete videos. Require exact paired coverage,
verified storage readback, and model request/response identity. A valid video
is not proof of correct physics. Unknown sampled-frame assertions do not pass;
retain negative results and distinguish VLM judgments from human assessment.
Preserve failed-stage evidence and standard durable run state when debugging.
