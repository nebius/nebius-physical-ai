---
name: physis-lang
description: Run or review the NPA Physis-Lang-inspired paired physical-prompting experiment with full Wan 2.1 14B generation and blinded video evaluation.
---

# Physis-Lang physical prompting

Use [the guide](../../../docs/workbench/physis-lang.md) and
[`physis-lang.yaml`](../../../workflows/testing/physis-lang.yaml) for this
independent inference experiment. Upstream at the recorded revision contains
paper assets only. Do not describe this implementation as released Physis-Lang,
PhysThinker, PhysCapBench, guideline evolution, retrieval, or fine-tuning.

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
