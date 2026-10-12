---
name: physical-ai-event-video-generation
description: Author, run, or inspect the NPA Cosmos Data Factory EVG workflow that turns seed images into Cosmos3 event videos with detection, tracking, captions, visual QA, and person attribute labels.
---

# Event Video Generation on NPA

Use `workflows/cosmos-data-factory/paidf-event-video-generation.yaml`.
Read [shared setup](../cosmos-data-factory-setup/SKILL.md) and the
[EVG guide](../../../workflows/cosmos-data-factory/event-video-generation.md).
Route clothing-image editing to IAA and video-to-video appearance augmentation
to the separate PAIDF VDA workflow.

Preserve all twelve states: provenance → prepare images → sample configs →
Cosmos3 Super service plus PAIDF augmentation → validate media →
detection/tracking → captions → anomaly Visual QA → person Visual QA → attribute
search → dataset assembly → terminal reopening. Do not invent generation-only
or labeling-only DAG variants. Service and batch share one state; upstream
component retries and failure results remain enforced.

Require an authorized S3 prefix containing selected seed images; a flat directory
makes source mapping clear. NPA enumerates prefixes recursively, does not
implement Airflow's `max_images`, and does not accept its HTTP/single-object
input forms. There is no implicit demo. Preparation verifies dimensions/decoding,
creates RGB JPEG, and retains source/prepared hashes. Stage the requested subset.

Use existing `num_augmentations` and `seed` controls. Event/environment sampling
is defined in the reviewed adapter; arbitrary upstream distribution JSON or
service-mode payloads are not NPA config. Keep the exact Cosmos3 Super
Image2Video revision and private generation/labeling digests. Generation requires
two B200s on one node; captioning/anomaly VQA retain GPU requests for CUVID
decoding even with a hosted VLM.

Retain enabled request guardrails, exact offline snapshots, source-bound
tokenizer/guardrail adaptations, and runtime lineage. Preserve ten-image VQA
controls and `request_media_contract`; the guide records the upstream
16-frame/12-crop difference. Do not disable guardrails or claim complete QA
coverage merely from successful protocol validation.

Validate/plan with intended overrides. For requested execution, follow the
guide's capability/image gates, `submit --runtime`, and monitoring with the same
run ID. Read `anomaly_dataset/dataset.json`, every labeling handoff/sidecar,
upstream provenance, and terminal validation. Decode generated MP4s and inspect
whether the event occurred. Report QA answer coverage/warnings separately from
protocol success. Trackless scenes may omit only supported track-dependent PAS
artifacts. Raw video and separately produced Rerun recordings are distinct evidence.

## Source and changes

Adapted from [NVIDIA's EVG skill](https://github.com/NVIDIA/paidf-orchestration/blob/f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9/skills/physical-ai-event-video-generation/SKILL.md).
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Upstream skill license:
CC-BY-4.0 AND Apache-2.0. Airflow setup, payload, trigger, monitor, and retrieval
instructions were replaced with NPA/SkyPilot operations. The guide discloses
input, placement, guardrail, and VQA differences. This is an Early Access
semantic adaptation; see [NOTICE-NVIDIA-PAIDF](../../NOTICE-NVIDIA-PAIDF).
