---
name: physical-ai-image-attribute-augmentation
description: Author, run, or inspect the NPA Cosmos Data Factory IAA workflow for person-image clothing, color, footwear, and accessory augmentation with attribute verification and search labels.
---

# Image Attribute Augmentation on NPA

Use `workflows/cosmos-data-factory/paidf-image-attribute-augmentation.yaml`.
Read [shared setup](../cosmos-data-factory-setup/SKILL.md) and the
[IAA guide](../../../workflows/cosmos-data-factory/image-attribute-augmentation.md)
for configuration, commands, inputs, and output contracts. Use EVG for
seed-image-to-event-video synthesis; IAA edits person images.

Preserve the nine-state graph: provenance → prepare images → sample configs →
Qwen Image Edit service plus real PAIDF augmentation → validate → upstream
postprocessing → Person Attribute Search → assemble dataset → reopen final
handoffs. Service and batch share one state so generation remains available
until its consumer finishes. Use generic workflow commands; upstream Airflow
manifests cannot be submitted as NPA specs.

Require authorized person images in a worker-readable S3 prefix. Keep source
person directories, but disclose that NPA processes individual images rather
than concatenating multiple views or applying Airflow's person-folder limit.
Preparation verifies dimensions/decoding, creates RGB JPEG, and records both
source/prepared hashes. There is no implicit dataset; stage the requested subset
explicitly and preserve source URIs for identity traceability.

Use `num_augmentations` and `seed` overrides for reproducible sampling. The
reviewed adapter owns attribute distributions; upstream
`cosmos.variable_distribution` is not a supported direct override. An unused
YAML key does not prove custom sampling. Keep the reviewed Qwen Image Edit
model/revision and exact private generation/attribute-search digests. Verify
explicitly selected hosted VLM/LLM availability before execution.

For planning, run `validate-spec` and `plan-spec --check-render` with intended
overrides and record deployment prerequisites separately. For a requested run,
follow the guide's access/image gates and `submit --runtime`, then monitor the
same run ID. Read `augmented_dataset/dataset.json`, generation/postprocessing
reports, PAS sidecars, `reports/upstream.json`, and terminal validation. Account
for skipped/rejected edits and review actual identity, pose, and clothing
quality; protocol success does not prove re-identification training suitability.

## Source and changes

Adapted from [NVIDIA's IAA skill](https://github.com/NVIDIA/paidf-orchestration/blob/f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9/skills/physical-ai-image-attribute-augmentation/SKILL.md).
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Upstream skill license:
CC-BY-4.0 AND Apache-2.0. Airflow deployment, payload, trigger, monitor, and
retrieval instructions were rewritten for NPA/SkyPilot. Input-preparation and
dataset-filename differences are explicit in the guide. This is an Early Access
semantic adaptation, not execution of NVIDIA's controller.
See [NOTICE-NVIDIA-PAIDF](../../NOTICE-NVIDIA-PAIDF).
