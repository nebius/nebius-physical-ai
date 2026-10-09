---
name: cosmos-data-factory-setup
description: Prepare the shared NPA runtime, storage, credentials, and private compatibility images for Cosmos Data Factory IAA or EVG workflows on Nebius and SkyPilot.
---

# Cosmos Data Factory setup

Read [the shared runbook](../../../workflows/cosmos-data-factory/README.md)
before choosing input, image, or GPU settings, then load the IAA or EVG skill.
NVIDIA's reference is Airflow on Kubernetes. NPA translates its DAGs into
`npa.workflow/v0.0.1`; upstream Helm values, Kubernetes manifests, Airflow API
calls, and payload JSON are not NPA workflow specs. Pinned augmentation templates
and label configs are consumed through the real adapters.

For authoring or review, validate and plan locally, leaving unavailable runtime
prerequisites unverified. For a requested run, resolve the authorized project,
exact context, staged S3 input, output bucket, and immutable private digests from
the current task/configuration. Reuse already supplied values; ask only for
missing required choices. Example storage and zero digests are placeholders.
IAA/EVG have no automatic demo dataset. Setup does not imply permission to
submit, publish images, or destroy infrastructure.

Check S3, Token Factory, HF, NGC, and Nebius authentication and the selected
`paidf-iaa`/`paidf-evg` plus labeling access capabilities before allocating GPUs.
Prove private-image pullability on the selected context. Keep credentials in
NPA's store or private environment and forward names through `--secret-env`;
never expose values in specs, reports, or public handoffs. Runtime access and
image redistribution are separate decisions.

IAA needs the private image-edit and attribute-search wrappers. EVG needs
event-video, detection, captioning, visual-qa, and the same attribute-search
wrapper. Refer to the compatibility recipes and acceptance evidence linked in
the runbook; source inspection does not qualify a rebuilt image. Generation
requests one B200 for IAA or two B200s on one node for EVG. Subsequent EVG
decoder/labeling stages remain GPU-backed even with a hosted VLM. Use the actual
sequential profiles rather than summing Airflow pool reservations.

Follow the runbook's generic configure, validate, plan, submit, status, logs,
artifacts, and cancel commands. Every run preserves `reports/upstream.json`
and reopens its final handoffs in `reports/terminal-validation.json`. Report
local validation, deployment readiness, protocol completion, and observed
media/QA quality separately. Cancel only the selected run and verify terminal
state before tearing down owned resources.

## Source and changes

Adapted from NVIDIA/paidf-orchestration's `paidf-orchestration-setup` skill and
shared getting-started guide at `f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9`.
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Setup was rewritten for
NPA/SkyPilot; no Airflow controller is installed by this skill.
See [NOTICE-NVIDIA-PAIDF](../../NOTICE-NVIDIA-PAIDF) for source URLs,
Apache-2.0 source and CC-BY-4.0 skill attribution, and changes.
