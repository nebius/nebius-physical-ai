---
name: opendm
description: Run or validate Dexmal OpenDM DM05 LIBERO preparation, full training, HTTP inference, closed-loop evaluation, and factual RRD/MP4 artifacts through the private NPA workflow.
---

# OpenDM DM05

Use this skill for the Dexmal OpenDM DM05 LIBERO path. It is a private BYOF
candidate, not a public NPA image, hosted service, or a claim of physical robot
success. For generic packaging and workflow mechanics, load `byof-onboard`,
`oss-solution-registry-onboard`, and `author-npa-workflow` as needed.

## Pinned upstream boundary

- OpenDM: `https://github.com/dexmal/opendm` at
  `7d52f1591437332cb0157be3303c1c46da811344`, Apache-2.0, Copyright 2026
  Dexmal.
- Base checkpoint: `Dexmal/DM05` at
  `5cd18734814abb075a9ccfd9ad6d16777b5cf10e`; the card declares Gemma terms.
  Fetch it only at runtime and do not borrow OpenPI's acceptance environment
  switch.
- Training data: `Dexmal/libero` at
  `f15a66b3975f8cd210c746991f80adde5ab05ca4`. Its card says only `license: cc`
  without a version. Keep it runtime-only and do not make a redistribution
  claim until the exact data license is resolved.
- Closed-loop evaluator: `dexmal/dexbotic-benchmark` at
  `789b87f50d9fadc7663d2e8bac057941221aab81`, MIT; its LIBERO submodule is
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`, MIT.

Keep source notices, source metadata, the model citation, checkpoint lineage,
and any local modifications in run provenance. A successful payload probe is
only operational access; it does not grant redistribution or hosted-service
rights.

## Required five-stage behavior

Use [`workflows/testing/dm05-opendm.yaml`](../../../workflows/testing/dm05-opendm.yaml).
The successful path must retain all five real stages:

1. Runtime-fetch and organize the exact LIBERO data, then compute actual
   OpenDM normalization statistics.
2. Runtime-fetch the exact base model and run upstream `dm05_libero.py`
   full SFT through `script/libero_runner.sh`.
3. Start upstream `/v1/infer` from that resulting checkpoint and capture a real
   action chunk.
4. Re-start that exact checkpoint-derived server and run Dexbotic's real
   closed-loop LIBERO evaluator.
5. Emit factual metrics/provenance RRD and preserve an evaluator-created MP4.

Do not replace any of these with import tests, an echo manifest, fabricated
metrics, or an assumption that a localhost model server persists between jobs.
The evaluation stage uses separate visible GPUs for the HTTP server and
evaluator in one job.

## Normalization contract

Fail closed before serving if the prepared data or model response differs from:

- dataset `libero_pi0_all`;
- camera keys `images_1`, `images_2` in the prompt order `Head`, `Left wrist`;
- 8D Franka state: six joint values followed by two gripper values;
- 7D action and chunk size 10; and
- the `norm_stats.json` generated for that data/action configuration and copied
  into the trained checkpoint.

Use the upstream HTTP endpoint, not a custom policy protocol. The Dexbotic
LIBERO evaluator sends ordered agent/wrist cameras and an 8D EEF-pose-plus-
gripper state. Record that it is not the six-joint training-state convention;
the pinned OpenDM LIBERO config has `add_state=False`, so the state is parsed
and shape-normalized but not model-conditioned. Do not conceal that distinction
or turn it into a state-conditioned benchmark claim.

## Build, execute, inspect

Before build, provisioning, or submission, run `npa workbench health preflight
--checks nebius,s3 --json` and probe the exact public model/data payloads under
the operator's credential context. Build only an operator-private,
digest-pinned image. It must exclude checkpoint/data/cache/credential/output
bytes and must not enable optional telemetry or privacy consent. Do not use the
published Dexbotic `latest` image as a base because it inherits unrelated
EULA/privacy environment variables.

Validate and plan the YAML, then submit with an operator bucket and the exact
private image digest. Use source staging because the stage adapter lives in
this NPA checkout. Independently inspect the report manifest, upstream
`results.json`, RRD, and decoded MP4 after completion. Distinguish an
operational smoke from the default 100,000-step / 50-trials-per-task full
configuration; neither is physical-robot evidence.

See [`docs/workbench/dm05-opendm.md`](../../../docs/workbench/dm05-opendm.md)
for the attribution, legal boundary, private image recipe, and current
acceptance status.
