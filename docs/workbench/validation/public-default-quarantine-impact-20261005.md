# Public default quarantine impact, 2026-10-05

PR #807 intentionally quarantined published images that violate the current
image contract. PR #869 supplies repaired, digest-bound development defaults for
the PAIDF actions. Historical release tags remain quarantined.

This audit examines both planning dispositions in all 133 shipped declarative
workflows, using their real resource profiles and no image or registry overrides.
It includes shell states whose resource profile selects a quarantined tool.
It checks image selection; it does not claim GPU execution or acceptance of the
blocked images. The guardrail compares the table below with actual default
resolution, so a wildcard image override cannot conceal another regression.

After PAIDF's repaired defaults, 34 workflows still fail default image planning:
27 select the stale image families withdrawn by #807, and seven require images
without an accepted public release. The audit checks every task, including
unbuilt and validation candidates outside the stale-image inventory.

## Repaired workflow defaults

The normal Physical AI Data Factory, NVIDIA PAIDF VDA Cosmos Transfer 2.5, and
PAIDF Cosmos3 variants select the repaired Evaluator and Curator candidates.
PAIDF Cosmos3 also selects the repaired Cosmos3 candidate for video preparation
and variant generation. These pins and their qualification scope live in
`npa/src/npa/deploy/public_release_manifest.json`, beside publication policy.

The scope remains explicit because image-byte safety and advertised runtime
capability are separate claims. Qualifying Nano variant generation does not
qualify checkpoint evaluation, policy training, or the separate Ray Serve image.
An explicit operator image or registry still takes precedence.

The live normal PAIDF run also exposed stale adapter behavior in the accepted
viewer image: its recording retained private S3 locations. The normal spec now
requests `source_overlay: true`, as the Cosmos3 and direct Transfer specs already
do, so every stage uses the submitted adapters and their portable report
projection. The guardrail checks this setting in all three specs. Image-byte
acceptance alone does not prove that an older baked SDK implements current
artifact behavior.

## Still blocked shipped workflows

These 27 workflows need a repaired, scanned and capability-qualified image or an
explicit operator-owned image. A repaired parent does not repair the layers of
an already-published derivative. The table is an outstanding qualification
inventory, not an instruction to remove quarantine.

| Workflow | Quarantined image tools |
| --- | --- |
| `workflows/partners/antioch/antioch-offline-policy-train.yaml` | `lerobot` |
| `workflows/testing/adversarial-scenario-hardening.yaml` | `isaac-lab` |
| `workflows/testing/cosmos3-checkpoint-eval.yaml` | `cosmos3` |
| `workflows/testing/cosmos3-fastwam-k2-eval.yaml` | `cosmos3` |
| `workflows/testing/cosmos3-generate.yaml` | `cosmos3` |
| `workflows/testing/cosmos3-policy-model-factory.yaml` | `cosmos3` |
| `workflows/testing/cosmos3-ray-batch.yaml` | `cosmos3-ray-serve` |
| `workflows/testing/hardening-with-insights.yaml` | `isaac-lab` |
| `workflows/testing/isaac-arena-evaluation-b200.yaml` | `isaac-arena` |
| `workflows/testing/isaac-arena-evaluation-rtxpro.yaml` | `isaac-arena` |
| `workflows/testing/isaac-franka-capture-reason.yaml` | `isaac-lab` |
| `workflows/testing/multicamera-rgbd-capture.yaml` | `isaac-lab` |
| `workflows/testing/rgbd-scan-to-isaac.yaml` | `isaac-lab, sonic` |
| `workflows/testing/rl-policy-training-sim-success.yaml` | `isaac-lab` |
| `workflows/testing/robocasa-data-policy.yaml` | `lerobot` |
| `workflows/testing/scan-to-isaac-navigation.yaml` | `isaac-lab` |
| `workflows/testing/shared-scene-navigation.yaml` | `isaac-lab` |
| `workflows/testing/sonic-eval.yaml` | `sonic` |
| `workflows/testing/sonic-export-eval.yaml` | `sonic` |
| `workflows/testing/sonic-export.yaml` | `sonic` |
| `workflows/testing/sonic-locomotion-finetuning.yaml` | `sonic` |
| `workflows/testing/sonic-train.yaml` | `sonic` |
| `workflows/testing/tokenfactory-cosmos-gate.yaml` | `lerobot-vlm-rl` |
| `workflows/testing/tokenfactory-rollout-judge-combo.yaml` | `lerobot` |
| `workflows/testing/tokenfactory-scene-to-rollout-judge.yaml` | `lerobot` |
| `workflows/testing/tokenfactory-train-triage.yaml` | `lerobot` |
| `workflows/testing/video-variant-sweep-cosmos3.yaml` | `cosmos3` |

The [video variant sweep guide](../../../workflows/guides/video-variant-sweep.md)
provides the explicit immutable image override used for its completed B200
validation. That run qualifies the documented override and workload; it does
not accept the quarantined catalog default.

## Other quarantined images and Sim2Real

The twelve stale image tools are `cosmos-curate`, `cosmos-evaluator`, `cosmos3`,
`cosmos3-ray-serve`, `isaac-lab`, `isaac-arena`, `genesis`, `lerobot`,
`lerobot-vlm-rl`, `loop-eval`, `reference-policy`, and `sonic`.
Genesis, Loop Eval and Reference Policy have no selected default in the above
shipped plans. They still affect direct CLI/SDK consumers and the legacy
Sim2Real configuration, which resolves Reference Policy, LeRobot VLM RL,
Loop Eval and Isaac Lab and therefore fails closed on public defaults.

The canonical `workflows/main/sim2real.yaml` already requires explicit immutable
controller, Transfer, EnvGen, Isaac and viewer images. That operator-input
contract is distinct from the legacy automatic public selection; PAIDF's
candidate mapping does not change it. Optional LeRobot transfer workflows use
their separately qualified 0.6.0 digest and do not select the quarantined default
LeRobot bytes.

Layer quarantine covers the first six tools listed above; the other six have
runtime or metadata violations. All remain in `publication_pending`. Neither
source repair nor unit tests alone promote a public release. The old release
quarantine and full capability acceptance gates remain enforced.

## Other defaults without an accepted public release

These seven workflows are additional qualification gaps, separate from the
previously accepted images withdrawn by #807. Curobo, LIBERO, MJLab and NCore
were already publication candidates before #807. That PR made the common
publication quarantine apply to default consumption as well; NCore's explicit
qualified-image requirement already existed. Open3D was registered later and
has not earned a supported public release. Do not treat a planning sentinel or
a development tag as evidence that its full workload is accepted.

| Workflow | Quarantined image tools |
| --- | --- |
| `workflows/testing/byof-libero.yaml` | `libero` |
| `workflows/testing/curobo-benchmark.yaml` | `curobo` |
| `workflows/testing/mjlab-eval.yaml` | `mjlab` |
| `workflows/testing/mjlab-render.yaml` | `mjlab` |
| `workflows/testing/mjlab-train-eval.yaml` | `mjlab` |
| `workflows/testing/nurec-colmap-reconstruct.yaml` | `ncore` |
| `workflows/testing/open3d-registration.yaml` | `open3d` |

These paths need exact independently qualified operator images or a separately
reviewed public default. Their existing license, customer authorization, model
access and workload acceptance requirements continue to apply.
