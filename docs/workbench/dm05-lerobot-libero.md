# DM05-Lerobot-LIBERO checkpoint comparison

This is an experimental checkpoint-evaluation workflow, not a new NPA model
release or an OpenDM replacement. It compares the public Dexmal
`DM05-Lerobot-LIBERO` checkpoint with its documented predecessor
`DM05-Lerobot` using the checkpoint card's native `lerobot-eval` LIBERO
protocol. The implementation is
[`dm05-lerobot-libero-comparison.yaml`](../../workflows/testing/dm05-lerobot-libero-comparison.yaml).

## Protocol and capability boundary

The successful path has five connected, substantive stages:

1. seal the 40-task, two-camera, seed-7 episode plan;
2. fetch the exact predecessor revision at runtime and run closed-loop LIBERO
   rollouts through upstream `lerobot-eval`;
3. fetch the exact enhanced revision at runtime and run the identical rollouts;
4. calculate per-suite and total success deltas from the two native
   `eval_info.json` artifacts; and
5. independently decode native rollout MP4s, emit a side-by-side MP4, and
   write a metrics/provenance RRD.

Both rollout stages consume the exact `protocol.json` checksum emitted by the
first stage. Metrics refuse mismatched protocol hashes, roles, incomplete
40-task grids, non-boolean episode success values, or unequal episode counts.
The report refuses absent/undecodable native MP4s and mismatched metric
provenance. It never turns the model-card table into a local result.

| Contract | Pinned value |
| --- | --- |
| Enhanced checkpoint | `Dexmal/DM05-Lerobot-LIBERO@c22df98af5a69e7b9f6bfc1086d1a6982e647b26` |
| Documented predecessor | `Dexmal/DM05-Lerobot@716afe317bfd01fa4d7ad7cfb84e3b19b7bd934d` |
| LeRobot evaluator source | `huggingface/lerobot@30da8e687a6dfc617fcd94afc367ac7071c376ce` (v0.6.0) |
| Simulator source | `Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Cameras | `agentview_image → front`, `robot0_eye_in_hand_image → wrist` |
| Observation / model contract | 256×256, state dimension 8, action dimension 7, `chunk_size=10`, `n_action_steps=10`, `add_state=false` |
| Action/controller distinction | model processor actions are **absolute** (`use_relative_actions=false`); LIBERO environment control is **relative** |
| Published evaluation | 40 tasks × 5 episodes, seed 7 = 200 episodes |

The card reports Spatial 49/50, Object 50/50, Goal 50/50, and LIBERO-10
48/50, for 197/200 (98.5%). That is a reported 200-episode experiment. It is
not a standard 2,000-episode complete LIBERO result, a matched baseline delta,
or physical-robot evidence. A workflow report only sets
`published_197_of_200_reproduced=true` when its candidate's actual native
rollout totals exactly 197 out of 200.

## Attribution and distribution boundary

- Dexmal Team's model-card citation is retained: *DM0.5: An Open-World
  Foundation Model for General-Purpose Embodied Intelligence* (July 2026),
  https://www.dexmal.com/blog/dm0.5/index_en.html. The enhanced model derives
  from `Dexmal/DM05-Lerobot`, itself a fine-tune of `Dexmal/DM05`.
- `dexmal/opendm@7d52f1591437332cb0157be3303c1c46da811344` is Apache-2.0;
  its documented LIBERO configuration is credited as the upstream DM05
  training/evaluation context. This checkpoint workflow does not claim OpenDM
  training, serving, or benchmark results.
- `huggingface/lerobot@30da8e687a6dfc617fcd94afc367ac7071c376ce` is Apache-2.0.
  `LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01` is MIT. Preserve their
  notices and citations when retaining derived artifacts.
- The checkpoint card labels both DM05 LeRobot checkpoints `gemma`. The current
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) govern use and set
  conditions for redistribution or a hosted service. The workflow ships no
  checkpoint, cache, or model-derived image layer; it fetches exact revisions
  only at runtime. No public redistribution, hosted service, or right to
  redistribute the checkpoint is claimed.

No NPA EULA checkbox, `ACCEPT_*` variable, telemetry/privacy consent, or
duplicate per-image attestation was added. The public exact-revision payload
range probes succeeded during onboarding; that proves technical access at that
time, not an independent legal conclusion or permission to redistribute.

## Validation state and resumption

The workflow validates, plans, and has a local five-stage execution harness
with mocked checkpoint fetch plus native-shaped `eval_info.json`, real encoded
MP4s, and a decoded RRD. This proves the artifact contract, not checkpoint
inference.

Live evaluation remains unverified. The public checkpoint declares custom
LeRobot policy type `dm05`, while the pinned NPA LeRobot v0.6.0 policy registry
does not provide that type and the checkpoint repository contains no custom
policy source. The first live run must use a reviewed immutable image that
demonstrably provides the exact DM05 implementation, then separately inspect
the final `metrics.json`, native `eval_info.json`, decoded comparison MP4, and
decoded RRD before making a live-ready or reproduction claim.
