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
| Exact DM05 policy source | `hbzfeng/lerobot@6eede4f7d2efe6b4f6a58ddb7b13ed55e2346b9c` (the source in closed/superseded upstream LeRobot PR [#4051](https://github.com/huggingface/lerobot/pull/4051)) |
| Maintained source, not checkpoint-parity source | `huggingface/lerobot@fea7153a06d449d5acf3783fa2daf8df8b49e79e` (PR #4721; requires the documented processor conversion) |
| Simulator code / data source | `Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01` |
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

## Attribution, terms, and packaged notices

The exact checkpoint-compatible policy is the Apache-2.0 implementation from
[`hbzfeng/lerobot`](https://github.com/hbzfeng/lerobot/tree/6eede4f7d2efe6b4f6a58ddb7b13ed55e2346b9c),
provided with LeRobot PR #4051. That PR was closed/superseded rather than
merged; the repository source at the pinned revision declares Copyright 2024
The Hugging Face team. Individual DM05 modules additionally credit Dexmal and
Hugging Face contributors. The revision has an Apache-2.0 `LICENSE` and no
separate `NOTICE` file. The private validation Dockerfile retains that source
and its license at `/opt/lerobot/dm05-source`, carries
`DM05-ATTRIBUTION.md`, and labels the source/revision/license. NPA does not
patch the upstream policy: its changes are the external five-stage adapter,
runtime-manifest check, and provenance/reporting logic only.

`Lifelong-Robot-Learning/LIBERO` at the pinned revision declares MIT (Copyright
2023 Lifelong Robot Learning) for its benchmark source and simulator assets,
which the operator-private validation image retains with its `LICENSE` at
`/opt/lerobot/libero-benchmark`. Its demonstration dataset is CC BY 4.0 and is
not copied into this derivative image. The enhanced checkpoint card identifies Dexmal Team as author, names
`Dexmal/DM05-Lerobot` as its base model, and labels the weights `gemma`. The
model card's [Gemma Terms of Use](https://ai.google.dev/gemma/terms) are a
separate weights/service/redistribution boundary. The image contains no model
weights, processor cache, dataset, output, or acceptance record; exact
checkpoint revisions are fetched at operator runtime using the configured
credential path. The private validation image is not an official NPA public
image release, and this does not claim a right to redistribute checkpoints,
model-derived artifacts, or a hosted service.

`dexmal/opendm@7d52f1591437332cb0157be3303c1c46da811344` is Apache-2.0 and is
credited only as the DM05 training/evaluation context. This independently owned
checkpoint workflow neither claims OpenDM training/serving nor treats its
capabilities as interchangeable.

No NPA EULA checkbox, `ACCEPT_*` variable, telemetry/privacy consent, or
duplicate per-image attestation was added. Exact-revision payload access is an
operational fetch result, not legal assent or a redistribution grant.

### Upstream citations

The checkpoint model card supplies:

```bibtex
@misc{dm05,
    title  = {{DM0.5}: An Open-World Foundation Model for General-Purpose Embodied Intelligence},
    author = {{Dexmal Team}},
    month  = {July},
    year   = {2026},
    url    = {https://www.dexmal.com/blog/dm0.5/index_en.html}
}
```

The pinned LeRobot source asks users to cite its repository:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Meftah, Khalil and Ellerbach, Maxime and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = {\url{https://github.com/huggingface/lerobot}},
    year = {2024}
}
```

The LIBERO repository supplies:

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

## Validation state and resumption

The workflow validates, plans, and has a local five-stage execution harness
with mocked checkpoint fetch plus native-shaped `eval_info.json`, real encoded
MP4s, and a decoded RRD. It also requires an exact source manifest and confirms
that the installed registry resolves `dm05` to
`lerobot.policies.dm05.DM05Policy` before it invokes `lerobot-eval`. A successful
native evaluator is the stage's checkpoint-load proof; its `rollout.json`
records the exact implementation provenance only after that evaluator exits.
These local checks prove the connected artifact contract and source selection,
not checkpoint inference or benchmark performance.

Live evaluation remains unverified until an operator-private digest built from
`npa/docker/workbench/lerobot/Dockerfile.dm05-validation` passes target pull
preflight and the five stages finish. Inspect the final `metrics.json`, both
native `eval_info.json` files, decoded comparison MP4, and decoded RRD from that
same run before making any live-ready or reproduction claim. The maintained
PR #4721 implementation must not be substituted for this checkpoint-parity
path unless its documented processor conversion is also executed and recorded.

### Baseline compatibility finding

The documented `Dexmal/DM05-Lerobot` predecessor is a real DM05 checkpoint but
its pinned published configuration has 14-dimensional state and action spaces
with `chunk_size=50`, `n_action_steps=50`, and `add_state=true`. It is not the
candidate's 8-state/7-action LIBERO representation. The rollout adapter
records this exact contract and rejects it before evaluation rather than adding
an uncredited action/state adapter or calling an invalid result a matched
baseline. A complete comparison needs a released, provenance-pinned 8/7
LIBERO-compatible predecessor or a separately documented native OpenDM
baseline; neither is substituted automatically.

### LIBERO runtime assets

Upstream LIBERO prompts interactively to create its first user-level path
configuration. The private noninteractive workflow instead writes a run-local
`LIBERO_CONFIG_PATH/config.yaml` pointing only to the pinned MIT benchmark
closure (BDDL files, initialization states, and assets). It does not answer a
terms prompt, add an acceptance flag, or bake/download a demonstration dataset.
The runtime manifest binds this closure's source revision and license; a missing
closure fails the rollout explicitly.
