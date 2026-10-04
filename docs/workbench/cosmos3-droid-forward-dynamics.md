# Cosmos3 DROID forward dynamics

`workflows/testing/cosmos3-droid-forward-dynamics.yaml` is an experimental,
five-stage evaluation path for the public
[`jere-mybao/cosmos3-nano-droid-forward-dynamics`](https://huggingface.co/jere-mybao/cosmos3-nano-droid-forward-dynamics)
derivative. It is a short-horizon video forward-dynamics model, not a robot
policy or a general benchmark result.

## What the workflow does

1. Reads one hash-verified DROID selection manifest, composes the three views,
   and converts absolute poses plus source `action.gripper_position` values to
   the checkpoint's exact action representation.
2. Runtime-fetches the immutable exported checkpoint, hashes its metadata and
   every safetensors shard independently, then runs upstream native inference
   with the true actions.
3. Runs the same native model with a matched temporal permutation and with
   zeroed actions.
4. Decodes the actual MP4 outputs to calculate frame-wise RGB MSE/PSNR against
   the held-out reference and true-vs-control image differences.
5. Emits a synchronized Rerun `.rrd`, verifies it with `rerun rrd verify`, and
   writes hash-bound source/checkpoint/provenance records.

All handoffs are run-scoped S3 prefixes. The image uses a submitted source
overlay, while checkpoint, VAE, tokenizer, DROID media, and mutable caches stay
outside image layers and outside this repository.

## Exact model contract

The pinned model revision is
`1dfff3cc3b86548b208341bb123d1c4f71043114`. Its card defines a 17-frame,
15-fps window: the first composite observation is conditioning and the next 16
frames are predicted. The action tensor is `[16,64]`: channels 0–2 are relative
translation, 3–8 are the relative rotation matrix's first two columns in rot6d,
and channel 9 is `1.0 - source.action.gripper_position`; remaining channels are
zero. No action normalization is applied. The workflow uses
`droid_lerobot` (domain id 8), the same checkpoint domain named in the card.
For the upstream native per-sample JSON, the stage writes the corresponding bare
`[16,10]` JSON action array and references it from `action_path`; Cosmos
Framework verifies that raw width and performs the documented zero-padding to 64
channels itself. The generic native inference entrypoint also receives the
17-frame composite video, 15-fps, DROID domain, and 16-action length through
that same sample JSON. The `[16,64]` action artifact remains in the handoff and
is hash-bound in provenance.

The selection manifest carries `poses_abs` with 17 absolute end-effector poses
derived from `observation.state.cartesian_position`: XYZ Euler orientation is
converted to a matrix and then right-multiplied by the upstream DROID-to-OpenCV
rotation. It also carries `gripper_actions_raw` with 16 values copied from the
separate source `action.gripper_position` column. Entry *i* is paired with the
pose delta from frame *i* to frame *i + 1*. The stage rejects a missing,
out-of-range, or mis-sized gripper array rather than substituting observation
gripper state; those are distinct DROID fields and substituting one for the
other changes the checkpoint input.

Each composite has the 640×360 wrist camera on top, with the two exterior
cameras resized to half size and concatenated along the bottom, yielding a
640×540 input canvas. This preserves the model's `concat_view` geometry.

The selection manifest must either use a scene/building identifier for a
stronger separation or declare the card's exact 3% per-subset episode split
(`seed=42`, `val_ratio=0.03`, `success=57,639` or `failure=14,268`). The latter
is reproduced with the card's `torch.randperm` procedure and is held out by
episode, not scene; it remains susceptible to scene correlation. The selection
also names the card's `droid_plus_lerobot_640x360_20260412` conversion and its
immutable source revision.

The temporal permutation and zero arms are diagnostics. The card's published
40-pair action probe is not reproduced or promoted into a held-out benchmark by
this workflow. A single successful window likewise says nothing about
convergence, long-horizon rollouts, or physical-robot success.

## Attribution, terms, and packaging

| Boundary | Identity and terms | Packaging decision |
| --- | --- | --- |
| Framework source | [NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework) [`5e67049cd94acb667786f1e6dd0dab821cb90c97`](https://github.com/NVIDIA/cosmos-framework/commit/5e67049cd94acb667786f1e6dd0dab821cb90c97), OpenMDW-1.1; preserve [`LICENSE`](https://github.com/NVIDIA/cosmos-framework/blob/main/LICENSE) and [`NOTICE`](https://github.com/NVIDIA/cosmos-framework/blob/main/NOTICE) | The private validation image records this source marker, verifies the native forward-dynamics JSON schema and `droid_lerobot` mapping at build time, and carries no weights. No public image publication is authorized. |
| Derivative checkpoint | `jere-mybao/cosmos3-nano-droid-forward-dynamics@1dfff3cc3b86548b208341bb123d1c4f71043114`, publisher `jere-mybao`, [OpenMDW-1.1](https://huggingface.co/jere-mybao/cosmos3-nano-droid-forward-dynamics/blob/main/LICENSE) | Runtime fetch only. The checkpoint is never copied to NPA image layers or a public NPA registry. |
| Base model | [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano), NVIDIA, OpenMDW-1.1 | Inherited derivative lineage; no separate base checkpoint is baked. |
| Data | [`nvidia/Cosmos3-DROID@5c11a20accb11497270a5247a7f1e66ad04c956c`](https://huggingface.co/datasets/nvidia/Cosmos3-DROID), NVIDIA's LeRobot conversion, OpenMDW-1.1; it credits [DROID](https://arxiv.org/abs/2403.12945), whose authors are Khazatsky, Pertsch, Nair, Balakrishna, Dasari, Karamcheti, et al. The derivative card further records the raw DROID source as CC-BY 4.0. | Operator-supplied, run-scoped inputs only; never baked, mirrored, or republished by NPA. |
| Runtime auxiliary payloads | Model card identifies `Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203` for the Wan VAE and `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` tokenizer; both cards currently declare Apache-2.0. | The upstream framework resolves these at runtime. Their terms/access are independent of the derivative checkpoint; no payload is distributed by this workflow. |
| Outputs | OpenMDW-1.1 does not impose an output-use restriction | Run outputs still carry DROID/Cosmos provenance and must be handled by the operator consistently with any source-data obligations. |

OpenMDW-1.1 permits dealing in covered Model Materials subject to retaining the
agreement and applicable origin notices when distributing them. This workflow
does not distribute the model material: it performs an operator-owned runtime
fetch and retains links, copyright, and provenance. It introduces no NPA EULA,
acceptance variable, telemetry consent, or duplicate attestation. Existing
provider access is checked through the normal NPA health/access surfaces before
execution; access is not a redistribution conclusion.

The model card records framework commit
`9cbd0841b50a1e667577292be1a4ad79cbc8e3d9` in its export manifest. NVIDIA's
current public Git remote no longer serves that commit. The qualification path
uses the available immutable OpenMDW-1.1 source pin above, writes it into the
image as `.npa_source_revision`, and fails inference if it is not present. The
image's build verifier also checks the upstream action sample JSON fields and
action loader plus DROID raw width/domain mapping. Artifacts retain both the
card export commit and actual runtime source revision, so this is not represented
as a source-level reproduction of the retired commit.

NPA’s modifications are limited to staging and hash-checking the selected input,
constructing the documented DROID action format, calling the upstream native
entrypoint, computing decoded-video diagnostics, and emitting Rerun/provenance
artifacts. It does not claim the derivative, Cosmos Framework, Cosmos3-DROID, or
DROID research as original NPA work. Each visualization provenance artifact
links the checkpoint publisher, framework license/NOTICE, data source, copyright,
and these modifications.

## Citation

Credit the derivative publisher, NVIDIA Cosmos, and DROID. The model card asks
for the following DROID citation:

```bibtex
@inproceedings{khazatsky2024droid,
  title={DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
  author={Khazatsky, Alexander and Pertsch, Karl and Nair, Suraj and Balakrishna, Ashwin and Dasari, Sudeep and Karamcheti, Siddharth and others},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024},
  url={https://arxiv.org/abs/2403.12945}
}
```

## Validation status

The local stage contract is tested with generated, decoded three-view MP4s,
hash-bound handoffs, actual MSE/PSNR calculations, and a Rerun file accepted by
`rerun rrd verify`. A genuine card-held-out DROID selection has also been
re-emitted privately from the correct state-pose and source-action columns. On
an operator-private RTX PRO 6000 route, the image completed remote preparation
and reached the qualified framework checkpoint-fetch path. That attempt exposed
a source-overlay provenance bug after the fetch receipt; its regression fix is
tested locally and awaits a freshly staged native run. Schema validation and an
explicit-private-image static plan pass, while the quarantined public default
correctly refuses to render. The adjacent
[hash-bound readiness record](../../workflows/testing/cosmos3-droid-forward-dynamics.readiness.json)
tracks the separately verified local contract and live prerequisites.
It is not yet a live-accepted capability: completion requires a real
NPA/Kubernetes RTX PRO 6000 GPU run on the selected held-out DROID window, the exact runtime
checkpoint fetch, and independent S3 artifact inspection. The unavailable
historical framework commit prevents a claim of exact source-level card
reproduction.
