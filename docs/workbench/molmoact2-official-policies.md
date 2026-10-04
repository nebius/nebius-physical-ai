# Official MolmoAct2 policies (LIBERO first)

This is an AllenAI MolmoAct2 BYOF candidate, not NPA-original research and not
a claim that its foundation checkpoint is a safe deployment policy. It starts
with the upstream-supported LIBERO path and uses the upstream-pinned LeRobot
integration rather than duplicating policy runtime code.

## Exact sources, credit, and lineage

| Boundary | Pinned identity / decision |
| --- | --- |
| Source | [AllenAI `molmoact2`](https://github.com/allenai/molmoact2) `6070080a20321b4f498ab30f28e1d09ac465edb7`, Apache-2.0 ([LICENSE](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/LICENSE)) |
| Upstream LeRobot integration | Apache-2.0 `experiments/lerobot` tree committed in the exact MolmoAct2 source revision; credit to the Hugging Face LeRobot contributors. The upstream root's separate `lerobot/` gitlink is `80633827176a0203064cb141383664fba024e050`, but the runnable trainer imports `experiments/lerobot`; they are not treated as interchangeable. |
| Foundation checkpoint | [`allenai/MolmoAct2`](https://huggingface.co/allenai/MolmoAct2) `e432d85f6e039edca44afb93c262f3084ab72a9c`; runtime fetch only. The upstream page identifies it as a foundation checkpoint, not a ready-to-run deployment policy. |
| Official LIBERO policy/data | [`allenai/MolmoAct2-LIBERO`](https://huggingface.co/allenai/MolmoAct2-LIBERO) `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`; [`allenai/MolmoAct2-LIBERO-Dataset`](https://huggingface.co/datasets/allenai/MolmoAct2-LIBERO-Dataset) `fe3ead447f44c0ea950396360b304cc2fb6be8f8`, dataset card `apache-2.0`; runtime fetch only. |
| LIBERO simulator | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), MIT, installed through the upstream LeRobot `libero` extra. |

The source repository states Apache-2.0 for the model family. At onboarding
time the listed Hugging Face checkpoint pages did not expose a separate model
repository `LICENSE` file or `license` card field. This is recorded rather than
silently broadened; checkpoint bytes remain outside the image and every run
records the exact checkpoint revision. The image has the upstream LICENSE,
NOTICE links, source revision, LeRobot revision, and a third-party notice; run
artifacts retain checkpoint/dataset/source provenance.

Use the upstream citation verbatim when reporting this work:

```bibtex
@misc{fang2026molmoact2actionreasoningmodels,
  title={MolmoAct2: Action Reasoning Models for Real-world Deployment},
  author={Haoquan Fang and Jiafei Duan and Donovan Clay and Sam Wang and Shuo Liu and Weikai Huang and Xiang Fan and Wei-Chuan Tsai and Shirui Chen and Yi Ru Wang and Shanli Xing and Jaemin Cho and Jae Sung Park and Ainaz Eftekhar and Peter Sushko and Karen Farley and Angad Wadhwa and Cole Harrison and Winson Han and Ying-Chun Lee and Eli VanderBilt and Rose Hendrix and Suveen Ellawela and Lucas Ngoo and Joyce Chai and Zhongzheng Ren and Ali Farhadi and Dieter Fox and Ranjay Krishna},
  year={2026}, eprint={2605.02881}, archivePrefix={arXiv}, primaryClass={cs.RO},
  url={https://arxiv.org/abs/2605.02881}
}
```

## Contracts and supported scope

The executable path is [`molmoact2-official-policies.yaml`](../../workflows/testing/molmoact2-official-policies.yaml).
Its five substantive stages are:

1. Download the immutable LIBERO LeRobot dataset, check state/action/camera
   features, then use upstream LeRobot `split_dataset` to make non-overlapping
   reindexed train and held-out datasets.
2. Fine-tune with upstream `launch_scripts/train_lerobot.py`; the stage maps
   only the generated train subset under the upstream-required repo path.
3. Run upstream LeRobot `lerobot-eval` in closed-loop LIBERO and retain its
   actual JSON and MP4 outputs.
4. Run upstream `run_open_loop_inference_lerobot.py` on the reindexed held-out
   episode, preserving raw action MSE and normalization metadata.
5. Emit a Rerun `.rrd` from those completed metrics and copy an
   evaluator-produced MP4. It refuses to create visualization evidence without
   an actual earlier MP4.

The LIBERO tag contract is upstream-native: `observation.state`, `action`,
`observation.images.image`, and `observation.images.wrist_image`; unnormalized
gripper; delta end-effector action; 10-step action horizon / 10 executed
actions. The adapter validates all keys before training and uses the checkpoint
`norm_tag=libero`; it does not quietly rename cameras or normalize the gripper.
The training stage uses the upstream LoRA fine-tuning form (`ft_vlm`, action
expert, LM head, rank-64 LoRA) so the upstream trainer emits its documented
complete `step*-merged` checkpoint for the downstream LeRobot rollout. It
does not supply a locally invented duration, GPU count, or benchmark-success
threshold.

The rollout stage passes LeRobot's upstream default of 50 evaluation episodes;
it is not shortened to an onboarding-only trial count. Its success metric is
reported as factual simulator evidence, not as a convergence benchmark or a
physical-robot result.

The source also publishes contracts for, but this workflow does not yet accept:

| Capability | Official upstream contract | Status |
| --- | --- | --- |
| DROID | `observation.state`; exterior-1-left, exterior-2-left, wrist-left cameras; absolute joint pose; unnormalized gripper; horizon/actions 15; `norm_tag=franka_droid` | Deferred. The anonymous exact `allenai/droid_lerobot` access probe returned HTTP 401. No NPA EULA or acceptance variable is added; an operator-authorized exact revision probe is the narrow next gate. |
| SO100/101 | `observation.state`, `action`; absolute joint pose; normalized gripper; horizon/actions 30; `norm_tag=so100_so101_molmoact2` | Contract documented only. Upstream's mixture deliberately has no fixed camera-key list; physical use must preserve the checkpoint/configured camera names, calibration, embodiment, safety limits, and attached hardware. No physical-robot success is claimed. |

The DROID and SO100/101 fine-tuned model pages are not substituted for each
other or for the foundation model. None is promoted to a deployment policy by
this integration.

## Packaging, terms, and execution boundary

`npa/docker/workbench/molmoact2/Dockerfile` is an operator-private BYOF recipe,
not a published NPA image. It clones the two pinned Apache-licensed sources
with Git LFS smudging disabled, is digest-pinned to its CUDA base, runs as a
non-root user, disables W&B telemetry, and has no checkpoints, datasets,
caches, signed URLs, or credentials in its layers. A private build and a real
GPU workflow run still require the normal secure-image, artifact inspection,
and secret-leak gates before any live-ready claim. It is intentionally absent
from the public container catalog until byte-level redistribution proof and
anonymous-pull verification exist.

No new NPA EULA, `ACCEPT_*` variable, or duplicate attestation is introduced.
The public LIBERO artifacts use anonymous runtime fetch. The operator scope
record is referenced privately by the onboarding run; it is not a declaration
of commercial/noncommercial use or acceptance of any new vendor term. No
optional telemetry or privacy consent is enabled.

Before a live submission, build into the operator-controlled registry, resolve
the resulting immutable digest, and pass it through the workflow as
`runtime_image`. Then run `npa workbench health preflight --checks nebius`
before provisioning. A completed run is accepted only after independent S3
artifact inspection confirms the evaluator's metrics, decoded MP4, and RRD;
an image build, import smoke, or foundation-checkpoint download is not
evidence of policy quality or physical-robot success.
