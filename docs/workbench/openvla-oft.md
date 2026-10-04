# OpenVLA-OFT LIBERO runtime

This is an integration of upstream work, not original NPA research. It executes
the official OFT continuous-action recipe and preserves the source, component,
and measured-artifact lineage needed to distinguish it from stock OpenVLA.

## Runtime contract

`workflows/testing/openvla-oft-libero.yaml` has five connected substantive
stages:

1. `prepare` materializes supplied RLDS TFRecords, decodes one trajectory through
   the pinned deterministic `dlimp.DLataset.from_tfrecords` reader, and records
   that decode evidence with the data inventory and normalization metadata.
2. `train` materializes the exact base-model Hub revision in the writable cache,
   then invokes the pinned upstream `vla-scripts/finetune.py` through upstream-
   style single-node eight-process `torchrun`, with its continuous L1 action
   head, 8-D proprioception projector, two images, and LoRA rank 32.
3. `rollout` invokes upstream `run_libero_eval.py` in closed loop and retains
   the upstream MP4s and final log.
4. `evaluate` independently checks the raw episode/success counts, recalculates
   the rate, and emits a 95% Wilson interval.
5. `visualize` writes a factual SVG/CSV from those verified metrics.

Every durable handoff is a hash-bound bundle. A training or rollout checkpoint
must include all of `lora_adapter`, `action_head--*`, and
`proprio_projector--*`; a stock OpenVLA decoder-only checkpoint is rejected.
This is the stable contract consumed by related LIBERO-Plus, Sylvest, and
LoRAFleet work. It does not imply their data, checkpoints, or results are
interchangeable.

The checked-in template carries upstream's published eight-process, 150,005-step
recipe and 50 trials per task. It requests the NPA `RTXPRO6000` accelerator alias
for both training and evaluation, so the route resolves both phases to one GPU
product. This is a deployment selection, not a paper-reproduction claim. It
makes no convergence, full-benchmark, or physical-robot claim. A deliberately
small operational smoke must be identified as such, not presented as those
results.

## Upstream lineage and credit

| Layer | Identity and credit | How NPA handles it |
| --- | --- | --- |
| OFT source | [moojink/openvla-oft](https://github.com/moojink/openvla-oft), commit `e4287e94541f459edc4feabc4e181f537cd569a8`; MIT; © 2025 Moo Jin Kim, Chelsea Finn, Percy Liang | Runtime-fetched into an operator-owned cache; its source remains out of the image. |
| OFT method | Kim, Finn, Liang, “Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success,” arXiv:2502.19645 (2025) | Cite this work when using OFT results. |
| Base model/source | [OpenVLA](https://github.com/openvla/openvla) commit `c8f03f48af692657d3060c19588038c7220e9af9`; code MIT; © 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti | Base `openvla/openvla-7b` resolves at runtime only. |
| Base weights | [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b), Hub revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | The current model card says MIT, but upstream OpenVLA’s own README says Llama-2-derived pretrained models are subject to the [Llama Community License](https://ai.meta.com/llama/license/). Treat that source-level statement as controlling for the base weight; do not redistribute it in an NPA image. |
| LIBERO simulator runtime | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`; MIT; © 2023 Lifelong Robot Learning | Fetched and installed only into the operator-owned runtime cache, following the OFT LIBERO guide. Its source, simulator dependencies, and assets are not image bytes. |
| OFT direct runtime dependencies | [custom Transformers fork](https://github.com/moojink/transformers-openvla-oft), commit `bc339d9ad707454c0c115970db43c260067c61ab`, Apache-2.0; [kvablack/dlimp](https://github.com/kvablack/dlimp), commit `92e3eca97af3b14d0b6aa15182c0dc240407698d`, Apache-2.0 | Both are runtime-only and pinned. OFT declares the moving, unlicensed `moojink/dlimp_openvla` fork, so NPA installs its editable source with `--no-deps`, installs the licensed dlimp parent with `--no-deps`, and supplies the fork's only behavioral delta: `dlimp/dataset.py` sets `options.deterministic = True`. The ready marker records the Apache notice hash, source modification hashes, and an installed parallel-map ordering probe. |
| LIBERO RLDS | [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds), revision `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`, model-card license `mit` | Supplied by the operator/staged to the run; never baked or republished by this solution. The dataset card asks users also to cite LIBERO. |
| OFT suite adapters | `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` and `...-spatial-object-goal-10`; exact revisions plus expected `lora_adapter`, `action_head--<step>_checkpoint.pt`, and `proprio_projector--<step>_checkpoint.pt` are in `OFFICIAL_SUITE_CHECKPOINTS` | Public Hub cards currently advertise MIT, but adapters and cache remain runtime-only; source/copyright/notice lineage stays in every artifact manifest. A newly trained bundle is never labelled as an official adapter. |

The OFT upstream README supplies the BibTeX entry for arXiv:2502.19645. The
OpenVLA source and modified LIBERO dataset cards supply the respective OpenVLA
and LIBERO citations. Preserve all of them in downstream reports that use their
data or results.

## Distribution and access decision

No source, base weights, adapter/checkpoint, RLDS data, generated cache,
credential, or user output is part of the solution image. The selected shape is
a neutral bootstrap plus operator-owned runtime fetch. That avoids asserting a
redistribution right for the Llama-derived base or a derived checkpoint merely
because an endpoint is publicly readable.

This onboarding inspected authoritative source and card metadata; it did not
treat a credential or metadata lookup as proof of exact-revision payload access.
NPA adds no `ACCEPT_*` environment variable, EULA checkbox, telemetry, or
privacy consent. If an upstream endpoint later presents an actual click-through
or paid entitlement, it blocks only that runtime fetch; record the exact
upstream clause and use the operator's documented acceptance path. It does not
authorize inventing an NPA-wide gate.

The pinned OFT package still declares an unlicensed, moving `dlimp_openvla`
dependency. NPA does not fetch it. A public blob-tree comparison at the pinned
commits established that Apache-2.0 `kvablack/dlimp` differs only by retaining
its `LICENSE` and by the fork's deterministic default. NPA installs the licensed
parent at the immutable revision above, makes that one documented
`deterministic=True` change in the operator cache, and records its before/after
hashes and installed runtime probe. This is a narrow compatibility modification,
not a claim of rights to the unlicensed fork.

Outputs are operator-owned run artifacts. Their manifests record upstream
revisions, the resolved runtime dependency inventory hash, component hashes,
source data inventory, command, and measured simulator results; they do not
grant rights to redistribute any upstream input.
