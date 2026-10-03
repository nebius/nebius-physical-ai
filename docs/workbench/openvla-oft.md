# OpenVLA-OFT LIBERO runtime

This is an integration of upstream work, not original NPA research. It executes
the official OFT continuous-action recipe and preserves the source, component,
and measured-artifact lineage needed to distinguish it from stock OpenVLA.

## Runtime contract

`workflows/testing/openvla-oft-libero.yaml` has five connected substantive
stages:

1. `prepare` reads supplied RLDS TFRecord bytes and records the real data
   inventory plus normalization metadata.
2. `train`, once its exact upstream dependency licensing is resolved, materializes
   the exact base-model Hub revision in the writable cache, then invokes the
   pinned upstream `vla-scripts/finetune.py` through upstream-style single-node
   eight-process `torchrun`, with its continuous L1 action head, 8-D
   proprioception projector, two images, and LoRA rank 32.
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
recipe and 50 trials per task. It uses H100 for both training and evaluation:
the pinned PyTorch 2.2.0 predates B200, while upstream reports A100 execution
and asks that training and evaluation use the same GPU product. This is a
compatibility selection, not a paper-reproduction claim. It makes no
convergence, full-benchmark, or physical-robot claim. A deliberately small
operational smoke must be identified as such, not presented as those results.

## Upstream lineage and credit

| Layer | Identity and credit | How NPA handles it |
| --- | --- | --- |
| OFT source | [moojink/openvla-oft](https://github.com/moojink/openvla-oft), commit `e4287e94541f459edc4feabc4e181f537cd569a8`; MIT; © 2025 Moo Jin Kim, Chelsea Finn, Percy Liang | Reserved for an operator-owned runtime cache after the direct-dependency blocker below is resolved. |
| OFT method | Kim, Finn, Liang, “Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success,” arXiv:2502.19645 (2025) | Cite this work when using OFT results. |
| Base model/source | [OpenVLA](https://github.com/openvla/openvla) commit `c8f03f48af692657d3060c19588038c7220e9af9`; code MIT; © 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti | Base `openvla/openvla-7b` resolves at runtime only. |
| Base weights | [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b), Hub revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | The current model card says MIT, but upstream OpenVLA’s own README says Llama-2-derived pretrained models are subject to the [Llama Community License](https://ai.meta.com/llama/license/). Treat that source-level statement as controlling for the base weight; do not redistribute it in an NPA image. |
| LIBERO simulator runtime | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`; MIT; © 2023 Lifelong Robot Learning | Fetched and installed only into the operator-owned runtime cache, following the OFT LIBERO guide. Its source, simulator dependencies, and assets are not image bytes. |
| OFT direct runtime dependencies | [custom Transformers fork](https://github.com/moojink/transformers-openvla-oft), Apache-2.0; plus `moojink/dlimp_openvla`, an unversioned direct source dependency in the pinned OFT `pyproject.toml` | The custom Transformers fork is runtime-only. The pipeline refuses before network/runtime mutation because `dlimp_openvla` has no declared repository license or `LICENSE` file and the upstream declaration has no immutable revision. The smallest resolution is an authoritative upstream license/permission for one exact revision; then the normal ready-marker inventory is available. |
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

Separately, the current OFT dependency declaration for `dlimp_openvla` provides
no repository license and does not pin a revision. That is a concrete upstream
term unknown, not a request for an NPA EULA: the pipeline fails closed before
it fetches or installs that dependency. The resolution is an upstream license or
permission for one exact revision; all source-independent work remains valid.

Outputs are operator-owned run artifacts. Their manifests record upstream
revisions, the resolved runtime dependency inventory hash, component hashes,
source data inventory, command, and measured simulator results; they do not
grant rights to redistribute any upstream input.
