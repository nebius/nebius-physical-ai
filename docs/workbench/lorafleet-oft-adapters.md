# LoRAFleet reconstructed OpenVLA-OFT adapters

This is an operator-built BYOF qualification path for LoRAFleet's public
`openvla-oft-libero-reconstructed-r64` release. It reconstructs four separate
LIBERO policies (`spatial`, `object`, `goal`, and `10`) from an immutable
OpenVLA base plus LoRAFleet's factors. It is not an NPA model release, a public
image, a common-base serving system, or original NPA research.

## Pinned sources and credit

| Component | Immutable source | Author / terms | How this integration uses it |
| --- | --- | --- | --- |
| Reconstruction release | [`LoRAFleet/openvla-oft-libero-reconstructed-r64@0b75d5b`](https://huggingface.co/LoRAFleet/openvla-oft-libero-reconstructed-r64/tree/0b75d5b19ba43ac3efa870bdf830192e98cb4d4e) | LoRAFleet; model card declares MIT | Runtime-fetched rank-64 factors, suite action/proprio heads, statistics, checksum inventory, and reconstruction diagnostics |
| Base VLA | [`openvla/openvla-7b@47a0ec7`](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f) | OpenVLA authors; MIT as declared by the model card | Runtime-fetched, complete-byte-verified base operand only |
| OFT runtime | [`moojink/openvla-oft@e4287e9`](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8) | Moo Jin Kim, Chelsea Finn, and Percy Liang; [MIT](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/LICENSE) | Pinned runtime and its native LIBERO policy/evaluation components |
| OFT data loader | [`kvablack/dlimp@92e3eca`](https://github.com/kvablack/dlimp/tree/92e3eca97af3b14d0b6aa15182c0dc240407698d) | Kevin Black; [Apache-2.0](https://github.com/kvablack/dlimp/blob/92e3eca97af3b14d0b6aa15182c0dc240407698d/LICENSE) | Licensed parent used in place of the unlicensed historical fork; its LICENSE and a modification notice remain in the private image |
| OFT transformer runtime | [`moojink/transformers-openvla-oft@bc339d9`](https://github.com/moojink/transformers-openvla-oft/tree/bc339d9ad707454c0c115970db43c260067c61ab) | Moo Jin Kim and Hugging Face Transformers contributors; [Apache-2.0](https://github.com/moojink/transformers-openvla-oft/blob/bc339d9ad707454c0c115970db43c260067c61ab/LICENSE) | Immutable bidirectional-attention runtime dependency, installed by the private image recipe |
| Original merged policy baselines | four revision-pinned repos recorded in the release [`manifest.json`](https://huggingface.co/LoRAFleet/openvla-oft-libero-reconstructed-r64/blob/0b75d5b19ba43ac3efa870bdf830192e98cb4d4e/manifest.json) | OpenVLA-OFT / moojink; cards declare MIT | Runtime-fetched only for the published-policy baseline stage |
| Simulator | [`Lifelong-Robot-Learning/LIBERO@8f1084e`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) | LIBERO contributors; [MIT](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/LICENSE) | Pinned simulation runtime for upstream-native rollouts |

The source repository's notice is preserved by the operator-built image and
this integration retains the source revision, release/model revisions,
publisher checksums, source checkpoint revisions, local modifications, and
runtime image digest in stage artifacts. The OpenVLA-OFT citation is:

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645}, year={2025}
}
```

Please credit OpenVLA, OpenVLA-OFT, LoRAFleet, and LIBERO when using any
artifacts produced from this workflow. The release attributes its randomized
SVD/FlashTSQR reconstruction implementation to LoRAFleet; this repository does
not claim to have produced that method or the source checkpoints.

### Licensed dlimp replacement

The upstream OFT project names `moojink/dlimp_openvla` without an immutable
revision or LICENSE. This integration does **not** fetch or redistribute that
fork. It uses the verified Apache-2.0 parent
`kvablack/dlimp@92e3eca97af3b14d0b6aa15182c0dc240407698d` instead. A recursive
blob-tree comparison established that the only code difference is
`dlimp/dataset.py`: the fork sets `options.deterministic = True`, while the
licensed parent sets it to `False`. The private recipe makes only that same
one-line change, retains the unmodified Apache `LICENSE`, writes
`NPA_MODIFICATIONS.md`, and records before/after source hashes in
`npa_lorafleet_dlimp_provenance.json`. The first workflow stage rejects a
missing, substituted, or non-deterministic loader before downloading model
payloads. It then runs a delayed multi-worker map and creates a two-step local
RLDS fixture that the real OFT reader consumes, proving deterministic ordering
and the replacement loader's actual OFT data path without fetching the
separately governed training dataset.

[`npa/scripts/build_lorafleet_oft_adapters.sh`](../../npa/scripts/build_lorafleet_oft_adapters.sh)
is the canonical `--build-command` content for the generic BYOF builder. It
installs OFT with `--no-deps`, so the historical fork cannot be resolved
transitively, then pins both licensed replacement sources. The resulting image
is operator-private; no public registry upload or public image claim is made.

## Reconstruction and behavioral boundary

The release marks these as approximate reconstructions, not original training
adapters. For all 439 target modules the executable stage requires:

```text
W_target = W_verified_base + FP32(B @ A)
```

where `B = U*S`, `A = Vh`, and `r = lora_alpha = 64`. It first validates the
release's `non_lora_exact` records, all 543 non-target tensors' audit records,
and the three base-shard hashes. It then **assigns** each reconstructed target
from a base tensor; it never adds factors to an already merged source target.
Each suite uses the release's matching action head, proprio projector, and
`dataset_statistics.json`. A stock OpenVLA action decoder is not substituted.

The publisher evaluated only task 0 with eight initial states per suite. That
smoke is retained as provenance, but it is not used as equivalence evidence.
The workflow instead requests every task in all four LIBERO suites and the
upstream default 50 initial states per task (2,000 simulator episodes for each
policy flavor). Its comparison emits empirical outcome counts for the exact
paired simulator states. It makes no claim about adapter equivalence, action
distribution equality, benchmark superiority, training convergence,
shared-base/multi-adapter serving, or physical-robot performance.

## Five real connected stages

[`workflows/testing/lorafleet-oft-adapters.yaml`](../../workflows/testing/lorafleet-oft-adapters.yaml)
has five connected GPU stages. Each output is a run-scoped S3 object, and the
next stage reads the exact URI rather than reconstructing provenance from a
name alone.

1. `verify-inputs` downloads and hashes public release/base/source metadata,
   validates immutable revisions and every suite's factors, heads, and stats.
2. `reconstruct-weights` consumes that verification artifact, byte-verifies the
   base weights, and performs all four base-plus-FP32-factor reconstructions.
3. `published-baseline-rollouts` consumes both earlier artifacts and runs the
   pinned original merged OFT checkpoints with upstream LIBERO components.
4. `reconstructed-adapter-rollouts` consumes the baseline's explicit task and
   initial-state protocol, then runs reconstructed policies on those identities.
5. `compare-and-visualize` consumes the two rollout reports, checks identity
   pairing, produces observed metrics, a validated `.rrd`, and inventories
   representative actual MP4s from each policy flavor.

The committed image value is an intentionally non-routable digest sentinel.
An operator-built, private immutable image digest must be supplied at submit
time through NPA configuration/overrides after local image inspection. The
ad-hoc BYOF image is not registered as an NPA container/catalog image and must
not be publicly published merely because this workflow is present.

## Terms and redistribution decision

The checked public source and model cards above declare MIT. This integration
does not add an NPA EULA, acceptance environment variable, checkbox, or
per-image attestation: no reviewed source required one for these public inputs.
Existing operator credentials are used only by normal runtime fetch and S3
transport; they are not evidence of broader redistribution rights.

The OpenVLA/OFT runtime's optional `openvla/modified_libero_rlds` training data
was independently access-controlled during onboarding and is neither fetched
nor needed for these evaluation stages. Its terms and redistribution decision
remain separate and unverified; that does not block the public model/simulator
qualification path. Runtime package, CUDA base-image, and cache terms remain
their respective upstream terms. The operator image is private and unpromoted
until a separate exact-byte licensing and publication review is complete.

Model factors, base weights, original merged checkpoints, task heads, simulator
assets, cache contents, and outputs are runtime fetched or generated under the
operator's scoped run. They are not copied into committed files or a public
NPA image. Generated evaluation artifacts retain source/revision/checksum
provenance but do not themselves widen any upstream redistribution grant.

## Validate, submit, and inspect

Use the isolated repository environment:

```bash
npa/.venv/bin/npa workbench workflow validate-spec workflows/testing/lorafleet-oft-adapters.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec workflows/testing/lorafleet-oft-adapters.yaml --json
```

Before a live submit, run `npa workbench health preflight --checks nebius` and
build/inspect the private BYOF image using the pinned OFT revision. The live
submission must override only the image sentinel with the private immutable
digest through the configured NPA surface, then independently download the
final `comparison.json`, `comparison.rrd`, and retained MP4s for decoding and
provenance inspection. A scheduler or capacity failure is a live-validation
blocker, not a reason to promote the result to accepted.
