# OpenWAM-alpha foundation → LIBERO policy workflow

This is an operator-private, **unvalidated** OpenWAM candidate. It implements
only the upstream OpenWAM-alpha → LIBERO path at the pinned revisions below. It
does not claim a full LIBERO aggregate, training convergence, paper-score
reproduction, physical-robot performance, or a public NPA image release.

## Supported architecture and boundary

The workflow invokes OpenWAM's native `scripts/train.sh`, `scripts/deploy.py`,
and `benchmarks/libero/single_eval.py` entrypoints. Its accepted architecture
contract is exactly:

| Item | Pinned supported value |
| --- | --- |
| OpenWAM source | [`OpenWAM-Official/OpenWAM`](https://github.com/OpenWAM-Official/OpenWAM) `48bd67b89d489b14d03b8d92bc66e65d306df32e` |
| Foundation | [`OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model`](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model/tree/52df4e66c82c5c8b480adcc8d01f4db7415dfb56) `52df4e66c82c5c8b480adcc8d01f4db7415dfb56` |
| Architecture | `dual_system` + `joint_self_attn`, mutual attention mask |
| Video backbone | [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e) `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` |
| Benchmark client | [`Lifelong-Robot-Learning/LIBERO`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Training-format data mirror | [`OpenWAM/LIBERO`](https://huggingface.co/datasets/OpenWAM/LIBERO/tree/bcb2eaf1121ae4cbd324f8862807abce282500e8) `bcb2eaf1121ae4cbd324f8862807abce282500e8` |

The checked-in workflow uses OpenWAM's `training.debug=true`, which upstream
defines as a 20-step operational smoke with checkpoint saves. It is useful to
prove the real model/data/trainer path and must never be represented as a
converged run. Full OpenWAM training uses the operator's upstream configuration
and appropriate GPU topology; it is not inferred from a passing smoke.

## Five connected executable stages

[`workflows/testing/openwam-libero-four-stage.yaml`](../../workflows/testing/openwam-libero-four-stage.yaml)
passes the exact artifact handoffs below through run-scoped S3-style URIs:

1. `prepare-assets` downloads the immutable foundation, Wan backbone, LIBERO
   training-format dataset, and LIBERO source; verifies the foundation checkpoint
   shape and archives source/license provenance.
2. `fine-tune-openwam` restores that archive and invokes OpenWAM's native
   LIBERO fine-tuner, retaining an archive of the actual checkpoint, config, and
   normalization statistics.
3. `deployed-policy-rollout` starts the upstream policy server from that exact
   trained archive and executes one native-action LIBERO spatial-suite episode.
4. `evaluate-heldout-episode` serves the same checkpoint anew and runs the next
   numbered initial-state episode. Its result is separate numerical evidence,
   not a benchmark aggregate.
5. `emit-factual-rrd` reads the three real reports, emits a Rerun `.rrd`, and
   verifies the decoded held-out success-rate entity before upload.

Every final report records the run ID, declared/observed image identity,
checkpoint hashes, source revisions, and input artifact URIs. Model/data/cache
artifacts remain in run-owned private storage; no stage fabricates scores,
videos, or RRD metrics.

## Attribution, terms, and redistribution

OpenWAM source and the foundation-model card are Apache-2.0. The pinned source
contains `LICENSE` and the complete `CITATION.cff` author list, but no separate
`NOTICE` file; preserve that exact upstream record and cite
OpenWAM Official Team, *OpenWAM: An Open, Modular Exploration Towards
Systematic World-Action Model Pretraining*, arXiv:2609.07398 (2026). The model
inherits the Wan 2.2 component: credit Wan-AI and preserve the Wan model-card
citation and Apache-2.0 notice with checkpoint provenance.

LIBERO code is MIT. The original LIBERO benchmark data is CC-BY-4.0; the
OpenWAM-format mirror's card label does not replace attribution to the original
LIBERO authors or its data license. The image's
[`THIRD_PARTY_NOTICES.md`](../../npa/docker/workbench/openwam/THIRD_PARTY_NOTICES.md)
and every prepared-assets manifest keep these identities separate.

The foundation, Wan backbone, and dataset cards were publicly accessible at the
listed immutable revisions during authoring. No documented click-through,
additional NPA EULA, `ACCEPT_*` environment variable, or per-image attestation
was necessary for this path, so none is added. Access and Apache labels still
do not settle redistribution of a complete built image, runtime dependencies,
data, trained checkpoints, or outputs. The
[`REDISTRIBUTION.md`](../../npa/docker/workbench/openwam/REDISTRIBUTION.md)
therefore classifies the recipe `unvalidated` and operator-private pending a
complete GPU workflow qualification and a separate public-redistribution
decision.

## Image and execution readiness

For a new candidate, build from
[`npa/docker/workbench/openwam/Dockerfile`](../../npa/docker/workbench/openwam/Dockerfile)
only into an operator-controlled registry, scan the exact digest, then replace
the workflow's placeholder `runtime_image` with that immutable digest. The
image carries two deliberately separate Python environments: OpenWAM's pinned
CUDA environment hosts training and serving; the upstream-documented Python
3.10 LIBERO client environment handles MuJoCo evaluation over the policy
WebSocket. The runtime fetches all model/data payloads and does not turn a
public checkpoint card into permission to publish derived files.

Before submit, run `npa workbench health preflight --checks nebius`, select an
owned writable bucket, prove the private image pull through
`npa workbench workflow preflight-images`, and submit the workflow. Retain the
workflow's readiness record alongside the result. Only a finished run with
independent artifact inspection may become a live-ready or accepted claim.
