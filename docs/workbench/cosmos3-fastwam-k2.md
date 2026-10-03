# Cosmos3 Edge FastWAM-K2 closed-loop screening

[Workbench](README.md) · [workflow](../../workflows/testing/cosmos3-fastwam-k2-eval.yaml)

This experimental workflow evaluates the supplied Cosmos3 Edge FastWAM-K2
checkpoint against its full-WAM base with matched **closed-loop RoboLab task
success** and the native policy-inference timing records. It does not establish
a benchmark result, model quality claim, training result, or physical-robot
success.

## Contract retained from the model card

The derivative checkpoint is the published evaluation snapshot
`geonmin-kim/Cosmos3-Edge-Policy-DROID-FastWAM-K2` at
[`04cc10f6f790153fa5db1ff90e95ecf9196e88c5`](https://huggingface.co/geonmin-kim/Cosmos3-Edge-Policy-DROID-FastWAM-K2/tree/04cc10f6f790153fa5db1ff90e95ecf9196e88c5),
using `step12000/`. The card identifies it as an EMA export with no training
state, warm-started from `nvidia/Cosmos3-Edge-Policy-DROID`; it is not relabelled
as original NPA research.

The only FastWAM mode this workflow permits is:

```text
--format-prompt-as-json True --no-guardrails --keep-generated-vision-frames 2
```

It never substitutes `--drop-generated-vision` (K=0). The K=2 contract keeps
one conditioning latent plus two leading generated vision latents. The first is
conditioned; the two retained generated frames remain in the denoising target.
The recorded expected packed evidence is 1,020 vision tokens and 680 vision
MSE-target tokens (340 per latent frame). The reported runtime overlay retains
the exact before/after hashes of the two upstream files it changes and fails if
its source anchors change.

The model-card export manifest pins the framework to
[`4e26181d87878a0b14c37ca021b0e2cd4f28dc5f`](https://github.com/NVIDIA/cosmos-framework/commit/4e26181d87878a0b14c37ca021b0e2cd4f28dc5f).
That revision does not expose `--keep-generated-vision-frames`; it cannot serve
the documented K=2 contract unmodified. The NPA runtime overlay is therefore a
disclosed compatibility modification, not an upstream flag or a K=0 fallback.
The full-WAM arm uses the base checkpoint revision
`68b17b3c959ccd0999de9a972c5f7c8b57112f86` without that overlay.

## Five connected real stages

1. `prepare` fetches the pinned RoboLab source, locates the selected task class
   definitions, hashes them, and publishes a matched input manifest.
2. `full_wam` starts the upstream RoboLab policy server with the baseline,
   drives native RoboLab episodes, and publishes result rows, MP4s, and timing.
3. `fastwam_k2` consumes both matched preparation and baseline evidence, applies
   the hash-bound K=2 overlay, and runs the same native simulator path.
4. `compare` consumes both completed arms and reports paired task-success and
   actual native policy latency; it excludes open-loop error as a decision
   signal.
5. `visualize` consumes those exact artifacts and emits a factual run-identified
   Rerun `.rrd` plus copied rollout MP4s; the prepared-input hash remains in
   the report as the comparison-provenance identity.

The card calls the two named tasks a discriminative screen, not a benchmark.
The supplied request uses `RubiksCubesInBinTask` and `StackYellowOnRedTask` with
RoboLab's adaptive sampling (`--num-episodes-adaptive 200 --ci-pp-width 0.14`).
A positive screen still requires a separately defined full-suite evaluation.

## Attribution, terms, and delivery

| Boundary | Upstream identity and credit | Terms / delivery decision |
| --- | --- | --- |
| K=2 checkpoint | Publisher: [geonmin-kim](https://huggingface.co/geonmin-kim); model card claims NVIDIA copyright and identifies the NVIDIA Edge base | The card metadata links the NVIDIA Open Model License while the shipped card text includes OpenMDW-1.1. This discrepancy is recorded rather than resolved by NPA. Fetch at runtime at the exact revision; no checkpoint, adapter, or cache is put in an image or committed artifact. |
| Cosmos Framework | [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework/tree/4e26181d87878a0b14c37ca021b0e2cd4f28dc5f), NVIDIA copyright/NOTICE and third-party attributions retained | `LICENSE` is OpenMDW-1.1. Source is fetched at runtime; the overlay's modification and hashes are recorded with the run. |
| RoboLab | [NVlabs/RoboLab](https://github.com/NVlabs/RoboLab/tree/ad45d4f974725d020f82c2b0d77d78533aeba2b3), authored by Xuning Yang and contributors | Apache-2.0 with upstream `THIRD_PARTY_NOTICES.md`. Source and Isaac runtime remain runtime-only. |
| Isaac Sim / Isaac Lab runtime | NVIDIA product runtime consumed by RoboLab's documented `uv sync --extra isaac50` path | No Isaac bytes or acceptance variable are baked into the NPA image or workflow. The upstream README identifies `OMNI_KIT_ACCEPT_EULA=Y` as its documented first-use mechanism outside tests; no new NPA checkbox, `ACCEPT_*` flag, or duplicate attestation is created here. A run may use only an existing operator-authorized product mechanism. |
| Output | Native RoboLab rows, MP4s, RRD, and NPA provenance | Output remains run-scoped. It carries upstream identities and is not treated as a redistribution grant for checkpoints, source, or simulator assets. |

The workflow routes through the NPA Cosmos image family with a source overlay.
The currently configured public Cosmos release is quarantined by the repository
stale-layer policy, so a live submission must use a freshly built,
source-matched immutable replacement after its local and registry gates pass.
This change does not publish an image or make an OCI-metadata claim. A live run
must record the exact resolved image digest and inspect the produced artifacts
before any live-ready claim.

The opt-in live-submit case seeds only the documented evaluation protocol, then
requires the five native stages to publish every result. Its independent
read-back verifies every declared artifact hash, finite native latency and task
success values, the K=2 serving record, decoded RRD application/run identity,
and an MP4 from each policy arm. It requires an explicit source-matched image
digest; it never falls back to the quarantined release.

RoboLab's upstream citation is retained below. Neither the model card nor the
pinned framework README supplied a separate BibTeX entry in the inspected
revisions.

```bibtex
@inproceedings{yang2026robolab,
    author    = {Xuning Yang and Rishit Dagli and Alex Zook and Hugo Hadfield and Ankit Goyal and Stan Birchfield and Fabio Ramos and Jonathan Tremblay},
    title     = {{RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies}},
    booktitle = {Proceedings of Robotics: Science and Systems},
    year      = {2026},
    address   = {Sydney, Australia},
    month     = {July},
    url       = {https://arxiv.org/abs/2604.09860}
}
```

## Readiness boundary

Local validation, specification planning, and artifact-contract tests verify
only the implementation. Before this workflow is accepted, an operator-owned
run must complete on an RT-core Kubernetes target, use the resolved immutable
image, materialize the exact runtime source/checkpoints, complete both
closed-loop arms, and have its final MP4/RRD/report independently read back.
Unavailable capacity, missing exact upstream access, or absent existing Isaac
product authorization leaves that target explicitly unverified; it is not a
reason to replace the K=2 contract with K=0 or to report a proxy benchmark.
