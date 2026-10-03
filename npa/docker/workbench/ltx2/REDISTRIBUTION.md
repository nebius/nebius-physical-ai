# npa-ltx2 container redistribution record

Review date: 2026-08-13. Engineering classification, not a legal opinion or a
substitute for counsel.

## What the vendor says versus what the licence says

LTX's own product pages describe LTX-2.5 as "fully open source", and one of them
names Apache-2.0 outright. It is not. The licence file in the source repository
(`LICENSE.md`) is the **LTX-2.x Community License Agreement, dated 2026-08-11**,
which GitHub classifies as `NOASSERTION`. It is not an OSI licence: it has a
revenue threshold, twenty use-based restrictions, and conditions on onward
transfer.

It also *supersedes* an earlier text. The LTX-2 Community License Agreement of
2026-01-05 is the version most search results and third-party write-ups still
quote. LTX-2.5 shipped on 2026-08-11 with a re-issued agreement, and the two
differ in ways that matter here — notably the Section 2.2 non-commercial
carve-out and the Section 6 AI-regulation and provenance obligations, neither of
which exists in the January text. Read the dated file in the repo at the pinned
ref, not a summary.

The published Python distributions do not resolve this either way: `ltx-core`
and `ltx-pipelines` on PyPI carry **no licence metadata at all** (`license:
None`, no classifiers, no license files). Silence is not permission, so the
repository's `LICENSE.md` governs.

## Three separately classified layers

1. **Source.** Section 1.9 defines "LTX-2.x" to include "machine-learning model
   code, inference-enabling code, training-enabling code, fine-tuning enabling
   code, accompanying source code, scripts". The `ltx-core` / `ltx-pipelines` /
   `ltx-trainer` packages are therefore licensed material, not an
   Apache-2.0-style wrapper around gated weights. **This is the trap specific to
   LTX**: on most model onboardings the code is permissive and only the weights
   are gated, so the habitual "bake the code, runtime-fetch the weights" split
   silently fails here.
2. **Baked runtime.** Nothing of LTX's is baked. The image carries the
   digest-pinned Debian/Python base, `uv`, `huggingface_hub`, ffmpeg, and our own
   scripts. No LTX code, no weights, no CUDA/`nvidia-*` wheels.
3. **Weights and data.** `Lightricks/LTX-2.5` on Hugging Face is a **gated**
   repository: access requires signing in, sharing contact information, and
   accepting the terms on the model page, with a token scoped to read gated
   repos. Never baked; fetched at run time under the operator's own entitlement.

## Why the image bakes nothing rather than baking the source

Section 3 does permit redistribution, but on conditions a public registry cannot
satisfy:

- **3.1** the use-based restrictions "MUST be included as an enforceable
  provision by you in any type of legal agreement (e.g. a license) governing the
  use and/or distribution", with notice to subsequent users. An anonymous
  `docker pull` forms no agreement, and a text file inside the image is not an
  enforceable provision against the puller.
- **3.2** every third-party recipient must be given a complete copy of the
  agreement.
- **3.5** a Derivative may not be transferred to a Commercial Entity **unless
  that entity has already obtained a paid licence**, and the transferor must
  notify it in writing beforehand. There is no way to establish either fact about
  an anonymous puller.

Had we baked `ltx-core`, this image would have been `restricted` /
build-your-own for the same reason the retired Cosmos3-serving parent was: a
derived-container grant whose conditions anonymous distribution does not establish. Baking nothing
makes the question moot instead of arguable, which is the stronger position and
the one the Isaac re-architecture already set as this repo's precedent. Prefer
arguments that survive the vendor changing their mind.

So `redistribution: public` here is a claim about **our** image only. It is not a
claim that LTX-2.5 is redistributable, and this image gives no one any right to
LTX-2.5: Lightricks delivers source and weights to each operator directly, under
that operator's own acceptance.

## The obligations that stay with the operator

Two survive into the operator's own conduct, and neither is something this
image can discharge, verify, or record on their behalf:

- **Section 2.1** — an Entity at or above **$10,000,000** annual revenue,
  aggregated across all affiliates under common Control (Section 1.6), needs a
  paid Commercial Use Agreement for any use outside the Section 2.2 carve-out
  ("testing, evaluation, or non-commercial research and development in a
  non-production or development environment"). Nebius cannot know an operator's
  revenue, ships no LTX bytes, and is not a distributor here, so it does not ask
  customers to self-certify it. `ltx-runtime terms` states the threshold and the
  vendor contact; the decision is the operator's.
- **Attachment A(18)** — "For commercial use only: To train, improve, or
  fine-tune any other machine learning model, artificial intelligence system, or
  competing model", reinforced by Section 2.2(c). **In a physical-AI workbench
  this is the load-bearing restriction**, because the obvious reason to generate
  synthetic video here is to train a robot policy, and a robot policy is another
  machine learning model. Complying with it is the operator's responsibility:
  the restriction governs Outputs sitting in their own bucket, which no
  packaging choice of ours reaches.

Note the asymmetry that makes Attachment A(18) easy to get wrong: it is scoped by
*use*, not by entity size. A company under the revenue threshold owes no licence
fee and is still barred from training other models on Outputs commercially.

Section 6 and Attachment A(5)/(19) add output-side duties — disclose
machine-generated content, and do not strip provenance, watermarking, or latent
disclosure. They attach to artifacts we never touch, so they are stated here and
in `docs/workbench/ltx2.md` rather than enforced.

## What the image does check

Acceptance of the agreement is not a variable we can meaningfully hold. The
agreement forms by conduct — "By downloading, using, accessing or distributing
any portion or element of LTX-2.x, you agree that you have read and accepted to
be bound by this Agreement" — so a local `ACCEPT=YES` never formed it. What *is*
checkable is the entitlement Lightricks grants: `Lightricks/LTX-2.5` is gated,
and access follows a human accepting the terms on that page. An anonymous `HEAD`
on a weight file returns 401; with an entitled token it returns 302.

So the container requires `HF_TOKEN` for **both** fetches — the weights and, per
Section 1.9, the source — and refuses with exit 78 without it. NVIDIA's CUDA
terms remain a separate vendor decision under
`NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS`. Neither value is ever baked.

## Verification

The claim is about bytes in layers, so it is checked against layers, not against
this file or the Dockerfile:

```bash
npa/.venv/bin/python npa/scripts/scan_image_ltx_payload.py <registry>/npa-ltx2:<tag>
```

The build itself asserts the refusal (exit 78) on the source fetch, the weight
fetch, and NVIDIA's runtime terms, asserts *which* gate refused in each case,
and asserts that no cache was written; `smoke.sh` repeats that against the
pushed image. Passing automation proves byte absence and refusal behaviour — not
human legal approval.

## Status

**Built, byte-scanned, GPU-validated, and released.** The exact zero-payload
digest recorded in `npa/src/npa/deploy/ltx2_image_manifest.json` completed a
real text-to-video generation and decoded-MP4 validation on one RTX PRO 6000.
The accepted manifest also binds the source/weight revisions and zero-finding
payload scan. `blackwell-dc-images.json` records the image as CPU-only at rest:
the GPU runtime is fetched into operator-owned storage, so that physical run
does not assert an in-image B200/B300 architecture. The shared submit matrix is
still plan-only because its generic credential presence check cannot establish
that a given operator's `HF_TOKEN` has the gated Lightricks entitlement;
`npa/tests/e2e/test_ltx2_live_e2e.py` owns that entitled live path.

One consequence of delegating installation to upstream's own `uv sync` is that
the transitive dependency closure is **not** hash-locked the way
`wan2-2/runtime-requirements.txt` is; it is pinned only by the immutable source
ref and upstream's own constraints. That is a deliberate trade — a lockfile we
invented and could not resolve or verify here would be fiction — and the
bootstrap captures the real resolved closure to
`npa_resolved_inventory.txt` in the cache on first sync. Hardening that captured
resolution into a checked-in hash lock remains follow-up work.

The container starts as UID 1000 with the runtime and model caches owned by that
user. As elsewhere in this workbench that is an ownership boundary, not a
security sandbox: passwordless sudo is retained for the shared SkyPilot
Kubernetes bootstrap. Operators needing a privilege boundary must enforce it via
pod security context outside the image.
