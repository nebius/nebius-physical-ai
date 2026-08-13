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
build-your-own for exactly the reason `cosmos3-serving` is: a derived-container
grant whose conditions anonymous distribution does not establish. Baking nothing
makes the question moot instead of arguable, which is the stronger position and
the one the Isaac re-architecture already set as this repo's precedent. Prefer
arguments that survive the vendor changing their mind.

So `redistribution: public` here is a claim about **our** image only. It is not a
claim that LTX-2.5 is redistributable, and this image gives no one any right to
LTX-2.5: Lightricks delivers source and weights to each operator directly, under
that operator's own acceptance.

## The obligations a container cannot discharge

Two survive into the operator's own conduct, so the image refuses to run until
the operator has answered for them (`ltx_gate.py`, backed by
`npa/src/npa/workbench/ltx2/licensing.py`):

- **Section 2.1** — an Entity at or above **$10,000,000** annual revenue,
  aggregated across all affiliates under common Control (Section 1.6), needs a
  paid Commercial Use Agreement for any use outside the Section 2.2 carve-out.
  Nebius cannot know an operator's revenue, and must not assume the permissive
  answer. Section 2.2 does allow a Commercial Entity to use LTX-2.5 for
  "testing, evaluation, or non-commercial research and development in a
  non-production or development environment" without a paid licence — which is
  what a dev-VM evaluation is, and it is the declaration such a run must make.
- **Attachment A(18)** — "For commercial use only: To train, improve, or
  fine-tune any other machine learning model, artificial intelligence system, or
  competing model", reinforced by Section 2.2(c). **In a physical-AI workbench
  this is the load-bearing restriction**, because the obvious reason to generate
  synthetic video here is to train a robot policy, and a robot policy is another
  machine learning model. Under a commercial declaration that is prohibited. The
  disposition is computed per run, stamped into
  `ltx_provenance.json` next to the artifacts, and re-checked fail-closed by
  `npa workbench ltx2 gate` before any trainer may consume them.

Note the asymmetry that makes Attachment A(18) easy to get wrong: it is scoped by
*use*, not by entity size. A company under the revenue threshold owes no licence
fee and is still barred from training other models on Outputs commercially.

Section 6 and Attachment A(5)/(19) add output-side duties — disclose
machine-generated content, and do not strip provenance, watermarking, or latent
disclosure. The provenance manifest records these as `output_obligations` so they
travel with the artifacts instead of living only here.

## Verification

The claim is about bytes in layers, so it is checked against layers, not against
this file or the Dockerfile:

```bash
npa/.venv/bin/python npa/scripts/scan_image_ltx_payload.py <registry>/npa-ltx2:<tag>
```

The build itself asserts the refusal (exit 78) in two directions and that no
cache was written; `smoke.sh` repeats that against the pushed image and adds the
weight-fetch refusals. Passing automation proves byte absence and refusal
behaviour — not human legal approval.

## Status

**Not yet built or GPU-validated.** No accepted-image manifest exists, no tag has
been published, and `ltx2` is absent from the Blackwell/GPU compatibility
manifests on purpose: those record physical evidence, and inventing entries for
an unbuilt image would be the kind of unearned claim those files exist to
prevent. `docs/workbench/ltx2.md` carries the dev-VM runbook that produces the
evidence, and `npa/tests/e2e/test_ltx2_live_e2e.py` is the gated live check.

One consequence of delegating installation to upstream's own `uv sync` is that
the transitive dependency closure is **not** hash-locked the way
`wan2-2/runtime-requirements.txt` is; it is pinned only by the immutable source
ref and upstream's own constraints. That is a deliberate trade — a lockfile we
invented and could not resolve or verify here would be fiction — and the
bootstrap captures the real resolved closure to
`npa_resolved_inventory.txt` in the cache on first sync. Hardening that captured
resolution into a checked-in hash lock is the follow-up once a real build exists.

The container starts as UID 1000 with the runtime and model caches owned by that
user. As elsewhere in this workbench that is an ownership boundary, not a
security sandbox: passwordless sudo is retained for the shared SkyPilot
Kubernetes bootstrap. Operators needing a privilege boundary must enforce it via
pod security context outside the image.
