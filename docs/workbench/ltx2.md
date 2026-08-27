# LTX-2.5 Workbench support

NPA packages Lightricks' LTX-2.5 video/audio model as a BYOF solution whose
image contains **no LTX-2.5 bytes at all** — no `ltx-core`, no `ltx-pipelines`,
no weights, no CUDA wheels. On first use the container fetches upstream's pinned
source and the gated weights under the operator's own credentials, and refuses
to fetch either without a Hugging Face token that has access to the gated
repository.

LTX-2.5 is a generative video model. This integration does not represent it as
an action-conditioned robotics simulator or an action-prediction model.

> **Status: exact public digest accepted after zero-payload and real GPU gates.**
> The immutable development digest recorded in
> `npa/src/npa/deploy/ltx2_image_manifest.json` passed every pre-publication
> check, then generated and independently decoded a real clip on one RTX PRO
> 6000 under the operator's own entitlement. The supported release tag is
> `2.5-rtfetch-20260817`; source and weights remain runtime-only.

## Why this image ships nothing

LTX-2.5 is **not** OSI open source, despite being described that way on
Lightricks' pages. The governing text is the **LTX-2.x Community License
Agreement** (2026-08-11), and its Section 1.9 folds "inference-enabling code,
training-enabling code … accompanying source code" into the licensed material.
That removes the split Wan 2.2 relies on, where Apache-licensed source is baked
and only the gated weights are fetched: for LTX, **the code is licensed material
too**.

Section 3 then puts three obligations on anyone who distributes it — impose the
use restrictions by contract on the recipient (3.1), deliver the full agreement
to them (3.2), and do not transfer to a Commercial Entity at all until it holds
a paid licence (3.5). An anonymous `docker pull` establishes none of those. So
we redistribute nothing of Lightricks', and Lightricks delivers to each operator
directly. `npa/docker/workbench/ltx2/REDISTRIBUTION.md` records the full
classification.

## Immutable upstream inputs

| Input | Revision | Packaging |
| --- | --- | --- |
| Official source | [`Lightricks/LTX-2` `fd4ded7…`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca) | fetched at run time with the operator's own `HF_TOKEN`; never baked |
| LTX-2.5 weights | [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) (gated) | fetched at run time with the operator's own `HF_TOKEN` |
| CUDA PyTorch stack | resolved by upstream's own `uv` pins from `download.pytorch.org/whl/cu132` | fetched at run time under separate NVIDIA acceptance |

`ltx_runtime.sh` verifies the fetched `HEAD` against the pinned ref and requires
upstream's `LICENSE.md` to be present before installing, because that ref is
what this documentation and the capability artifact claim.

## What the workbench requires: a token, not a declaration

**A valid `HF_TOKEN` with access to `Lightricks/LTX-2.5` is the only
workbench-side requirement.** Both fetches refuse with exit 78 without one —
the weights *and* the source, because Section 1.9 of the agreement folds
"inference-enabling code, training-enabling code … accompanying source code"
into the licensed material, so the GitHub repository is licensed material too.

Acceptance happens on Hugging Face, not here. `Lightricks/LTX-2.5` is a gated
repository (`gated: auto`): you sign in, share contact information, and accept
Lightricks' terms on that page, and Lightricks grants your account access. That
is checkable, and we checked it — an anonymous `HEAD` on a weight file returns
`401`, the same request with an entitled token returns `302`. A fine-grained
token needs the "read gated repos" scope.

**The licence binds by use.** Its opening line is "By downloading, using,
accessing or distributing any portion or element of LTX-2.x, you agree that you
have read and accepted to be bound by this Agreement." An earlier version of
this integration asked the operator to set
`NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES` plus an entity class and a use class.
Those are gone. The local variable never formed the contract — conduct did — and
the class answers were unverifiable self-certification that Nebius, which ships
zero LTX bytes and is not a distributor here, has no standing to collect. The
entitlement is the stronger evidence, and it is the vendor's own.

A separate `NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS=YES` covers NVIDIA's CUDA
runtime, which is a different vendor's decision. Neither it nor the token is
ever baked into the image.

Print what governs LTX-2.5 without holding anything:

```bash
npa/.venv/bin/npa workbench ltx2 terms
```

## Two obligations that remain yours

Compliance with the agreement is the operator's own responsibility. Two clauses
in particular cannot be discharged, verified, or recorded by a container, and
this workbench does not pretend otherwise:

- **Section 2.1** — an Entity whose annual revenue is at or above $10,000,000,
  counting all affiliates under common Control (Section 1.6), needs a **paid
  Commercial Use Agreement** for any use outside the Section 2.2 carve-out
  ("testing, evaluation, or non-commercial research and development in a
  non-production or development environment"). Contact
  `ltxv-licensing@lightricks.com`.
- **Attachment A(18)** — for commercial use, the Outputs may not be used "to
  train, improve, or fine-tune any other machine learning model, artificial
  intelligence system, or competing model". Section 2.2(c) says the same from
  the other side.

In a physical-AI workbench the second one is not an edge case. Generating
synthetic video to train a robot policy is the obvious reason to want this
model, and a robot policy is another machine learning model. No packaging choice
reaches it, because it governs artifacts in your own bucket — which is exactly
why it is your call rather than a value this pipeline computes for you. Note the
asymmetry: Attachment A(18) is scoped by *use*, not entity size, so a company
below the revenue threshold owes no licence fee and is still bound by it. Any
model trained on Outputs is a Derivative of LTX-2.x under Section 1.5 and stays
bound by the Agreement, including the Section 3.5 transfer conditions.

Section 6 and Attachment A(5)/(19) add output-side duties: disclose
machine-generated content, and do not strip provenance, watermarking, or latent
disclosure markers.

## The pipeline

`byof-ltx2.yaml` runs:

```text
generate (GPU, npa-ltx2) -> curate (CPU, FiftyOne)
```

- **generate** proves the refusal on this image first (`ltx-runtime
  assert-refusal`, which also asserts both caches are still empty), then fetches
  source and weights under your token, runs `python -m ltx_pipelines.distilled`
  at the pinned ref, and validates the decoded MP4.
- **curate** runs real FiftyOne Brain uniqueness and duplicate detection, and is
  the terminal state. Curation inspects Outputs without training on them.

**Curation is where the shipped pipeline stops, deliberately.** An earlier draft
ended in a LeRobot training state, which made it look like a complete
demonstration. It was not: that state trained on the `lerobot/pusht` hub dataset,
so no LTX Output ever reached it. LTX-2.5 text-to-video output is not a LeRobot
dataset — no actions, no `meta/info.json` — and no honest conversion exists here,
so training is your own next step, taken under your own reading of Attachment
A(18).

Validate and plan the checked-in spec:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-ltx2.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-ltx2.yaml --run-id ltx2-plan
```

## Dev VM runbook

All six steps were executed for the accepted digest. Repeat them for any future
candidate; an older proof never transfers to new image bytes.

**1. Build.** Requires Docker and a registry you control. The build itself
proves the refusal works and that proving it downloaded nothing; it fails if any
LTX, weight, or CUDA payload appears in any layer.

```bash
npa/docker/workbench/ltx2/build.sh --registry <your-registry> --push
```

**2. Scan the pushed bytes, not the Dockerfile.** The claim is about bytes in
layers, so check bytes in layers. The scanner is mutation-tested in both
directions by `npa/tests/docker/test_ltx_image_payload_scan.py`.

```bash
npa/.venv/bin/python npa/scripts/scan_image_ltx_payload.py \
  <your-registry>/npa-ltx2:dev-<full-git-sha>
```

Only the immutable public development tag is tested before release promotion.

**3. Re-prove the refusal against the pushed image.** This is the same mode the
build ran, now against the artifact anyone would pull.

```bash
docker run --rm <your-registry>/npa-ltx2@sha256:<digest> ltx-runtime assert-refusal
```

It must print `NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_ENTITLEMENT_OK`. Confirm the
unentitled refusal separately, and that it names the token rather than one of
the gates behind it:

```bash
docker run --rm <your-registry>/npa-ltx2@sha256:<digest> ltx-runtime ensure
# exits 78, names HF_TOKEN and the gated repository, downloads nothing
```

**4. Accept the terms yourself.** Read the
[Agreement](https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/LICENSE.md) and the
[Acceptable Use Policy](https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf),
request access on the gated repository with your own Hugging Face account, and
export the resulting `HF_TOKEN`. Nobody else can hold that entitlement for you —
which is why neither the workflow, the test suite, nor the image contains one.

**5. Run the live GPU smoke.** One RTX PRO 6000 Blackwell, 500 GB disk (the
weight set alone is ~66 GiB, so a default 100 GB disk fails the pull rather than
the generation, which is a much less obvious failure).

```bash
export NPA_INTEGRATION_E2E=1 NPA_LTX2_LIVE_GPU=1
export NPA_LTX2_REUSE_IMAGE=<your-registry>/npa-ltx2@sha256:<digest>
npa/.venv/bin/python -m pytest npa/tests/e2e/test_ltx2_live_e2e.py -q
```

The live test refuses to run unless `HF_TOKEN` is already in the environment; it
never sets it, and `npa/tests/guardrails/test_live_tests_never_declare_a_licence.py`
asserts by AST walk that no live test starts accepting a vendor's terms on your
behalf. Forwarding to a submitted workflow goes through the secret channel —
`--secret-env HF_TOKEN` and `--secret-env NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS` —
never through a spec or rendered YAML.

**6. Record what the run proved.** Replace the accepted manifest's digest and
scan/GPU evidence in one change. Never carry the current proof to a future
digest: publishing bytes whose payload scan and GPU workflow have not run hands
out a claim we have not earned.

## Accepted validation record

The accepted manifest binds all checks to immutable digest
`sha256:c04b5b4e4c7f1c26e21671b3826ce8da75755c98bab2c54cd46137c609c2410b`:

| Check | Result |
| --- | --- |
| Public development build | all packaging, licensing, secret, payload, base-pin, SBOM, vulnerability, provenance, source-revision, non-root, and bootstrap gates passed before/after push |
| `scan_image_ltx_payload.py <digest>` | `pass` — zero LTX source, weights, credentials, acceptance, gated assets, customer data, or live infrastructure data |
| `docker run <digest> ltx-runtime assert-refusal` | printed `NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_ENTITLEMENT_OK`, exit 0 |
| `docker run <digest> ensure` without a token | exit 78, and the refusal names `HF_TOKEN` — the source fetch is entitlement-gated now, not just the weights |
| `docker run <digest> fetch-weights` without a token | exit 78 |
| `docker run <digest> status` | `source: absent`, `weights: absent`, `weights_revision: unknown` |
| Real GPU workflow | one RTX PRO 6000; 1536×1024 H.264, 121 decoded frames, 1,994,625 bytes; both required capabilities passed with no deferred work |
| Independent operator decode | matched MP4 SHA-256 `8899f722b746da93bf79a5fd3bf81d1fdb8b64ea70c5ae3b9fa97a8aac779724` |

| `/opt/npa/ltx2/smoke.sh` in-image | `OK (refusals enforced, no LTX payload present)` |

The first build failed three times, and each failure was a real defect that no
amount of local shell testing had reached:

1. `COPY --chmod=0444` also set `0444` on the directories BuildKit created for
   those files, so `/opt/npa/ltx2` had no execute bit and the runtime user got
   Permission denied on every file inside it.
2. The entrypoint's `ltx-runtime` arm forwarded the literal word as the mode, so
   the runbook's own `docker run <image> ltx-runtime assert-refusal` died as an
   unknown mode.
3. The build's "the refusal wrote nothing" check searched `/workspace` by depth
   and tripped over the empty cache mount points created two lines earlier.

A fourth surfaced in the scan: the payload scanner passed empty audited-bytes
allowlists on the assumption that this image installs no crypto SDK, but it
installs `openssh-server` and `ffmpeg`, whose binaries carry key-format literals
and a CUDA/NVENC ELF reference. Those are now audited by path *and* exact
SHA-256, verified with `dpkg -V` against Debian's package manifests; substituted
bytes at an audited path still fail as `audited_literal_byte_drift`.

All four are regression-tested without Docker, and each test fails against the
pre-fix file.

## Output validation

A file that exists proves nothing, so the smoke decodes the pixels through the
repo's own `video_check` module (copied into the image) and rejects an
unreadable container, a flat single-colour render, and one still frame repeated.
The build exercises the validator on ffmpeg's synthetic sources — a moving clip
that must pass and a black clip that must fail — so a validator that always
passed could not ship. The live test downloads the published MP4 and re-runs the
same check from the operator side, so an in-pod result is never the only
evidence.

The smoke's hard gate requires two capabilities and fails the run if either is
missing: `ltx2_5_text_to_video` and `ltx2_5_decoded_mp4_validation`.
Image-to-video, audio-to-video, and LoRA fine-tuning are not claimed.

## How the refusal is kept honest

The image has two independent refusals — the Hugging Face entitlement and
NVIDIA's CUDA terms — and both fail closed with exit 78. That is a trap: the
first version of the build-time proof checked only the exit code, so a gate
rewritten to accept everything still passed, because the next gate refused
downstream of it and produced the same 78.

`assert-refusal` asserts *which* gate refused, by the variable each refusal
names, across the source fetch, the weight fetch, and NVIDIA's terms.
`npa/tests/docker/test_ltx_runtime_bootstrap.py` executes the shipped script
with `bash` and breaks each gate in turn to prove the proof still notices —
including dropping the token check from `fetch_source` alone, which leaves the
weight path fully guarded and would otherwise slip through. It also runs a
control on the unmutated script, so the mutation cases cannot pass on an
unrelated error. None of it needs Docker or a GPU.

## Validation

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/docker/test_ltx_runtime_bootstrap.py \
  npa/tests/docker/test_ltx_image_payload_scan.py \
  npa/tests/workbench/test_ltx2_licensing.py \
  npa/tests/workbench/test_ltx2_video_check.py \
  npa/tests/workflows/test_byof_solution_smokes.py \
  npa/tests/e2e/test_ltx2_live_e2e.py -q
npa/.venv/bin/python -m pytest npa/tests/docker/ npa/tests/deploy/ -q
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The workbench states these terms and refuses to fetch what an operator is not
entitled to receive. It does not, and cannot, certify their compliance with the
agreement — that is theirs, and this page is not legal advice.
