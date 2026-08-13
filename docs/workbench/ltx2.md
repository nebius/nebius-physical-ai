# LTX-2.5 Workbench support

NPA packages Lightricks' LTX-2.5 video/audio model as a BYOF solution whose
image contains **no LTX-2.5 bytes at all** — no `ltx-core`, no `ltx-pipelines`,
no weights, no CUDA wheels. On first use the container fetches upstream's pinned
source and the gated weights under the operator's own credentials, and refuses
to fetch anything until the operator has made a licensing declaration that only
they can make.

LTX-2.5 is a generative video model. This integration does not represent it as
an action-conditioned robotics simulator or an action-prediction model.

> **Status: built and byte-scanned; not yet generated on a GPU.** The image has
> been built and pushed on the dev VM, the payload scan passes against the
> pushed digest, and the refusal has been re-proved against those exact bytes
> pulled back from the registry (see "Validated on the dev VM" below). What has
> *not* happened is a generation run: that needs the operator's own LTX-2.x
> declaration, which nobody else may make. `ltx2` therefore stays in
> `UNVALIDATED_PUBLICATION_TOOLS` — publication needs the GPU capability
> evidence as well as the byte evidence, and only one of the two exists.

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
| Official source | [`Lightricks/LTX-2` `fd4ded7…`](https://github.com/Lightricks/LTX-2/tree/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca) | fetched at run time; never baked |
| LTX-2.5 weights | [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) (gated) | fetched at run time with the operator's own `HF_TOKEN` |
| CUDA PyTorch stack | resolved by upstream's own `uv` pins from `download.pytorch.org/whl/cu132` | fetched at run time under separate NVIDIA acceptance |

`ltx_runtime.sh` verifies the fetched `HEAD` against the pinned ref and requires
upstream's `LICENSE.md` to be present before installing, because that ref is
what the provenance manifest claims.

## The declaration

Isaac needs one answer: has the operator accepted the EULA? LTX needs three,
because its obligations depend on facts about the operator that nobody upstream
of them can know.

| Variable | Values | Question it answers |
| --- | --- | --- |
| `NPA_LTX_ACCEPT_COMMUNITY_LICENSE` | `YES` | Have you read and accepted the Agreement and its Attachment A? |
| `NPA_LTX_ENTITY_CLASS` | `community` \| `commercial` | Is your annual revenue, counting all affiliates under common Control (Section 1.6), at or above $10,000,000? |
| `NPA_LTX_USE_CLASS` | `non-commercial` \| `commercial` | Is *this* use the Section 2.2 non-commercial carve-out, or production/revenue-generating? |
| `NPA_LTX_COMMERCIAL_AGREEMENT_REF` | free text | Required when both answers above are `commercial`: Section 2.1 prohibits that combination without a paid Commercial Use Agreement. |

A separate `NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS=YES` covers the CUDA runtime,
and `HF_TOKEN` covers the gated weight repository. Every one of these is absent
by default and none is ever baked. Missing or unrecognised values refuse with
exit 78 before anything is downloaded.

Print the terms without declaring anything:

```bash
npa/.venv/bin/npa workbench ltx2 terms
npa/.venv/bin/npa workbench ltx2 declare      # validates what you have set
```

## The restriction that shapes the workflow

**Attachment A(18)** prohibits using the Outputs "to train, improve, or
fine-tune any other machine learning model, artificial intelligence system, or
competing model" — for commercial use. Section 2.2(c) says the same from the
other side.

In a physical-AI workbench that is not an edge case. Generating synthetic video
to train a robot policy is the obvious reason to want this model, and a robot
policy is another machine learning model. Packaging cannot help: the restriction
governs artifacts in the customer's own bucket. So the control lives in the
pipeline.

`byof-ltx2.yaml` runs:

```text
generate (GPU, npa-ltx2) -> stamp (CPU) -> curate (CPU, FiftyOne) -> gate (CPU)
```

- **stamp** writes `ltx2_provenance.json` (`npa.ltx2.provenance.v1`) next to the
  video: which licence text was accepted, by what class of entity, for what
  class of use, and the resulting Attachment A(18) disposition.
- **curate** runs real FiftyOne Brain uniqueness and duplicate detection. It
  inspects Outputs without training on them, so it is licence-neutral and runs
  before the gate.
- **gate** re-reads the manifest and refuses the named consumer under a
  commercial declaration. It also denies on a missing manifest, an unrecognised
  schema, an unknown disposition, and a manifest that does not cover the
  artifact the consumer named — the breach is a licence-termination event under
  Section 13, so "we couldn't tell" must never mean "proceed".

**The gate is the terminal state, and that is deliberate.** An earlier draft
ended in a LeRobot training state, which made the pipeline look like a complete
demonstration. It was not: that state trained on the `lerobot/pusht` hub dataset,
so no LTX Output ever reached it and the gate in front of it was decorative in
the one place meant to show it is not. LTX-2.5 text-to-video output is not a
LeRobot dataset — no actions, no `meta/info.json` — and no honest conversion
exists here, so training is the operator's own next step, taken after calling the
gate. `npa.workbench.ltx2.check_training_consumer` is the integration point.

Any model trained on cleared Outputs is a Derivative of LTX-2.x under Section 1.5
and stays bound by the Agreement, including the Section 3.5 transfer conditions.

Validate and plan the checked-in spec:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-ltx2.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-ltx2.yaml --run-id ltx2-plan
```

## Dev VM runbook

Nothing below has been executed. Each step produces the evidence the next one
depends on, and the tree records no accepted digest until they have all run.

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
  <your-registry>/npa-ltx2:2.5-rtfetch-unbuilt
```

The tag comes from `supported_tool_version("ltx2")` and carries `-unbuilt`
deliberately, so a tag that has never been produced cannot be mistaken for one
that has. Renaming it is part of step 6, not step 1.

**3. Re-prove the refusal against the pushed image.** This is the same mode the
build ran, now against the artifact anyone would pull.

```bash
docker run --rm <your-registry>/npa-ltx2@sha256:<digest> ltx-runtime assert-refusal
```

It must print `NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_DECLARATION_OK`. Confirm the
undeclared refusal separately, and that it names the licence gate rather than
one of the gates behind it:

```bash
docker run --rm <your-registry>/npa-ltx2@sha256:<digest> ltx-runtime ensure
# exits 78, prints the LTX-2.x terms, downloads nothing
```

**4. Accept the terms yourself.** Read the
[Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md) and the
[Acceptable Use Policy](https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf),
request access on the gated weights repository with your own Hugging Face
account, and export your own answers. Nobody else may set these for you — which
is why neither the workflow, the test suite, nor the image contains them.

**5. Run the live GPU smoke.** One RTX PRO 6000 Blackwell, 500 GB disk (the
weight set alone is ~66 GiB, so a default 100 GB disk fails the pull rather than
the generation, which is a much less obvious failure).

```bash
export NPA_INTEGRATION_E2E=1 NPA_LTX2_LIVE_GPU=1
export NPA_LTX2_REUSE_IMAGE=<your-registry>/npa-ltx2@sha256:<digest>
npa/.venv/bin/python -m pytest npa/tests/e2e/test_ltx2_live_e2e.py -q
```

The live test refuses to run unless the declaration and `HF_TOKEN` are already
in the environment; it never sets them, and a test in the same file asserts by
AST walk that no future edit starts setting them. Forwarding to a submitted
workflow goes through the secret channel — `--secret-env HF_TOKEN`,
`--secret-env NPA_LTX_ACCEPT_COMMUNITY_LICENSE`, and the two class variables —
never through a spec or rendered YAML.

**6. Record what the run proved.** Add the accepted digest and the scan and GPU
evidence, and remove `ltx2` from `UNVALIDATED_PUBLICATION_TOOLS` in the same
change. Not before: publishing an image whose payload scan has never run hands
out a claim we have not earned.

## Validated on the dev VM

Steps 1–3 of the runbook have been executed. The image was built with
`build.sh --push` and scanned by immutable digest
`sha256:246c69c0501e30640413ae49113b5f0cdd82e606d21c7f25b7625cf178ebc084`:

| Check | Result |
| --- | --- |
| `build.sh --push` | succeeded, including every in-build proof: `health`, `assert-refusal`, no files written under `/workspace`, the ffmpeg moving-clip/flat-clip validator pair, and the no-`ltx_core`/no-`*.safetensors`/no-`nvidia` layer checks |
| `scan_image_ltx_payload.py <digest>` | `pass` — 18 archives scanned, zero findings |
| `docker run <digest> ltx-runtime assert-refusal` | printed the marker, exit 0, on bytes pulled fresh from the registry |
| `docker run <digest> ensure` (undeclared) | exit 78, names the licence gate, "Nothing has been downloaded." |
| `docker run <digest> fetch-weights` (undeclared) | exit 78 |
| `docker run <digest> status` | `source: absent`, `weights: absent` |
| `/opt/npa/ltx2/smoke.sh` in-image | `OK (refusals enforced, no LTX payload present)` |

The first build failed three times, and each failure was a real defect that no
amount of local shell testing had reached:

1. `COPY --chmod=0444` also set `0444` on the directories BuildKit created for
   those files, so `/opt/npa/ltx2` had no execute bit and the runtime user got
   Permission denied on the licensing gate itself.
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

The smoke's hard gate requires four capabilities and fails the run if any is
missing: `ltx2_5_text_to_video`, `ltx2_5_decoded_mp4_validation`,
`ltx2_5_license_gate_refusal`, `ltx2_5_license_provenance_stamp`. Image-to-video,
audio-to-video, and LoRA fine-tuning are not claimed.

## How the refusal is kept honest

The image has three independent refusals — Lightricks' terms, NVIDIA's CUDA
terms, and the Hugging Face entitlement — and all three fail closed with exit
78. That is a trap: the first version of the build-time proof checked only the
exit code, so a licence gate rewritten to accept everything still passed,
because NVIDIA's gate refused downstream of it and produced the same 78.

`assert-refusal` now asserts *which* gate refused, by the variable each refusal
names. `npa/tests/docker/test_ltx_runtime_bootstrap.py` executes the shipped
script with `bash` and breaks each gate in turn to prove the proof still
notices; it also runs a control on the unmutated script, so the mutation cases
cannot pass on an unrelated error. None of it needs Docker or a GPU.

## Validation

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/docker/test_ltx_runtime_bootstrap.py \
  npa/tests/docker/test_ltx_image_payload_scan.py \
  npa/tests/workbench/test_ltx2_licensing.py \
  npa/tests/workbench/test_ltx2_gate_cli.py \
  npa/tests/workbench/test_ltx2_video_check.py \
  npa/tests/workflows/test_byof_solution_smokes.py \
  npa/tests/e2e/test_ltx2_live_e2e.py -q
npa/.venv/bin/python -m pytest npa/tests/docker/ npa/tests/deploy/ -q
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

This is engineering enforcement of a recorded classification, not legal advice.
Passing these gates does not substitute for the organization's own publication
and legal approval.
