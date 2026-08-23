---
name: solution-licensing
description: Use when adding, onboarding, or repackaging any solution, tool, container image, model, weights, dataset, or runtime cache; classify each artifact separately, keep restricted weights out of images, design safe runtime caching, and record redistribution decisions where guards enforce them.
---

# Solution Licensing And Redistribution

Use this skill whenever work adds something new to what NPA ships: a workbench
tool, a BYOF/OSS solution, a container image, a base-image swap, a model, or a
dataset. It answers one question:

> We are about to bundle someone else's software. **Who may we hand the result
> to, and in what form?**

"It's open source" does not answer that. The whole point of this skill is that
the answer usually depends on components the top-level license does not cover.

This is engineering guidance for classifying and recording a decision, not legal
advice. When a license is novel, ambiguous, or the vendor's own sources
disagree, record the finding and escalate to a human rather than picking the
convenient reading.

## When To Use

- Onboarding an OSS or partner solution (pairs with
  `skills/workflows/oss-solution-registry-onboard/SKILL.md`, whose License
  admission gate this skill implements)
- Adding a workbench image or changing a Dockerfile's `FROM`
- Adding a model, checkpoint, or dataset to a workflow
- Deciding whether something may be published to a public registry
- Reviewing a PR that does any of the above

Before any provisioning, build, download, or submission that depends on a
third-party EULA, also load
`skills/atomic/third-party-eula-preflight/SKILL.md`; licensing classification
does not itself establish operator consent or upstream asset access.

## The Six Artifact Boundaries

Classify each boundary separately. A permissive answer at one boundary says
nothing about the others.

| Boundary | What it is | Typical trap |
| --- | --- | --- |
| **Source** | The project's own code | Apache/MIT badge on a repo whose *build* pulls proprietary parts |
| **Baked runtime** | Everything the image carries: base image, wheels, SDKs, binaries, assets, fonts, textures | Free to *use*, not to *redistribute* |
| **Weights** | Model checkpoints, adapters, tokenizers, and auxiliary model files | Treating registry access control as permission to redistribute baked bytes |
| **Datasets** | Training, evaluation, calibration, and example data | Assuming a model license also covers its data |
| **Runtime caches** | Downloaded weights, SDKs, compiled kernels, and mutable runtime state | Assuming runtime fetch is durable, or persisting credentials beside cached bytes |
| **Outputs** | What the model generates when an operator runs it | Terms that restrict what the *generated artifacts* may be used for, and that therefore bind pipeline stages downstream of the model |

The decisive layer is normally the **baked runtime**, because publishing an image
distributes every byte in it to whoever pulls it.

**Outputs** is the boundary people forget, because it is the one that is not
about redistribution at all. Ship an image containing none of the vendor's bytes
and every other boundary goes quiet — while a term like "you may not use the
Outputs to train another model" stays fully in force and lands squarely on the
next stage of the workflow. See the LTX-2.5 precedent below.

## Procedure

### 1. Enumerate what the artifact actually ships

Do not read the README. Read the Dockerfile and the lockfile.

```bash
grep -nE '^(FROM|ARG .*(BASE|IMAGE|VERSION))' npa/docker/workbench/<tool>/Dockerfile
grep -nE 'pip install|apt-get install|curl|wget|COPY --from' npa/docker/workbench/<tool>/Dockerfile
```

List: base image, every SDK/wheel installed from a vendor index (for example
`pypi.nvidia.com`, `nvcr.io`), anything downloaded at build time, baked weights,
datasets, and every cache path populated or mounted at runtime.

### 2. Find each component's real license

Prefer the vendor's own licensing page or the package's own metadata over a
repo badge or a summary. Useful checks:

```bash
pip download --no-deps --no-binary :all: <pkg>   # then read the sdist metadata
npa/.venv/bin/python -c "import importlib.metadata as m; print(m.metadata('<pkg>')['License'])"
```

A package whose `License` field literally reads *"NVIDIA Proprietary Software"*
settles the question regardless of what the GitHub repo's badge says.

### 3. Ask the redistribution question explicitly

For every component, answer these four separately — permission for one is not
permission for another:

1. May we **use** it internally?
2. May we **redistribute** it to third parties (shipping an image counts)?
3. May we **run it as a service** for third parties? Vendors often treat
   "install and operate it for a customer" as redistribution even though no
   bits change hands.
4. Are there **field-of-use limits** (non-commercial, research-only, no
   competing service, evaluation-only)?
5. What may the operator do with the **Outputs**? Restrictions here survive
   every packaging trick, because they attach to artifacts we never touch.
6. Do any obligations depend on **facts about the operator** — revenue,
   headcount, entity type, whether this particular use is commercial? If so,
   nobody upstream of the operator can answer them, which changes what the
   image must ask for at run time (see below).

### 4. Resolve conflicting vendor sources

Vendors publish general terms and product-specific terms, and they diverge.
When they do, prefer the source that is **more specific to the component we
actually ship**, and then the **more recent**. Record which source you relied
on and its date, because the next person will find the other one and assume we
were wrong.

### 5. Record the decision where the build enforces it

A conclusion in a PR description is not a control. Encode it:

- `npa/docker/workbench/packaging-contract.yaml` — set `redistribution: public`
  or `restricted` on the image entry.
- `npa/src/npa/deploy/images.py` — add restricted tools to
  `RESTRICTED_PUBLICATION_TOOLS` so `publicly_publishable_tools()` and
  `publish_public` exclude them, and so resolving them from a public registry
  fails loudly. Compatibility aliases retain the original Omniverse-named API,
  but new code must use the general inventory because any vendor runtime can be
  non-redistributable.
- `npa/src/npa/deploy/images.py` — add tools that are licence-eligible but have
  no built, byte-scanned artifact yet to `UNVALIDATED_PUBLICATION_TOOLS`.
  "Restricted" and "unproven" are different answers to different questions, and
  conflating them is wrong in both directions: a tool here is not restricted,
  it simply has no evidence yet, and it leaves the set in the same change that
  records its accepted digest and scan.
- For a solution's weights/datasets, record the license and the runtime-fetch
  requirement in the capability table from the onboarding skill.
- For an **output-layer** restriction, the record has to travel with the
  artifacts: stamp a provenance manifest next to them and put a fail-closed gate
  in front of every downstream stage the restriction reaches. A note in the docs
  cannot stop a workflow; a gate can.

The packaging-contract guards then fail the build if a Dockerfile bakes a
restricted marker, or is built `FROM` a restricted image, while claiming
`public`.

## Patterns That Keep Us Compliant

Three patterns do the real work. Prefer them over asking for an exception.

**Runtime fetch under the customer's own credentials.** Never bake gated or
redistribution-restricted weights merely because a token can gate image access.
The image ships the downloader; the operator supplies their own HF/NGC
credential at runtime and fetches an exact immutable revision when the selected
asset requires authorization. Do not require a token for genuinely public,
anonymous weights. For Hugging Face, the token and its actual upstream repository
permission are the only local access gate: probe every required repository before
provisioning, with no NPA terms boolean or model-check bypass. An HF or NGC token
proves authorization to fetch; it is not EULA acceptance and does not change
redistribution rights.

**Build-your-own.** For a runtime we may not redistribute, ship the Dockerfile
and the build tooling, not the built image. Each operator builds into their own
registry (`build.sh --registry <their-registry> --push`), pulling the vendor
base with their own credentials and EULA acceptance. The vendor delivers to
each operator under that operator's own acceptance; we ship only instructions.

**Runtime fetch of the whole SDK.** Build-your-own has a real cost: the customer needs
vendor credentials, and *we* cannot publish a working image at all. Where the vendor
serves the runtime from an index the customer can reach directly, the stronger move is to
ship an image containing none of it and fetch on first run on the customer's runtime.
The absence of proprietary bytes from published layers is the redistribution control;
EULA acceptance governs runtime use separately. This is how the Isaac images became
publishable; see the worked precedent.
Check whether the vendor's index actually requires a credential before assuming
build-your-own is the only option: `pypi.nvidia.com` serves Isaac Sim anonymously, so the
credential was never the gate — acceptance was.

All three patterns share one idea: **move the vendor's delivery to the customer**, so we
are never the redistributor.

## Runtime Weight Cache Policy

EULA simplification changes acceptance UX only. It does not make a runtime
weight cache durable. Name the cache tier in every workflow design:

| Cache tier | Lifetime and policy |
| --- | --- |
| **Image layer** | Immutable and redistributed with the image. Never use it for gated or redistribution-restricted weights, datasets, credentials, or populated runtime caches. |
| **Node-local ephemeral** | Reuses downloads only on the same surviving node or pod volume. Treat a reschedule, node replacement, or cleanup as a cold cache. |
| **Shared durable PVC/object storage** | Survives workers only when the workflow explicitly provisions and mounts a PVC or stages objects to configured storage. It is not implied by runtime fetch. |

The workbench ships that third tier: `npa/src/npa/workbench/model_cache.py` is the
one place that decides where downloaded weights land, and it is wired into the
SkyPilot renderer, the sim2real sibling GPU Jobs, Serverless Job envs, and VM
Docker deploys. It stays inert until the operator supplies storage
(`NPA_MODEL_CACHE_PVC`, `NPA_MODEL_CACHE_HOST_PATH`, or an explicit
`NPA_MODEL_CACHE_DIR`), then redirects the whole cache family — `HF_*`,
`TORCH_HOME`, `NPA_COSMOS3_CACHE`, `NPA_COSMOS_CURATE_WEIGHTS_DIR`, the LeRobot,
Wan and LTX caches — into it. Reach for the claim in
`npa/docker/workbench/common/model-weight-cache.yaml` and
`docs/workbench/model-weight-cache.md` before designing a per-workflow cache;
a new one-off cache path is how the family fragments and one stage silently
re-downloads.

For a durable cache:

1. Wire the PVC or object-storage location explicitly; do not rely on an image
   path or an ambient host directory.
2. Key cache identity by provider, repository/artifact, exact immutable revision
   or digest, and relevant format/version. Never let mutable `latest` aliases
   overwrite an existing identity.
3. Populate safely under concurrency: download to a unique temporary location,
   verify expected files/checksums, then atomically publish a ready marker or
   immutable prefix. Use a lock or single-writer warm stage where the backend
   needs one.
4. Inject HF/NGC credentials through runtime secret plumbing only. Never write
   tokens into cache metadata, manifests, logs, object keys, or image layers.
5. Reuse without redistribution: run an explicit warm/fetch stage with the
   operator's credential, then mount the immutable cache read-only in consumer
   stages or pass its durable URI and verified identity. Never `COPY` that cache
   into a later Docker build context or publish it as a derived image.

Document the selected asset license, immutable revision/digest, cache tier,
storage wiring, population protocol, and consumer mount/URI in the workflow or
capability record. If those are absent, describe the cache as ephemeral.

## Worked Precedent: Isaac Sim / Omniverse Kit

The canonical case in this repo, and the best template for reasoning — because the
first two answers it produced were both wrong, and the third one changed the product
rather than the argument.

**The layers.**

- **Source:** Isaac Sim's GitHub source is Apache-2.0. Isaac Lab's GitHub repo
  (`isaac-sim/IsaacLab`) is BSD-3-Clause. Both freely redistributable.
- **Baked runtime:** the *shipped binary* bundles NVIDIA-owned components (Omniverse Kit
  SDK, models, textures) under the **NVIDIA Isaac Sim Additional Software and Materials
  License**. Redistributing Isaac Sim with Omniverse Kit to third parties, or delivering
  it to them as a service, requires NVIDIA AI Enterprise. Internal R&D is free with no
  seat limit.
- **Weights and data:** already handled by runtime fetch (Cosmos, GR00T N1,
  Cosmos-Reason) at runtime. Public Hugging Face assets work anonymously; gated
  assets use the operator's token and must pass a real upstream access probe before
  provisioning. There is no second NPA acceptance switch or bypass; NGC credentials
  apply only to NGC-hosted pulls.

**Wrong answer #1: "the source is Apache-2.0, so the image is fine."** The decisive layer
is the baked runtime, and publishing an image distributes every byte in it.

**Wrong answer #2: "Isaac Lab's repo is BSD-3, so we can bake that half and only
runtime-fetch Isaac Sim."** This is the trap worth memorising, because it looks like
diligence. Read the *package metadata*, not the repo badge:

```
$ curl -sL https://pypi.nvidia.com/isaaclab/isaaclab-2.3.2.post1-cp311-none-manylinux_2_35_x86_64.whl -o w.whl
$ npa/.venv/bin/python -c "import zipfile; z=zipfile.ZipFile('w.whl'); print(z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode()[:400])"
Name: isaaclab
License: NVIDIA Proprietary Software
Classifier: License :: Other/Proprietary License
```

and `isaaclab/__init__.py` opens with *"distribution of this software … without an
express license agreement from NVIDIA CORPORATION is strictly prohibited"*. The **wheel**
is a differently-licensed repackaging of the BSD-3 **repo**. Same project, same version,
two licences, and only one of them is on the artefact you would ship.

**Also wrong: "gate the image behind a runtime token."** A token gates a *download*. If
the bytes are already in the layers, a token protects nothing — you have just added a
speed bump in front of a redistribution you have already performed. Any proposal of the
form "we keep baking it but add an access control" is answering the wrong question.

**The answer that worked: move the vendor's delivery to the customer — for the whole
SDK, not just the weights.** The images were re-architected to contain **no NVIDIA Isaac
bytes at all**. On first run they download Isaac Sim and Isaac Lab from
`https://pypi.nvidia.com` into a cache volume. NPA defaults NVIDIA's documented
`ACCEPT_EULA=Y` for these non-interactive workloads and preserves an explicit opt-out.
NVIDIA still delivers the runtime directly to each operator; we redistribute no Isaac
bytes, so the redistribution conclusion does not depend on the EULA UX default.
The clean runtime-fetch `isaac-lab`, `sonic`, and `groot` images may therefore be
classified `redistribution: public`. Historical SONIC L40S and inherited MuJoCo
images remain restricted and quarantined because their built layers contain the
old payload. The replacement MuJoCo architecture is built independently from a
digest-pinned public Python base and must pass exact-layer scans plus real GPU
validation before its digest can replace the quarantined variant.

Three things made that verdict defensible rather than merely plausible, and a new
solution should expect to produce all three:

1. **Default acceptance and explicit opt-out are tested features.** Unset acceptance
   must run non-interactively. Empty, `N`, `NO`, `0`, and `FALSE` must refuse
   before downloading; `Y`, `YES`, `1`, and `TRUE` normalize to acceptance;
   unrecognized values fail separately as invalid. The public-image control
   remains the verified absence of Isaac bytes.
2. **The absence is verified on the artefact.** `npa/scripts/scan_image_omniverse_payload.py`
   streams the built image's filesystem and layer history and fails on Kit payload
   signatures. Reading the Dockerfile is not evidence — the claim is about bytes in
   layers, so check bytes in layers.
3. **The guard was redesigned, not relaxed.** The old check flagged any Dockerfile
   *mentioning* `isaacsim`; the new images legitimately mention it in bootstrap plumbing,
   so the check now distinguishes **baked at build time** from **referenced for run
   time**, and is mutation-tested in both directions. When a guard blocks a change that
   is genuinely fine, the fix is to make the guard encode the real distinction — never to
   widen its exceptions.

**The other trap, still worth knowing.** As of May 2026 NVIDIA announced that Omniverse
is free for development, production *and redistribution*. Read alone that looks like it
lifts the restriction. It does not: the Isaac Sim Additional Software and Materials
License is the product-specific licence, and the Isaac Sim 6.0 documentation — GA'd
4 June 2026, i.e. *after* that announcement — still requires AI Enterprise for
third-party redistribution. More specific and more recent wins. Note that the
reclassification above **did not rely on that announcement at all**; it rests on the
images containing none of the licensed material, which is a much stronger position than
a favourable reading of a vendor's marketing page. Prefer arguments that survive the
vendor changing their mind.

**Useful carve-outs** (unchanged): selling simulation *outputs* (datasets, videos,
reports), or selling custom code and USD assets that the customer runs on *their own*
Isaac Sim, do not require a licence.

**One more layer this case surfaced.** Auditing the images for Omniverse Kit turned up a
*separate* redistribution question nobody had asked: `npa-sonic` also baked NVIDIA
**driver userspace** libraries (`libnvidia-*-580`, apt-pinned and held). Different
vendor licence, same shape of problem. It turned out to be unnecessary — the container
runtime injects the host driver and the Vulkan ICD given
`NVIDIA_DRIVER_CAPABILITIES=all` — so it was removed. When you enumerate what an image
ships, enumerate *all* of it; the component you were sent to look at is rarely the only
one with terms attached.

## Worked Precedent: LTX-2.5 — When The Licence Restricts The Output

Isaac showed how to make a redistribution question moot. LTX-2.5 is the case
where doing that correctly still leaves you with a live problem, so it is the
better template for any model whose licence is a bespoke community agreement
rather than an OSI licence.

**The layers.** Lightricks markets LTX as open source; the LTX-2.x Community
License Agreement is not an OSI licence. Section 1.9 folds "inference-enabling
code, training-enabling code … accompanying source code" into the licensed
material, so unlike Wan — Apache source, gated weights — there is no clean split
where we bake the code and runtime-fetch only the weights. **The code is the
licensed material too.** Section 3 then makes a distributor responsible for
imposing the use restrictions by contract (3.1), delivering the full agreement
(3.2), and not transferring to a Commercial Entity at all until it holds a paid
licence (3.5). An anonymous `docker pull` establishes none of those, which is an
independent reason not to bake even the source. So `npa-ltx2` contains no LTX
bytes at all and fetches both source and weights at run time — the Isaac answer,
reached by a different route.

**Prefer verifiable vendor-side consent to customer self-certification.** This is
the transferable lesson, and it is the opposite of what this integration first
shipped. An earlier version asked the operator for three declarations —
`NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES`, an entity class, and a use class —
because LTX's obligations turn on facts only the operator knows: whether the
entity's revenue crosses Section 2.1's $10M threshold across all affiliates
under common Control, and whether *this* use is commercial. All three were
removed, for two reasons worth carrying forward:

- **The agreement binds by conduct, so a local variable never formed it.** Its
  opening line is "By downloading, using, accessing or distributing any portion
  or element of LTX-2.x, you agree that you have read and accepted to be bound
  by this Agreement." Whoever runs it is bound whether or not they typed `YES`
  to us. Read the acceptance clause before building an acceptance gate: if the
  licence forms on use, an `ACCEPT=YES` collects nothing and only looks like a
  control.
- **The vendor's own gate is checkable; a typed answer is not.**
  `Lightricks/LTX-2.5` is a gated Hugging Face repository, and Lightricks grants
  access only after a human accepts its terms there. Verified empirically: an
  anonymous `HEAD` on a weight file returns `401`, and the same request with an
  entitled token returns `302`. A token that can read the repository is
  therefore *evidence* of acceptance, obtained from the party whose terms they
  are — strictly stronger than a self-typed variable, which is unfalsifiable.
  `npa-ltx2` now requires `HF_TOKEN` for the source fetch as well as the weight
  fetch, because Section 1.9 makes the source licensed material too.

An infrastructure provider that ships zero vendor bytes is not a distributor of
them, and should not ask a customer to self-certify their revenue to it. Look
for an entitlement the vendor already issues — gated repo, licence key, signed
URL, registry credential — and require that. Ask the customer to declare only
what no vendor-side entitlement can stand in for, and prefer stating an
obligation plainly over pretending to have checked it.

**The restriction that packaging cannot reach.** Attachment A(18) prohibits
using the Outputs "to train, improve, or fine-tune any other machine learning
model" for commercial use. In a physical-AI workbench that is not an edge case —
generating synthetic video to train a robot policy is the *reason* to want the
model, and a robot policy is another machine learning model. No amount of
careful packaging touches this, because it governs artifacts sitting in the
customer's own bucket. Which is also why no gate of ours can decide it: whether
A(18) applies turns on the operator's own commercial position. `byof-ltx2.yaml`
therefore stops at curation, which inspects Outputs without training on them,
and leaves training to the operator under their own reading of the clause; the
docs name the obligation instead of computing a disposition for it. Do not build
a control that must guess a fact you cannot observe — a gate keyed to a guess
reads as compliance while enforcing nothing.

**The trap in proving it: gates that mask each other.** `npa-ltx2` has
independent refusals — the operator's own Hugging Face entitlement and NVIDIA's
CUDA terms — that all fail closed with exit 78. The build-time proof originally
checked only that `ensure` exited 78, which meant a gate rewritten to accept
everything still passed, on the strength of another gate refusing downstream of
it. A refusal proof must assert **which** gate refused, and must be
mutation-tested by breaking each gate in turn
(`npa/tests/docker/test_ltx_runtime_bootstrap.py` runs the shipped script for
real and does exactly that, including dropping the token check from the source
fetch alone while the weight path stays guarded). Any time several controls
share an exit code, assume they are hiding each other until a mutant proves
otherwise.

**Do not publish on the strength of the classification alone.** The licence work
concluded `redistribution: public`, and the pushed image has since been scanned
by digest, but no GPU has run it — so `ltx2` sits in
`UNVALIDATED_PUBLICATION_TOOLS` and `publish_public` refuses it by name.
Eligible and proven are different claims.

## Red Flags

Any of these means stop and classify carefully rather than assuming OSS:

- A build step authenticates to a vendor registry or index (`nvcr.io`,
  `pypi.nvidia.com`, a login wall, an API key)
- The Dockerfile sets an EULA acceptance variable (`*_ACCEPT_EULA`,
  `ACCEPT_EULA`, `PRIVACY_CONSENT`) — something in there has terms attached
- The package's own metadata says proprietary, even on an OSS-branded project
- The license permits "use" or "internal use" but is silent on distribution;
  silence is not permission
- `FROM` points at another restricted image — restriction is inherited, and the
  child Dockerfile will show no marker of its own
- Weights or assets are downloaded during `docker build` rather than at run time
- The license names an entity type we might be (cloud provider, service
  provider, competitor) in a carve-out
- The license names an entity type the **operator** might be, or turns on a
  revenue/headcount threshold — we cannot answer that and neither can a variable
  they type, so state the obligation and require the vendor's own entitlement
- The license says anything about the model's *outputs*: how they may be used,
  that they must be disclosed as machine-generated, that provenance markers may
  not be stripped, or that they may not train another model
- The project is described as open source but ships a bespoke "community
  license" instead of an OSI licence; read it for restrictions on the code, not
  just the weights

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/docker/ npa/tests/deploy/ -q
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m npa.deploy.publish_public --dry-run

# For an image that runtime-fetches a vendor runtime, also prove the artefact is clean:
npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py <registry>/<image>:<tag>
npa/.venv/bin/python npa/scripts/scan_image_ltx_payload.py <registry>/npa-ltx2:<tag>
```

The publish dry run is the end-to-end check: whatever it lists is what we would
hand to the public, so a restricted image appearing there is a hard stop.

## Gotchas

- "Free" means free of charge, not free to redistribute. They are unrelated
  questions and vendors price them separately.
- Running a workload for a customer can be redistribution in license terms even
  though no artifact is delivered. Check layer 3 whenever we host something.
- An image is not a bag of licenses; it is a single artifact you hand over
  whole. The most restrictive component governs the whole image.
- Access control is not a license. Making a registry private limits who *can*
  pull, but if a third party pulls a prebuilt restricted image with our
  blessing, that is still redistribution.
- Deleting a public tag does not undo publication. Treat a mistaken publish as
  an incident, not a revert.
- An output restriction cannot be packaged away. Shipping zero vendor bytes is
  the answer to "may we redistribute this?" and no answer at all to "what may
  the customer do with what it generates?" — check whether the pipeline's next
  stage is the thing the licence forbids.
- Several gates that all fail closed with the same exit code will mask each
  other. A refusal test that only checks the code is worth less than it looks;
  assert which gate refused, and mutate each one to prove it.
- Record the license decision at onboarding time. Reconstructing why an image
  was classified two years later, after the vendor's page has changed, is far
  harder than writing one paragraph now.
