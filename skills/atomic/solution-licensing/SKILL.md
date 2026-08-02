---
name: solution-licensing
description: Use when adding, onboarding, or repackaging any solution, tool, container image, model, or dataset, to determine whether what we ship may be redistributed, and to record that decision where the build enforces it.
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

## The Three Layers

Classify each layer separately. A permissive answer at one layer says nothing
about the others, and it is almost always a lower layer that constrains us.

| Layer | What it is | Typical trap |
| --- | --- | --- |
| **Source** | The project's own code | Apache/MIT badge on a repo whose *build* pulls proprietary parts |
| **Baked runtime** | Everything the image carries: base image, wheels, SDKs, binaries, assets, fonts, textures | Free to *use*, not to *redistribute* |
| **Weights and data** | Model checkpoints, datasets, assets fetched or baked | Gated licenses with field-of-use or non-commercial terms |

The decisive layer is normally the **baked runtime**, because publishing an image
distributes every byte in it to whoever pulls it.

## Procedure

### 1. Enumerate what the artifact actually ships

Do not read the README. Read the Dockerfile and the lockfile.

```bash
grep -nE '^(FROM|ARG .*(BASE|IMAGE|VERSION))' npa/docker/workbench/<tool>/Dockerfile
grep -nE 'pip install|apt-get install|curl|wget|COPY --from' npa/docker/workbench/<tool>/Dockerfile
```

List: base image, every SDK/wheel installed from a vendor index (for example
`pypi.nvidia.com`, `nvcr.io`), anything downloaded at build time, and any baked
weights or assets.

### 2. Find each component's real license

Prefer the vendor's own licensing page or the package's own metadata over a
repo badge or a summary. Useful checks:

```bash
pip download --no-deps --no-binary :all: <pkg>   # then read the sdist metadata
python -c "import importlib.metadata as m; print(m.metadata('<pkg>')['License'])"
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
  `OMNIVERSE_RESTRICTED_TOOLS` so `publicly_publishable_tools()` and `publish_public`
  exclude them, and so resolving them from a public registry fails loudly. That set is
  currently **empty**, and deliberately kept that way rather than deleted — a mechanism
  that is removed when unused has to be rebuilt and re-reviewed under time pressure. Its
  tests monkeypatch a synthetic restricted tool in, so it cannot rot while unused.
- For a solution's weights/datasets, record the license and the runtime-fetch
  requirement in the capability table from the onboarding skill.

The packaging-contract guards then fail the build if a Dockerfile bakes a
restricted marker, or is built `FROM` a restricted image, while claiming
`public`.

## Patterns That Keep Us Compliant

Two patterns do the real work. Prefer them over asking for an exception.

**Runtime fetch under the customer's own credentials.** Never bake gated weights.
The image ships the *code* to download them; the operator supplies their own
HF/NGC token at run time and accepts the model license directly. We never
redistribute weights, and the customer's entitlement is theirs. This is how
Cosmos, GR00T N1, and Cosmos-Reason weights are handled.

**Build-your-own.** For a runtime we may not redistribute, ship the Dockerfile
and the build tooling, not the built image. Each operator builds into their own
registry (`build.sh --registry <their-registry> --push`), pulling the vendor
base with their own credentials and EULA acceptance. The vendor delivers to
each operator under that operator's own acceptance; we ship only instructions.

**Runtime fetch of the whole SDK.** Build-your-own has a real cost: the customer needs
vendor credentials, and *we* cannot publish a working image at all. Where the vendor
serves the runtime from an index the customer can reach directly, the stronger move is to
ship an image containing none of it and fetch on first run under the customer's own EULA
acceptance — with a hard refusal when acceptance is absent, since that refusal is the
mechanism. This is how the Isaac images became publishable; see the worked precedent.
Check whether the vendor's index actually requires a credential before assuming
build-your-own is the only option: `pypi.nvidia.com` serves Isaac Sim anonymously, so the
credential was never the gate — acceptance was.

All three patterns share one idea: **move the vendor's delivery to the customer**, so we
are never the redistributor.

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
  Cosmos-Reason) with the operator's own HF/NGC token.

**Wrong answer #1: "the source is Apache-2.0, so the image is fine."** The decisive layer
is the baked runtime, and publishing an image distributes every byte in it.

**Wrong answer #2: "Isaac Lab's repo is BSD-3, so we can bake that half and only
runtime-fetch Isaac Sim."** This is the trap worth memorising, because it looks like
diligence. Read the *package metadata*, not the repo badge:

```
$ curl -sL https://pypi.nvidia.com/isaaclab/isaaclab-2.3.2.post1-cp311-none-manylinux_2_35_x86_64.whl -o w.whl
$ python -c "import zipfile; z=zipfile.ZipFile('w.whl'); print(z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode()[:400])"
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
`https://pypi.nvidia.com` into a cache volume, and **refuse to run** unless the operator
has set both `OMNI_KIT_ACCEPT_EULA=YES` and `ISAACSIM_ACCEPT_EULA=YES`. Nothing is baked
with acceptance pre-granted. NVIDIA delivers to each operator under that operator's own
acceptance; we redistribute nothing; the licensing question is moot rather than argued.
So `isaac-lab`, `sonic`, `sonic-mujoco` and `groot` are now `redistribution: public`.

Three things made that verdict defensible rather than merely plausible, and a new
solution should expect to produce all three:

1. **The refusal is a tested feature, not a comment.** It is the legal mechanism, so it
   is asserted in a unit test, inside the image build itself, and against the built
   image. If it can be bypassed by forgetting an environment variable, it is not a
   mechanism.
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

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/docker/ npa/tests/deploy/ -q
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m npa.deploy.publish_public --dry-run

# For an image that runtime-fetches a vendor runtime, also prove the artefact is clean:
npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py <registry>/<image>:<tag>
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
- Record the license decision at onboarding time. Reconstructing why an image
  was classified two years later, after the vendor's page has changed, is far
  harder than writing one paragraph now.
