# Contributing a Containerized Solution

This guide is for contributors who want an open-source Physical AI solution to
run as an NPA container and, when appropriate, become a published Workbench
image. It covers the contribution and evidence expected in the pull request.

You do not need Nebius registry credentials to contribute. The Nebius Physical
AI team rebuilds reviewed code, pushes the official image, validates the
registry artifact, and decides whether to publish it. Never request, copy, or
commit an official registry credential.

## 1. Choose the integration level

Start at the lowest level that proves the solution's real capability:

| Level | Use it when | Main contribution |
| --- | --- | --- |
| BYOF candidate | A public repository needs to be containerized and tested | Pinned source, install command, capability smoke, and BYOF workflow |
| Solution workflow | Users need a repeatable train, evaluate, infer, or generate pipeline | BYOF or curated image plus an `npa.workflow` spec |
| First-class Workbench tool | The solution has a stable maintained API/CLI surface | Dockerfile, image registration, CLI/SDK/YAML integration, golden evaluation, docs, and skill |

See `docs/architecture/oss-onboarding-ladder.md` for the promotion criteria.
Do not add a first-class tool when a BYOF candidate or solution workflow is the
honest product boundary.

## 2. Ground the proposal in upstream behavior

Pin the upstream repository to a full commit SHA or immutable release tag. Read
its maintained README, install guide, examples, model/data instructions, and
license before writing the container.

Describe each capability you want NPA to accept using the upstream project's
own name. For every capability, record:

| Field | Required evidence |
| --- | --- |
| Capability | Upstream environment id, config, script, API, or example name |
| Upstream source | Repository URL, exact ref, and documentation path or URL |
| Command | Smallest documented command that proves the capability |
| Runtime | Ubuntu, Isaac Lab runtime-fetch, custom base, service, or job |
| Compute | CPU, GPU model/architecture, GPU count, memory, and other required devices |
| Inputs and outputs | Real input contract and the artifact produced |
| Status | `accepted` only after a live pass; otherwise `deferred` with the blocker |

An import check, CUDA availability check, or successful container start is not
proof of a solution capability.

## 3. Prepare the containerization change

Fork the repository, branch from current `main`, and keep all reconstruction
inputs in the pull request.

For a BYOF registry candidate:

1. Add `npa/workflows/workbench/npa-workflows/byof-<solution>.yaml`.
2. Use `workload: solution-smoke`, not only `container-verify`.
3. Pin the install in `build_command`.
4. Put the documented capability command in `smoke_command`.
5. Write a named JSON result under `$NPA_SMOKE_OUTPUT_DIR`; it must contain at
   least `solution`, `capability`, and a capability-specific proof value.

Use the CLI dry run to check that the BYOF request is accepted without building,
pushing, or launching infrastructure:

```bash
npa/.venv/bin/npa workbench byof run \
  --repo-url <public-repository-url> \
  --repo-ref <full-commit-or-immutable-tag> \
  --base-profile ubuntu \
  --workload solution-smoke \
  --build-command '<pinned-install-command>' \
  --smoke-command '<real-capability-command>' \
  --solution-name <solution-slug> \
  --capability-name <upstream-capability-id> \
  --smoke-artifact-name <solution>_<capability>.json \
  --skip-push \
  --skip-run \
  --dry-run \
  --output json
```

Use `--base-profile isaac-lab` only for a solution that actually requires that
stack. Isaac-dependent runtime commands must run through
`/isaac-sim/python.sh` and must test refusal when the operator has not accepted
the required EULAs.

For a first-class Workbench image, add or update all applicable surfaces:

- `npa/docker/workbench/<tool>/Dockerfile` and preferably `build.sh`;
- `npa/docker/workbench/packaging-contract.yaml`;
- `npa/src/npa/deploy/images.py` and the supported version source;
- the real CLI, Python wrapper/SDK, and workflow/toolRef resolution;
- focused tests and a golden capability evaluation;
- human documentation and the relevant root skill/index entry.

Use a new additive tag. Never reuse or overwrite a released public tag.

## 4. Make the image reproducible and safe to inspect

The final image must:

- use digest-pinned base images where the registry supports resolution;
- run as a non-root user;
- contain no build credential, token, private key, or tenant identifier;
- keep gated weights, datasets, proprietary SDKs, and model caches out of image
  layers;
- fetch gated material at runtime using the operator's credential and license
  acceptance when that design is legally valid;
- expose an explicit entrypoint and health behavior appropriate to its
  `service`, `job`, or `interactive` packaging tier; and
- retain required license and attribution files.

Public-image builds must not require an application credential. If a build can
only succeed with a gated vendor token, stop and document the licensing and
distribution question instead of hiding the token in BuildKit.

## 5. Classify what the image actually distributes

Review these layers separately:

1. the upstream source;
2. the complete baked runtime, including the base image, wheels, system
   packages, SDKs, binaries, fonts, textures, and assets; and
3. model weights and data.

Record the authoritative license source and access date for each. Set
`redistribution: public` only if every byte in the resulting image may be given
to third parties. If a license is ambiguous, product-specific terms conflict
with general terms, or field-of-use restrictions apply, mark the decision as
blocked and request human review.

This is engineering evidence, not legal advice. A permissive repository badge
does not establish that a vendor wheel, runtime, model, or dataset is
redistributable.

## 6. Run real CPU and GPU capability tests

The pull request must include real execution evidence appropriate to the
solution's claims:

- **CPU-only solution:** run the documented capability on a real CPU and prove
  its functional output. Do not substitute an import or `--help` check.
- **GPU solution:** run the documented capability on a real compatible GPU.
  Record the GPU model, architecture, count, driver/runtime versions, command,
  exit status, and output artifact. A `torch.cuda.is_available()` check alone
  is not sufficient.
- **CPU and GPU modes claimed:** exercise both paths.
- **Multi-GPU claim:** require the requested device count in the test and prove
  that the application created the intended mesh or distributed workers.

Prefer the smallest real example, but let it perform the actual inference,
training step, simulation step, render, conversion, query, or service request
being advertised. Use real representative input rather than an echo or a
manifest-only stub.

Include this test record in the PR:

| Evidence | What to report |
| --- | --- |
| Source | Exact repository ref and PR commit |
| Image | Candidate name/tag and immutable digest, if one was pushed to a contributor-owned registry |
| Hardware | CPU model or GPU model/architecture/count and relevant memory |
| Command | Exact build and capability commands with secrets removed |
| Result | Exit status and concise measured output |
| Artifact | Filename, type, size, hash or key metrics, and how it was inspected |
| Limits | Untested paths, unavailable hardware/assets, residual vulnerabilities, and deferred capabilities |

If the needed hardware is unavailable, keep the capability `deferred`. The NPA
team may perform the missing live run during review, but the capability is not
registry-ready until a real test passes.

## 7. Run repository checks

Use the repository virtual environment:

```bash
python3 -m venv npa/.venv
npa/.venv/bin/pip install -e "npa[dev]"
npa/.venv/bin/python -m pytest \
  npa/tests/docker/ npa/tests/deploy/ \
  npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m pytest \
  npa/tests/workflows/test_byof_solution_smokes.py -q
```

Also run the focused tests for the solution and validate/plan any new workflow
spec. Do not commit project IDs, tenant IDs, registry IDs, bucket names, IPs,
private endpoints, customer names, credentials, or generated model/data
artifacts.

## 8. Open the pull request

In addition to the repository PR template, include:

- the upstream repository and exact ref;
- the capability table and links to upstream documentation;
- base image name and digest, approximate image size, and target architecture;
- a statement that no application credential is required at build time;
- source, baked-runtime, weights, data, and asset licensing evidence;
- the proposed packaging tier and redistribution class;
- exact unit, image, CPU/GPU, and live capability results;
- the named output artifacts and how they were validated; and
- known vulnerabilities, residual risks, and deferred capabilities.

Do not include an official registry URI or ask for registry credentials. A
candidate in a contributor-owned registry is optional and is never treated as
the official artifact.

## 9. What the Nebius Physical AI team does

After the code and evidence are reviewed, the Nebius Physical AI team will:

1. build the exact trusted commit using the checked-in build path;
2. assign the approved additive tag and push it to the private Nebius source
   registry;
3. inspect the pushed registry bytes, including manifest/config, non-root user,
   layer history, secrets, SBOM/vulnerabilities, and any payload whose absence
   is part of the licensing decision;
4. pull that exact image in the real NPA/SkyPilot/Kubernetes path and repeat the
   required CPU and/or GPU capability test;
5. record the immutable source and public digests; and
6. after explicit publication authorization, mirror only a `public`-classified
   image to GHCR and verify anonymous pulls.

Publishing is not performed by untrusted fork CI. A new GHCR package also needs
a one-time manual visibility decision by an administrator; later tags inherit
that package visibility and unchanged images are skipped by the incremental
publisher.

## Related guidance

- `skills/workflows/contribute-workbench-image/SKILL.md`
- `skills/workflows/oss-solution-registry-onboard/SKILL.md`
- `skills/workflows/byof-onboard/SKILL.md`
- `skills/atomic/solution-licensing/SKILL.md`
- `docs/workbench/container-packaging.md`
- `docs/security/container-golden-evals.md`
- `docs/security/image-reproducibility.md`
