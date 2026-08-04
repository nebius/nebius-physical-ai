---
name: contribute-workbench-image
description: Use when an external contributor or maintainer adds or changes an NPA workbench container image and must carry it safely from a fork pull request through licensing review, trusted build, private source-registry push, live validation, and incremental GHCR publication.
---

# Contribute A Workbench Image

Keep source contribution, trusted building, and public publication as separate
approval boundaries. Let anyone propose reproducible image code. Let only a
maintainer run reviewed code with registry access, and let an authorized human
make the irreversible public-release decision.

## Load The Governing Rules

Read these before changing an image:

- `docs/workbench/contributing-a-containerized-solution.md`
- `docs/workbench/container-packaging.md`
- `skills/atomic/solution-licensing/SKILL.md`
- `skills/atomic/build-and-push-image/SKILL.md`
- `skills/atomic/testing-conventions/SKILL.md`

Use `npa/.venv/bin/python` for repository validation. Keep tenant, project,
registry, bucket, service-account, key, IP, and secret values out of committed
files. Preserve unrelated dirty-tree changes.

## Choose The Role

### External contributor

Prepare code and evidence in a fork. Do not request, receive, or use the
official registry credentials. An optional candidate image may live in the
contributor's own registry, but maintainers must rebuild the official artifact
from reviewed repository source.

### Maintainer

Review the fork with unprivileged CI, merge or reproduce the reviewed commit on
a maintainer-owned branch, build the exact trusted commit, validate the pushed
bytes, and publish only after explicit authorization.

Never check out or execute untrusted fork code in a privileged
`pull_request_target` or `workflow_run` job. Never expose registry credentials
to a fork PR, a contributor-controlled action, or a build step that requires a
secret. Public-image builds must need no build-time application credential;
gated runtimes, weights, and data belong in an operator-authorized runtime
fetch.

## Prepare The Contribution

1. Fork the repository and branch from current `origin/main`.
2. Use a new, additive tag. Do not reuse or overwrite a released public tag.
3. Build from checked-in source under `npa/docker/workbench/<tool>/`; do not
   make a detached Dockerfile or an image that can only be reconstructed from a
   developer machine.
4. Add or update the applicable surfaces:

   - `Dockerfile` and, when useful, `build.sh`
   - `npa/docker/workbench/packaging-contract.yaml`
   - `npa/src/npa/deploy/images.py` and the supported version source
   - CLI, SDK, and `npa.workflow` image resolution when behavior changes
   - golden evaluation, focused tests, human documentation, and the relevant
     tool skill

5. Make the final stage non-root, pin resolvable bases by digest, keep secrets
   out of layers, and make service health behavior machine-checkable.
6. Classify the source, baked runtime, and weights/data separately. Record
   `redistribution: public` only when every shipped component may be handed to
   third parties. Record ambiguity and stop for human review; never weaken a
   guard or relabel an artifact merely to make publication pass.
7. Keep gated weights, datasets, and proprietary SDKs out of layers. Fetch them
   at runtime using the operator's own credential and acceptance when that
   design is legally valid. Test any refusal or acceptance gate as product
   behavior.

## Validate The Pull Request

Create the repository virtualenv if needed, then run focused checks before the
full non-live suite:

```bash
python3 -m venv npa/.venv
npa/.venv/bin/pip install -e "npa[dev]"
npa/.venv/bin/python -m pytest \
  npa/tests/docker/ npa/tests/deploy/ \
  npa/tests/guardrails/test_skills_index.py -q
make test PYTHON=npa/.venv/bin/python
make lint PYTHON=npa/.venv/bin/python
```

Build locally without credentials when practical. For a runtime-fetch image,
test both refusal without acceptance/credentials and success with the
operator-supplied runtime inputs. Add committed live-infrastructure coverage
for a new or changed workflow/tool path, following `testing-conventions`.

Open the fork PR against `main`. Include:

- exact image name and proposed tag;
- base image and digest;
- expected architecture and approximate size;
- build-time credential statement;
- source, runtime, weights, and data license evidence;
- packaging tier and redistribution class;
- unit, image, and live functional results; and
- known vulnerabilities or residual release risks.

## Build The Trusted Artifact

After review, build from the exact trusted commit. Prefer the checked-in
`build.sh`; otherwise use `docker buildx build --push` so a large image streams
to the configured private source registry rather than unpacking into the local
daemon. Resolve the registry from NPA configuration and pass it as an argument;
do not commit its concrete identifier.

```bash
npa/docker/workbench/<tool>/build.sh \
  --registry "$NPA_REGISTRY" \
  --tag <new-tag> \
  --push

crane manifest "$NPA_REGISTRY/npa-<tool>:<new-tag>" >/dev/null
```

If the checked-in script has different flags, use its help rather than
inventing an alternate build path. Match the pushed name and tag exactly to the
pin resolved by `npa.deploy.images`.

Inspect the registry artifact, not only the Dockerfile or local daemon:

- run the appropriate Trivy/SBOM and layer-history checks;
- scan the full filesystem for payloads whose absence is part of the license
  decision;
- confirm non-root runtime and absence of credentials, caches, weights, and
  checkpoints;
- run the real capability test on the intended CPU/GPU architecture; and
- record the immutable manifest digest and exact source commit.

Never publish a `restricted` image. A successful build is not evidence that
redistribution is permitted.

## Verify Source And Public State

Do not infer current coverage from the number of GHCR package pages. A package
may be public while the repository's newly pinned tag is absent.

Set `NPA_PUBLIC_REGISTRY` to the intended public target before either check.
The explicit `--target` arguments below avoid silently validating a different
configured registry.

Check every pinned source tag with the configured read credential:

```bash
: "${NPA_PUBLIC_REGISTRY:?set NPA_PUBLIC_REGISTRY to the intended public registry}"
unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN
npa/scripts/nebius_registry_docker_login.sh
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target "$NPA_PUBLIC_REGISTRY" --preflight
```

Check every exact target tag through the anonymous path:

```bash
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target "$NPA_PUBLIC_REGISTRY" --verify-public
```

Interpret the results precisely:

- source missing: build/push the pinned tag or correct the pin;
- source present and target HTTP 404: the tag still needs mirroring;
- both present with equal `crane digest` output: current and complete;
- both present with different digests: stop and investigate tag reuse or an
  unintended rebuild before copying anything; and
- source denial: fix identity or permissions; never treat it as a missing
  image.

## Publish Incrementally

Public publication is an explicit human decision. Do not dispatch it, change a
package's visibility, or accept a release risk without authorization for the
exact plan.

1. Run **Publish public images** on `main` with `dry_run=true` and
   `skip_missing=false`.
2. Review the plan, source preflight, license gates, and exact pins.
3. After authorization, run it once with `dry_run=false`.
4. Confirm the summary says every planned image is anonymously pullable.

The publisher compares manifest digests and skips unchanged images. It does not
republish the entire mirror. A new tag in an existing package inherits that
package's public visibility. A completely new package is private after its
first push and needs one manual **Danger Zone -> Change visibility -> Public**
operation by an administrator, followed by verification. Treat that change as
irreversible.

## Completion Evidence

Report the PR and merge commit, source and GHCR digests, scan results, real
functional-test result, publication run URL, and anonymous verification count.
State missing or mismatched tags explicitly. Clean up temporary credentials,
builders, worktrees, and scratch files after preserving the evidence needed in
the PR.
