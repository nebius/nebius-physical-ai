---
name: contribute-workbench-image
description: Use when an external contributor or maintainer adds or changes an NPA workbench image and must carry it through licensing review, trusted public-dev build, real validation, and digest-identical GHCR release promotion.
---

# Contribute A Workbench Image

Keep source contribution, trusted building, and supported release promotion as
separate approval boundaries. Public development pushes are irreversible public
disclosures even when their tags are later deleted.

## Governing Rules

Before acting, read:

- `skills/atomic/audit-container-docs/SKILL.md`
- `skills/atomic/secure-image-build/SKILL.md`
- `skills/atomic/solution-licensing/SKILL.md`
- `skills/atomic/third-party-eula-preflight/SKILL.md`
- `skills/atomic/build-and-push-image/SKILL.md`
- `skills/atomic/testing-conventions/SKILL.md`
- `docs/workbench/container-packaging.md`
- `docs/workbench/container-image-catalog.md`
- `docs/workbench/contributing-a-containerized-solution.md`

Use `npa/.venv/bin/python`. Keep live infrastructure identifiers, credentials,
and customer data out of commits, workflow inputs, logs, and PR prose.

## Contributor Boundary

An external contributor supplies reproducible code and evidence in an
unprivileged fork. Never expose official package credentials to fork-controlled
code or run untrusted code in privileged `pull_request_target`/`workflow_run`
jobs. Maintainers rebuild the reviewed commit through the repository workflow.

Add or update the Dockerfile/build script, packaging contract, image resolver
and version source, CLI/SDK/workflow surfaces, golden evaluation, tests, docs,
and relevant skill together. Use non-root final stages, digest-pinned bases when
resolvable, and runtime fetch for gated or non-redistributable material.

Classify source, baked runtime, weights, data, caches, and outputs separately.
Only `redistribution: public` images may enter official GHCR. Restricted images
remain build-your-own in an operator registry.

## Trusted Public Development Build

After review, build the exact trusted full SHA with the manual
`publish-public-images.yml` workflow on a ref whose code contains that workflow:

```bash
gh workflow run publish-public-images.yml \
  --ref <reviewed-branch> \
  -f dry_run=true \
  -f development_sha=<full-git-sha> \
  -f build_development_tools=<tool> \
  -f tool=<tool>
```

The workflow must finish all pre-publication gates before it pushes
`ghcr.io/nebius/nebius-physical-ai/npa-<tool>:dev-<full-git-sha>`. It must not
attach a bootstrap/attestation label unless it actually proved the declared
contract. After push, record the digest and prove anonymous pullability.

Inspect the registry artifact rather than trusting the Dockerfile: layers,
history, OCI config, SBOM/provenance attestations, vulnerabilities, secrets,
licenses, non-root user, credentials, caches, checkpoints, customer data, live
infrastructure data, and tool-specific restricted-payload scans must all pass.

## Functional And Release Gates

Run the smallest real capability on the intended physical GPU from the exact
public development digest. Imports, startup, and CUDA availability are not
functional evidence. Preserve redacted hardware, digest, command, exit status,
and output-artifact measurements.

Promote only that validated digest:

```bash
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target ghcr.io/nebius/nebius-physical-ai \
  --development-sha <full-git-sha> --dry-run
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target ghcr.io/nebius/nebius-physical-ai \
  --development-sha <full-git-sha>
```

Verify the supported tag anonymously and require it to resolve to the same
digest. On failure, delete only an exact run-owned development tag/version after
matching package, tag, and digest; deletion cannot revoke earlier downloads.

## Completion Evidence

Report the reviewed commit, public development reference/digest, gate summaries,
real capability result, release reference/digest when promoted, anonymous pull
checks, and exact cleanup. Mark untested capability or release claims deferred.
