---
name: secure-image-build
description: Use whenever an NPA container image is built, tagged, pushed, copied, or promoted, including public full-SHA development builds, release publication, build scripts, and image-producing GitHub Actions workflows. Enforce mandatory safety gates and refuse insecure or restricted publication.
---

# Secure Image Build

Treat every official development push as public and irreversible. Use one
official namespace: `ghcr.io/nebius/nebius-physical-ai/<image>`. Tag reviewed
development bytes `dev-<full-git-sha>` and promote only that exact digest to the
supported release tag.

## Load The Governing Procedures

Read and follow these before acting; do not duplicate their detailed commands:

- `skills/atomic/solution-licensing/SKILL.md`
- `skills/atomic/build-and-push-image/SKILL.md`
- `skills/workflows/contribute-workbench-image/SKILL.md`
- `skills/atomic/third-party-eula-preflight/SKILL.md`
- `skills/atomic/testing-conventions/SKILL.md`
- `docs/workbench/container-packaging.md`

For live validation also load `skills/atomic/gpu-selection/SKILL.md`,
`skills/atomic/submit-workflow/SKILL.md`, and
`skills/atomic/protect-nebius-infra-details/SKILL.md`.

## Mandatory Sequence

1. Resolve the exact checked-out commit. Require a full 40-character Git SHA
   and the immutable tag `dev-<full-git-sha>`; refuse moving or abbreviated
   development tags.
2. Require `redistribution: public` in
   `npa/docker/workbench/packaging-contract.yaml`. Hard-refuse `restricted`
   images, including `cosmos3-serving`, from every official GHCR tag.
3. Before any public push, run the repository packaging/license guards and
   inspect the locally built artifact, including layers, history, and OCI config.
   Refuse credentials, secrets, customer data, live infrastructure identifiers,
   gated weights/data, proprietary SDK/runtime payloads, or cached EULA
   acceptance anywhere in the artifact.
4. Require policy-approved, digest-pinned bases/dependencies or an existing
   documented packaging exception. Enforce the non-root runtime and packaging
   contract. When SkyPilot needs the bootstrap contract, prove the behavior and
   exact OCI attestation; never add the label without the proof.
5. Produce the SBOM and run the repository-supported vulnerability, secret,
   license, payload, revision/provenance, and bootstrap checks. Do not push the
   public development tag until every pre-publication gate passes.
6. Push with explicit `packages: write`, then resolve the tag once to its OCI
   digest. Verify revision/provenance and SBOM attestations, rerun exact-digest
   payload/security checks, and prove an anonymous pull without ambient auth.
7. Run the real functional workflow on a compatible physical GPU using the
   immutable development digest. An import, startup, or CUDA-availability check
   is not functional evidence.
8. Promote only the validated digest to the supported public release tag through
   `npa.deploy.publish_public`. Verify anonymous pullability and exact digest
   identity independently after promotion; record the exact digest identity in
   `public_release_manifest.json`. Scheduled health must compare the release tag
   anonymously with that `published_digest`, without depending on dev-tag retention.
9. On failure, delete only the exact run-owned development version after
   matching package, tag, and digest. Never infer ownership from a name or delete
   a shared/release digest. Record that deletion cannot revoke prior downloads.
   Retain a successful dev tag only when the documented release policy requires
   it to preserve the release's shared digest/provenance.

## Refusal Conditions

Stop before publication if any mandatory evidence is missing, a scan is
unavailable or inconclusive, the source commit/tag is mutable, redistribution is
not public, the pushed digest differs from the inspected artifact, anonymous
pull verification fails, or real GPU validation has not passed. Never weaken a
gate or substitute a label, unit test, or deletion promise for artifact evidence.
