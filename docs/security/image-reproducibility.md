# Image Build Reproducibility

This document describes the reproducibility posture of npa workbench container
images. Customers verifying what they run can use this to validate image
provenance and understand the current limits of rebuild determinism.

## Tag and Digest Strategy

### Base Images

Base images should be pinned by SHA256 digest in Dockerfiles. The human-readable
tag stays in the reference, but the digest is the load-bearing content selector:

```dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04@sha256:d49bb8a4ff97fb5fe477947a3f02aa8c0a53eae77e11f00ec28618a0bcaa2ad1
```

To update a base image:

1. Identify the new tag, usually an upstream security patch release.
2. Resolve the new tag's digest through the registry HTTP API. Do not use
   `docker pull` just to resolve a digest.
3. Update the Dockerfile `FROM` line and the CI scan matrix.
4. If the pinned upstream base still carries a fixable OS-package CVE in a
   package that is not needed at runtime, remove only that package from final
   image layers after build-time compilation and mirror that removal in the CI
   base-image scan's minimal derivative.
5. Run CI and verify the Trivy scan passes against the resulting scan target.

At the time this document was added, public Docker Hub bases were digest-pinned.
The `nvcr.io/nvidia/isaac-lab:2.3.2` base is still tag-only because anonymous
manifest access did not return a digest; its Dockerfile carries a TODO marker
until CI or an operator has registry auth for NGC digest resolution.

### npa Image Tags

npa-published runtime images use two tag families:

- `cuda12`: CUDA 12.x runtime for H200, L40S, A100, and earlier supported GPUs.
  This is the production tag family for current H200/L40S validation.
- `cuda13-b300`: CUDA 13.x runtime for Blackwell validation images, including
  B300 and RTX PRO 6000 `sm_120` targets. Production readiness still depends on
  each upstream framework and the target fleet's CUDA 13-compatible host driver.

Customers should select the tag family by target GPU:

- H200 / L40S: `cuda12`
- B300 / RTX PRO 6000 Blackwell: `cuda13-b300` when that path is declared stable

The canonical mapping is maintained in `npa/docker/workbench/tags.yaml`, and CI runs
`npa/docker/workbench/check_tag_consistency.py` to reject tag-family drift.

## CVE Scanning

CI runs Trivy in `.github/workflows/image-security-scan.yml`:

- Pull requests that modify Dockerfiles, `npa/docker/workbench/**`, or the scan workflow
- Pushes to `main`
- A weekly scheduled scan to catch newly disclosed CVEs in already-pinned bases

The workflow scans Dockerfile/config issues and the digest-pinned public base
image lineages. Dockerfile/config misconfigurations fail on HIGH and CRITICAL
findings. Base-image CVE jobs are intentionally OS-package only, use
`--ignore-unfixed`, and fail on fixed CRITICAL vulnerabilities. When a pinned
CUDA base contains a fixable CRITICAL in a build-only OS package that consuming
Dockerfiles remove before the final runtime layer, CI builds a minimal purged
derivative of that pinned base and scans that derivative. This keeps the
digest-pinned lineage visible while validating the remediation that is present
in the shipped workbench images. HIGH base-image CVEs stay visible in SARIF and
scheduled scan output, but they are advisory while the repo is not shipping
those upstream bases unchanged as final runtime images.

The repository-level `trivy.yaml` mirrors the base-image CVE policy for local
operator scans: fixed CRITICAL OS-package vulnerabilities are the blocking gate,
and unfixed findings are not shown. The root `.trivyignore` is reserved for
explicit accepted-risk exceptions. Keep it empty unless a finding has an owner,
a short rationale, and a planned remediation or review date. Do not use it to
blanket-suppress fixable criticals.

To investigate a CI scan failure:

1. Read the Trivy output in the failed CI run.
2. Determine whether the CVE is reachable in npa's use of the affected
   component.
3. Prefer updating the base image to a patched tag and re-pinning the digest.
4. If a fix is unavailable or the finding is not exploitable, document the
   accepted risk before adding a `.trivyignore` entry.
5. If a fixed CRITICAL CVE remains in a vendor image after checking newer
   compatible tags, keep the PR red or update the affected base; do not suppress
   it without an explicit accepted-risk decision.

The workflow uses `aquasecurity/trivy-action@v0.36.0`. That version is
intentional: Aqua's March 2026 incident notes identify older Trivy action tags
as affected and list `v0.35.0` or later safe releases as the remediation floor.

## Build Determinism

Build determinism is partial:

- Base layers: pinned by digest for public Docker Hub bases.
- Python/runtime dependencies: many are versioned through Dockerfile `ARG`
  values, for example Cosmos, GR00T, Genesis, LeRobot, FiftyOne, PyTorch, and
  selected wheel versions.
- npa source: copied from the checked-out repository and installed into the
  image, so the Git commit is part of provenance.
- Apt packages: installed by name without package-version pins. Rebuilds can
  drift when Ubuntu repositories update.
- Python installer tooling: several Dockerfiles upgrade `pip`, `setuptools`,
  and `wheel` without exact versions.
- Build timestamps and layer metadata: not normalized, so independent rebuilds
  should not be expected to be bit-identical.

For stricter rebuild determinism, future work should pin apt package versions or
snapshot apt repositories, consume `npa/requirements-lock.txt` where applicable,
pin installer tooling, and normalize image timestamps with BuildKit/buildx
timestamp rewriting.

## Provenance and Verification

Customers who need to verify an image can compare the base digest in the image
metadata against the Dockerfile source used for the build:

```bash
docker inspect <image>:<tag> --format='{{json .RootFS.Layers}}'
grep '^FROM ' npa/docker/workbench/<tool>/Dockerfile
```

For published images, also capture the repository digest:

```bash
docker inspect <image>:<tag> --format='{{index .RepoDigests 0}}'
```

If the Dockerfile base digest or published image digest differs from the
expected release record, treat that as provenance drift and investigate before
using the image in a customer environment.

## Open Items

- NGC base digest pinning for Isaac Lab once registry auth is available in CI.
- Full deterministic rebuilds, including apt snapshotting and timestamp
  normalization.
- Customer-facing image catalog: which image exists, what it contains, and which
  GPU or workload each image targets.
- `cuda13-b300` remains blocked on upstream Blackwell support in Taichi and
  flash-attn plus host driver readiness.
