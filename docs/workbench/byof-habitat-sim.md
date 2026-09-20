# Habitat-Sim runtime-fetch image

The public `Dockerfile.bootstrap` ships Ubuntu/Python/SSH prerequisites, NPA
launchers and immutable upstream locks. Every inherited and installed Ubuntu
source component accompanies the image, fetched from the same signed official
snapshot and verified against its source descriptor. Habitat-Sim, its native
build dependencies, scientific wheels and the scene are obtained only inside the
operator workload. The interpreter shim also works when SkyPilot overrides the
container entrypoint. Runtime source and venv directories are private, ephemeral
and removed after the workload.

Public development publication requires actual layer-wide payload absence,
bootstrap source-delivery, security, SBOM and provenance checks. Supported release
promotion still requires a real exact-digest RTX RGB-D/navigation/Bullet result.
The previous baked `Dockerfile` and `verify_image.py` remain a quarantined private
candidate; their complete source-closure and retained-byte guards are unchanged.
The legacy packaging details below apply to that candidate only.

## Source and maintenance boundary

- Simulator: [`facebookresearch/habitat-sim`](https://github.com/facebookresearch/habitat-sim)
  at immutable revision `57ee4941dc4765240f0f91f70b2c97a919bf9038`.
- Source license: MIT, from the pinned upstream
  [`LICENSE`](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/LICENSE).
- Image base: `ubuntu:22.04` at manifest digest
  `sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986`.
- Ubuntu snapshot: `20260903T121500Z`, with a checksum-pinned trust bootstrap
  containing only the exact CA, OpenSSL CLI, and libssl packages needed to parse
  the sorted 121-certificate set. Its configuration and bundle are separately
  size/hash bound before HTTPS package transport, Ubuntu archive `Signed-By`, and
  exact Perl-family compatibility refusal. Every direct build/runtime package is
  downloaded separately, matched to its lock-declared package/version and
  `amd64`/`all` architecture, and SHA-256 verified before its local `.deb` is
  passed to APT. Transitive dependencies remain resolved by the signed immutable
  snapshot and are bound by the final complete-package inventory rather than
  falsely claimed as individually hash locked.

The runtime's GPL-3.0+ `rsync` bootstrap binary is accompanied in the image by
the complete four-file Ubuntu source package from the same signed immutable
snapshot. `apt-runtime.lock` binds the signed `Sources.xz` metadata and every
source artifact's name, size, and SHA-256. The build verifies the exact compressed
index size/hash and parses the unique source/version, directory, and complete
Files/Checksums-Sha256 records before accepting the four rsync source artifacts. The OCI
verifier requires those exact source bytes. No mutable source offer is used.

The build obtains only official GitHub codeload archives bound by size and SHA-256.
It materializes an allowlisted projection, excludes unused audio and GUI gitlinks,
and emits a path/size/SHA-256 inventory for every resulting build-source file. The
wheel build enables Bullet and disables CUDA, GUI, audio, tests, and the Basis
compressor. Python wheels are platform-specific and hash locked; the final stage
does not contain build tools, wheel archives, package caches, trust-bootstrap
package archives, CUDA, or NVIDIA vendor libraries. The exact rsync corresponding
source described above is the sole intentional source-archive payload.

Upstream warns that **beyond v0.3.4, Meta internal teams do not officially maintain
releases or provide active development**. This candidate pins the upstream commit
and makes no newer-maintenance claim.

## Six independent licensing boundaries

| Boundary | Treatment |
| --- | --- |
| Source | The exact MIT Habitat-Sim projection and the exact permissively licensed build dependencies are baked with their notices and source inventory. |
| Baked runtime | Public delivery is the objective, not accepted eligibility. Exact inherited and installed copyleft/source delivery, native wheel closure, all layers, notices, and built-byte verification remain pending. |
| Weights | None. Any model, checkpoint, or weight path is forbidden. |
| Data/assets | The image contains no scene. The smoke runtime-fetches only the official Meta archive and verifies the entire archive plus the exact Skokloster GLB/navmesh members. |
| Runtime cache | Unique, mode-restricted, bounded, and ephemeral. Cleanup of partial/archive bytes is attempted on every outcome; failures are reported without masking the initiating error, and operators must verify the cache location is clear. Unrelated archive members are never extracted. |
| Outputs | Operator-owned RGB/depth and JSON artifacts preserve title, creator, scan credit, CC BY link, original asset URL, modification notice, source/image identity, and archive/member hashes. They are never image inputs or public CI artifacts. |

Runtime fetch changes delivery only. It does not create permission, terms
acceptance, or a commercial-use conclusion. Credentials do not authorize content.
The aggregate Hugging Face collection is not a source and grants no permission.
There is no separate EULA or acceptance switch for this selected source/asset path.

## Scene and attribution boundary

The runtime locator is the official Meta HTTPS endpoint
[`habitat-test-scenes.zip`](https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip)
referenced by pinned upstream
[`examples/settings.py`](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/examples/settings.py).
The URL is mutable and is not the asset identity. The smoke requires:

| Item | Bytes | CRC32 | SHA-256 |
| --- | ---: | --- | --- |
| complete archive | 94,590,970 | — | `1231420c6482e79e25beea7ab25121e0421a5fd67b68dd9502145442c288db06` |
| `data/scene_datasets/habitat-test-scenes/skokloster-castle.glb` | 38,295,764 | `7a0ced74` | `b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56` |
| `data/scene_datasets/habitat-test-scenes/skokloster-castle.navmesh` | 28,192 | `a694cab0` | `1a9a5bd123af8001f0ea2c5c8d326cb3fd39808ca771fc766856af8f0772391d` |

The pinned upstream README identifies the scene as [*The King's Hall* by
Skokloster Castle](https://sketchfab.com/3d-models/the-kings-hall-d18155613363445b9b68c0c67196d98d),
scanned by Erik Lernestål, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).
NPA extracts the two processed members byte-for-byte and labels rendered
observations as derived output. No Matterport3D, HM3D, Replica, Gibson, semantic,
or other scene data is permitted.

## Image and publication quarantine

The packaging contract records `redistribution: unvalidated`, while public-neutral
delivery remains the objective. `licenses.json` and the OCI contract record pending
source closure; `UNVALIDATED_PUBLICATION_TOOLS` prevents publication. The candidate has no
public-image-table row and no supported-release manifest entry.

The verifier inventories distinct dpkg and Python package identities from every
layer, including superseded base versions. A reviewed complete inventory and exact
accompanying source bytes are required for every package; rsync alone is not a
complete closure. The checked-in source-delivery contract remains pending and
fails closed. SBOM/native-component and license review must establish sufficiency
before a later authorized byte-qualified publication. Inert tests prove refusal
and byte matching, not legal clearance or image qualification.

The local build output requires an existing owner-controlled directory that is
not group/world writable. BuildKit writes to a private temporary file; only a
successful nonempty regular output is atomically linked to the absent requested
name, mode 0600. An existing destination is never replaced. Cleanup attempts each
exact temporary target independently and reports failures without replacing the
original command exit status; callers must verify cleanup separately.

Source preparation likewise requires owner-controlled output parents. It verifies
the complete projection in private sibling staging before Linux atomic no-clobber
directory publication. Existing projection or inventory paths are refused,
including symlinks. Partial staging is cleaned best-effort on failure; no caller
output is replaced, and cleanup warnings do not conceal the original result.

`npa/docker/workbench/habitat-sim/verify_image.py` accepts only a closed,
attested, single-linux/amd64 OCI graph. It verifies graph/config/layer identities,
reads every regular byte in every ordered layer, checks whiteouts, enforces final
`USER ubuntu`, labels, entrypoint, required payloads, and refuses known scene,
source-archive, cache, credential, CUDA/NVIDIA, FFmpeg, and gated-data paths or
hashes. It also hashes the embedded runtime locks, notices, and smoke module;
reconciles installed apt and Python distributions to those locks; binds every
regular file and link under `/opt/venv` into an operator-reviewed exact
installed-file inventory; requires exactly one wheel `RECORD` beside every
installed distribution's metadata, verifies every hashed `RECORD` entry, and
accepts pip's hashless generated-bytecode rows only when the target exists and
the complete venv inventory independently binds it; checks every projected
source byte against its immutable inventory; binds native venv ELF
files to wheel `RECORD` hashes; parses every ELF `DT_NEEDED`, `DT_RPATH`, and
`DT_RUNPATH` entry, records the loader-order search path, and refuses unresolved
or multiply compatible same-architecture targets; and requires the OCI revision
label to equal the reviewed full Git SHA. The verifier also binds the selected Ubuntu
base diff ID and requires operator-reviewed SHA-256 values for the complete dpkg
inventory, Python venv installed-file inventory, and native closure. The dpkg
inventory includes every installed binary version, architecture, source
package/version, exact dpkg `.list`
path/hash, and resolved copyright path/hash. A missing or duplicate package
file list fails closed. The Python inventory includes every installed venv file
path, size, SHA-256, and link target, independently of in-image metadata. The
native inventory includes every ELF hash, owner, architecture, SONAME, and
resolved `DT_NEEDED` edge. Its first fail-closed scan may expose
those three public inventories for review, but cannot pass until all expected
hashes are supplied explicitly. The independent complete-byte scanner, Trivy, SBOM,
provenance, license, secret, vulnerability, native-library, and exact payload
checks remain mandatory on actual candidate bytes.

A later trusted public workflow must rebuild an exact reviewed full Git SHA. It
must not transport privately built OCI bytes. The trusted rebuild is a new digest:
all byte scans and the real exact-public-digest RTX run must be repeated before an
anonymous pull or catalog claim is accepted. Private and public digest equality is
never inferred from equivalent source.

## Workflow and hard gate

Validate and plan without building or submitting:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  workflows/testing/habitat-sim-smoke.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec \
  workflows/testing/habitat-sim-smoke.yaml --run-id habitat-sim-plan --json
npa/.venv/bin/npa workbench workflow run-spec \
  workflows/testing/habitat-sim-smoke.yaml --run-id habitat-sim-plan \
  --plan-only --scheduler-plan --json
```

These structural planning commands do not submit or qualify an image. The
quarantined `-unbuilt` reference is not runnable: SkyPilot task rendering refuses
it until an explicit registry-qualified immutable image digest is supplied.
A real run requires a later reviewed and byte-qualified exact image digest.

The spec has one `workflow.habitat_sim.smoke` state, selects
`tool://habitat-sim`, and requests exactly
`RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`. Habitat-Sim is a renderer: use a
manager-proven STRICT RTX PRO 6000 Blackwell target, never B200.

The renderer checks that exact accelerator literal and count after normalization.
An identity mapping is supported; a remap to `RTXPRO6000:1`, B200, H100, or a
different count is refused. No independently verified short-alias mapping is
currently bound to this workflow, so scheduler-alias support is deferred. Neither
the literal nor successful rendering proves hardware identity or placement:
the separately authorized live gate must prove STRICT binding to exactly one
RTX PRO 6000 Blackwell. Rendered Habitat tasks also require an explicit
registry-qualified SHA-256 image reference, even with `require_baked_npa` unset
or false; unresolved tags are not runnable-image evidence.

The required local proof is `$NPA_SMOKE_OUTPUT_DIR/habitat-sim-smoke.json`.
Before submission, the operator must hash a mode-restricted immutable rendered-plan
contract and replace the workflow's all-zero `plan_sha256` planning sentinel; the
workload refuses the sentinel, a malformed run ID, and root execution. Success
requires multiple distinct saved RGBA/depth frames, declared shapes and hashes,
finite depth statistics, genuine pathfinder actions, nonzero agent displacement,
Bullet build/enable/step/world-time evidence, measured FPS, NVIDIA EGL/GL library
and vendor evidence, exactly one RTX PRO 6000 with compute capability 12.0, exact
source/scene/archive/member provenance, immutable image identity, and exit zero.
The workload stages every proof and observation under a unique run-owned storage
prefix, verifies each byte, publishes a complete provenance manifest, and only
then conditionally creates `habitat-sim-publication-ready.json`. Consumers follow
that marker to the manifest and reject missing markers, missing or extra staged
objects, and any hash mismatch. Failed publication deletes and verifies only the
exact keys attempted by that transaction. An independent live selector compares
the Kubernetes `containerStatuses` image ID, command, arguments, selected
environment, provider-bound node, and termination message with the immutable
rendered-plan contract before it accepts the staged proof.

The dedicated selector is intentionally inert unless all three owner-controlled
gates are present: `NPA_INTEGRATION_E2E=1`,
`NPA_HABITAT_SIM_IMAGE_LIVE=1`, and a mode-restricted
`NPA_HABITAT_SIM_IMAGE_LIVE_RECEIPT`. The receipt binds the exact Git/workflow
bytes, private image digest, pod UID, run identity, rendered-plan bytes, and the hashes of
separate owner-only registry and provider readbacks. Registry evidence must show
an anonymous 401/403 refusal and authenticated exact-digest pull; known public
registry hosts and equivalent GHCR spellings are rejected. The STRICT provider
receipt binds one node-group ID and the exact Kubernetes node name/provider ID;
the selector requires a `READY` readback performed no earlier than the live
transaction start, reads that node back, and requires the completed pod to name it.
Owner-only JSON is read through mode- and UID-checked no-follow descriptors and
bound to the same directory entry before and after parsing. The selector also
requires the pod's declared container image, observed image ID, non-root runtime
identity, and proof-bound termination receipt to name the same immutable digest.
A succeeded pod or caller-selected proof hash alone cannot pass.
A live transaction and cleanup require separate manager
authorization.

## Deferred

- all proprietary or gated datasets and every scene beyond the two pinned members;
- semantic annotations and semantic-sensor claims;
- Habitat-Lab installation, policy training, and distributed training;
- GUI, interactive viewer, audio, CUDA build, multi-GPU, and B200 rendering;
- built-image, private/public pull, RTX capability, anonymous publication, catalog,
  and supported-release claims until their separate exact-byte gates pass.
