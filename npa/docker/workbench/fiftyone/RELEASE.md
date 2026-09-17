# FiftyOne 1.21 release

The supported image tag is `1.21.0-skypilot-v1-20260915`, bound to
`sha256:9ba5e723b2af8bad4e442781f3e09e2e5649499352ae583b7508bfdd902d7119`.
It was built from source revision `17e3c9e82ceee7db1af59efef132b4c1187ede3b`.
The [trusted build](https://github.com/nebius/nebius-physical-ai/actions/runs/35029858954)
passed all 29 bare-image, initial-install and post-install checks. Independent
review verified its layers, source annex, configured scans and signed provenance
and SBOM. The archive payload scan covered 47,181 paths. These checks retain
the coverage limits described below.

The Dockerfile and VM installer now require FiftyOne 1.21.0, whose App and media
routes default to same-origin access. This removes the wildcard CORS behavior
that allowed a malicious website to read local media responses. Trusted origins
remain an explicit operator setting through `FIFTYONE_ALLOWED_ORIGINS`.

NPA also binds the App to loopback: the stock interface can read files accessible
to its service account and is intended for trusted operators. VM commands use
verified SSH forwards, and `npa workbench fiftyone open` keeps the browser tunnel
alive. Kubernetes uses local port-forward access and rejects public LoadBalancer
exposure. Redeploy existing containers and Kubernetes deployments to replace
their previous listeners. CORS is not an authentication boundary.

The release also includes datasets 5.0.1 with the upstream folder-based
builder metadata-reference fix for
[CVE-2026-66007](https://www.vulncheck.com/advisories/datasets-path-traversal-via-unsanitized-file-name-metadata),
Pillow 12.3.0, and Paramiko 5.0.0. The datasets fix applies to
`FolderBasedBuilder._generate_examples`; it does not establish general metadata
containment for `Dataset.load_from_disk` or `DatasetDict.load_from_disk`.
PAIDF Brain curation constructs FiftyOne samples from staged media without
calling these Hugging Face loaders. FiftyOne 1.21 supports the patched Starlette
1.3.1 and ETA 0.17 dependency closure. The VM installer rebuilds an
existing environment if any of these security requirements is missing.

The bundled database is MongoDB 7.0.40. Its complete official archive hash is
checked before extraction, its notices are retained, and its executable replaces
any wheel-provided copy within the installation layer.

MongoDB Community Server uses SSPL v1, which is
[not OSI-approved](https://www.mongodb.com/legal/licensing/server-side-public-license/faq).
The image contains a verified source annex with the matching
`mongodb-source.tar.gz`, `source.json` provenance and `SOURCE.md` delivery
directions in `/opt/fiftyone/mongodb-notices/`. It also installs
`MONGODB_SOURCE.md` beside the bundled `mongod`. Exact-image verification
checked the archive, binary-to-source mapping, required source members and
recipient access; the [source-delivery directions](SOURCE.md) describe the
supplied files. Retain these checks for each replacement image. The three
notices alone do not establish source delivery.
The [SSPL's distribution and service provisions](https://www.mongodb.com/legal/licensing/server-side-public-license)
need separate review; satisfying image redistribution conditions does not decide
an operator's service-use obligations.

The implemented annex preserves the source archive unchanged and supplies a separate
`version.json` for source-build tooling using the public source commit identity.
Building MongoDB still requires the compiler, build tools and dependencies
documented by that source tree. Archive verification and source delivery do not
establish a byte-identical rebuild of the bundled binary.

Existing registry tags describe previously published bytes. Source changes do
not update them or establish a new validated release. Build an immutable
`dev-<full-git-sha>` candidate and run the packaging, vulnerability, payload,
bootstrap, dependency, source-delivery and functional gates before any public
push. The validation coverage below distinguishes source identity from runtime
checks. Promote only the tested digest through the normal reviewed release
process and update supported image metadata together after publication.

## Validate a local candidate

Run the canonical image validator from the repository root after the reviewed
source is committed and its image is built locally. Set `FIFTYONE_IMAGE_ID` to
the already loaded image's exact `sha256:<64-hex>` ID and
`FIFTYONE_SOURCE_REVISION` to its full 40-character Git revision. That revision
must match the checkout's `HEAD` and the image's OCI revision label. The driver
archives the committed `npa/` subtree and checks its own scripts against those
bytes; uncommitted working files are not the source used for installation.

```bash
umask 077
validation_root="$(mktemp -d)"
npa/.venv/bin/python npa/docker/workbench/fiftyone/validate_image.py \
  --image-id "$FIFTYONE_IMAGE_ID" \
  --revision "$FIFTYONE_SOURCE_REVISION" \
  --source-root "$PWD" \
  --output-path "$validation_root/result"
```

The output path must be a new child of an existing private directory. This
command runs real checks in two owned local containers and verifies their
removal. Bare-image and post-install functional phases run offline; the initial
NPA source install uses Docker bridge networking, disconnected before the
post-install phase. Preserve the original exit and private evidence on failure;
do not push an image whose required checks or cleanup failed.

Before initial setup, the validator assigns only its empty receipt bind mount
to the image's non-root user through the existing passwordless sudo contract.
This supports runners whose host UID differs from the image UID: a writable
foreign-owned file in sticky `/tmp` can still reject shell redirection. The
validator checks the receipt's identity, ownership and empty contents before
running the unchanged setup script; it leaves `/tmp` hardening in place.

The driver verifies actual non-root bootstrap behavior, including passwordless
sudo, writable paths, rsync, SSH start/restart/stop and entrypoint forwarding.
It checks dependencies and the existing environment and CPU Brain/App smokes
before and after the actual initial NPA setup, and verifies the source annex.
The datasets gate binds the installed folder-builder file to the reviewed
5.0.1 wheel bytes. A separate harmless synthetic saved-dataset roundtrip checks
its own emitted files and rows. These checks neither reproduce a vulnerability
nor establish universal metadata containment. The App check inspects the real
ASGI root response for default same-origin behavior; the functional smoke
separately exercises TCP App startup, readiness and stop. Neither result claims
CORS testing of media routes.

Raw logs, package inventories and detailed receipts remain private. The trusted
publication workflow uploads only `public-summary.json`, with the allowlisted
`npa.fiftyone.public-validation.v1` schema: image/source identity, named check
outcomes and counts, selected package versions, source-annex summary and owned
cleanup. Do not upload the containing evidence directory. This validator
complements the required packaging, vulnerability and payload gates; a passing
summary is not publication authorization.

For the full payload gate, `scan_image_omniverse_payload.py --tarball` accepts a
complete Docker-save or OCI-layout archive. It checks that referenced configs,
manifests and readable layers are present before reporting `scan_complete`.
Some Docker image stores can export only metadata while returning success;
that incomplete export is not image coverage. Preserve the failed input and
use a complete archive of the same independently verified digest. This archive
check establishes reference completeness; retain the separate digest, package,
vulnerability and source-delivery checks.

The CPU golden evaluation runs the existing standalone `smoke_functional.py`:
version, dataset creation and query, Brain curation, and App launch on loopback.
The lighter `smoke_env.py` remains useful for environment checks. Qualify the
rebuilt image through the actual initial NPA source install as well, then run
`pip check` in the selected FiftyOne interpreter and preserve the installed
FiftyOne, ETA and Paramiko versions. The image's build-time dependency check
does not establish compatibility after source installation.

The existing non-root SkyPilot prerequisites and bundled MongoDB/Brain path are
retained. The earlier bootstrap correction added sudo, SSH and rsync and replaced
the bare Bash entrypoint with command passthrough.
Prove these behaviors on the exact rebuilt image; the bootstrap label records
the contract but does not execute or establish it.

The release also updates the Python base security release and installs the
hash-locked packaging tools in both Python environments, so the global pip and
wheel copies cannot retain older vulnerable versions.
The same installation layer removes an upstream token-bearing historical image
recipe using the existing exact-source validator. It preserves the loader AST,
primary documentation, package RECORD integrity, and regenerated bytecode.
