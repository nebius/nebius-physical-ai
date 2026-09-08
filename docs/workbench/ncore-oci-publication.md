# NCore OCI development publication

NCore remains quarantined from supported releases pending complete conversion,
RTX NRE training/rendering and readback acceptance. This command publishes only
`ghcr.io/nebius/nebius-physical-ai/npa-ncore:dev-<full-source-sha>`. It does not
fill acceptance records, change a release tag or add a public catalog row.

The NCore branch of `publish-public-images.yml` uses
`npa/scripts/publish_ncore_oci.py`. Other tools retain their existing workflow.
The CLI requires Linux amd64, the checkout's CPython 3.12 environment with the
NPA development dependencies, Docker/buildx, the Debian archive keyring, and
`skopeo` plus `gh` for publication. Use an exclusively controlled builder and
registry writer for this immutable tag. CI dispatches sharing a development SHA
are serialized by the existing workflow concurrency group. Skopeo/OCI registry
tag writes do not offer a portable atomic create-if-absent operation: a separate
writer that ignores this serialization can race a tag check. Do not run another
publisher for the same tag concurrently.

Run from the exact reviewed, committed coordinator checkout. Source authorization
requires `HEAD` and the full requested SHA to agree, and verifies the committed
build context and executable gate sources. Unrelated dirty paths are permitted.
Private scanner policy must already be available as `CUSTOMER_DENYLIST` and
`INFRA_DENYLIST`. These values never become Docker build arguments or container
environment variables. Keep their provisioning outside source files and command
logs. Do not upload the analysis directory as a public CI artifact.

An operator with an existing exact-literal policy can instead supply
`--policy-mode exact-literals --literal-inventory <private-file>` to each
command. The nonempty owner-only JSON inventory must be inside the analysis
directory. It uses the existing verified native matcher with
`exact-substring-v1`; all literal values remain private. CI keeps its existing
regex policy. Neither mode changes the credential detector or byte coverage.

Choose a fresh owner-only `NCORE_OCI_ROOT` directory outside the checkout and set
`SOURCE_SHA` to the reviewed full SHA. The directory must already exist with
mode 0700. Commands below create their own new child directories and refuse to
replace evidence:

```bash
umask 077
npa/.venv/bin/python npa/scripts/publish_ncore_oci.py prepare \
  --source-sha "$SOURCE_SHA" --analysis-root "$NCORE_OCI_ROOT"
npa/.venv/bin/python npa/scripts/publish_ncore_oci.py build \
  --source-sha "$SOURCE_SHA" --analysis-root "$NCORE_OCI_ROOT" \
  --builder "$NCORE_BUILDX_BUILDER"
npa/.venv/bin/python npa/scripts/publish_ncore_oci.py check \
  --source-sha "$SOURCE_SHA" --analysis-root "$NCORE_OCI_ROOT" \
  --output-dir "$NCORE_OCI_ROOT/check" \
  --annex "$NCORE_OCI_ROOT/sources" --native-source "$NCORE_OCI_ROOT/sources" \
  --metadata "$NCORE_OCI_ROOT/metadata" \
  --bootstrap-source "$NCORE_OCI_ROOT/bootstrap-source"
```

`prepare` invokes the existing pinned Go/Gitleaks and Aho-Corasick preparation
and real native integration checks. It downloads the locked source annex,
separate signed Debian metadata, and the two hash-pinned SkyPilot bootstrap
sources. It requires the `base_sources.py --metadata` integration; an older
helper fails rather than combining build metadata with delivered source.
`build` calls the existing committed-context `ncore/build.sh --oci-output`
with `--provenance=mode=max` and `--sbom=true`. The actual buildx metadata,
original archive hash, complete committed context archive hash and invocation
are retained in `build/`. Direct `build.sh --load` smoke images cannot enter this
publication command.

`check` performs all gates without registry mutation. Graph, archive, source,
payload and confidentiality checks operate on local artifacts. Trivy requires
its vulnerability database, and disposable CPU bootstrap probes fetch locked
runtime dependencies and run the real pinned SkyPilot APT/provisioner commands;
the full command therefore requires network access. Its initial `--image-only`
packaging probe runs with networking disabled. Cold and warm runtime checks run
through both the normal entrypoint and a Bash entrypoint override. Runtime
installs and caches remain in disposable containers and are never committed
back into the image. No GPU workload is part of this prepublication command.

The inspection adapter accepts one assembled scratch-root layer, optionally
followed by the exporter's exact canonical `WORKDIR` no-op: 1,024 zero decoded
bytes with the fixed stored gzip digest, or those same uncompressed zeros.
Both stored and decoded identities must match. Changed gzip headers, padding,
extra entries and further layers are rejected. The existing strict
`ncore_verification.py` graph validator still checks every physical layer and
rejects partial graphs, extra blobs and unsupported layouts. Rootfs derivation
rejects whiteouts, duplicate paths, special files and hardlinks, and retains:

- The unchanged original OCI archive and complete-byte scan, including config,
  history, all physical archive bytes and embedded attestations.
- A derived Docker-save archive for existing payload and source-inventory
  tooling, with the exact original config and every original stored layer blob
  in order, including the optional no-op. It carries no repository tag.
- The final filesystem tar and its derivation receipt. Its SHA256 must equal
  the first original rootfs diff ID. The optional tail changes no filesystem
  entries; it remains in the physical layer counts and complete-byte scan.
- Genuine buildx SPDX and source-bound SLSA statements, the ordinary pinned
  Trivy vulnerability/secret/license reports, and the separate selected-base
  SPDX/Trivy scan generated by `npa.deploy.ncore_selected_sbom.scan_archive`.
  The latter requires actual `--list-all-pkgs` coverage; lock-only output cannot
  qualify. It does not replace the mandatory embedded buildx SPDX predicate.
- Source authentication using the actual shipped lock, final-file inventory,
  corresponding-source annex, separate signed repository metadata and keyring.
  Private denylists never enter these public source inputs.

Before any NCore bootstrap container starts, the command captures the single
immutable identity reported by the local Docker load, inspects that identity
and saves it back to a private archive. Docker's containerd store can identify
the loaded image by a Docker v2 manifest digest. That local manifest must bind
the exact original config bytes and every stored layer blob, with only the
known OCI-to-Docker media-type mapping. The saved index, manifest and payload
must agree, with no missing or unrelated image blobs. Classic Docker config
IDs require the same config byte proof and every original layer's stored bytes
or exact decoded tar bytes; equal extracted files alone cannot qualify.

The private `local-image-binding.json` receipt records the relationship between
the runnable local identity and the original publication index, platform,
config and ordered layers. Unknown or ambiguous load identities fail before
container execution. All probes use the verified immutable local ID with
`--pull=never`. Use an exclusively controlled Docker daemon for these probes.
This local representation never replaces the original OCI archive, config,
attestations or graph used for publication and anonymous readback.

Publication requires an owner-only Docker/containers auth file under
`NCORE_OCI_ROOT`, supplied by the coordinator's existing registry login, and
the existing `gh` authentication. Never put credentials on the command line.
`publish` reruns all gates into a new directory; a previous pass JSON cannot
authorize a later write:

```bash
npa/.venv/bin/python npa/scripts/publish_ncore_oci.py publish \
  --source-sha "$SOURCE_SHA" --analysis-root "$NCORE_OCI_ROOT" \
  --output-dir "$NCORE_OCI_ROOT/publication" \
  --annex "$NCORE_OCI_ROOT/sources" --native-source "$NCORE_OCI_ROOT/sources" \
  --metadata "$NCORE_OCI_ROOT/metadata" \
  --bootstrap-source "$NCORE_OCI_ROOT/bootstrap-source" \
  --authfile "$NCORE_OCI_ROOT/registry-auth.json"
```

The command verifies the index selected by Skopeo before its first registry
write, refuses a divergent existing dev tag, and avoids copying an already
equal tag. A proven absent tag is checked again immediately before transfer.
It invokes `skopeo copy --all --preserve-digests` on the original OCI archive
using a digest destination, checks the dev tag again after that upload, then
copies the exact registry digest to the absent tag;
Docker load/push and later GitHub referrers are not the publication path.
These flags preserve the full list and require digest preservation.
See the [Skopeo copy reference](https://github.com/containers/skopeo/blob/main/docs/skopeo-copy.1.md).

Only after every prepublication gate passes can a push or visibility mutation
occur. An existing private package can become public only if every version
belongs to this verified graph and every tag is this dev tag. Readback explicitly
disables credentials, downloads the entire graph, compares index/platform/config
and every blob descriptor, and reruns the unchanged complete-byte gate on the
downloaded archive. Equality binds source, payload, security, bootstrap and SBOM
evidence to those same registry bytes. A failed readback never produces a
`published.json` receipt.

Dispatch the same supported path after the coordinator commits the integrated
metadata, native-delivery, selected-SBOM and publication changes:

```bash
gh workflow run publish-public-images.yml --ref "$REVIEWED_REF" \
  -f dry_run=true -f development_sha="$SOURCE_SHA" \
  -f build_development_tools=ncore -f tool=ncore
```

Here `dry_run` controls release promotion, as in the existing workflow; requesting
a development build publishes after its gates. NCore does not enter generic
failed-build cleanup because the tag may have existed before this run. Retain
failed evidence privately, establish exact creation/tag/digest ownership before
any cleanup, and use the existing explicit cleanup procedure. Public downloads
cannot be revoked by deleting a tag.

No passing image result follows from unit tests or this wiring. A fresh build
and the actual native byte, source/license/payload, image and supplemental Trivy,
bootstrap and registry readback gates must pass on the final committed artifact.
The existing full-input conversion and RTX acceptance schema remains the release
boundary, independent of development publication.
