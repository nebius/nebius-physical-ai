# NCore OCI development publication

NCore remains quarantined from supported releases pending complete conversion,
RTX NRE training/rendering and readback acceptance. This command publishes only
`ghcr.io/nebius/nebius-physical-ai/npa-ncore:dev-<full-source-sha>`. It does not
fill acceptance records, change a release tag or add a public catalog row.

The NCore branch of `publish-public-images.yml` uses
`npa/scripts/publish_ncore_oci.py`. Other tools retain their existing workflow.
The CLI requires Linux amd64, the checkout's CPython 3.12 environment with the
NPA development dependencies, Docker/buildx, `dpkg-deb`, `gpgv`, and
`skopeo` plus `gh` for publication. Use an exclusively controlled builder and
registry writer for this immutable tag. CI dispatches sharing a development SHA
are serialized by the existing workflow concurrency group. Skopeo/OCI registry
tag writes do not offer a portable atomic create-if-absent operation: a separate
writer that ignores this serialization can race a tag check. Do not run another
publisher for the same tag concurrently.

Run from the exact reviewed, committed coordinator checkout. Source authorization
requires `HEAD` and the full requested SHA to agree, and verifies the committed
build context and executable gate sources. Runtime repository imports, including
the payload-history classifier, must resolve inside the expected checkout and
match their committed blobs before execution; the loader
compiles those compared bytes without using cached bytecode. The source check
also compares the actual loaded repository module population. The packaging
guards execute from a complete committed repository snapshot, including
transitive imports, conftests, configuration and data. Their receipt requires
every collected guard to pass setup, call and teardown, with no skips,
deselections or expected failures. Python environment controls, site startup
and pytest plugin autoload are disabled for that guard subprocess. Unrelated
dirty paths never enter the snapshot and are permitted. The operator's host
interpreter, installed dependencies and initial CLI startup remain trusted.
The CLI starts project imports with a fresh private bytecode-cache namespace,
and each Python subprocess receives its own empty namespace with cache writes
disabled. Disabling writes alone would still allow old bytecode to be read.
Buildx metadata permissions are restricted through a verified regular-file
descriptor; content, identity and single-link ownership are checked again before
the metadata becomes a bound gate input.

Private scanner policy must already be available as `CUSTOMER_DENYLIST`, with
optional `INFRA_DENYLIST`. These values never become Docker build arguments or container
environment variables. Keep their provisioning outside source files and command
logs. Do not upload the analysis directory as a public CI artifact.

The CLI emits flushed stderr markers such as
`NCore OCI phase=byte-scan status=begin`, followed by `status=pass` or
`status=failure`. Only fixed, allowlisted action and phase identifiers appear;
exception text, subprocess output, paths and policy matches remain private.
Preparation separately identifies the pinned keyring, native scanner tools,
literal engine, native integration checks, source annex and bootstrap-source
downloads. A preparation failure therefore identifies the failing public
operation without publishing its private subprocess log.
Anonymous public dependency fetches retry HTTP 429, 500, 502, 503 and 504
responses up to three times with exponential backoff before failing. No error
body is delivered as artifact bytes, and hash, TLS, host, redirect and
authentication checks remain mandatory. Interrupted payload reads still fail
so the caller can discard its partial file.
The byte scan has nested `byte-scan-authorization`, `byte-scan-execution` and
`byte-scan-report` markers, so a stalled or failed boundary is distinguishable.
Its report phase emits fixed `outcome`, `coverage` and `findings` summaries that
contain only strict booleans and bounded nonnegative counts. Missing, malformed
or oversized count groups report `available=false`; no policy, match, path,
argument, environment or exception value is included.
When the scanner exits nonzero after writing a report, the same bounded summary
is still emitted and the original failure is preserved. A missing or unreadable
report emits only `available=false`. The optional `credential-findings` count
comes from the native credential detector; the total includes confidentiality
findings too. These diagnostic counts never approve findings or publication.
The NCore path also hashes the complete private `report.json` and
`records.jsonl` bytes. These `raw-report` and `raw-ledger` SHA-256 values let an
operator correlate retained evidence without uploading it. If, and only if,
the separate attribution verifier succeeds, an `attribution-acceptance` line
hashes that complete receipt as separate provenance evidence. None of these
hashes is a scan verdict, disposition, exemption or release acceptance.

When both raw files have the expected scanner schema and mutually consistent
bounded counts, fixed `byte-scan-detail` lines count the implementation's
`customer-denylist`, `infra-denylist` and `private_literal` rule identifiers,
plus fixed native-credential, structural and unclassified categories. They
also count each fixed byte-scanner record kind. Unexpected classes contribute
only to the fixed `unclassified` field; malformed, partial or inconsistent
evidence emits only `byte-scan-detail available=false`. Arbitrary rule names,
record kinds, report metadata, paths, offsets, lines, patterns, matches,
exceptions, arguments and environment values are never printed.

For a `layer_regular_content` record with findings, the diagnostic prints its
bounded byte size and finding count with
`SHA256(ASCII(raw-content-sha256))`. It never prints the raw content digest or
logical path, and it never fingerprints path or structural records. To compare
a separately built local image's known-public dependency file, hash the file's
exact bytes once, encode that lowercase digest as 64 ASCII characters, and hash
those characters again. Equality identifies equal whole-file content without
making the diagnostic an allowlist: any finding still fails and still requires
the existing attribution verifier or policy outcome.
Phases distinguish source and provenance checks, byte and payload scans,
image and component security, bootstrap, registry transfer, visibility and
anonymous verification. The anonymous verification phase includes a second
`byte-scan`. On failure, the innermost phase fails first, followed by its
enclosing phases. A begin marker without a terminal marker means the phase
did not report completion. Pass requires the phase's checks to return
successfully; an evidence file alone is insufficient. The existing stdout
success marker and mandatory publication gates are unchanged.

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
  --source-sha "$SOURCE_SHA" --analysis-root "$NCORE_OCI_ROOT" \
  --keyring "$NCORE_OCI_ROOT/keyring/debian-archive-keyring.gpg"
npa/.venv/bin/python npa/scripts/publish_ncore_oci.py build \
  --source-sha "$SOURCE_SHA" --analysis-root "$NCORE_OCI_ROOT" \
  --keyring "$NCORE_OCI_ROOT/keyring/debian-archive-keyring.gpg" \
  --builder "$NCORE_BUILDX_BUILDER"
npa/.venv/bin/python npa/scripts/publish_ncore_oci.py check \
  --source-sha "$SOURCE_SHA" --analysis-root "$NCORE_OCI_ROOT" \
  --output-dir "$NCORE_OCI_ROOT/check" \
  --annex "$NCORE_OCI_ROOT/sources" --native-source "$NCORE_OCI_ROOT/sources" \
  --metadata "$NCORE_OCI_ROOT/metadata" \
  --keyring "$NCORE_OCI_ROOT/keyring/debian-archive-keyring.gpg" \
  --bootstrap-source "$NCORE_OCI_ROOT/bootstrap-source"
```

`prepare` first downloads the public Debian keyring package identified by the
existing base lock over verified HTTPS. It checks the package hash, reads the
single regular keyring member without installing the package, and checks the
keyring's independently locked hash. `--keyring` defaults to
`$NCORE_OCI_ROOT/keyring/debian-archive-keyring.gpg`; CI passes that path explicitly.
Build, check and publish require the prepared owner-only file and verify its
hash before their expensive work. No ambient Ubuntu keyring is used. The
existing signed Debian metadata checks retain the same exact keyring hash.

`prepare` then invokes the existing pinned Go/Gitleaks and Aho-Corasick preparation
and real native integration checks. It downloads the locked source annex,
separate signed Debian metadata, and the two hash-pinned SkyPilot bootstrap
sources. It requires the `base_sources.py --metadata` integration; an older
helper fails rather than combining build metadata with delivered source.
`build` calls the existing committed-context `ncore/build.sh --oci-output`
with `--provenance=mode=max` and a digest-pinned BuildKit SBOM generator.
Provenance must name that exact generator in addition to the locked base;
unknown, duplicate or substituted materials are refused. The actual buildx metadata,
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
back into the image. A failed bootstrap preserves its internal APT diagnostics
in private stderr before the disposable container is removed. The log location
comes from the pinned upstream script, whose body and exit status are preserved.
No GPU workload is part of this prepublication command.

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
- The SLSA materials must include the exact Python base reference, linux/amd64
  platform and digest from the base lock, exactly once, and the digest-pinned
  SBOM generator declared by the committed build script. Only the frontend
  reference declared by the committed Dockerfile may accompany those materials.
  The frontend digest receives structural checks only: its tag is not an
  independent digest pin. This does not authenticate the producer or every
  network-fetched build input. The committed context hash,
  shipped-source comparison and separate source delivery checks provide their
  own bindings. The local build receipt is not a signed builder identity.
- Source authentication using the actual shipped lock, final-file inventory,
  corresponding-source annex, separate signed repository metadata and keyring.
  Private denylists never enter these public source inputs.

The shipped-source check requires the exact three-file `/opt/ncore/native/`
population, including the downloader from `npa/src/npa/_public_https.py`, plus
the shell launcher in `/opt/ncore/bin/native-bootstrap.sh`. It compares every
file with its real committed source path. Missing, additional,
duplicate, nonregular or changed native inputs fail. Payload layer-path scanning
retains the exact text-only Python `.pth` hook check. Separately,
`payload-history.json` records the existing payload classifier's evaluation of
every history entry from the original digest-bound config, including empty
layers; it records the classified entry count and fails on prohibited installs.
The tarball path scanner's empty history result is not history coverage.

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
  --keyring "$NCORE_OCI_ROOT/keyring/debian-archive-keyring.gpg" \
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

Only after every prepublication gate passes can a push or administrator handoff
occur. Already public packages continue directly to anonymous verification.
For a private or internal package, the command uses supported package GET and
fully paginated version GET requests, requires exactly the validated graph's
manifest versions with only the original index carrying this dev tag, then
refreshes the inventory and rechecks the tag. Missing versions, additional
versions or tags, malformed responses and inventory changes fail without a
handoff. The REST package resource has no supported visibility PATCH operation;
the command never attempts it. GitHub documents visibility changes through
[package-admin settings](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility#configuring-visibility-of-packages-for-an-organization),
separately from the [Packages REST API](https://docs.github.com/en/rest/packages/packages).

After those checks pass, a non-public package still ends with a typed
`administrator-handoff-required` failure and nonzero exit status. The private
`transfer/administrator-handoff.json` receipt binds the full source SHA, exact
immutable dev tag, index/platform/config digests, original archive hash, validated
graph and package inventory hash, and hashes of the required evidence files.
It contains no raw authentication, scanner policy or process output. The CLI
prints only strictly validated receipt/source/image identities and a fixed
instruction identifying organization `nebius`, container package
`nebius-physical-ai/npa-ncore`, and **Package settings → Danger Zone → Change
visibility → Public**. It does not print private receipt locations or API URLs.
The handoff is neither publication success nor release acceptance.

A package administrator must refresh the complete exact inventory immediately
before that UI action and compare it with the private handoff, stopping if
anything changed. Inventory reads cannot lock the package or authorize later
unvalidated versions. After the UI change, retry `publish` using the same
original OCI archive and build receipt with a new output directory. All
prepublication gates run again; an equal immutable tag is not copied. A fresh,
nondeterministic build from the same source SHA does not validate or replace the
existing tag: a different digest is refused.

Anonymous full graph, hash and byte verification remains mandatory after the
UI change. Readback explicitly disables credentials, downloads the entire
graph, compares index/platform/config
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
