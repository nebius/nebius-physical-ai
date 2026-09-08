# NCore CPU ingestion image

This image packages the official pinned NCore COLMAP converter and V4 reader.
It remains a quarantined development candidate; no image acceptance or GPU
readiness is implied. See [REDISTRIBUTION.md](REDISTRIBUTION.md) for the source,
application dependency and acceptance boundaries.

At container startup, the unprivileged user fetches two exact Debian native
libraries needed by real APT/curl. The fixed isolated root installer verifies
those bytes before installing their original library paths and SONAME links.
Both the normal entrypoint and SkyPilot's Bash override run this setup before
APT. Network, hash, ownership or path failures stop startup.

| Setting | Default | Requirement |
| --- | --- | --- |
| `NPA_NCORE_NATIVE_CACHE` | `${XDG_CACHE_HOME:-<account-home>/.cache}/npa/ncore/native` | Absolute worker-owned `0700` POSIX directory; files/lock are `0600` |
| `BASH_ENV` | `/opt/ncore/bin/native-bootstrap.sh` in the final stage only | Preserve for SkyPilot's `/bin/bash -c` override so native setup runs before APT |

For an explicitly mounted private native cache, set
`NPA_NCORE_NATIVE_CACHE=/workspace/native-cache` and make that directory owned by
the worker with mode `0700`. It must support `flock` and atomic rename. The default
is ephemeral; the image provisions no persistent storage. Both complete `.deb`
artifacts and selected members are hash-checked on warm startup. Unsafe or corrupt
existing entries fail closed; remove the affected cache entry before retrying.
Downloaded bytes must never enter a Docker context, final build input or layer.

The separate application dependency cache uses `NPA_NCORE_RUNTIME_CACHE`; its
existing precedence and pip hash lock are described in
[REDISTRIBUTION.md](REDISTRIBUTION.md#runtime-cache). Native setup does not install
application wheels or invoke package scripts. It supports the published non-root
user and needs the existing passwordless sudo bootstrap contract.

[BASE-SOURCES.md](BASE-SOURCES.md#native-startup-delivery) records the exact native
selection, retained source inventory, private cache and privileged installation
checks, and outstanding exact-image/bootstrap acceptance gates.

## Selected-base publication evidence

NCore ships selected loose Debian files and an empty dpkg database. The
supplemental `npa.deploy.ncore_selected_sbom` scan verifies those files against
the lock **inside the final merged filesystem export**. It checks selected
regular-file SHA256, exact symlinks, delivered source/notices and Debian 12
identity, then emits SPDX 2.3 with actual SHA1 checksums. Package identities
retain binary and source versions (including epochs), architecture and Debian
purls. Partial-package scope is explicit; no installed dpkg state or package
license conclusion is invented.

After independently verifying the final OCI index, sole linux/amd64 child and
config, derive a merged filesystem tar from that exact platform and run:

```bash
npa/.venv/bin/python -m npa.deploy.ncore_selected_sbom \
  --rootfs-tar "$NCORE_FINAL_ROOTFS_TAR" \
  --image-digest "$NCORE_OCI_DIGEST" \
  --platform-digest "$NCORE_AMD64_DIGEST" \
  --config-digest "$NCORE_CONFIG_DIGEST" \
  --scan-output "$NCORE_SELECTED_EVIDENCE_DIR"
```

`--scan-output` must be a new directory. It retains `selected.spdx.json`, raw
`selected.trivy.json`, and, only on success, `selected.receipt.json`. The receipt
binds the supplied OCI identities, merged-tar hash, shipped-lock hash, SPDX and
Trivy hashes, exact package-set hash and observed counts. `files_verified` counts
regular Debian selection/source/notice files excluding the lock; exact symlinks
are counted separately. Selection counts follow the shipped lock and can shrink
when files move to runtime delivery. Build-only repository metadata is excluded
only when the lock identifies it as signed repository input, never required source.

The equivalent API is `scan_archive(rootfs_tar, new_evidence_directory,
image_digest=..., platform_digest=..., config_digest=...)`. The caller must prove
that the merged tar was derived from the verified OCI graph and retain that
derivation alongside the original OCI artifact. Supplying digests alone does
not establish the relationship. This API supports the local pre-publication
gate; its workflow/build-script integration is separate. `publish_public`
rechecks the accepted index/platform/config and exports that exact platform
before repeating this supplemental scan at promotion and release recheck.

Trivy uses `--list-all-pkgs`; its exact binary/source/version/architecture set
must equal the selected SPDX inventory. Fixed CRITICAL findings and secrets
refuse the gate; unfixed CRITICAL counts remain visible. The scan uses an empty
configuration/ignore file and removes ambient `TRIVY_*` filters. The existing
image vulnerability/secret scan is still required. This supplemental inventory
does not cover CPython, runtime-fetched dependencies or ancestor bytes; the
independent source, complete-byte, license, payload and bootstrap gates remain
required. See [BASE-SOURCES.md](BASE-SOURCES.md) for source delivery.

For a diagnostic inventory only, use `--lock base-source-lock.json --output
selected.spdx.json`. It omits unobserved SHA1/file-analysis claims and cannot
produce a passing scan receipt. Keep the supplemental SPDX separate from the
mandatory buildx SPDX predicate: the release parser refuses duplicate SPDX
predicates. The checked-in acceptance template remains unvalidated and NCore
remains quarantined until all exact-image and full-input functional evidence,
including native NRE completion and independently decoded RRDs, is accepted.
