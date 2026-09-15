# NCore non-Debian component coverage

[Workbench docs](README.md)

NCore's supplemental component gate is required **in addition to** the selected
Debian SPDX scan and ordinary Trivy vulnerability, secret and license scans.
It is specific to the selected NCore scratch image. It neither fabricates an
installed package database nor grants publication or release acceptance.

The implementation is `npa.deploy.ncore_component_scan.scan_archive`. The
publication command invokes it unconditionally before bootstrap/transfer, after
the OCI graph, source and selected-base gates. The host source checks include
the component modules. Its image-file mapping authenticates committed NPA
source and notices, exact generated revision/import-hook bytes, and the duplicate
Apache license against the separately authenticated upstream source.

## Reviewed population and required evaluation

The reviewed CPython 3.12.14 file selection contains these non-Debian components:

| Component | Identity and evaluation | License scope |
| --- | --- | --- |
| CPython and selected stdlib | 3.12.14, explicit Grype Python CPE | Core Python terms plus incorporated-code notices |
| Embedded Expat | 2.8.3, explicit Grype Expat CPE | Exact upstream Expat notice |
| Embedded libmpdec | Official 2.5.1 release; PSF-declared CPE plus bounded upstream release-note review | BSD-2-Clause notice |
| HACL* hash implementations | Upstream revision recorded by CPython's refresh script, OSV commit query | MIT notice |
| Embedded KaRaMeL runtime helpers, including F* arithmetic | HACL* snapshot mapped to KaRaMeL revision `95968326f0ca1d6f9056347496482d285e6a9f1e`; separate OSV commit query | Two Apache-2.0 headers plus delivered full terms |
| Embedded BLAKE2 | CPython snapshot mapped to the referenced upstream libb2 commit; OSV commit query | Reference-code and CPython-wrapper CC0 dedications plus delivered full terms |
| NVIDIA NCore | Exact source-lock commit, OSV commit query | Apache-2.0 notice |
| trueprice pycolmap | Exact source-lock commit, OSV commit query | MIT notice |
| NPA source and bootstraps | Exact reviewed commit, OSV commit query | NPA Apache-2.0 notice; existing adaptation/source checks remain required |

The reviewed profile binds CPython commit
`2abcf904b8dac8c999d2b3aac76681abb333798a` and the canonical 61-file ELF inventory
SHA256 `fd9c96ddcf94b6e8cc74d04e0d555bf6964c47f0117514def0cc8d7ac180e698`.
Its 665 selected interpreter/stdlib paths remain subject to exact hash/link
verification. Expat is 2.8.3, libmpdec remains 2.5.1, and the HACL* revision is
unchanged. The BLAKE2 snapshot identity follows the new CPython commit.

The earlier CPython 3.12.12 profile was refused for fixed CRITICAL
[CVE-2026-6100](https://cveawg.mitre.org/api/cve/CVE-2026-6100).
The PSF record identifies 3.12.14 as the fix for the 3.12 branch. That source
fact does not replace evaluating the new component versions and final image.
Fixed CRITICAL findings still refuse; no severity, fix-state or scanner filter
has changed.

The KaRaMeL source relationship and standalone query are now verified by a
required third bundled-source proof. This resolves the source-mapping gap;
**final-image acceptance remains required**. Each replacement image must pass
fresh component, selected-base, ordinary image, complete-byte, source, license
and functional gates. A source proof or empty advisory response cannot accept
an image.

There are nine code identities and twelve required license scopes. These are
separate populations: incorporated Python notices, the two KaRaMeL/F* notice
headers and the BLAKE2 wrapper dedication are not installed packages. Embedded
component file sets overlap their parent CPython/HACL* files; their counts must
not be added as if they were additional shipped bytes.

## Verified bundled source and advisory scopes

`npa.deploy.ncore_component_sources` downloads source archives as data into the
private evidence directory. Each archive must match its pinned SHA256 before
parsing. The proof records every vendored file, upstream path, both file hashes,
and the SHA256 of their exact unified diff, including unchanged files. It also
binds the complete CPython extension source tree, including wrappers. Source
attribution to the image depends on the reviewed ELF/build profile; downloading
source does not independently prove how a binary was compiled.

For **libmpdec**, the [official download page](https://www.bytereef.org/mpdecimal/download.html)
publishes the 2.5.1 archive hash
`9f9cd4c041f99b5c49ffb7b59d9f12d95b683d88585608aa56a6307667b2b21f`.
CPython's [pinned SPDX entry](https://github.com/python/cpython/blob/2abcf904b8dac8c999d2b3aac76681abb333798a/Misc/sbom.spdx.json)
independently declares that release URL, hash and
`cpe:2.3:a:bytereef:mpdecimal:2.5.1:*:*:*:*:*:*:*`. The gate verifies these
fields from the authenticated source archive before querying Grype. This is a
PSF-declared identity; independent NVD dictionary recognition is **not
established**, so zero matches cannot alone establish library coverage.

All 52 files below `Modules/_decimal/libmpdec/` map to that release, including
examples and literature; 49 are identical. The three differences are README
text, the ClangCL integer-width workaround in `mpdecimal.c`, and CPython's
configuration, visibility and inline-declaration adaptations of
`libmpdec/mpdecimal.h.in`. All 64 extension-tree files are bound. The complete
mapping hashes are retained in the inventory's `source_mapping` field and the
private source proof.

The gate additionally fetches the current [official upstream changelog](https://www.bytereef.org/mpdecimal/changelog.html)
and requires its bytes to match the reviewed snapshot through 4.0.1. The review
evaluates its entries against installed 2.5.1 and the vendored file scope:

- Earlier release fixes are included in 2.5.1.
- The later `z` format and transcendental status-flag changes are upstream
  features. The nonzero-status handling of `mpd_qset_*_exact` is classified by
  upstream as a reliability fix; it remains recorded as a known difference.
- The C++ input-validation changes concern unvendored `libmpdec++`; the
  remaining later changes concern upstream build/install code, tests and C++.

The snapshot SHA256 is
`00a5a2d83efd91425faea4fb14975acac94a3330ff6218b1462e0aa61d43b6d8`.
No security advisory is identified in this reviewed release-note scope. A
changed page or download failure refuses pending review, including changes
that merely alter formatting. This is a finite upstream review plus a real
CPE evaluation, not a claim of exhaustive CVE discovery or a verified upstream
Git identity. The same-spelled Music Player Daemon and distribution package
versions are not substitutes for this source library.

For **BLAKE2**, [CPython's update PR](https://github.com/python/cpython/pull/6286)
explicitly references [libb2 commit 620681a](https://github.com/BLAKE2/libb2/commit/620681a3b15c4d7239b9323b9da5ea208a959d3d).
Its codeload archive SHA256 is
`92e0566634e3fd89407e5d959e3022b04f97d23c9e2e1d253604b6a0b7bfde1f`.
All 14 `Modules/_blake2/impl/` files map to that commit's `src/` files; four
are identical. The finite differences cover CPython's symbol naming and
configuration, unaligned load/store copies, zeroing/compiler portability,
`memmove` in the finalizers, removal of an unused convenience wrapper and
whitespace. All 21 CPython extension-tree files are bound. The gate submits
the real OSV query `{"commit":"620681a3b15c4d7239b9323b9da5ea208a959d3d"}`
and consumes every page. Any returned advisory blocks pending review; no
claim is made that CPython ships all of libb2 or every upstream code path.

All three bundled source proofs require the full Python CPE evaluation for
the installed parent version, covering CPython's wrappers and adaptations under the existing
policy. KaRaMeL additionally requires the actual HACL* commit evaluation.
That shared parent scope is recorded explicitly and is never counted
as a separate unique library scan. The fixed-CRITICAL rule remains unchanged.

## Invocation and evidence

Call the Python function from the trusted publication coordinator:

```python
from npa.deploy.ncore_component_scan import scan_archive

scan_archive(
    rootfs_tar,
    new_private_evidence_directory,
    base_lock=committed_base_lock_bytes,
    source_lock=committed_upstream_lock_bytes,
    source_sha=reviewed_full_sha,
    committed_files=verified_image_path_to_committed_sha256,
    image_digest=verified_index_digest,
    platform_digest=verified_platform_digest,
    config_digest=verified_config_digest,
)
```

All parameters shown are required. `committed_files` must come from the
coordinator's authenticated, complete NPA source mapping, including the native
bootstrap/downloader, recipes, notices and compatibility aliases. It must not be
constructed by trusting image-reported hashes. Known generated source-revision
and import-hook bytes must be authenticated by the coordinator as well.

The module checks every non-directory rootfs member. Selected Debian files and
source annex artifacts retain their lock identities. CPython files retain their
lock hashes; the complete CPython ELF population must match the reviewed
embedded-component profile. Upstream source files must match their inventory
and committed source lock. All remaining files must be authenticated NPA files
or one of the finite assembly/venv paths. Unknown files, new distributions,
extra native payloads, changed locks and unreviewed interpreter binaries refuse.
The upstream stager/source-verification gate remains responsible for comparing
patched NCore/pycolmap source with the actual upstream archives.

By default, the gate downloads official Grype 0.118.0 and Trivy 0.72.0 Linux
amd64 archives, verifies their pinned SHA256 values and exact executable hashes,
and uses private executables, configuration and caches. Optional
`grype_executable` and `trivy_executable` arguments accept local files with the
same exact hashes. It does not execute the image. Tool commands, exits, stdout,
stderr, raw OSV request/response pages, database identity/hash, staged-license
origins and hashes, and each component's file-population hash stay in the
private evidence directory. HTTP/scanner failures fail closed.
Bundled evaluations also retain their exact mapping profiles and source-proof
hashes. KaRaMeL retains its source query and both required parent scopes;
libmpdec retains the exact upstream release-review dispositions. Missing or substituted proof,
query, parent evaluation or release review refuses.

Grype uses its supported [direct CPE scan interface](https://oss.anchore.com/docs/guides/vulnerability/scan-targets/).
The report must name that exact CPE, pinned scanner and valid database, with
CPE matching enabled and no applicable filters or suppressed results. All
severity/fix states are retained; a fixed CRITICAL blocks publication.

[OSV commit queries](https://google.github.io/osv.dev/post-v1-query/) run for
exact source commits and consume every response page. An empty response means
no known advisory for that query, not proof of source security. Any returned
source advisory blocks pending review because these source-only products have
no approved package-version/severity disposition. OSV advisory absence cannot
replace source guards or validate local patches. The
[PSF advisory database](https://github.com/psf/advisory-database) uses OSV Git
ranges; its contents are not a substitute for evaluating the affected commit.

Trivy additionally scans exact copies of delivered notices with `fs --scanners
license --license-full`. Copies use `notice/LICENSE` so Trivy recognizes
NPA's nonstandard notice filenames; their bytes and original image paths are
bound in `license-inputs.json`. Each short Apache-2.0 or CC0 notice is followed
by one LF and its complete, separately hash-pinned delivered terms. Every byte
of both inputs is retained and the combined scanner input is hashed. Full
terms never replace the actual copyright/dedication header. Every required
notice scope must have its expected license identifiers in a loose-file
analyzer result. Core Python terms cannot satisfy missing Expat, libmpdec,
HACL* or BLAKE2 notices. License identifiers
are scanner observations, not a legal determination or source-delivery waiver.

## Required CPython notice inputs

For the reviewed 3.12.14 profile, the following image paths must contain exact
bytes from [CPython commit 2abcf90](https://github.com/python/cpython/tree/2abcf904b8dac8c999d2b3aac76681abb333798a).
The implementation pins their SHA256 values. The base lock and Docker notice
copies specify delivery of all eight rows below; the final merged image must
prove those deliveries before acceptance.

| Image path relative to `/usr/share/doc/npa-ncore/cpython/` | Exact source |
| --- | --- |
| `LICENSE.third-party` | `Doc/license.rst`, unchanged |
| `LICENSE.expat` | Whole `Modules/expat/COPYING` file for Expat 2.8.3 |
| `LICENSE.mpdecimal` | First complete C comment in `Modules/_decimal/libmpdec/mpdecimal.h`, followed by one LF |
| `LICENSE.hacl` | First complete C comment in `Modules/_hacl/Hacl_Hash_SHA2.c`, followed by one LF |
| `LICENSE.hacl-krml` | First complete C comment in `Modules/_hacl/include/krml/lowstar_endianness.h`, followed by one LF |
| `LICENSE.hacl-fstar` | First complete C comment in `Modules/_hacl/include/krml/FStar_UInt128_Verified.h`, followed by one LF |
| `LICENSE.blake2` | First complete C comment in `Modules/_blake2/impl/blake2.h`, followed by one LF |
| `LICENSE.blake2-python` | First complete C comment in `Modules/_blake2/blake2module.c`, followed by one LF |

The existing `/usr/local/lib/python3.12/LICENSE.txt` must match CPython's `LICENSE`;
its hash is unchanged. All eight additional notices belong to the CPython
partition, even though selected-base delivery also verifies them. The complete
`/usr/share/common-licenses/Apache-2.0` and `CC0-1.0` files retain their Debian
partition and exact hashes. Both short KaRaMeL/F* headers require the former;
both BLAKE2 dedications require the latter. Missing, altered or unrecognized
notice/terms inputs refuse the license gate.

## HACL*, KaRaMeL and F* source boundary

CPython's pinned [refresh script](https://github.com/python/cpython/blob/2abcf904b8dac8c999d2b3aac76681abb333798a/Modules/_hacl/refresh.sh)
copies four hash implementations from HACL* and five helper headers from its
[vendored KaRaMeL tree](https://github.com/hacl-star/hacl-star/tree/bb3d0dc8d9d15a5cd51094d5b69e70aa09005ff0/dist/karamel).
The helpers are `lowstar_endianness.h`, `internal/target.h`,
`FStar_UInt_8_16_32_64.h`, `fstar_uint128_struct_endianness.h` and
`FStar_UInt128_Verified.h`. The script adjusts include paths, removes unused
external declarations/macros, and generates the small `krml/types.h` wrapper.
The pinned HACL* [INFO.txt](https://github.com/hacl-star/hacl-star/blob/bb3d0dc8d9d15a5cd51094d5b69e70aa09005ff0/dist/gcc-compatible/INFO.txt)
records KaRaMeL revision
`95968326f0ca1d6f9056347496482d285e6a9f1e`. Its
[Makefile](https://github.com/hacl-star/hacl-star/blob/bb3d0dc8d9d15a5cd51094d5b69e70aa09005ff0/Makefile)
records the KaRaMeL checkout revision and copies the runtime directories into
`dist/karamel`. The verifier authenticates both source records and verifies
that all 21 vendored files match that
[recorded KaRaMeL revision](https://github.com/FStarLang/karamel/tree/95968326f0ca1d6f9056347496482d285e6a9f1e)
byte for byte. The tree contains ordinary files, not a gitlink; this proof
establishes a recorded revision with identical runtime bytes, without claiming
the headers uniquely identify that commit or originated at its date.

`npa.deploy.ncore_karamel_source.verify_karamel_source` checks all three pinned
CPython/HACL*/KaRaMeL archives, the exact INFO/Makefile/refresh records, the
complete HACL* vendor tree, all five header transformations and the six-file
CPython helper directory including local `types.h`. Two upstream headers are
unchanged; three have the reviewed include rewrites and declaration/macro
removals. Every original, delivered-source and exact unified-diff hash is
bound. Archives are read as data; no upstream scripts or compiler are run.
The already fetched CPython archive is reused and checked against the same
fixed hash before parsing.

The inventory attaches a copy of the exact KaRaMeL profile to the actual
`karamel-runtime` component. It includes every archive URL/root/hash,
source-record hash, header-mapping hash, whole vendor/local-tree hash, the
standalone query and both parent scopes. The retained source proof's exact
JSON SHA256 is
`362fa6f8dfa800e7145d8b9c3efe0b74cc0ad018f5a1bce842afa54ef8669c98`;
a missing or substituted proof, profile or query refuses. The publication
source closure includes the new verifier and both component/KaRaMeL test
modules. Its committed-snapshot runner requires complete collection and
passing setup, call and teardown reports for every mandatory test.

The [build selection](https://github.com/python/cpython/blob/2abcf904b8dac8c999d2b3aac76681abb333798a/Modules/Setup.stdlib.in)
compiles HACL* into `_md5`, `_sha1`, `_sha2` and `_sha3`.
Its [SHA2 implementation](https://github.com/python/cpython/blob/2abcf904b8dac8c999d2b3aac76681abb333798a/Modules/_hacl/Hacl_Hash_SHA2.c)
uses the copied inline F* 128-bit arithmetic and endian helpers. This supports
an embedded `karamel-runtime` advisory component bound to those four extension
files. The recorded version denotes the HACL* vendored snapshot, not a claimed
KaRaMeL release. Its standalone OSV query is
`{"commit":"95968326f0ca1d6f9056347496482d285e6a9f1e"}`. The evaluation must also
bind `parent_advisory_scope` to the actual CPython 3.12.14 version/CPE evaluation
and `hacl_advisory_scope` to the actual HACL* commit evaluation. Neither parent
query can substitute for the standalone KaRaMeL query. Every returned OSV
advisory still blocks; zero findings describe only that database query's scope.
The F* compiler and KaRaMeL translator are not selected runtime components;
no compiler package is synthesized from a copyright header. Attribution is
based on authenticated source/build inputs and exact ELF hashes, not on
symbol-level recovery from stripped binaries.

The current profile is intentionally fixed at CPython 3.12.14. A replacement
must update the parent version/CPE, source commit/archive hash, actual ELF
inventory, wrapper/source mapping hashes and notices together after review.
The gate checks that the Python file lock agrees with the pinned ELF hashes
and that the observed ELF population agrees with that profile. Unchanged
library versions alone do not authorize a new interpreter's bytes.

## Limits

This gate binds a merged rootfs to OCI identities supplied by the coordinator;
it does not independently prove the rootfs's graph derivation. Original config,
ordered layers, ancestor contents, archive bytes, private policies, secrets,
source authenticity, restricted payloads, quarantine and publication refusal
remain governed by the existing mandatory gates. Runtime-fetched dependencies
are absent from the baked population and require their existing separate
runtime checks. No image, registry or runtime acceptance follows from this
module's unit tests or a standalone scanner exit zero.
