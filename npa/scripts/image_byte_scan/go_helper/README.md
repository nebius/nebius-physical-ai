# Whole-file Gitleaks helper

This analysis tool is built outside container contexts. It calls the pinned
Gitleaks 8.28.0 `Detector.Detect` API on each complete raw record, including binary
and empty records. It does not use Gitleaks file discovery, MIME filtering,
stdin chunking, a baseline, or image-authored ignore files.

An explicit Linux amd64 bootstrap runs native regression tests before writing a
terminal dependency receipt:

```bash
npa/.venv/bin/python npa/scripts/image_byte_scan/go_helper/build.py \
  --analysis-root /absolute/private-analysis \
  --trusted-root /absolute/trusted-checkout \
  --output-dir /absolute/private-analysis/tools
```

The analysis directory must be outside the checkout and owner-only. Each output
directory permits one preparation. Failures retain logs without a success
receipt; retry in a fresh output directory. `--toolchain-archive` accepts an
already downloaded archive only when its complete SHA-256 matches the same pin.
The normal Python test suite never downloads or invokes Go. The explicit command
runs all Go tests, including real detector, process and inherited-descriptor
canaries. Hermetic bootstrap tests are collected by normal repository CI and can also run separately:

```bash
npa/.venv/bin/python -m pytest npa/tests/docker/test_image_byte_go_build.py -q
```

The bootstrap pins [Go 1.27.1](https://go.dev/dl/) Linux amd64 to SHA-256
`63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445`.
It uses isolated module, compilation and temporary caches, disables automatic
Go toolchain switching and workspace discovery, verifies the locked module
checksums, and builds a source snapshot without changing `go.mod` or `go.sum`.
The output receipt binds the exact source, trusted `.gitleaks.toml`, binary,
raw readiness JSON, toolchain, downloaded module closure, notices and tests.
All path components are opened through descriptors without following symlinks;
parent traversal is rejected before normalization. Cancellation stops and joins
the command session owned by the bootstrap, including children that ignore
termination. Native test JSON must account for every declared test with zero
skips; the module response must match the complete requested checksum set.
Outputs include `whole-file-scanner`, `helper-ready.json`,
`gitleaks-config.toml`, `dependency-receipt.json`, and `licenses-go/`.
Python literal matching is prepared separately by the archive scanner.

## Protocol and policy

Launch with exactly one of `--config PATH` or `--config-fd FD`. The descriptor
must be inherited, regular, seekable, at offset zero, and greater than 2. The
helper checks configuration metadata before and after its complete read.

The helper emits one readiness JSON line, then consumes an unsigned 64-bit
big-endian byte length followed by exactly that many bytes, repeatedly. It emits
one JSON result per record, retaining every finding but returning only rule and
line information, record ordinal, byte count and SHA-256. Clean EOF between
records produces a final summary. A truncated header or payload is an error.
Exit 0 means no findings, 1 means findings after processing all complete records,
and 2 means a protocol, configuration or IO failure. Failures never establish a
clean scan. The caller separately records exact coverage and handles resource
exhaustion as failure.

All embedded default rules and trusted repository additions remain active.
`MaxTargetMegaBytes=0`, inline `gitleaks:allow` suppression is disabled, and
matching plaintext is never emitted. Readiness binds deterministic before/after
policy hashes. The only policy change removes path prerequisites from these
four content rules, leaving their content expressions and other settings intact:

- `freemius-secret-key`: `(?i)\.php$`
- `hashicorp-tf-password`: `(?i)\.(?:tf|hcl)$`
- `kubernetes-secret-yaml`: `(?i)\.ya?ml$`
- `nuget-config-password`: `(?i)nuget\.config$`

This makes those content checks apply to extensionless records too. Unknown
path rules fail closed. The path-only `pkcs12-file` selector
`(?i)(?:^|\/)[^\/]+\.p(?:12|fx)$` is reported in readiness and must be enforced by
the archive scanner against every actual logical image path. Controlled ordinal
paths in this helper cannot activate repository path allowlists.

## Licensing and provenance

The helper's NPA source is covered by the repository license. Gitleaks 8.28.0 is
[MIT licensed](https://github.com/gitleaks/gitleaks/blob/v8.28.0/LICENSE); its exact
notice is retained as `LICENSE-GITLEAKS`. The Go toolchain carries the BSD-style
notice in `LICENSE-GO`, taken from the pinned official archive. The pinned module
sum is `h1:XXeibrt4XbdrYm3FnzXR3uUPs9HbgGduroICjBl6PMw=`. `go.sum` binds the
transitive dependency closure. The bootstrap copies each downloaded module's
exact license, copying, copyright and notice files into the external tool output
and binds them in its receipt; missing notices fail preparation. These tools,
modules, caches and notices belong to the analysis host, never the scanned image.
There are no model weights or datasets in this helper.
