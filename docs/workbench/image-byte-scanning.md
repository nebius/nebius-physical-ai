# Complete-byte checks for cuRobo images

The trusted cuRobo publication workflow adds an archive byte scan before pushing
an image and after pulling its exact digest. It retains the existing Trivy,
license, runtime, provenance, SBOM, and payload checks. Other workbench images
keep their existing publication path.

The additional check addresses exclusions in file-oriented scanners. For example,
the pinned [Trivy secret analyzer](https://github.com/aquasecurity/trivy/blob/v0.70.0/pkg/fanal/analyzer/secret/secret.go)
filters some inputs before detection. The pinned
[Gitleaks detector](https://github.com/gitleaks/gitleaks/blob/v8.28.0/detect/detect.go)
also supports path, size, and inline suppression policies. The NPA bridge calls
the detector directly on each complete record, with explicit policy receipts and
image-authored suppression disabled.

The scanner verifies the saved-image hash, OCI config, ordered layer identities,
and file/byte counts against cuRobo's payload verifier. It reads every ancestor
layer, including files subsequently deleted, plus raw archive headers, logical
names, optional gzip headers, and padding. It never extracts image members into
the host filesystem. Unsupported layer encodings or ambiguous archives fail.

## Policy and dependencies

CI requires `CUSTOMER_DENYLIST` from the repository secret store;
`INFRA_DENYLIST` is optional. An early policy check runs before image construction.
The regex policy preserves the existing case rules and one search per text line,
then adds whole-record matches. UTF-8 with `surrogateescape` retains invalid bytes
and maps findings back to exact byte offsets. Missing or invalid required policy
fails. Local exact-literal inventories are a separate, explicitly selected policy;
they do not claim equivalence with CI's regex secrets.

The Go detector and CPython 3.12 native literal matcher use pinned dependencies.
Build their receipts with `go_helper/build.py` and `prepare.py dependencies` under
`npa/scripts/image_byte_scan/`. Both take `--analysis-root`, `--trusted-root`, and
`--output-dir`. Keep the analysis directory owner-only and outside the checkout;
scanner binaries, caches, and private policies must stay outside the image build
context. Standard pytest uses synthetic protocol fixtures without downloading
tools. The separate native CI job executes the actual detector and matcher and
fails when either is missing.

Before launching the detector, the scanner copies the verified helper and
configuration into sealed Linux memory files and checks their hashes again.
The child uses these immutable copies. Original-file checks remain in place to
detect changes to the authorized inputs; a later refusal alone cannot prevent
changed executable bytes from running.

## Run and inspect

Use `prepare.py authorize --help` to prepare an authorization from the exact
archive, successful cuRobo verifier report, expected image identity, and native
dependency receipts. For CI policy mode, provide the denylist values through the
private process environment, never command arguments. Authorization snapshots
and binds the policy, trusted source, and input hashes.

Run with the repository interpreter and an unused output directory:

```bash
npa/.venv/bin/python npa/scripts/scan_image_bytes.py \
  --analysis-root "$SCAN_ROOT" --trusted-root "$PWD" \
  --authorization "$SCAN_AUTHORIZATION" \
  --output-dir "$SCAN_ROOT/result"
```

A passing receipt requires complete accounting, unchanged inputs, joined native
children, configured confidentiality policy, and no unresolved findings. Review
the report and ordinal-based findings together. Reports do not contain matching
bytes or member names; retain the archive privately when investigating findings.
A failed scan stops publication and keeps its inputs for investigation. Hosted
CI retains those private inputs only for the runner's lifetime; preserve needed
evidence in authorized private storage before disposing of a persistent runner.
Never upload a failed image or policy file as a public CI artifact.

Reading all bytes is not a proof that every possible secret is recognizable.
Image-layer gzip is decoded; archives contained inside ordinary files are scanned
as stored. Encrypted content and unsupported semantic encodings remain limits of
pattern detection. Whole-record regexes can also require substantial memory or
CPU. An interrupted or exhausted scan is incomplete, with no truncation fallback.
Functional GPU workloads and artifact inspection remain separate release gates.
