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

## Review specific findings locally

An exact match may be a public cryptographic self-test or an inert source
example. Public provenance alone does not establish that it is safe: a working
bearer token remains a credential even if upstream published it in an example.
Remove such unnecessary material before the installation layer commits and
rebuild; never approve it merely because the surrounding code is inactive.

`image_byte_scan/adjudicate.py` verifies an explicit independent review of a
complete scan. The original report and findings stay unchanged, including its
`valid: false` verdict. A separate acceptance receipt requires every native,
regex and literal occurrence, including identical repeated detections, to have
an independently reviewed proof. Missing, new, duplicate, unreviewed or changed
occurrences fail. Structural archive findings cannot be accepted by this
protocol.

Each proof binds the exact record bytes, an explicit non-operational content
role, retained provenance evidence and a separate semantic review. The review
must investigate actual credential use; a role label, filename or origin URL is
not proof. Keep manifests, occurrence proofs, reviews and their evidence in an
owner-only directory outside Git. Obtain the exact manifest and review hashes
from the independent review's accepted result. These hashes authorize particular
bytes; they are not reviewer signatures. Never calculate approval inputs
automatically from an unreviewed bundle.

```bash
npa/.venv/bin/python npa/scripts/image_byte_scan/adjudicate.py \
  --analysis-root "$SCAN_ROOT" --trusted-root "$PWD" \
  --authorization "$SCAN_AUTHORIZATION" \
  --report "$SCAN_REPORT" --records "$SCAN_RECORDS" \
  --manifest "$REVIEWED_MANIFEST" --manifest-sha256 "$REVIEWED_MANIFEST_SHA256" \
  --review "$INDEPENDENT_REVIEW" --review-sha256 "$INDEPENDENT_REVIEW_SHA256" \
  --output-dir "$SCAN_ROOT/accepted-review"
```

The command rechecks the archive, OCI config, scanner source, toolchain, policy,
raw report, complete ledger and every evidence binding. It distinguishes the
image's committed source revision from the committed scanner revision; changing
either requires the corresponding new evidence. Its input format is a Docker
save archive accepted by the scanner, including its `manifest.json`, rather than
an arbitrary OCI directory. A source change after scanning requires a new scan;
editing hashes in an old report is not a substitute.

This local review does not publish an image, authorize a registry operation, or
replace licensing, vulnerability, SBOM/provenance, runtime or physical-GPU
validation. The hosted publication workflow continues to require zero raw
findings by default. It has no implicit access to an operator's private review
bundle. Do not upload that bundle as a public Actions artifact or add private
evidence to Git to transport it.

## Reviewed public native content

The optional public content policy is distinct from private occurrence review.
A trusted scanner invocation may provide `--public-native-policy PATH` and
`--public-native-policy-sha256 SHA256` together. The expected hash must come from
independently reviewed publishing configuration. Computing it from an unreviewed
catalog does not establish authorization. The catalog and every referenced proof
must belong to the exact committed scanner source closure.

The scanner performs a fresh scan in the same invocation, binds the exact emitted
ledger, and conserves every native occurrence, including duplicate coordinates
and identical files in ancestor layers. Entries bind the record kind, complete
byte hash and size, pinned detector identity, exact native finding multiset, and
independently reviewed public provenance and semantic evidence. An upstream file
name, package name or public origin does not establish noncredential semantics.
Typed path metadata requires an explicit exact-byte proof of its own role.

Literal and confidentiality-regex findings always refuse this public policy.
Structural findings, unknown bytes, changed native populations, incomplete scans,
changed inputs or unreviewed evidence also refuse. The raw report retains its
original verdict; a successful separate `public-policy-acceptance.json` receipt
binds the actual image source, scanner source, archive, config, manifest, policy,
proofs, authorization, ledger and raw report. This receipt alone does not prove
other security gates, redistribution eligibility or a real GPU workload.

Python bytecode entries require whole-file equality, including every header byte.
For cuRobo, the supported build paths pass the exact source commit epoch as a
build-only argument so pinned CPython emits checked-hash bytecode. Source and
compiler provenance, full marshal structure and actual layer readback remain
required before reviewing a catalog entry; header masking is never permitted.

The trusted cuRobo publishing workflow pins the reviewed catalog hash in its
build job and passes that same pin to both the local-image and pushed-digest
complete scans. It obtains no approval hash from an unreviewed file at runtime.
A failed policy gate stops subsequent publication actions and retains failed
inputs for private investigation. Confidentiality configuration remains required
before building; exact public native content cannot authorize denied identifiers.
