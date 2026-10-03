# Whole-file Gitleaks helper

[Image scanning](../../../../docs/workbench/ncore-oci-publication.md) · [Repository](../../../../README.md)

This analysis tool is built outside container contexts. It calls the pinned
Gitleaks 8.28.0 `Detector.Detect` API on each complete raw record, including binary
and empty records. It does not use Gitleaks file discovery, MIME filtering,
stdin chunking, a baseline, or image-authored ignore files.

Distinct records are detected concurrently, with at most 64 workers, and results
are emitted in record order. Admission reserves payload bytes against a 512 MiB
budget before allocating them, so the bytes held for detection are bounded
independently of archive size. A record larger than that budget is admitted alone
up to the derived framed-record ceiling.

The helper has a 12 GiB address-space ceiling. Current measurements peak at
9.85x payload bytes for one record, so admission rounds that observation to 10x,
retains 4 GiB of fixed/process headroom, and derives a record ceiling of
858,993,459 bytes (about 819 MiB). The same formula covers the 512 MiB aggregate
payload budget used by concurrent smaller records. The measurement is not
treated as a permanent detector constant: `RLIMIT_AS` remains the hard,
fail-closed backstop if implementation drift uses more memory.

The archive scanner routes a larger regular-file record through an isolated
one-shot mode, with an independent 1 GiB complete-record ceiling. It prefers
`O_TMPFILE`; filesystems without it use an `O_EXCL` mode-`0600` name that is
unlinked immediately inside the private analysis directory. After every byte is
written, the scanner fsyncs and changes the inode to mode `0400`, reopens it
read-only through its descriptor, and closes the writer before launching either
worker. The helper maps that unlinked descriptor read-only, exposes the complete
mapping to the same single `Detector.Detect` call without a second Go payload
copy, and hashes and stats it before and after detection. Mapping, detector
allocation, mutation, unmap, process death, or receipt failure leaves the scan
incomplete. No chunked or overlap matching path exists.
The 1 GiB ceiling is a finite hostile-input admission bound, not a promise that
every admitted byte pattern will fit: the unchanged 12 GiB address-space limit
still rejects detector-memory exhaustion rather than accepting partial coverage.

## Cgroup memory awareness

Before loading the detector, the helper resolves its cgroup-v2 membership through
`/proc/self/cgroup` and `/proc/self/mountinfo`. For each readable visible ancestor
with finite `memory.max`, it subtracts that ancestor's `memory.current`, reserves
10% of the remaining allowance, and uses the smallest result as Go's
[`debug.SetMemoryLimit`](https://pkg.go.dev/runtime/debug#SetMemoryLimit). This
leaves room for the Python parent and other memory already charged to the group;
subtracting the helper's small startup footprint as well is conservative.
An already smaller runtime limit is preserved for directly launched helpers.

Unlimited, unavailable or non-v2 metadata leaves the runtime setting unchanged;
readable ancestor limits still apply when a descendant limit is unavailable.
A sample with no headroom, or no more allowance than the helper already owns, is
skipped: `memory.current` includes reclaimable page cache, so transient pressure
must not prevent scanning. Usable limits from other ancestors and an already
stricter runtime limit remain effective. A usage file disappearing with ENOENT
or ENODEV is also skipped. Malformed finite-limit or usage data and other usage
read failures produce a controlled error before detection. Values above Go's
signed addressable range are treated as unlimited. The policy is sampled at
startup: hidden ancestors, later limit changes and competing allocations cannot
be predicted.

This is a **soft limit on Go-managed memory**, not a process-memory ceiling or an
OOM guarantee. The [Go GC guide](https://go.dev/doc/gc-guide#Memory_limit) explains
that the runtime can exceed the limit to preserve progress. Detector working
memory and a single oversized record can still exhaust the cgroup; no record is
truncated or skipped, and the existing payload admission policy is unchanged.
If the operating system terminates the helper, the scan fails and cannot
establish a clean result.

The address-space ceiling and complete-record admission checks described above
remain independent hard limits. Cgroup sizing does not weaken those checks.

## Prepare the helper

Run on **Linux amd64 with kernel 5.9 or newer**, from an NPA checkout with its
development environment installed. Linux 5.9 is required because the
built-binary TSYNC proof binds the pre-existing thread's `Seccomp_filters`
counter. The bootstrap downloads its pinned Go toolchain and modules; a local Go
installation is not required. macOS, ARM, and older kernels are rejected by this
native preparation path.

Choose a private analysis directory outside the checkout. Replace the three
absolute paths below before running. The bootstrap runs native regression tests
before writing a terminal dependency receipt:

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
runs all Go tests, then executes the built helper's production containment
startup probe. The probe applies the same installation call as normal scans to
an already-running locked thread, checks every denied syscall route, and launches
`/proc/self/exe` to prove that a fresh child inherits the policy. Hermetic
bootstrap tests are collected by normal repository CI and can also run separately:

```bash
npa/.venv/bin/python -m pytest npa/tests/docker/test_image_byte_go_build.py -q
```

## What successful preparation produces

The bootstrap pins [Go 1.27.1](https://go.dev/dl/) Linux amd64 to SHA-256
`63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445`.
It uses isolated module, compilation and temporary caches, disables automatic
Go toolchain switching and workspace discovery, verifies the locked module
checksums, and builds a source snapshot without changing `go.mod` or `go.sum`.
The output receipt binds the exact source, trusted `.gitleaks.toml`, binary,
raw readiness JSON, successful built-binary containment probe, toolchain,
downloaded module closure, notices and tests.
The bootstrap holds the built binary and config descriptors across both probes
and hashes those same held bytes for the receipt; a path replacement or in-place
mutation fails preparation.
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

Normal framed mode optionally accepts `--ordinal-base N` so a restarted helper
retains globally controlled synthetic record paths. One-shot mode requires
`--config-fd`, `--record-fd`, `--record-length`, and `--record-ordinal`
together. Its record descriptor must be an owner-only, unlinked, read-only
regular file of the exact declared length, no larger than 1,073,741,824 bytes.
It emits the same readiness, actual global ordinal, result, summary, and
exit-status contract as a one-record framed session.

The helper emits one readiness JSON line, then consumes an unsigned 64-bit
big-endian byte length followed by exactly that many bytes, repeatedly. It emits
one JSON result per record, returning only rule and line information, record
ordinal, byte count and SHA-256. More than 4,096 findings in one record fails the
scan before response population; findings are never silently discarded. Clean
EOF between records produces a final summary. A truncated header or payload is
an error. The scanner process uses these exit codes:

The caller may keep several records in flight; results are emitted strictly in
record order, so the response bytes are identical at every depth. Both directions
are live at once, so a caller must continue reading results while it writes
records. A caller that blocks in a write without reading deadlocks against a
helper that has filled its output pipe and stopped reading, which is why
`core.Detector` transfers record bytes and collects results through one readiness
wait rather than draining only before each write.

Before reading configuration or records, the Linux amd64 helper installs an
inherited seccomp policy that forbids changing process group/session membership
and creating or joining namespaces. Readiness binds that containment policy.
`clone3` is reported unavailable so ordinary runtime thread creation falls back
to `clone`; namespace-bearing `clone` calls are still rejected.
The built-binary probe records the ambient seccomp-filter count on a locked
pre-existing thread and requires the production TSYNC installation to increase
that exact thread's count by one.
The caller sends `SIGKILL` to the isolated helper group after direct exit and
accepts a terminal result only after stdout reaches EOF and no live member of
that group remains. A missing policy, retained pipe, surviving member, or policy
installation failure rejects the scan rather than accepting incomplete cleanup.

The caller may keep several records in flight; results are emitted strictly in
record order, so the response bytes are identical at every depth. Both directions
are live at once, so a caller must continue reading results while it writes
records. A caller that blocks in a write without reading deadlocks against a
helper that has filled its output pipe and stopped reading, which is why
`core.Detector` transfers record bytes and collects results through one readiness
wait rather than draining only before each write.

| Code | Meaning |
| --- | --- |
| `0` | All records processed; no findings. |
| `1` | All complete records processed; findings retained. |
| `2` | Protocol, configuration, or I/O failure. | Failures never establish a
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
