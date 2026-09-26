# Workbench performance and reliability improvements: validation (2026-09-19)

[Workbench docs](README.md)

## Integration onto current main

The timing measurements in the later sections record the original campaign.
The PR was subsequently extracted onto main at `8c9c2909b`, preserving the
newer S3 URI validation, `require_empty` upload preflight, bucket-root prefix
handling, and portable object keys. The extracted source at `0365b774e`
passed 334 focused tests (80 optional skips), 3,869 guardrails, and all 7
committed live credential, manifest-discovery, and transfer tests against
fresh Nebius storage. Independent Claude Code review found no actionable
issue. All required Linux CI checks passed, including five Python test
shards, merged coverage, security, browser/compatibility, and repository gates.

Fresh GPU validation also exercised that extracted source, with import paths
and SHA-256 hashes checked for all five changed production modules. Both
B200 and RTX PRO 6000 performed a real CUDA matrix multiplication, 16 SGD
training steps, checkpoint publication/download, `weights_only=True` loading
into a fresh model, and 8 resumed steps with decreasing loss. All 37 output
artifacts per device matched their hashes after S3 round trips. Independent
operator-side NumPy checks recomputed the matrix product, checkpoint
predictions, and resumed loss from the generated tensor storage bytes.
Exact checkpoint downloads retained mode `0600`; manifest discovery and
credential preflight also ran from the staged candidate source.

RTX completed successfully. B200 deliberately exited nonzero **after**
publishing its verified results and a distinguishing failure marker, providing
a real failed-job fixture for the separate runtime-diagnostics change. The
expected terminal failure was checked together with that marker and every
artifact; it was not treated as an ordinary successful job. This validates
synthetic tensor training and shared Workbench I/O, not a Genesis simulation
or a Cosmos/model-specific training workload. It does not refresh the timing
measurements below or establish an overall 5x speedup.

The attempted full macOS base/candidate comparison was invalidated by host
disk exhaustion. Its failure counts are not evidence of a pass or regression;
the supported Linux CI run supplies the complete-suite result for the
extracted source. Concrete infrastructure identities, credentials, and raw
operator evidence remain in access-controlled external records.

Scope: three shared, high-frequency code paths in `npa` — credential
preflight, S3 directory transfer, and workflow-run manifest discovery —
changed to run independent I/O concurrently instead of serially. Base commit
`41130c9d9a913fd88d8304fbf17c78468f6b858d`. All real-infrastructure numbers
below were measured against a dedicated, isolated validation project with
its own writable object storage; concrete resource identifiers are kept out
of this document per the repository's confidentiality conventions and are
retained in access-controlled private evidence.

**Method note on the speedup numbers:** every before/after timing
measurement in this document (credential preflight, manifest discovery,
storage transfer) was run from the operator's local machine against
remote Nebius endpoints in one region — not measured from inside a GPU
worker or from a machine co-located with the service. Absolute latencies
therefore reflect that specific client-to-remote-endpoint topology and the
tested workload shapes; the relative speedups (concurrent vs. serial, same
client, same network path each trial) are the reproducible claim, not the
absolute seconds in isolation.

## Summary

| Target | What changed | Real measured result |
|---|---|---|
| Credential preflight (`npa workbench health preflight`) | 5 independent provider checks (Hugging Face, NGC, S3, Token Factory, Nebius CLI) now run in a bounded thread pool instead of serially | **2.70x** median speedup over 5 alternated trials, all 5 checks PASS against real providers both before and after |
| Workflow-run manifest discovery (`workflow list` / `status` / `artifacts` without an explicit S3 URI) | Candidate run manifests below a bucket prefix are fetched concurrently through one shared client instead of one client built per candidate | **11.88x** (`list_runs`) / **12.53x** (`discover_workflow_run_state`) median speedup over 3 alternated trials against 40 real seeded manifests; identical results verified between the old and new code every trial |
| S3 directory transfer (`upload_directory` / `download_directory` / `download_path`) | Per-file transfers run concurrently (bounded pool) with an adaptive per-file multipart thread budget | **6.2x–7.2x** for many small files, **2.4x–3.0x** for a realistic mixed checkpoint directory; a disclosed, workload-dependent tradeoff for one large file alone in a directory, from ~7% slower (fresh connection) to ~2.1x slower (after prior small-file traffic on the same client) — see below — **no blanket 5x claim for storage** |

Workflow-run manifest discovery is the strongest real result (11.9x-12.5x).
Credential preflight measured 2.70x. The storage result is reported
honestly as mixed: strong for the common multi-file case, a disclosed
workload-dependent cost for one narrow case.

## 1. Credential preflight

`npa/src/npa/workflows/credential_preflight.py` (`run_credential_preflight`)
and `npa/src/npa/workflows/sim2real_health.py` (`run_preflight`) previously
ran each selected check as a plain serial list comprehension. Each check is
an independent network or subprocess probe (Hugging Face identity, NGC
token exchange, one S3 list call, Token Factory model list, and a Nebius
CLI identity + IAM-token mint), so serial execution paid the sum of every
check's latency. Both now run their selected checks through a shared
bounded-thread-pool helper (`run_checks_concurrently`, added to
`sim2real_health.py`), preserving result order and per-check error
isolation: all independent probes still run to completion even after one
fails (unlike the old serial version, which stopped launching further
checks the moment one raised), and the earliest-ordered check's exception
is the one re-raised to the caller, only after every probe has finished.

The credential-preflight worker pool is capped at the number of distinct
supported checks (5), independent of how many times a caller repeats a
check name, so a caller passing duplicate names cannot scale the thread
count unboundedly. The Sim2Real health-check runner needed no such cap: its
6 checks are fixed and each selectable at most once.

**Real evidence** (5 alternated trials — which source ran first was flipped
each trial so neither side systematically benefited from running warm — all
5 checks against real Hugging Face, NGC, S3, Token Factory, and Nebius CLI
providers, every trial PASS on both sides):

- Base median: 1.77s. Candidate median: 0.66s. **2.70x.**

**Tests:** deterministic `threading.Barrier`/`Event`-based proofs (no
sleep-based timing assertions, which are flaky under parallel test load) in
`npa/tests/workflows/test_credential_preflight.py` and
`npa/tests/workflows/test_sim2real_health.py` — concurrent execution, result
ordering independent of completion order, first-ordered-exception
propagation, and worker-count capping under repeated check names. Committed
gated live coverage in `npa/tests/e2e/test_credential_preflight_live.py`
(run with `NPA_INTEGRATION_E2E=1` and `NPA_E2E_PROJECT=<alias>`).

## 2. Workflow-run manifest discovery

`npa/src/npa/orchestration/skypilot/workflow_state.py`'s `list_runs` and
`discover_workflow_run_state` (used by `workflow list`/`status`/`artifacts`
whenever no explicit `--workflow-s3-uri` is given) previously built one
boto3 client per candidate run manifest — constructing that many clients
through boto3's shared default session concurrently is a documented hazard,
so parallelizing this loop required precreating and sharing one client per
listing operation first. Candidate manifests are now fetched concurrently
in bounded, page-sized batches through that one shared client;
`list_runs`'s early exit once its `limit` is reached still respects listing
order at batch granularity, and `discover_workflow_run_state`'s
newest-copy tie-break is unchanged (a retried/migrated run can leave more
than one durable copy, so every candidate is still read before the newest
is picked).

**Real evidence** (3 alternated trials against 40 real, seeded durable run
manifests in a dedicated bucket prefix; both the returned run-id set and the
discovered run's prefix were verified equal to the expected seeded set for
both the old and new code on every trial, not merely equal to each other):

- `list_runs`: base median 5.48s, candidate median 0.46s. **11.88x.**
- `discover_workflow_run_state`: base median 5.46s, candidate median 0.44s.
  **12.53x.**

Fixture manifests were deleted and their absence verified after every run.

**Tests:** `npa/tests/orchestration/skypilot/test_workflow_state_manifests.py`
covers single-shared-client-per-operation, filtering/sort correctness,
early-exit bound, newest-copy tie-breaking, response-body closing, and a
deterministic Barrier-based concurrency proof. Committed gated live
coverage in `npa/tests/e2e/test_workflow_manifest_discovery_live.py`.

## 3. S3 directory transfer

`npa/src/npa/clients/storage.py`'s `upload_directory`, `download_directory`,
and `download_path` previously transferred files one at a time. Files now
transfer through a bounded thread pool (`_run_bounded`, capped at
`_DIRECTORY_TRANSFER_WORKERS`), with each individual file's multipart
thread count sized adaptively (`_adaptive_transfer_config`) so a directory
with many files stays within a fixed total thread budget, while a
directory with very few files — down to the common single-large-checkpoint
case — is not needlessly capped below boto3's own default multipart
concurrency.

Two behaviors are worth stating precisely, since one is new and one is
unchanged from before this change:

- **New:** `download_file`'s exact-object GET is now atomic. It streams into
  a staging file created with the destination's exact permission bits from
  the first byte: when replacing an existing file, `os.fchmod` sets those
  exact bits on the open file descriptor before any bytes are written (not
  just `os.open`'s mode argument, which alone can still be masked by a
  stricter-than-default process umask), then renames onto the destination
  only once the transfer completes and the byte count matches the
  provider's declared length. Replacing an existing restrictively-permissioned
  file therefore never has a window where the in-progress replacement is
  more permissive than the file it is replacing, and a transport or disk
  failure partway through never leaves a truncated file where a complete
  one previously existed.
- **Unchanged:** `download_path`'s exact-object-vs-tree precedence —
  a literal object key always wins over a same-named descendant tree — is
  preserved through the concurrency refactor. For a non-`/`-ending,
  non-empty key, a direct HEAD checks for the exact object first; when
  that HEAD is ambiguous (403) or is skipped (a `/`-ending, non-empty prefix),
  one bounded, single-item listing probe checks for the same exact-key
  precedence before the prefix is treated as a tree. An empty prefix skips
  this probe entirely and is always treated as a tree.

**Real evidence** (5 alternating real-Nebius rounds per study,
hash-verified readback every round, fixture prefixes deleted and verified
absent afterward, zero provider retry attempts observed). Fixture sizes:
"64 small files" is 64 x 32KiB; "mixed directory" is one 128MiB file plus
63 x 32KiB files; "one large file alone" is a single 128MiB file. Two
separate studies isolate a genuine workload-dependent effect for the
single-large-file case:

| Scenario | Study | Upload | Download |
|---|---|---|---|
| 64 small files | fresh client per scenario | 6.22x | 6.94x |
| 64 small files | client reused across scenarios | 6.23x | 7.16x |
| Mixed directory (64 files, one large) | fresh client per scenario | 2.51x | 3.03x |
| Mixed directory (64 files, one large) | client reused across scenarios | 2.72x | 2.42x |
| One large file alone in a directory | fresh client per scenario | 0.94x (~7% slower) | 1.00x (~same) |
| One large file alone in a directory | client reused after small-file traffic | 0.47x (**~2.15x slower**) | 1.05x |

**Known, disclosed tradeoff — not a blanket "no regression" claim:** the
single-large-file-alone-in-a-directory upload case is measurably slower
than the prior implementation in both studies, and the size of that
slowdown varies with the scenario: smaller (~7% slower) when the client
used for that transfer was fresh, larger (~2.15x slower) when the same
client had just finished a small-file-heavy workload immediately before.
An independent, read-only code review found no algorithmic difference from
the prior implementation in the relevant code path (a single file takes
the non-pooled fast path and omits the per-file `Config=` override,
matching boto3's own default concurrency). The fresh-client study narrows
the gap but does not eliminate it, and no specific cause (network
conditions, connection state, or something else) has been established —
this is reported as an open, measured tradeoff, not a resolved or fully
explained one. Treat the storage speedup as proven for the common
multi-file case (consistently 2.4x-7.2x across both studies) and treat the
single-large-file case as a real cost under at least the reused-client
condition tested. It does not change the atomicity or correctness evidence
above.

Atomicity is scoped precisely: `download_file`'s exact-object GET is
atomic (staging + rename). `download_directory`/`download_path`'s
multi-file case moves each file through boto3's own per-file managed
transfer (itself atomic via a temp file and rename), not a
whole-directory transaction — a failure partway through a multi-file
transfer leaves the files that already completed in place and raises
without truncating any file, but does not roll back files that already
succeeded.

**Tests:** `npa/tests/clients/test_storage_transfer_regressions.py` (the
atomic-replacement and exact-object-precedence behaviors described above),
extensive coverage in `npa/tests/test_clients.py` including a deterministic
multi-file failure-propagation proof
(`test_download_directory_joins_concurrent_siblings_before_raising`, using
a `threading.Barrier` — no sleep-based timing), and existing containment
tests in `npa/tests/clients/test_download_containment.py`. Committed gated
live correctness coverage (small multi-file and one file above the
multipart threshold, hash-verified, dedicated-prefix cleanup) in
`npa/tests/e2e/test_storage_transfer_live.py`.

## Real GPU execution

**Scope:** this GPU job proves the changed source runs correctly, and that
its S3/manifest-discovery/credential-preflight behavior is correct, on
real GPU compute. It is not a training-throughput benchmark and does not
extend any "5x" claim to GPU compute — the SGD steps and matmul below are
a correctness/liveness check (real gradients, real device math), not a
performance measurement. It is also not a Genesis simulator qualification:
the published Genesis image was used only as a ready-made PyTorch/CUDA
runtime container, and no Genesis simulation capability was exercised or
validated.

Both GPU jobs (RTX first, B200 second) ran the same frozen source
archive, staged once for all five modules (the three changed modules,
`sim2real_health.py`, and the incidental `workbench/data` fix) before
either job started. SHA256 of each staged module file matched that frozen
reviewed snapshot exactly, verified independently on the operator's own
machine, not just the jobs' self-report. That frozen snapshot predates one
documentation-only change (a docstring clarification, no behavior change)
that later landed in `sim2real_health.py`; the final original-campaign snapshot differed from
what both GPU jobs actually ran only by that one change, confirmed
executable-AST-equivalent (source with docstrings stripped is identical) —
the other 4 modules remain byte-identical. Both jobs exercised the real
changed behavior, not a synthetic stand-in:

- 12 durable run manifests correctly listed and the exact target run
  correctly discovered through the concurrent manifest-discovery path.
- 2 concurrent real credential-preflight checks, both `s3`, proving two
  real concurrent S3 probes execute correctly (the separate, deterministic
  unit test proves the repeated-check-name worker-pool cap; this GPU run
  does not exercise that cap with only 2 checks).
- An exact-object download preserved its destination's `0600` permission
  bits through the atomic replace path.
- 34 output artifact hashes verified identical between the worker and an
  independent operator-side readback (input artifacts were verified
  separately: the worker checked them after its own download, independent
  of this output check).
- A real CUDA computation, not just a driver check: a 2048x2048 matmul
  with zero max-absolute-error against an independently recomputed
  expected result, and 16 real SGD optimizer steps with monotonically
  decreasing loss (0.2275 -> 0.1432).
- The written checkpoint's ZIP container CRC is valid and its float32
  parameter buffers contain only finite values on both GPUs (no local
  `torch.load` was performed for this check; the operator machine has no
  torch installed).

| GPU | Result | Compute capability |
|---|---|---|
| RTX PRO 6000 Blackwell Server Edition | PASSED | `(12, 0)` |
| B200 | PASSED (after one retry) | `(10, 0)` |

The B200 job's first launch attempt failed before reaching any worker code
(a provider-side `StartFailed`); root cause was the private launch
harness assuming the wrong interpreter path for that image, not the
changed source. The failed attempt's job was confirmed terminal and
removed, and the retry (a distinct job) passed with the same checks above.

## Full-suite and guardrail status

The full hermetic unit suite reports `155 failed`, `21,111 passed`, `235
skipped`, `1 xpassed`. Its failing-test set was diffed node-ID-by-node-ID
against an independently generated clean-base run on the same machine: the
two failing-test sets are identical (0 new failures, 0 resolved). Both runs used `pytest-timeout`'s signal method; the base run's
timeout threshold was 180 seconds and the candidate run's was 45 seconds
(lowered after an unrelated hang during an earlier attempt) — the timeout
threshold was not identical between the two runs, so the failure-set match
is evidence this change did not add or remove failures, not proof of an
otherwise fully controlled comparison, and it does not by itself establish
the individual root cause of every one of the 155 pre-existing failures. All 3,460
repository guardrail tests pass. `ruff check` is clean on every changed
file.

## Cleanup

The original campaign's dedicated validation infrastructure was torn down:
GPU compute (both jobs) reached terminal `COMPLETED` state, and an
independent post-completion inventory audit of that project found zero
remaining jobs, endpoints, instances, disks, filesystems, snapshots, GPU
clusters, Kubernetes clusters, or IP allocations. The project's writable
object-storage bucket and its owned storage service account (including
its access key) were then deleted and their absence independently
verified through two separate provider queries. The project itself was
retained, not deleted, per the task's scope.
