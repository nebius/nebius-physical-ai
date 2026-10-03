# PR #846: held-out publication authority

Status: **not merge-ready**. Source and transport controls do not establish
full-pipeline GPU execution, policy quality, or supported-image release.

## Immutable source

Original PR head: `5687434aea366cfa08eb576a5aea1df83903f90f`.
Original committed S3 execution: `4457c5b1ede9561fd61901452524d1d39f791011`.
Current runtime successor: `2cac4009d97c326bd6f5298c3bc718031bdbf62b`.
Successor tree: `09ad0aff7ca982c038e866f4c2bb2f620a84a26c`.
Integrated main: `f1ecd4255131374c417cdbb1f6c3e0d84b2a3514`.

Main's frame grouping, sampling, simulation conversion, workflow staging, and
landed evaluator provenance are composed with the publication authority changes.
Exact SHA-256 comparisons show eleven relevant publication, authority, storage,
live-control, canonical Stage8/direct-client and visualization files unchanged
between the S3 execution source and the successor. The original execution stays
at its original SHA; it is not a successor-SHA execution. The changed MCAP cache
consumer and generic evaluator composition have separate fresh controls.

## Results and retained failures

| Execution | Measured result |
| --- | --- |
| Original-head CI | One coverage shard failed, four canceled, coverage skipped |
| Frozen full suite at4457c5b1 | 40,207 passed; 213 skipped; 1 XPASS; 2 failed; 77.67% package coverage |
| Successor affected composition | 1,284 passed; 2 explicit self-hosted GPU opt-in skips |
| Successor simulation conversion | 21 passed |
| Successor required security regressions | 898 passed |
| Successor smoke | 118 passed |
| Successor CLI install | 12 passed |
| Successor precheck and merge precheck | 270 passed; combined tree clean |
| Browser executed at2cac4009d | 28 native checks and 235 Cypress tests passed |
| Python3.10/3.14 compatibility executed at2cac4009d | 164 passed on each interpreter |
| Full suite executed at2cac4009d | 40,262 passed; 213 skipped; 1 existing XPASS; zero failures; 77.69% package coverage |
| Aggregate confidentiality and secret scans | No findings |

The canonical workflow validates and its runtime wave plan resolves. Production
render checking fails closed at the first controller step because no immutable
source-attested image is configured. No provider is contacted or secret
materialized by this check. Task-scoped public-key verification accepts both
signed execution commits without changing shared Git configuration.

These browser/compatibility rows refer to fresh executions in the isolated
2cac4009d checkout. Separate earlier4457c5b1 executions are retained with their
original identities; they are not relabeled. The runtime checkout remained clean
and fixed at2cac4009d throughout the fresh checks.

The completed successor suite took2925.39seconds. Its original XML SHA-256 is
`43e152400d69cb5604e6d81a6fcb5f1bb6363dd50b9886e91b388496ace57a9a`.
The non-strict XPASS is the existing LanceDB registration marker; it is not
collapsed into an ordinary pass or removed. Opt-in GPU and unavailable optional
dependency skips remain skips, not model or component acceptance.

The two frozen-suite failures remain retained. One positive encoder-failure
fixture lacked the newly mandatory GPU evidence and failed before reaching its
injected encoder error. The successor repairs its baseline, preserving the
encoder failure and zero-publication assertions. The second failure was a
harness-path false positive: temporary test homes located under the real account
home triggered the isolation guard. The same three isolation tests pass with a
private temporary root outside the account home; the guard is unchanged. An
early successor attempt using the same unsuitable root was interrupted after
393 passes and 8 skips; it is not a completed gate. Its XML and diagnostics are
retained separately from the correctly isolated execution.

Four missing-GPU authority failures were reproduced by deleting both device
fields from rehashed Stage3/7/9/10 records. These GPU-mandatory stages now reject
absent or incomplete device provenance; CPU stages remain valid without it.
These checks do not turn operator-provided metadata into cryptographic proof of
actual GPU execution. A legacy-alias read without strong identity was separately
reproduced and repaired through retained listing ETags, authorized HEAD fallback,
and conditional reads.

Two actual-MCAP encoding/decoding controls reproduced cache corruption: a newer
valid committed generation overwrote another reader's verified pathname, and a
journal-rejected hostile replacement still changed the prior verified file.
The successor isolates temporary staging, verifies journal bytes and the strong
object snapshot, and atomically exposes content-addressed verified files. Both
controls now pass, preserving prior bytes on rejected reads. These are real
format/consumer controls with an in-memory storage transport, not a GPU rollout
or physical-policy result.

A separate actual legacy converter/readback control emits two messages, decodes
the fixed numeric inputs (`count=2`, `success_rate=0.5`), reuses identical
canonical bytes on the second read, checks both summary paths, and confirms
temporary staging cleanup. Its timestamps remain explicitly `synthetic-fps`,
not sensor time. The numeric values are fixture controls, not measured policy
performance. This CPU control also uses in-memory storage rather than live S3.

## Original real S3 execution

At committed4457c5b1, three real S3 tests passed in2.31seconds with no skips:

- Conditional streaming create/update/delete rejects stale versions without
  altering the winner's bytes.
- Four-object committed publication rejects stale writers and hostile immutable
  replacement while canonical aliases and the journal remain unchanged.
- An injected interruption leaves publication incomplete and unreadable;
  recovery verifies immutable bytes and rolls forward exactly once.

Only disposable task-owned fixture objects were cleaned up. The fixtures are
explicitly transport bytes, not policy, RRD or MCAP pipeline artifacts. Raw
commands, errors, source identities and results remain private. The original
committed S3 XML SHA-256 is
`8e255bcdbb61b4d9443260fc38eca01d477c641edac68f5da74652b6d91ef7c4`.

## Remaining acceptance conditions

The inherited CPU-only claim is insufficient. Changed checkpoint/producer/render
contracts require genuine Cosmos Transfer, EnvGen and Isaac rollout/PPO/gold
execution, hosted Stage8 evaluation, and decoded RRD/MCAP plus numeric lineage
review. All13 upstream records must bind exact source-attested images. No archived
run is relabeled, no source overlay or stub substitutes for live execution, and
no private image is published. An authorized five-image exact-source immutable
set or non-publication delivery path remains required. Authentication and exact
capability access passed but do not prove execution.

Independent Codex receipt846-S1 concerns the original5687434 head and confirms
the missing-GPU defect. Separate successor review ran55 focused tests with zero
skips at2cac4009d and resolves that exact source finding. This is not full source
or live-evidence acceptance. A proof-label finding has fresh exact-SHA browser
and compatibility receipts supplied, pending reviewer reconciliation. Broader
immutable successor review remains open, including archive replay producer
identity. Native Claude is unavailable: no native executable/tool was found,
and a proxy using another upstream model is not Claude. No human approval is
claimed.

Full live acceptance, independent source/evidence review,
current published-head CI and the actual merge-queue integration gate remain.
The PR stays draft; no merge or queue action has been taken.
