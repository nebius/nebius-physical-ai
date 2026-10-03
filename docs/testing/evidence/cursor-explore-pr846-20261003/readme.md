# PR #846: held-out publication authority

Status: **not merge-ready**. Source and transport controls do not establish
full-pipeline GPU execution, policy quality, or supported-image release.

## Immutable source

Original PR head: `5687434aea366cfa08eb576a5aea1df83903f90f`.
Original committed S3 execution: `4457c5b1ede9561fd61901452524d1d39f791011`.
Frozen full-suite runtime source: `2cac4009d97c326bd6f5298c3bc718031bdbf62b`.
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

## Independent successor findings

### Later immutable successor observations

#### Preserved subsequent failures and review disposition

The later frozen execution at
`0703581562ef7e6c9001adb3997502702a6137d7` completed with 40,562 passes,
200 skips, one existing non-strict XPASS, seven failures and one error;
coverage was 77.76%. Three failures concern stale generated dependency pins,
one concerns the actual standalone storage bundle importing a newly separated
module, and three concern older viewer fixtures/assertions. The error comes
from an extra pytest plugin introduced through the Rerun test extra. This is
a failed full gate, not a pass at a later head. Its original logs and XML remain
unchanged. Its separately passing root precheck, 238 affected CPU controls and
18 real-S3/retained-response CPU controls do not override that failed gate.

The sole focused first-party Claude follow-up reviewed
`3a208dece20e1fac8f832592216e4c0b048fde81` and again returned changes required.
Its repaired runtime files are byte-identical at the frozen execution above;
incoming-main integration and additional files remain separate scopes. The
actual substantive model was `claude-opus-5`, provider `firstParty`. Both
reviews were text-only AI reviews, not human approval or media inspection.
No third Claude review is claimed or requested.

Two additional CPU controls at the frozen execution above reproduced retained
cache growth across eight access generations and an unidentified-client cache
namespace. Two separate fault-injected CPU controls reproduced acceptance of a
known pre-transport SDK parameter rejection when another actor created identical
bytes. These are narrow create-CAS counterexamples, not a demonstrated live PUT
failure or revival of the withdrawn generic-CAS claim. During the subsequent
working repair, an actual filesystem control also showed that same-sized local
rewrites can retain identical timestamp metadata; the failed control is kept.

The subsequent repair therefore bounds cache retention, reclaims stale access
generations, avoids recyclable client identities, and re-authenticates retained
local bytes by SHA-256 rather than trusting timestamps alone. The default
1 GiB/128-entry budgets limit retention only: larger valid recordings continue
through verified temporary streaming. Unknown clients do not reuse retained
bytes. Local storage failure is distinct from a journal integrity conflict.
Known pre-transport parameter errors remain errors; exact post-write lost-response
recovery remains a separately tested supported path. Standalone storage bundles
retain their own complete transport adapter, without a new package dependency.

Cold listing still reads and verifies complete recording bytes against the
journal and selected object version. This is an explicit authenticity policy,
not a claim of metadata-only listing or bounded cold-read latency. Warm reuse
avoids another provider whole-body GET but still incurs local SHA-256 I/O.
There is no HEAD-only acceptance shortcut. Fresh frozen-source execution and
independent delta review are required for these later repairs; none of the
historical results approve their future source or complete the GPU pipeline.

At `28385f1baab8ba47151dfa4ada374315082acedf`, the fresh standard CPU
coverage gate completed with 40,529 passes, 213 skips, one existing non-strict
XPASS and three failures; coverage was 77.72%. Two viewer fixtures lacked the
now-required object identity. One older positive expected a symlinked containment
root to be allowed, conflicting with the new fail-closed containment policy.
That execution is retained as failed, not relabeled as a future successor pass.

Fresh actual S3 retained-input controls passed five cases. Actual synthetic CPU
RRD encode/decode plus viewer-cache S3 controls passed one case and failed one:
the legacy conditional GET error entered botocore's XML parser as a
`StreamingChecksumBody`, producing `TypeError` before the expected conflict
translation. The separate actual production-client reproduction also failed
that GET negative; its successful checksummed GET and non-mutating stale-writer
conditional file PUT controls both passed. This does not establish a blanket
journal PUT failure. Historical actual-MCAP in-memory successes above remain
bound to their original executions.

First-party operator Mac Claude Code pass 1 reviewed the exact successor as a
text-only AI code review. The actual substantive model was `claude-opus-5`
with provider `firstParty`; its verdict was changes required. Independent CPU
controls confirmed repeated listing/range whole-body reads, healthy large-report
rejection and untranslated inventory conflicts. The separate real-client GET
failure remains material; the proposed generic-CAS defect was not established
by prewrite, stale-writer and exact lost-response recovery controls.

Independent actual RRD decoding additionally found two semantic collisions:
different auxiliary-timeline winners and recursive-clear order shared a semantic
identity. These executed counterexamples contradict Claude's unexecuted soundness
assessment of that part; both review opinions and failed recordings are retained.
A local CLI RRD refresh also installed rejected bytes before verification and
deleted a prior verified destination. Agent viewer and local CLI cache paths
are distinct. No original or later failure is erased by a repair proposal.

Current replay-writer attribution is accepted only within the independently
reviewed canonical path at this exact head. The hypothesized marker-free,
ComponentRecord-free shape rejects before upload; no reachable canonical bypass
was demonstrated. This is not complete pipeline or future-head approval.

Repairs and the one focused Claude follow-up require a separate immutable head,
affected tests/live controls and independent receipt. No new hosted inference
or GPU acceptance is inferred. Full 14-stage acceptance still requires the exact
qualified controller, Transfer, EnvGen, Isaac and viewer images.

At the original `2cac4009d` review, five runtime findings required changes: cleanup containment,
rejected RRD cache preservation, static/blueprint semantic identity, verified
report re-reads, and current replay writer attribution. Owner repairs require
independent immutable successor review and full live replay acceptance. Replay must
preserve immutable original input authority separately from the actual current
writer/output generation; changing the retained report source or requiring all
historical inputs to match the new writer would erase provenance.

Published892050d67 CI also exposed two root-collection test-helper import
errors. Import-only7acc repairs that exact CI invocation with unchanged production
source and assertions. Owner root/package collection each found41,048 tests and
each focused population passed22; independent two-file collection/execution
each found/passed3 in both cwd modes and closed S2 only. Current7acc CI passes;
the failed892 attempt remains retained, not relabeled green.

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
