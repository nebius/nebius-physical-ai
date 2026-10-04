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

## Completed repaired CPU gate and current integration

The later frozen runtime `064b4ddf5d9083b9077f103d3a40bf92388948d8`
completed the local Linux coverage suite: 41,109 passed, 193 skipped, one existing
non-strict XPASS, zero failures/errors, 77.85% package coverage, in 2319.56 seconds.
Its source and environment remained frozen. Original XML SHA-256:
`d9bcf2e7d71ab35706aecbc18c33171cd8776ae3ccea2f6bcdeff3a16fd23ef2`.
This local run is not the distributed current-head CI admission gate.

Exact064 also passed 898 required security controls, 1,134 affected controls
with one explicit self-hosted GPU opt-in skip, 37 cache/SDK/CAS controls on each
of Python 3.10 and 3.14, 12 CLI installation checks, and 28 native plus 235
Cypress browser checks. Browser API/protocol fixtures are not a rendered Isaac
workload. Fresh actual S3 publication controls passed three tests; four further
real S3 controls deliberately induced native pretransport validation and
postwrite lost-response phases. They are phase controls, not a demonstrated
production PUT failure. Disposable fixture cleanup was verified.

The earlier `070358156` full result remains failed: 40,562 passed, 200 skipped,
one existing XPASS, seven failures and one error, 77.76% coverage. The earlier
`28385f1ba` full result remains failed: 40,529 passed, 213 skipped, one existing
XPASS and three failures, 77.72% coverage. Original import, SDK GET, semantic,
cache, dependency, transport and private protocol failures are retained.
Corrections do not relabel these executions or erase their original assertions.

Signed runtime candidate `c1755debcb8fd08c7865b063343895a83bc07a12`, tree
`2391cb70fa2377e1906511410f002fe6c945b64a`, composes actual main
`88fd1b1356c942745244e49d2316800634c530ae`. Its own isolated environment passed
1,599 affected prompt/rubric/content/grade/paired/metadata/CLI/SDK/registration,
cache and hostile-input controls, with one explicit live-GPU opt-in skip.
Root-cwd precheck passed 6,696 tests with six skips and collected 42,642 tests.
Original integration XML SHA-256:
`a58bd547b44fbca55dbfb2a3e8967f6f6d4438b2e5c4677d774fa98b19c22675`.
Fifteen relevant publication, regeneration, authority and storage production
files are byte-identical between064 and this candidate. Changed cache and
incoming VLM/grade paths have fresh affected controls, not a fabricated
candidate-SHA full execution. An earlier borrowed-environment attempt retained
13 failures because protected subprocesses imported the old editable source;
the isolated current environment passes without changing those assertions.

## Separate cache, review and shared-storage scopes

Independent Codex AI review at exact064 closed the auxiliary-timeline and
recursive-clear semantic collisions and local CLI rejected-RRD cache defect.
It reopened all four original failed recordings, preserved decoded disagreements,
and ran 52 scoped CPU controls with no skips. Current replay writer and immutable
original input authority remain separate. The relevant producer and regeneration
files are unchanged throughc175; this is a reviewed-source bridge, not a later
full-pipeline execution or human approval.

A separate parent review found a late process-wide cache installation defect:
three failed postrename observations left 48 unindexed bytes under a 20-byte,
one-entry retention budget. Candidatec175 registers renamed-leaf ownership
before observation and cleans only the matching candidate on failure. Failed
unlink remains indexed for retry; the primary exception and previously opened
readers survive. Five new negative controls fail the frozen original runtime
and pass in the fresh affected candidate population. Cleanup budget limits
retention, not accepted recording size or workload count. Full byte/journal/
version authentication remains mandatory; no HEAD-only shortcut was added.
The exact new delta still requires its separate independent receipt.

Actual first-party operator Mac Claude Code primary and sole focused follow-up
both returned **changes required**. The substantive model was `claude-opus-5`,
provider `firstParty`. Confirmed findings were repaired or separately adjudicated;
the original results remain unchanged and no third review was requested. An
independent reviewed-delta bridge is required for later repairs. These are AI
text/source reviews, not human approval, Claude pixel inspection or GPU proof.
Earlier native-VDI unavailability observations are historical; the Mac route
became available without copying credentials.

The committed595/846 shared-storage composition passed 234 CPU controls and four
real S3 controls. The S3 execution remains at
`4e7330642d6104db7af1912637fa330f0d05178b`; its capability/storage source is
byte-identical through the completed CPU composition
`831f667148de06c34b28202b167001e688643d0c` and relevant latest595 dependency
files. The cases exercise conditional create/retry, exact full-byte recovery,
hostile replacement rejection and unrelated-prefix preservation. Advisory
digest metadata absence was observed and retained; it is not authentication.
595's CopyObject failure and846's conditional GET parser failure are distinct.
No IAM expansion, private image publication or shared-service change occurred.

## Current request evidence versus historical responses

Main634 changes the default rubric, prompt, interleaved Frame labels, paired
request contract and grade reconstruction. Old `ae74dd3b` responses remain
traceable CPU replay inputs but now fail the current prompt-digest gate; they
are not new-payload hosted proof or current promotion evidence.

The separately retained main634 four-control execution at
`4cbaef3273e210b92ec0937af8b2e22575848de2` has an exact bridge to the candidate:
eight committed request/profile/parser helper ASTs match, all four full wire
bodies match their frozen and actual transmitted bytes, and frame, model,
prompt, rubric and protocol hashes match. Thirteen fresh CPU bridge/writer/
grade/canonical-precedence controls passed. Original bridge XML SHA-256:
`24be665d139085138e1945a834b52f776df7fc3647e5bb48430e0e9c29b354dd`.
No new provider call occurred in this bridge. Changed inherited transport and
current consumers have CPU controls; original hosted execution stays at4cba.

The [original narrow qualification](../../../workbench/evidence/vlm-terminal-ordinal-qualification.md)
retains one complete and three incomplete/ambiguous/blank controls at the frozen
0.8 threshold, plus documented rationale inaccuracies. It does not calibrate
model quality, qualify alternate models, establish physical safety or reverse
earlier rejected panels. Inherited paired audits remain audit-only; they cannot
be renamed into a scalar promotion artifact.

Full14-stage GPU/model/render acceptance still requires the exact qualified
controller/Transfer/EnvGen/Isaac/viewer delivery. Source/CPU/S3 closure, independent
final evidence review, current published-head CI and queue integration are
distinct gates. The PR remains draft and not merge-ready. Only the campaign
coordinator may enqueue after every gate; this owner has not merged or queued it.

## Completion-guard integration and delivery preflight

Signed source `6c5979195bef07ce088198f088bed4a55debb9b9` integrates landed main
`935bc9de16aebd9c08600fdba195781615c15eca`. The common scalar producer now rejects
incomplete completions before parsing on either backend; paired judgments also
require exact `finish_reason=stop`. Seven original634 request/profile/parser
helper ASTs remain identical. The eighth hosted helper changes because its
completion check moved into the common producer; the earlier eight-helper
statement remains scoped to its original candidate, not blanket current equality.

Fresh affected controls: 711 pass, one explicit live self-hosted GPU skip.
Fifty retained-response/consumer/completion controls pass with no skips,
including malformed content combined with 11 invalid completion forms on both
scalar backends and paired judgments. Root precheck: 6,712 pass, six skip;
42,913 tests collected. Twenty-six other critical modules are byte-identical,
including storage, grade, captions, direct Stage8, publication and regeneration.
Original hosted responses retain their original source and scientific limits;
there was no new hosted call, local full, GPU run or third Claude review.

Completed earlier-head `946b8fdb` CI remains its own fresh execution: six full
shards, 42,016 pass, 889 skip and one existing XPASS; all three required checks
succeed from app15368. It is not relabeled as execution of this successor.
The historical064 missing terminal journal is disclosed separately and is not
a source-readiness gate. Neither image qualification nor queue authorization
follows from CPU CI.

Actual immutable946 preflight passed five credential checks and recorded six
pinned-revision payload-access entries as Ready in task-owned state. No credential
values were retained and no legal assent was performed. The broader unselected
public dataset warning and degraded cluster result remain explicit. Three
existing one-free-RT-GPU targets were observed; only two numerically fit the
16-CPU/128-GiB tasks. This is volatile capacity, not a reservation or GPU proof.
All production training, shard and quality defaults are preserved; a planned
GPU concurrency of one serializes the eight shards without truncating them.

The independent read-only image audit found no qualified current-source set in
the inspected supply. Four exact historical manifest/config reads succeeded,
but their source is stale; historical Isaac is absent from accepted releases
and quarantined. This is not a claim that all registries lack images or that
current registry access was denied. No layers were read and no image was built,
copied, pushed or promoted. Audit receipt SHA-256:
`d01d5bfaa9298817d4b6d97f53a31f5d771ab78f2299c7db1ebacab5249645e1`.

The canonical graph requires five delivered roles: controller, Transfer,
EnvGen, Isaac and viewer. Stage8 is hosted CPU work in the controller image,
not a separate Reason image. Structural validation passes; both actual static
branch renders fail closed at the first missing immutable controller image.
No placeholder digest, source overlay or stub was substituted.

The remaining delivery requirement is five immutable source/build/byte/security/
bootstrap/capability bundles, compatible same-digest Isaac cache and verified
task seed/assets. The real14-stage GPU/model/render and restart path has not
executed. Direct use of qualified operator images requires no registry copying
or public promotion; private or quarantined bytes must not be publicly released.
Technical payload access is not full auxiliary-download inventory, product
acceptance, image qualification or model-quality proof. The PR remains draft.

## Combined review closure and model-identity integration

The independent combined review accepts the exact946 source, CPU/full-CI,
retained storage transport and hosted-request populations, with their existing
limits:174 immutable source bindings and111 artifacts. Original four4e733 S3
controls and four4cba sealed hosted requests remain those original executions;
23 submitted image occurrences are eight unique images. Rationale errors,
scientific uncertainty, failures and both first-party Opus changes-required
receipts remain retained. This is AI review, not human approval, future-head
approval, image qualification or model-quality acceptance. Receipt SHA-256:
`1d03098eccabb74f8d1c8f908046397c27337b9adc2958b717e43d47da046b77`.

Publishedf4 CI run[37168098903](https://github.com/nebius/nebius-physical-ai/actions/runs/37168098903)
completed successfully:42,066 passed,889 skipped and one XPASS across six full
shards; coverage displays78%. Twenty checks succeed and four are distinctly
skipped, with no pending/failing/canceled checks in the retained observation.
Required gitleaks, scan and security-regression are actual app15368 successes.
Execution merge7a2078ddbbd8f1842f349e7a2933ef5d44f031ad has parents935/f4
and exactly the publishedf4 tree. Validated artifact ZIP SHA-256:
`20db4e5976012d5caa05183a17b5b6808ab753ea2898078bda5792be929c35e5`.
Completed CI receipt SHA-256:
`e8d3497237a0a80b4375145ea3bfac076dadca844e933c907b94cb01a33a1e9d`.
Two failed collector name assumptions remain diagnostic failures, not failed
CI or rewritten artifacts. No CI/full rerun was needed to bind the logs.

Actualmain833 `a5a934af1516b78d88b2b2755247ec59bb1559b5` is composed in
separate signed source `82aa856129556c85ff81f680df00e234c2c8e71d`, tree
`ea0976171a7002f58c4c07c2b7c56975be5928ba`. Only two critical runtime modules
change fromf4: effective model/enforcement disclosures through scalar, loop and
benchmark results, and optional grade-claim validation. Twenty-five other
critical modules, including publication/cache/regeneration/storage/directStage8,
retain byte equality. Seven original request/profile/parser helpers remain
unchanged; the eighth hosted helper changed completion/disclosure behavior.
No blanket evaluator equality or new hosted execution is claimed.

Exact82aa affected tests:820 pass and one explicit live self-hosted GPU skip.
Root precheck:6,714 pass,six skip;42,948 tests collected. The original replay
attempt passes56 controls and fails four newly added direct-dataclass checks:
the serialized-report validator correctly rejects tuple-bearing in-memory
frames. Separate corrected production JSON-writer/readback/grade controls pass
all four and retain that tuple rejection as a negative assertion. The original
50-control response/completion panel and six hostile enforcement claims pass
in the56; no single60-pass execution is invented. Original requests/responses,
thresholds and frame hashes are unchanged; no hosted retry, local whole-full,
GPU repeat or third Claude was selected. Bridge receipt SHA-256:
`76b829d5895860290d6197dd18df400c1263b328abe79b408ec1ba6152833f8c`.

The proof appendix alone differs from this source execution. New published-head
CI remains a separate final gate; neither priorf4 CI nor independent946 review
is relabeled. Historical064 terminal-journal absence remains disclosure only.
Five qualified immutable images, compatible cache/seed and the genuine14-stage
GPU/model/render and Stage8/Stage14 restart acceptance remain absent. No private
image publication, draft removal, owner merge or enqueue has occurred.
