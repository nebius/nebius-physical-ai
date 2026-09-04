# Making npa and Workbench first-class: documentation and usability audit

`npa` already presents Workbench as its primary solution. The largest remaining
usability gap is continuity: users can find a command or recipe but cannot always
follow one verified path from installation to a completed workload, an inspected
artifact, and recovery. Several narrative examples also describe interfaces or
workflows that the current implementation does not provide.

This audit identifies those gaps, makes focused documentation improvements, and
sets acceptance criteria for the remaining product work. It does not treat help
output, a validated spec, or a submitted job as proof of a completed workload.

## Scope and evidence

The source review started from `918196ed8c96c354a95f297eb1c188257bbecd2a` on
2026-09-04. It covered the repository/package landing pages, installation and
onboarding, documentation indexes, CLI help, Python and HTTP access, representative
workload guides, workflow/resource setup, artifact discovery, error recovery,
and contributor documentation.

Evidence is separated into three kinds:

| Evidence | What it establishes |
| --- | --- |
| Source and documentation inspection | Which interfaces exist and where published instructions contradict them |
| Local command and validation probes | Installation/help behavior, argument handling, imports, spec validation, and documentation consistency |
| Real workload and artifact checks | Actual execution, outcomes, output integrity, and inspected results for the specific exercised journey |

Exact operational records remain in access-controlled external evidence. Public
findings use code references, aggregate measurements, and artifact digests rather
than customer inputs, credentials, live resource names, or private locations.

## Existing strengths to preserve

- The [root README](../../README.md) names `npa` as the CLI/SDK and Workbench as
  the primary solution. The root CLI already places Workbench in a
  `Primary solution` help panel
  ([main.py](../../npa/src/npa/cli/main.py), application registration).
- Workflows already have `status`, `logs`, `artifacts`, `list`, cancellation, and
  teardown. Their [run resolver](../../npa/src/npa/orchestration/npa_workflow/run_resolution.py)
  distinguishes unavailable verification from conclusive absence. Improve
  discovery of those capabilities before adding another competing command.
- [Credential/access preflight](../../skills/atomic/health-preflight/SKILL.md),
  [GPU selection](../../skills/atomic/gpu-selection/SKILL.md), the
  [workflow catalog](../../npa/workflows/workbench/npa-workflows/README.md),
  [run lifecycle](../run-lifecycle.md), and [teardown](../teardown.md) provide
  much of the necessary operational guidance already.
- [Generated CLI documentation](../../scripts/build_docs.sh) is checked for
  drift. [Contributing](../../CONTRIBUTING.md) already describes registration,
  shared implementation, tests, containers, and agent skills and acknowledges
  that existing tool interfaces differ.

## Implemented improvements

These changes improve documentation and discovery. They do not add runtime
capabilities or establish new training-quality claims.

| Changed surface | Result and user/developer impact |
| --- | --- |
| [Repository README](../../README.md) | Direct Workbench and Python/API links; explicit integration, extension, and documentation-index routes |
| [Documentation index](../README.md) | Task-based entry points through setup, GPU workloads, programmatic access, recovery, viewing, and contribution; general workflows precede niche examples |
| [Workbench index](../workbench/README.md) | An ordered setup → workload → runtime → submit → inspect → cleanup path; direct workflow, guide, lifecycle, and contributor references |
| [Guide index](../workbench/guides/README.md) | Generation, reconstruction, and data workflows alongside robot guides; prior backend checks retain their measurements and are scoped to the recorded runs |
| [Franka / Genesis guide](../workbench/guides/franka-pick-and-place-genesis.md) | Real serverless PPO training on H200, actual checkpoint/summary filenames, separate rendering constraints, and recovery guidance for a CLI failure while the same provider job continues |
| [Detailed installation](../install.md) | Clone and enter the repository before creating the environment, so installation and later activation resolve the same directory; preserve `.venv` for user examples and `npa/.venv` for repository validation |
| [Package README](../../npa/README.md) | Current Workbench scope, source installation, dependency boundaries, and programmatic/workflow monitoring entry points |
| [SDK error reference](../sdk/errors.md) | Source-checkout installation and a link to the canonical install guide |
| [CLI / SDK / workflow walkthrough](../workbench/cli-sdk-yaml-walkthrough.md) | Detection-training examples reflect actual service behavior and checkpoint discovery; identify the existing local CLI limitation and per-tool API differences; use the maintained declarative workflow for composition |

The navigation changes deliberately point to actual supported surfaces. For
example, `workflow artifacts` lists durable outputs generally, while
`workflow load-artifact` currently retries the final viewer load for a successful
PAIDF run. The latter is not presented as a universal viewer command.

## Prioritized remaining work

P1 findings prevent a promised journey or misstate its contract. P2 findings
increase discovery and integration effort. Suggested owners name the functional
area responsible for the outcome, not a particular person.

| Priority | Work item | Suggested owner | Status in this change |
| --- | --- | --- | --- |
| P1 | Reconcile the PushT guide with the workflow it actually submits | Workflows / robotics docs | Recommendation |
| P1 | Repair the local detection-training CLI-to-SDK contract | Detection training | Limitation documented; runtime fix recommended |
| P1 | Make default authenticated detection-training deployment become Ready | Detection training / deployment | Blocking probe mismatch documented; runtime fix recommended |
| P1 | Finish the advertised train → evaluate → inspect guide journeys | Tool maintainers | Recommendation |
| P1 | Publish accurate Python/API capability contracts | SDK / tool maintainers | Example claims corrected; catalog recommended |
| P1 | Make onboarding prerequisites depend on the selected runtime | Onboarding / platform | Entry labels corrected; runtime matrix recommended |
| P1 | Preserve durable completion and artifact identity across clients | Workflows / tool services | Current limits documented; runtime work recommended |
| P2 | Group help around user tasks and common submit operations | CLI | Recommendation |
| P2 | Separate current capability claims from historical validation | Docs / tool maintainers | Historical index scope clarified; Genesis training claim updated from a real run; further per-tool refresh recommended |
| P2 | Standardize installation and integration entry points | Docs / packaging | Main navigation/install/package fixes implemented; full example sweep recommended |
| P2 | Give contributors a path for each change type | Developer experience | Direct contributor links added; task-specific entry table recommended |
| P1 | Preserve exact job identity through cold-start phases | Serverless / runtime | Real failure and same-job recovery documented; runtime fix recommended |
| P1 | Align readiness with the exact write and execution scope | Platform / onboarding | Live scope mismatch and GPU discovery documented; unified preview recommended |
| P2 | Validate narrative journeys beyond generated help | Docs / testing | Local probes added as audit evidence; continuing journey coverage recommended |

### 1. Reconcile the PushT recipe with the actual workflow

The [PushT guide](../workbench/guides/pusht-sim-to-real.md), under "The
one-command live run", submits `sim2real.yaml` with legacy `NPA_SIM2REAL_*`
variables and describes an H100 wrapper with automatic cluster teardown.
The actual [spec](../../npa/workflows/workbench/npa-workflows/sim2real.yaml)
defines the compositional 14-stage standard runtime, defaults to
`Isaac-Lift-Cube-Franka-v0`, uses `bucket` and `trigger_uri`, requests RT-core
GPU profiles, and requires immutable image inputs. This is a different task and
resource contract.

**Impact:** a reader choosing the prominently linked PushT journey cannot
reproduce its advertised task, hardware, or lifecycle from the live block.

**Acceptance:** either provide a separately named, real PushT workflow with
registered live coverage and held-out evaluation artifacts, or withdraw the
incompatible live command, label the legacy local example, and link to the current
Sim2Real guide. The documented submit must resolve the advertised task, inputs,
config keys, GPU profile, and cleanup behavior. A Franka workflow must not be
presented as a PushT implementation.

### 2. Repair local detection-training argument forwarding

A local CliRunner invocation of the documented training command exited 1 with
`TypeError: train() got an unexpected keyword argument 'label_map'` before
training or input access. The installed console command reproduced the same
argument error with exit 2 through the top-level error formatter. In
[detection_training.py](../../npa/src/npa/cli/workbench/detection_training.py),
`train_cmd` serializes `TrainRequest`, including `label_map`, then forwards its
fields to the [SDK train function](../../npa/src/npa/sdk/workbench/detection_training.py),
whose signature does not accept that field.

**Impact:** the example selected as the clear CLI/API/SDK integration model has
a broken local path. Removing the failing recipe from the walkthrough helps
readers select the supported service path but does not repair this defect.

**Acceptance:** CLI, request schema, and shared training function preserve both
absent and explicit label maps. A test exercises the actual CLI-to-SDK adapter
with the expensive training implementation replaced at its call site, then a
real training/evaluation journey proves the repaired path and its artifacts.

### 3. Fix authenticated deployment readiness

The default detection-training deployment uses token authentication, but its
[_kubernetes_manifest](../../npa/src/npa/cli/workbench/detection_training.py)
emits an HTTP `/health` readiness probe without authentication. The
[service health route](../../npa/src/npa/workbench/detection_training/service.py)
requires the token. An in-process probe returned HTTP 401 without credentials and
200 with a synthetic authorized header. This establishes a probe/service
contract mismatch; it is not evidence of a live deployment attempt.

**Impact:** the default authenticated deployment cannot become Ready, so a user
following the advertised service recipe is blocked before training. The revised
walkthrough now requires an existing reachable authenticated service and states
the deployment limitation.

**Acceptance:** default deployment reaches readiness while preserving token
authentication and protecting tokens from printed/public manifests, probe
definitions, and logs; runtime secrets remain in the protected secret store.
Test the actual
generated probe against the service contract, then verify a real default
deployment, authenticated capability access, and rejection of unauthenticated
capability requests. Disabling service authentication is not the remedy.

### 4. Complete each guide's promised outcome

The [Reachy guide](../workbench/guides/reachy2-lerobot-policy.md) leaves its
public dataset as `pollen-robotics/<reachy-dataset>`. Its "Evaluate and serve"
block invokes `eval --help`, `serve --help`, and `infer --help`; those commands
do not evaluate, serve, or infer. Several guide index records concern smoke or
local checks, so they cannot substantiate a claim that every guide completed
training, evaluation, and visualization.

**Impact:** readers stop after training or help output without the evidence
needed to assess the policy or reuse its outputs.

**Acceptance:** each featured first-run guide pins a compatible public or
suitably sanitized input, names the execution mode, and provides working
training/generation, evaluation, artifact retrieval, and viewing steps. Its
completion evidence includes actual terminal status, a readable checkpoint or
output, a metric with its denominator where applicable, and an inspected
artifact. State hardware/environment requirements for real rollouts before the
run; do not substitute a help command for an unavailable evaluation.

### 5. Describe Python and HTTP support per tool

The [compatibility SDK namespace](../../npa/src/npa/sdk/workbench/__init__.py)
exports several kinds of clients.
[LeRobot](../../npa/src/npa/workbench/lerobot/__init__.py) and
[Genesis](../../npa/src/npa/workbench/genesis/__init__.py) wrap CLI callbacks
through [_sdk.py](../../npa/src/npa/_sdk.py), while detection training has typed
models and explicit local/service modes. The
[workflow SDK](../../npa/src/npa/sdk/workbench/workflow.py) exposes monitoring
functions rather than mirroring every CLI verb. A single uniform signature,
return type, or service endpoint contract is not available across all tools.
The [quickstart](../quickstart.md), under its CLI/SDK/YAML introduction, still
claims every capability has all three surfaces; this remaining claim needs the
same per-tool qualification as the corrected walkthrough.

**Impact:** notebooks and agents cannot infer supported imports, output types,
side effects, authentication, or completion semantics from the old parity claim.

**Acceptance:** a checked capability table links each tool to its Python import,
function signature, return type, execution modes, authentication, endpoint schema,
and error contract. Advertised imports and examples execute against their actual
adapters. Typed APIs evolve through shared implementation while preserving
existing CLI callback compatibility.

### 6. Choose the workload before requiring its infrastructure

[Workbench getting-started](../workbench/getting-started.md) requires the full
platform quickstart and initially lists Kubernetes, H100, RT-core capacity,
object storage, AWS CLI, Docker, kubectl, and Terraform together. Runtime-specific
skip instructions occur later. This explains why a page formerly labeled
fresh-clone onboarding can feel like repeated platform setup.

**Impact:** users of an existing service, a VM, or a serverless workload cannot
readily distinguish their actual prerequisites from those for other runtimes.

**Acceptance:** maintain one canonical install/project/access path followed by
a per-runtime requirements table. Each workload selects that table's row and a
verified GPU profile; only required tools, permissions, model access, and storage
are enforced. A clean-shell walkthrough for each supported runtime reaches a
real artifact without setting up unrelated infrastructure. GPU workloads remain
the primary entry path; hosted inference is documented for its captioning,
generation, and reasoning roles.

### 7. Make completion and artifact identity durable

[Detection training service](../../npa/src/npa/workbench/detection_training/service.py)
returns a running response and starts background work. Its run-state dictionary
is process memory. The
[training implementation](../../npa/src/npa/workbench/detection_training/training.py)
writes checkpoints under `<output>/<run-id>/checkpoints/epoch_<epoch>.pt`;
the former walkthrough guessed `model_final.pt` immediately after submission.
The revised walkthrough uses waiting and checkpoint discovery and describes the
service lifetime limitation.

Generic workflow artifact listing and PAIDF viewer retry are separate commands
in [workflow CLI](../../npa/src/npa/cli/workbench/workflow/__init__.py).
Tool-service run IDs and durable workflow IDs are also different contracts.

**Impact:** users may inspect an unfinished job, guess a nonexistent output,
lose service status after restart, or send the wrong kind of ID to monitoring.

**Acceptance:** completed runs emit durable typed artifact metadata containing
roles, media/schema types, existence checks, and exact follow-up commands. A
fresh client can retrieve the final model/report after service restart. Training,
generation, and dataset journeys each demonstrate status, artifact retrieval,
and a supported viewer or download path. Retrying viewing never repeats a
completed expensive stage.

### 8. Make help reveal the common task first

The captured installed CLI tree contains 30 visible Workbench groups without
task category panels. `workflow submit` declares 69 parameters, including the
positional spec, and its captured help occupied 538 lines. All ten inspected
help invocations succeeded; discoverability rather than parser availability is
the issue. See [Workbench registration](../../npa/src/npa/cli/workbench/__init__.py)
and [submit_cmd](../../npa/src/npa/cli/workbench/workflow/__init__.py).

**Impact:** readers and coding agents must scan tool names and advanced
recovery, registry, credential, and runtime options to assemble a common command.

**Acceptance:** group tools by tasks such as simulation, training, generation,
curation, evaluation, viewing, and operations. Keep workflow and health prominent.
Show one minimal verified submit example and separate advanced options into help
panels. Preserve existing names/flags, verify visibility in help tests, and
regenerate the CLI reference.

### 9. Keep historical evidence distinct from current capability

The [Franka guide](../workbench/guides/franka-pick-and-place-genesis.md) formerly
called serverless teacher training an import check with a placeholder checkpoint.
The current [_genesis_serverless_train_teacher_command](../../npa/src/npa/cli/genesis/__init__.py)
imports and calls the actual `train_teacher`, supplies iteration and environment
settings, and writes a checkpoint summary. The audit episode verified full PPO
training on one H200: 500 iterations, 1,024 environments, and 12,288,000
transitions, producing `model.pt` and its architecture/summary artifacts. The
guide now reflects this behavior and separates headless training from rendering
constraints. Producing the trained checkpoint does not establish task success;
evaluation evidence is a separate result.

**Impact:** old evidence can hide supported capabilities or be misread as proof
that a newer runtime already works. Broad tool-level "validated" statements also
hide the difference between an optimizer smoke and a complete evaluated policy.

**Acceptance:** record source/image revision, workload/input identity, GPU class,
actual outcome, and limitations for validation rows. Maintain current capability
prose separately and recheck it against implementation. Preserve historical
failures and scope; never upgrade placeholder or smoke evidence into proof of
training quality.

### 10. Standardize installation and integration entry points

The detailed install sequence formerly created the environment before cloning,
then told readers to reactivate it from inside the clone. That produced a working
first shell but a missing activation path in the next shell. The ordering is now
fixed and the user/contributor environment distinction remains explicit.
The [SDK errors page](../sdk/errors.md) also formerly said `pip install npa`;
it now follows the editable source-installation contract in
[installation](../install.md). The package README formerly described only
legacy LeRobot/Genesis functions and now introduces current Workbench scope.

**Impact:** users entering through a secondary page can install the wrong thing,
activate a different interpreter, or miss most of Workbench.

**Acceptance:** all installation examples use the supported source workflow and
consistent paths for their audience. Root/docs/package/Workbench pages link to
the same first workload, CLI/Python overview, workflow catalog, artifact recovery,
and contributor instructions. An isolated install/import/help check protects the
entry path, while real workload checks establish the operational result.

### 11. Give contributors the applicable contract without duplicating it

[Contributing](../../CONTRIBUTING.md) is a substantial complete-tool contract.
It is valuable for a new service but requires readers fixing one CLI option or
writing one workflow to infer which sections apply. The
[pre-PR skill](../../skills/atomic/pre-pr-validation/SKILL.md) provides a useful
change-to-gate map but should remain aligned with current scripts and CI.

**Impact:** contributors overrun the needed scope or miss registrations, workflow
contracts, documentation, or live coverage. Stale Known Deviations can be copied
into new implementations.

**Acceptance:** add an entry table for docs-only, CLI/SDK, workflow, tool/image,
and agent changes, linking exact files and applicable gates to the existing
contract. Keep one worked implementation per extension class and verify stated
deviations against current exports. Avoid a second independent contributor manual.

### 12. Check narrative journeys as well as help generation

The [docs generator](../../scripts/build_docs.sh) detects help drift, not
semantic recipe failures. The local training error above exists despite valid
help. The [BDD100K reference](../../npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml)
is a declarative state machine, while the prior walkthrough described raw
multidocument SkyPilot curl tasks. The revised walkthrough points to the
maintained spec; local validation and planning both exited zero.

**Impact:** green documentation generation can coexist with examples that use
the wrong workflow or never reach a result.

**Acceptance:** inventory the featured journeys, assign a maintainer and current
tool/spec target, and check links, imports, real argument compatibility, and
narrative commands locally. Run representative real journeys with artifact
checks. Publish static, mocked, smoke, and full workload evidence as separate
scopes so readers can assess what a pass means.

### 13. Preserve exact job identity through cold-start phases

A real `genesis train-teacher --runtime serverless` submission on H200 exited
1 after the provider create request exceeded the client's 300-second response
window. The client recovered the existing job by name, but its provider state
was `IMAGE_PULLING`. The [status mapper](../../npa/src/npa/clients/serverless.py)
does not recognize that phase, returns `unknown`, and the
[supervisor](../../npa/src/npa/serverless_common/supervision.py) stops on
unclassified evidence. The exact job continued, completed all 500 PPO iterations,
and wrote a valid checkpoint. Inspection by its existing identity recovered the
result without another training submission. These source files are unchanged
from the audit base.

**Impact:** a failed client command can leave real GPU work running, and an
operator may launch duplicate training because the result looks lost. A useful
first-class experience must report both the failed client operation and the
continuing backend job.

**Acceptance:** normalize documented provider startup phases while preserving
unknown states as uncertainty. Return the exact job identity, observed provider
phase, and safe monitoring/recovery action whenever a create call outlives its
response window. Test the timeout → same-job recovery → image pull → running →
terminal sequence without a second create call, then repeat a real cold-image
training journey and verify the final checkpoint. Never report unknown evidence
as absence or cancel a different job.

### 14. Align readiness with the exact write and execution scope

Live `health preflight` passed its four credential checks for the default
configured project. The separately saved cluster context passed SkyPilot
verification, but that cluster project's storage credentials returned
`AccessDenied` on a task-scoped `PutObject`. The default project's bucket passed
write/read/delete verification and belonged to a different tenant; it was not
substituted for the cluster's storage. No workflow launch or storage/IAM change
followed the denied write.

The selected serverless region also had no H100 platform. Read-only platform
discovery identified H200 as a compatible available product for the headless
Genesis training path; the H200 job then ran. A key-scoped hosted model listing
likewise establishes catalog availability, not successful inference.

**Impact:** individually green configuration, credential, and cluster reports
can concern different resources. A generic model/GPU default or a green listing
check cannot establish the next operation's prerequisites.

**Acceptance:** preview one effective project/tenant/context/bucket/endpoint and
GPU product for the selected journey, with sources of each setting. Gate the
actual target prefix's write/read behavior, exact model inference where needed,
and regional GPU compatibility before launch. Keep authorization failures scoped;
never silently borrow another tenant's configuration or change IAM. Preserve
secret values outside ordinary output.

## Validation and real-workload evidence

The audit used an independently installed Python 3.12 environment from this
checkout's `npa[dev,adapter]` dependencies. It exercised public stock simulation
and self-authored operational prompts. Generated model text is an output to
inspect, not a source for the code findings above.

| Check | Observed result | Evidence scope |
| --- | --- | --- |
| Documentation relative links and headings | All local targets and heading anchors in the ten changed documents resolve | Navigation validation; excludes external websites |
| User-facing install and onboarding guardrails | 15 passed | `test_docs_green_path.py`; documentation contracts |
| Narrative example review | Eight command argument sets, 27 callable imports, and three SDK bindings verified | Parser/signature checks; no inferred API parity |
| Representative CLI help | Ten probes exited zero | Discoverability and parser availability, not workload execution |
| Maintained BDD100K spec | `validate-spec` and `plan-spec` exited zero | Static authoring/runtime-plan checks |
| Local detection-training reproduction | CliRunner exit 1; installed console exit 2; unsupported `label_map` forwarding | Confirmed pre-training product limitation |
| Detection-training readiness contract | HTTP 401 without authentication; HTTP 200 with a synthetic authorized header | Local service/probe mismatch; no live deployment claim |
| Ruff, test collection, generated CLI reference | Passed | Full `npa` lint scope, 13,333 collected tests, and `build_docs.sh --check` |
| Full guardrails | 2,411 passed | Repository contracts, including the focused documentation checks |
| Mocked browser gate | 118 passed across agent, LeIsaac, and Foxglove specs | Real Cypress execution against mocked service responses; no live UI claim |
| Local `make test` suite | 12,932 passed, 36 skipped, one xpassed, one failed | The same Cosmos failure was reproduced on the pristine tracked audit base; explanation below |

The local suite used the Makefile's normal live/E2E exclusions, with pytest-xdist
parallel execution. Its failure is
`test_run_cosmos_transfer_names_gated_access_denial_without_leaking_prompt`.
It expects an inference subprocess's gated-access error, but the test leaves
`prepare_guardrail_nltk_data` unmocked. In the supported development installation,
that earlier step reports missing `huggingface_hub`. The failure was reproduced
with the tracked files restored to the unchanged audit base in this same clone;
all task document bytes were then restored and verified. No test, dependency
contract, or runtime code was changed to obtain a pass. Two earlier failures from
an ambient Rerun executable passed with this clone's environment first on `PATH`;
the final full-suite run used that environment. This is a documented baseline
failure, not a fully green suite.

### Real Genesis training and held-out evaluation

The GPU journey invoked the actual `genesis train-teacher` CLI using its
serverless runtime, one H200, 1,024 environments, and the default 500 PPO
iterations. At 24 steps per iteration, this is 12,288,000 environment transitions.
The public immutable container was:

```text
ghcr.io/nebius/nebius-physical-ai/npa-genesis@sha256:80cd1c4c7f7a5466533de29ee5cad1213202f348545346fea9ed6746fa03b1ca
```

The client command exited 1 during cold-start supervision as described in finding
13. The exact existing provider job subsequently reached `COMPLETED`, and its
summary recorded all 500 iterations. The audit recovered and inspected its real
outputs without resubmitting training: 18 objects totaling 30,805,018 bytes,
including checkpoint history, architecture, summaries, and training logs. The
2,535,555-byte final checkpoint passed archive integrity checks and has SHA-256
`3eb740a0b5caa6a406c01a52293faa36ead0041c7211a47c1502b59258c5f4bb`.

A separate job on the same immutable image downloaded that checkpoint, loaded all
17 model tensors, confirmed every tensor was finite, and invoked the image's real
`npa.genesis.generate_demos.eval_teacher` on 1,024 environments with held-out seed
7777. This used the library through a serverless payload because a native
serverless evaluation CLI was not established by the audit. The job completed,
wrote its evaluation JSON, and verified the exact stored bytes by reading them
back. The evaluation artifact has SHA-256
`5b72466c8a5f736dc486366b0344418ab2ecf21da9ba0ba400f8c5b3b8959a95`.

**Policy quality: zero successful episodes out of 1,024.** The training and
evaluation components ran and produced inspectable artifacts; this checkpoint
does not solve the task. The updated guide makes no solved-policy or successful
robot demonstration claim. Further training or reward/configuration work belongs
to the policy-development journey and needs its own evaluation evidence.

### Real hosted text generation

The storage → inference → stored output journey used 32 self-authored public
operations scenarios through `npa workbench token-factory generate`, with an
explicit key-visible `meta-llama/Llama-3.3-70B-Instruct` model. A second distinct output prefix exercised the same corpus with the explicitly
selected `Qwen/Qwen3-30B-A3B-Instruct-2507` model. Both CLI commands exited zero.
Independent verification found all 32 unique input identifiers in order, nonempty
text for every row, exact equality with the CLI's returned rows, and identical
bytes on a second storage read. The prompt corpus itself was also checked against
its pre-upload digest.

| Public model | Stored output | SHA-256 |
| --- | --- | --- |
| Llama 3.3 70B Instruct | 32 JSONL rows; 57,362 bytes | `9640fa3533b3ef73b410f6af8f0945799a2bf910517046bb03a45ec0ac227001` |
| Qwen3 30B A3B Instruct 2507 | 32 JSONL rows; 77,820 bytes | `58210bdf9bb92f99fd861436608f22eb72f2bedb1ca8361dfa06e572da28cf6a` |

These are real batches of hosted inference requests and persistent outputs. They
establish the two exercised paths, not every advertised model or GPU service.
Inspection found invented NPA command forms and incorrect GPU advice in the
ungrounded generated answers. These outputs prove generation and artifact
handling; they are not reliable NPA instructions and were not incorporated into
the documentation. Code-grounded review remains necessary before publishing
model-written operational guidance.

### Scoped limits and resource lifecycle

The saved cluster context verified successfully, but its project's task-prefix
storage write was denied. A multi-stage SkyPilot workflow was therefore blocked
before launch. The separately writable default project's bucket was not
substituted across tenant boundaries. No Kubernetes workflow completion is
claimed, and no bucket or IAM change was made. Read-only regional discovery also
showed why the default H100 selection could not launch there; the compatible H200
path above ran after explicit selection.

Only two serverless GPU jobs were created by this audit. Both reached terminal
completion. Their artifacts were preserved outside the checkout before the
terminal-safe cancel calls and exact-ID deletion checks. Existing shared
clusters, controllers, projects, and buckets remain intact. Task output objects
remain as durable outcome evidence; no new bucket or standing compute service was
created.

The evidence distinguishes a failed client command, a completed backend job,
and a policy that fails its quality evaluation. Recommendations remain proposed
work until their acceptance criteria are demonstrated.
