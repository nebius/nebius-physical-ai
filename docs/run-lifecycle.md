# Run lifecycle: identity, gates, and status

[Docs](README.md)

What `npa workbench workflow submit` verifies before it launches, how a run keeps
its identity across interruptions, and how to read the status it reports back.

If you just want to launch something, start with
[the quickstart](quickstart.md) or the
[Physical AI Data Factory runbook](workbench/guides/physical-ai-data-factory-deploy.md).
This page is the reference for what those paths are doing underneath.

<a id="everything-is-verified-before-the-run-starts"></a>

## Checks before submission

`submit` repeats prerequisite checks **before** input or source staging, so
detected image and identity problems can be resolved before uploading the run's
inputs and source. It reports
detected prerequisites together with remediation commands. These checks do not
prove that the workload will complete or produce usable artifacts.

Run specification validation, planning, and image checks in this order.
Validation and planning do not launch a workload. Image checks can create and
delete a temporary Kubernetes probe pod when bootstrap evidence is absent:

```bash
npa workbench workflow validate-spec "<spec.yaml>"
npa workbench workflow plan-spec "<spec.yaml>" --run-id "<run-id>" \
  --var bucket="<bucket>" --check-render
npa workbench workflow preflight-images "<spec.yaml>" \
  --project "<alias>" --infra "k8s/<cluster>" --var bucket="<bucket>"
```

Use the same target and overrides throughout; public images need no `--registry`.
The [workflow quick start](workbench/npa-workflow-guide.md#quick-start) shows the
complete sequence.

`plan-spec --check-render` also sends the resolved static plan through the same
SkyPilot renderer used for submission. It catches pod configuration and
first-party image startup-contract errors that schema validation cannot see. The
check stays local, does not contact a provider, and does not materialize registry
secrets.

The renderer also compiles simple quoted Python heredocs in top-level `setup`
and `run` scripts with the local NPA interpreter, so syntax errors fail before
submission. This check is deliberately narrow: the quoted identifier delimiter
must end the command line, and the command must invoke a literal `python`,
`python3`, or versioned `python3` executable that reads its program from standard
input. Fixed environment assignments may precede it. Unquoted or nested
heredocs, continued command lines, dynamically expanded commands or assignments,
other interpreters, and heredocs passed to a Python script as data are left to
the shell and runtime.

The renderer rejects explicit non-root main containers that disable the sudo
access required by first-party images. It also rejects individual command
arguments of 131,072 UTF-8 bytes or more, including an oversized `run.shell`.
Keep large scripts and archives in declared workflow inputs or mounted storage;
use a short launcher to verify and execute them.

`preflight-images` reports each image as `ok` / `not_found` / `forbidden` and
prints the exact build command for anything missing. It unions every declared
decision outcome, and `--infra k8s/<cluster>` lets the pull check verify that
exact cluster's declared pull-secret authority. `submit` uses the same complete
image plan by default, so a branch-specific missing image surfaces before the
run instead of as an `ImagePullBackOff` on the cluster. If complete-path
planning itself fails, `preflight-images` exits before any registry or
Kubernetes probe; it never reports that failure as `images: none`.

### Image override compatibility

**Compatibility note:** an `--image-override` selector that matches no toolRef
now fails before preparation, credential resolution, or staging, including with
preflight disabled. Older versions silently ignored it. Scope shared override
sets to each workflow; use an exact toolRef, a family prefix such as
`workbench.fiftyone` (without glob characters), or bare `*`. Selectors for
unselected decision branches remain valid because matching uses every state in
the workflow. Library rendering and planning enforce the same contract.

### Quota is arithmetic, and it is checked first

The plan treats `compute.disk.size.network-ssd` as a **byte allowance**,
separately from `compute.disk.count`, and prints exact `required`, `available`,
and `shortfall` values in both bytes and GiB.

The default whole path needs **1,251 GiB** of new NETWORK_SSD capacity when
nothing exists yet: 100 GiB for the agent root disk, 128 GiB for the CPU node,
and 1,023 GiB for the GPU node. So an account with 21 GiB available is blocked
with a 1,230 GiB shortfall *before* Terraform, networking, the Kubernetes
control plane, VMs, or disks are created.

Proven existing resources are deducted on retries. Unknown or contradictory
quota evidence is never treated as permission to mutate.

### Images resolve to digests

Image preflight resolves each selected tag to an **immutable digest**. NPA
verifies first-party OCI bootstrap-contract metadata; an arbitrary unattested
image gets one exact, bounded capability probe in the selected context, whose pod
must be deleted successfully. Results are cached by digest plus contract version.
First-party images cannot replace their declared user with `runAsUser: 0`.

Image identity binds the render registry, GPU target, image variant, and every
image override selector to its reference, then binds each preflight reference
to its resolved immutable image. Mapping order does not matter; reassigning the
same references or digest values changes identity. Direct immutable image
overrides are included even when the preflight pin map is empty. Older
value-only or pin-only identities cannot prove this complete selection to an
upgraded controller; resume those runs with their original controller instead.

For each wave with rendered images, a versioned composite binds that v3
selection fingerprint to each state's tool and exact resolved reference. The
ledger retains those bindings, the references and their separate set hash, and
whether every reference is content addressed. Swapping selector assignments or
catalog defaults cannot hide behind an unchanged set of resolved images.

**Upgrade note:** earlier value-only, pin-only, and reference-set-only records,
as well as unversioned v3 selection-only records for waves with images, cannot
prove this binding in completed replay, durable output reuse, or supervised
recovery. Those identity checks reject weaker records; they do not migrate saved
identities in place. Ordinary in-flight provider adoption is a separate legacy
path and is not made into an input-identity fence by this change. Follow the
[controller rollout procedure](#controller-rollout-with-existing-runs) below.
Accelerator-name overrides remain outside image identity because they select
cluster resources rather than container images.

Multi-tool workflows can pin distinct validated images with repeatable
`--image-override TOOL_REF=IMAGE`. An exact tool override beats the optional
global `--image` fallback, and the rendered task uses the digest that preflight
verified.

Reordering selector or digest-pin mappings preserves the selected images and
permits completed-wave replay. Exact tool selectors take precedence over family
selectors, and the longest matching family takes precedence over `*`. Moving an
image between those selectors changes the run identity, even when the set of
image digests stays the same. Resource-level image declarations are covered by
the separate workflow identity.

### Controller rollout with existing runs

Use a separate controller revision for new runs while existing runs finish on
their original revision. A source merge does not require replacing a running
driver or resuming its ledger under the new controller.

1. Preserve each unfinished run's original controller revision, environment,
   prepared source, immutable images, and recorded workflow options. Keep exact
   run and provider identifiers in private operational records.
2. Leave those controllers and their ledgers unchanged. If an old driver needs
   to restart, use its original revision and recorded inputs with the same run
   ID. Do not point an existing run at the upgraded controller or rewrite saved
   identity fields.
3. Install the upgraded controller separately and use it for new work with new
   run IDs. A replacement of unfinished work is permitted only after the prior
   attempt is terminal or its exact cancellation has been verified; a new run
   ID alone does not prevent duplicate work.
4. Retire the original controller only after every run assigned to it is
   terminal. If its revision or evidence cannot be recovered, preserve the run
   and resolve that uncertainty before upgrading or relaunching it.

This procedure does not require interrupting a running GPU job. An alternative
is to wait until all affected runs are terminal before upgrading their existing
controller. A snapshot showing no active runs is valid only at the time it was
checked; it is not a durable substitute for preserving controller revisions.

### The controller launch is one transaction

Every Kubernetes managed-job launch crosses a single controller-launch
transaction: NPA probes the selected context through the same `KUBECONFIG`
environment SkyPilot uses, requires three consecutive `/readyz` successes
spanning 10 seconds, then reconciles the exact job name through structured
SkyPilot queue output under an owner-only logical-launch lock.

A transient controller-creation refusal is retried automatically only after
exact job absence is proven and API stability is re-established; an accepted
request is adopted by immutable job id. Ambiguous existence blocks rather than
risking a duplicate launch or a name-based cancellation. JSON exposes
`launch_transaction`. Full decision record:
[SkyPilot controller launch transaction](architecture/skypilot-controller-launch-transaction.md).

## Run identity

`prepare-run` reserves the exact run identity up front, so validation, planning,
and image gates all describe the run you are about to submit:

```bash
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT")"
```

It writes an **atomic, locked state record** scoped by stable project identity
and workflow identity. Fresh submits never inherit a historical global
`~/.npa/paidf-first-run-id`; an ambiguous legacy file is warned about, but never
reused and never deleted for you.

Each submission also writes an owner-only local receipt at
`~/.npa/workflow-submissions/<project>/<run>.json`. It holds location, plan, and
job identity **only — never credentials** — and it removes any dependency on
`NPA_SRC_S3_URI` in a later shell.

Resource profiles retain Kubernetes `automountServiceAccountToken` when its
value is a JSON boolean, so disabling or enabling that pod behavior survives a
restart. String, numeric, and structured lookalikes remain blocked by the
receipt's credential filter.

## Restart safety

Provisioning resumes the same secret-free operation journal under
`~/.npa/operations/`. It preserves configured credentials, storage, and durable
Terraform state, and prints one deterministic resume command.

Source staging and submission are content-addressed and idempotent for an
explicit `RUN_ID`, so repeating a submit repairs and reuses derived artifacts
instead of duplicating work.

The owner-only submission receipt under
`~/.npa/workflow-submissions/<project>/<run>.json` is part of that safety
boundary. If it is unreadable, malformed, symlinked, or belongs to a different
project/run identity, NPA preserves it and refuses pre-launch mutation. Audit
the path and its exact ownership, back it up, then repair the receipt before
retrying; do not delete it merely to bypass the check because it may retain the
only exact managed-job identity. If corruption occurs after provider acceptance,
submit still prints a `submission_warnings` entry only when the launch proves
`submitted` or `adopted` plus a nonblank job ID, so that job can be cancelled or
reconciled. Any weaker result remains a hard receipt error. Receipt warnings and
optional post-success artifact-handoff diagnostics redact URL query strings,
secret assignments, bearer tokens, and resolved credential values before JSON
output or local persistence. Every workflow CLI `Error:` boundary applies the
same shape-based redaction before rendering, while preserving multiline
recovery commands; credential-aware submit failures also redact the exact
resolved values even when a provider quotes an opaque token. Provider
diagnostics that enter durable workflow records use that same redactor for URL
userinfo, signed URL queries, bearer and known token shapes, secret assignments,
and PEM private-key blocks while preserving line structure for troubleshooting.

**A stale or ambiguous run is never selected silently.** Resume by naming it:

```bash
npa workbench workflow submit "$SPEC" --project "$PROJECT" --resume-run "$RUN_ID" ...
```

or call `prepare-run` again to create a distinct run. Non-interactive recovery
must use `--resume-run <id>`.

A run's committed source is immutable: retries repair or reuse derived
artifacts, but never replace a user-supplied source with a default.

For an attempt whose exact scheduler record is absent, use
`--retry-absent-in-flight` with the same explicit resume command after repairing
the reported dependency. Recovery rechecks both the exact job identity and every
wave-unique output URI, including structured output declarations with schemas.
A prior storage-read block remains retryable when the ledger retains verified
launch absence; a new lookup must still prove absence. Existing outputs,
unreadable storage, missing output declarations, or uncertain scheduler status
continue to block a new launch. Prior attempts remain in the run history.

A durable `block_relaunch` decision remains unresolved unless the same record
already contains verified terminal cancellation. After the reported dependency
is repaired, a record with provider identity is reconciled against that exact
job ID: a live job is adopted and polled, unavailable queue evidence remains
blocked, and neither case resets the attempt number or submits replacement
work. A previously verified terminal attempt keeps the ordinary terminal retry
path and does not become queue-dependent again.
An unrecognized non-terminal recovery decision also defaults to exact
reconciliation; only explicit completion/absence decisions, provider-terminal
evidence, or durable proof that launch never started may skip that fence.
A cancellation claim without a terminal provider status, or a malformed launch
sequence, remains uncertain and cannot authorize replacement work.

Directory-style output evidence scans every S3 list page for a non-empty
descendant. Zero-byte directory markers do not prove completion or absence,
and malformed/truncated pagination blocks recovery rather than authorizing
duplicate work.

Image override selectors must be an exact toolRef, a boundary-safe family
prefix such as `workbench.fiftyone`, or the bare `*`. Glob-like and unmatched
selectors fail before run preparation, provisioning, or source staging, even
when image preflight is disabled. Validation considers every workflow branch.

## Reading status

```bash
npa workbench workflow status "$RUN_ID" --project "$PROJECT" --watch
```

The finite infrastructure-recovery allowance limits relaunches, not reuse.
When every declared durable output is valid, recovery marks the wave complete
even at the allowance boundary. If the exact provider attempt is still live,
its cancellation must reach a verified terminal state before that reuse is
accepted.
The workflow completion result is separate from the observed provider status.
The verified cancellation and output-reuse decision are written as an immutable
supervisor event before the mutable runtime attempt is terminalized. If the
driver stops between those writes, resume binds that event to the exact attempt,
workflow, source, and image identities, revalidates every declared output, and
completes the same attempt without submitting replacement work. Missing,
malformed, conflicting, mismatched, or temporarily unreadable evidence remains
blocked and retryable under the same run ID. The provider's verified terminal
status remains factual even while output revalidation is blocked.
The wave record and immutable terminal event retain `output_reuse_error_type`
(for example, `PermissionError` or `ValueError`) beside the sanitized reason.
This distinguishes access errors from rejected evidence without exposing raw
credentials or changing the fail-closed recovery decision. The field clears
after successful evidence recovery; exception type alone does not authorize
a retry or a replacement job.

An unreadable supervisor history blocks reconciliation without creating a reuse
claim. Restoring access lets the same attempt reconcile its exact provider job.
If the driver stopped after the reuse decision but before recording verified
cancellation, resume validates the decision's identity and output declarations,
then reconciles that exact job. A live job is adopted; a succeeded job must still
pass output validation. A cancelled or failed job retains its terminal failure
and launches no replacement during that resume. A later resume can request the
ordinary explicit workload retry. An attempt already marked for reuse still
requires its immutable cancellation proof; missing or corrupt proof stays blocked.


`status` resolves the exact run from the selected project's receipt, the
canonical workflow prefix, or the pinned managed-job identity — even while the
final manifest is still pending. `logs` uses the same resolver. Both JSON and
text output identify every source they checked.

New submission receipts record the resolved SkyPilot executable and isolated
controller state directory used for launch. Later `status`, `logs`, and
`cancel` calls reuse that exact route. A legacy receipt with a managed-job ID
but no controller route requires explicit `--sky-bin`. Runs launched with
isolated state also require the matching `--isolated-config-dir`. NPA reports
live verification as unavailable instead of treating absence from an ambient
SkyPilot queue as proof that the job is absent.

| State | What it means |
| --- | --- |
| `MANIFEST_PENDING` | Submission evidence exists (a receipt, job, or task identity) and the manifest has not landed yet |
| `NOT_SUBMITTED` / `PLAN_ONLY` | A reservation, plan, or partial staging prefix exists, but no submission evidence — so it is *not* pending |
| `VERIFICATION_UNAVAILABLE` | S3, SkyPilot, or provider verification failed. **Never** treated as absence |
| `NOT_FOUND` | Emitted only after every applicable exact source answered authoritatively |
| `CACHED` | The explicit offline mode (`--cached`). Not live-verified, and not automation-trustworthy |

Unrelated nested S3 keys are never guessed as runs.

Per-stage status reads use the same cause-aware boundary. A genuinely missing
optional status object may fall back to manifest evidence; denied, throttled,
or unreachable storage makes `status` return
`VERIFICATION_UNAVAILABLE`/exit 2 while retaining the manifest's last-known
state. `cancel` makes no cancellation call and records only a
verification-failed receipt for the same uncertainty.
Client setup and response-body failures also remain unavailable, even when
their underlying exception resembles a missing file or key.

The optional `supervisor` object is a durable snapshot from its `recorded_at`
time. Status labels it `observation_scope: durable_supervisor_snapshot` and
`current_artifact_state_verified: false`; reading status does not recheck that
snapshot's outputs or checkpoints. A live `RUNNING` job can therefore appear
beside an earlier supervisor observation that outputs were absent. The top-level
`automation_may_trust_state` flag covers the current lifecycle-state query and
does not make those historical artifact observations current.

Status also corrects live-job adoption advice in terminal snapshots written by
older drivers. Such views identify `recovery_guidance_basis:
recorded_terminal_attempt`; they do not rewrite the stored event or verify
current artifact availability. Other recovery decisions remain unchanged.

When the driver records a terminal attempt, it replaces earlier live-job adoption
advice with the terminal outcome. A failed job directs the operator to its error
and artifacts; a completed wave is reusable only after its declared outputs pass
validation. This reporting update preserves the configured retry policy.

### Pending storage claims

When a managed job has no remaining worker pod, live status can still report a
typed storage blocker from its rendered persistent-volume claim. This lookup is
deliberately narrow: NPA uses only claim names declared by the exact rendered
stage and attributed to the exact recorded managed-job ID. It then requires one
current claim with the same namespace, name, and Kubernetes UID. The claim must
still be `Pending` and must not be deleting. Events are queried in that claim's
namespace and accepted only when their involved-object identity matches all
three fields.

`ProvisioningFailed` warnings become `STORAGE_QUOTA_EXCEEDED`,
`STORAGE_CAPACITY_UNAVAILABLE`, or `STORAGE_PROVISIONING_FAILED`. Normal
provisioning progress such as `ExternalProvisioning`, `Provisioning`, and
`WaitForFirstConsumer` remains pending and is not treated as a failure. A
warning caused by a provisioning timeout, temporary service outage, or rate
limit also remains pending and cannot authorize cancellation. A prior
warning on a claim that is now `Bound`, a warning for an older claim UID, an
ambiguous claim, or a malformed declared name also falls back to the existing
unknown/no-pod status.

Status may display a current UID-bound event without proving that it belongs to
the latest launch attempt. The supervisor acts only when the event's latest
observed timestamp is at or after the durable attempt start. Otherwise it fails
closed: it neither cancels nor relaunches. An actionable quota or provisioning
error can terminalize only the exact recorded managed job; a capacity blocker
can retain that same attempt for recovery. A declared claim name identifies
placement, not ownership, so it never authorizes broad claim discovery or
cancellation.

If a shell cannot resolve the project storage location, point status at the
prefix explicitly:

```bash
npa workbench workflow status "$RUN_ID" --project "$PROJECT" \
  --workflow-s3-uri "s3://$BUCKET/<workflow>/$RUN_ID/npa-workflow"
```

Valid outputs from a provider-succeeded attempt are reused only when the
recorded workflow, source, and image identities still match the requested run.
Missing or changed immutable identity evidence blocks reuse and requires the
recorded identity to be restored or a new run ID to be started.

## Where the kubeconfig goes

`provision-if-absent` writes the cluster kubeconfig to
`~/.npa/clusters/<context>/kubeconfig` rather than merging it into
`~/.kube/config`. `submit --infra k8s/<context>` finds that file on its own. For
`kubectl` in your own shell, export it (the command prints this line for you):

```bash
export KUBECONFIG="$HOME/.npa/clusters/<context>/kubeconfig"
```

A reserved recovery successor retains the provider's verified terminal state
when its outputs complete the workflow. Logical workflow success can coexist
with a provider CANCELLED or FAILED state; a SUCCEEDED race remains SUCCEEDED.
Adoption must not overwrite that provider evidence.

## Related

- [Known footguns](workbench/troubleshooting/known-footguns.md) — the failures these gates are designed to catch
- [Physical AI Data Factory runbook](workbench/guides/physical-ai-data-factory-deploy.md) — the whole path, end to end
- [Workflow authoring guide](workbench/npa-workflow-guide.md) · [tool catalog](workbench/npa-workflow-tool-catalog.md)
- [Tear it all down](teardown.md) — cancellation and cleanup ordering
- [debug-failed-run skill](../skills/atomic/debug-failed-run/SKILL.md) — triage order for a run that failed or hung

### Exact artifact lookup while a manifest is pending

An explicit storage locator remains scoped to that workflow. Artifact lookup
may remove a trailing control-directory suffix only when the requested run ID
and path layout, or the manifest's exact run-prefix provenance, identify the
parent as the run root. Workflow names and pending status alone never authorize
listing a parent prefix that could contain a sibling workflow's outputs.

Completed-wave replay requires nonempty, matching workflow, source, and image
identities before checking its retained outputs. A changed or missing identity
blocks both replay and replacement submission, even when an output is missing.
With matching identities, missing declared outputs retain the normal recovery
path; a replacement must publish and verify its own declared outputs.
