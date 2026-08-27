# Run lifecycle: identity, gates, and status

What `npa workbench workflow submit` verifies before it launches, how a run keeps
its identity across interruptions, and how to read the status it reports back.

If you just want to launch something, start with
[the quickstart](quickstart.md) or the
[Physical AI Data Factory runbook](workbench/guides/physical-ai-data-factory-deploy.md).
This page is the reference for what those paths are doing underneath.

## Everything is verified before the run starts

`submit` repeats its deterministic checks **before** input or source staging, so
a missing image or an identity mismatch is caught locally rather than after the
~1,225-file source tree has been uploaded and a cluster is waiting. It prints
**everything** still missing in one list, each with the command that fixes it, so
you are not discovering prerequisites one failed run at a time.

The read-only gates are meant to be run in this order:

```bash
npa workbench workflow validate-spec    <spec.yaml>
npa workbench workflow plan-spec        <spec.yaml> --run-id "$RUN_ID"
npa workbench workflow preflight-images <spec.yaml> --registry "$REGISTRY"
```

`preflight-images` reports each image as `ok` / `not_found` / `forbidden` and
prints the exact build command for anything missing. `submit` runs the same
check by default, so a missing image surfaces on your machine instead of as an
`ImagePullBackOff` on the cluster.

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

Multi-tool workflows can pin distinct validated images with repeatable
`--image-override TOOL_REF=IMAGE`. An exact tool override beats the optional
global `--image` fallback, and the rendered task uses the digest that preflight
verified.

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

## Restart safety

Provisioning resumes the same secret-free operation journal under
`~/.npa/operations/`. It preserves configured credentials, storage, and durable
Terraform state, and prints one deterministic resume command.

Source staging and submission are content-addressed and idempotent for an
explicit `RUN_ID`, so repeating a submit repairs and reuses derived artifacts
instead of duplicating work.

**A stale or ambiguous run is never selected silently.** Resume by naming it:

```bash
npa workbench workflow submit "$SPEC" --project "$PROJECT" --resume-run "$RUN_ID" ...
```

or call `prepare-run` again to create a distinct run. Non-interactive recovery
must use `--resume-run <id>`.

A run's committed source is immutable: retries repair or reuse derived
artifacts, but never replace a user-supplied source with a default.

## Reading status

```bash
npa workbench workflow status "$RUN_ID" --project "$PROJECT" --watch
```

`status` resolves the exact run from the selected project's receipt, the
canonical workflow prefix, or the pinned managed-job identity — even while the
final manifest is still pending. `logs` uses the same resolver. Both JSON and
text output identify every source they checked.

| State | What it means |
| --- | --- |
| `MANIFEST_PENDING` | Submission evidence exists (a receipt, job, or task identity) and the manifest has not landed yet |
| `NOT_SUBMITTED` / `PLAN_ONLY` | A reservation, plan, or partial staging prefix exists, but no submission evidence — so it is *not* pending |
| `VERIFICATION_UNAVAILABLE` | S3, SkyPilot, or provider verification failed. **Never** treated as absence |
| `NOT_FOUND` | Emitted only after every applicable exact source answered authoritatively |
| `CACHED` | The explicit offline mode (`--cached`). Not live-verified, and not automation-trustworthy |

Unrelated nested S3 keys are never guessed as runs.

If a shell cannot resolve the project storage location, point status at the
prefix explicitly:

```bash
npa workbench workflow status "$RUN_ID" --project "$PROJECT" \
  --workflow-s3-uri "s3://$BUCKET/<workflow>/$RUN_ID/npa-workflow"
```

## Where the kubeconfig goes

`provision-if-absent` writes the cluster kubeconfig to
`~/.npa/clusters/<context>/kubeconfig` rather than merging it into
`~/.kube/config`. `submit --infra k8s/<context>` finds that file on its own. For
`kubectl` in your own shell, export it (the command prints this line for you):

```bash
export KUBECONFIG=~/.npa/clusters/<context>/kubeconfig
```

## Related

- [Known footguns](workbench/troubleshooting/known-footguns.md) — the failures these gates are designed to catch
- [Physical AI Data Factory runbook](workbench/guides/physical-ai-data-factory-deploy.md) — the whole path, end to end
- [Workflow authoring guide](workbench/npa-workflow-guide.md) · [tool catalog](workbench/npa-workflow-tool-catalog.md)
- [Tear it all down](teardown.md) — cancellation and cleanup ordering
- [debug-failed-run skill](../skills/atomic/debug-failed-run/SKILL.md) — triage order for a run that failed or hung
