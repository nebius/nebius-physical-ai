# Known Operational Footguns

These are known operational failure modes surfaced during W10 Isaac Lab BYOF
validation. They are documented here so partners can request the right operator
action before discovering each issue through a failed run.

## L40S Capacity Is On-Demand-Zero

Symptom: SkyPilot keeps backing off while trying to schedule an L40S job.

Root cause: the workbench cluster may have no provisioned L40S capacity, and
on-demand L40S availability can be zero for the target region.

Current workaround: ask your Nebius support or operations contact to provision
an L40S node group before the run. If your workflow can use another RT-core GPU
and your region has it available, use RTX Pro 6000 in US Central.

Category for follow-up: capacity.

## Default L40S Preset Has Insufficient CPU

Symptom: Kubernetes pod scheduling fails with a CPU resource error even though
an L40S node group exists.

Root cause: the default L40S preset can have less allocatable CPU than the
SkyPilot workflow request, such as a 16-CPU request.

Current workaround: ask for a larger L40S preset, or reduce the SkyPilot CPU
request in the workflow YAML when that is acceptable for the workload.

Category for follow-up: platform.

## Writable S3 But Terraform State Returns 403

Symptom: an ordinary object-storage health check passes, while Terraform fails
to read or list its remote state prefix.

Root cause: Nebius API/IAM authentication and Object Storage HMAC credentials
are separate. A root-level write probe also does not prove access to Terraform's
exact state object and prefix.

Mitigation: use the project-scoped agent preflight/deploy path. NPA HEADs/GETs
the exact state object, validates an isolated conditional sibling in the same
prefix, and carries the resolved HMAC environment through init, plan, apply,
state, output, rollback, and destroy. It fails before resource creation when the
contract is missing or forbidden. Do not broaden IAM merely to compensate for a
dropped subprocess environment.

Category for follow-up: platform.

## Registry Pull Secret Expires Silently

Symptom: the task pod fails to pull the Workbench image with a registry
authentication error such as `401 Unauthorized`.

Root cause: Nebius IAM-backed registry tokens expire, and an old
`npa-nebius-registry` pull secret can remain in the namespace.

Mitigation: the standard workflow runtime refreshes registry credentials while
submitting its SkyPilot tasks. Manual workaround if needed: refresh the registry
token and recreate the `npa-nebius-registry` image pull secret in the SkyPilot
namespace, normally `default`. The older per-stage sibling-Job refresh helper is
retained only for archived pre-standard-runtime replay and is scheduled for
removal under the Sim2Real legacy compatibility contract.

Category for follow-up: security.

## Registry Pull Returns 403 Even Though The Tag Exists

Symptom: an operator confirms the tag exists (`GET /v2/<repo>/tags/list` returns
`200`), but every worker pod fails to pull with `403 Forbidden` and the managed
job sits in `PENDING` / `ImagePullBackOff` instead of failing.

Root cause: Nebius Container Registry speaks the standard Docker Registry v2
auth flow. Listing tags and pulling a manifest are *different permissions*, and
the token endpoint issues tokens optimistically — the registry only enforces the
permission on the final manifest request. So a readable tag list proves nothing
about whether the identity the run injects can pull. Kubernetes then retries
image pulls indefinitely, so the run never fails; it just stops.

Mitigation: `npa workbench workflow submit` now reproduces each planned step's
pull with the very credentials it is about to inject and refuses to submit when
a Nebius registry image comes back `403`. Run it standalone with:

```bash
npa workbench workflow preflight-images <spec.yaml>
```

The fix is to grant the run's identity pull access to that repository. Pass
`--no-preflight-images` to skip the gate.

Category for follow-up: security.

## Submit Hangs With No Output (SkyPilot Controller pod_config Loop)

Symptom: `npa workbench workflow submit` prints nothing for 15+ minutes after
`deployIfAbsent` and never returns.

Root cause: SkyPilot 0.12.2 declares `kubernetes!=32.0.0,>=20.0.0` with no upper
bound, so a fresh install resolves the newest client. Client `36.0.0` renamed the
generated `openapi_types` entries from `dict(str, str)` to `dict[str, str]`, which
SkyPilot's pod validator turns into an import of
`kubernetes.client.models.dict[str, str]`. Every `pod_config` then fails
validation and the managed-jobs controller retries forever.

Mitigation: `npa skypilot bootstrap` pins `kubernetes>=20.0.0,!=32.0.0,<36` and
repairs an already-bootstrapped venv in place; `npa skypilot status` reports the
installed client and fails when it is out of range. Submit also streams SkyPilot's
output live and names this failure when it appears, instead of buffering silently.

Category for follow-up: platform.

## Requesting `NAME:2` On A Fleet Of 1-GPU Nodes

Symptom: a job fails `FAILED_PRECHECKS`, or never schedules, on a cluster that
clearly has enough GPUs in total.

Root cause: SkyPilot places all GPUs of one task on a **single node**. A cluster
of 2 nodes × 1 GPU can never satisfy `NAME:2`, and adding nodes does not help.
Multi-GPU fan-out documentation assumes N GPUs on one pod, which is a different
cluster shape than "N single-GPU node presets".

Mitigation: `npa workbench workflow gpus --cluster <name>` prints each
accelerator's *requestable quantity per node*. Submit rejects a request above that
maximum with a one-line fix rather than letting it fail later.

Category for follow-up: platform.

## Kubernetes GPU Names Do Not Match Workflow Specs

Symptom: a spec pinned to `RTXPRO6000` fails prechecks, and `sky gpus list`
reports a different name — sometimes `RTX6000`, later
`RTXPRO-6000-BLACKWELL-SERVER-EDITION`.

Root cause: on Kubernetes, SkyPilot derives accelerator names from node labels.
Nebius sets `nebius.com/gpu-name`, and the NVIDIA GPU operator later adds
`nvidia.com/gpu.product`, so the advertised name changes as labelling completes.

Mitigation: submit resolves the spec's accelerator against what the cluster
advertises and remaps it automatically. Run
`npa workbench workflow gpus --cluster <name>` to see the names yourself, and
`--no-resolve-accelerators` to submit the spec's values verbatim.
## Controller Teardown Refuses Right After Workflow Cancel

Symptom: `npa workflow cancel <run-id> --project <alias>` succeeds,
and immediate shared-controller teardown reports that managed jobs are still
in progress.

Root cause: managed-job cancellation returns as soon as cancellation is
*scheduled*. The controller keeps reporting the job as `CANCELLING` for a while, and teardown
refuses while any managed job is non-terminal.

Mitigation: NPA resolves the canonical run prefix, reads authoritative workflow
and per-stage/runtime-wave state, and cancels every exact non-terminal managed
job ID. A job becoming terminal or absent during cancellation is successful
convergence; malformed/ambiguous/auth/partial failures remain errors. Once every
workflow is terminal, run `npa skypilot cleanup-controller --project <alias>
--context <context> --yes`; it waits for the queue to drain and retries the
specific controller refusal. The explicit/selected project and exact NPA-saved
context are cross-checked against immutable provider identity. Remote deletion
runs without changing real local SkyPilot state; NPA independently proves the
controller absent, durably checkpoints that evidence, and only then converges
matching local metadata. An ambient current context, first/stale profile, or
auth/RBAC/connectivity uncertainty is never treated as proof. Planned/staged runs that never launched report
`already_absent` and remain repeat-safe without calling SkyPilot.

Category for follow-up: platform.

## A Managed Job Sits In PENDING Forever

Symptom: a managed job stays `PENDING` for hours and never becomes `FAILED`,
burning wall-clock until somebody cancels it by hand.

Root cause: Kubernetes retries image pulls and pod scheduling indefinitely, so a
worker pod that cannot start never fails — and SkyPilot keeps reporting the job as
`PENDING`. There is no signal distinguishing "slow to start" from "will never
start".

Mitigation: `npa workbench workflow status` inspects the pods behind a `PENDING`
job (via SkyPilot's own `skypilot-cluster-name` label) and reports the container's
waiting reason — `ImagePullBackOff`, `Unschedulable`, `CreateContainerConfigError`
— with a remedy. `npa cleanup` also lists non-terminal managed jobs, since one of
them will block controller teardown.

Category for follow-up: platform.

## Cluster Teardown Goes Quiet For Several Minutes

Symptom: `npa cluster down` appears to hang while draining a node, then completes
after roughly five to seven minutes.

Root cause: draining respects PodDisruptionBudgets. Platform add-ons such as
`coredns`, `coredns-autoscaler`, `cilium-operator`, and `metrics-server` can
exhaust their disruption allowance on a one-node/default CPU pool because no
spare node exists for a healthy replacement. A preview that inherits the wrong
kubeconfig can also try browser authentication or fail RBAC before it can explain
which protections were checked.

Mitigation: `npa cluster down` selects NPA's saved kubeconfig for the target
cluster, runs its exec credential in non-interactive/no-browser mode, and
distinguishes authentication, authorization, kubeconfig, and API failures. Those
failures affect only the best-effort preview; Terraform destroy is still attempted,
and NPA says explicitly that PDB safety was not verified. The successful preview
uses one cluster-wide inventory of nodes, pods, their controllers, and every PDB,
then applies selector, placement, health, `disruptionsAllowed`, and unhealthy-pod
policy semantics. It reports which workload blocks which node and why the
one-node pool cannot temporarily satisfy it. NPA requests normal eviction first.
For only an explicitly confirmed whole-cluster destroy with exact NPA project,
context, cluster, and provider identity, it can then temporarily remove the
exact kube-system PDBs for cilium-operator, CoreDNS, the CoreDNS autoscaler, and
metrics-server. It snapshots each object and restores its exact spec if destroy
aborts while the cluster remains. Shared clusters, node-pool operations,
unverified contexts, and user/application budgets are never weakened or
force-deleted.

During node-group reconciliation, a `ComputeInstanceDeletionFailed` event whose
detail confirms `NotFound` means the instance is already absent; NPA reports that
race as idempotent progress. The same event with PermissionDenied or any other
real deletion failure remains visible verbatim.

Category for follow-up: platform.

## No-Cluster Teardown Downloads Terraform Providers Into The Checkout

Symptom: `npa cluster down --force` runs authentication and `terraform init` even
though no cluster, Terraform resource state/inventory, or NPA kubeconfig exists;
`deploy/cluster/.terraform` can grow by hundreds of MB. A later provider download
may also fail against tracked lock-file checksums.

Mitigation: the no-evidence path is now a clear no-op and crosses no Terraform,
Nebius-auth, Kubernetes, or RBAC boundary. Real apply/destroy runs use an exact
marked directory under `~/.npa/terraform-data/cluster/` as `TF_DATA_DIR` and
remove it on every exit path. Interrupted apply records a non-secret lifecycle
inventory immediately before apply, so a later down is not incorrectly skipped.
After configuration removal, use `--receipt <opaque-id>` or both exact
`--project-id` and `--cluster-id`. Verified provider absence is also a no-op;
insufficient identity and a present cluster without recoverable owned state fail
before binary lookup or `.terraform` creation.
`npa cleanup --full --yes` detects/removes marked scratch and the exact validated
legacy source cache; a failed removal stays visible on the next report.

Checksum verification is never disabled and runtime init uses
`-lockfile=readonly`. On mismatch, verify the provider source/release and mirror,
then reconcile in a clean reviewed checkout (replace the platform deliberately):

```bash
terraform -chdir=deploy/cluster providers lock -platform=linux_amd64
git diff -- deploy/cluster/.terraform.lock.hcl
```

Do not delete the lock file or use an unverified provider package merely to make
teardown proceed.

Category for follow-up: platform.

## Teardown Is Seven Ordered Steps With No Single Entry Point

Symptom: an environment looks torn down but still has a hung managed job, a local
SkyPilot venv, empty `~/.npa/agents` / `~/.npa/clusters` directories, or an IAM
service account nothing removed.

Root cause: teardown spans cancel → agent destroy → cluster down → bucket delete →
owned storage-IAM delete → forget project → remove local state, and nothing checks
the order or reports what is left.

Mitigation: `npa cleanup` reports residual local state (with sizes), empty per-alias
state directories, and any managed job still non-terminal — the step most often
missed, because such a job keeps the jobs controller alive — then prints the ordered
runbook. `npa cleanup --yes` removes the local caches and clears
`skypilot.sky_bin` from `config.yaml` (`--keep-sky` keeps `~/.sky`) while
preserving credentials. The deliberately broader `npa cleanup --full --yes`
also removes the locally saved Hugging Face, Token Factory, and NGC entries and
prunes empty `config.yaml`, `clusters/`, and `~/.npa`. It also removes only
validated NPA Terraform scratch/legacy `deploy/cluster/.terraform` residue and
performs a read-only storage-IAM verification; non-empty, unrelated, ambiguous,
or symlinked paths are preserved. Cleanup already owns the isolated SkyPilot
venv, so there is no dead `npa skypilot uninstall` step afterwards. It
intentionally does not own the repository-local environment containing the
running `npa` command. Preview that separate scope with `npa uninstall`; actual
removal requires both `--remove-environment --yes` and is deferred until the
invoking process exits.

Managed jobs are audited and written to a versioned, atomic, non-secret receipt
before `~/.sky` or the isolated SkyPilot venv can be removed. Active jobs,
provider uncertainty, or receipt-write failure preserves the operational state
needed to cancel/verify them. Receipts remain under
`~/.npa/teardown-receipts/` after config/resource removal and are audit evidence,
not operational residue. `npa cleanup --list-receipts` lists them; only old
receipts whose every phase is terminal can be explicitly pruned with
`--prune-receipts --receipt-retention-days <days> --yes`.
Receipt v2 additively retains immutable non-secret recovery identity. Exact flags
take precedence over a receipt, which takes precedence over live config; any
overlap conflict fails before action. `configure --forget-project` prints the
receipt ID before removing the stanza, and every recovery command accepts that
opaque selector rather than a receipt filesystem path.

Cleanup JSON separates `operational_residue_present`,
`audit_receipts_retained`, and `verification_unresolved`. The receipt file is
never operational residue by itself, even when an unresolved event inside it
correctly leaves operator action outstanding.

Cloud IAM stays explicit. Configure records provenance only when its create call
made `lerobot-training`, before the next fallible provider/configuration step.
An interrupted or partially rolled-back setup keeps that non-secret journal for
retry/teardown; access-key secrets are never journaled or requested by list
inventory. After bucket deletion,
`npa storage service-account delete --project <alias> --dry-run` shows the exact
account/access keys and `--yes` removes them. An ID or familiar account name alone
is not ownership proof, so legacy, reused, mismatched, and user-managed identities
are left untouched. Bucket credentials and IAM provenance have separate lifecycle
records: deleting the bucket cannot erase the `storage_iam` proof or a legacy ID,
and agent bootstrap cannot replace the owned storage identity with `npa-agent`.
Verified absence/deletion exits 0. No trustworthy ownership (including a matching
name with no provenance) and provider/auth verification failures are
operator-actionable partial cleanup and exit 2. Use the project ID if the alias
was already forgotten:

```bash
npa storage service-account delete --project-id <project-id> --dry-run
npa storage service-account delete --project-id <project-id> --yes
npa cleanup --full --yes --project <alias>
```

For alias-free journaling, also pass `--id <exact-service-account-id>` and, when
required by scope verification, `--tenant-id`/`--profile` or a receipt. Exact
NotFound is verified absence; auth, RBAC, network, and parse failures remain
unresolved. NPA never recreates a project stanza merely to record the result.

Category for follow-up: platform.

## Raw Access-Key List JSON Can Disclose The Secret

Symptom: ordinary `nebius iam v2 access-key list --format json` output contains
an access-key secret that an operator expected only the explicit secret endpoint
to return. Piping that object through `jq` does not undo the disclosure: the raw
response has already crossed the process pipe and may reach tracing, debug, or
error capture.

Ownership boundary: this response shape is behavior of the external Nebius CLI
and API, not an NPA implementation. NPA does not patch or claim to fix that
binary. Every NPA-owned inventory/cleanup path instead asks the CLI to select
only IDs, names, service-account references, state, and expiry with JSONPath
before stdout is written. Provider diagnostics are defensively redacted as a
second line of protection.

For a human-readable inventory, use the CLI's supported output field selection
and do not enable debug output:

```bash
nebius iam v2 access-key list --parent-id <project-id> --all \
  --format 'jsonpath={range .items[*]}{.metadata.id}{"\t"}{.metadata.name}{"\t"}{.status.state}{"\n"}{end}'
```

This deliberately cannot recover a secret. If a workload no longer has the
secret saved through its creation flow, rotate/create a key rather than listing
raw JSON. See the Nebius CLI's
[JSONPath output documentation](https://docs.nebius.com/cli/jsonpath-output).

Category for follow-up: upstream CLI + platform mitigation.

## A Default Security Group Is Deleted With Its Owned Parent Network

Symptom: Nebius rejects direct deletion of a network's default security group.
Retrying `nebius vpc security-group delete` cannot make that lifecycle valid;
Nebius documents that only non-default security groups are directly deletable.

Mitigation: use the existing owner-level cleanup action. `npa agent destroy
--project <alias> --name <name> --yes` and `npa cluster down --force` tear down
the complete NPA-owned network. If agent Terraform encounters the provider's
specific default-group refusal, it resolves the parent network only from that
stack's Terraform state, deletes the proven NPA-owned parent, and reconciles the
already-absent child resources. An absent network/security group is safe to
retry.

For an existing subnet, reused network, shared network, or any stack without
Terraform ownership proof, NPA does not broaden deletion. It preserves the
network and explains that only the network owner may remove the parent network;
unrelated permission, dependency, and non-default security-group failures remain
ordinary failures. See Nebius'
[security-group deletion rules](https://docs.nebius.com/vpc/security-groups/manage#deleting-security-groups).

Category for follow-up: platform.

## Literal AWS Endpoint In SkyPilot YAML

Symptom: S3 uploads fail and logs show the literal string `${AWS_ENDPOINT_URL}`
instead of `https://storage.eu-north1.nebius.cloud`.

Root cause: SkyPilot 0.12.2 does not interpolate environment variables inside
YAML `envs` blocks at submission time.

Mitigation: reference Isaac Lab / BYOF SkyPilot YAMLs now ship the concrete
`https://storage.eu-north1.nebius.cloud` endpoint, and
`npa/scripts/run_isaac_lab_rl.py` always materializes `AWS_ENDPOINT_URL` /
`NEBIUS_S3_ENDPOINT` before submit. Prefer the runner for custom endpoints.

Category for follow-up: docs + platform.

## Sky Check Reports HTTP 403 Anonymous

Symptom: `sky check` cannot connect to Kubernetes and reports an HTTP 403 for
an anonymous user.

Root cause: the active local kube context is missing, expired, or not
authenticated against the Nebius managed Kubernetes cluster.

Current workaround: refresh the Nebius MK8s credentials, select the correct
kube context, and verify access with
`kubectl auth can-i create pods -n default`.

Category for follow-up: docs.

## Deploy Reports Replacement Required

Symptom: `npa workbench <tool> deploy` stops after Terraform planning and
reports that critical resources would be replaced or destroyed.

Root cause: the requested change affects infrastructure that cannot be updated
in place, such as the VM, boot disk, network, subnet, or security group.

Current workaround: if replacement is intentional, rerun with `--replace` and
use `--yes` for non-interactive automation. For environment-only updates, use
the tool's in-place deploy or `reload-env` path instead of replacing the VM.

Category for follow-up: deploy safety.

## BYOVM Live Commands Use SSH Fallback

Symptom: BYOVM deploy succeeds through SSH-local health checks, but a later
`status`, `serve`, `infer`, or FiftyOne app command would historically time out
against the public endpoint when public ports were blocked.

Root cause: public endpoint reachability can differ from SSH reachability on
partner BYOVM hosts.

Current behavior: BYOVM aliases record `endpoint_strategy: ssh_fallback` when
deploy health checks use SSH. Live commands honor the saved strategy and can
self-heal legacy aliases by falling back through a transient SSH-local route.

Category for follow-up: BYOVM networking.

## Xet Transfer Rejects Gated Cosmos Downloads

Symptom: a Hugging Face download of a gated Cosmos repo fails with `Unable to
parse string as hex hash value`, with a valid token and an accepted license.

Root cause: huggingface/xet-core#895 breaks the Xet transfer client on exactly
`huggingface_hub==1.23.0` plus `hf-xet==1.5.1`. Newer releases fix it.

The current `npa-cosmos3:1.2.2-cu130-r6` image bakes the compatible
`huggingface_hub==0.36.2` / `hf-xet==1.3.2` pair and its build rejects the
known-bad pair. In other runtimes, set `HF_HUB_DISABLE_XET=1` and retry or
upgrade the pair. `npa workbench cosmos3 generate` warns on stderr only for the
exact bad pair while Xet is enabled (`HF_HUB_DISABLE_XET=0` therefore still
warns). The warning stays silent when the workaround is active. See
`docs/workbench/cosmos3-access-preflight.md`.

Category for follow-up: dependencies.
