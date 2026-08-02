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

## Registry Pull Secret Expires Silently

Symptom: the task pod fails to pull the Workbench image with a registry
authentication error such as `401 Unauthorized`.

Root cause: Nebius IAM-backed registry tokens expire, and an old
`npa-nebius-registry` pull secret can remain in the namespace.

Mitigation: Sim2Real sibling Kubernetes Jobs now call
`ensure_registry_pull_secret_for_images()` immediately before each `kubectl
apply`, in addition to the initial `k8s_submit` refresh. Manual workaround if
needed: refresh the registry token and recreate the `npa-nebius-registry`
image pull secret in the SkyPilot namespace, normally `default`.

Category for follow-up: security.

## `sky down` Refuses Right After `sky jobs cancel`

Symptom: `sky jobs cancel -a` succeeds, and an immediate `sky down` of the jobs
controller fails with `NotSupportedError: In-progress managed jobs found. To avoid
resource leakage, cancel all jobs first` — telling the operator to do what they
just did.

Root cause: `sky jobs cancel` returns as soon as cancellation is *scheduled*. The
controller keeps reporting the job as `CANCELLING` for a while, and `sky down`
refuses while any managed job is non-terminal.

Mitigation: NPA's teardown now waits for cancelled jobs to reach a terminal state
before running `sky down`, recognizes this specific error, and retries once the
queue drains. If a job genuinely will not drain, teardown names the job ids and
says to wait for `sky jobs queue --all` to show them terminal.

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

Root cause: draining respects PodDisruptionBudgets. Single-replica platform add-ons
(`coredns`, `cilium-operator`, `metrics-server`) declare budgets that allow zero
disruptions on a small node pool, so eviction retries until their pods reschedule.
It is expected, but indistinguishable from a hang.

Mitigation: `npa cluster down` now previews the budgets that currently allow no
evictions and says the wait is expected, so the silence is bounded and explained.

Category for follow-up: platform.

## Teardown Is Six Ordered Steps With No Single Entry Point

Symptom: an environment looks torn down but still has a hung managed job, a local
SkyPilot venv, empty `~/.npa/agents` / `~/.npa/clusters` directories, or an IAM
service account nothing removed.

Root cause: teardown spans cancel → agent destroy → cluster down → bucket delete →
forget project → remove local caches, and nothing checks the order or reports what
is left.

Mitigation: `npa cleanup` reports residual local state, configured project entries,
non-terminal managed jobs, and the service accounts `npa configure` creates, then
prints the ordered runbook. `npa cleanup --yes` removes the purely-local caches
(`--keep-sky` keeps `~/.sky`). It never deletes cloud resources, and it never
deletes service accounts — `lerobot-training` in particular is frequently shared
with unrelated work in the same project, so it is reported for you to remove with
the Nebius CLI if it really is unused.

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
