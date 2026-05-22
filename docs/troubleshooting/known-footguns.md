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

Current workaround: refresh the registry token and recreate the
`npa-nebius-registry` image pull secret in the SkyPilot namespace, normally
`default`.

Category for follow-up: security.

## Literal AWS Endpoint In SkyPilot YAML

Symptom: S3 uploads fail and logs show the literal string `${AWS_ENDPOINT_URL}`
instead of `https://storage.eu-north1.nebius.cloud`.

Root cause: SkyPilot 0.12.2 does not interpolate environment variables inside
YAML `envs` blocks at submission time.

Current workaround: use `npa/scripts/run_isaac_lab_rl.py`, which materializes
endpoint values before submission, or substitute the literal endpoint value in
the YAML before submitting.

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
