# Recover a workflow whose client state is missing

[Workbench docs](README.md)

Ordinary recovery reuses the original isolated SkyPilot state and
`npa workbench workflow cancel`. Another caller's empty queue cannot prove that
the original controller stopped. A failed cancellation followed by client-queue
absence is therefore verification unavailable.

For an orphaned single-job controller, retain the original execution binding in
an owner-only JSON file outside Git. Recovery independently checks the durable
NPA ledger, live controller database, and worker metadata. It never deletes a
pod, cluster, bucket or controller. Controllers with multiple job records are
refused, including records for other users.

Construct the record from retained original evidence, cross-checked live. Never
guess a caller or transfer a numeric job ID between controllers. Hash the original
UTF-8 configuration strings; do not serialize their secret-bearing contents.

| Field | Required source |
| --- | --- |
| `schema` | `npa.workflow.controller-recovery.v1` |
| `run_id`, `project`, `workflow_s3_uri` | Original NPA submission and durable state prefix |
| `context`, `namespace` | Original execution configuration |
| `controller`, `controller_uid`, `container` | Exact original controller metadata |
| `job_id`, `name`, `submitted_at` | Original controller job ID, exact name and timestamp |
| `user_hash`, `workspace` | Original caller and workspace |
| `attempt_id` | Durable wave `logical_launch_id`, also present in the submitted task |
| `image` | Original `repository@sha256:...`, matching the live worker |
| `dag_yaml_content_sha256`, `config_file_content_sha256` | SHA-256 of original controller definition and configuration strings |

Set `KUBECONFIG` to the verified owner configuration containing that exact context.
The record must be a regular file owned by the caller with mode `0600`. First
reconcile read-only, then request cancellation of the verified exact job:

```bash
npa workbench workflow reconcile-controller "$NPA_CONTROLLER_RECOVERY_RECORD" --json
npa workbench workflow reconcile-controller "$NPA_CONTROLLER_RECOVERY_RECORD" --cancel --json
```

Repeat the second command to verify convergence. `cancel_requested=true` is only
an acknowledgement. Require `cleanup_verified=true`, terminal controller state,
and `active_workers=0`. The operation is idempotent once terminal and leaves the
controller available. Missing, ambiguous, or changed identities fail closed.

Before mutation, sanitized observations are retained in an owner-only sibling
directory and immutable content-addressed objects beneath the exact run's
`controller-recovery/` state prefix. Read-after-write verification is required;
diagnostic write failure prevents cancellation. Preserve the original failed
runtime ledger and raw evidence. Recovery does not turn a failed payload into a
successful evaluation or valid visual proof.
