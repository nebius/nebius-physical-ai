# Run Cosmos3 PAIDF as a team

A shared Workbench user signs in, selects a LeRobot episode and camera, and
submits the administrator-supplied PAIDF workflow. The service drives Cosmos3
augmentation, hosted captioning and evaluation, curation, and visualization on
the enrolled Kubernetes cluster. The client can run on a small VDI; models and
GPU jobs run on the destination cluster.

Start with [Workbench setup](shared-workbench-quickstart.md). The
[Cosmos3 PAIDF runbook](../../workflows/guides/paidf-cosmos3.md) describes the
actual components, input formats, quality decisions, and output evidence.

## Operator: prepare accounts and inputs

For a fresh Nebius project, finish [project configuration](../configuration.md)
and [GPU cluster setup](kubernetes.md), including credential, quota, capacity,
image-access, and GPU readiness checks. Creating a project does not grant a new
human account access to it or provide GPU capacity. Keep this operator
configuration separate from each user's client state. Run provisioning and
SkyPilot readiness on the Linux operator VDI; macOS supports the HTTPS client
and offline planning but cannot run the isolated scheduler smoke.

Create one authoritative Workbench installation with durable account state.
For example, use two local groups, `aloha` and `pusht`, each granted the runner
role in its workspace. Create separate accounts for the people in each group;
give each person an explicit cluster allocation, concurrent GPU allowance,
personal namespace, and narrowly scoped storage principal. Follow
[account setup and enrollment](team-access.md#administrator-establish-local-accounts-and-policy).
Local groups are Workbench permissions; they are not synchronized from Nebius.

Prepare these additional allocation fields for PAIDF:

| Field | Operator prepares |
| --- | --- |
| `storage` | Private output bucket/prefix, endpoint, principal, and mode-0600 storage credential file |
| `workload_secrets_file` | Mode-0600 JSON with the approved `HF_TOKEN` and `NEBIUS_TOKEN_FACTORY_KEY`; add other supported provider keys only when the workload needs them |
| `source_s3_uri` | Reviewed NPA package source prefix for the workflow's `source_overlay: true` and stages without a baked workbench image |
| `shared_inputs` | Any additional read-only dataset/source prefixes, with corresponding read permission granted in cloud IAM |

Stage the source and datasets under the personal allocation's prefix or an
explicitly permitted shared input prefix. `shared_inputs` declares the boundary;
it does not create cloud permissions. A source prefix contains an installable
NPA package, including `pyproject.toml` and `src/`, rather than a tarball. Never
upload the operator's credentials, CLI cache, or virtual environment.

Check exact gated-model access with the allocated provider credential before
spending GPU time. Approve a workflow whose accelerator name, CPU request,
memory, and variant parallelism fit the actual per-node capacity and personal
GPU allowance. GPU quota alone does not establish physical capacity. A
single-GPU node also needs CPU headroom for Kubernetes system workloads.

Use version-pinned datasets and retain their cards, metadata, episode tables,
and video files. The runbook's
[LeRobot v3 example](../../workflows/guides/paidf-cosmos3.md) provides pinned
ALOHA files and checksums. Read the dataset metadata to select the camera and
confirm all requested episode indices exist. A camera name from another
dataset may be invalid: ALOHA commonly uses `observation.images.top`, while
PushT uses `observation.image`. Preserve shared-video timestamps in v3 data;
renaming an episode's video cannot replace its episode metadata.

Privately deliver the endpoint, workspace and cluster aliases, approved
workflow, input URI, camera, and either a personal key or an explicit Nebius
identity link. A person who has never used Nebius can use the personal-key
path immediately. Nebius SSO first needs that person's human account, tenant
membership, and administrator-linked immutable identity; `npa login` does not
create those cloud identities.

## User: sign in and select an episode

Install the NPA client, then use the method your operator prepared:

```bash
npa login --endpoint https://workbench.example.invalid --token-file ./workbench.key
```

For a linked human identity on a VDI, complete the official browser flow:

```bash
npa login --endpoint https://workbench.example.invalid --nebius \
  --no-browser --ssh-host my-vdi
```

Open the printed callback tunnel on your laptop before opening the sign-in link.
Keep passwords, MFA, callback codes, and tokens in that browser/CLI flow. Use
the [identity guide](team-identity.md#sign-in-once-with-npa-login) for profile
selection and a private CA. Verify the resulting allocation:

```bash
npa workbench whoami
```

Set `DATASET_URI` to the input URI delivered by your operator, then submit:

```bash
npa workbench submit --spec paidf-cosmos3.yaml \
  --var input_kind=lerobot --var lerobot_dataset_uri="$DATASET_URI" \
  --var input_camera=observation.images.top --var input_episode=0
```

Each `--var KEY=VALUE` overrides one string in `config` without changing the
YAML. The service assigns run output storage and namespace from your allocation.
Changing the episode changes the submitted document and its retry identity.
Repeating identical inputs recovers the original run. Add `--new-run` only to
deliberately repeat an acknowledged execution.

## User: process 20 episodes with durable retries

The shared SDK uses the same saved login and retry receipts as the CLI. This
example processes episode indices 0–19, waiting for each run before submitting
the next. Several users can run it independently in their own allocations.
It needs neither operator credentials nor a direct SkyPilot connection.
To divide those 20 episodes between two people, use `range(0, 20, 2)` for one
and `range(1, 20, 2)` for the other, with their separate saved logins.

```python
import copy
import os
import time
from pathlib import Path

import yaml
from npa.sdk.workbench.team import open_connection, submit_workflow

template = yaml.safe_load(Path("paidf-cosmos3.yaml").read_text())
dataset_uri = os.environ["DATASET_URI"]
camera = os.environ["INPUT_CAMERA"]


def inspect_run(run_id, action):
    client, _ = open_connection()
    try:
        return client.run(run_id, action)
    finally:
        client.close()


for episode in range(20):
    workflow = copy.deepcopy(template)
    workflow["config"].update(
        input_kind="lerobot",
        lerobot_dataset_uri=dataset_uri,
        input_camera=camera,
        input_episode=str(episode),
    )
    client, session = open_connection()
    try:
        run = submit_workflow(client, session, workflow)
    finally:
        client.close()
    run_id = run["id"]
    print(episode, run_id, flush=True)
    while True:
        status = inspect_run(run_id, "status")["status"]
        if status == "succeeded":
            print(inspect_run(run_id, "artifacts"), flush=True)
            break
        if status in {"failed", "cancelled", "recovery_required"}:
            raise RuntimeError(f"Inspect {run_id}: {status}")
        time.sleep(5)
```

Set `INPUT_CAMERA` to the actual camera in the staged dataset. Preserve the
saved connection's `requests/` receipts and the unchanged template/input values.
The example reopens the saved connection for each operation so human SSO can
refresh through the official CLI during a long batch.
Rerunning the script recovers earlier submissions, including a lost response.
Inspect a failed run's logs and artifacts before deciding on a new run. For
`recovery_required`, retain the run ID and ask the operator to reconcile its
existing scheduler state. The operator's `workflow batch` command follows a
different submission path; it does not use a saved shared Workbench login.
Status includes the latest recorded stage names and completed/recorded wave
counts once the service has persisted runtime progress. These are observations,
not a percentage estimate; refinement can add waves. They use private server
records without a cloud lookup, and the run's lifecycle status stays authoritative.
Fetching logs reports pending worker state for jobs that are still starting.
It downloads available owned wave logs, which can take longer than an ordinary
status request. Interrupting that read does not cancel the workflow.

Verify actual episode provenance, nonempty decoded generated videos, quality
dispositions, curator outputs, and Rerun evidence. A submission response is
admission evidence; it does not prove successful augmentation. This workflow
produces augmented videos and reports for the selected camera. It does not
rebuild a training-ready LeRobot dataset with action/state alignment or prove
that an appearance variant is suitable for policy training.

## User: download and inspect the result

Set `RUN_ID` to the ID returned by submission. After the run succeeds, list its
outputs with your saved login:

```bash
export RUN_ID
npa workbench run "$RUN_ID" --action artifacts
```

The command returns an index. Download the recording through the same personal
connection; the reference workflow publishes `reports/sim2real.rrd`. If your
operator changed its output path, use the identifier from the index.

```python
import os
from pathlib import Path

from npa.sdk.workbench.team import open_connection

os.umask(0o077)
run_id = os.environ["RUN_ID"]
output = Path(f"{run_id}.rrd")
client, _ = open_connection()
try:
    with output.open("xb") as destination:
        client.download(run_id, "reports/sim2real.rrd", destination)
finally:
    client.close()
print(output.resolve())
```

The exclusive write preserves any earlier download. Open the resulting file
with `rerun "$RUN_ID.rrd"` on your desktop; copy it there first if the VDI is
headless. Scrub the source and generated video timelines and read the quality,
caption, and curation panels. The recording embeds the videos, so this local
inspection uses the downloaded file and your desktop viewer.

## Operator: audit the completed batch

Operators can audit a completed two-dataset, four-account batch with
`npa/tests/e2e/test_team_paidf_live.py`. Set `NPA_INTEGRATION_E2E=1` and
`NPA_TEAM_PAIDF_LIVE_CONFIG` to an absolute, mode-0600 JSON file containing
`endpoint`, optional `ca_file`, and `accounts`. Each account supplies its private
`token_file`, dataset label, and `runs` records with `id`, `episode`, `camera`,
and `run_uri`. The test reads through each person's authenticated API access,
requires all episode indices 0–19 for each dataset, rejects cross-owner access,
and checks the canonical PAIDF artifacts and decoded videos. It does not submit
jobs. A quality rejection must be reviewed as a measured outcome; it cannot
satisfy this successful-batch gate by lowering thresholds after the run.
