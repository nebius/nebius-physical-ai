# BDD100K SkyPilot Pipeline

This cookbook describes the SkyPilot workflow at
`npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml`.

> This pipeline reproduces LanceDB's autonomous-vehicle perception walkthrough on
> Nebius Physical AI Workbench (adding a FiftyOne/Voxel51 review stage). See
> LanceDB's [Unifying the AV ML Stack](https://www.lancedb.com/blog/unifying-the-av-ml-stack-lancedb)
> blog and the [lancedb/training object-detection](https://github.com/lancedb/training/tree/main/object-detection)
> reference code.

The workflow composes the BDD100K reproduction stages:

1. Import the BDD100K subset into LanceDB with `POST /import-bdd100k`.
2. Backfill CPU UDF columns with five sequential `POST /backfill` calls.
3. Backfill `clip_embedding` with `POST /backfill` on an H100 task.
4. Create the three failure-mode materialized views with `POST /create-mv`.
5. Train one detector per failure-mode view with `POST /train`.
6. Evaluate each trained detector with `POST /eval`.
7. Launch a FiftyOne App on `--address 0.0.0.0 --port 5151` with the SkyPilot port exposed for public review.

SkyPilot 0.12.2 supports serial pipelines and all-parallel job groups, but not
mixed dependency graphs in one YAML. This pipeline therefore serializes the
three training tasks and three evaluation tasks. The logical DAG is still:

```text
ingest -> CPU backfill -> CLIP backfill -> materialized views -> training x3 -> eval x3 -> FiftyOne app
```

## Prerequisites: Provision Infrastructure

The pipeline tasks assume the object store, the Kubernetes cluster with GPU
node groups, and the in-cluster workbench services already exist. The repo
provisions all of these; provision them in this order before running any live
submission. The `--mock-endpoints` dry validation below needs none of this
infrastructure and is the right first step.

### 0. Operator setup

Complete the platform quickstart and workbench getting-started first. They cover
`npa` install, Nebius CLI auth, `~/.npa/credentials.yaml`, the `kubectl`/Docker/
Terraform/AWS-CLI tooling, and the SkyPilot runtime bootstrap.

- [docs/quickstart.md](../../quickstart.md)
- [docs/workbench/getting-started.md](../getting-started.md)
- [docs/orchestration/skypilot-setup.md](../../orchestration/skypilot-setup.md)

Export the non-secret identifiers used throughout this cookbook:

```bash
export NPA_S3_BUCKET=<your-bucket>            # bucket name only, no s3:// prefix
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
export NPA_STORAGE_ENDPOINT=storage.eu-north1.nebius.cloud
```

### 1. Object store (S3 bucket)

The pipeline reads raw frames from and writes every artifact under the bucket in
`NPA_S3_BUCKET`. Creating the bucket itself is a Nebius account action (Console
or `nebius storage bucket create`); the access keys belong in
`~/.npa/credentials.yaml` under `storage.*`, as described in getting-started.

Verify the bucket is reachable:

```bash
aws s3 ls "s3://${NPA_S3_BUCKET}/" --endpoint-url "${AWS_ENDPOINT_URL}"
```

For a real-data run, stage a 3-5k frame BDD100K subset in standard BDD100K
format at the source prefix the YAML expects:

```text
s3://${NPA_S3_BUCKET}/raw-bdd100k/subset-demo/
```

For a first run, skip staging and use `--synthetic`, which generates rows in the
ingest service instead of reading the source prefix.

### 2. Kubernetes cluster and GPU node groups

`npa cluster up` provisions a Managed Kubernetes cluster from
`deploy/cluster` with the NVIDIA GPU Operator, the Nebius Network Operator, a
default StorageClass, and a default GPU node group. Copy
`deploy/cluster/terraform.tfvars.example` to `terraform.tfvars` and set your
`tenant_id`, `parent_id`, `region`, and `subnet_id` first.

```bash
npa cluster up --terraform-dir deploy/cluster
```

This writes a kubeconfig under `~/.npa/clusters/<cluster-name>/kubeconfig`,
validates GPU nodes and the default StorageClass, and runs a SkyPilot GPU smoke
task. The default node group in `terraform.tfvars.example` is RTX PRO 6000
(`gpu-rtx6000`).

The CLIP-embedding, training, and evaluation stages request `H100:1`. If your
default node group is not H100, attach an H100 node group so SkyPilot can place
those tasks:

```bash
npa cluster node-group add \
  --cluster-name npa-cluster \
  --gpu-type h100 \
  --node-count 1
```

The CPU stages (ingest, CPU backfill, materialized views) request only CPU and
schedule on any node. Confirm SkyPilot can place pods and that the registry pull
secret exists in the namespace SkyPilot uses (normally `default`):

```bash
export KUBECONFIG=~/.npa/clusters/npa-cluster/kubeconfig
kubectl auth can-i create pods -n default
kubectl get secret npa-nebius-registry -n default
```

### 3. In-cluster workbench services

Each pipeline task calls existing workbench services over HTTP. Deploy them into
the `workbench` namespace so the in-cluster DNS names in the YAML resolve:

```text
http://npa-lancedb.workbench.svc.cluster.local:8686
http://npa-detection-training.workbench.svc.cluster.local:8790
```

Deploy detection-training with the output prefix it writes checkpoints under:

```bash
npa workbench detection-training deploy \
  --namespace workbench \
  --output-path "s3://${NPA_S3_BUCKET}/bdd100k-pipeline/"
```

Deploy LanceDB so the ingest, backfill, and materialized-view stages can reach
it. The container and cloud paths are production-ready today; the in-cluster
`workbench`-namespace service deploy is still partly operator-owned in this
build (see the "Known Limitation" section of
[lancedb-deploy-runbook.md](lancedb-deploy-runbook.md)). Until that path is a
one-command deploy, provide a reachable LanceDB endpoint and override it at
submission time:

```bash
python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000 \
  --lancedb-endpoint http://<your-lancedb-endpoint>:8686 \
  --run-id <your-run-id>
```

See [lancedb-deploy-runbook.md](lancedb-deploy-runbook.md) for deploy runtimes,
auth, and storage, and [lancedb-vector-search.md](lancedb-vector-search.md) for
table, query, and import usage.

### Teardown

After the demo, remove the GPU node group and the cluster to stop GPU spend:

```bash
npa cluster node-group remove --cluster-name npa-cluster --name <node-group-name>
npa cluster down --terraform-dir deploy/cluster
```

Object-store artifacts persist independently; delete the run prefix under
`s3://${NPA_S3_BUCKET}/bdd100k-pipeline/<run-id>/` when you no longer need it.

## Dry Validation

The mock-endpoint path validates the full pipeline — every task's `run:` script,
the curl/jq request plumbing, and the request ordering — with **no cloud, GPU,
or credentials**. It is the reproducible first step and the basis of the
recorded demo below.

First install the package into the repo virtualenv (one time):

```bash
python3 -m venv npa/.venv
npa/.venv/bin/pip install -e npa
```

Then run the dry validation. Writing `--output-json` gives a machine-checkable
summary in addition to stdout:

```bash
npa/.venv/bin/python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000 \
  --mock-endpoints \
  --run-id demo-validate \
  --output-json /tmp/bdd100k-validation.json
```

Expected result: exit code `0`, all 11 tasks return `0`, no failures, and the
recorded request order is
`import-bdd100k -> 6x backfill -> 3x create-mv` (LanceDB) and
`3x train -> 3x eval` (detection-training). Confirm the summary:

```bash
npa/.venv/bin/python - <<'PY'
import json
d = json.load(open("/tmp/bdd100k-validation.json"))
print("tasks:", len(d["task_results"]), "all rc==0:",
      all(t["returncode"] == 0 for t in d["task_results"]))
print("failures:", d["failures"])
PY
```

### Record the validation

To capture a shareable recording of the validation (and the full unit-test
suite) run it under `script(1)`:

```bash
script -q -e -c '
  npa/.venv/bin/python -m pytest npa/tests/ --ignore=npa/tests/e2e --timeout=120 -q
  npa/.venv/bin/python npa/scripts/run_bdd100k_pipeline.py \
    --yaml npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
    --synthetic 5000 --mock-endpoints --run-id demo-recording
' /tmp/bdd100k-demo-recording.log
```

View the recording (it is a plain text typescript; `script` inserts carriage
returns, so strip them when reading in a pager):

```bash
# Quick read
cat /tmp/bdd100k-demo-recording.log

# Clean view in a pager (strips CR control characters)
col -b < /tmp/bdd100k-demo-recording.log | less

# Just the pass/fail summary lines
grep -E 'passed|DEMO STATUS|rc=' /tmp/bdd100k-demo-recording.log
```

## Full Submission

Full submission requires a working SkyPilot 0.12.2 binary:

```bash
export NPA_SKYPILOT_BIN=/opt/npa/skypilot/bin/sky
python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000 \
  --run-id bdd100k-pipeline-$(date -u +%Y%m%dT%H%M%SZ) \
  --cleanup
```

The wrapper renders run-specific S3 paths before calling
`npa.orchestration.skypilot.submit_workflow`.

## Required Services

The task pods call the in-cluster workbench services deployed in
[Prerequisites step 3](#3-in-cluster-workbench-services) over HTTP. Override
these defaults if the service names differ:

```bash
--lancedb-endpoint http://npa-lancedb.workbench.svc.cluster.local:8686
--detection-endpoint http://npa-detection-training.workbench.svc.cluster.local:8790
```

The default input source is
`s3://${NPA_S3_BUCKET}/raw-bdd100k/subset-demo/`. Full submission requires the
configured S3 credentials to list and read this prefix.

## Images

The committed YAML pins the first-party LanceDB and detection-training images:

- `cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-lancedb:0.30.3`
- `cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-detection-training:bdd100k-golden-eval-smoke-20260614T210000Z`

The optional final FiftyOne app can still be replaced with a BYO registry image:

- `cr.eu-north1.nebius.cloud/<your-registry-id>/npa-fiftyone:<fiftyone-image-tag>`

The final FiftyOne task exposes port `5151` through SkyPilot. The app does not add authentication; restrict the run inputs to datasets that are safe to show publicly and use `sky status --endpoint 5151 <cluster>` to resolve the public URL.

## Output Layout

For `run_id=<run-id>`, outputs are rooted at:

```text
s3://${NPA_S3_BUCKET}/bdd100k-pipeline/<run-id>/
```

The LanceDB URI is `<root>/lancedb/`. Training outputs are under
`<root>/training/<view-name>/`, and evaluation outputs are under
`<root>/eval/<view-name>/`.
