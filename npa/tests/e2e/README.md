# Live tests

## Existing cluster adoption and profile selection

Run the profile regression against an existing cluster the operator can read:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_PROJECT='<configured-project-alias>' \
NPA_E2E_ADOPT_CLUSTER_NAME='<existing-cluster-name>' \
NPA_NEBIUS_PROFILE='<verified-nebius-profile>' \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_cluster_adoption_profile_live.py -q
```

All four variables are required; the test skips without them. There is no default
target. The test reads the selected project's configuration and the live cluster,
then fetches a temporary kubeconfig with a deliberately conflicting
`NEBIUS_PROFILE`. It verifies that the generated credential plugin retains
`NPA_NEBIUS_PROFILE`. Child-process NPA state stays under pytest's private
temporary directory; the test creates no cloud resources or GPU jobs.
Set `NPA_CONFIG_DIR` to select an isolated operator configuration.

## PAIDF native variant recovery

After executing the reviewed candidate source through the standard PAIDF runtime,
run the read-only receipt audit with `NPA_INTEGRATION_E2E=1` and
`NPA_PAIDF_VARIANT_RECOVERY_LIVE_CONFIG` pointing to an owner-only JSON file outside
Git. The configuration has `project`, `run_uri` (the exact run prefix ending in
`/`), `run_id`, `fresh_after` (an aware UTC timestamp recorded before submission),
`input_sha256`, `runtime_image` (an immutable reference), `variant_count`, and
`require_recovery` (a boolean). Set the latter to true to require actual reuse
after worker recovery. Neither variable has a default target.

```bash
NPA_INTEGRATION_E2E=1 \
NPA_PAIDF_VARIANT_RECOVERY_LIVE_CONFIG="$PRIVATE_RECOVERY_CONFIG" \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_paidf_variant_recovery_live.py -q
```

The test reads only that run's exact source, manifest and new per-variant
receipts. It verifies full video decoding, all object bytes, native controls,
guardrails, immutable model binding and actual source fingerprints. It does not
submit, cancel, infer, repair infrastructure or publish objects. Run it from the
same reviewed checkout staged to the workload. Fixture tests and an audit of an
older source without these receipts do not establish native recovery acceptance.

## SkyPilot GPU bridge labels

Use an owner-only JSON configuration containing `context`, `kubeconfig`,
`accelerator`, `cpus`, `memory`, and `evidence_path`. The evidence destination's
parent directory must also be owner-only. Select the exact authorized cluster
and keep its identifiers and evidence outside Git.

```bash
NPA_INTEGRATION_E2E=1 \
NPA_SKYPILOT_GPU_LABEL_LIVE_CONFIG="$PRIVATE_GPU_LABEL_CONFIG" \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_skypilot_gpu_label_live.py -q
```

The tests read actual node and pod metadata. They verify reviewed bridge labels
against the effective accelerator request and independently check actual free
GPU gang capacity. The latter skips when a running workload occupies the GPU;
a successful label check does not prove free placement or native inference.
No node labels, cloud resources, pods, or workloads are changed.

Run the Insights storage tests against an explicitly selected project's own
configured credentials:

```bash
NPA_INTEGRATION_E2E=1 NPA_E2E_PROJECT=<project-alias> \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_insights_storage_live_e2e.py -q
```

Both variables are required; tests skip without them. No infrastructure is
provisioned. Fixtures write and delete only a unique test-owned prefix and
verify removal of current objects, object versions, and delete markers.
Set `NPA_CONFIG_DIR` to use an isolated operator configuration.

Set `NPA_E2E_INSIGHTS_BUCKET_ROOT=1` to also verify JSONL discovery from both
bucket-root URI forms. This additional opt-in has no default and enumerates
key metadata across the named bucket. It does not read unrelated object
bodies, and its writes and deletions remain inside the unique fixture prefix.
See [the test module](test_insights_storage_live_e2e.py) for the full contract.

## Soperator user namespaces

After deploying a dedicated Soperator cluster with the default unconfined
worker profile, verify its actual host setting and non-root user-namespace
creation on every worker:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_SOPERATOR_CONTEXT=<dedicated-kube-context> \
NPA_E2E_SOPERATOR_WORKER_COUNT=<exact-worker-count> \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_soperator_userns_live.py -q
```

All three variables are required; the test skips without them. The test creates
no cloud resources or GPU jobs. It runs a short namespace probe as the existing
non-root `soperatorchecks` identity in each worker container. It does not change
host settings and does not replace the deployment's CUDA creation checks.
`NPA_E2E_SOPERATOR_NAMESPACE` defaults to `soperator`.
`NPA_E2E_SOPERATOR_EVIDENCE_DIR` optionally names a private output directory for
a new `userns-live.json` receipt; an existing receipt is never overwritten.

### Provisioning boolean ingress

`test_agent_provision_boolean_live.py` sends malformed boolean fields to all
three provisioning route aliases on an isolated, authenticated agent. Every
request must return HTTP 400 with `ok: false`, `status: invalid`, and a field
error. These rejected requests must not issue confirmations or start provisioning;
the rendered-backend unit tests separately verify those side-effect boundaries.

Set `NPA_AGENT_PROVISION_BOOLEAN_LIVE_CONFIG` to an owner-only JSON file outside
the checkout containing `base_url` (the agent API URL ending in `/api/`),
`username`, and `password`. TLS verification uses system trust; for a private
certificate authority, set optional `ca_bundle` to its PEM bundle path. This
variable has no default; the test skips when
it is absent. Deploy the candidate to a disposable CPU agent in an isolated
project first, then run with `NPA_INTEGRATION_E2E=1`. Destroy the exact test agent
and its owned storage after capturing evidence, following the teardown skill.
