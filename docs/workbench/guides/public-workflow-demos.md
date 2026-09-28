# Run the four public workflow demos

Use the same commands to reconstruct a scan and train a navigation policy,
generate industrial sensor data, compare a continued policy, or reconstruct a
photographic scene. Each demo prepares its public sample automatically and
writes a compact HTML report alongside the full artifacts.

These reference experiments use public data. They do not qualify a private
robot or site. Navigation policies use range observations; the four-camera
synthetic-data experiment is a separate sensor-generation workflow.

## Configure once

Follow [Workbench setup](../getting-started.md) to configure a project, its
object-storage credentials, and an RTX PRO 6000 Kubernetes target. The shipped
demo presets select RTX PRO 6000; the recorded native qualification uses that
GPU. An L40S target requires an explicit accelerator override and its own
qualification.
NuRec additionally requires NVIDIA NGC access. Isaac runtime downloads use the
existing Workbench EULA policy; optional telemetry remains off.

Install the complete combined demo revision. A checkout containing only one of
the original implementation PRs does not contain all the required adapters.
The launcher stages the installed source automatically for every run.

```bash
npa workbench workflow demo list
npa workbench health preflight --project '<project>' --checks s3,nebius
```

For NuRec, also run `npa workbench health access --capability nurec` before
starting the GPU workload. Standard workflow submission checks exact storage
destinations, image access, source staging, and the requested GPU product.

The NuRec GPU pod pulls its NVIDIA image using the operator-managed Kubernetes
Secret `ngc-nvcr-imagepullsecret` in the workload namespace (`default` for these
demos). Create it once on your chosen cluster after configuring your NGC key.
Workbench submission checks this Secret but does not create it. From the
installed checkout, this command sends the saved key directly to Kubernetes
through stdin, without putting it in command arguments or a temporary file:

```bash
set -o pipefail
npa/.venv/bin/python - <<'PY' | kubectl --kubeconfig '<kubeconfig-path>' --context '<rtx-context>' --namespace default create -f -
import base64
import json
from npa.clients.config import resolve_credentials

key = resolve_credentials().ngc_api_key
if not key:
    raise SystemExit("Configure your NGC key first.")
auth = base64.b64encode(f"$oauthtoken:{key}".encode()).decode()
config = json.dumps({"auths": {"nvcr.io": {"auth": auth}}})
print(json.dumps({
    "apiVersion": "v1", "kind": "Secret",
    "metadata": {"name": "ngc-nvcr-imagepullsecret", "namespace": "default"},
    "type": "kubernetes.io/dockerconfigjson",
    "stringData": {".dockerconfigjson": config},
}))
PY
```

`create` refuses to overwrite an existing Secret. If your operator already
provisioned it, proceed to the demo command; its image preflight verifies access.
For a different workload namespace, change both namespace values above to match
your SkyPilot configuration.

## Run a demo

```bash
npa workbench workflow demo run real-to-sim --project '<project>' --infra 'k8s/<rtx-context>'
npa workbench workflow demo run synthetic-data --project '<project>' --infra 'k8s/<rtx-context>'
npa workbench workflow demo run rl-improvement --project '<project>' --infra 'k8s/<rtx-context>'
npa workbench workflow demo run nurec --project '<project>' --infra 'k8s/<rtx-context>'
```

Select one command, or use separate terminals to run independent demos. The
standard scheduler handles available GPU capacity. No manual sample archive,
baseline checkpoint, case file, source URL, or bucket substitution is required.
The configured project supplies the storage destination. The native workflow
specification owns the stages and full workload sizes.

Demo launch and viewing both use that project's complete saved storage
credentials. Unrelated shell storage credentials and source pointers are
temporarily ignored during demo submission; the command restores the shell
environment afterward. Report downloads are kept separately for each project
and storage destination.

Standard workflow status, live logs, and cancellation use the resolved run's
project and storage credentials for controller queries too. After an owned local controller
stops, these commands recover it with the same storage identity used to read the
run. They leave your shell credentials unchanged; a controller already running
under a different identity still fails verification.

Append `--plan-only` to inspect the standard submission plan without launching.
Each execution receives a fresh run ID and its own output prefix. Record the run
ID printed by the command. The launcher does not add a per-stage time limit.

| Demo | Sample prepared automatically | Result to inspect |
| --- | --- | --- |
| `real-to-sim` | Complete hash-verified TUM office RGB-D capture, calibration, poses, measured collision scene and supported reset cases | Reconstruction measurements, trained checkpoint, and scored navigation episodes |
| `synthetic-data` | NVIDIA warehouse and calibrated four-camera route | 265 poses, 1,060 RGB-D views, colored point clouds, and independent geometry checks |
| `rl-improvement` | Public office/warehouse navigation inputs and baseline training | Baseline/candidate evaluation on separate final cases, safety metrics, and a measured recommendation |
| `nurec` | Pinned public PPISP NCore capture | Native Gaussian USDZ, novel views, image-quality metrics, and a Rerun recording |

## View, inspect, and resume

```bash
npa workbench workflow demo view synthetic-data '<run-id>' --project '<project>'
npa workbench workflow status '<run-id>' --project '<project>'
npa workbench workflow logs '<run-id>' --project '<project>'
```

`demo view` downloads only the compact, self-contained `reports/index.html` and
opens it in your browser. It does not download the complete sensor dataset or
put storage credentials in a browser URL. Use `--no-open` on a remote terminal
to print the local report path. Preserve the full S3 artifacts for training and
independent review; the HTML is a visual summary.

Resume an interrupted execution with its exact recorded identity:

```bash
npa workbench workflow demo run synthetic-data --project '<project>' \
  --infra 'k8s/<rtx-context>' --resume-run '<run-id>'
```

The existing workflow engine verifies immutable inputs, source, images, and
completed outputs before reusing them. A failed or ambiguous execution must be
reconciled through that engine; rerunning with a fresh ID is a new experiment.

## Read quality separately from execution

A completed GPU stage is not proof that a policy improved. Reports preserve
the measured success rates, safety metrics, and failed quality gates. Training
scene performance does not establish cross-scene transfer. Final evaluation
cases must remain separate from training and development selection.

The earlier scan-trained candidate scored 0/4,000 on the held-out warehouse;
that negative result remains historical evidence. The new replay experiment
must establish its own result before being described as effective.

The complete SDG and NuRec demo commands have now passed fresh full GPU
qualification, complete artifact readback, and offline report checks. See the
[source-bound measurements](../evidence/public-demos/README.md), including the
NuRec report's documented metric-display correction. The scan-to-policy and RL
improvement demos remain under qualification; their latest full attempts stopped
before learning and do not establish policy quality.

To stop a run, use `npa workbench workflow cancel '<run-id>' --project '<project>'`.
Cancel and verify terminal status before removing dedicated infrastructure.
