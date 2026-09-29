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
installed checkout with its virtual environment activated, this command sends
the saved key directly to Kubernetes through stdin, without putting it in
command arguments or a temporary file:

```bash
set -o pipefail
python - <<'PY' | kubectl --kubeconfig '<kubeconfig-path>' --context '<rtx-context>' --namespace default create -f -
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
| `rl-improvement` | Public office/warehouse inputs, baseline training, and measured baseline simulation failures | Failure-admitted reconstruction, continued training, development selection or rejection report; final comparison only after selection passes |
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

Keep the original installed checkout until each run finishes. Resume with that
checkout, the same project and runtime target, and the exact recorded run ID.
A newer driver may reject the previous run's saved source or image identity.

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

The earlier scan-trained candidate scored 0/4,000 on the held-out warehouse.
A subsequent full replay experiment improved development success from 82.475%
to 93.0%, but failed 371 metric checks across 257 cases. Selection retained the
baseline and left final outcomes untouched. Both negative results remain
historical evidence. The current fixed-baseline-penalty experiment also rejected
its candidate: development success declined from 82.475% to 82.05%, with 699
metric violations across 399 cases. The original baseline was retained and
final evaluation was not run. The workflow's full training, comparison and
rejection behavior are verified; effective promotion remains unproven.

Scan-to-policy, SDG and NuRec have full GPU qualification, complete artifact
readback and offline report checks in the
[source-bound measurements](../evidence/public-demos/README.md). The fresh scan
run qualifies the updated shared navigation runtime: 3,386/4,000 successful
held-out routes (84.65%) after 500 training updates and 16 million transitions.
All cases completed by native step 20 within the unchanged 300-step maximum.
This evaluates goals within the reconstructed public office; it does not establish
transfer to unseen buildings, physical robots or camera-input policies.

Each receipt identifies its exact native source revision and report bytes.
The earlier scan result remains preserved separately. SDG and NuRec retain their
unchanged workload source. The RL comparison workflow is verified through its
rejection path. Successful policy promotion still requires the unchanged
development and final quality gates.

To stop a run, use `npa workbench workflow cancel '<run-id>' --project '<project>'`.
Cancel and verify terminal status before removing dedicated infrastructure.
