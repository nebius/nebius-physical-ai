# Render digital-twin scenes on Nebius GPUs

[Workbench](../README.md) · [NuRec reconstruction](neural-reconstruction.md)

For a large authored environment, the
[RTX campus workflow](../../../workflows/testing/digital-twin-campus-render.yaml)
renders an 18-hectare site with native Blender OptiX. Its cutaway robotics and
fulfillment halls, freight terminal, operations tower, solar field and process
yard share one OpenUSD/glTF scene. The offline HTML provides four inspection
routes, thumbnails, scrubbing, keyboard navigation, playback and fullscreen views.
Scene counts come from native geometry; they do not represent live operations.

Use the RTX path to reconstruct a real capture into a Gaussian USDZ scene, render
new camera views on a GPU, and inspect the results in a portable HTML file.
The infrastructure uses a dedicated project, a CPU pool for coordination and
reports, one RTX PRO 6000 worker, and project-scoped object storage. SkyPilot
executes the existing NuRec workflow; stages exchange their artifacts through S3.

For an authored reference environment, the companion CUDA workflow renders an
industrial cell with native Blender Cycles and exports OpenUSD and binary glTF.
It can target B200 compute infrastructure without depending on RT cores. These
are separate renderer contracts: B200 availability does not make NuRec compatible.

| Input and purpose | Renderer | Infrastructure |
| --- | --- | --- |
| Authored 18-hectare industrial campus → aerial and close inspection routes | Blender 4.5.3 Cycles OptiX | RTX PRO 6000, on-demand or explicitly authorized preemptible |
| Real calibrated capture → appearance reconstruction and new views | NVIDIA NuRec/NRE | [RTX PRO 6000](../../../npa/examples/fleet/digital-twin.yaml) |
| Repository-authored factory cell → rendered views and portable geometry | Blender 4.5.3 Cycles CUDA | [B200](../../../npa/examples/fleet/digital-twin-cuda.yaml); RTX CUDA qualification is separate |

The HTML shows recorded render viewpoints with independent playback controls.
It does not stream a running renderer or provide a free-camera XR client.
Appearance reconstruction does not establish collision geometry, articulated
physics, or a synchronized live sensor twin.

## Prepare the RTX infrastructure

Run deployment and workflow submission from a Linux operator host with `/proc`.
Use an isolated checkout, virtualenv, and private `NPA_CONFIG_DIR`. Local macOS
planning and viewing remain supported. Install NPA following the
[getting-started guide](../getting-started.md) and prepare the operator's own
Nebius and NGC credentials.

Configure an explicitly selected project and create its object storage:

```bash
npa workbench health preflight --checks nebius,ngc --json
npa workbench health access --capability nurec --json
npa configure --no-interactive --provision \
  --tenant-id '<tenant-id>' --project-id '<project-id>' \
  --region us-central1 --project-alias twin
npa workbench health preflight --project twin --checks nebius,s3 --json
```

Copy [the infrastructure spec](../../../npa/examples/fleet/digital-twin.yaml)
outside the repository. Set the authorized tenant, exact project ID, region,
and a project name matching the configured alias in that private copy. The
template requests one RTX PRO 6000 and a separate CPU worker. Adjust the shape
to the workload and verified capacity; it imposes no run budget or job limit.

For reserved capacity, set `defaults.gpu_nodes.capacity_block_group` to the
private reservation ID. Fleet verifies the reservation and uses strict binding.
Omitting that field selects ordinary on-demand allocation. A full reservation
does not silently select another pool. Preemptible allocation requires an
explicit operator decision.

When preemptible use is authorized, remove the reservation field and set
`defaults.gpu_nodes.preemptible: true` in the private spec. Keep the CPU pool
on-demand and preserve GPU count, driver profile and disk allocations. A reclaim
can interrupt a render; inspect the same run and use its standard resume path.
Only completed, verified bundles receive a publication receipt.

For another region, select a project belonging to that region and use the
provider's advertised platform and preset. Some regions expose
`gpu-rtx6000-a`; preserve that exact platform so Fleet configures the matching
GPU Operator driver selector. Fleet preflight verifies project identity,
regional quotas and any reservation before provisioning.

```bash
npa fleet plan --spec '<private-fleet.yaml>'
npa fleet deploy --spec '<private-fleet.yaml>' --yes
```

On a fresh cluster, create SkyPilot's service account before the first image
preflight. Use the deployment's exact kubeconfig and context; an existing
SkyPilot service account should be inspected and reused:

```bash
kubectl --kubeconfig '<private-kubeconfig>' --context '<render-context>' \
  --namespace default create serviceaccount skypilot-service-account
```

This creates the identity only. SkyPilot manages its runtime role bindings during
bootstrap; image preflight verifies that exact identity's pull configuration.

The `rtx-rendering` profile selects GPU Operator mounted graphics drivers and
requires stable GPU capacity, CUDA execution, and GLX/EGL/Vulkan readiness.
The template supplies signed Ubuntu HTTPS repositories for the driver build
and uses S3 handoffs, so it does not require a shared filesystem.

The NRE image requires the operator's NGC credential. Before submitting, create
the workflow's `ngc-nvcr-imagepullsecret` in the selected cluster's `default`
namespace from an owner-private Docker config scoped to `nvcr.io`:

```bash
kubectl --kubeconfig '<private-kubeconfig>' --context '<rtx-context>' \
  --namespace default create secret generic ngc-nvcr-imagepullsecret \
  --type=kubernetes.io/dockerconfigjson \
  --from-file=.dockerconfigjson='<private-docker-config.json>'
```

For registry preflight, set `SKYPILOT_DOCKER_SERVER=nvcr.io`,
`SKYPILOT_DOCKER_USERNAME='$oauthtoken'`, and `SKYPILOT_DOCKER_PASSWORD` to the
same key in the private process environment. Keep the Docker config, keys and
rendered secret-bearing runtime documents outside Git and shared reports.

Keep B200 workloads on a separate compatible compute profile with managed
drivers. B200 may serve supported training or inference components; it cannot
replace RTX in this NuRec renderer. Availability alone does not establish
image or workload compatibility.

## Reconstruct and render

Use the registered project and exact context from deployment. Confirm the
requested accelerator name after the GPU Operator has labeled the worker:

```bash
npa workbench workflow gpus --context '<rtx-context>' --project twin --json
npa skypilot verify --cluster '<rtx-context>'
npa workbench workflow demo run nurec \
  --project twin --infra 'k8s/<rtx-context>'
npa workbench workflow demo view nurec '<run-id>' --project twin
```

The public default uses NVIDIA's real PPISP capture under CC BY 4.0. Its
attribution is retained in the HTML. Custom captures need the appropriate NCore
input, calibrated camera poses, and source permissions. See the
[NuRec guide](neural-reconstruction.md) for the
full workflow and camera controls.

## Rendering evidence and portable output

The native render stage writes `novel_views/render-evidence.json` after NRE
exits successfully and produces frames. It binds the consumed USDZ and ordered
rendered image bytes with SHA-256 hashes. It also samples GPU model, driver,
utilization and memory during execution. These are allocated-device observations,
not process-level or cryptographic hardware attestation. Missing or inactive
telemetry is explicitly labeled unverified.

Rendering reserves a fresh output generation and preserves earlier files. Its
evidence binds only the new generation, so stale frames cannot produce a success
record.
A failed renderer does not publish a success record. The HTML builder verifies
the scene and media hashes before displaying the record; changed bytes or frame
counts fail rather than inheriting evidence from another render. Older runs
without a record remain viewable with an unavailable-evidence label.

`reports/index.html` embeds up to 32 images per timeline, at up to 960 pixels.
Playback presents sampled output frames, not a newly synthesized video. The file
works offline and blocks network connections through its content security policy.
Only selected hardware measurements and content hashes enter the report; private
infrastructure identifiers, credentials, storage locations, and arbitrary receipt
metadata are excluded. Full-resolution images, the USDZ and Rerun recording
remain separate run artifacts.

## Render the industrial campus on RTX

Prepare the dedicated RTX infrastructure and project storage above, then select
the [campus workflow](../../../workflows/testing/digital-twin-campus-render.yaml):

```bash
npa workbench health preflight --project twin --checks nebius,s3 --json
npa workbench workflow validate-spec workflows/testing/digital-twin-campus-render.yaml
npa workbench workflow prepare-run 'workflows/testing/digital-twin-campus-render.yaml' --project twin
npa workbench workflow preflight-images workflows/testing/digital-twin-campus-render.yaml \
  --project twin --infra 'k8s/<rtx-context>'
npa workbench workflow submit workflows/testing/digital-twin-campus-render.yaml \
  --run-id '<prepared-run-id>' --project twin --infra 'k8s/<rtx-context>' \
  --var bucket='<your-bucket>' --var prefix='digital-twin/<prepared-run-id>' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Use the deployment's exact kubeconfig and the prepared run ID, as in the CUDA
example below. The default renders 32 views at 2560 × 1440 and 128 path-tracing
samples. `views` and `samples` are quality controls; the campus requires at least
four views so every inspection route has real rendered media. Native OptiX
selection disables both CPU and CUDA render devices and refuses fallback if
OptiX is unavailable. GPU activity is required before the campus HTML is emitted.

The scene contains 32 industrial robots, 600 modeled warehouse inventory units,
108 solar arrays, autonomous carriers, stacked freight containers, two gantry
cranes and an operations tower. Shared meshes keep repeated geometry compact.
The native receipt records measured object and triangle counts, asset-role
counts and every camera matrix. SHA-256 evidence binds both scene source modules,
the pinned Blender archive, native receipt, scene exports and rendered frames.

The bundle resolves through `rendered/completion.json`. Its `index.html` embeds
1920-pixel previews derived from the original 1440p PNG frames; it makes no
network requests. It presents recorded viewpoints of an authored reference
environment, not live simulation, synchronized telemetry or an XR session.

After independently materializing the completed publication:

```bash
NPA_INTEGRATION_E2E=1 NPA_DIGITAL_TWIN_CAMPUS_LIVE_DIR='<materialized-campus-directory>' \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_digital_twin_render_live.py -q
```

This verifies native RTX OptiX selection, observed GPU activity, source hashes,
full-resolution distinct frames, all four camera routes and a populated exported
scene. The adjacent readiness record distinguishes planning from live execution.

### Verified RTX campus execution

The campus workflow completed on one preemptible NVIDIA RTX PRO 6000 with
Blender 4.5.3 Cycles OptiX. It produced all 32 distinct 2560 × 1440 views at 128
samples, with CPU rendering disabled. The native scene contains 8,404 mesh
objects, 22 shared meshes and 208,984 instanced triangles, including 139 freight
containers. Native execution, including scene setup and exports, took 247.162
seconds. Device-wide observations covered 238 samples and reached 98% utilization
and 4231 MiB GPU memory. This is one execution, not a throughput benchmark.

Immutable publication and independent readback passed. Live acceptance verified
both executed scene-source hashes, native OptiX selection, decoded original
frames and populated scene exports. Desktop and mobile browser checks passed
for all four routes, thumbnails, scrubbing, play/pause and fullscreen, with no
horizontal mobile overflow, network requests or JavaScript errors. An independent
OpenUSD 26.8 reader loaded 8,409 mesh primitives, including converted lettering,
and verified the foundation's 500 × 360 metre extent at one metre per scene unit.
Exact resource
identities and operational receipts remain in access-controlled evidence.

## Render the authored CUDA reference scene

Use [the B200 fleet template](../../../npa/examples/fleet/digital-twin-cuda.yaml)
in a dedicated project with project-scoped object storage. Set its tenant,
project and optional reservation privately, and prepare the fresh cluster's
SkyPilot service account as described above. Its managed-image
driver profile is separate from the RTX graphics profile; do not change a
partially provisioned RTX pool into B200 or select GPU Operator on an NVSwitch
system. Reconcile the owned failed allocation before choosing another profile.

The [CUDA workflow](../../../workflows/testing/digital-twin-cuda-render.yaml)
uses an existing immutable workbench image and the submitting checkout's source
overlay. It downloads the official Blender 4.5.3 Linux archive at runtime and
checks its pinned SHA-256 before extraction. Blender is GPL-licensed; see
[Blender's license information](https://www.blender.org/about/license/).
No Blender distribution or third-party scene assets are added to an image or
this repository. The workflow uses no Isaac, NGC, model weights or gated assets.

```bash
npa workbench health preflight --project twin --checks nebius,s3 --json
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
npa workbench workflow validate-spec workflows/testing/digital-twin-cuda-render.yaml
npa workbench workflow plan-spec workflows/testing/digital-twin-cuda-render.yaml --run-id twin-render
npa workbench workflow prepare-run 'workflows/testing/digital-twin-cuda-render.yaml' --project twin
npa workbench workflow preflight-images workflows/testing/digital-twin-cuda-render.yaml \
  --project twin --infra 'k8s/<b200-context>'
npa workbench workflow submit workflows/testing/digital-twin-cuda-render.yaml \
  --run-id '<prepared-run-id>' --project twin --infra 'k8s/<b200-context>' \
  --var bucket='<your-bucket>' --var prefix='digital-twin/<prepared-run-id>' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Use the fresh ID returned by `prepare-run` in both run placeholders. Select the
deployment's verified private kubeconfig through `KUBECONFIG`; do not rely on an
ambient context. The `bucket` override is the bucket name, without `s3://`.
Source staging is automatic. If submission fails, inspect the
same run's status and retained diagnostics before using `--resume`; creating a
second run does not reconcile the first.

The default is 24 camera viewpoints at 1280 × 720 with 64 path-tracing samples.
`--var views=<count>` and `--var samples=<count>` control rendering quality and
view coverage. `--var accelerator=RTXPRO6000:1` selects the CUDA backend on a
separately qualified RTX cluster; it does not select NRE or OptiX.
Native Cycles device discovery must find CUDA GPUs, and every CPU rendering
device is explicitly disabled. A failed renderer publishes no success artifact.

The completed bundle contains `index.html`, original PNG frames, `scene.usdc`,
`scene.glb`, camera matrices in `native-render.json`, and `render-evidence.json`.
Publication uses an immutable attempt and verified object-storage readback.
The declared stage output is `rendered/completion.json`, which binds all files
under its immutable attempt; consumers resolve this receipt before loading media.
The HTML embeds the real rendered frames, provides playback and scrubbing,
and displays the native backend, selected GPU family and device-wide observations.
It includes no remote scripts, network requests, infrastructure identifiers or
storage routes. Hashes bind the native receipt, exported scene and all frames;
they prove byte consistency, not cryptographic hardware attestation.

This scene is an authored factory-cell demonstrator. It has no connection to a
captured facility, synchronized sensors, collision validation or a live XR client.
OpenUSD/glTF exports are geometry handoffs; verify their materials and scale in
the receiving application before a simulation or XR integration.

After materializing the completed bundle into a private local directory:

```bash
NPA_INTEGRATION_E2E=1 NPA_DIGITAL_TWIN_CUDA_LIVE_DIR='<materialized-render-directory>' \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_digital_twin_render_live.py -q
```

The read-only acceptance check requires native CUDA selection, positive observed
GPU activity, distinct non-flat full-resolution images, a populated glTF scene,
bound OpenUSD bytes and an offline HTML viewer. The adjacent workflow readiness
record reports which execution prerequisites have actually been verified.

### Verified B200 execution

The reference workflow completed on one NVIDIA B200 with Blender 4.5.3 Cycles
CUDA, producing 24 distinct 1280 × 720 PNG views at 64 samples, OpenUSD/glTF
exports, and portable HTML. Native execution took 318.613 seconds, including
initial CUDA kernel setup and exports; this is not a steady-state render benchmark.
Device-wide telemetry collected 311 samples, with 99% peak utilization and
4308 MiB peak allocated GPU memory. The selected native renderer had CPU devices
disabled. The executed scene script's SHA-256 was
`b2ea5a30c7751d480b61e8c957c419ec983bc63a49199dede19ded61592e2f80`.

Immutable publication and independent object-storage readback succeeded. The
live acceptance test passed against the materialized bundle, including the
checkout's scene-script hash. Browser inspection verified image decoding,
24-view scrubbing, and play/pause with zero network requests or JavaScript errors.
Exact operational receipts remain in access-controlled evidence. This verifies
the authored CUDA reference on B200; RTX/NuRec execution is separately qualified.

## Verify a captured NuRec run

After downloading the complete run into an owner-private directory, verify it:

```bash
NPA_INTEGRATION_E2E=1 NPA_DIGITAL_TWIN_LIVE_DIR='<materialized-run-directory>' \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_digital_twin_render_live.py -q
```

This read-only live check validates bound scene/media hashes, observed RTX
activity, a readable USDZ archive, non-empty decoded renders, and offline HTML.
It does not launch another GPU run.

## Cleanup

Cancel an unfinished workflow by its exact run ID before removing infrastructure.
After terminal workflow completion, remove its owned SkyPilot controller using
the exact project/context, then destroy the dedicated fleet:

```bash
npa skypilot cleanup-controller --project twin --context '<rtx-context>' --yes
npa fleet destroy --spec '<private-fleet.yaml>' --yes
```

Fleet destroy retains object storage and the project. Keep the rendered assets
until the intended handoff is complete, then follow the explicit storage/project
cleanup steps in the [teardown guide](../../teardown.md).
