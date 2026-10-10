# Marble robot inspection presets

Choose a Go1 scenario, provide the project, cluster, bucket and credentials,
then submit its YAML. It generates a World Labs Marble world, records simulated
robot observations on Nebius, and publishes a standalone HTML replay. These
presets do not require an existing `world_uri`.

To use your own environment, [add a new world](#add-a-new-world). All preset
YAMLs and the five-site fan-out live in `workflows/testing/`; the
[testing catalog](../../../workflows/testing/README.md#generation-and-reconstruction)
pairs each workflow with this guide.

These remain testing workflows. All five generated-site parameterizations
completed native live collection and standalone reporting on 2026-10-10 using
four RTX PRO 6000 workers followed by the fifth collection on a reused worker.
See [live acceptance](#live-acceptance) for measurements and the tested scope.
Newly generated geometry must still pass route and motion checks.

| Preset | YAML in `workflows/testing/` | Commanded speed | Observations | Intended experiment |
| --- | --- | --- | --- | --- |
| Warehouse | [marble-go1-warehouse.yaml](../../../workflows/testing/marble-go1-warehouse.yaml) | 1.2 m/s | 500 | Fast aisle patrol, turns and return passes for visual navigation data |
| Shipyard | [marble-go1-shipyard.yaml](../../../workflows/testing/marble-go1-shipyard.yaml) | 0.7 m/s | 750 | Inspection approaches beside hull sections and dock equipment |
| Home | [marble-go1-home.yaml](../../../workflows/testing/marble-go1-home.yaml) | 0.35 m/s | 1,000 | Slow hallway survey for localization and household robot perception |
| Factory | [marble-go1-factory.yaml](../../../workflows/testing/marble-go1-factory.yaml) | 0.5 m/s | 1,000 | Inspection views beside closed machine enclosures |
| Utility | [marble-go1-utility.yaml](../../../workflows/testing/marble-go1-utility.yaml) | 0.5 m/s | 1,000 | Service-aisle observations around pumps, pipes and valves |

Every preset uses the supported **Unitree Go1** articulation and pinned pretrained
walking policy. ANYmal and Spot require their own assets, controllers and
validated collector. The separate [ANYmal navigation workflow](marble-navigation-rl.md)
requires a qualified Isaac runtime and still has an outstanding GPU training
and evaluation acceptance gap. It does not produce this Go1 replay.

## Run one preset

Use the checkout containing these workflows from a Linux operator with NPA and
SkyPilot installed. Configure your Nebius project, exact Kubernetes context,
writable S3 bucket and a schedulable RTX PRO 6000 first. The source overlay
stages this checkout's collector code; an older installed release is insufficient.

Supply `WLT_API_KEY` through the environment or private NPA credentials. It needs
World API credit; the Marble website subscription alone is insufficient. Keep
the key out of YAML, command-line values and HTML. Supply S3 credentials through
the secret environment variables below.

```bash
npa workbench health preflight --project <project-alias> --checks nebius,s3 --json
npa skypilot verify --cluster <exact-context> --output-format json
npa workbench workflow validate-spec workflows/testing/marble-go1-home.yaml
npa workbench workflow plan-spec workflows/testing/marble-go1-home.yaml \
  --run-id <run-id> --var bucket=<owned-bucket> --waves --json
npa workbench workflow submit workflows/testing/marble-go1-home.yaml \
  --project <project-alias> --infra k8s/<exact-context> \
  --run-id <run-id> --var bucket=<owned-bucket> \
  --stage-src --runtime --max-wait-seconds 0 \
  --secret-env WLT_API_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --isolated-config-dir /var/lib/npa/<run-id>-sky
```

Select another preset by changing the YAML filename. Native submission resolves
the World API token before launch; the generation stage verifies that key and
credit work. The Nebius/S3 preflight does not establish World API credit. Preserve
the isolated SkyPilot directory for resume and cleanup; use a different run ID
and directory for a new run.

Successful single-site output is under
`s3://<owned-bucket>/runs/<run-id>/marble-go1-<site>/`:

- `world/world.json`: provider provenance and downloaded asset hashes.
- `results/`: RGB, raw depth, observer frames, robot states, actions, contacts,
  trajectory and measured GPU/render evidence.
- `report/index.html`: one offline HTML file embedding three synchronized videos
  and telemetry. Download it to view locally; playback requires no credentials
  and does not submit another job.

To reuse the exact world without another generation request, use the
[existing-world workflow](marble-warehouse-quadruped.md) with its S3 prefix as
`world_uri`, without a trailing slash.

## Add a new world

For a new environment with the same Go1 robot, override `world_prompt` on any
single-site preset. No Python changes or container rebuild are needed. The
existing acquisition, collection and HTML stages consume the new world.
Use the setup above and a fresh run ID; the optional `prefix` override gives
the new site's artifacts an easy-to-recognize location.

```bash
npa workbench workflow submit workflows/testing/marble-go1-factory.yaml \
  --project '<project-alias>' --infra 'k8s/<exact-context>' \
  --run-id '<new-run-id>' --var 'bucket=<owned-bucket>' \
  --var 'prefix=runs/{{run.id}}/marble-go1-custom' \
  --var 'world_prompt=A realistic aircraft maintenance hangar with stationary aircraft beside a clear, flat concrete service aisle at least ten meters long and three meters wide. Camera centered near floor level, turning space at both ends, equipment outside the aisle, no people, robots, stairs or floor holes.' \
  --stage-src --runtime --max-wait-seconds 0 \
  --secret-env WLT_API_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --isolated-config-dir '/var/lib/npa/<new-run-id>-sky'
```

Before launching, preview the same YAML with `npa workbench workflow plan-spec`,
the same run ID and `--var` values, and `--waves --json`. The acquisition command
should contain your prompt. The example keeps the factory preset's Go1, 0.5 m/s
speed, 1,000 observations and turnaround profile. Its completed HTML is
`s3://<owned-bucket>/runs/<new-run-id>/marble-go1-custom/report/index.html`.

Describe the site, lighting and equipment, then keep the flat-floor, clear-route
and turning-space requirements. Generated geometry must still pass the
collector's checks. A new environment does not add a new robot, stair-climbing
controller or inspection detector.

For another generation, use a new run ID and output prefix. To resume the same
generation, retain its original prompt and run settings; the acquisition journal
rejects a different prompt in an existing generation prefix.

### Reuse a world you already generated

Run `marble-warehouse-quadruped.yaml` using the
[existing-world instructions](marble-warehouse-quadruped.md#run-headlessly),
with `--var world_uri=s3://<owned-bucket>/<world-prefix>` and a fresh run ID.
The name of that workflow does not restrict the input to a warehouse; the
world still must meet its route and motion requirements. Tune speed and frames
using the same guide if you want a slower inspection pass.

`world_uri` points to the NPA acquisition bundle containing `world.json`,
`world.spz` and `collider.glb`, without a trailing slash. A Marble browser link
or HTML replay is not that bundle. Reuse needs S3 access and does not make a
new World Labs request or require `WLT_API_KEY`.

### Save a reusable preset

Copy the closest single-site YAML to `workflows/testing/marble-go1-<site>.yaml`.
Give it a distinct `metadata.name`, update `metadata.description`, and set
`config.world_prompt` and `config.prefix` for the new site. Keep the existing
stage graph, toolRefs and URI templates; adjust `speed_mps` and `frames` if needed.
Validate and plan the copied file before submitting it.

When contributing that preset, add its workflow-to-guide row to the
[testing table](../../../workflows/testing/README.md), register its live coverage,
and create a matching readiness record as described in the
[workflow authoring instructions](../../../skills/workflows/author-npa-workflow/SKILL.md).
Record successful live validation only after that new scene actually runs.

### Replace a world in the five-site fan-out

In [marble-go1-five-sites.yaml](../../../workflows/testing/marble-go1-five-sites.yaml),
edit `params.world_prompt` under the chosen `acquire-<site>` state. For example,
change `acquire-factory` to generate the hangar. Keep its state name and the
matching collection/report paths to replace that slot while retaining five
scenarios. Update its description to identify the new environment. If needed,
change `speed_mps` and `frames` in the matching `collect-<site>` state too.

The fan-out uses per-state prompts, so a global `--var world_prompt=...` does
not replace them. Keep `preset_count: 5` and the three five-member lists when
replacing a site. Validate and inspect `plan-spec --waves` after editing, and
refresh the readiness record when saving a changed YAML.

## Fan out all five

Submit `workflows/testing/marble-go1-five-sites.yaml` with the same flags above.
It runs three native Workbench parallel waves:

```text
5 World Labs API acquisitions (CPU)
                 ↓ all acquisitions succeed
5 Go1 collectors (1 Nebius RTX PRO 6000 each)
                 ↓ all collections succeed
5 HTML report builders (CPU)
```

This requests **five concurrent GPUs**, one per site, during collection. It
makes five distinct hosted world-generation requests. Inspect the wave plan:

```bash
npa workbench workflow plan-spec workflows/testing/marble-go1-five-sites.yaml \
  --run-id <run-id> --var bucket=<owned-bucket> --waves --json
```

If your cluster has fewer than five available GPUs, add `--max-concurrency 4`
to the submission above for four available GPUs, or set it to your available
count. NPA batches every wave: all five sites still run with their full frame
counts and quality settings. With four GPUs, collection runs as four jobs and
then the fifth. Select this at initial submission; SkyPilot starts a parallel
job group together, so a five-member group needs capacity for all five workers.

Each site's artifacts use
`s3://<owned-bucket>/runs/<run-id>/marble-go1-five-sites/<site>/`, where `<site>` is
`warehouse`, `shipyard`, `home`, `factory` or `utility`. Each has its own
`report/index.html`; world and result paths are separate. `preset_count` must
remain five unless the member lists change too. Use individual YAMLs to run a
subset independently. A failed member stops the next wave; completed artifacts
remain available for diagnosis and native resume.

## What this measures

The GPU performs gsplat RGB rendering, Warp mesh-depth raycasts and Cycles robot
rendering. PyBullet torque dynamics, pretrained policy inference and denoising
run on CPU. Every turnaround collection must travel at least one simulated
meter, retain foot contact in at least 80% of observations, turn at least 150
degrees, and record a two-meter return leg.

Prompts request clear level routes with room to turn. Generated geometry is
approximate: a prompt does not establish a supported floor, collision clearance
or calibrated metric scale. The collector searches near the scene origin for
a supported straight route and turns along it. It does not explore arbitrary
rooms, climb stairs or plan a shipyard mission. Unsupported geometry or failed
motion produces a failed stage; no sample world is substituted.

The outputs support perception, localization and inspection-view experiments.
Factory and utility presets collect views; they do not detect defects, read
gauges or simulate thermal measurements. Home does not manipulate objects.
None trains a policy or proves production inspection performance. Navigation
learning requires the separate training/evaluation path and held-out evidence.

## Live acceptance

The [sanitized acceptance record](../validation/marble-go1-five-sites-20261010.json)
captures five distinct worlds generated with a funded `WLT_API_KEY`, all fifteen
acquire/collect/report stages, and 4,250 observations at 1280 × 720 and 25 Hz.
Native submission used `--max-concurrency 4`: four collectors ran concurrently,
then utility ran on one worker. All batch barriers passed the native workflow
assertion. The default five-concurrent-collector plan was not completed.

| Site | Observations | Recorded travel (m) | Return leg (m) | Peak speed (m/s) |
| --- | --- | --- | --- | --- |
| Warehouse | 500 | 13.04 | 5.76 | 1.31 |
| Shipyard | 750 | 15.52 | 5.66 | 0.81 |
| Home | 1,000 | 10.96 | 4.04 | 0.40 |
| Factory | 1,000 | 16.95 | 5.64 | 0.61 |
| Utility | 1,000 | 16.81 | 5.68 | 0.61 |

Each site passed the turn and contact checks and produced three embedded videos:
observer, robot RGB and mesh depth. The five native reports were repackaged
through NPA's HTML writer after fixing chapter-boundary rounding, preserving
every recorded video and telemetry byte. Both original and final report hashes
are retained. The final files were checked offline in Chromium desktop and
mobile viewports for playback, synchronized seeking,
motion chapters and telemetry downloads, with zero external requests or
JavaScript errors. A sixth overview uses NPA's
`npa.workflows.preview_html.write_preview` with hash-verified sampled frames;
the individual reports contain the full recordings.

The single-site YAMLs were validated and their matching parameterizations were
executed in the combined workflow; five independent entrypoint submissions were
not repeated. Collection used source `eb81e7039`. Subsequent XML hardening was
qualified separately using the pinned real Go1 URDF and a recorded pose on CUDA:
all 38 visual parts retained identical vertices, oriented topology, transforms
and materials. The largest image difference was one 8-bit color level, also
observed when repeating the original renderer. This is not a full fan-out rerun
of the hardened source.

GPU utilization statistics describe the recorded sampling windows, which may
include setup, CPU physics, rendering, artifact upload and group-barrier waiting.
Utility sampling began during rendering and includes a short renderer
qualification. Peak utilization does not establish sustained throughput.
Exact infrastructure and credential evidence is retained privately.
Visual inspection found reconstruction artifacts and a noticeably soft utility
background; the reports preserve that generated-world quality.

Main's newer credential-binding scheme intentionally refuses the older running
API session's identity seal. A separate fresh API using the merged source passed
Kubernetes access, identity revalidation, queue and controller-status checks,
then was stopped. That check launched no cloud jobs and does not replace the
collection's source-scope disclosure above.
