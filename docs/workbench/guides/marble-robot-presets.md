# Marble robot inspection presets

Choose a Go1 scenario, provide the project, cluster, bucket and credentials,
then submit its YAML. It generates a World Labs Marble world, records simulated
robot observations on Nebius, and publishes a standalone HTML replay. These
presets do not require an existing `world_uri`.

To use your own environment, [add a new world](#add-a-new-world). All preset
YAMLs and the five-site fan-out live in `workflows/testing/`; the
[testing catalog](../../../workflows/testing/README.md#generation-and-reconstruction)
pairs each workflow with this guide.

These are testing workflows. Their schema, resolved commands and fan-out are
checked locally. The earlier warehouse demo completed on one real RTX PRO 6000;
these five new generated worlds and the combined fan-out have not completed
live acceptance. Generated geometry must pass route and motion checks.

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
