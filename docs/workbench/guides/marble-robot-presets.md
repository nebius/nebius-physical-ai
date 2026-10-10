# Marble robot inspection presets

Choose a Go1 scenario, provide the project, cluster, bucket and credentials,
then submit its YAML. It generates a World Labs Marble world, records simulated
robot observations on Nebius, and publishes a standalone HTML replay. These
presets do not require an existing `world_uri`.

These are testing workflows. Their schema, resolved commands and fan-out are
checked locally. The earlier warehouse demo completed on one real RTX PRO 6000;
these five new generated worlds and the combined fan-out have not completed
live acceptance. Generated geometry must pass route and motion checks.

| Preset | YAML in `workflows/testing/` | Commanded speed | Observations | Intended experiment |
| --- | --- | --- | --- | --- |
| Warehouse | `marble-go1-warehouse.yaml` | 1.2 m/s | 500 | Fast aisle patrol, turns and return passes for visual navigation data |
| Shipyard | `marble-go1-shipyard.yaml` | 0.7 m/s | 750 | Inspection approaches beside hull sections and dock equipment |
| Home | `marble-go1-home.yaml` | 0.35 m/s | 1,000 | Slow hallway survey for localization and household robot perception |
| Factory | `marble-go1-factory.yaml` | 0.5 m/s | 1,000 | Inspection views beside closed machine enclosures |
| Utility | `marble-go1-utility.yaml` | 0.5 m/s | 1,000 | Service-aisle observations around pumps, pipes and valves |

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
