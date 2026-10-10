# Marble robot inspection presets

Use **one workflow**, [`marble-go1-inspection.yaml`](../../../workflows/testing/marble-go1-inspection.yaml),
and choose **one situation** with `--preset`. Each submission generates one
World Labs Marble world, records one simulated Go1 patrol on one Nebius RTX PRO
6000, and publishes one standalone HTML replay. It never launches the other four
situations. Omitting `--preset` uses the warehouse defaults.

The workflow lives in `workflows/testing/`. The
[testing catalog](../../../workflows/testing/README.md#generation-and-reconstruction)
links its single entry to this guide.

| `--preset` | Commanded speed | Recorded duration at 25 FPS | Intended experiment |
| --- | --- | --- | --- |
| `warehouse` | 1.2 m/s | 20 s / 500 frames | Fast aisle patrol, turns and return passes for visual navigation data |
| `shipyard` | 0.7 m/s | 30 s / 750 frames | Inspection approaches beside hull sections and dock equipment |
| `home` | 0.35 m/s | 40 s / 1,000 frames | Slow hallway survey for localization and household robot perception |
| `factory` | 0.5 m/s | 40 s / 1,000 frames | Inspection views beside closed machine enclosures |
| `utility` | 0.5 m/s | 40 s / 1,000 frames | Service-aisle observations around pumps, pipes and valves |

All five use the supported **Unitree Go1** articulation and pinned pretrained
walking policy. ANYmal and Spot need their own assets, controllers and qualified
collector. The separate [ANYmal navigation workflow](marble-navigation-rl.md)
still has an outstanding GPU training and evaluation acceptance gap; it does
not produce this Go1 replay.

## Run one preset

Use this checkout from a Linux operator with NPA and SkyPilot installed.
Configure your Nebius project, exact Kubernetes context, writable S3 bucket and
one schedulable RTX PRO 6000. The source overlay stages the checkout's collector
and report code; an older installed release is insufficient.

Supply `WLT_API_KEY` through the environment or private NPA credentials. It needs
World API credit; a Marble website subscription alone is insufficient. Keep the
key out of YAML, command-line values and HTML. Provide S3 credentials through
the secret environment variables below.

```bash
npa workbench health preflight --project <project-alias> --checks nebius,s3 --json
npa skypilot verify --cluster <exact-context> --output-format json
npa workbench workflow validate-spec workflows/testing/marble-go1-inspection.yaml \
  --preset home
npa workbench workflow plan-spec workflows/testing/marble-go1-inspection.yaml \
  --preset home --run-id <run-id> --var bucket=<owned-bucket> --waves --json
npa workbench workflow submit workflows/testing/marble-go1-inspection.yaml \
  --preset home --project <project-alias> --infra k8s/<exact-context> \
  --run-id <run-id> --var bucket=<owned-bucket> \
  --stage-src --runtime --max-wait-seconds 0 \
  --secret-env WLT_API_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --isolated-config-dir /var/lib/npa/<run-id>-sky
```

Change only `--preset home` to select another built-in situation. Validation,
planning and submission resolve the same preset. Unknown names and conflicting
`--var` overrides of preset-owned settings fail before submission. There is no
`all` preset. The plan contains three sequential stages: `acquire`, `collect`,
`report`; only `collect` requests a GPU.

Native submission resolves the World API token before launch. The acquisition
stage establishes that the key and API credit work; the Nebius/S3 preflight does
not prove World API credit. Preserve the isolated SkyPilot directory for resume
and cleanup. Use a new run ID and directory for a new run.

Output is under `s3://<owned-bucket>/runs/<run-id>/marble-go1-<preset>/`:

- `world/world.json`: provider provenance and downloaded asset hashes.
- `results/`: RGB, raw depth, observer frames, states, actions, contacts,
  trajectory and measured GPU/render evidence.
- `report/index.html`: one offline HTML file with three synchronized **continuous
  MP4 videos at 25 FPS**, every recorded frame, telemetry, seeking and motion
  chapters. Download and open it; playback needs no credentials or GPU job.

## Add a new world

Use the **same YAML**, omit `--preset`, and override the prompt and output label.
No new workflow file, Python change or container rebuild is required. This
example explicitly selects the factory collection settings for a custom hangar:

```bash
npa workbench workflow submit workflows/testing/marble-go1-inspection.yaml \
  --project '<project-alias>' --infra 'k8s/<exact-context>' \
  --run-id '<new-run-id>' --var 'bucket=<owned-bucket>' \
  --var workflow_preset=hangar --var frames=1000 --var speed_mps=0.5 \
  --var 'world_prompt=A realistic aircraft maintenance hangar with stationary aircraft beside a clear, flat concrete service aisle at least ten meters long and three meters wide. Camera centered near floor level, turning space at both ends, equipment outside the aisle, no people, robots, stairs or floor holes.' \
  --stage-src --runtime --max-wait-seconds 0 \
  --secret-env WLT_API_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --isolated-config-dir '/var/lib/npa/<new-run-id>-sky'
```

Preview the same configuration using `plan-spec`, the same run ID and `--var`
values, and `--waves --json`. The acquisition command should contain your prompt.
The report will be under `runs/<new-run-id>/marble-go1-hangar/report/index.html`.
The built-in presets own their prompt, name, generation source, frame count and
speed, so omit `--preset` when changing those values.

Describe the environment and lighting, retaining the flat-floor, clear-route
and turning-space requirements. Generated geometry must pass the collector's
checks. A new scene does not add stair climbing, a new robot or a detector.
Use a new run ID for another generation. Resuming a generation requires its
original prompt and settings; the journal rejects mismatched requests.

To reuse a generated world's exact assets without another World API request,
use the [existing-world instructions](marble-warehouse-quadruped.md#run-headlessly)
with `world_uri` set to its S3 prefix, without a trailing slash.

## What the GPU does

World Labs generates the environment behind its API. The acquisition stage
downloads the scene assets to S3. The Nebius collector loads those assets and
performs gsplat RGB rendering, Warp mesh-depth raycasts and Cycles articulated
robot rendering on one RTX PRO 6000. PyBullet torque dynamics, pretrained policy
inference and denoising run on CPU.

Every turnaround collection must travel at least one simulated meter, retain
foot contact in at least 80% of observations, turn at least 150 degrees and
record a two-meter return leg. The collector searches near the scene origin for
a supported straight route and turns along it. It does not explore arbitrary
rooms or plan a shipyard mission. Unsupported geometry or failed motion fails
the stage; no sample world is substituted.

These outputs support perception, localization and inspection-view experiments.
They do not detect defects, read gauges, simulate thermal measurements, train a
new policy or prove production inspection performance. Generated geometry and
scale require calibration before real-site or sim-to-real claims.

## Continuous proof for all five situations

A single run produces its selected situation's continuous report. For comparing
already completed runs, NPA's native `write_quadruped_gallery` packages their full
recordings into one standalone HTML with a situation selector:

```python
from pathlib import Path
from npa.workbench.marble.quadruped_report import write_quadruped_gallery

output = Path("continuous-proof")
output.mkdir(exist_ok=True)
write_quadruped_gallery(output, {
    "Warehouse": Path("warehouse/index.html"),
    "Shipyard": Path("shipyard/index.html"),
    "Home": Path("home/index.html"),
    "Factory": Path("factory/index.html"),
    "Utility": Path("utility/index.html"),
})
```

The gallery replays existing results; it submits no jobs. It embeds every
original MP4 and trajectory row, with no frame sampling or external media
requests. Only the selected situation's three videos are loaded for playback.
The full five-situation proof is about 247 MB because it preserves all recordings.
The [continuous-proof verification record](../validation/marble-go1-presets-continuous-20261010.json)
binds the HTML hash to all five command-equivalence checks and eleven offline
browser checks, including presented video frames, seeking, downloads and rapid
situation switching. Desktop and mobile layouts made no external requests.

## Live acceptance

The [original acceptance record](../validation/marble-go1-five-sites-20261010.json)
records five distinct API-generated worlds and 4,250 observations at 1280 × 720,
25 Hz. These recordings were collected on Nebius using the earlier combined
workflow before the interface was consolidated into one preset-selectable YAML.
They establish real GPU work for the five parameterizations; they do not establish
five fresh submissions of the new entrypoint.

| Site | Recorded travel (m) | Return leg (m) | Peak speed (m/s) |
| --- | --- | --- | --- |
| Warehouse | 13.04 | 5.76 | 1.31 |
| Shipyard | 15.52 | 5.66 | 0.81 |
| Home | 10.96 | 4.04 | 0.40 |
| Factory | 16.95 | 5.64 | 0.61 |
| Utility | 16.81 | 5.68 | 0.61 |

Every site passed turn and contact checks. The original collection used source
`eb81e7039`; subsequent XML hardening was qualified with pinned real Go1 assets
and a recorded CUDA pose, preserving all 38 visual parts. This was not a full
collection rerun. GPU utilization covers measured sampling windows, including
setup and CPU work; it does not prove sustained throughput. The utility world
has a visibly soft reconstruction, preserved in the proof. Exact infrastructure
and credential evidence remains private.
