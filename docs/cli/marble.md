# `npa workbench marble`

## Command Tree

```text
Usage: npa workbench marble [OPTIONS] COMMAND [ARGS]...

World Labs Marble worlds, CUDA camera datasets, and spatial scans.

Options
--help  Show this message and exit.
Commands
navigation-prepare  Prepare a Marble collider for native ANYmal navigation learning.
rover-collect  Collect RGB, depth, actions, and contact states from a wheel-driven rover.
acquire  Acquire a hosted world or the explicitly attributed upstream example.
capture  Render real Gaussian splats with gsplat on a required CUDA GPU.
scan  Measure depth and clearance against real mesh triangles on CUDA.
report  Build the interactive HTML report from verified GPU results.
pallet-preflight  Require the World API key and snapshot disjoint labeled pallet data.
pallet-benchmark  Train matched-budget pallet detectors and compare held-out real-image AP.
pallet-report  Publish the measured pallet comparison as static JSON and HTML.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `navigation-prepare` | Prepare a Marble collider for native ANYmal navigation learning. |
| `rover-collect` | Collect RGB, depth, actions, and contact states from a wheel-driven rover. |
| `acquire` | Acquire a hosted world or the explicitly attributed upstream example. |
| `capture` | Render real Gaussian splats with gsplat on a required CUDA GPU. |
| `scan` | Measure depth and clearance against real mesh triangles on CUDA. |
| `report` | Build the interactive HTML report from verified GPU results. |
| `pallet-preflight` | Require the World API key and snapshot disjoint labeled pallet data. |
| `pallet-benchmark` | Train matched-budget pallet detectors and compare held-out real-image AP. |
| `pallet-report` | Publish the measured pallet comparison as static JSON and HTML. |

## Examples

```bash
npa workbench marble --help
npa workbench marble navigation-prepare --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `marble`.
