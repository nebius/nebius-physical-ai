# `npa workbench retargeting`

## Command Tree

```text
Usage: npa workbench retargeting [OPTIONS] COMMAND [ARGS]...

Motion retargeting for SONIC locomotion workflows.

Options
--help  Show this message and exit.
Commands
run  Retarget source motion artifacts into the SONIC embodiment schema.
workflow  Show the SkyPilot YAML template for retargeting.
status  Show retargeting tool status.
list  List supported retargeting source formats.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `run` | Retarget source motion artifacts into the SONIC embodiment schema. |
| `workflow` | Show the SkyPilot YAML template for retargeting. |
| `status` | Show retargeting tool status. |
| `list` | List supported retargeting source formats. |

## Examples

```bash
npa workbench retargeting --help
npa workbench retargeting run --help
```

Convert retargeted Bones-SEED/G1 CSVs to a SONIC motion library:

```bash
npa workbench retargeting run \
  --input-path s3://bucket/bones-seed/g1/csv/210531/ \
  --output-path s3://bucket/sonic-locomotion/run-1/retargeted/ \
  --source-format bones-seed-csv \
  --frame-rate 30 \
  --source-frame-rate 120 \
  --individual \
  --output json
```

The command writes real SONIC motion-lib `.pkl` artifacts plus
`retargeting_result.json` metadata. It no longer writes a manifest-only shim.

Supported robot motion-lib inputs are `auto`, `soma-csv`, `bones-seed-csv`,
`deploy-pkl`, `teleop-pkl`, and `motion-lib`. `bvh` invokes upstream SONIC's
SOMA skeleton extractor and writes SOMA skeleton PKLs; upstream SONIC does not
bundle the raw BVH-to-G1 robot retargeter.

Regenerate this page with `bash scripts/build_docs.sh` after changing `retargeting`.
