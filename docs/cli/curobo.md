# `npa workbench curobo`

## Command Tree

```text
Usage: npa workbench curobo [OPTIONS] COMMAND [ARGS]...

NVIDIA cuRobo V2 motion planning and complete benchmark evaluation.

Options
--help  Show this message and exit.
Commands
prepare  Write a complete benchmark recipe (both full datasets; no problem cap).
benchmark  Run every benchmark problem in each selected dynamics configuration.
plan  Plan each Franka start/goal and cuboid scene from an S3 input manifest.
validate  Verify journal hashes, identities, finite trajectories and all denominators.
visualize  Build and verify RRD joint timelines and actual forward-kinematics paths.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `prepare` | Write a complete benchmark recipe (both full datasets; no problem cap). |
| `benchmark` | Run every benchmark problem in each selected dynamics configuration. |
| `plan` | Plan each Franka start/goal and cuboid scene from an S3 input manifest. |
| `validate` | Verify journal hashes, identities, finite trajectories and all denominators. |
| `visualize` | Build and verify RRD joint timelines and actual forward-kinematics paths. |

## Examples

```bash
npa workbench curobo --help
npa workbench curobo prepare --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `curobo`.
