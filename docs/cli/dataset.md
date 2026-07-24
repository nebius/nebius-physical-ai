# `npa workbench dataset`

## Command Tree

```text
Usage: npa workbench dataset [OPTIONS] COMMAND [ARGS]...

Dataset-of-record: ingest, validate, curate, and query production sensor data.

Options
--help  Show this message and exit.
Commands
ingest  Ingest raw sensor data into a versioned dataset-of-record manifest.
validate  Validate a dataset manifest against schema + quality thresholds.
curate  Slice a dataset version by event/location/quality with lineage.
query  Query dataset records by event/location/quality facets.
status  Fetch a registered dataset version status.
system-info  Show dataset-of-record runtime information.
list  List service-managed dataset versions.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `ingest` | Ingest raw sensor data into a versioned dataset-of-record manifest. |
| `validate` | Validate a dataset manifest against schema + quality thresholds. |
| `curate` | Slice a dataset version by event/location/quality with lineage. |
| `query` | Query dataset records by event/location/quality facets. |
| `status` | Fetch a registered dataset version status. |
| `system-info` | Show dataset-of-record runtime information. |
| `list` | List service-managed dataset versions. |

## Examples

```bash
npa workbench dataset --help
npa workbench dataset ingest --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `dataset`.
