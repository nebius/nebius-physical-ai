# `npa workbench insights`

## Command Tree

```text
Usage: npa workbench insights [OPTIONS] COMMAND [ARGS]...

Insights: lineage graph + common metrics store over workflow-run artifacts.

Options
--help  Show this message and exit.
Commands
record  Record metric emissions + lineage into the store.
ingest-run  Non-invasively ingest a run prefix into the store.
query  Query metric records by facet.
lineage  Traverse the provenance graph for an artifact/version.
compare  Compare a metric set between two run ids; flag regressed/improved.
dashboard  Build a dashboard rollup + optional static HTML report.
status  Report store totals and (optionally) a per-run rollup.
system-info  Show insights runtime information.
list  List service-tracked insights stores.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `record` | Record metric emissions + lineage into the store. |
| `ingest-run` | Non-invasively ingest a run prefix into the store. |
| `query` | Query metric records by facet. |
| `lineage` | Traverse the provenance graph for an artifact/version. |
| `compare` | Compare a metric set between two run ids; flag regressed/improved. |
| `dashboard` | Build a dashboard rollup + optional static HTML report. |
| `status` | Report store totals and (optionally) a per-run rollup. |
| `system-info` | Show insights runtime information. |
| `list` | List service-tracked insights stores. |

## Examples

```bash
npa workbench insights --help
npa workbench insights record --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `insights`.
