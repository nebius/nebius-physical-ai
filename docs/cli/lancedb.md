# `npa workbench lancedb`

## Command Tree

```text
Usage: npa workbench lancedb [OPTIONS] COMMAND [ARGS]...

Deploy and query LanceDB vector-search workbenches.

Options
--help  Show this message and exit.
Commands
deploy  Deploy or register a LanceDB service.
status  Check endpoint reachability.
list  List tables in a LanceDB instance.
create-table  Create or update a LanceDB table.
query  Run a vector search query.
import-lerobot  Import a LeRobot dataset into a LanceDB table.
import-bdd100k  Import BDD100K rows through local mode or a deployed service endpoint.
backfill  Backfill one BDD100K UDF column through local mode or a deployed service.
create-mv  Create a filtered materialized view through local mode or a deployed service.
refresh-mv  Refresh a registered materialized view through local mode or a deployed service.
query-table  Run a bounded SQL-filtered LanceDB table query.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `deploy` | Deploy or register a LanceDB service. |
| `status` | Check endpoint reachability. |
| `list` | List tables in a LanceDB instance. |
| `create-table` | Create or update a LanceDB table. |
| `query` | Run a vector search query. |
| `import-lerobot` | Import a LeRobot dataset into a LanceDB table. |
| `import-bdd100k` | Import BDD100K rows through local mode or a deployed service endpoint. |
| `backfill` | Backfill one BDD100K UDF column through local mode or a deployed service. |
| `create-mv` | Create a filtered materialized view through local mode or a deployed service. |
| `refresh-mv` | Refresh a registered materialized view through local mode or a deployed service. |
| `query-table` | Run a bounded SQL-filtered LanceDB table query. |

## Examples

```bash
npa workbench lancedb --help
npa workbench lancedb deploy --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `lancedb`.
