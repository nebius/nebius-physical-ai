# `npa workbench encord`

## Command Tree

```text
Usage: npa workbench encord [OPTIONS] COMMAND [ARGS]...

Register S3 media with Encord SaaS and materialize curated results.

Options
--help  Show this message and exit.
Commands
import-labels  Create an ontology and project, then save unreviewed video prelabels.
render-labels  Verify exported Encord labels and render them over returned video bytes.
push  Register or explicitly upload S3 media and write a durable receipt.
pull  Materialize an Encord source to S3 with an exact lineage manifest.
verify-roundtrip  Verify identity, destination existence, size, and compatible checksums.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `import-labels` | Create an ontology and project, then save unreviewed video prelabels. |
| `render-labels` | Verify exported Encord labels and render them over returned video bytes. |
| `push` | Register or explicitly upload S3 media and write a durable receipt. |
| `pull` | Materialize an Encord source to S3 with an exact lineage manifest. |
| `verify-roundtrip` | Verify identity, destination existence, size, and compatible checksums. |

## Examples

```bash
npa workbench encord --help
npa workbench encord import-labels --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `encord`.
