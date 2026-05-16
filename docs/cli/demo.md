# `npa demo`

## Command Tree

```text
Usage: npa demo [OPTIONS] COMMAND [ARGS]...

Demo artifact bootstrap and verification helpers.

Options
--help  Show this message and exit.
Commands
stage  Stage demo artifacts into an operator-owned bucket.
verify  Verify staged demo artifacts without downloading object contents.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `stage` | Stage demo artifacts into an operator-owned bucket. |
| `verify` | Verify staged demo artifacts without downloading object contents. |

## Examples

```bash
npa demo --help
npa demo stage --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `demo`.
