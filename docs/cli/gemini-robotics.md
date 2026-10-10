# `npa workbench gemini-robotics`

## Command Tree

```text
Usage: npa workbench gemini-robotics [OPTIONS] COMMAND [ARGS]...

Gemini Robotics hosted planning and evaluation (plan, eval). Live-validated adapter: API base URL and model id must be supplied explicitly; no live access has been validated.

Options
--help  Show this message and exit.
Commands
plan  Run advisory ER planning and publish one immutable S3 receipt.
eval  Evaluate one durable plan/rubric input and publish an S3 receipt.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `plan` | Run advisory ER planning and publish one immutable S3 receipt. |
| `eval` | Evaluate one durable plan/rubric input and publish an S3 receipt. |

## Examples

```bash
npa workbench gemini-robotics --help
npa workbench gemini-robotics plan --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `gemini-robotics`.
