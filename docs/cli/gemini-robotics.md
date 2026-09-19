# `npa workbench gemini-robotics`

## Command Tree

```text
Usage: npa workbench gemini-robotics [OPTIONS] COMMAND [ARGS]...

Gemini Robotics API-backed toolRef (plan, adapt, eval). Provisional adapter: API base URL and model id must be supplied explicitly; no live access has been validated.

Options
--help  Show this message and exit.
Commands
plan  Run ER embodied-reasoning planning via the Gemini API.
adapt  Submit an on-device adaptation job via the Gemini API.
eval  Evaluate a plan against a rubric via the Gemini API.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `plan` | Run ER embodied-reasoning planning via the Gemini API. |
| `adapt` | Submit an on-device adaptation job via the Gemini API. |
| `eval` | Evaluate a plan against a rubric via the Gemini API. |

## Examples

```bash
npa workbench gemini-robotics --help
npa workbench gemini-robotics plan --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `gemini-robotics`.
