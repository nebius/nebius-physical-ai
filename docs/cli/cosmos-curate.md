# `npa workbench cosmos-curate`

## Command Tree

```text
Usage: npa workbench cosmos-curate [OPTIONS] COMMAND [ARGS]...

NVIDIA Cosmos Curator: split, transcode, motion-score, and catalog video clips.

Options
--help  Show this message and exit.
Commands
curate-augmented  Curate a run's augmented variants with the real Cosmos Curator stages.
curate-videos  Run the curator stages over a local directory of videos.
plan-pipeline  Print upstream's `video-pipeline split` command for the curator container.
fetch-models  Download curator model weights with your own Hugging Face token.
models  Show the curator model sets, their upstream pins, and what is present.
engine  Report whether the upstream curator can run in this environment.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `curate-augmented` | Curate a run's augmented variants with the real Cosmos Curator stages. |
| `curate-videos` | Run the curator stages over a local directory of videos. |
| `plan-pipeline` | Print upstream's `video-pipeline split` command for the curator container. |
| `fetch-models` | Download curator model weights with your own Hugging Face token. |
| `models` | Show the curator model sets, their upstream pins, and what is present. |
| `engine` | Report whether the upstream curator can run in this environment. |

## Examples

```bash
npa workbench cosmos-curate --help
npa workbench cosmos-curate curate-augmented --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cosmos-curate`.
