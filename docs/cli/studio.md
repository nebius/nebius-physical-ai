# `npa studio`

## Command Tree

```text
Usage: npa studio [--registry studio.json] <film> <command> [options]

Commands: list, brief, scenes, narrate, draft, watch, preview, final.
Example: npa studio inference draft --scene 01-opening --open

Create an empty portable studio from the repository renderer:
npa studio init --directory ./my-studio --renderer docs/demos/executive-film

The registry selects a local renderer directory (default: renderer beside it).
Add film aliases under projects in studio.json. Each alias names a film-project.json.
Rendering needs FFmpeg, Pillow and NumPy; narration additionally needs edge-tts.
Credentials belong in external NPA configuration, never in a studio project.
```

## Options

No command-specific options are listed by `--help`.

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa studio --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `studio`.
