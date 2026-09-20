# `npa studio`

## Command Tree

```text
Usage: npa studio [--registry studio.json] <film> <command> [options]

Commands: init, create, list, brief, scenes, narrate, draft, watch, preview, final.
Example: npa studio inference draft --scene 01-opening --open

Search accessible object storage using external NPA configuration:
npa studio search --all-projects --query cosmos --kind video

Create a portable studio using the installed renderer:
npa studio init --directory ./my-studio
cd my-studio
npa studio create demo --input-path ./my-clip.mp4 --duration 30

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
