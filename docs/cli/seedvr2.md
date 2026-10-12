# `npa workbench seedvr2`

## Command Tree

```text
Usage: npa workbench seedvr2 [OPTIONS] COMMAND [ARGS]...

Restore low-resolution video with official ByteDance SeedVR2-3B.

Options
--help  Show this message and exit.
Commands
probe  Decode the exact input video and publish a verified media manifest.
restore  Run pinned one-step SeedVR2-3B and publish immutable artifacts.
verify  Re-download and independently verify a delivered SeedVR2 result.
review  Build a non-blended bicubic-baseline versus SeedVR2 review package.
status  Report local batch-stage readiness without claiming model-cache state.
system-info  Report immutable source, model, and payload identities.
list  List the real video-restoration and evidence operations.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `probe` | Decode the exact input video and publish a verified media manifest. |
| `restore` | Run pinned one-step SeedVR2-3B and publish immutable artifacts. |
| `verify` | Re-download and independently verify a delivered SeedVR2 result. |
| `review` | Build a non-blended bicubic-baseline versus SeedVR2 review package. |
| `status` | Report local batch-stage readiness without claiming model-cache state. |
| `system-info` | Report immutable source, model, and payload identities. |
| `list` | List the real video-restoration and evidence operations. |

## Examples

```bash
npa workbench seedvr2 --help
npa workbench seedvr2 probe --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `seedvr2`.
