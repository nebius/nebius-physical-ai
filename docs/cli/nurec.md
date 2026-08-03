# `npa workbench nurec`

## Command Tree

```text
Usage: npa workbench nurec [OPTIONS] COMMAND [ARGS]...

NVIDIA Omniverse NuRec / Neural Reconstruction Engine: sensor recordings -> 3DGUT Gaussian reconstruction -> renderable USDZ -> novel-view renders. Requires an RT-core GPU (L40S or RTX PRO 6000
Blackwell); never route the render path at H100/H200.

Options
--help  Show this message and exit.
Commands
check  Check NRE container access, dataset download rights, and GPU suitability.
fetch  Download and unpack the real NCore V4 shards for a scene.
reconstruct  Train a 3DGUT Gaussian reconstruction and publish the renderable USDZ.
render  Render novel views from a trained reconstruction with ``nre render``.
visualize  Build the run's Rerun recording so it renders in the NPA agent viewer.
finalize  Aggregate the run tree into a real final report.
status  Summarize what a NuRec run prefix currently holds, stage by stage.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `check` | Check NRE container access, dataset download rights, and GPU suitability. |
| `fetch` | Download and unpack the real NCore V4 shards for a scene. |
| `reconstruct` | Train a 3DGUT Gaussian reconstruction and publish the renderable USDZ. |
| `render` | Render novel views from a trained reconstruction with ``nre render``. |
| `visualize` | Build the run's Rerun recording so it renders in the NPA agent viewer. |
| `finalize` | Aggregate the run tree into a real final report. |
| `status` | Summarize what a NuRec run prefix currently holds, stage by stage. |

## Examples

```bash
npa workbench nurec --help
npa workbench nurec check --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `nurec`.
