# `npa workbench openvla`

## Command Tree

```text
Usage: npa workbench openvla [OPTIONS] COMMAND [ARGS]...

OpenVLA-OFT LIBERO preparation, training, rollout, and evidence.

Options
--help  Show this message and exit.
Commands
prepare  Inspect real RLDS data and publish normalization provenance.
train  Run upstream OFT fine-tuning; no stock-decoder fallback exists.
rollout  Run upstream closed-loop LIBERO rollouts and retain MP4 evidence.
evaluate  Compute held-out numerical success from verified raw rollout evidence.
visualize  Create CSV/SVG comparison artifacts from verified evaluation metrics.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `prepare` | Inspect real RLDS data and publish normalization provenance. |
| `train` | Run upstream OFT fine-tuning; no stock-decoder fallback exists. |
| `rollout` | Run upstream closed-loop LIBERO rollouts and retain MP4 evidence. |
| `evaluate` | Compute held-out numerical success from verified raw rollout evidence. |
| `visualize` | Create CSV/SVG comparison artifacts from verified evaluation metrics. |

## Examples

```bash
npa workbench openvla --help
npa workbench openvla prepare --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `openvla`.
