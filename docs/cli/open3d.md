# `npa workbench open3d`

## Command Tree

```text
Usage: npa workbench open3d [OPTIONS] COMMAND [ARGS]...

Open3D point-cloud registration and surface reconstruction.

Options
--help  Show this message and exit.
Commands
prepare  Index the scans under a prefix into a registration manifest.
stage-demo  Publish the upstream open3d.data DemoICPPointClouds scans as input.
register  Register consecutive pairs with RANSAC/FPFH, then refine with ICP.
multiway  Build the full pairwise pose graph, optimize it, and fuse the scans.
validate  Re-verify a published registration from its artifacts alone.
reconstruct  Reconstruct a Poisson surface from the fused multiway cloud.
visualize  Build and verify an RRD of the optimized scans, fusion and surface.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `prepare` | Index the scans under a prefix into a registration manifest. |
| `stage-demo` | Publish the upstream open3d.data DemoICPPointClouds scans as input. |
| `register` | Register consecutive pairs with RANSAC/FPFH, then refine with ICP. |
| `multiway` | Build the full pairwise pose graph, optimize it, and fuse the scans. |
| `validate` | Re-verify a published registration from its artifacts alone. |
| `reconstruct` | Reconstruct a Poisson surface from the fused multiway cloud. |
| `visualize` | Build and verify an RRD of the optimized scans, fusion and surface. |

## Examples

```bash
npa workbench open3d --help
npa workbench open3d prepare --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `open3d`.
