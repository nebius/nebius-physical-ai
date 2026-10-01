# Workbench benchmarks

Benchmarks pair runnable Workbench workflows with measured results, fixed
workload controls, and an explanation of what the measurements establish.

| Benchmark | Hardware | What it measures |
| --- | --- | --- |
| [Cosmos3-Super](cosmos3-super/README.md) | Eight-GPU HGX B200 and HGX H200 nodes on Nebius | Request latency versus technically valid video output per node-hour across four serving topologies |

Each benchmark should include its workflow YAMLs, hardware and software pins,
request and validation protocol, results with units and source dates, and
reproduction instructions. Keep published reference results distinct from new
Workbench runs and single-GPU checks distinct from full-node measurements.
Credit the authors and link every upstream repository used for its methodology,
software, or results; preserve applicable licenses when adapting material.

The Cosmos3-Super YAMLs are relative symlinks to the maintained
[`workflows/testing/`](../workflows/testing/) specs. They can be passed directly
to the CLI from a Git checkout, while catalog discovery and live-submit coverage
continue to use the original paths. Edit the original spec to update both entry
points. Clone with symlink support; on a platform that checks links out as text,
use the original workflow paths.

For setup and operation, see the [Workbench docs](../docs/workbench/README.md)
and [workflow catalog](../workflows/README.md).
