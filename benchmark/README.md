# Workbench benchmarks

Benchmarks pair runnable Workbench workflows with measured results, fixed
workload controls, and an explanation of what the measurements establish.

| Benchmark | Hardware | What it measures |
| --- | --- | --- |
| [Cosmos3-Super](cosmos3-super/README.md) | Eight-GPU HGX B200 and HGX H200 nodes on Nebius | Request latency versus technically valid video output per node-hour across four serving topologies |
| [Cosmos3-Nano](cosmos3-nano/README.md) | One eight-GPU HGX H200 node on Nebius | Reference results without a Workbench workflow: technically valid video output per node-hour for Nano across ten cells, against the Cosmos3-Super reference |

Each benchmark should include its workflow YAMLs, hardware and software pins,
request and validation protocol, results with units and source dates, and
reproduction instructions. Keep published reference results distinct from new
Workbench runs and single-GPU checks distinct from full-node measurements.
Credit the authors and link every upstream repository used for its methodology,
software, or results; preserve applicable licenses when adapting material.

The Cosmos3-Super YAMLs are byte-identical copies of the maintained
[`workflows/testing/`](../workflows/testing/) specs. They can be passed directly
to the CLI from a Git checkout, while catalog discovery and live-submit coverage
continue to use the original paths. Update both copies together; the existing
documentation guardrails check the complete file set and byte equality to
prevent drift. These regular files also work in downloads without symlink
support.

For setup and operation, see the [Workbench docs](../docs/workbench/README.md)
and [workflow catalog](../workflows/README.md).
