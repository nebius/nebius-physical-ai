# Upstream comparison and remaining gaps

Reviewed against Workbench base
[`59514b9905c5427a250c2d0d3455b1f8e374b7e5`](https://github.com/nebius/nebius-physical-ai/tree/59514b9905c5427a250c2d0d3455b1f8e374b7e5)
and Stewart Tong's
[`cosmos3-super-serving` revision `951dc1162fcc6c47d694e29616a1f833f293b877`](https://github.com/stewtong/cosmos3-super-serving/tree/951dc1162fcc6c47d694e29616a1f833f293b877)
on October 1, 2026. These findings describe those revisions. This showcase adds
documentation and reference aggregates; it does not change the serving runtime
or measurement protocol.

## What Workbench already covers

The existing [runner](../../npa/src/npa/workbench/cosmos/super_benchmark.py)
pins the model, image, prompt hashes, output shape, sampler, and seed cycle;
checks the physical GPU family; partitions the node into the four topologies;
requires valid warmups; and records failures without video-second credit.
It validates full MP4 decode and blank/frozen checks, derives shared-window
throughput, and persists per-cell evidence with completion markers for resume.
The B200 full suite already includes concurrency-two cells and delayed repeats.
The two single-GPU suites have an explicitly different resource scope.

## Meaningful differences

| Area | Stewart's public reference | Workbench at the reviewed base | Consequence |
| --- | --- | --- | --- |
| Request scheduling | v1 dispatches synchronized rounds across all replicas | `dispatch_cell` synchronizes initial workers, then each replica drains its own request queue | Compare each schedule separately; concurrency-two latency is particularly sensitive to queue shape |
| Timer boundary | v1 stops request timing after complete MP4 persistence and validates outside the production window | `one_attempt` calls `validate_video` before setting `client_wall_seconds`; `dispatch_cell` waits for those calls | Workbench includes validation overhead; exact v1 reproduction needs explicit response-completion timing and deferred validation |
| H200 study coverage | Seven cells: four primary, `H2C2`, `H3C2`, and `H1R` | Four primary cells, plus a separate single-GPU check; no `h200-full` suite | The H200 YAML cannot reproduce all 168 published attempts |
| Routed serving | Named latency/balanced/throughput profiles behind one endpoint, replica-health admission, bounded queue, overload and degraded-capacity controls | Benchmark services are addressed directly and exist only for the current cell | Router behavior needs separate integration and live evidence before Workbench can claim it |
| Routed benchmark protocol | v2 uses a work-conserving client and an explicit `cosmos3_super_serving_v2` comparison basis | No routed v2 benchmark surface | Neither Workbench's current schedule nor upstream v2 results should be relabeled as historical v1 results |
| Public evidence verification | Embedded attempts/windows, aggregate rederivation, checksums, and generated-table drift checks | Live attempt/window/derived artifacts exist, but there is no equivalent standalone public-reference rederivation command | Keep source records and verifiers accessible; a static table alone is not independent raw-evidence verification |

The priority for exact historical reproduction is to align scheduling and timing
under an explicit versioned protocol, with tests and a new live run. Existing
results must retain their original timing meaning. Adding the three missing
H200 cells comes next if complete cross-platform study coverage is needed.
The router is a separate serving feature with its own acceptance work.

The historical
[Workbench B200 run](../../docs/workbench/cosmos3-super-serving.md#complete-live-b200-reproduction-2026-09-03)
reports concurrency-two mean latency 20.87–25.94% above the public record despite
close shared-window throughput. The code differences above establish a protocol
mismatch; they do not by themselves quantify how much of that observed delta
each difference caused.

## Source evidence and verification

Use the upstream revision linked above to inspect or independently verify the
reference evidence:

| Source | Purpose |
| --- | --- |
| [BENCHMARKS.md](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/BENCHMARKS.md) | All 17 primary cells, concurrency deltas, and delayed repeats |
| [METHOD.md](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/reproduce/METHOD.md) | Precise v1 controls, timer boundaries, dispatch schedule, and validity gate |
| [REPRODUCE.md](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/reproduce/REPRODUCE.md) | Full v1 matrix and separate v2 commands |
| [derive-results.py](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/reproduce/derive-results.py) | Rederive aggregates from embedded attempts/windows with `--verify-embedded` |
| [render-benchmarks.py](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/reproduce/render-benchmarks.py) | Check the upstream rendered tables with `--check` |
| [SHA256SUMS](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/results/SHA256SUMS) | Verify source record integrity |
| [SERVING.md](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/SERVING.md) | Routing, queueing, health, and lifecycle behavior |
| [runtime-validation-20260903.json](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/results/runtime-validation-20260903.json) | Live H200 operability evidence, excluded from performance results; no B200 launcher validation claim |

Upstream supplemental startup, guardrail-overhead, and determinism observations
have narrower controls and aggregate-only evidence. Workbench's existing serving
guide already covers those subjects. They are not added to the primary
408-attempt table. See Stewart's
[OBSERVATIONS.md](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/OBSERVATIONS.md)
for their scope and attribution.
