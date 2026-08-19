# Antioch / OpenPI continuous control demo

Status: **passed** on 2026-08-19.

This public-safe report records the production-mode integration proof without
live infrastructure identities, endpoints, credentials, frames, prompts, or
customer data. The run used an Antioch-hosted RT-capable RTX Isaac/Franka
scenario and a separate B200 OpenPI pi0.5-DROID policy workload. The policy was
reachable only through the authenticated relay to a private Kubernetes
ClusterIP service. Both workloads used digest-pinned images; model and NVIDIA
runtime payloads were fetched into operator-owned runtime caches and were not
image layers.

## Sustained result

| Signal | Measured result |
| --- | ---: |
| elapsed interval | 38.55 s |
| advancing render ticks | 126 |
| render rate | 3.27 fps |
| advancing exterior + wrist observations | 115 |
| observation rate | 2.98 fps |
| policy requests / completed round trips | 28 / 18 |
| response contract | finite `[15,8]` chunks |
| safely applied position targets | 69 |
| control-step rate | 2.98 fps |
| inference latency p50 / p95 | 70.62 / 182.00 ms |
| observation age p95 | 270.82 ms |
| response age p95 | 392.89 ms |
| latest-frame replacements | 83 |
| chunk underruns | 5 |
| reconnects / transport failures | 10 / 8 |
| rejected responses / stale responses | 10 / 0 |
| safe holds | 46 |

Readiness was true only after camera timestamps advanced, repeated policy round
trips completed, and multiple targets were safely applied over the sustained
interval. Rendering and physics continued while inference was in flight.
Latest-observation replacement bounded backpressure instead of accumulating
stale frames.

The naturally exercised transport faults caused safe holds and epoch-reset
reconnects; fresh validated chunks resumed control afterward. No stale response
executed. Deterministic fault tests separately inject late, disconnected,
malformed, and unsafe responses and assert stale-response rejection, bounded
queues, epoch reset, and fail-closed no-action behavior.

This is measured **soft-real-time** behavior. Python, WebSocket, Kubernetes, and
the authenticated relay do not provide hard-real-time or deterministic latency
guarantees.
