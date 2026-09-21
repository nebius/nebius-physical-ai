# Simulation fanout experiment

Six Workbench specialists using GLM and DeepSeek completed one full 36-episode
sweep **2.81 times faster**, with **68.8% lower estimated API-equivalent token
cost**, than a single Astra agent with the same simulator concurrency. Across
all three primary trials, however, specialists completed **96/108 episodes**
and Astra completed **108/108**. This experiment shows a cost and latency
opportunity, with a reliability gap; it does not establish overall superiority.

## What actually ran

On 2026-09-21, six independent LangGraph worker processes operated scoped
Workbench file and command tools. Three used `zai-org/GLM-5.3` (requested low
reasoning) and three used `deepseek-ai/DeepSeek-V4-Pro-0813` (requested
`reasoning_effort=none`), hosted by Token Factory. Assignments rotated between
rounds. Explicit assignment was
used; Jev was not involved in this experiment.

Each task started with a broken goal position and missing mass variations.
Agents read a geometric instruction, observed failing validation, repaired
`plan.json`, validated it, and ran the sweep. Tasks covered translation,
reflection, diagonal displacement, rotation, crossing and offset coordinates.
Each swept mass multipliers `[0.8, 1.0, 1.2]` against sliding-friction multipliers
`[0.8, 1.2]`: six cases per task, 36 per arm and seed.

The baseline was one actual `gpt-6-astra` Codex agent with the identical
Workbench tool executor and scopes, plus an explicit batch operation to launch
all six sweeps concurrently. No shell or additional model workers were allowed
in that baseline. Medium reasoning was primary; xhigh was measured separately.
This measures the proposed specialist architecture against a single agent on
this interface. It does not isolate model quality from orchestration, or compare
against an unrestricted Codex team.

Both arms ran the same existing Workbench MuJoCo Fetch controller, with up to
six simulator processes on one Apple M4 Pro host (12 CPU cores, 48 GiB RAM).
MuJoCo was 3.3.7 and Gymnasium Robotics was 1.4.1. No remote GPU resources were
provisioned. Seeds were 11, 23 and 37; execution order was reversed in round 2.
The task matrix, prices and execution-source hashes were retained before the
primary runs. No failed trial was replaced with a favorable rerun.

## All measured trials

Time includes agent calls, tools and worker startup after workspace preparation.
It excludes independent replay verification and initial implementation/setup.

| Seed | Arm | Verified episodes | Agent + tools | Estimated token cost |
|---|---|---:|---:|---:|
| 11 | Six specialists | 30/36 | 46.32 s | Unknown; recorded portion $0.11945 |
| 11 | Astra medium | 36/36 | 55.23 s | $0.41279 |
| 23 | Six specialists | 36/36 | 22.10 s | $0.14237 |
| 23 | Astra medium | 36/36 | 62.18 s | $0.45562 |
| 37 | Six specialists | 30/36 | 45.10 s | Unknown; recorded portion $0.12385 |
| 37 | Astra medium | 36/36 | 83.27 s | $0.50391 |
| 23 | Astra xhigh, secondary | 36/36 | 68.33 s | $0.57694 |

The complete seed-23 specialist run cost approximately **$0.00395 per accepted
episode**, versus Astra medium's **$0.01266**. Against the secondary xhigh run,
the same specialist run was 3.09 times faster and its estimated token cost was
75.3% lower. These are individual paired observations, not confidence intervals
or fleet-wide savings estimates. Six agents also create concurrent model
requests; provider quotas and host contention can change the result.

Token costs use actual retained usage counters and a dated
[rate snapshot](../../npa/examples/specialists/simulation/prices.json).
Token Factory input/output rates came from its
[public model catalog](https://tokenfactory.nebius.com/api/public/models_info);
all input was charged at the normal rate, with no assumed cache discount.
Astra uses the [published Standard API rates](https://developers.openai.com/api/docs/models/gpt-6-astra),
including its reported cached-input and cache-write counters. Reasoning output
is included in output tokens and counted once. Astra was accessed through a
ChatGPT-authenticated CLI; these estimates are **not its subscription invoice**.
Host compute, development and human intervention are excluded. Because two
specialist responses lost usage telemetry, aggregate specialist cost and
aggregate savings are unknown.

## Physics evidence and failures

All **240 produced episodes** across the seven measured runs passed independent
recorded-action replay and physical acceptance. Each retained an actual
185-frame, 25-fps camera video and physics trace. Grading also checked the
requested scene, complete mass/friction coverage and artifact hashes. Separate
negative controls rejected a gripper that never closed and a modified joint
trace with a recomputed file hash. These checks establish that completed
artifacts represent the requested simulation, rather than accepting model text
as evidence.

The two missing sweeps belonged to the DeepSeek `offset` worker. The original
runtime rejected incomplete responses before recording their usage, leaving
both the exact finish reason and billed tokens unknown. The PR now retains
usage and finish reason for rejected responses, displays their rejection in the
monitor, and still executes none of their proposed tools.

Two separately labeled checkpoint retries, one per failed task, also stopped.
The new telemetry recorded `finish_reason=length` and 8,192 completion tokens
for each. No caller token cap was supplied. Those diagnostic retries took an
additional 33.77 and 36.86 seconds and are not folded into or substituted for
the original trials. The telemetry bug is fixed; the provider/model completion
failure remains unresolved. The other five workers completed independently in
each affected run.

The simulation controller was a known scripted policy, and the healthy
reference configurations passed all cases before evaluation. Once these
configurations are known, an ordinary script can execute the same sweeps without
an LLM. The demonstrated value is interpreting tasks, repairing configuration,
independent execution and inspectable evidence. This does not demonstrate
policy learning, sim-to-real transfer, Cosmos throughput or autonomous repair
of arbitrary research code.

## Reproduce and inspect

The [example instructions](../../npa/examples/specialists/simulation/README.md)
include installation, the full ordered matrix and independent scoring. Reviewed
[aggregate measurements](../../npa/examples/specialists/simulation/results/2026-09-21.json)
retain per-model token counters, per-task outcomes and the frozen protocol.
Exact checkpoints, task transcripts, raw operational receipts and videos remain
in operator-owned evidence outside Git.
