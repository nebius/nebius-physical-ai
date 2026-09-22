# Simulation fanout and reliability experiments

After the changes below, a fresh three-pair confirmation completed **108/108
physics cases on each side**. Specialists finished **2.79 times faster** with
**68.6% lower estimated API-equivalent token cost** than one Astra
agent with matched Workbench tools and simulator concurrency. This matches
completion on these six known tasks; it does not establish general model parity
or statistical non-inferiority. All earlier failures remain below and in the
[reviewed measurements](../../npa/examples/specialists/simulation/results/2026-09-21-reliability.json).

## Changes that address the observed failures

DeepSeek's retained failed prompt showed repeated visible planning, with no
native tool call. It also confused the coordinate bound with the displacement
between object and goal. Both arms now receive the same clarified coordinate
instruction. DeepSeek uses temperature 1.0 and top-p 0.95, following its
[agentic sampling recommendation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/main/README.md).
The diagnostic and experiments requested `reasoning_effort=none`; GLM retained
its requested low-reasoning template option. Changes were combined, so the
experiment does not isolate each contribution.

Profiles can explicitly authorize a backup endpoint. A rejected generation
advances to that model at a checkpoint, preserving completed work, grants and
all reported usage. Required-operation gates reject final answers until
validation and simulation have succeeded after the latest edit. Cached receipts
from earlier checks cannot certify a later edit. Provider identity mismatches,
refusals and uncertain tool effects do not trigger model fallback.

A separate live regression resumed the original eight-message failing prompt
with its original greedy DeepSeek settings. DeepSeek again returned
`finish_reason=length` at 8,192 completion tokens. The runtime automatically
switched to GLM, which repaired the plan and completed **6/6 replay-verified
cases**, without human intervention after launch. Total new agent/tool time was
**49.96 seconds** and the estimated token cost was **$0.05424**, including the
rejected response. The imported history contained only two reads and one failed
validation, with no prior successful edit or simulation. This regression is
separate from every primary trial.

Refreshing latest main also exposed an independent install regression:
Gymnasium Robotics 1.4.2 plus MuJoCo 3.13.0 failed during Fetch initialization.
Its joint utility uses enum tuple membership on NumPy scalars; the newer enum
comparison rejects that check. Upstream's [current utility source](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/utils/mujoco_utils.py)
casts these values to Python integers, but that fix was absent from the released
1.4.2 wheel. The `robot-sdg` extra keeps MuJoCo 3.3.7 with Robotics 1.4.2 until a
compatible release can pass the real simulator tests. No vendor code is patched
at runtime. Both physical negative controls pass with the declared pair.

## Fresh-install confirmation

Three pairs were declared before execution, after the final receipt-gate fix,
rebase and dependency correction: seeds 11, 23 and 37, reversing arm order in
round two. The final code and environment were frozen before these runs. All
six known task families and their mass/friction cases were retained, with fresh
workspaces and fresh Astra runs. These are repeats, not held-out task families.

| Seed | Arm | Verified cases | Agent + tools | Estimated token cost |
|---|---|---:|---:|---:|
| 11 | Astra medium | 36/36 | 61.22 s | $0.48058 |
| 11 | Specialists | 36/36 | 23.66 s | $0.14570 |
| 23 | Astra medium | 36/36 | 65.11 s | $0.48008 |
| 23 | Specialists | 36/36 | 22.83 s | $0.15583 |
| 37 | Astra medium | 36/36 | 67.38 s | $0.47908 |
| 37 | Specialists | 36/36 | 22.84 s | $0.15050 |

Summed wall time was **69.34 s versus 193.70 s**; summed token estimates were
**$0.45203 versus $1.43974**. Ratios use those sums, not the best pair.
The final specialists needed no model fallback in these trials. Each final
answer passed the configured completion gate. All **216** produced episodes
passed independent action replay, and all **39,960** video frames decoded.
The controller was still scripted; no policy learning or physical robot transfer
was tested. Jev was not used. Astra had medium reasoning, the same scope and a
concurrent batch tool; unrestricted Codex teams were not compared.

Costs include every recorded failed generation and backup call when present.
They use the dated rates and actual usage described in the original experiment
below, with no assumed Token Factory cache discount. They are not subscription
invoices or total deployment costs. Model/tool timers exclude preparation and
independent grading. Source hashes, package versions, per-model usage and
per-task outcomes are retained in the linked data.

## Earlier six-pair improvement matrix, including the host failure

The first improvement matrix declared six pairs before execution, on MuJoCo
3.3.7 and Robotics 1.4.1. Five complete pairs each delivered 36/36 on both sides.
During seed 53, the shared host exhausted disk space. SQLite writes and the
experiment's final evidence write failed; the specialists retained **18/36**
verified episodes. Durable receipts were recovered without resuming that trial.
Partial artifacts from the three interrupted tasks remain in private evidence;
they lack completed sweep receipts and are excluded from accepted totals.
Its full wall time and cost are unknown. Re-downloadable package cache was
pruned, and the remaining scheduled rounds ran without replacing the failed
one. Overall this matrix is **198/216 versus 216/216**, not equal completion.

| Seed | Specialists | Astra | Time: specialists / Astra | Token estimate: specialists / Astra |
|---|---:|---:|---:|---:|
| 11 | 36/36 | 36/36 | 78.05 s / 92.10 s | $0.14845 / $0.50495 |
| 23 | 36/36 | 36/36 | 32.77 s / 84.90 s | $0.16102 / $0.48652 |
| 37 | 36/36 | 36/36 | 28.77 s / 68.67 s | $0.16770 / $0.50441 |
| 53 | 18/36 | 36/36 | Unknown (disk failure) / 61.52 s | Unknown / $0.38299 |
| 71 | 36/36 | 36/36 | 25.94 s / 71.60 s | $0.16364 / $0.45523 |
| 89 | 36/36 | 36/36 | 22.78 s / 74.38 s | $0.16133 / $0.67154 |

All **414** accepted episodes passed replay and all **76,590** video frames
decoded. A subsequent gate review found the cached-old-receipt edge case; the
final runtime rejects it. Auditing all 36 task histories found no repeated tool
receipt IDs, and all 33 completed tasks passed the corrected gate. The separate
three-pair confirmation above tests that final code and the declared dependency
pair. It supplements this matrix and does not erase its infrastructure failure.

## Original simulation fanout experiment

Six Workbench specialists using GLM and DeepSeek completed one full 36-episode
sweep **2.81 times faster**, with **68.8% lower estimated API-equivalent token
cost**, than a single Astra agent with the same simulator concurrency. Across
all three primary trials, however, specialists completed **96/108 episodes**
and Astra completed **108/108**. This experiment shows a cost and latency
opportunity, with a reliability gap; it does not establish overall superiority.

### What actually ran

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

### All measured trials

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

### Physics evidence and failures

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
the original trials. At the end of this original experiment, telemetry was fixed but the model
completion failure remained unresolved; the later recovery experiment above
addresses that observed failure mode. The other five workers completed independently in
each affected run.

The simulation controller was a known scripted policy, and the healthy
reference configurations passed all cases before evaluation. Once these
configurations are known, an ordinary script can execute the same sweeps without
an LLM. The demonstrated value is interpreting tasks, repairing configuration,
independent execution and inspectable evidence. This does not demonstrate
policy learning, sim-to-real transfer, Cosmos throughput or autonomous repair
of arbitrary research code.

### Reproduce and inspect

The [example instructions](../../npa/examples/specialists/simulation/README.md)
include installation, the full ordered matrix and independent scoring. Reviewed
[aggregate measurements](../../npa/examples/specialists/simulation/results/2026-09-21.json)
retain per-model token counters, per-task outcomes and the frozen protocol.
Exact checkpoints, task transcripts, raw operational receipts and videos remain
in operator-owned evidence outside Git.
