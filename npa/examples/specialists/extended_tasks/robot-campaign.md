# Task: build a resumable robot data campaign

Read the [common benchmark contract](README.md). Planning estimate: **2–4 hours
per arm**, subject to native calibration. This is a local CPU simulation and
dataset-engineering task. It does not train a policy or demonstrate sim-to-real
transfer.

## Agent goal

Turn Workbench's prepared Fetch simulation into a campaign that survives a
controller restart and partial dataset publication. Implement the missing
campaign orchestration, repair discovered integration issues, execute all 216
cases, and publish independently readable LeRobot shards with a complete campaign
index. Make progress and failures inspectable through specialist operation
receipts. Retain the existing controller, physical thresholds and source data.

The existing [workflow driver](../robot_workflow/workload.py) accepts one fresh
output directory and executes cases sequentially. It is not already a resumable
campaign runner. Build around its real simulator/export path; a JSON task list
without actual execution does not satisfy this task.

## Frozen workload

Expand these supported scene factors into explicit immutable input JSON before
either arm runs. Use the existing `npa.robot-workflow.v1` matrix format and validate
every case with `RobotScene` and `load_matrix`.

| Factor | Values |
| --- | --- |
| Object XY → goal XY, meters | `(-0.10,-0.08) → (0.10,0.08)`; `(-0.08,0.10) → (0.09,-0.10)` |
| Lighting | `0.7`, `1.0`, `1.2` |
| Object/target colors | red/green, blue/red, orange/blue |
| Development seeds | `42`, `44`, `142`, `144` |
| Final-validation seeds | `242`, `244`, `342`, `344` |

The Cartesian product gives 72 development and 72 final-validation `pick_place` cases.
Add one `open_gripper` negative control for each development case: **216 cases
total**, 144 expected positive and 72 expected negative. Native calibration must
establish those expectations before freezing. If an expectation is physically
wrong, revise the task version before measurement, never the measured result.

Each episode retains the real 185-step trajectory, both 480×360 camera streams,
actions, states, contacts and object traces: 39,960 control transitions and
79,920 recorded camera samples. There should be 26,640 accepted timesteps and
53,280 accepted camera samples if the frozen positive expectations hold. These
are acceptance targets, not observations from a completed campaign.

These seeds are public. The final-validation partition is disjoint from the
development partition, not a secret benchmark. Run it only on the integrated
candidate, retaining every attempt. The grader and private fault fixtures remain
outside agent write grants. Physical controls remain separate from dataset split
membership; never publish final-validation episodes in the development dataset.

## Engineering milestones

1. Implement stable case and shard identities bound to complete source, scene,
   dependency and configuration hashes. Store durable completion records. Resume
   only verified outputs whose inputs still match; invalidate changed work and
   downstream publication explicitly. Expose status without launching new work.
2. Execute the development matrix as separate shards, preserving rejected
   episodes and deterministic global case ordering. Publish each shard
   transactionally, then a campaign index with exact membership and hashes.
   The index may refer to multiple native datasets; do not hand-edit LeRobot
   parquet/video metadata to pretend they are a merged dataset.
3. Diagnose and recover from the two controlled failures below. Add regression
   checks for each repaired boundary. Integrate all specialist patches and
   rerun affected work under the final source. An unchanged native artifact can
   be reused only with a recorded dependency proof, not a new source label.
4. Run the final-validation matrix from the integrated revision, independently verify
   every case and native shard, and deliver the index, preview videos, patch,
   recovery record and measurements.

## Controlled recovery exercises

The harness kills the experiment's campaign controller immediately after the
first shard's bytes are complete but before its campaign-level acknowledgement.
After restart, the agent must reconcile those bytes and complete the receipt
without generating a duplicate case or re-running already verified simulation.

On another shard, the harness raises an encoder/publication error after raw
episodes exist but before final dataset publication. Raw episodes must remain
intact, no partial dataset can appear as complete, and recovery must encode from
verified raw evidence. The fault adapter and its trigger receipt are immutable;
agents must repair production recovery behavior rather than disable the fixture.

## Independent acceptance

The existing [replay and reader](../robot_workflow/README.md) verify trajectories,
camera alignment, physical outcomes and native LeRobot samples for each shard.
The new trusted campaign grader must additionally check:

- Exact 216-case coverage, no duplicate or cross-split case identities, expected
  physical outcomes, and negative controls excluded from accepted datasets.
- Complete source/input bindings, real native loading of every accepted timestep,
  matching language labels, both camera timelines and contiguous shard indices.
- Valid campaign-to-shard membership and global ordering; no phantom rows after
  the failed publication and no mutation of reused artifacts.
- Recorded restart and encoder failures, successful reconciliation, and absence
  of duplicated work after a lost acknowledgement.
- A source-change fixture invalidates affected cached work; stale output cannot
  satisfy final-source acceptance.

Freeze the campaign grader outside agent grants. The existing per-shard reader
alone cannot establish restart behavior or campaign-wide uniqueness.

## Suggested specialist ownership

- **Campaign operations:** durable scheduling, operation status and resume logic.
- **Dataset publication:** staging, shard indexing and native format compatibility.
- **Evidence integration:** production provenance and campaign reports; this role
  does not author or weaken the independent grader.

Relevant source is under `npa/src/npa/workbench/token_factory/`,
`npa/src/npa/adapter/sim_to_lerobot.py`, and
`npa/examples/specialists/robot_workflow/`. Freeze exact disjoint write grants
after inspecting the source. Keep orchestration in a named campaign module;
avoid modifying the simulator merely to simplify acceptance.
