# Simulation fanout comparison

Compare six independently running Workbench specialists against one Codex Astra
agent using the same scoped tools and simulator concurrency. Both repair scene
configurations from task instructions and execute actual MuJoCo Fetch
pick-and-place sweeps. This is an opt-in paid experiment, not a CI benchmark.

The [recorded experiment](../../../../docs/workbench/specialists-simulation-experiment.md)
retains the original reliability failures and the subsequent recovery experiment.
This harness now implements protocol v2: explicit reciprocal model backups,
required successful validation/simulation receipts, vendor-recommended DeepSeek
sampling, and clarified coordinate bounds shared by both arms. Keep every
predeclared trial, including failures, when reproducing it. Original v1 results
retain their original source hashes; running this version is a new measurement.

## Install and preflight

Use an isolated checkout and its own Python 3.12 environment:

```bash
uv venv npa/.venv --python 3.12
uv pip install --python npa/.venv/bin/python -e 'npa[agent-specialists,robot-sdg]'
uv pip install --python npa/.venv/bin/python -r npa/examples/specialists/simulation/requirements.txt
npa/.venv/bin/python -m npa workbench health preflight --checks token_factory --json
npa/.venv/bin/python -m npa workbench token-factory models
codex login status
```

The optional MCP dependency is for the Astra baseline only. This example uses
MCP 2's `MCPServer`, not the earlier `FastMCP` import. The recorded baseline used
Codex CLI 0.154.0 with access to `gpt-6-astra`. Verify your account's access to the
exact model IDs in `experiment.py`; availability can differ by account.
Store the Token Factory key through NPA's credential store or
`NEBIUS_TOKEN_FACTORY_KEY`, never in team JSON or source. Rendering needs a
working MuJoCo OpenGL backend. The recorded rendering path was macOS; headless
Linux was not validated. For a Linux adaptation, configure EGL/OSMesa and add
the required `MUJOCO_GL`/`PYOPENGL_PLATFORM` names to each operation's `pass_env`
in `_operations` before measuring. Workbench deliberately filters command
environments, so exporting renderer variables alone does not forward them.

Each run creates six workspaces and a private durable state directory outside
the checkout. The model can read only `TASK.md` and `plan.json`, edit only
`plan.json`, and invoke fixed `validate` and `simulate` operations. These are
tool grants, not an operating-system sandbox. Commands and state remain under
the operator's account. Do not publish raw checkpoints, model output or logs.

## Run the fixed matrix

Choose a new private output directory, set `BENCHMARK_ROOT`, and preserve every
run. Each task has six cases: three cube masses crossed with two sliding-friction
values. A simulator process runs those cases sequentially; both arms can run six
task processes concurrently. The specialist arm alternates GLM and DeepSeek;
`--rotation` reverses assignments between paired rounds.

```bash
umask 077
mkdir -p "$BENCHMARK_ROOT"
cp npa/examples/specialists/simulation/prices.json "$BENCHMARK_ROOT/prices.json"
round=0
for seed in 11 23 37 53 71 89; do
  round=$((round + 1))
  rotation=$((round - 1))
  if [ $((round % 2)) -eq 1 ]; then
    arms=(specialists astra-medium)
  else
    arms=(astra-medium specialists)
  fi
  for arm in "${arms[@]}"; do
    npa/.venv/bin/python npa/examples/specialists/simulation/experiment.py --live \
      --output "$BENCHMARK_ROOT/round-$round/$arm" --arm "$arm" \
      --seed "$seed" --rotation "$rotation"
  done
done
npa/.venv/bin/python npa/examples/specialists/simulation/score.py --root "$BENCHMARK_ROOT"
```

`prices.json` is a dated rate snapshot. Verify the linked provider rates and
update your private copy before a new measurement. The output is an estimated
API-equivalent token cost, not a subscription invoice or total deployment cost.

The baseline receives an additional `run_operations` batch tool, so Astra can
launch all six sweeps at once. Its native shell and model delegation are
disabled to keep the Workbench tool interface matched. Medium reasoning is the
primary comparison; `astra-xhigh` is available for separately labeled sensitivity
runs and was used in v1. This is not a test of
unrestricted Codex coding or Astra with its own model workers.

The timer covers agent calls, tools and worker startup after preparation,
including every failed generation and automatic backup call. Backups use the
other configured open-weight model, never Astra. Seeds are repeat trials on the
same six tasks, not held-out task families. Record your source hashes, package
versions and protocol before starting; dependency upgrades may change physics.
Independent grading happens afterwards. `task-receipts.json` and
`execution.json` are the end-of-trial snapshots; keep them immutable. The scorer
requires a successful simulation receipt from that snapshot before accepting
on-disk artifacts. Supplemental recovery must not replace the original result.
The harness imposes no token, runtime or iteration cap. Provider limits still
apply. Existing output directories are rejected rather than overwritten.

## Evidence and validation

Every completed case retains a 185-step physics trace and a 7.4-second,
25-fps two-camera video. `latest.json` locates each task report. The grader checks
task coordinates, colors, lighting, exact parameter coverage, plan/video/trace
hashes, grasp/lift/place/release/settling acceptance, and a fresh replay of every
recorded action. It compares joint state, object/gripper positions, contacts and
environment success with tolerance `1e-6`. The controller is a fixed privileged
state script from Workbench; the agents repair configuration and orchestrate
runs, rather than learn a controller.

```bash
npa/.venv/bin/python -m pytest npa/tests/agent_eval/test_specialist_simulation_benchmark.py -q
NPA_SIMULATION_BENCHMARK_LIVE=1 npa/.venv/bin/python -m pytest npa/tests/agent_eval/test_specialist_simulation_benchmark.py -q
```

The second command also runs real rendered negative controls: an always-open
gripper must fail grasp/lift checks, and a modified state trace must fail replay
even when its file hash is updated. These checks do not use paid inference.
