# Prepared robot workflow for agent repair experiments

This example runs Workbench's real MuJoCo Fetch pick-and-place simulator and
LeRobot exporter on operator-prepared scenes. Each case records 185 control
steps, two 480×360 camera streams, nine joint positions, four actions, contact
and object traces, and a preview video. The open-gripper case is an intentional
physical failure and must remain outside the training dataset.

The executable path is **local CPU Workbench execution**. It invokes the same
`robot_sim` and `robot_artifacts` implementation used by
[`token-factory-robot-sdg.yaml`](../../../../workflows/testing/token-factory-robot-sdg.yaml).
It does not submit a cloud workflow, call a planning model, train a policy, or
demonstrate physical robot transfer. Prepared scenes keep simulation inputs
identical when comparing different agents. The ordinary hosted scene-planning
workflow remains available independently.

`multimodel-scenes.json` extends the original matrix to six cases by varying
seed, lighting and colors for both transfer layouts and the failed-grasp control.
Use it with the same commands below. The
[multi-model repair report](../../../../docs/workbench/specialists-multimodel-repair.md)
records real Flash and full-GLM code changes, failed attempts and native validation.

Install `npa[robot-sdg,adapter]` in the checkout's own virtualenv and provide
`ffmpeg`. On headless Linux install `libosmesa6` and set `MUJOCO_GL=osmesa` and
`PYOPENGL_PLATFORM=osmesa` before starting Python. Native dataset verification
uses a separate interpreter with `lerobot==0.5.1` and PyAV; its dependency pins
can differ from Workbench's environment.

```bash
npa/.venv/bin/python npa/examples/specialists/robot_workflow/workload.py \
  --matrix npa/examples/specialists/robot_workflow/scenes.json --validate-only
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  npa/.venv/bin/python npa/examples/specialists/robot_workflow/workload.py \
  --matrix npa/examples/specialists/robot_workflow/scenes.json \
  --output "$NEW_RUN_DIRECTORY"
```

The output directory must be new. Failed execution retains partial artifacts
and a failure receipt. `report.json` binds every completed artifact, the exact
matrix bytes, simulator versions and both imported Workbench source files.
The example processes every input once; it adds no runtime, job or model-token
budget and never retries simulation until it obtains a success.

## Independent verification

Run verification from a trusted source snapshot, outside the candidate's
editable files and execution sandbox. Supply the expected source hashes from
the operator's submitted snapshot, rather than trusting hashes read from the
candidate's own report. The JSON format is:

```json
{
  "robot_sim.py": "operator-computed SHA-256",
  "robot_artifacts.py": "operator-computed SHA-256"
}
```

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  npa/.venv/bin/python npa/examples/specialists/robot_workflow/verify.py \
  --input "$COMPLETED_RUN_DIRECTORY" \
  --matrix "$FROZEN_SCENE_MATRIX" \
  --expected-source "$SUBMITTED_SOURCE_HASHES" \
  --native-python "$NATIVE_LEROBOT_PYTHON" \
  --output "$NEW_VERIFICATION_DIRECTORY"
```

The verifier reconstructs the scene with upstream Gymnasium-Robotics and MuJoCo.
It independently checks every control action, pre/post state, contact,
object/goal observation and both raw camera frames. It independently re-encodes
verified raw frames using the fixed H.264 settings, then checks every decoded
preview/dataset frame and its exact 25 Hz timestamp. This catches shifted or
duplicated frames that whole-image similarity can miss. Native LeRobot
verification checks all exported state/action/image
samples, timestamps, task labels and episode indices, decodes dataset videos
and forms a real training batch. Neither verifier imports candidate simulation
or export helpers. Exact raw-pixel replay requires matching simulator, renderer
and platform dependencies; differing runtimes must not receive an automatic
tolerance increase.

## Comparing agents

Use the [shared playbook](PLAYBOOK.md) and the
[coordinator runner](../workflows/README.md). Both arms need identical task text,
matrices, source grants, operation grants, verifier versions and fallback rules
appropriate to their architecture. Start from disjoint source and output
directories. A task is complete only after final-source simulation/export and
independent verification succeed.

If testing repairs, introduce explicitly recorded fault patches only in
disposable candidate copies. Never modify production source to create a
benchmark. Keep the correct reference, fault transformations and held-out
verification fixtures outside model-readable grants. Calibrate that the
faulted source fails and the unchanged reference passes before freezing either
measured arm. Record all attempts and operator interventions.

Tool grants are not an OS sandbox. A trusted operation adapter must execute
candidate source without network, host credentials, writable dependencies or
access to the grader. Bind immutable source and scenes read-only, grant only a
fresh output directory, and verify output after the process exits. Do not pass
arbitrary agent-authored shell commands through the adapter.

Account for coordinator, specialist, router, fallback and rejected model
responses, including missing usage. Simulation, native verification, setup,
host allocation and human/development effort are separate quantities. A
successful local workflow does not establish cloud deployment or savings. No
comparative result is implied by the example alone. The separate
[measured repair report](../../../../docs/workbench/specialists-robot-workflow-experiment.md)
retains three matched pairs, their source/artifact audits, model-cost estimates
and observed provider failures.
The subsequent [model-selection report](../../../../docs/workbench/specialists-model-selection-experiment.md)
retains cheaper-model screening and a separate direct-dispatch experiment on
the same task. Its endpoint recipe changes model selection and coordination;
the native verification contract remains the same.
