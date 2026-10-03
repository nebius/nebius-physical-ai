# Repair and verify the prepared robot workflow

Read the task description and the permitted source files. Use scoped directory
listing and line-range reads, then make small exact replacements against the
latest full-file hash. The scene matrix, controller task, physics thresholds
and independent verifier are fixed inputs.

1. Run `diagnose` to observe the supplied defect and `validate` after a source edit.
2. Run `submit` to execute the current source through the real local Workbench
   simulation and dataset export. A returned attempt identifier is not success.
3. Use `status`, `wait` and `logs` to inspect the same attempt. Let native work
   finish independently; repeated submission must not duplicate uncertain work.
4. Run `verify` after terminal execution. It must bind the current source hash
   to the executed source and validate real artifacts independently.
5. If verification fails, inspect the recorded error, repair the permitted
   source, rerun the affected workflow and verify the new attempt. Retain failed
   attempts. Do not edit records to turn a failure into success.

Preserve every prepared case, including physical failures. Observation/state at
time `t` must precede action `t`; the trace also records resulting state `t+1`.
Both camera views must correspond to that same pre-action simulator state.
Only physics-accepted episodes belong in LeRobot, with contiguous dataset
indices and the correct instruction, frames, actions and timestamps. Rejected
cases retain their original recordings and receive no dataset episode index.

For delegation, give the worker the concrete outcome and permitted workspace.
Let the configured router choose among models within that workspace's existing
authority. Wait for durable completion or attention, then inspect receipts.
The coordinator may take over a resolved escalation and must preserve the
original worker outcome. Do not call polling tools repeatedly when an
asynchronous wait is available.

Report the final patch, verifier outcome, failed attempts and remaining
limitations. Distinguish simulator execution, dataset validity and physical
task success. These scripted demonstrations are not a trained policy or proof
of transfer to a real robot.
