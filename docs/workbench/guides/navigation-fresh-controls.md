# Fresh native scenes for reference controls

The built-in `static_raycast` reference runs each physical control in a separate
native process before training or evaluation. The stage parent starts solo,
repeat, coincident-peer and obstacle controls sequentially, then starts a fifth
process for the actual workload. Each process constructs the full robot
population in one shared scene. The repeat arm repeats the solo setup in a new
scene; it does not test a warm reset of the preceding scene.

This changes the control protocol, not the reset implementation. Native contact
history can survive articulation teleports, and removing actors is not equivalent
to creating a new physics scene. See NVIDIA's
[simulation limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/guides/current_limitations.html)
and [determinism boundary](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/API.html#determinism).
Fresh processes give the comparison arms equivalent construction history; they
do not guarantee identical GPU trajectories. The original numerical tolerance
and the same-placement repeat control still apply. This protocol requires its
own native qualification. Historical warm-reset results remain evidence only
for their original source and protocol.

Every control uses the original stage mode, including its environment seed,
runner seed, checkpoint initialization, renderer startup settings, source,
image binding, collision scene, controller, population and action sequence.
Only the prescribed focal and peer reset cases differ. Training controls load
the same initial PPO state but perform no learning; evaluation controls use the
same checkpoint-loading and inference initialization as the final evaluator.
The raw initial policy-state digest must agree across all five processes.

The parent recomputes the original free-motion, collision-free baseline,
same-placement, peer-isolation and obstacle-positive gates from all four traces.
It binds them to a fresh attempt identifier, exact input bytes, native provenance
and child exit records. Each native process records its actual process group;
the control receipt must match the new session owned by its parent launcher. Before learning or evaluation, the final native process
independently verifies the request, complete file hashes, native logs, raw traces,
physical gates and its own current initialization. There is no public skip flag
or reusable qualification cache. An incomplete child, native PhysX error, changed
trace, different mode or receipt copied to another attempt prevents execution.

Artifacts are retained within the normal stage publication:

| Path | Evidence |
| --- | --- |
| `controls/request.json` | Fresh attempt, original mode and complete input/source bindings |
| `controls/<arm>/native-start.json` | Actual native PID, mode, scene/controller inventory and local clock origin |
| `controls/<arm>/probe-<arm>.npz` | Full-population states and focal observation/native arrays; completed snapshots survive failure |
| `controls/<arm>/runtime.log` and `process.json` | Child output, exit code, log digest and parent-observed lifetime |
| `controls/<arm>/control.json` | Completed native control and its artifact hashes |
| `controls/accepted.json` | Parent's recomputed four-arm acceptance |
| `isolation-comparisons.json` | Exact A/A and A/B stream differences, retained before gate rejection |
| `native-process.json` | Final process's independent acceptance and current native provenance |
| `runtime.log` and `process.json` | Final training/evaluation process output and exit record |

Native clocks are local to each process. Receipts retain separate origins and
measured step intervals; elapsed time must match the observed step count times
the actual timestep within floating-point representation error. Absolute times
are not compared between processes.
Rendered evaluation still binds each frame to the final process's actual native
and Fabric clocks. Failure publications retain available logs, partial traces
and contact/reset diagnostics, with no successful qualification claim. An
interruption terminates the owned child process group and waits for the launcher;
a second interruption escalates that same group to a kill signal. The original
interruption is then re-raised, with partial stage evidence published when storage
remains available. No workload timeout is imposed.

This path currently supports the reference's static range observations. It does
not qualify camera-conditioned navigation or 4000 simultaneous RGB-D policies.
External BYOF adapters retain their existing in-process control integration.
