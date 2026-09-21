# cuRobo V2: real B200 controls

**The rebuilt image passed these preliminary controls. Full benchmark and image-release acceptance remain pending.**

![Actual B200 trajectory and joint traces](trajectory.png)

[Open the full-size plot](trajectory.png) · [Unchanged two-case journal](problems.jsonl) · [Numeric report](report.json) · [SHA-256 manifest](SHA256SUMS)

| Control or measurement | Observed result |
| --- | --- |
| Valid planning problem | Success; 61 retained trajectory samples |
| Goal-blocked problem | Failed as expected; no successful trajectory |
| Malformed robot input | Rejected; output directory absent |
| Independent forward-kinematics replay difference | 0 m |
| Planner-reported endpoint position error | 1.3719009928e-7 m |
| Endpoint distance recomputed from published positions | 1.4277366110e-7 m |
| Planner-reported orientation error | 1.6226947253e-7 rad |
| Planned trajectory duration | 1.5000000224 s |
| Tool path length recomputed from all samples | 0.6550320881 m |
| Durable artifact readback | 25 objects, 969,621 bytes; every hash matched |
| Original Rerun recording | Verified; decoded data matched; 61 samples |

The plot uses every recorded tool-position sample and all seven arm-joint traces. The unchanged journal also includes both finger-joint columns, velocities, acceleration, jerk, and orientation. Planned time is derived from the recorded timestep; it is separate from GPU execution time. The two endpoint position figures describe different measurements: the planner's metric and an independent calculation from retained coordinates.

The image was produced from source `b387643682271d867c816fba1a598295ffe57f37`. PR [#625](https://github.com/nebius/nebius-physical-ai/pull/625) head `1479d3716311cc0543028973acd00ff534bc5691` adds only formatting in its host image verifier and three test files; runtime/build inputs are unchanged. The immutable image and configuration digests are in the report.

The managed B200 job completed successfully. An independent consumer downloaded the stored outputs; root checked all 25 retained files against that manifest, confirmed the public journal is byte-for-byte identical, and recomputed the plotted endpoint distance and path length. Root did not make another object-store request.

These are controlled positive and negative cases, not a benchmark success-rate estimate. Kinematics replay does not independently establish collision freedom or suitability for robot hardware. No dynamics benchmark or VLM quality gate is claimed here. The original RRD and operational metadata stay private.

Remaining gates include the complete image-byte scan, RTX controls, full benchmark matrices, broad tests, and current-commit CI. Earlier failed workload evidence remains retained separately.
