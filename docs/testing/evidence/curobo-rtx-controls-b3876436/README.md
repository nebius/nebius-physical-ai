# cuRobo V2: real RTX PRO 6000 controls

**The rebuilt image passed these preliminary application controls after a recorded bootstrap repair. Full benchmark and image-release acceptance remain pending.**

[Unchanged RTX two-case journal](problems.jsonl) · [Numeric report](report.json) · [SHA-256 manifest](SHA256SUMS)

![Matching trajectory from the independent B200 run](../curobo-b200-controls-b3876436/trajectory.png)

The plot above was made from the **B200** run. Root independently compared the RTX and B200 journals: both problem identities, outcomes, and every retained trajectory array are exactly equal. The RTX journal is linked separately; the complete journals differ in runtime measurements.

| Control or measurement | RTX result |
| --- | --- |
| Hardware | RTX PRO 6000; CUDA compute capability 12.0 |
| Valid planning problem | Success; 61 retained samples |
| Goal-blocked problem | Failed as expected; no successful trajectory |
| Malformed robot input | Rejected; output directory absent |
| Independent forward-kinematics replay difference | 0 m |
| Planner-reported position error | 1.3719009928e-7 m |
| Endpoint distance recomputed from published coordinates | 1.4277366110e-7 m |
| Planner-reported orientation error | 1.6226947253e-7 rad |
| Durable artifact readback | 25 objects, 969,692 bytes; every hash matched |
| Original Rerun recording | Verified; decoded data matched; 61 samples |

SkyPilot's package setup was interrupted before the application started, leaving `patch` absent. The existing isolated pod was repaired by completing pending package configuration and installing `patch`; the same job then completed successfully. The image digest stayed unchanged, and all nine installed cuRobo Python files matched the image-producing source. Pre-repair package/runtime logs and a post-repair package manifest are retained privately; no complete pre-repair package manifest was captured. This proves the application after the disclosed repair, not an unmodified bootstrap.

The image source is `b387643682271d867c816fba1a598295ffe57f37`. PR [#625](https://github.com/nebius/nebius-physical-ai/pull/625) head `1479d3716311cc0543028973acd00ff534bc5691` changes only formatting in the host verifier and three test files; runtime and build inputs are unchanged. Immutable image digests and receipt hashes are in the report.

Root checked all 25 downloaded originals against the retained durable-readback manifest and independently recalculated the endpoint distance and path length from this unchanged journal. Root did not issue an additional object-store request. Original RRD files and operational records remain private.

These positive and negative controls do not estimate benchmark success rates, certify collision freedom, or establish robot-hardware safety. No inverse-dynamics benchmark or VLM quality acceptance is claimed. Both full benchmark matrices, complete image-byte qualification, broad tests, and current-commit CI remain separate gates. Prior failed attempts remain retained.
