# Base-image CI disk ownership and isolation

The signed source `5f8123940b8e631a934c34ce22f113b4b5713338` passes the complete local suite: **25,534 passed, 155 skipped, 1 xpassed**, with zero failures and unchanged source. The actual trusted-base Bandit, zizmor and Trivy comparison reports **607 baseline / 607 candidate findings, zero new blocking findings**. Its real scanner regression controls pass. See [validation.json](validation.json) for the exact source/tree and retained receipt hashes.

The change assigns every validated base image its own ordinary CI runner and requires all applicable results through a fail-closed aggregate. Patched bases export a private archive from a uniquely named BuildKit builder; the builder and cache volume are removed before Trivy reads the archive. Each entry then reclaims its archive, mutable cache and temporary layer data. Unmodified images use their remote digest. No global pruning or vulnerability-policy relaxation is introduced.

Independent review reproduced a failed-Docker-execution cleanup bug, then verified the repair retains the ownership receipt, Buildx configuration and original exception. The focused suite has 68 passing controls, including matrix failure/cancellation handling and unrelated-state preservation.

[Eight actual base-image scans](native-eight-images.json) passed on the retained scanner prototype. Seven isolated builders/cache volumes were absent before their archive scans; unrelated sentinel state survived. The three-worker trial sampled a **54,829,871,104-byte combined owned peak**. The final source adds the matrix/entry CLI and reviewed failed-exec repair; the native run is not relabeled as a final-source all-eight rerun.

**Final-source hosted validation passed all eight image scans**, their required aggregate and the full CI gate: 28 successful checks and two expected skips. Every job used a separate GitHub Actions `ubuntu-latest` runner. The observed filesystem capacity was 154,894,188,544 bytes; the minimum available space was 92,115,558,400 bytes before a scan and 90,231,042,048 bytes afterward. These are before/after readings, not continuous peak measurements or a guarantee for smaller runners. See [hosted-ci.json](hosted-ci.json) for the complete measurements and original log hashes.

| Image inventory entry | Final-source hosted result |
| --- | --- |
| nvidia-cuda-12-4-1-devel-ubuntu22-04 | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/35631932553/job/106440072300) |
| nvidia-cuda-12-6-3-devel-ubuntu22-04 | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/35631932553/job/106440072255) |
| nvidia-cuda-12-8-1-cudnn-devel-ubuntu22-04 | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/35631932553/job/106440072343) |
| nvidia-cuda-12-8-1-cudnn-devel-ubuntu24-04 | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/35631932553/job/106440072497) |
| nvidia-cuda-13-0-2-cudnn-devel-ubuntu24-04 | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/35631932553/job/106440072295) |
| nvidia-cuda-13-0-2-base-ubuntu24-04 | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/35631932553/job/106440072459) |
| python-3-11-slim-trixie | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/35631932553/job/106440072378) |
| nvidia-cuda-13-0-1-cudnn-devel-ubuntu22-04 | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/35631932553/job/106440072501) |

These results establish scan completion on the recorded hosted runners. They do not mean zero vulnerabilities, a measured throughput improvement, or GPU/VLM/application-image qualification.
