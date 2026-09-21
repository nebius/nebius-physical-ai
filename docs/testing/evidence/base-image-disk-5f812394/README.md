# Base-image CI disk ownership and isolation

The signed source `5f8123940b8e631a934c34ce22f113b4b5713338` passes the complete local suite: **25,534 passed, 155 skipped, 1 xpassed**, with zero failures and unchanged source. The actual trusted-base Bandit, zizmor and Trivy comparison reports **607 baseline / 607 candidate findings, zero new blocking findings**. Its real scanner regression controls pass. See [validation.json](validation.json) for the exact source/tree and retained receipt hashes.

The change assigns every validated base image its own ordinary CI runner and requires all applicable results through a fail-closed aggregate. Patched bases export a private archive from a uniquely named BuildKit builder; the builder and cache volume are removed before Trivy reads the archive. Each entry then reclaims its archive, mutable cache and temporary layer data. Unmodified images use their remote digest. No global pruning or vulnerability-policy relaxation is introduced.

Independent review reproduced a failed-Docker-execution cleanup bug, then verified the repair retains the ownership receipt, Buildx configuration and original exception. The focused suite has 68 passing controls, including matrix failure/cancellation handling and unrelated-state preservation.

[Eight actual base-image scans](native-eight-images.json) passed on the retained scanner prototype. Seven isolated builders/cache volumes were absent before their archive scans; unrelated sentinel state survived. The three-worker trial sampled a **54,829,871,104-byte combined owned peak**. The final source adds the matrix/entry CLI and reviewed failed-exec repair; the native run is not relabeled as a final-source all-eight rerun.

**Hosted-runner fit remains unproven.** The new workflow records actual free space before and after each image; all eight hosted outcomes and the required aggregate remain acceptance gates. These observations establish neither a throughput improvement nor GPU/VLM/application-image qualification.
