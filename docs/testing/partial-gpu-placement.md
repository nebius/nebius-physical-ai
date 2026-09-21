# Live validation of partial-node GPU requests

Catalog and inventory tests do not establish physical GPU placement or collective
performance. Qualify each hardware topology with an actual managed job before
claiming that a partial allocation preserves its interconnect behavior.

## Procedure

Use an existing, operator-approved eight-GPU node with no other GPU workload.
Keep the selected project, context, node selector, registry authentication and
evidence outside Git. Use an isolated NPA checkout and SkyPilot state; do not
replace the production launcher or reuse its managed-job identity.

1. Run the credential health preflight and targeted `npa skypilot verify`. Import
   the existing cluster with `npa cluster kubeconfig` if using a new NPA config
   directory; specify its exact provider name, project and local context. Copying
   config files alone leaves absolute kubeconfig paths pointing at the old state.
2. Read `sky show-gpus` through the selected pinned SkyPilot runtime. Feed that
   live output to `parse_kubernetes_gpu_catalog` and
   `resolve_kubernetes_accelerator`. A six-GPU request must resolve even when the
   display lists `1, 2, 4, 8`. Preserve that catalog evidence.
3. Submit sequential eight- and six-GPU jobs through `submit_workflow` or
   `npa workbench workflow submit`, using the same digest-pinned PyTorch image and
   the same CPU, memory and shared-memory resources. Use one node per job and a
   task-level `kubernetes.pod_config.spec.nodeSelector` to select the same node.
   Raw SkyPilot YAML must use the cluster's exact advertised accelerator name.
   Do not use raw `sky jobs launch` or bypass execution preflight.
4. Put [gpu_collective_probe.py](../../npa/tests/e2e/gpu_collective_probe.py) in
   each task and execute it with the image's PyTorch interpreter:

   ```bash
   export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET
   "$TRAINING_PYTHON" -m torch.distributed.run \
     --standalone --nproc_per_node="$GPU_COUNT" gpu_collective_probe.py
   ```

   The probe checks the visible GPU count, records each rank's UUID and peer
   access, captures physical topology and software versions, and checks
   all-reduce correctness. For each of five FP32 message sizes (1 KiB through
   256 MiB), it measures three repeats of 100 collectives after 20 warm-ups.
   Timing uses the slowest rank and includes synchronization. Keep NCCL's
   transport selection unchanged; record any environment or system configuration
   overrides present in the image.
5. While each worker exists, capture its pod UID, assigned node, GPU requests and
   limits, image digest and container state. Require one pod requesting exactly
   six GPUs for the partial case, six distinct rank UUIDs from that node, all
   correctness checks passing, and the exact managed job reaching `SUCCEEDED`.
6. Compare medians, NCCL connection logs and GPU subsets. Repeat the full-node
   control after the partial allocation when investigating a difference.
   All-reduce normalized bus bandwidth is algorithm bandwidth multiplied by
   `2 * (N - 1) / N`; raw algorithm bandwidth alone is not a like-for-like
   comparison between six and eight ranks. This follows NVIDIA's
   [NCCL performance convention](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).
7. Retain logs and placement receipts privately before teardown. Verify each
   owned worker is gone and the node's GPUs are released. Remove only the
   isolated test controller and API process after its exact jobs are terminal.
   Preserve other controllers, workloads, datasets and checkpoints.

## What the result establishes

A successful six-GPU job establishes real partial-node placement and collective
correctness for the tested image and hardware. A matched throughput comparison
can detect a transport change or a performance loss in that configuration. It
does not establish training convergence or application throughput.

PCIe results do not qualify NVLink, NVSwitch or NVLS. For those claims, use a node
that actually has that fabric, retain its `nvidia-smi topo -m` evidence, and verify
the relevant NCCL transport remains active for the allocated subset. A passing
collective using shared memory must not be described as NVLink validation.

Publish only aggregate measurements, software versions and evidence hashes.
Raw topology, pod manifests, node names, UUIDs and private image references stay
in access-controlled evidence as required by the infrastructure confidentiality
policy.
