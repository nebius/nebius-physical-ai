# CUDA profiling on two B200 nodes

A separate 110-update sixteen-B200 profiling run completed successfully. Native
process time was 1,364.239 s; Slurm allocation was 1,458 s. This instrumented
run is excluded from the throughput comparison.

![Two actual CUDA rank timelines](cuda-timeline.png)

The profiler observed global ranks **0 and 8**, one per host, over native
`ProfilerStep#98` and `ProfilerStep#99`. Each window spans about **15.074 s**
and contains **198,042 kernels**, of which 197,956 link to CPU operators.
Observed CUDA kernel-interval unions are **8.005 s and 7.957 s**. Each trace
has 540 NCCL events; summed durations are 1.722 s and 3.214 s. These sums are
not exposed communication overhead: intervals overlap. Timeline rows merge
intervals within each category and must not be stacked into wall-time fractions.
The observed union excludes memcpy, CPU work and unobserved activity. Fused
kernels without stronger attribution remain `other`.

[evidence.json](evidence.json) hashes both original traces, the analyzer,
[profile-summary.json](profile-summary.json), exporter and numeric timelines.
The original traces remain in the full-GET-verified private profiling archive.
This is two sampled ranks, not sixteen-GPU aggregate utilization.

## Reproduce

Use a fresh `--nodes 2 --steps 110 --profile` plan from the
[recipe](../../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md), complete
it, then run `profile_report.py --run-dir "$WAM_PROFILE_RUN"`. Export real traces:

```bash
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-profile-16/export.py   --recipe-dir "$WAM_RECIPE" --run-dir "$WAM_PROFILE_RUN"   --output-dir "$WAM_PROFILE_EXPORT"
```

Choose a fresh output directory. `--summary-path` selects an existing verified
summary and `--bin-seconds` defaults to 0.25. To regenerate the committed figure
with Matplotlib 3.11.2 and NumPy 2.5.3:

```bash
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-profile-16/plot.py
```

The renderer checks numeric hashes; repeat rendering reproduces the image hash
in [render-manifest.json](render-manifest.json).
