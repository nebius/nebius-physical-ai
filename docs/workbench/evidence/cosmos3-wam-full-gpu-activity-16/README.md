# Full sixteen-GPU training telemetry

**252,656 actual device samples** cover the union of both completed native
training-process intervals. Both hosts recorded their eight B200s at a nominal
one-second cadence on synchronized UTC clocks. The largest observed gap was
**7.9 seconds**. Sampled per-device memory maxima range from **57.67 to 59.05 GiB**.

![Complete sixteen-GPU activity](full-gpu-activity.png)

The figure shows all global ranks, with checkpoint completions aligned onto
the earliest native-process start. Slurm node rank comes from the actual
allocation: the controller role does not imply rank zero. One node's
utilization falls at saves while the other's remains high. This alone does
not identify useful work, synchronization or exposed communication overhead.
Utilization is a sampled whole-device metric, not model FLOP utilization;
memory maxima are not allocator or subsecond peaks. Means are arithmetic
sample means, not time-weighted integrals.

[evidence.json](evidence.json) and the two `node-*` directories record original
completion/settings hashes, selected raw-row hashes and compressed/uncompressed
CSV hashes. [export.py](export.py) verifies both native completions and GPU
identity before extracting the entire training interval. Raw frozen recorders
remain in private evidence. No samples were selected to hide checkpoint pauses.

## Reproduce

Run the recorder alongside the [native Slurm recipe](../../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md).
After both training processes finish, export each host's actual recorder with
its recorded node rank into a fresh directory:

```bash
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-full-gpu-activity-16/export.py   --run-dir "$WAM_FULL_RUN" --node-rank "$WAM_NODE_RANK"   --telemetry "$WAM_GPU_CSV" --output-dir "$WAM_NODE_EXPORT"
```

The exporter requires both completion files and original settings. To
regenerate the committed figure with Matplotlib 3.11.2 and NumPy 2.5.3:

```bash
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-full-gpu-activity-16/plot.py
```

It verifies both node exports and the sibling full-run checkpoint times.
The second rendering reproduced [render-manifest.json](render-manifest.json).
