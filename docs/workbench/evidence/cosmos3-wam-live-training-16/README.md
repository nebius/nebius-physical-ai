# Real sixteen-B200 WAM training execution

![Actual two-node GPU readings over a fixed two-minute window](training-window.png)

Two reserved eight-B200 nodes executed the native Cosmos 3 Nano WAM trainer
under one Slurm allocation. All sixteen GPU processes were independently
matched to the native training command, exact training configuration, Slurm
cgroup, local rank and global rank. The distributed preflight passed on
sixteen distinct devices across two hosts, with NCCL all-reduce sum 136.

This folder is live execution evidence from an incomplete 2,000-update run.
It does not establish completed duration, throughput scaling or policy quality.
The measured timing repetitions and full checkpoint evaluations answer those
questions separately.

## Why both snapshots and a time window are included

The first process-attributed snapshots were taken near updates 59 and 63,
with independent timestamps. They happened to report 0% utilization on
fifteen devices and 24% on one. Those original readings are retained in
`node-0.json`, `node-1.json` and the [snapshot figure](training-snapshot.png).
A single instantaneous observation does not describe the execution cycle.

After those snapshots, we chose a fixed **prospective 120-second window**,
recorded its endpoints at the start, and collected every available
device sample within it. The window was not selected for high utilization.
It ran from 02:32:37.898 through 02:34:37.898 UTC on September 28, 2026.

The window contains 1,920 samples: 120 per GPU. Every device has observations
at both 0% and 100%; arithmetic sample means range from 56.41% to 62.39%.
The heatmap shows the recurring low- and high-utilization periods on both
nodes. Each cell averages available readings in one second; missing bins
remain blank, with no interpolation. This is reported GPU utilization,
not model FLOP utilization or exposed communication overhead.

Sampled device memory ranges from 57.56 to 59.01 GiB. The maximum observed
interval between samples is 1.019 seconds. Memory is whole-device usage at
those sampling times, not an allocator peak or a guarantee about shorter
spikes. Each node's samples have their own timestamps; the physical GPU
indices are local to that node. Node labels here use actual Slurm node rank,
verified from each process's environment without publishing that environment.

## Reproduce the capture and rendering

`capture.py` verifies attribution on the host where it runs, then writes a
new receipt using `--run-dir`, `--job-id` and `--output-path`. Run it on both
workers after a completed timed update. It reads process identities privately
and exports only the rank assignments, numeric GPU readings and source hashes.

`live-window.json` preserves the preselected observation window. Each
`window-node-*/` folder contains the numeric samples and the actual exporter
receipt. `export_window.py` selects raw recorder rows by those endpoints,
requires all eight local GPUs and checks timestamp order and boundary coverage:

```bash
"$WAM_SHARED_ROOT/framework/.venv/bin/python" "$WAM_WINDOW_EXPORTER" \
  --window "$WAM_OBSERVATION_WINDOW" --node-rank "$WAM_NODE_RANK" \
  --telemetry "$WAM_NODE_TELEMETRY" --output-dir "$WAM_NODE_EXPORT"
```

Set the variables to this exporter, the recorded window, actual Slurm node
rank, that host's readable private telemetry file and a fresh output directory.
Ensure the recorder file is readable by the operator; it need not be public.
Keep host identifiers, raw command lines and provider details private.

With Matplotlib 3.11.2 and NumPy 2.5.3, regenerate the figures from this checkout:

```bash
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-live-training-16/plot.py
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-live-training-16/plot_window.py
```

The renderers verify the snapshot, window and sample hashes before drawing.
Both figures were inspected and reproduced with identical image hashes.
The manifests record the renderer versions and output hashes. Raw telemetry
remains available in private campaign evidence; selected original row hashes
are preserved alongside the sanitized CSVs.
