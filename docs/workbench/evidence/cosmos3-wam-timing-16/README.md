# Three sixteen-B200 timing repetitions

Three separate unprofiled two-node runs completed 200 updates each. Mean
optimizer-step times were **6.8740, 6.8942 and 6.8817 s**; their mean was
**6.8833 s**, with across-run sample standard deviation **0.0102 s**.
Each retained 149 native timing records (updates 52–200); all 447 records and
actual token counters remain in the three `repeat-*` directories.

The 50-update timing exclusion differs from the 500-update learning-rate
warmup. These windows measure stable iteration duration during learning-rate
warmup. The final checkpoint save is outside the timed loop but inside the
native process durations: 1,959.009, 1,961.936 and 1,964.940 seconds. Slurm
allocations were 2,063, 2,066 and 2,081 seconds, all `COMPLETED`, `0:0`.

All runs start from the same base checkpoint and seed 42, with nominal global
batch 2,048, cap 64, accumulation two, and fixed source/data/model pins. Caches
were warm. Neither evaluation nor archival overlapped any repetition.
Pooled throughput was **132,745.58 tokens/s**. The standard deviation describes
three run means, not 447 independent trials or quality variation across seeds.

[evidence.json](evidence.json) hashes original reports, CSVs, complete final
checkpoint manifests, both node completion receipts and Slurm accounting.
Every checkpoint component matched its full-GET-verified archive before local
temporary payload pruning. Original report and CSV bytes are unchanged.

## Reproduce

Follow the [recipe](../../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md)
with three fresh `--nodes 2 --steps 200` plans, without `--profile`. Run
`report.py` after each completed run, retain all outputs, and use
`scaling_report.py` together with three corresponding eight-GPU reports.
The [complete comparison](../cosmos3-wam-scaling/README.md) provides the exact
command using the committed numeric records and both regenerated figures.
