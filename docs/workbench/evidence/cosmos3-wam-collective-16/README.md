# Sixteen-B200 collective correctness and transport proof

Two reserved eight-B200 nodes completed a native Slurm allocation with sixteen
distinct GPU ranks. Every element of every tested FP32 all-reduce equaled
136, the sum of rank inputs 1 through 16. The job completed with exit `0:0`
in 34 allocated seconds. All sixteen ranks reported `Using network IB`;
none reported `Using network Socket`. The retained logs also contain
GPUDirect RDMA data-channel records.

This is a custom PyTorch diagnostic. It measures synchronized host duration,
including Python dispatch, and is not an execution of the `nccl-tests` binary.
It establishes a working collective path, not training scaling or NIC line rate.

| Buffer per rank | Mean operation time | Algorithm bandwidth | Normalized bus bandwidth |
| --- | --- | --- | --- |
| 1 MiB | 0.4078 ms | 2.57 GB/s | 4.82 GB/s |
| 16 MiB | 0.4955 ms | 33.86 GB/s | 63.48 GB/s |
| 256 MiB | 1.2893 ms | 208.21 GB/s | 390.39 GB/s |
| 1 GiB | 3.3450 ms | 321.00 GB/s | 601.87 GB/s |

Each size has five warmup operations and twenty measured operations. Each
recorded duration is the maximum across ranks for that operation; the table
uses the mean of those twenty maxima. Correctness checks and the preceding
barrier are outside the timed interval. Bandwidth uses decimal GB/s.
The normalized value multiplies bytes/time by `2 × (16 − 1) / 16`, following
the [NVIDIA all-reduce convention](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/doc/PERFORMANCE.md).
This topology combines intra-node NVLink with inter-node communication, so
the normalized number must not be interpreted as one network link's speed.

## Evidence and reproduction

`bandwidth.json` is the unmodified runtime output, including every duration,
runtime version and the executed Python source hash. `transport-proof.json`
records the independently checked NCCL transport markers, raw log hashes,
and allocation result. Generic stream roles replace private log filenames.
Raw logs remain private because they contain host
and network identifiers. `collective_bandwidth.py` is the exact executed
Python source; its hash must match `bandwidth.json`.

Stage that script in shared storage and set `WAM_COLLECTIVE_SCRIPT` to its
absolute path. Set `WAM_SHARED_ROOT` to the prepared recipe root and
`WAM_COLLECTIVE_OUTPUT` to a new absolute JSON output path. On the configured
native Slurm controller, submit this directory's parameterized shell launcher:

```bash
export WAM_SHARED_ROOT WAM_COLLECTIVE_SCRIPT WAM_COLLECTIVE_OUTPUT
sbatch --output="$WAM_COLLECTIVE_LOG" --error="$WAM_COLLECTIVE_ERROR" \
  collective.sbatch
```

Choose private log paths in `WAM_COLLECTIVE_LOG` and `WAM_COLLECTIVE_ERROR`
before submission. The shell launcher exposes the same launch settings with
operator-selected paths. Retain its exact submitted bytes and Slurm accounting
with each new result. Inspect the actual NCCL logs for sixteen IB selections
and data-channel transport; a successful numeric result alone does not prove
which transport ran. Run this diagnostic separately from timed training.
