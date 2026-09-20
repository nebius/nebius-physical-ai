# Recompute the synthetic evo proof

This companion to PR #584 exposes all **690 unrounded error values** from six
real native evo result archives: APE and RPE for exact, bounded, and drifting
synthetic controls. Download `synthetic-errors.json` and run the snippet below
from its directory with Python 3. Only the standard library is needed; no VM,
GPU, customer credentials, NumPy installation, or inference call is required.

The vectors were extracted from the retained `error_array.npy` members, with
pickle loading disabled. Each vector retains its original ZIP SHA-256 and NPY
member SHA-256. The exporter verified all six archive hashes against the frozen
live-run manifest and checked that the exported JSON numbers round-trip to the
original float64 bytes exactly. Original ZIPs and their internal path metadata
are not republished. Their hashes identify retained provenance; recomputation
from this package verifies the exported error vectors, not unseen ZIP bytes.

```python
import json
import math

with open("synthetic-errors.json", encoding="utf-8") as handle:
    proof = json.load(handle)

assert proof["frozen_gate"] == {
    "ape_rmse_m_max": 0.05,
    "rpe_rmse_m_max": 0.02,
    "minimum_errors_per_metric": 100,
}
for name, expected in (("exact", True), ("bounded", True), ("drifting", False)):
    control = proof["controls"][name]
    rms = {}
    counts = {}
    for metric, expected_count in (("ape", 120), ("rpe", 110)):
        result = control["metrics"][metric]
        errors = result["errors_m"]
        assert len(errors) == result["error_count"] == expected_count
        assert all(math.isfinite(value) and value >= 0 for value in errors)
        rms[metric] = math.sqrt(math.fsum(value * value for value in errors) / len(errors))
        counts[metric] = len(errors)
        assert math.isclose(rms[metric], result["reported_rmse_m"], rel_tol=1e-12, abs_tol=1e-30)
    passed = rms["ape"] <= 0.05 and rms["rpe"] <= 0.02 and min(counts.values()) >= 100
    assert passed == expected == control["expected_gate_pass"]
    print(name, "APE", format(rms["ape"], ".12g"), "RPE", format(rms["rpe"], ".12g"), "pass", passed)
```

Expected output, with errors measured in metres:

```text
exact APE 7.17503714778e-16 RPE 7.45474572329e-16 pass True
bounded APE 0.00354543382727 RPE 0.00594981427373 pass True
drifting APE 0.4438116059 RPE 0.142876706784 pass False
```

These are translation errors after SE(3) Umeyama alignment. RPE uses a
10-frame delta and all eligible pairs. The frozen gate requires APE RMSE at
most 0.05 m, RPE RMSE at most 0.02 m, and at least 100 errors in each metric.
This export reproduces two accepted and one rejected vector-bearing controls.
The malformed-input control deliberately produced no result archive and is
documented separately in the execution proof; this export does not reproduce
that process exit or claim a four-case confusion matrix from three cases.

All input paths were generated independently of sensor data. For index
`i = 0, ..., 119`, let `p = i / 119`: `x = 5*cos(2*pi*p)`,
`y = 5*sin(2*pi*p)`, and `z = 0.5*sin(4*pi*p)`. Exact replay uses the same
trajectory; bounded adds `0.005*sin(0.2*i)` to y; drifting adds `1.5*p*p` to y.
The reference and all three control input files were independently regenerated
and matched the retained live inputs byte-for-byte. Their hashes are included.
No KITTI data, customer trajectories, private paths, or infrastructure
identifiers are in this export.

The real workload ran on CPU Kubernetes through a direct SkyPilot BYOF launch
using upstream [evo v1.35.1](https://github.com/MichaelGrupp/evo/tree/8dd6cfe0ec1747f9e1b5b569edd82c54d1a3f422).
The producing NPA commit is
`2f872775be4b7f116c0b0b8195c45ef2c4dbcca0`; the workflow is byte-identical at
qualified candidate `c0e53c67f99075f8fc75fd24997ca4b6ce849353`. Exact workflow,
command, image-index/platform-manifest, plan, and frozen-evidence hashes are in
the JSON. This does not claim execution at a later PR head or through an outer
`npa.workflow` launch.

Recomputing RMS establishes consistency of the released vectors and gate
decisions. It does not rerun evo, independently recreate the container workload,
or establish navigation performance, sensor accuracy, robot safety, or real-world
generalization. The container image and raw private operational bundle are not
included.
