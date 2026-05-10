# Rerun Adapters

This package contains standalone Rerun `.rrd` exporters for Unitree G1
LeRobotDataset artifacts and GR00T prediction artifacts.

The adapters are intentionally separate from `npa viz lerobot --backend rerun`.
The `viz` backend renders MP4s for quick review; these adapters create reusable
Rerun recordings for customer artifacts, handoff, and post-demo inspection.

## LeRobotDataset to Rerun

```python
from pathlib import Path

from npa.viz.adapters import lerobot_to_rerun

lerobot_to_rerun(
    dataset_path="s3://bucket/path/to/LeRobotDataset/",
    output_rrd_path=Path("/tmp/isaac-lab-trajectory.rrd"),
    duration_s=5.0,
)
```

The recording logs:

- `world/skeleton/joints` as cyan `Points3D`
- `world/skeleton/bones` as cyan `LineStrips3D`
- `world/skeleton/angles/<joint_name>` scalar time series

## GR00T Predictions Overlay

```python
from pathlib import Path

from npa.viz.adapters import groot_predictions_to_rerun

groot_predictions_to_rerun(
    predictions_path="s3://bucket/path/to/groot-predictions/",
    input_dataset_path="s3://bucket/path/to/LeRobotDataset/",
    output_rrd_path=Path("/tmp/groot-predictions-overlay.rrd"),
    duration_s=5.0,
)
```

The overlay recording uses one shared `frame_time` timeline:

- `world/skeleton/...` for input trajectory in cyan
- `world/predictions/...` for predicted trajectory in orange

Prediction inputs may be JSON, `.npy`, `.npz`, or a directory containing one of
those artifacts. REAL_G1 53D action vectors are mapped onto the canonical 43D G1
state layout before logging.

## Output Paths

Local outputs are written directly. For `s3://...` outputs, the adapter first
writes a local temporary `.rrd`, then uploads it to the requested object path.

Both adapters apply a default 5 second duration cap and evenly subsample longer
sources. Pass `duration_s` to request a shorter cap.

## Viewer

Open a recording locally with:

```bash
rerun /tmp/isaac-lab-trajectory.rrd
rerun /tmp/groot-predictions-overlay.rrd
```

The saved blueprint opens with a dark 3D spatial view for `world/**`, a time
series panel for `world/**/angles/**`, hidden Blueprint and Selection panels, and
the Time panel visible on the `frame_time` timeline.
