"""Re-log the live run's own geometry through the current runner, for viewer capture."""
import sys, numpy as np, open3d as o3d
from npa.workbench.open3d.runner import run_visualize
fused = "/a/fused.ply"; mesh = "/a/mesh.ply"
out = sys.argv[1]
import json, pathlib
pathlib.Path(out).mkdir(parents=True, exist_ok=True)
run_visualize({"pose_graph": {"nodes": [{"fragment_id": "f0", "pose": np.eye(4).tolist()}], "fused_sha256": "0"*64},
               "fragments": {"f0": fused}, "fused_sha256": "0"*64,
               "fused_path": fused, "mesh_path": mesh,
               "voxel_size": 0.02}, pathlib.Path(out), "aspectfix")
print("wrote", out)
