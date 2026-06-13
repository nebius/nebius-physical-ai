# Sim2Real — Customer asset handoff (Sereact)

Per-category plan for initial stock setup; custom UR/Flexiv OBJ uploads later.

| Category | Plan | Formats |
| --- | --- | --- |
| **1. Robot / embodiment** | **Stock** (Franka Panda built-in) | Later: custom URDF/MJCF + meshes, or USD |
| **2. Manipulated objects** | **Stock** for now | Later: OBJ/STL/GLB/PLY/USD per object + dimensions/mass |
| **3. Scene / environment** | **Stock** (table + simple bins) | Later: fixture meshes or SceneSpec JSON / USD layout |
| **4. Cameras / sensors** | **Stock** (workspace + wrist defaults) | Later: custom poses/intrinsics in SceneSpec |

Wire custom uploads via `ASSETS_URI`, `SCENE_SPEC_URI`, `ROBOT_PRESET` / `ROBOT_SPEC_URI` on workflow submit.
