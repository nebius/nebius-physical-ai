#!/usr/bin/env python3
"""Genesis serverless rendering diagnostic v3.

This probe targets the actual pick-place camera path used by
``generate_demos()``. A synthetic empty-scene camera can pass while the real
demo camera path fails, so the verdict is gated on ``env.get_camera_obs()`` in
``FrankaPickPlaceEnv`` with cameras enabled.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


results: dict[str, Any] = {}


def section(name: str) -> None:
    print(f"\n===== {name} =====", flush=True)


def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, check=False, text=True)


section("ENVIRONMENT")
id_result = run_capture(["id"])
whoami_result = run_capture(["whoami"])
print(id_result.stdout.strip() or id_result.stderr.strip(), flush=True)
print(whoami_result.stdout.strip() or whoami_result.stderr.strip(), flush=True)
results["uid"] = os.getuid()
results["gid"] = os.getgid()
results["groups"] = os.getgroups()
results["user"] = os.environ.get("USER", "<unset>")
results["home"] = os.environ.get("HOME", "<unset>")
print(f"groups: {results['groups']}", flush=True)

section("/etc/group")
try:
    with open("/etc/group", encoding="utf-8") as handle:
        group_lines = [
            line.strip()
            for line in handle.readlines()
            if line.strip() and not line.startswith("#")
        ]
    results["etc_group_count"] = len(group_lines)
    for target_gid in [44, 992, 993, results["gid"]]:
        for line in group_lines:
            parts = line.split(":")
            if len(parts) >= 3 and parts[2] == str(target_gid):
                members = parts[3] if len(parts) > 3 else ""
                results[f"gid_{target_gid}_name"] = parts[0]
                results[f"gid_{target_gid}_members"] = members
                print(f"GID {target_gid} = {parts[0]!r}, members: {members!r}", flush=True)
                break
        else:
            results[f"gid_{target_gid}_name"] = None
            results[f"gid_{target_gid}_members"] = ""
            print(f"GID {target_gid}: not found in /etc/group", flush=True)
    results["etc_group_full"] = group_lines
except Exception as exc:
    results["etc_group_error"] = str(exc)
    print(f"/etc/group read FAILED: {exc}", flush=True)

section("/dev/dri")
dri_listing = run_capture(["ls", "-la", "/dev/dri"])
print((dri_listing.stdout or dri_listing.stderr).strip(), flush=True)
results["dri_devices"] = {}
for dev in ["/dev/dri/card0", "/dev/dri/card1", "/dev/dri/renderD128", "/dev/dri/renderD129"]:
    if os.path.exists(dev):
        st = os.stat(dev)
        info: dict[str, Any] = {
            "exists": True,
            "uid": st.st_uid,
            "gid": st.st_gid,
            "mode_perms": oct(stat.S_IMODE(st.st_mode)),
        }
        try:
            fd = os.open(dev, os.O_RDWR)
            os.close(fd)
            info["rw_access"] = True
        except PermissionError as exc:
            info["rw_access"] = False
            info["rw_error"] = str(exc)
        except OSError as exc:
            info["rw_access"] = False
            info["rw_error"] = f"{type(exc).__name__}: {exc}"
        results["dri_devices"][dev] = info
        print(f"{dev}: {info}", flush=True)
    else:
        results["dri_devices"][dev] = {"exists": False}
        print(f"{dev}: missing", flush=True)

section("nvidia-smi")
nvidia_smi = run_capture(["nvidia-smi", "-L"])
results["nvidia_smi_returncode"] = nvidia_smi.returncode
results["nvidia_smi_output"] = (
    nvidia_smi.stdout[:1000] if nvidia_smi.returncode == 0 else nvidia_smi.stderr[:1000]
)
print(results["nvidia_smi_output"], flush=True)

section("EGL init")
try:
    from OpenGL import EGL

    display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    results["egl_display"] = bool(display) and display != EGL.EGL_NO_DISPLAY
    major, minor = EGL.EGLint(), EGL.EGLint()
    init_result = EGL.eglInitialize(display, major, minor)
    results["egl_initialize"] = bool(init_result)
    results["egl_version"] = f"{major.value}.{minor.value}" if init_result else None
    if not init_result:
        results["egl_error_code"] = int(EGL.eglGetError())
    print(f"eglInitialize: {init_result}", flush=True)
except Exception as exc:
    results["egl_initialize"] = False
    results["egl_error"] = str(exc)
    print(f"EGL init FAILED: {exc}", flush=True)
    traceback.print_exc()

section("vendored genesis.ext.pyrender")
try:
    import genesis.ext.pyrender as vendored_pyrender

    results["vendored_pyrender_import"] = True
    results["vendored_pyrender_path"] = str(Path(vendored_pyrender.__file__))
    print(
        f"genesis.ext.pyrender imported, path: {vendored_pyrender.__file__}",
        flush=True,
    )
except Exception as exc:
    results["vendored_pyrender_import"] = False
    results["vendored_pyrender_error"] = str(exc)
    print(f"vendored pyrender import FAILED: {exc}", flush=True)
    traceback.print_exc()

section("Real demo path probe")
try:
    import numpy as np
    import torch

    from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv

    torch.manual_seed(42)
    np.random.seed(42)
    env_cfg = EnvConfig(
        n_envs=4,
        enable_cameras=True,
        domain_randomize=True,
        camera_fps=20,
        action_space="joint",
    )
    env = FrankaPickPlaceEnv(env_cfg)
    env.reset()
    cam_obs = env.get_camera_obs()
    workspace = cam_obs["workspace"]
    wrist = cam_obs["wrist"]
    results["real_demo_probe"] = True
    results["real_demo_workspace_shape"] = (
        list(workspace.shape) if hasattr(workspace, "shape") else "no-shape"
    )
    results["real_demo_wrist_shape"] = (
        list(wrist.shape) if hasattr(wrist, "shape") else "no-shape"
    )
    print(
        "Real demo probe OK: "
        f"workspace={results['real_demo_workspace_shape']}, "
        f"wrist={results['real_demo_wrist_shape']}",
        flush=True,
    )
except Exception as exc:
    results["real_demo_probe"] = False
    results["real_demo_error"] = str(exc)
    results["real_demo_traceback"] = traceback.format_exc()
    print(f"Real demo probe FAILED: {exc}", flush=True)
    traceback.print_exc()

section("VERDICT")
dri_devices = results.get("dri_devices", {})
assert isinstance(dri_devices, dict)
dri_devs_present = any(
    isinstance(device, dict) and bool(device.get("exists"))
    for device in dri_devices.values()
)
dri_rw_ok = (
    all(
        bool(device.get("rw_access"))
        for device in dri_devices.values()
        if isinstance(device, dict) and bool(device.get("exists"))
    )
    if dri_devs_present
    else False
)
egl_ok = bool(results.get("egl_initialize", False))
vendored_pyrender_ok = bool(results.get("vendored_pyrender_import", False))
real_demo_ok = bool(results.get("real_demo_probe", False))

results["probe_dri_rw_ok"] = dri_rw_ok
results["probe_egl_init_ok"] = egl_ok
results["probe_vendored_pyrender_ok"] = vendored_pyrender_ok
results["probe_real_demo_ok"] = real_demo_ok

if not dri_rw_ok:
    verdict = "PERMS_BLOCKER"
elif not egl_ok:
    verdict = "EGL_INIT_FAIL"
elif not vendored_pyrender_ok:
    verdict = "VENDORED_PYRENDER_FAIL"
elif real_demo_ok:
    verdict = "ALL_OK"
else:
    verdict = "GENESIS_DEMO_RENDER_FAIL"

results["verdict"] = verdict
print(f"VERDICT: {verdict}", flush=True)
print(
    "sub-probes: "
    f"dri_rw={dri_rw_ok}, "
    f"egl={egl_ok}, "
    f"pyrender={vendored_pyrender_ok}, "
    f"real_demo={real_demo_ok}",
    flush=True,
)
print(f"FULL_RESULTS_JSON: {json.dumps(results, default=str)}", flush=True)

out_file = "/tmp/diagnostic-results.json"
with open(out_file, "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2, default=str)

sys.exit(0 if verdict == "ALL_OK" else 1)
