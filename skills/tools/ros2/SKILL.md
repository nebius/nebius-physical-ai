---
name: ros2
description: Use when working on ROS 2 Jazzy prerequisite detection, the ROS 2 bridge / bag-conversion / Open-RMF fleet deployment planning specs, or the npa workbench ros2 CLI.
---

# ROS 2

ROS 2 is the robotics middleware tool for prerequisite detection and
deployment planning.

The toolRef intentionally exposes only what is real: the `preflight`
command detects whether a usable ROS 2 Jazzy environment is present
(`ros2` CLI on PATH, `ROS_DISTRO` check, `rclpy` importable) and fails fast
with a remediation message when it is not. Bridge (bag/MCAP <-> topics),
bag conversion, and the Open-RMF fleet adapter are not implemented and are
not exposed. Deployment planning (stage specs, Jetson Thor target) lives in
the pipeline module, which emits specs without executing ROS 2 code.

Preflight measures the environment of its own process. Direct CLI use measures
the machine where it runs; a workflow measures its task container. Select an
image with ROS 2 Jazzy installed and sourced through the workflow resource image
or image override. There is no built-in ROS 2 image route, and the default CPU
image will report missing prerequisites. This tool never probes a remote robot
host or certifies that host from a pod-local result.

## Interfaces

- CLI: `npa workbench ros2 preflight --help`
- Python SDK (workbench-first): `npa.workbench.ros2` (`preflight`)
- Workflow module: `npa.workflows.byof.ros2_pipeline`
  (stage specs, Jetson Thor deploy target, argparse entrypoint)
- Catalog toolRef: `workbench.ros2.preflight`
  (`python3 -m npa.workflows.byof.ros2_pipeline --preflight`)

## Conventions

- The CLI lives at `npa.cli.workbench.ros2` (multi-command package form for
  new tools); the SDK surface lives at `npa.workbench.ros2` and does not go
  through `make_cli_wrapper`.
- Targets ROS 2 Jazzy (`SUPPORTED_ROS_DISTRO = "jazzy"`); Jetson Thor nodes
  run Jazzy + Isaac ROS inside the runtime-fetched container.

## Native validation

In an operator-selected Jazzy runtime, source `/opt/ros/jazzy/setup.bash` and
install NPA into a Python environment that can import the native `rclpy` package:

```bash
NPA_INTEGRATION_E2E=1 NPA_ROS2_PREFLIGHT_LIVE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_ros2_preflight_live.py -q
```

The three tests exercise the real SDK, the ToolRef module entry point, and
wrong-distribution refusal without mocked ROS modules. Explicitly enabled tests
fail if Jazzy is unavailable. They do not validate a remote robot, bridge,
bag conversion, or fleet adapter, and they make no cloud or robot mutations.
The command disables unrelated ROS pytest plugins; this suite does not use
launch-testing hooks and needs no external pytest plugins.
