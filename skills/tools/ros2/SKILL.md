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
