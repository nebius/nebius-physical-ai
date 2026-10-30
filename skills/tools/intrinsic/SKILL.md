---
name: intrinsic
description: Use when working on Intrinsic Core read-only validation (preflight, ICON status, digital-twin reachability) or the npa workbench intrinsic CLI.
---

# Intrinsic Core

Intrinsic Core (Apache 2.0, announced at ROSCon 2026) is Google/Alphabet's
open industrial-robotics stack: a k3s-packaged local runtime, the ICON
real-time control engine, motion/grasp planning, FoundationPose 6-DoF
perception, and ROS 2 interoperability (requires ROS 2 Lyrical Luth).

The host-only CLI/SDK intentionally exposes only what is real and read-only:

- `preflight`: host checks (Ubuntu >= 22.04, `ROS_DISTRO=lyrical`, k3s,
  `inctl` on PATH, GPU warning) plus runtime checks (k3s pods, TCP
  ingress at `--address`, `inctl service state list` with no errored
  services). Fails fast with remediation when unusable.
- `icon-status`: read-only ICON real-time control status
  (`inctl icon status --instance_name=icon`).
- `world-probe`: read-only digital-twin reachability — TCP ingress plus
  a world/ObjectWorld entry in `inctl service state list`. There is no
  read-only `inctl world` query (`inctl world reset` mutates state and is
  deliberately not exposed).

## Interfaces

- CLI: `npa workbench intrinsic preflight --help`
- Python SDK (workbench-first): `npa.workbench.intrinsic`
  (`preflight`, `icon_status`, `world_probe`)

These probes are **not** `npa.workflow` toolRefs: they inspect local host
state (`systemctl`, k3s, ROS, and the local `inctl` installation). Run them
directly on an Intrinsic Core host. A generic workflow pod would observe its
own environment rather than the target runtime.

## Conventions

- The CLI lives at `npa.cli.workbench.intrinsic`; the SDK surface lives
  at `npa.workbench.intrinsic` and does not go through `make_cli_wrapper`.
- Requires ROS 2 Lyrical Luth (`SUPPORTED_ROS_DISTRO = "lyrical"`,
  `ROS_DISTRO` env); the ingress address defaults to
  `localhost:17080` (override with `INTRINSIC_ADDRESS`).
- All subprocess invocations are argv lists, never `shell=True`.
- Unit tests mock subprocess/network; no live infrastructure.
