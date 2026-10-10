"""ROS 2 bridge + Open-RMF fleet workflow (BYOF).

"Train on Nebius -> run on robot" pipeline stages: package workbench
artifacts as ROS 2 topic streams, convert datasets between ROS 2 bags and
MCAP, run the Open-RMF fleet adapter, and target NVIDIA Jetson Thor edge
nodes.

Stage functions return plain ``dict`` specs (name / image / argv / env) so
schedulers (the orchestration catalog, skypilot, plain k8s) can materialize
them without importing ROS 2.  Nothing in this module executes ROS 2 code;
see :mod:`npa.cli.workbench.ros2` for the operator-facing commands.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: ROS 2 distribution this pipeline targets.
ROS_DISTRO = "jazzy"

#: Open-RMF release line the fleet adapter stub tracks.
OPEN_RMF_VERSION = "2.12.0"

#: Isaac ROS release line used for Jetson Thor deploy targets.
ISAAC_ROS_RELEASE = "4.0"

#: Runtime-fetch Isaac ROS container (never vendored in this repo).  Nodes
#: pull it at deploy time from NVIDIA's registry after accepting the
#: Isaac ROS EULA (``--accept-isaac-ros-eula`` / ``ISAAC_ROS_ACCEPT_EULA=1``).
ISAAC_ROS_IMAGE_TEMPLATE = "nvcr.io/nvidia/isaac/ros:{release}-{arch}"


@dataclass(frozen=True)
class JetsonThorTarget:
    """Deploy target profile for NVIDIA Jetson Thor edge nodes.

    Pattern: cross-build (or runtime-fetch) ``aarch64`` containers on Nebius,
    push to the operator registry, and pull them on the Thor node at deploy
    time with ``nvidia-container-runtime``.  ROS 2 Jazzy + Isaac ROS 4.0 run
    inside the container; ``ROS_DOMAIN_ID`` isolates the robot's DDS domain
    from the training VPC.
    """

    name: str = "jetson-thor"
    arch: str = "aarch64"
    jetpack: str = "7.0"
    container_runtime: str = "nvidia-container-runtime"
    ros_distro: str = ROS_DISTRO
    isaac_ros_release: str = ISAAC_ROS_RELEASE
    ros_domain_id: int = 42

    def isaac_ros_image(self) -> str:
        """Runtime-fetch Isaac ROS container reference for this target."""
        return ISAAC_ROS_IMAGE_TEMPLATE.format(
            release=self.isaac_ros_release, arch=self.arch
        )

    def deploy_steps(self, image: str) -> list[dict[str, Any]]:
        """Ordered deploy steps for rolling ``image`` out to the Thor node."""
        return [
            {
                "step": "preflight",
                "run": [
                    "nvidia-smi",
                    "jetson_release",
                ],
                "detail": "Verify GPU + JetPack on the Thor node.",
            },
            {
                "step": "eula-preflight",
                "run": ["npa", "third-party-eula-preflight", "isaac-ros"],
                "detail": (
                    "Accept the Isaac ROS EULA before the first runtime "
                    "fetch (third-party-eula-preflight)."
                ),
            },
            {
                "step": "pull",
                "run": ["docker", "pull", image],
                "detail": "Runtime-fetch the aarch64 workbench image.",
            },
            {
                "step": "run",
                "run": [
                    "docker",
                    "run",
                    "--runtime",
                    self.container_runtime,
                    "--gpus",
                    "all",
                    "-e",
                    f"ROS_DOMAIN_ID={self.ros_domain_id}",
                    "--network",
                    "host",
                    image,
                ],
                "detail": "Launch the bridge/fleet node with GPU access.",
            },
        ]


@dataclass
class Ros2PipelineConfig:
    """Configuration for one ros2_pipeline plan."""

    direction: str = "bidir"
    topics: list[str] = field(default_factory=list)
    input_path: str = ""
    output_path: str = ""
    domain_id: int = 0
    rmw: str = ""
    rate: float = 1.0
    duration: float = 0.0
    rmf_server: str = "http://localhost:8000"
    fleet_config: str = ""
    target: JetsonThorTarget = field(default_factory=JetsonThorTarget)
    accept_isaac_ros_eula: bool = False
    dry_run: bool = True


def bridge_stage(config: Ros2PipelineConfig) -> dict[str, Any]:
    """Stage spec for the bidirectional bag/MCAP <-> ROS 2 topic bridge."""
    return {
        "name": "ros2-bridge",
        "toolref": "workbench.ros2.bridge",
        "argv": [
            "npa",
            "workbench",
            "ros2",
            "bridge",
            "--direction",
            config.direction,
            "--domain-id",
            str(config.domain_id),
            *(["--rmw", config.rmw] if config.rmw else []),
            *([item for t in config.topics for item in ("--topics", t)]),
            *(["--input", config.input_path] if config.input_path else []),
            *(["--output", config.output_path] if config.output_path else []),
        ],
        "requires_ros2": True,
    }


def bag_convert_stage(
    input_path: str, output_path: str, topics: list[str] | None = None
) -> dict[str, Any]:
    """Stage spec for converting a recording between ROS 2 bag and MCAP."""
    return {
        "name": "bag-convert",
        "toolref": "workbench.ros2.bag_convert",
        "argv": [
            "npa",
            "workbench",
            "ros2",
            "bag-convert",
            input_path,
            output_path,
            *([item for t in (topics or []) for item in ("--topics", t)]),
        ],
        "requires_ros2": True,
    }


def fleet_adapter_stage(config: Ros2PipelineConfig) -> dict[str, Any]:
    """Stage spec for the Open-RMF fleet adapter (stub)."""
    return {
        "name": "open-rmf-fleet-adapter",
        "toolref": "workbench.ros2.fleet",
        "open_rmf_version": OPEN_RMF_VERSION,
        "argv": [
            "npa",
            "workbench",
            "ros2",
            "fleet-status",
            "--rmf-server",
            config.rmf_server,
            *(["--fleet-config", config.fleet_config] if config.fleet_config else []),
        ],
        "requires_ros2": True,
        "status": "stub",
    }


def jetson_deploy_stage(config: Ros2PipelineConfig, image: str) -> dict[str, Any]:
    """Stage spec for deploying the workbench image to a Jetson Thor node."""
    target = config.target
    return {
        "name": "jetson-thor-deploy",
        "target": asdict(target),
        "isaac_ros_image": target.isaac_ros_image(),
        "accept_isaac_ros_eula": config.accept_isaac_ros_eula,
        "steps": target.deploy_steps(image),
    }


def build_plan(config: Ros2PipelineConfig, image: str = "") -> list[dict[str, Any]]:
    """Build the ordered stage plan for ``config``."""
    plan = [bridge_stage(config)]
    if config.input_path or config.output_path:
        plan.append(
            bag_convert_stage(
                config.input_path or "<bag>",
                config.output_path or "<mcap>",
                config.topics,
            )
        )
    plan.append(fleet_adapter_stage(config))
    plan.append(jetson_deploy_stage(config, image or "<workbench-image>"))
    return plan


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m npa.workflows.byof.ros2_pipeline`` argument parser."""
    parser = argparse.ArgumentParser(
        description="ROS 2 bridge + Open-RMF fleet workflow plan builder."
    )
    parser.add_argument(
        "--plan", action="store_true", help="Print the stage plan as JSON."
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check ROS 2 Jazzy prerequisites; exit 3 when unusable.",
    )
    parser.add_argument(
        "--jetson-target",
        action="store_true",
        help="Print the Jetson Thor deploy target spec as JSON.",
    )
    parser.add_argument(
        "--isaac-ros-image",
        action="store_true",
        help="Print the runtime-fetch Isaac ROS container reference.",
    )
    parser.add_argument(
        "--image", default="", help="Workbench image for the deploy stage."
    )
    parser.add_argument(
        "--direction",
        default="bidir",
        choices=["to-ros", "from-ros", "bidir"],
        help="Bridge direction.",
    )
    parser.add_argument("--topics", default="", help="Comma-separated ROS 2 topics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m npa.workflows.byof.ros2_pipeline ...``."""
    args = build_parser().parse_args(argv)

    if args.preflight:
        from npa.workbench import ros2 as ros2_workbench

        payload = ros2_workbench.preflight()
        print(
            json.dumps(
                {"ok": payload["ok"], "detail": payload["detail"]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["ok"] else 3

    target = JetsonThorTarget()
    if args.isaac_ros_image:
        print(target.isaac_ros_image())
        return 0
    if args.jetson_target:
        print(json.dumps(asdict(target), indent=2, sort_keys=True))
        return 0
    config = Ros2PipelineConfig(
        direction=args.direction,
        topics=[t.strip() for t in args.topics.split(",") if t.strip()],
        target=target,
    )
    if args.plan:
        print(json.dumps(build_plan(config, args.image), indent=2))
        return 0
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
