# `npa workbench ros2`

Two-way middleware bridge between Physical AI workbench artifacts and ROS 2
(Jazzy LTS): bag/MCAP files <-> live ROS 2 topics, plus an Open-RMF fleet
adapter stub and a Jetson Thor deploy target. Unlike the foxglove/lichtblick
viz bridges (one-way, artifacts -> viewer), the ros2 bridge is bidirectional:
workbench outputs are *published* onto robot topics and robot telemetry is
*recorded* back into artifacts, closing the sim-to-real loop
"train on Nebius -> run on robot".

Every command requires ROS 2 Jazzy installed and sourced
(`source /opt/ros/jazzy/setup.bash`). Without it the commands fail fast with
a remediation message instead of a traceback.

## Command Tree

```text
Usage: npa workbench ros2 [OPTIONS] COMMAND [ARGS]...

ROS 2 bridge: bag/MCAP <-> topics (bidirectional), bag conversion, Open-RMF fleet.

Options
--help  Show this message and exit.

Commands
bridge        Bidirectional bridge between bag/MCAP artifacts and ROS 2 topics.
bag-convert   Convert recordings between ROS 2 bags and MCAP files.
fleet-status  Show Open-RMF fleet adapter status (stub).
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `bridge` | Bidirectional bridge between bag/MCAP artifacts and ROS 2 topics. |
| `bag-convert` | Convert recordings between ROS 2 bags and MCAP files. |
| `fleet-status` | Show Open-RMF fleet adapter status (stub). |

### `bridge`

```bash
npa workbench ros2 bridge --direction bidir \
  --topics /camera/image_raw,/joint_states \
  --input run_042.mcap --output run_042_ros2.mcap \
  --domain-id 0 --rate 1.0
```

| Option | Description |
| --- | --- |
| `--direction` | `to-ros` (publish artifacts onto topics), `from-ros` (record topics into artifacts), or `bidir` (default: both). |
| `--topics` | ROS 2 topics to bridge. Repeat the flag or comma-separate. |
| `--input`, `-i` | Input bag/MCAP file to publish (to-ros/bidir). |
| `--output`, `-o` | Output bag/MCAP file to record (from-ros/bidir). |
| `--domain-id` | ROS_DOMAIN_ID. |
| `--rmw` | RMW implementation override, e.g. `rmw_zenoh_cpp`. |
| `--rate` | Replay rate multiplier. |
| `--duration` | Run duration in seconds; 0 runs until interrupted. |
| `--output-format` | `text` or `json`. |

### `bag-convert`

```bash
npa workbench ros2 bag-convert run_042/ run_042.mcap \
  --from-format ros2bag --to-format mcap --compression zstd
```

| Option | Description |
| --- | --- |
| `--from-format` | `ros2bag` or `mcap`. |
| `--to-format` | `ros2bag` or `mcap`. |
| `--topics` | Only convert these topics. |
| `--compression` | Output compression, e.g. `zstd`. |

### `fleet-status`

```bash
npa workbench ros2 fleet-status --fleet-config fleet.yaml \
  --rmf-server http://rmf:8000
```

The Open-RMF fleet adapter is currently a stub: `fleet-status` validates its
arguments and reports prerequisites. A full adapter requires ROS 2 Jazzy,
the `rmf_fleet_adapter` packages, and a reachable Open-RMF API server
(tracked Open-RMF release line: v2.12.0).

## Jetson Thor deploy target

The pipeline's edge target is NVIDIA Jetson Thor (`aarch64`, JetPack 7.x,
Isaac ROS 4.0). The deploy pattern, implemented in
`npa.workflows.byof.ros2_pipeline.JetsonThorTarget`:

1. **Build/fetch on Nebius.** Workbench images are built (or the Isaac ROS
   container is referenced) as `aarch64` images and pushed to the operator
   registry. Nothing Jetson-specific is vendored in this repo.
2. **Runtime-fetch on the Thor node.** The node pulls the image at deploy
   time; the Isaac ROS container (`nvcr.io/nvidia/isaac/ros:4.0-aarch64`)
   follows the same runtime-fetch model as the thin Isaac images.
3. **EULA preflight.** Isaac ROS ships under NVIDIA's EULA: run
   `npa third-party-eula-preflight isaac-ros` (or set
   `ISAAC_ROS_ACCEPT_EULA=1`) before the first fetch.
4. **Launch with GPU access.** `docker run --runtime nvidia-container-runtime
   --gpus all -e ROS_DOMAIN_ID=<id> --network host <image>` -- ROS 2 Jazzy
   runs inside the container, and `ROS_DOMAIN_ID` isolates the robot's DDS
   domain from the training VPC.

Print the target spec or plan without ROS 2 installed:

```bash
python3 -m npa.workflows.byof.ros2_pipeline --jetson-target
python3 -m npa.workflows.byof.ros2_pipeline --plan --image <registry>/workbench-ros2:aarch64
```

## Examples

```bash
npa workbench ros2 --help
npa workbench ros2 bridge --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `ros2`.
