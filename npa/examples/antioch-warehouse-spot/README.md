# Spot warehouse patrol through Workbench

This Antioch project records a Boston Dynamics Spot model walking through the
native Isaac Sim warehouse. A pretrained flat-terrain locomotion policy drives
the joints from a scripted velocity command. A native RTX camera follows the
robot and records H.264 video alongside physical-state and camera-clock evidence.
This is a locomotion demonstration, not an autonomous warehouse-navigation or
collision-avoidance benchmark.

The project runs in Antioch's managed GPU infrastructure. `npa` packages the
source, submits the scenario, reconciles its state, and collects artifacts into
Nebius S3. No local simulator or GPU is required. Use a private Antioch project
image; do not publish the derived engine image to the public NPA registry.

## Prerequisites

- Install NPA, configure writable S3 storage, and authenticate the operator's
  Antioch CLI as described in the [adapter guide](../../../docs/workbench/antioch.md).
- Use the reviewed Antioch CLI `0.4.289` and Isaac Sim `6.0.1` engine image pinned
  in the Dockerfile. The engine supplies the proprietary simulator and retrieves
  the native robot, warehouse and policy assets under the operator's vendor terms.
- Review the Antioch terms and set the documented run-scoped acceptance value.
  Keep credentials and acceptance values outside the project archive.

From the repository root:

```bash
export NPA_ANTIOCH_ACCEPT_TERMS=YES
npa workbench health preflight --checks s3 --json
npa workbench antioch terms-preflight --output json
npa workbench antioch health --output json
```

## Package and submit

Choose unique S3 input/output prefixes and a workflow identity, then export
`WAREHOUSE_INPUT_URI`, `WAREHOUSE_OUTPUT_URI`, and `WAREHOUSE_RUN_ID`. Reuse the
same identities when retrying one operation. A different source or parameter set
requires a new operation identity and output prefix.

```bash
: "${WAREHOUSE_INPUT_URI:?Set a unique S3 input prefix}"
: "${WAREHOUSE_OUTPUT_URI:?Set a unique S3 output prefix}"
: "${WAREHOUSE_RUN_ID:?Set a unique workflow identity}"
export WAREHOUSE_PACKAGE_DIR="$(mktemp -d)"
WAREHOUSE_REVISION="$(git rev-parse HEAD)"
WAREHOUSE_SOURCE_SHA256="$(git archive HEAD:npa/examples/antioch-warehouse-spot | shasum -a 256 | cut -d ' ' -f 1)"
npa workbench antioch package-project \
  --project-dir npa/examples/antioch-warehouse-spot \
  --package-dir "$WAREHOUSE_PACKAGE_DIR" \
  --source-name npa-warehouse-spot --source-revision "$WAREHOUSE_REVISION" \
  --source-license Apache-2.0 --source-sha256 "$WAREHOUSE_SOURCE_SHA256" \
  --output json
```

Package committed source so the declared revision and digest describe the input.
Upload the package with the configured NPA storage resolver:

```python
import os
from pathlib import Path
from npa.workbench.antioch.storage_config import resolve_storage_client

storage = resolve_storage_client()
prefix = os.environ["WAREHOUSE_INPUT_URI"].rstrip("/")
for path in Path(os.environ["WAREHOUSE_PACKAGE_DIR"]).iterdir():
    destination = prefix + "/" + path.name
    storage.upload_file(str(path), destination)
    assert storage.read_bytes_with_etag(destination)[0] == path.read_bytes()
```

```bash
npa workbench antioch submit \
  --input-path "$WAREHOUSE_INPUT_URI" --output-path "$WAREHOUSE_OUTPUT_URI" \
  --workflow-run "$WAREHOUSE_RUN_ID" --state-id warehouse \
  --robot-type boston_dynamics_spot \
  --task "Record physical quadruped locomotion in a warehouse" \
  --scenario warehouse_spot_patrol \
  --parameters-json '{"seconds":24.0,"speed":0.8,"width":1920,"view":"tracking"}' \
  --output json
npa workbench antioch status \
  --output-path "$WAREHOUSE_OUTPUT_URI" \
  --workflow-run "$WAREHOUSE_RUN_ID" --state-id warehouse --output json
```

`seconds` is simulated recording duration, not a wall-clock performance claim.
`speed` is the commanded forward velocity in m/s. `width` must be divisible by
32 and at least 320; height is 9/16 of width. The alternative `overview` camera
helps inspect placement. `start_x` and `start_y` place the initial robot in metres.
The robot's root is not repositioned during the recording.

## Collect and inspect

After the scenario completes successfully, collect its outputs:

```bash
npa workbench antioch collect \
  --output-path "$WAREHOUSE_OUTPUT_URI" \
  --workflow-run "$WAREHOUSE_RUN_ID" --state-id warehouse \
  --allow-artifacts-only --output json
```

The artifact manifest points to `warehouse_spot.mp4` and `measurements.json`.
The scenario checks finite physical state, upright posture, actual travel,
joint articulation, advancing camera clocks, nonblank images and the expected
frame count. Inspect the decoded video as well as the checks. A smooth recording
does not establish real-robot safety or general policy reliability.

`--allow-artifacts-only` is intentional: the recording is review evidence, not a
LeRobot training dataset. The adapter verifies transferred bytes and publishes
`_SUCCESS.json` only after artifact collection completes.

To recheck a completed capture without launching another job, save its original
request as a private JSON file, including `source_sha256` from packaging. Then run:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_ANTIOCH_WAREHOUSE_REQUEST=/path/to/private/request.json \
npa/.venv/bin/python -m pytest npa/tests/e2e/test_antioch_warehouse_live.py -q
```

The live test verifies the durable completion receipt, collected checksums,
observed motion, advancing clocks and every decoded MP4 frame.

Background scenario compute retires automatically. To stop an unfinished owned
operation, cancel it through Workbench and verify the terminal state:

```bash
npa workbench antioch cancel \
  --output-path "$WAREHOUSE_OUTPUT_URI" \
  --workflow-run "$WAREHOUSE_RUN_ID" --state-id warehouse --output json
```

## Native implementation references

- [Isaac Sim 6.0.1 Spot standalone example](https://github.com/isaac-sim/IsaacSim/blob/v6.0.1/source/standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py)
- [NVIDIA policy deployment guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/robot_simulation/ext_isaacsim_robot_policy_example.html)

The project uses the upstream policy class directly. Robot, environment and
policy payloads are not included in this repository.
