# Public Genesis runtime

The canonical recipe requires an explicitly selected, source-attested CUDA 13
Python 3.11 workbench base and retains Genesis 0.4.6. It packages rigid Panda/Box simulation, teacher training,
checkpoint export, camera-demo conversion, native LeRobotDataset reading through
PyAV, and ACT student dependencies. A rebuilt candidate must pass its normal
publication gates and real GPU teacher/export/camera qualification before it
can become a public default. The old release remains quarantined.

The base's Python 3.11/Torch 2.13 CUDA 13 runtime stays intact. Genesis instead
creates `/opt/genesis/venv` with Python 3.11: its retained LeRobot 0.4.4
upstream contract requires `torch<2.11`, which cannot safely replace the
source-attested base runtime. The image selects that dedicated environment with
both `PATH` and `NPA_BAKED_PYTHON`; EnvGen preserves the same selection after
its sanitizing scratch stage. Isolation prevents the LeRobot resolver from
downgrading the base runtime; it is neither GPU/hardware compatibility evidence
nor a substitute for a complete secure dependency closure.

## Current source build boundary

The current source-attested recipe does **not** have a supported secure
LeRobot/Datasets closure, so it must not be presented as a completed manual
build path. Its `lerobot==0.4.4` dependency resolves `datasets==4.8.5`, whose
declared `fsspec[http] <=2026.2.0` ceiling conflicts with the enforced
`fsspec==2026.6.0` security floor. Lowering that floor, ignoring the conflict,
or relabelling a historical image is not a repair.

Newer released LeRobot lines require Python 3.12 but retain incompatible
Datasets, Pillow, and Torch ceilings; the official upstream main branch has the
same boundary. The existing RoboCasa-specific LeRobot derivative is qualified
only for that image's ACT evaluation and is not Genesis evidence. Accordingly,
the commands below are a source-attested build sequence to use only after a
separately reviewed, licensed, source-bound closure has passed its own
dependency, CPU-functional, image-security, and GPU qualification. They do not
currently produce a qualified Genesis or Envgen image.

The public recipe excludes TetGen's AGPL distribution and rejects volume
tetrahedralization through a repository-owned import shim. Rigid-body scenes
do not use TetGen. This does not classify every soft-body operation as
tetrahedralization; capabilities outside the advertised rigid workload need
their own license and execution review.

LeRobot 0.4.4 requires Diffusers below 0.36, while the security floor requires
at least 0.38. A forced upgrade leaves an inconsistent policy environment.
The public image therefore excludes Diffusers and W&B and sets
`NPA_GENESIS_SUPPORTED_STUDENT_POLICIES=act`. This optional comma-separated
image capability declaration makes student loading reject unsupported policy
types before importing their frameworks. The SDK keeps its existing policy
classes available to separately qualified operator images when the variable
is absent. Diffusion and SmolVLA evaluation need such an image with a compatible,
scanned dependency closure; changing this variable cannot install those
dependencies.

The public recipe also excludes TorchCodec: its newly selected version is
incompatible with Torch 2.6, and the compatible 0.2.1 decoder cannot consume the
file handles passed by this LeRobot release. Native dataset reading chooses its
existing PyAV backend without an override. The image pins `av>=15,<16` and
Hugging Face Hub 0.35.3 to the retained upstream dependency bounds.

All exclusions occur in the original dependency installation layer with wheel
caching disabled. `prepare_public_runtime.py` binds both original METADATA
files to the exact reviewed wheel hashes, removes only the documented optional
dependency edges, retains licenses/notices, refreshes RECORD hashes and sizes,
and requires excluded distributions to be absent before writing. `pip check`
also verifies the resulting dependency closure. LeRobot 0.4.4 eagerly imports
GR00T and other optional families while importing ACT. The public recipe binds
the exact policy registration and factory sources to reviewed hashes, retains
the native ACT branches and saved-processor loader, and rejects every other
factory family. It recompiles affected bytecode and updates both source and
bytecode RECORD entries in the same installation layer;
an unknown source or original RECORD fails before either file changes.

The CPU build smoke converts real raw demonstration arrays with system FFmpeg,
reads both cameras through the native dataset reader and its default PyAV
backend, constructs ACT, performs an action, and saves/reloads the checkpoint
and normalization processors through the NPA student loader. It prevents
network fetches and unexpected excluded-package imports. This smoke proves
dependency behavior; it does not prove Genesis GPU physics, policy quality or
camera rendering. Both `Dockerfile` and `Dockerfile.sm120` accept an explicitly
selected Python 3.11 parent and record the exact NPA source SHA. Build the
canonical base recipe first and pass its current-source local tag (or image ID):

```bash
SOURCE_SHA="$(git rev-parse HEAD)"
npa/docker/workbench/base/cuda13-blackwell/build.sh --tag "dev-${SOURCE_SHA}"
BASE_IMAGE="npa-base:cuda13-blackwell-dev-${SOURCE_SHA}"
npa/docker/workbench/genesis/build.sh \
  --base-image "${BASE_IMAGE}" --tag "dev-${SOURCE_SHA}"

GENESIS_IMAGE="npa-genesis:dev-${SOURCE_SHA}"
test "$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "${GENESIS_IMAGE}")" = "${SOURCE_SHA}"
docker build --build-arg "BASE_IMAGE=${GENESIS_IMAGE}" \
  --build-arg "NPA_SOURCE_SHA=${SOURCE_SHA}" \
  -f npa/docker/workbench/sim2real-envgen/Dockerfile \
  -t "npa-envgen:dev-${SOURCE_SHA}" npa
```

Historical catalog digests are not current-source parents or fallbacks for this
sequence. `sim2real-build.sh` requires `SIM2REAL_GPU_BASE_IMAGE` (or its
`GENESIS_IMAGE` compatibility alias) for the same reason. Neither recipe's CPU
proof qualifies GPU physics, policy quality, or hardware transfer.
