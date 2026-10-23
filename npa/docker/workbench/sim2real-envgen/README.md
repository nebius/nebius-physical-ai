# Sim2Real EnvGen runtime

`npa-envgen` is the source-attested runtime for canonical Sim2Real Stage 4.
It generates deterministic task/scenario records and split inputs; it does not
execute Genesis physics, train a policy, or evaluate a rollout.  Those are
separate Isaac workflow stages and must carry their own image evidence.

The image deliberately does not inherit the Genesis/LeRobot environment.  Its
Python 3.11 venv is rebuilt from
`../common/sim2real-envgen-requirements.txt`, which pins the current NPA
closure, Pillow 13, and fsspec 2026.6.0 without Genesis, LeRobot, datasets, or
Torch.  This keeps the unresolved Genesis upstream closure visible rather than
lowering a security floor or incorrectly blocking EnvGen.

## Build boundary

Run the canonical builder from the repository root with an immutable,
compatible CUDA parent for the Cosmos3-reason role:

```bash
SOURCE_SHA="$(git rev-parse HEAD)"
BASE_IMAGE="<immutable-source-attested-cuda-parent>" \
NPA_SOURCE_SHA="${SOURCE_SHA}" \
npa/docker/workbench/sim2real-build.sh
```

The builder first produces `npa-cosmos3-reason` at the exact source revision
and uses that newly built, source-labelled image as EnvGen's parent.  Set
`SIM2REAL_ENVGEN_BASE_IMAGE` only to a different image whose
`org.opencontainers.image.revision` label equals `NPA_SOURCE_SHA`; the builder
refuses a missing or mismatched label before building EnvGen.  A remote parent
must be addressed by immutable digest; the local freshly-built Cosmos role may
use its current-source tag only within the same build invocation.  `--push`
still requires the normal secure-image-build, access, license, byte-scan, and
immutable-digest gates.

Before selecting an explicit parent, inspect its OCI label locally and require
the exact checkout SHA. This does not authenticate a registry or replace the
builder's check; it makes the source-attestation prerequisite actionable:

```bash
PARENT_IMAGE="${SIM2REAL_ENVGEN_BASE_IMAGE:?set an immutable parent image}"
PARENT_SHA="$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "${PARENT_IMAGE}")"
test "${PARENT_SHA}" = "${SOURCE_SHA}"
```

`BUILD_LEGACY_GENESIS_EVAL=1` is intentionally separate.  That old individual
image is not a canonical Sim2Real workflow role and remains unqualified until
the independent Genesis runtime has a secure supported closure.

## Capability proof

The role's image smoke is
`python -m npa.smoke.test_sim2real_envgen_functional`.  It runs as `ubuntu`,
removes `PYTHONPATH`, imports the exact Stage 4 modules, and generates a local
raw-scenario sample.  It does not claim GPU/Genesis, Cosmos3, Token Factory,
policy, quality, or hardware success.  Those claims require a complete
current-source canonical workflow receipt.
