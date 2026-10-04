# EmbodiedGen V2: image to rigid object

EmbodiedGen is offered as one operator-private BYOF workflow, not a persistent
service. It runs the upstream V2.1.0 `img3d-cli` with the TRELLIS backend,
exports the resulting mesh and URDF, and proves that exact generated URDF has
collision geometry and settles under gravity in PyBullet. It does not claim
articulated-object generation, calibrated physical properties, or policy
improvement.

The workflow also invokes EmbodiedGen's documented URDF-to-MJCF conversion for
a MuJoCo/Genesis handoff. The checked physics result is the direct PyBullet
URDF load; MJCF conversion itself is not presented as a separate simulator run.

## Delivery and terms

The image uses a digest-pinned CUDA development base so upstream TRELLIS
extensions can compile. CUDA's distribution conditions do not establish an
anonymous-registry right for this recipe, so build it only in the operator's
private registry. EmbodiedGen source is Apache-2.0; TRELLIS source and the
selected `microsoft/TRELLIS-image-large` revision are MIT. The immutable image
contains only its CUDA/OS bootstrap and this fetcher; EmbodiedGen source,
TRELLIS source, model weights, task input, Python application dependencies,
cache, outputs, and credentials are never baked. Runtime fetching does not
change upstream use or output rights.

The workflow maps the configured Token Factory credential into EmbodiedGen's
OpenAI-compatible VLM adapter for URDF property estimation. That estimate is
explicitly marked non-calibrated in the result JSON.

## Build and run

First run the configured health/access preflights and use the resolved private
registry rather than a public image namespace. The build tag must bind the full
committed NPA SHA:

```bash
npa workbench health preflight --checks nebius --json
npa workbench health preflight --json
NPA_SOURCE_SHA=$(git rev-parse HEAD)
IMAGE_TAG="${NPA_REGISTRY}/npa-embodiedgen:dev-${NPA_SOURCE_SHA}"
bash npa/docker/workbench/embodiedgen/build.sh \
  "${IMAGE_TAG}"
IMAGE_DIGEST=$(docker buildx imagetools inspect "${IMAGE_TAG}" \
  --format '{{.Manifest.Digest}}')
IMAGE_REF="${NPA_REGISTRY}/npa-embodiedgen@${IMAGE_DIGEST}"
```

`NPA_REGISTRY` must begin with a fully-qualified private registry host;
`build.sh` rejects Docker shorthand and registries configured as anonymous/public,
then only pushes the exact `dev-<full-source-sha>` tag. Resolve that tag to the
registry-reported digest before planning or submitting; a tag is not an accepted
execution input. Validate and plan the workflow with that immutable private image
and pass the Token Factory secret through normal workflow secret plumbing:

```bash
npa workbench workflow validate-spec workflows/testing/byof-embodiedgen.yaml
npa workbench workflow plan-spec workflows/testing/byof-embodiedgen.yaml \
  --run-id embodiedgen-check \
  --var "base_image=${IMAGE_REF}" \
  --check-render
npa workbench workflow submit workflows/testing/byof-embodiedgen.yaml \
  --var "base_image=${IMAGE_REF}" \
  --image-override "workbench.byof.repo=${IMAGE_REF}" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY
```

Use one RTX PRO 6000 Blackwell GPU. The checked-in input is an exact upstream
example; replace `input_uri` with an operator-controlled HTTPS or S3 image URI
for a real task. The output includes the source/model revisions, GPU identity,
hashes for generated mesh and collision geometry, the VLM-estimate disclaimer,
PyBullet contact/settling facts, a PNG, and a fully decoded MP4.
