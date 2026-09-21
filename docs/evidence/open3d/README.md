# Open3D measurement evidence

The numbers quoted in `skills/tools/open3d/SKILL.md` and in the docstrings of
`npa/src/npa/workbench/open3d/runner.py` come from here. They were previously kept outside the
repository, which meant a reader had to take them on trust; a reviewer asked for them to be
committed so the claims can be checked against the data that produced them.

Every JSON here was written by a script in `harness/`, run inside the `npa-open3d` container so the
Open3D version and sampling behaviour match what the tool actually uses. Each harness documents its
own question and method in its module docstring.

## What the load-bearing claims rest on

| claim | file |
|---|---|
| The headline `unsupported_area_fraction` ranks cases backwards; the share past three voxels separates them | `band-profile-across-scenes.json` |
| The reading never over-calls: no zero-error reconstruction reads as an invented shell | `poisson-procedure-and-overcall-audit.json`, `band-validity-sweep.json` |
| It does under-call below a sensitivity floor, so a low reading is a lower bound | `floor-across-geometry-and-sampling.json` |
| The floor is set by the voxel-to-sample-spacing ratio, not by geometry, sampling or mesh resolution | `floor-vs-voxel-multiple.json`, `floor-vs-resolution.json` |
| Even sampling zeroes the headline fraction, so its failure is a capture-time artefact | `poisson-procedure-and-overcall-audit.json` |

## Reproducing

```bash
docker build -f docker/workbench/open3d/Dockerfile \
    --build-arg NPA_SOURCE_SHA="$(git rev-parse HEAD)" -t npa-open3d:local npa/
docker run --rm -v "$PWD/docs/evidence/open3d:/data" --entrypoint python npa-open3d:local \
    /data/harness/poisson_procedure_and_overcall_audit.py
```

Sampling is unseeded in several harnesses, deliberately: where a single cell's value carries
run-to-run noise comparable to the spread being measured, that is itself the finding and the harness
says so rather than publishing a precise-looking number. `floor-vs-resolution.json` is the clearest
case. Re-running will move those values; the directions and orders of magnitude are what the claims
rest on.

## View framing

`capability-record-view-framing.json` is the entry point. The operator reported a clipped scene and
root's independent inspection of the delivered PNGs found the top-down clipped at the bottom and the
baseline overview past the frame edge. There were two separate causes, and the interesting part is
that neither was a missing camera fit.

`camera-framing-probe.json` measures the product viewer: one camera fitted to the fused cloud, shared
across four Rerun views, in frame half-extents where 1.0 is the edge. Three views sit at 0.93-0.95 and
the fourth at 1.233. The one that clipped is the audit view, whose whole purpose is showing the
surface the crop removed -- geometry that lies outside the cloud by construction.

`frame-edge-audit-before.json` and `-after.json` measure the evidence renders from PNG bytes, so
"clipped at the bottom" is a count rather than an impression. `background-choice-sweep-refit.json`
chooses the page by measuring contrast against the content's own luminance; it also records that I
expected a trade there and the measurement found none, and separates the controlled comparison from
the confounded one.

## The live workflow-runtime gate

`live-runtime-attempt.json` records how far a real run gets and where it stops. It is the honest
counterpart to the container results above: the image, the private digest-pinned delivery route, real
Open3D capability executing on a cluster node, and a six-state plan that renders are all proven; the
`npa.workflow` runtime itself is not. A pod is not a substitute for the runtime, and the file says so
rather than letting four cleared gates imply a fifth.

## Results that are retained but do not support anything

`band-false-negative-sweep.json` is marked inconclusive and is kept for that reason. Occluding a
watertight mesh to manufacture a false negative does not work: Poisson reconstruction closes a hole
in a solid object by tracking the true surface, so there is almost nothing invented to miss. The
false-negative direction was eventually measured by a different construction, a polar-cap sweep in
the review lane's harness, and the curve it produced is pinned as a regression test in
`npa/tests/workbench/test_open3d.py`.
