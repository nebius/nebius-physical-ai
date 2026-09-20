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

## Results that are retained but do not support anything

`band-false-negative-sweep.json` is marked inconclusive and is kept for that reason. Occluding a
watertight mesh to manufacture a false negative does not work: Poisson reconstruction closes a hole
in a solid object by tracking the true surface, so there is almost nothing invented to miss. The
false-negative direction was eventually measured by a different construction, a polar-cap sweep in
the review lane's harness, and the curve it produced is pinned as a regression test in
`npa/tests/workbench/test_open3d.py`.
