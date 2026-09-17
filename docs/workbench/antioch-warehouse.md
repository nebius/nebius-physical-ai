# Run the Antioch-authored warehouse on Workbench

The [warehouse workflow](../../workflows/testing/antioch-warehouse.yaml) carries
the local Antioch fulfillment demo into a reproducible Nebius GPU job. It runs
one six-carton batch with native Isaac Sim, saves three review images and measured
physics evidence to S3, then downloads those artifacts in a separate CPU stage
and checks them again.

This is a port of the current warehouse in the local `antioch-neon-domino-demo`
project. The earlier domino and orbital demos are not its input. Antioch was the
authoring environment; this workflow uses NPA's existing Isaac Lab runtime image
and needs no Antioch login, project identity, notebook, or session.

## Run

Configure an NPA project with writable object storage and an RTX PRO 6000
Kubernetes worker. The spec requests one RTX PRO 6000 GPU; an operator can adapt
its resource profile to L40S. Isaac rendering needs an RTX GPU. H100/H200 are not
substitutes. Install the current checkout so source staging includes the new
package, and retain the returned run ID for monitoring.

```bash
npa workbench health preflight --project '<project-alias>' --checks s3
npa workbench workflow validate-spec workflows/testing/antioch-warehouse.yaml
npa workbench workflow plan-spec workflows/testing/antioch-warehouse.yaml --run-id preview
npa workbench workflow preflight-images workflows/testing/antioch-warehouse.yaml
npa workbench workflow submit workflows/testing/antioch-warehouse.yaml \
  --project '<project-alias>' --infra 'k8s/<context>' \
  --run-id '<new-run-id>' --s3-bucket '<bucket>' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

`config.source_overlay: true` installs the submitted source in the existing
Isaac runtime. No new container build is required. Use the standard
[workflow lifecycle](../run-lifecycle.md) for status, logs, artifacts, and exact
job cancellation. The batch completes after six placements; there is no demo
reset, polling deadline, or extra workload limit. A stalled batch remains visible
as a running job and can be cancelled through the normal workflow command.

NVIDIA warehouse USD assets and their materials are fetched at runtime from the
Isaac asset library. Source code and generated evidence contain no redistributed
vendor asset bundle. NPA's existing Isaac runtime-fetch and `ACCEPT_EULA` policy
apply; explicit opt-out refuses before download. See
[container packaging](container-packaging.md) for runtime and license boundaries.

## Evidence and acceptance

All paths are under `s3://<bucket>/antioch-warehouse/<run-id>/` by default.
Override `config.prefix` through the generic workflow configuration when needed.

| Artifact | Contents |
| --- | --- |
| `simulation/validation.json` | Runtime engine, 120 Hz timestep, geometry and rigid-body counts, per-carton position errors and speeds, source SHA-256 inventory, fidelity limits |
| `simulation/events.json` | Ordered accumulation, pickup, transfer, release, and placement events for every carton |
| `simulation/trajectory.json` | Physics-backed carton positions and gantry targets on the simulation timeline |
| `simulation/aisle.png`, `overview.png`, `packing.png` | Actual 1280×720 rendered views of the completed batch |
| `simulation/verification.json` | Producer checks and exact input artifact hashes |
| `reports/verification.json` | Independently recomputed checks and hashes after S3 download |
| `simulation/failure.json` | Failure type when simulation or artifact validation raises before completion |

Both stages fail on failed acceptance checks. All six distinct cartons must be
placed within 2 cm and have residual speed below 4 cm/s. Checks also require
PhysX at 120 Hz, the populated warehouse, eight rigid bodies, ordered events for
each carton, measured conveyor transport, bounded attachment error, and nonblank,
properly exposed images. The verifier decodes PNG pixels instead of trusting
producer image statistics. Verification reports bind all six required evidence
files by SHA-256. Storage failures propagate; application shutdown cannot turn
an earlier exception into a successful stage.

To recheck downloaded evidence without Isaac or a GPU:

```bash
npa/.venv/bin/python -m npa.workflows.antioch_warehouse verify \
  --input-path '<downloaded-simulation-directory>' \
  --output-path '<new-report-directory>'
```

These module paths are worker/developer interfaces. The workflow is the public
composition surface and uses S3 for its cross-stage handoff.

## Fidelity and provenance

The conveyor uses contact surface velocity at 0.45 m/s in world space. Six
2.5 kg dynamic cartons collide, fall under gravity, and settle on a pallet.
Gantry motion is scripted and kinematic, the vacuum attachment is an ideal
fixed joint, and the accumulation trigger reads carton position. The geometry
illustrates guards and controls; it does not establish certified safety,
suction performance, learned-policy quality, or a calibrated facility twin.

The original local `warehouse_demo.py` and primitive methods from
`astra_showcase.py` were adapted into the installed package. Layout and control
remain separate; notebook startup and viewport helpers were replaced with
native [Isaac standalone execution](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/introduction/workflows.html)
and [Kit viewport capture](https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.utility/latest/omni.kit.viewport.utility/omni.kit.viewport.utility.capture_viewport_to_file.html).
The workflow preserves native
[NVIDIA warehouse asset references](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_environments.html)
and records their units and bounds.

See [validation evidence](../testing/antioch-warehouse.md) for the tested source,
commands, numerical results, and remaining execution limits. Planning and unit
checks alone are not evidence of physical simulation success.
