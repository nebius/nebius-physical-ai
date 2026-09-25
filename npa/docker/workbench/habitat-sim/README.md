# Habitat-Sim image builds

From a clean, committed checkout, build the public runtime-fetch recipe into a
new OCI archive in an existing owner-only directory outside the checkout:

```bash
bash npa/docker/workbench/habitat-sim/build.sh /path/to/private/bootstrap.oci.tar
```

The default selects `Dockerfile.bootstrap`, matching `packaging-contract.yaml`
and the trusted public image workflow. It projects the Dockerfile and every
repository-owned `COPY` input from the exact Git commit and passes that full SHA
as `NPA_SOURCE_SHA`. The output retains BuildKit provenance and SBOM attestations.
It contains prerequisites and launchers; Habitat-Sim, its native dependencies,
scientific wheels, and the scene are fetched only when the workload runs.

The quarantined baked recipe is available only through an explicit option:

```bash
bash npa/docker/workbench/habitat-sim/build.sh --legacy-baked /path/to/private/legacy.oci.tar
```

That mode retains the legacy `Dockerfile`, source-provenance context, and
`NPA_SOURCE_MANIFEST_SHA256` contract consumed by `verify_image.py`. Its pending
runtime/source closure is separate from the public bootstrap's source-delivery
and payload-absence checks in `verify_bootstrap.py`.

Neither mode publishes or loads the image. The output must not already exist;
successful export creates a mode-0600 archive without replacing another file.
A local build is not a publication or GPU qualification. `verify_bootstrap.py`
expects a Docker-save archive, while `verify_image.py` checks the legacy attested
OCI graph; their archive formats and acceptance contracts are not interchangeable.

See the [Habitat-Sim guide](../../../../docs/workbench/byof-habitat-sim.md)
for workflow commands, verification, runtime inputs, and release quarantine.
