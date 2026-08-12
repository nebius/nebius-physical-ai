# FiftyOne image release

The source contract after PR #218 is SkyPilot-ready, but the shipped workflow
must continue to resolve the existing `npa-fiftyone:1.15.0` tag until a rebuilt
image has been published and smoke-tested. Do not move the resolver to a tag
that does not exist.

## Failure trace and root cause

The PAIDF `curate` state resolves `workbench.fiftyone` through
`SUPPORTED_TOOL_VERSIONS["fiftyone"]` in `npa/src/npa/deploy/images.py` and
`[tool.npa.supported-tools].fiftyone` in `npa/pyproject.toml` to the existing
`npa-fiftyone:1.15.0` manifest. `npa/docker/workbench/fiftyone/Dockerfile` runs as the
non-root `ubuntu` user. SkyPilot 0.12.2's Kubernetes template overrides the
image command with `[/bin/bash, -c, --]`, then uses passwordless `sudo` to
install/check `openssh-server` and `rsync`, prepare sshd, and keep the
`ray-node` container alive.

The published image was missing `sudo` (as well as the SSH/rsync prerequisites),
so that `set -e` initialization could not elevate from `ubuntu`; the container
exited and Kubernetes deleted it before SkyPilot could exec subsequent setup.
Controller retrying therefore produced the observed `container not found
("ray-node")` / `cannot exec in a deleted state` loop. The old bare
`ENTRYPOINT ["/bin/bash"]` was also not a safe general command-passthrough
contract, even though SkyPilot's current Kubernetes template overrides it.
This Dockerfile fixes both contracts and preserves the bundled MongoDB/Brain
path.

After this change merges, an image release operator should build and test the
additive candidate locally:

```bash
docker build -f npa/docker/workbench/fiftyone/Dockerfile \
  -t npa-fiftyone:1.15.0-skypilot1 npa
docker run --rm npa-fiftyone:1.15.0-skypilot1 \
  bash -lc 'sudo service ssh restart; command -v rsync; printf NPA_FIFTYONE_COMMAND_OK'
docker run --rm npa-fiftyone:1.15.0-skypilot1 \
  python /opt/npa/docker/workbench/fiftyone/smoke_functional.py
```

Then push `npa-fiftyone:1.15.0-skypilot1` through the normal reviewed
workbench-image release process and, only after the registry manifest exists,
update `SUPPORTED_TOOL_VERSIONS["fiftyone"]` and
`[tool.npa.supported-tools].fiftyone` together in a follow-up change. This PR
does not push or mutate registry tags.
