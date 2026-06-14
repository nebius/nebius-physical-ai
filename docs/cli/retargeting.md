# `npa workbench sonic retargeting`

Motion retargeting for SONIC locomotion workflows. Registered under the SONIC
tool namespace.

## Command Tree

```text
Usage: npa workbench sonic retargeting [OPTIONS] COMMAND [ARGS]...

Motion retargeting for SONIC locomotion workflows.

Options
--help  Show this message and exit.
Commands
run      Retarget source motion artifacts into the SONIC embodiment schema.
workflow Print the bundled SkyPilot workflow path and default image.
status   Report retargeting tool status.
list     List supported source formats and defaults.
```

## Examples

```bash
npa workbench sonic retargeting --help
npa workbench sonic retargeting run --help
npa workbench sonic retargeting run \
  --input-path s3://<your-bucket>/motions/source/ \
  --output-path s3://<your-bucket>/sonic-locomotion/<run-id>/retargeted/
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `retargeting`.
