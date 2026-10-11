# VLA-JEPA candidate image

Build this recipe from the exact reviewed NPA source checkout, with `npa/` as
the build context. Pass its full lowercase commit as the existing
`NPA_ADAPTER_REVISION` build argument. The build rejects a missing or malformed
revision; the caller must also verify that the context bytes belong to it.

The image copies NPA to `/opt/npa/src`, declares the dependency-complete vendor
interpreter through `NPA_BAKED_PYTHON`, and bakes the revision into
`NPA_IMAGE_SOURCE_SHA`. The maintained workflow renderer imports the real
VLA-JEPA adapter and compares that marker with the workflow's `source_sha`.
Do not inject a marker at submission time to relabel an older image.

Use the [VLA-JEPA workflow guide](../../../../docs/workbench/guides/lerobot-vla-jepa.md)
for private image routing, runtime-fetched inputs, and native validation.
