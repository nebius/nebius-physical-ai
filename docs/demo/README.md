# Demo runbooks

[Docs](../README.md)

These are operator and presentation references for the multi-tool Workbench
demo. They require prepared infrastructure and demo artifacts. For a first run
from your own input, use the [workload guides](../workbench/guides/README.md).

| Task | Runbook | Before you start |
| --- | --- | --- |
| Recreate the multi-tool demo | [8-GPU H200 demo](8gpu-h200.md) | H200 inference host, separate RT-core simulation host, S3, and access to the selected artifact manifest |
| Use the RTX PRO simulation variant | [Architect live pack](architect-live-rtxpro.md) | Your own project settings and compatible images; read its recorded runtime gaps |
| Present prepared recordings in the browser | [Dedicated agent UI](architect-live-agent-ui.md) | A deployed agent and staged G1/GR00T recordings |

Replace example aliases and infrastructure values with your own private
configuration. The historical source-artifact manifest may need separate read
access; use your own manifest if unavailable. The architecture pack records
known failures, so it is not a guarantee that every demo component currently
runs unchanged.

For image/GPU selection, use the [compatibility matrix](../workbench/image-gpu-compatibility-matrix.md).
After a demo, cancel active jobs and follow [teardown](../teardown.md) for only
the resources you created.
