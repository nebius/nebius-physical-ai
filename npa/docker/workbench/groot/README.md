# GR00T Container Image

Build locally for the Linux x86_64 Nebius runtime:

```bash
docker/workbench/groot/build.sh
```

Build and push to an operator-controlled registry:

```bash
docker/workbench/groot/build.sh --registry <your-registry>/<namespace> --push
```

The image bakes Isaac-GR00T N1.7 at commit
`3df8b3825d67f755e69141446f4315f281b9b7e6`, GR00T package version `0.1.0`,
Isaac Lab `2.3.2.post1`, and the Cosmos
Reason2 model revision patch used by the VM deploy path. Model weights,
datasets, HuggingFace cache, checkpoints, and outputs stay outside the image
under `/opt/groot-data`.

The same environment supports real single-node training. The reference
`npa.workflow` spec at
`npa/workflows/workbench/npa-workflows/groot-1-7-finetune.yaml` runs the pinned
vendor launcher locally in the allocated image and uses `torchrun` when more
than one GPU is selected.
