# NPA workflow golden specs

Declarative `apiVersion: npa.workflow/v0.0.1` workflows. Authoring guide:
[docs/workbench/npa-workflow-guide.md](../../docs/workbench/npa-workflow-guide.md).

| Spec | Stages | Pattern |
| --- | --- | --- |
| [vlm-eval-single.yaml](vlm-eval-single.yaml) | 1 | Single tool, terminal |
| [tokenfactory-rollout-judge.yaml](tokenfactory-rollout-judge.yaml) | 2 | Serial chain with inputs |
| [sim2real-vlm-rl.yaml](sim2real-vlm-rl.yaml) | 11 | Nested loops + dynamic gate |
| [bdd100k-pipeline.yaml](bdd100k-pipeline.yaml) | 11 | LanceDB → train → eval → review |

```bash
npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml
npa workbench workflow plan-spec npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml --run-id demo
```

SkyPilot execution for BDD100K remains at
`npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml` (submitted via
`npa/scripts/run_bdd100k_pipeline.py`).
