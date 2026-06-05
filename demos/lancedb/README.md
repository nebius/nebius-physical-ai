# LanceDB Demo

Demo-specific assets for LanceDB belong here. Keep reusable setup in
`infra/bootstrap/` and reusable workflow YAML under
`npa/workflows/workbench/skypilot/`.

The existing BDD100K pipeline remains the runnable LanceDB reference:

```bash
npa workbench workflow submit npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --run-id bdd100k-demo \
  --var NPA_S3_BUCKET=<bucket>
```
