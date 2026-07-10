# NPA workflow golden specs

Declarative `apiVersion: npa.workflow/v0.0.1` workflows. Authoring guide:
[docs/workbench/npa-workflow-guide.md](../../docs/workbench/npa-workflow-guide.md).

Agent skills: `skills/workflows/author-npa-workflow/SKILL.md` (edit) and
`skills/workflows/generate-npa-workflow/SKILL.md` (design new pipelines).

| Spec | Stages | Pattern |
| --- | --- | --- |
| [vlm-eval-single.yaml](vlm-eval-single.yaml) | 1 | Single tool, terminal |
| [tokenfactory-rollout-judge.yaml](tokenfactory-rollout-judge.yaml) | 2 | Serial chain with inputs |
| [tokenfactory-cosmos-gate.yaml](tokenfactory-cosmos-gate.yaml) | 6 | Creative reason → augment → VLM gate loop |
| [sim2real-vlm-rl.yaml](sim2real-vlm-rl.yaml) | 11 | Nested loops + dynamic gate |
| [bdd100k-pipeline.yaml](bdd100k-pipeline.yaml) | 11 | LanceDB → train → eval → review |
| [byof-maniskill.yaml](byof-maniskill.yaml) | 1 | OSS registry candidate: ManiSkill pinned image + PickCube smoke |
| [byof-mujoco-playground.yaml](byof-mujoco-playground.yaml) | 1 | OSS registry candidate: MuJoCo Playground pinned image + Cartpole smoke |
| [byof-robocasa.yaml](byof-robocasa.yaml) | 1 | OSS registry candidate: RoboCasa pinned image + headless kitchen-task smoke |
| [byof-openpi.yaml](byof-openpi.yaml) | 1 | OSS registry candidate: OpenPI pinned image + pi05 DROID config smoke |
| [byof-droid-policy-learning.yaml](byof-droid-policy-learning.yaml) | 1 | OSS registry candidate: DROID policy learning pinned image + RLDS config smoke |

```bash
npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/tokenfactory-cosmos-gate.yaml
npa workbench workflow plan-spec npa/workflows/workbench/npa-workflows/tokenfactory-cosmos-gate.yaml \
  --run-id demo --assume-decision loop_back
```

SkyPilot execution for BDD100K remains at
`npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml` (submitted via
`npa/scripts/run_bdd100k_pipeline.py`).
