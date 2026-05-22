# Workflow Engine Recommendation - W8

## Verdict

Verdict C: Hybrid. Use Argo Workflows as the Workbench Composition layer
orchestrator. Use Ray, through KubeRay or Anyscale, inside compute-intensive
steps where distributed Python, RL, data processing, training, tuning, or
serving is the actual requirement.

Do not choose Ray-only. The deciding evidence is that Ray Workflows is not a
stable workflow engine. It was alpha in older docs, then deprecated, and the
`ray.workflow` package was removed in Ray 2.48. Ray remains strategically and
technically important, but as a distributed compute substrate, not as the
durable cross-tool workflow controller.

This confirms the current strategy memory: Argo Workflows for orchestration,
Ray for distributed compute within pipeline steps. The update is sharper:
Workbench should avoid saying "Ray Workflows" as a future orchestration
candidate unless Ray reintroduces a stable, supported equivalent.

## Load-bearing reasons

1. Argo matches the Workbench workflow shape. Workbench needs to compose
   heterogeneous tools: FiftyOne, Cosmos, LeRobot, Genesis, Isaac Lab, Rerun,
   partner APIs, and storage movement. Argo's native unit is a Kubernetes
   container step in a DAG with retries, conditions, parameters, artifacts,
   scheduling, suspend/resume, and UI/API visibility.

2. Ray matches the Workbench distributed compute shape. Ray is strongest where
   Argo is intentionally thin: Python-native distributed tasks, actors, object
   refs, Ray Data, Ray Train, Ray Tune, Ray Serve, RLlib, and high-throughput
   CPU/GPU scheduling inside a compute phase.

3. The Nebius-Anyscale partnership is real, but it does not rescue Ray-only.
   Nebius and Anyscale have an integrated managed Ray story for multimodal and
   physical AI, which Workbench should use. But that partnership is strongest
   when Ray is used for Ray-shaped work. It does not change the fact that the
   open-source Ray workflow engine was removed.

## Scores

| Area | Argo Workflows | Ray + KubeRay |
| --- | --- | --- |
| Native workflow capability score | 13/16 | 8/16 |
| Distributed compute inside a step | Plugin/external | Native |
| Workflow engine maturity | CNCF graduated Argo project; Argo Workflows mature | Ray Workflows deprecated/removed |
| Partner/GTM leverage | Weak direct Nebius partnership | Strong Anyscale partnership |
| Marketplace partner friction | Low, container/service/API native | Higher for non-Ray tools |

The scores count native support against the requested Workbench workflow
capability matrix. They are not weighted. The Ray score would be much higher
for distributed compute, but lower for durable cross-tool orchestration.

## Capability matrix

| Capability | Argo Workflows | Ray + KubeRay |
| --- | --- | --- |
| DAG-style multi-step pipelines | Native | None for durable workflows; Ray task graphs are native but not a workflow engine |
| Heterogeneous container per step | Native | None |
| Per-step GPU resource requests | Native | Native logical resources |
| Step retry/backoff policies | Native | Native basic retries; weaker workflow backoff |
| Conditional branching | Native | Native Python control flow |
| Parallel fan-out / fan-in | Native | Native |
| Long-running steps | Native | Native |
| Suspend / resume workflows | Native | None |
| Workflow templates / parameterization | Native | Plugin/config |
| Artifact passing, S3-backed | Native | Plugin/library-level |
| Secret/credential injection | Native via Kubernetes | Plugin via Kubernetes/Anyscale |
| Distributed compute within a step | Plugin/external | Native |
| Python SDK authoring | Plugin via Hera | Native |
| Monitoring/UI | Native | Native |
| Backfill / scheduled runs | Native | Plugin via RayCronJob/job scheduling |
| Reproducibility / versioning | Plugin around native specs/archive | Plugin |

## Architecture recommendation

The Workbench Composition layer should own Argo WorkflowTemplates as the
customer-facing workflow contract. Each stage should be a containerized tool
step or a resource step that creates/runs a subordinate system.

Ray should appear as an implementation choice for a step, not as the outer
workflow model. Examples:

- Argo step launches a LeRobot distributed training job backed by Ray Train or
  tool-native Accelerate/DDP.
- Argo step launches a Genesis or Isaac Lab RL rollout/training phase backed by
  Ray/RLlib when the workload benefits from actors and distributed rollouts.
- Argo step creates or targets a KubeRay RayCluster for a bounded job, or
  delegates to Anyscale on Nebius for managed Ray.
- Argo remains responsible for cross-tool order, retries, artifacts, metadata,
  user-visible status, and cleanup.

This keeps the product understandable: Argo is the workflow plane, Ray is the
distributed compute plane.

## Nebius-Anyscale leverage

The Anyscale partnership should be a first-class integration path:

- Add a Ray step type in Workbench workflow templates after the initial Argo
  install path exists.
- Support two Ray backends: self-managed KubeRay on MK8s and managed Anyscale
  on Nebius where commercial/customer context warrants it.
- In customer materials, say Workbench composes physical-AI workflows on
  Kubernetes and can use Anyscale/Ray for distributed steps.

Avoid implying that Ray replaces Argo for workflow orchestration. The stronger
message is that Nebius can offer both a standard Kubernetes workflow plane and
a partner-backed managed Ray scale-out plane.

## Partner orientation

The public partner scan did not find partners shipping Argo-bound or Ray-bound
cross-tool reference architectures. Most are engine-agnostic:

| Partner | Orientation |
| --- | --- |
| Voxel51 / FiftyOne | Kubernetes/Helm service and SDK/data platform |
| Encord | SaaS/API/SDK data and annotation workflows |
| Bifrost | Terraform/Kubernetes service deployment |
| MetAI | NVIDIA Omniverse/Isaac-centered simulation data |
| Genesis | Simulation/RL platform; Ray-adjacent in RoboGen, not workflow-bound |
| Lightwheel | Data collection/simulation outputs compatible with training workflows |
| Foxglove | Robotics data platform, hosted or customer-hosted Kubernetes services |
| Rerun | SDK/viewer/data stack for multimodal robotics data |
| Antioch | Insufficient credible public evidence in this pass |

Partner implication: use Argo to compose containers, services, and APIs without
forcing each partner into a Ray programming model.

## Cost of being wrong

If Hybrid is wrong because two substrates are too much, the escape path is
manageable. Keep Argo workflows as the stable outer contract and make Ray steps
optional. A Ray-backed step can be replaced with a plain container, PyTorch DDP,
or a tool-native launcher without rewriting the whole workflow.

If Hybrid is wrong because Ray becomes the dominant orchestrator again, the cost
is a larger migration. Workbench would need to translate Argo DAGs and
WorkflowTemplates into a Ray/Anyscale-native application model. That would be
justified only if Ray gains a stable durable workflow engine again and partners
start shipping Ray-native workflow references.

If Hybrid is wrong because Argo becomes insufficient at scale, the likely fix is
not Ray-only. The more likely fix is queueing/scheduling and controller scaling:
Kueue, Run:ai/KAI Scheduler, workflow archival hygiene, artifact repository
tuning, and namespace quotas.

## What would have changed the verdict

The verdict would have moved toward Ray-only if all of these were true:

- Ray Workflows or a successor was GA/stable, actively developed, and supported
  in current Ray releases.
- It supported durable DAGs, suspend/resume, scheduled/backfill runs,
  artifact passing, retries/backoff, status/history, and operational UI/API.
- It handled heterogeneous container-per-step workflows without contorting
  every partner tool into a Python Ray task.
- A meaningful share of Workbench partners shipped Ray-native reference
  architectures.

The opposite is true today: Ray Workflows was removed, while partner tools are
mostly container/service/API oriented.

## Concrete next-step diff

Proceed with W8-argo-install. Do not rename it to W8-ray-install.

After Argo installation:

1. Add a minimal Workbench `WorkflowTemplate` for one real composition path,
   for example FiftyOne curation -> LeRobot train -> Rerun/FiftyOne eval.
2. Define a Ray step contract in docs before implementing it:
   input artifact paths, output artifact paths, image/runtime env, GPU
   requirements, logs, metrics, cleanup behavior, and backend selection
   (`kuberay` or `anyscale`).
3. Schedule a separate W8-kuberay-or-anyscale-spike only after the first Argo
   workflow skeleton is running.
4. Update strategy language to: "Argo Workflows orchestrates Workbench
   compositions; Ray/KubeRay/Anyscale distributes compute inside selected
   steps."

## Sources

- Argo overview: https://argoproj.github.io/workflows/
- Argo DAGs: https://argo-workflows.readthedocs.io/en/latest/walk-through/dag/
- Argo conditionals: https://argo-workflows.readthedocs.io/en/latest/walk-through/conditionals/
- Argo loops/fan-out: https://argo-workflows.readthedocs.io/en/latest/walk-through/loops/
- Argo retries: https://argo-workflows.readthedocs.io/en/latest/retries/
- Argo suspend/resume: https://argo-workflows.readthedocs.io/en/latest/walk-through/suspending/
- Argo artifacts: https://argo-workflows.readthedocs.io/en/latest/configure-artifact-repository/
- Argo templates: https://argo-workflows.readthedocs.io/en/latest/workflow-templates/
- Argo managed namespaces: https://argo-workflows.readthedocs.io/en/latest/managed-namespace/
- CNCF Argo graduation: https://www.cncf.io/announcements/2022/12/06/the-cloud-native-computing-foundation-announces-argo-has-graduated/
- Argo Helm chart: https://artifacthub.io/packages/helm/argo/argo-workflows
- Hera Python SDK: https://hera.readthedocs.io/
- Ray Workflows alpha API page: https://docs.ray.io/en/latest/workflows/api/doc/ray.workflow.run.html
- Ray Workflows deprecation discussion: https://discuss.ray.io/t/ray-workflows-deprecated/22132
- Ray 2.48 removal notes: https://newreleases.io/project/github/ray-project/ray/release/ray-2.48.0
- Ray tasks: https://docs.ray.io/en/latest/ray-core/tasks.html
- Ray resources: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
- Ray on Kubernetes: https://docs.ray.io/en/latest/cluster/kubernetes/
- KubeRay Helm: https://ray-project.github.io/kuberay/deploy/helm/
- Ray Train and Hugging Face: https://docs.ray.io/en/latest/train/getting-started-transformers.html
- Ray Tune: https://docs.ray.io/en/latest/tune/tutorials/tune-distributed.html
- Ray Serve: https://docs.ray.io/en/latest/serve/index.html
- Ray Data: https://docs.ray.io/en/latest/data/data.html
- RLlib: https://docs.ray.io/en/master/rllib/index.html
- Nebius-Anyscale announcement: https://nebius.com/blog/posts/anyscale-partnership
- Nebius third-party integrations: https://docs.nebius.com/3p-integrations
- Anyscale on Nebius docs: https://docs.anyscale.com/k8s/nebius/
- Nebius partner program: https://nebius.com/nebius-partner-program
- NeMo Kubernetes/Argo playbook: https://docs.nvidia.com/nemo-framework/user-guide/24.07/playbooks/kubernetes.html
- LeRobot multi-GPU: https://huggingface.co/docs/lerobot/main/multi_gpu_training
- FiftyOne Helm: https://helm.fiftyone.ai/
- Encord SDK: https://docs.encord.com/sdk-documentation/getting-started-sdk/sdk-intro
- Bifrost Kubernetes deployment: https://docs.getbifrost.ai/deployment-guides/k8s
- MetAI solutions: https://www.met-ai.com/solutions/
- Lightwheel platform: https://www.lightwheel.ai/lightwheel-platform
- Foxglove primary sites: https://docs.foxglove.dev/docs/primary-sites/introduction
- Rerun docs: https://docs.rerun.io/dev/getting-started/
- NVIDIA NIM Kubernetes deployment: https://docs.nvidia.com/nim/large-language-models/2.0.0/deployment/kubernetes-deployment/index.html
- NVIDIA Run:ai scheduler docs: https://run-ai-docs.nvidia.com/saas/platform-management/runai-scheduler/scheduling/default-scheduler

