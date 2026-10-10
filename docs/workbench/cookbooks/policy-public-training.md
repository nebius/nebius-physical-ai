# Public policy training and deployed simulation proof

[One workflow YAML](../../../workflows/testing/policy-public-training.yaml)
executes public SmolVLA continued training and deployment through the standard
Workbench runtime. Every stage is discoverable in the tool catalog; the agent can
validate, plan, submit, inspect artifacts, resume and cancel using the existing
workflow surfaces. No customer data or private training scripts are needed.

Run on a Linux operator with Workbench installed, an authorized Nebius managed
Kubernetes context, configured private S3 storage and two available RTX PRO 6000
GPU workers. Training uses one GPU with 16 CPUs and 64 GiB; deployment uses two
separate GPU workers with 4 CPUs and 24 GiB each. CPU stages use 4 CPUs and 16 GiB.
The immutable published images are checked through the standard image preflight.
Public HF snapshots are revision-pinned and fetched without an HF credential.

```bash
npa workbench health preflight --checks s3,nebius --project <project> --json
npa skypilot verify --cluster <context> --kubeconfig <kubeconfig>
npa workbench workflow validate-spec workflows/testing/policy-public-training.yaml --json
npa workbench workflow submit workflows/testing/policy-public-training.yaml \
  --project <project> --infra k8s/<context> --runtime --stage-src \
  --var bucket=<private-bucket> \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Select the exact kubeconfig with `KUBECONFIG` before submission. Secret names
resolve from the selected project's configuration or the process environment;
never put their values in YAML. The runtime persists each wave, provider job
identity, gate decision and artifact under the run prefix. For interruption,
use the same checkout/source selection, project/context, storage and isolated
runtime state with `--resume --resume-run <run-id>`. Training additionally saves
complete native model, optimizer, scheduler, RNG and sampler recovery state to
S3 during execution. Failed gates resume their preceding native candidate and
add another epoch increment. There is no implicit iteration or cost cap; cancel
with `npa workbench workflow cancel <run-id> --project <project> --json`.

The recipe in YAML controls epochs, batch size, seed, workers, evaluation trials
and the success threshold. Change it before starting a new run. Three disjoint
LIBERO initial-state sets require at most 16 trials in each set. The standard
runtime workflow identity is sealed into the recipe and final proof.

| Stage | Real computation and proof |
| --- | --- |
| Prepare | Fetch pinned `lerobot/libero`; audit trajectories and decode every episode preview. |
| Curate | Real FiftyOne measures brightness/sharpness and selects episodes; quality flags and the complete measurements are retained. This is preview-frame quality, without an object detector or an in-house review claim. |
| Split | Trajectory hashes group duplicates within each task; selected episodes become train, validation and test partitions. Training normalization is fitted using training frames only. |
| Pretrain | Native LeRobot SmolVLA via torchrun performs continued multi-task training. Actual optimizer loss, gradient norms, changed weights and durable checkpoints are retained. |
| Evaluate / first gate | Measure policy loss over all validation demonstration frames and native simulation completion on initial states starting at zero. Failure returns to pretraining. |
| Fine-tune | Start from the first promoted checkpoint; train only the chosen task's training episodes. |
| Evaluate / second gate | Measure loss over the reserved task test frames and native simulation completion on the next reset set. Failure returns to fine-tuning. |
| Export | Require both promotion records and export the exact final specialist with portable model/tokenizer dependencies. |
| Serve | One GPU runs authenticated HTTP SmolVLA inference; a distinct GPU worker renders native LIBERO while applying returned actions in MuJoCo CPU physics. This third reset set is untouched by model selection. |
| Report | Reconcile every served/applied action, fully decode every episode video and create `reports/demo.html`, `demo.mp4`, `proof.json` and checksums. Deployment qualification failure preserves the measured proof and fails the workflow. |

The HTML embeds footage, all deployment trials including failures, actual action
vectors, measured responses, fresh model chunk calls, native optimizer curves
and stage identities. It opens offline with no external libraries or network
requests. Only HTML, MP4 and sanitized proof JSON are public-safe; raw worker
artifacts, S3 locations, logs and the authenticated rendezvous remain private.

Collect the private run's `reports/` and `serving/{server,client}/` directories
through the configured artifact transport, then verify the same-run evidence:

```bash
NPA_POLICY_PUBLIC_RESULTS=<collected-run-directory> \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_policy_public_live.py -q
```

This implements the two training/gate loops and deployed benchmark-container
pattern as a public executable baseline. It does not qualify in-house foundation
pretraining, weighted customer data soup, physical robot transfer or concurrent
client batching. It uses managed Kubernetes and torchrun, **not Slurm**. The
separate [Slurm workflow](policy-training-slurm.md) supports operator training
scripts and Soperator access. The SDG pipeline remains separate.

Runtime configuration: `NPA_WORKFLOW_SHA256` is supplied by the renderer;
`NPA_VLA_RECOVERY_URI`, `NPA_VLA_RECIPE_SHA256` and
`NPA_VLA_SELECTION_SHA256` are supplied by the training stage, not operator
settings. GPU stages require CUDA and NVIDIA EGL. Native package installation
and stage errors retain private diagnostics under the stage's diagnostics prefix.
