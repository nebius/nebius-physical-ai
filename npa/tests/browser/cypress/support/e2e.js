const WORKFLOW_YAML = [
  "apiVersion: npa.workflow/v0.0.1",
  "kind: Workflow",
  "metadata:",
  "  name: cypress-sim2real",
  "spec:",
  "  states:",
  "    - id: draft",
  "      toolRef: workbench.sim2real.status",
  "      description: mocked browser workflow state",
].join("\n");

const COMPLEX_WORKFLOW_YAML = [
  "apiVersion: npa.workflow/v0.0.1",
  "kind: Workflow",
  "metadata:",
  "  name: cypress-vlm-rl-loop",
  "spec:",
  "  states:",
  "    - id: rollout",
  "      toolRef: workbench.sim2real.policy_rollout",
  "      description: Roll out a policy on non-stock customer assets.",
  "      outputs:",
  "        rrd_uri: s3://mock/non-stock-customer-run/reports/sim2real.rrd",
  "    - id: vlm_gate",
  "      toolRef: workbench.token_factory.reason",
  "      description: Score rollout quality and decide whether to promote.",
  "      transitions:",
  "        promote_checkpoint: finalize",
  "        loop_back: rollout",
  "    - id: finalize",
  "      toolRef: workbench.sim2real.status",
  "      description: Publish run-specific report and Rerun recording.",
].join("\n");

const GENERIC_WORKFLOW_YAML = [
  "apiVersion: npa.workflow/v0.0.1",
  "kind: Workflow",
  "metadata:",
  "  name: cypress-cosmos-reason",
  "spec:",
  "  states:",
  "    - id: fetch_checkpoint",
  "      toolRef: workbench.cosmos.fetch",
  "      description: Fetch Cosmos checkpoint assets.",
  "    - id: reason",
  "      toolRef: workbench.token_factory.reason",
  "      description: Run Token Factory reasoning over staged inputs.",
  "    - id: publish",
  "      toolRef: workbench.artifacts.upload",
  "      description: Publish reasoning artifacts.",
].join("\n");

const GENERIC_WORKFLOW_RUN_DETAILS = {
  run: {
    run_id: "cosmos-reason-run",
    status: "running",
    result: "pending",
    updated_at: "2026-07-11T00:40:00Z",
    stages: [
      { id: "fetch_checkpoint", label: "Fetch checkpoint", status: "succeeded", summary: "Cosmos checkpoint staged." },
      { id: "reason", label: "Reason", status: "running", summary: "Token Factory reasoning in progress." },
      { id: "publish", label: "Publish", status: "pending", summary: "Waiting for reasoning outputs." },
    ],
    logs: [{ timestamp: "2026-07-11T00:40:00Z", level: "info", message: "generic workflow stages active" }],
  },
};

const RERUN_RECORDING_PATH = `/rerun/recordings/cap-${"A".repeat(43)}.rrd`;

const SIM_VIZ = {
  run_id: "mock-run",
  active_run_id: "mock-run",
  stage: "demo",
  camera: "workspace",
  rrd_uri: "file:///opt/npa-agent/sim2real.rrd",
  rrd_updated_at: "2026-07-07T03:33:00Z",
  rerun_ready: true,
  rerun_iframe_url: `/rerun/?url=https://example.test${RERUN_RECORDING_PATH}&hide_welcome_screen=1&camera=workspace`,
  mcap_uri: "file:///opt/npa-agent/recordings/sim2real.mcap",
  lichtblick_ready: true,
  lichtblick_iframe_url: "/lichtblick/?ds=remote-file&ds.url=%2Flichtblick%2Frecordings%2Fsim2real.mcap",
  // Intentionally not alphabetical — UI must keep latest-first order.
  available_run_ids: ["submitted-run", "mock-run"],
  available_runs: [
    { run_id: "franka-demo", last_modified: "2026-07-09T12:00:00Z", stage: "demo", source_type: "local_demo", source_label: "Local demo" },
    { run_id: "cosmos-reason-run", last_modified: "2026-07-08T13:00:00Z", stage: "running", source_type: "workflow_history", source_label: "Workflow history" },
    { run_id: "submitted-run", last_modified: "2026-07-08T12:00:00Z", stage: "submitted", source_type: "workflow_history", source_label: "Workflow history" },
    { run_id: "mock-run", last_modified: "2026-07-07T03:33:00Z", stage: "demo", source_type: "workflow_history", source_label: "Workflow history" },
  ],
};

const NON_STOCK_RUN_ID = "non-stock-customer-run";

const NON_STOCK_SIM_VIZ = {
  run_id: NON_STOCK_RUN_ID,
  active_run_id: NON_STOCK_RUN_ID,
  stage: "stage_14_rerun_viz",
  camera: "customer-overhead",
  rrd_uri: "file:///opt/npa-agent/recordings/sim2real.rrd",
  rrd_updated_at: "2026-07-07T04:12:00Z",
  rerun_ready: true,
  rerun_iframe_url: `/rerun/?url=https://example.test${RERUN_RECORDING_PATH}&hide_welcome_screen=1&camera=customer-overhead`,
  available_run_ids: [NON_STOCK_RUN_ID, "mock-run", "submitted-run"],
  available_runs: [
    { run_id: NON_STOCK_RUN_ID, last_modified: "2026-07-11T18:00:00Z", stage: "stage_14_rerun_viz", source_type: "workflow_history", source_label: "Workflow history" },
    { run_id: "franka-demo", last_modified: "2026-07-09T12:00:00Z", stage: "demo", source_type: "local_demo", source_label: "Local demo" },
    { run_id: "cosmos-reason-run", last_modified: "2026-07-08T13:00:00Z", stage: "running", source_type: "workflow_history", source_label: "Workflow history" },
    { run_id: "submitted-run", last_modified: "2026-07-08T12:00:00Z", stage: "submitted", source_type: "workflow_history", source_label: "Workflow history" },
    { run_id: "mock-run", last_modified: "2026-07-07T03:33:00Z", stage: "demo", source_type: "workflow_history", source_label: "Workflow history" },
  ],
  artifact_render: "rerun",
  artifact_key: `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`,
  artifact_uri: `s3://mock/${NON_STOCK_RUN_ID}/reports/sim2real.rrd`,
  artifact_preview_url: RERUN_RECORDING_PATH,
  artifact_download_url: RERUN_RECORDING_PATH,
  mcap_uri: `file:///opt/npa-agent/recordings/sim2real.mcap`,
  lichtblick_ready: true,
  lichtblick_iframe_url: "/lichtblick/?ds=remote-file&ds.url=%2Flichtblick%2Frecordings%2Fsim2real.mcap",
};

const RUN_DETAILS = {
  run: {
    run_id: "mock-run",
    status: "running",
    result: "pending",
    updated_at: "2026-07-07T03:33:00Z",
    stages: [
      { id: "select_assets", label: "Select assets", status: "succeeded", summary: "Stock Franka selected" },
      { id: "render", label: "Render", status: "running", summary: "Rerun recording available" },
    ],
    logs: [{ timestamp: "2026-07-07T03:33:00Z", level: "info", message: "mock run log" }],
  },
};

const NON_STOCK_RUN_DETAILS = {
  run: {
    run_id: NON_STOCK_RUN_ID,
    status: "completed",
    result: "promoted",
    updated_at: "2026-07-07T04:12:00Z",
    stages: [
      { id: "stage_02_assets", label: "Customer assets", status: "succeeded", summary: "Loaded BYO scene mesh and custom robot." },
      { id: "stage_10_eval_heldout", label: "Heldout eval", status: "succeeded", summary: "Non-stock heldout rollout passed." },
      { id: "stage_14_rerun_viz", label: "Rerun viz", status: "succeeded", summary: "Run-specific Rerun recording published." },
    ],
    logs: [
      { timestamp: "2026-07-07T04:10:00Z", level: "info", message: "loaded customer scene mesh" },
      { timestamp: "2026-07-07T04:12:00Z", level: "info", message: "published non-stock sim2real artifacts" },
    ],
  },
};

const CAMERAS = {
  selected: ["workspace"],
  cameras: [
    {
      name: "workspace",
      placement: "stock_workspace",
      fov: 60,
      pos: [1, 2, 3],
      look_at: [0, 0, 0],
      resolution: [640, 480],
    },
    {
      name: "wrist",
      placement: "stock_wrist",
      fov: 70,
      pos: [0.1, 0.2, 0.3],
      look_at: [0, 0, 0.1],
      resolution: [640, 480],
    },
  ],
};

const ASSETS = {
  scene_spec: { uri: "stock://scene/default" },
  robot_spec: { uri: "stock://robot/franka" },
  camera_spec: { uri: "stock://cameras/default" },
  selection: {
    scene_spec_uri: "stock://scene/default",
    robot_spec_uri: "stock://robot/franka",
    cameras_uri: "stock://cameras/default",
    robot_preset: "franka",
    sim_backend: "isaac",
    props: ["cube"],
  },
  resolved_uris: {
    scene_spec_uri: "stock://scene/default",
    robot_spec_uri: "stock://robot/franka",
    cameras_uri: "stock://cameras/default",
  },
};

const NON_STOCK_ARTIFACTS = [
  {
    key: `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`,
    s3_uri: `s3://mock/${NON_STOCK_RUN_ID}/reports/sim2real.rrd`,
    render: "rerun",
    inline: true,
    size: 8192,
  },
  {
    key: `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`,
    s3_uri: `s3://mock/${NON_STOCK_RUN_ID}/reports/sim2real.mcap`,
    render: "mcap",
    inline: true,
    size: 16384,
  },
  {
    key: `${NON_STOCK_RUN_ID}/recordings/native-single-camera.mcap`,
    s3_uri: `s3://mock/${NON_STOCK_RUN_ID}/recordings/native-single-camera.mcap`,
    render: "mcap",
    inline: true,
    size: 4096,
  },
  {
    key: `${NON_STOCK_RUN_ID}/rollouts/customer-camera.mp4`,
    s3_uri: `s3://mock/${NON_STOCK_RUN_ID}/rollouts/customer-camera.mp4`,
    render: "video",
    inline: true,
    size: 4096,
  },
  {
    key: `${NON_STOCK_RUN_ID}/reports/sim2real-report.json`,
    s3_uri: `s3://mock/${NON_STOCK_RUN_ID}/reports/sim2real-report.json`,
    render: "json",
    inline: true,
    size: 2048,
  },
  {
    key: `${NON_STOCK_RUN_ID}/logs/orchestrator.log`,
    s3_uri: `s3://mock/${NON_STOCK_RUN_ID}/logs/orchestrator.log`,
    render: "text",
    inline: true,
    size: 1024,
  },
  {
    key: `${NON_STOCK_RUN_ID}/raw/custom-dynamics.fooz`,
    s3_uri: `s3://mock/${NON_STOCK_RUN_ID}/raw/custom-dynamics.fooz`,
    render: "download",
    inline: false,
    size: 512,
  },
];

const JSON_ONLY_RUN_ID = "json-only-storage-run";
const JSON_ONLY_ARTIFACTS = [
  {
    key: `${JSON_ONLY_RUN_ID}/evaluation/aggregate.json`,
    s3_uri: `s3://project-artifacts/${JSON_ONLY_RUN_ID}/evaluation/aggregate.json`,
    render: "json",
    inline: true,
    size: 768,
  },
  {
    key: `${JSON_ONLY_RUN_ID}/checkpoints/policy.ckpt`,
    s3_uri: `s3://project-artifacts/${JSON_ONLY_RUN_ID}/checkpoints/policy.ckpt`,
    render: "download",
    inline: false,
    size: 4096,
  },
];

const ARTIFACT_ONLY_RUN_ID = "artifact-observation-run";
const ARTIFACT_ONLY_ARTIFACTS = [
  ["capture", "frame.png", "image"],
  ["capture", "metadata.json", "json"],
  ["dataset", "records.parquet", "download"],
  ["training", "checkpoint.bin", "download"],
  ["evaluation", "metrics.json", "json"],
  ["visualization", "preview.rrd", "rerun"],
  ["novel-layout", "future-format.xyz", "download"],
].map(([stage, name, render]) => ({
  key: `tenant-runs/${ARTIFACT_ONLY_RUN_ID}/${stage}/${name}`,
  s3_uri: `s3://project-artifacts/tenant-runs/${ARTIFACT_ONLY_RUN_ID}/${stage}/${name}`,
  render,
  inline: render !== "download",
  size: 512,
}));

const ARTIFACT_ONLY_RUN_DETAILS = {
  run: {
    run_id: ARTIFACT_ONLY_RUN_ID,
    source_type: "artifact_storage",
    source_label: "S3 artifacts",
    project_id: "project-a",
    bucket: "project-artifacts",
    status: "status_unavailable",
    status_label: "Status unavailable",
    result: "artifacts_available",
    updated_at: "2026-08-07T00:00:00Z",
    stages: [...new Set(ARTIFACT_ONLY_ARTIFACTS.map((item) => item.key.split("/")[2]))].map((stage) => {
      const count = ARTIFACT_ONLY_ARTIFACTS.filter((item) => item.key.split("/")[2] === stage).length;
      const reason = `${count} artifact${count === 1 ? " was" : "s were"} observed; execution status is unavailable.`;
      return {
        evidence_version: "npa.stage-evidence/v1",
        id: stage,
        stage_key: stage,
        label: stage.replaceAll("-", " "),
        status: "observed_output",
        status_label: "Observed output",
        artifact_count: count,
        evidence_type: "artifact_observation",
        evidence_source: "artifact_listing",
        authority: "observed",
        confidence: "high",
        diagnostic_reason: reason,
        evidence: { type: "artifact_observation", source: "artifact_listing", authority: "observed", confidence: "high", reason },
        summary: reason,
      };
    }),
    stage_summary: {
      evidence_version: "npa.stage-evidence/v1",
      text: "6 observed groups · execution status unavailable",
      displayed_stage_count: 6,
      observed_stage_count: 6,
      authoritative_stage_count: 0,
      execution_status_available: false,
      succeeded_count: 0,
      failed_count: 0,
      not_run_count: 0,
    },
    logs: [{ timestamp: "2026-08-07T00:00:00Z", level: "info", message: "Artifact observations only." }],
  },
};

// A Physical AI Data Factory run whose artifacts span every pipeline stage and
// whose augment is a REAL Cosmos Transfer 2.5 GPU render — used to exercise the
// per-stage provenance panel (counts, click-to-filter, honest engine banner).
const DF_MOCK_RUN_ID = "paidf-mock-gpu-run";
const DF_MOCK_ARTIFACTS = [
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/input/video_0.mp4`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/input/video_0.mp4`, render: "video", size: 4096 },
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/input/frame_00.png`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/input/frame_00.png`, render: "image", size: 2048 },
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/input/frame_01.png`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/input/frame_01.png`, render: "image", size: 2048 },
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/configs/manifest.json`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/configs/manifest.json`, render: "json", size: 512 },
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/cosmos_augmented/aug0/augmented_video.mp4`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/cosmos_augmented/aug0/augmented_video.mp4`, render: "video", size: 8192 },
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/cosmos_augmented/aug0/metadata.json`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/cosmos_augmented/aug0/metadata.json`, render: "json", size: 256 },
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/curation/report.json`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/curation/report.json`, render: "json", size: 256 },
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/reports/sim2real.rrd`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/reports/sim2real.rrd`, render: "rerun", size: 8192 },
];
const DF_MOCK_PROVENANCE = {
  ok: true,
  run_id: DF_MOCK_RUN_ID,
  components: [
    { stage: "Config generation", stage_key: "configs", component: "Appearance-variable sampler", runtime: "CPU", artifact_count: 1 },
    { stage: "Source frames", stage_key: "input", component: "Uploaded source clips", runtime: "input", artifact_count: 3 },
    { stage: "Augment", stage_key: "cosmos_augmented", component: "Cosmos Transfer 2.5", runtime: "GPU (Nebius K8s)", artifact_count: 2, engine: "cosmos_transfer_2.5_gpu", detail: "real Cosmos Transfer 2.5 diffusion on GPU", model: "nvidia/Cosmos-Transfer2.5-2B" },
    { stage: "Curation", stage_key: "curation", component: "FiftyOne-style curation report", runtime: "CPU", artifact_count: 1 },
    { stage: "Visualize + finalize", stage_key: "reports", component: "Rerun recording + aggregate report", runtime: "CPU", artifact_count: 1 },
  ],
  summary: "mock data-factory provenance",
  origin: { original_present: true, original_inputs: [{ key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/input/video_0.mp4`, stage: "input", kind: "video" }], summary: "mock origin" },
};

// A Data Factory run that has only raw input (augment never produced output) —
// the panel must warn so a raw input clip is not mistaken for a result.
const DF_INPUT_ONLY_RUN_ID = "paidf-mock-input-only";
const DF_INPUT_ONLY_ARTIFACTS = [
  { key: `physical-ai-data-factory/${DF_INPUT_ONLY_RUN_ID}/input/video_0.mp4`, s3_uri: `s3://mock/${DF_INPUT_ONLY_RUN_ID}/input/video_0.mp4`, render: "video", size: 4096 },
  { key: `physical-ai-data-factory/${DF_INPUT_ONLY_RUN_ID}/configs/manifest.json`, s3_uri: `s3://mock/${DF_INPUT_ONLY_RUN_ID}/configs/manifest.json`, render: "json", size: 512 },
];

const WORKFLOW_VALIDATION = {
  ok: true,
  status: "valid",
  name: "cypress-sim2real",
  states: ["draft"],
};

const COMPLEX_WORKFLOW_VALIDATION = {
  ok: true,
  status: "valid",
  name: "cypress-vlm-rl-loop",
  states: ["rollout", "vlm_gate", "finalize"],
};

const CHAT_SESSIONS = [
  { id: "default", title: "Default chat", message_count: 0 },
  { id: "session-two", title: "Second session", message_count: 2 },
];

const STATIC_BUTTON_IDS = [
  "mobilePanelsToggle",
  "newChatSession",
  "mobileChatAuthBtn",
  "chatSend",
  "chatActionS3",
  "chatActionCosmos",
  "chatActionWatch",
  "chatActionWorkflow",
  "workflowUpload",
  "workflowValidate",
  "workflowPlan",
  "workflowSubmitYaml",
  "workflowStatus",
  "loadRunData",
  "artifactRefreshRuns",
  "artifactLoadRunArtifacts",
  "openRerun",
  "loadRerunViewer",
  "downloadMcap",
  "foxgloveOpenWeb",
  "describeVisual",
];

const FIELD_IDS = [
  "agentAccessProjectSelect",
  "agentAccessBucketSelect",
  "chatSessionSelect",
  "chatModel",
  "chatLog",
  "chatForm",
  "chatInput",
  "workflowName",
  "workflowValidation",
  "workflowStates",
  "workflowYaml",
  "workflowPlanOutput",
  "runSummary",
  "stageList",
  "runLog",
  "stagesPanel",
  // Selection / Scene-mode controls were removed from the UI.
  "runIdInput",
  "runIdSelect",
  "artifactTypeFilter",
  "artifactSort",
  "artifactStageFilter",
  "runsArtifactsPanel",
  "artifactRunSummary",
  "artifactList",
  "simRunId",
  "simStage",
  "simCamera",
  "renderedDataSummary",
  "rerunFrame",
  "lichtblickFrame",
  "renderModeLichtblick",
  "viewerPaneLichtblick",
  "artifactPreviewHost",
  "tabMain",
  "tabRerun",
  "panelChat",
  "panelRerun",
  "statusBar",
  "toastHost",
];

function json(body) {
  return {
    statusCode: 200,
    headers: { "content-type": "application/json" },
    body,
  };
}

function renderForArtifactKey(key) {
  const artifact = NON_STOCK_ARTIFACTS.find((item) => item.key === key);
  if (artifact) {
    return artifact.render;
  }
  if (String(key || "").endsWith(".rrd")) return "rerun";
  if (String(key || "").endsWith(".mcap")) return "mcap";
  if (String(key || "").match(/\.(mp4|webm|mov)$/)) return "video";
  if (String(key || "").match(/\.(png|jpg|jpeg|gif|webp)$/)) return "image";
  if (String(key || "").endsWith(".json")) return "json";
  if (String(key || "").match(/\.(txt|log|csv|yaml|yml|md)$/)) return "text";
  return "download";
}

function simVizForArtifact(key) {
  const render = renderForArtifactKey(key);
  const isNonStock = String(key || "").startsWith(`${NON_STOCK_RUN_ID}/`);
  const isJsonOnly = String(key || "").startsWith(`${JSON_ONLY_RUN_ID}/`);
  const base = isNonStock
    ? NON_STOCK_SIM_VIZ
    : (isJsonOnly
      ? { ...SIM_VIZ, run_id: JSON_ONLY_RUN_ID, active_run_id: JSON_ONLY_RUN_ID }
      : SIM_VIZ);
  const previewPath = `/api/artifacts/file/${encodeURIComponent(key.replaceAll("/", "__"))}`;
  if (render === "rerun") {
    return { ...base, artifact_render: render, artifact_key: key, artifact_uri: `s3://mock/${key}` };
  }
  if (render === "mcap") {
    return {
      ...base,
      rrd_uri: "",
      rerun_ready: false,
      rerun_iframe_url: "/rerun/",
      artifact_render: render,
      artifact_key: key,
      artifact_uri: `s3://mock/${key}`,
      mcap_uri: "file:///opt/npa-agent/recordings/sim2real.mcap",
      lichtblick_ready: true,
      lichtblick_iframe_url: "/lichtblick/?ds=remote-file&ds.url=%2Flichtblick%2Frecordings%2Fsim2real.mcap",
      artifact_preview_url: "/lichtblick/recordings/sim2real.mcap",
      artifact_download_url: "/lichtblick/recordings/sim2real.mcap",
    };
  }
  return {
    ...base,
    source_type: "artifact_storage",
    source_label: "S3 artifacts",
    rrd_uri: "",
    rerun_ready: false,
    rerun_iframe_url: "/rerun/",
    artifact_render: render,
    artifact_key: key,
    artifact_uri: `s3://mock/${key}`,
    artifact_preview_url: previewPath,
    artifact_download_url: previewPath,
  };
}

function installAgentApiMocks() {
  let activeSimViz = SIM_VIZ;
  const deployment = {
    deployment_id: "npa-agent-mocked-wan",
    deployment_name: "wan-pr261",
    project_alias: "mock-project",
    runtime_namespace: "mock-project/wan-pr261",
    repository: "nebius/nebius-physical-ai",
    branch: "codex/wan-pr261",
    commit: "0123456789abcdef0123456789abcdef01234567",
    short_commit: "0123456789ab",
    workspace_label: "Wan Workbench",
    bootstrap_timestamp: "2026-08-10T00:00:00Z",
  };
  cy.intercept("GET", "/api/health", json({ ok: true, tool_refs: 19, deployment })).as("health");
  cy.intercept("GET", "/api/access*", json({
    apiVersion: "npa.agent.access/v1",
    identity: {
      tenant_id: "tenant-test",
      deployment_project_id: "project-a",
      deployment_project_name: "Project Alpha",
    },
    status: "partial",
    scope: "partial_tenant",
    capabilities: {},
    projects: [
      {
        id: "project-a",
        name: "Project Alpha",
        deployment_project: true,
        status: "available",
        capabilities: {
          artifact_discovery: { status: "available", reason: "Readable object storage is available." },
          workflow_submission: { status: "available", reason: "Deployment project only." },
        },
        resources: [
          {
            type: "object_storage_bucket",
            id: "resource-a",
            name: "project-artifacts",
            project_id: "project-a",
            capabilities: {
              artifact_discovery: { status: "available", reason: "Object listing was verified.", scope: "read_only" },
              artifact_read: { status: "available", reason: "Object reads were verified.", scope: "read_only" },
            },
          },
          {
            type: "object_storage_bucket",
            id: "resource-archive",
            name: "archive-artifacts",
            project_id: "project-a",
            capabilities: {
              artifact_discovery: { status: "available", reason: "Object listing was verified.", scope: "read_only" },
              artifact_read: { status: "available", reason: "Object reads were verified.", scope: "read_only" },
            },
          },
          {
            type: "object_storage_bucket",
            id: "resource-denied",
            name: "denied-artifacts",
            project_id: "project-a",
            capabilities: {
              artifact_discovery: { status: "denied", reason: "Permission denied while listing objects.", scope: "read_only" },
              artifact_read: { status: "denied", reason: "Permission denied while reading objects.", scope: "read_only" },
            },
          },
          {
            type: "object_storage_bucket",
            id: "resource-unavailable",
            name: "unavailable-artifacts",
            project_id: "project-a",
            capabilities: {
              artifact_discovery: { status: "unavailable", reason: "The object service is unavailable.", scope: "read_only" },
              artifact_read: { status: "unavailable", reason: "Object reads could not be verified.", scope: "read_only" },
            },
          },
        ],
      },
    ],
    errors: [],
  })).as("agentAccess");
  cy.intercept("GET", "/api/models", json({
    ok: true,
    model: "nvidia/Cosmos3-Super-Reasoner",
    models: ["nvidia/Cosmos3-Super-Reasoner", "mock/model"],
  })).as("models");
  cy.intercept("GET", "/api/session", json({
    deployment,
    selection: ASSETS.selection,
    sim_viz: SIM_VIZ,
    latest_submit: { run_id: "mock-run" },
    camera_selection: ["workspace"],
    chat_history: [],
    active_chat_session_id: "default",
    chat_sessions: CHAT_SESSIONS,
    llm: {
      model: "nvidia/Cosmos3-Super-Reasoner",
      default_model: "nvidia/Cosmos3-Super-Reasoner",
      models: ["nvidia/Cosmos3-Super-Reasoner", "mock/model"],
    },
    workflow_draft: { yaml: WORKFLOW_YAML, validation: WORKFLOW_VALIDATION },
  })).as("session");
  cy.intercept("GET", "/api/chat/sessions", json({
    active_session_id: "default",
    sessions: CHAT_SESSIONS,
  })).as("chatSessions");
  cy.intercept("POST", "/api/chat/sessions", json({
    active_session_id: "new-session",
    session: { id: "new-session", title: "New chat", chat_history: [] },
    sessions: [{ id: "new-session", title: "New chat", message_count: 0 }, ...CHAT_SESSIONS],
  })).as("newChatSession");
  cy.intercept("POST", "/api/chat/sessions/*/select", json({
    active_session_id: "session-two",
    session: {
      id: "session-two",
      chat_history: [
        { role: "user", content: "show status" },
        { role: "assistant", content: "**run_id**: `mock-run`" },
      ],
    },
    sessions: CHAT_SESSIONS,
  })).as("selectChatSession");
  cy.intercept("POST", "/api/chat", (req) => {
    const messages = Array.isArray(req.body.messages) ? req.body.messages : [];
    const lastMsg = messages.length ? messages[messages.length - 1] : null;
    const lastContent = lastMsg ? lastMsg.content : "";
    const lastText = Array.isArray(lastContent)
      ? lastContent
          .filter((part) => part && part.type === "text")
          .map((part) => String(part.text || ""))
          .join("\n")
      : String(lastContent || "");
    const lowered = lastText.toLowerCase();
    const visualContext = req.body && req.body.visual_context;
    if (visualContext || lowered.includes("[npa-visual-feedback]") || lowered.includes("describe this")) {
      const hasImage = Array.isArray(lastContent)
        && lastContent.some((part) => part && String(part.type || "").startsWith("image"));
      req.reply(json({
        ok: true,
        model: req.body.model || "Qwen/Qwen2.5-VL-72B-Instruct",
        session_id: req.body.session_id || "default",
        grounded: false,
        tier: hasImage ? "vision" : "reasoning",
        apis_used: [],
        reply: hasImage
          ? [
              "**What I see**: Dark 3D grid with orange and cyan skeleton wireframes (G1 trajectory style).",
              "**Likely meaning**: Locomotion / trajectory overlay in the Rerun viewer.",
              "**Operator feedback**: Structured sim content is visible — not a blank frame.",
              "**Next actions**: Scrub timeline; compare held-out cameras; keep this recording.",
            ].join("\n")
          : [
              "**What I see**: No viewer frame was attached — metadata only.",
              "**Likely meaning**: Capture could not read a non-blank canvas.",
              "**Operator feedback**: Wait for the viewer to settle past splash, then Describe this again.",
              "**Next actions**: Reload Rerun data; retry Describe this; try Video/Image artifacts.",
            ].join("\n"),
      }));
      return;
    }
    if (lowered.includes("outer loop") || lowered.includes("vlm") || lowered.includes("quality gate")) {
      req.reply(json({
        ok: true,
        model: req.body.model || "nvidia/Cosmos3-Super-Reasoner",
        session_id: req.body.session_id || "default",
        grounded: true,
        apis_used: ["workflows/draft", "workflows/validate", "workflows/plan"],
        reply: [
          "Here is a VLM/RL loop workflow for non-stock Sim2Real assets.",
          "```yaml",
          COMPLEX_WORKFLOW_YAML,
          "```",
        ].join("\n"),
        workflow_yaml: COMPLEX_WORKFLOW_YAML,
        workflow_validation: COMPLEX_WORKFLOW_VALIDATION,
        workflow_draft: {
          yaml: COMPLEX_WORKFLOW_YAML,
          validation: COMPLEX_WORKFLOW_VALIDATION,
          runnable: true,
        },
      }));
      return;
    }
    if (lowered.includes("non-stock") || lowered.includes("customer run") || lowered.includes("what can i view")) {
      req.reply(json({
        ok: true,
        model: req.body.model || "nvidia/Cosmos3-Super-Reasoner",
        session_id: req.body.session_id || "default",
        grounded: true,
        apis_used: ["artifacts/runs", "artifacts/run/{run_id}", "sim-viz/load-artifact", "sim-viz/status"],
        reply: [
          "**Non-stock Sim2Real artifacts**",
          `- **run_id**: \`${NON_STOCK_RUN_ID}\``,
          "- **preferred**: `reports/sim2real.rrd` (`rerun`)",
          "- **interactive surfaces**: Rerun recording, rollout video, report JSON, logs, and download fallback.",
          "- Use the Artifact browser to Discover runs, List artifacts, then Load the explicit object.",
        ].join("\n"),
      }));
      return;
    }
    req.reply(json({
      ok: true,
      model: req.body.model || "nvidia/Cosmos3-Super-Reasoner",
      session_id: req.body.session_id || "default",
      grounded: true,
      apis_used: ["sim-viz/status", "workflows/validate"],
      reply: [
        "Here is a 2-step workflow.",
        "```yaml",
        WORKFLOW_YAML,
        "```",
      ].join("\n"),
      workflow_yaml: WORKFLOW_YAML,
      workflow_validation: WORKFLOW_VALIDATION,
    }));
  }).as("chat");
  cy.intercept("GET", "/api/sim-assets", json(ASSETS)).as("simAssets");
  cy.intercept("GET", "/api/sim-assets/catalog", json({ ok: true, scenes: ["stock"], robots: ["franka"] })).as("simCatalog");
  cy.intercept("GET", "/api/sim-assets/cameras", json(CAMERAS)).as("cameras");
  cy.intercept("POST", "/api/sim-assets/selection", (req) => {
    req.reply(json({ ok: true, selection: { ...ASSETS.selection, ...(req.body || {}) }, sim_viz: SIM_VIZ }));
  }).as("setSelection");
  cy.intercept("GET", "/api/sim-assets/selection", json(ASSETS.selection)).as("getSelection");
  cy.intercept("PUT", "/api/sim-assets/cameras/selection", (req) => {
    req.reply(json({ ok: true, selected: req.body.selected || ["workspace"] }));
  }).as("setCamera");
  cy.intercept("GET", "/api/sim-viz/status*", (req) => {
    const url = new URL(req.url);
    const runId = url.searchParams.get("run_id") || "";
    if (runId === NON_STOCK_RUN_ID) {
      req.reply(json(activeSimViz.run_id === NON_STOCK_RUN_ID ? activeSimViz : NON_STOCK_SIM_VIZ));
      return;
    }
    req.reply(json(runId ? { ...SIM_VIZ, run_id: runId, active_run_id: runId } : activeSimViz));
  }).as("simVizStatus");
  cy.intercept("GET", "/api/sim-viz/rrd-blob*", {
    statusCode: 200,
    headers: { "content-type": "application/octet-stream" },
    body: "mock-rrd-payload",
  }).as("rrdBlob");
  cy.intercept("GET", "/api/sim-viz/rrd*", {
    statusCode: 200,
    headers: { "content-type": "application/octet-stream" },
    body: "mock-rrd-payload",
  }).as("rrd");
  cy.intercept("POST", "/api/sim-viz/load-franka-demo", (req) => {
    activeSimViz = {
      ...SIM_VIZ,
      run_id: "franka-demo",
      active_run_id: "franka-demo",
      source_type: "local_demo",
      source_label: "Local demo",
      stage: "demo",
    };
    req.reply(json({ ok: true, sim_viz: activeSimViz }));
  }).as("loadFranka");
  cy.intercept("POST", "/api/sim-viz/load-run", (req) => {
    const runId = String(req.body.run_id || "mock-run");
    activeSimViz = runId === NON_STOCK_RUN_ID
      ? NON_STOCK_SIM_VIZ
      : { ...SIM_VIZ, run_id: runId, active_run_id: runId };
    req.reply(json({ ok: true, sim_viz: activeSimViz }));
  }).as("loadRun");
  cy.intercept("POST", "/api/sim-viz/camera-preview", (req) => {
    req.reply(json({
      ok: true,
      entity_path: `world/camera_frustums/${req.body.camera || "workspace"}/frustum`,
      sim_viz: { ...SIM_VIZ, camera: req.body.camera || "workspace" },
    }));
  }).as("cameraPreview");
  cy.intercept("POST", "/api/sim-viz/load-artifact", (req) => {
    const key = String(req.body.key || "mock-run/preview.png");
    const render = renderForArtifactKey(key);
    activeSimViz = simVizForArtifact(key);
    req.reply(json({
      ok: true,
      render,
      sim_viz: activeSimViz,
      artifact_uri: `s3://mock/${key}`,
    }));
  }).as("loadArtifact");
  cy.intercept("GET", "/api/artifacts/preview/mock-run/preview.png", {
    statusCode: 200,
    headers: { "content-type": "image/png" },
    body: "",
  }).as("artifactPreview");
  // Inline thumbnails (stage inspector, Voxel51, artifact browser) fetch object
  // bytes through the authenticated download proxy.
  cy.intercept("GET", "/api/artifacts/download*", (req) => {
    const url = new URL(req.url);
    const uri = url.searchParams.get("s3_uri") || "";
    if (uri.match(/\.(png|jpe?g|gif|webp)$/i)) {
      req.reply({ statusCode: 200, headers: { "content-type": "image/png" }, body: "mock-image-bytes" });
      return;
    }
    req.reply({ statusCode: 200, headers: { "content-type": "application/octet-stream" }, body: "mock-bytes" });
  }).as("artifactDownload");
  cy.intercept({ method: /GET|HEAD/, url: "/api/artifacts/content*" }, (req) => {
    const url = new URL(req.url);
    const key = url.searchParams.get("key") || "";
    const download = url.searchParams.get("download") === "true";
    if (!download && key.endsWith(".json")) req.alias = "artifactContentJson";
    else if (!download && key.match(/\.ya?ml$/i)) req.alias = "artifactContentYaml";
    else if (!download && key.match(/\.(log|txt)$/i)) req.alias = "artifactContentText";
    else if (!download && key.match(/\.(png|jpe?g|gif|webp)$/i)) req.alias = "artifactContentImage";
    else if (!download && key.match(/\.(mp4|webm|mov)$/i)) req.alias = "artifactContentVideo";
    const baseHeaders = {
      "accept-ranges": "bytes",
      "cache-control": "private, no-store",
      "x-content-type-options": "nosniff",
      "content-disposition": download
        ? `attachment; filename="${key.split("/").pop() || "artifact.bin"}"`
        : `inline; filename="${key.split("/").pop() || "artifact.bin"}"`,
    };
    if (req.method === "HEAD") {
      req.reply({ statusCode: 200, headers: { ...baseHeaders, "content-length": "4096" }, body: "" });
      return;
    }
    if (!download && key.endsWith("manifest.json")) {
      const text = JSON.stringify({ run_id: key.split("/")[0], status: "completed" }, null, 2);
      req.reply({
        statusCode: 200,
        headers: { ...baseHeaders, "content-type": "application/json" },
        body: { ok: true, render: "json", text, bytes_read: text.length, total_bytes: text.length, truncated: false, redacted: false },
      });
      return;
    }
    if (!download && key.endsWith(".json")) {
      const text = JSON.stringify({ run_id: NON_STOCK_RUN_ID, result: "promoted", non_stock: true }, null, 2);
      req.reply({
        statusCode: 200,
        headers: { ...baseHeaders, "content-type": "application/json" },
        body: { ok: true, render: "json", text, bytes_read: text.length, total_bytes: text.length, truncated: false, redacted: false },
      });
      return;
    }
    if (!download && key.match(/\.ya?ml$/i)) {
      const text = "apiVersion: npa.workflow/v0.0.1\nmetadata:\n  name: groot-1-7-finetune\n";
      req.reply({
        statusCode: 200,
        headers: { ...baseHeaders, "content-type": "application/json" },
        body: { ok: true, render: "text", text, bytes_read: text.length, total_bytes: text.length, truncated: false, redacted: false },
      });
      return;
    }
    if (!download && key.match(/\.(log|txt)$/i)) {
      const text = key.endsWith("orchestrator.log")
        ? "loaded customer scene mesh\npublished non-stock sim2real artifacts\n"
        : "training completed\ntrain_loss=1.03125\n";
      req.reply({
        statusCode: 200,
        headers: { ...baseHeaders, "content-type": "application/json" },
        body: { ok: true, render: "text", text, bytes_read: text.length, total_bytes: text.length, truncated: false, redacted: false },
      });
      return;
    }
    if (!download && key.match(/\.(png|jpe?g|gif|webp)$/i)) {
      req.reply({ statusCode: 200, headers: { ...baseHeaders, "content-type": "image/png" }, body: "mock-image-bytes" });
      return;
    }
    if (!download && key.match(/\.(mp4|webm|mov)$/i)) {
      const ranged = Boolean(req.headers.range);
      req.reply({
        statusCode: ranged ? 206 : 200,
        headers: {
          ...baseHeaders,
          "content-type": "video/mp4",
          ...(ranged ? { "content-range": "bytes 0-99/4096", "content-length": "100" } : {}),
        },
        body: "mock-video-bytes",
      });
      return;
    }
    req.reply({
      statusCode: 200,
      headers: { ...baseHeaders, "content-type": "application/octet-stream" },
      body: "mock artifact payload",
    });
  }).as("artifactContent");
  cy.intercept("GET", "/api/artifacts/file/*", (req) => {
    const decoded = decodeURIComponent(req.url.split("/").pop() || "");
    if (decoded.includes("sim2real-report.json")) {
      req.reply({
        statusCode: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ run_id: NON_STOCK_RUN_ID, result: "promoted", non_stock: true }),
      });
      return;
    }
    if (decoded.includes("aggregate.json")) {
      req.reply({
        statusCode: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ run_id: JSON_ONLY_RUN_ID, evaluations: 4, checkpoint: "policy.ckpt" }),
      });
      return;
    }
    if (decoded.includes("orchestrator.log")) {
      req.reply({
        statusCode: 200,
        headers: { "content-type": "text/plain" },
        body: "loaded customer scene mesh\npublished non-stock sim2real artifacts\n",
      });
      return;
    }
    if (decoded.match(/\.(mp4|webm|mov)$/i)) {
      req.reply({
        statusCode: 200,
        headers: { "content-type": "video/mp4" },
        body: "mock-video-bytes",
      });
      return;
    }
    if (decoded.match(/\.(png|jpe?g|gif|webp)$/i)) {
      req.reply({
        statusCode: 200,
        headers: { "content-type": "image/png" },
        body: "mock-image-bytes",
      });
      return;
    }
    req.reply({
      statusCode: 200,
      headers: { "content-type": "application/octet-stream" },
      body: "mock artifact payload",
    });
  }).as("artifactFile");
  cy.intercept("GET", "/api/artifacts/runs*", (req) => {
    // Latest-first order from the API (non-stock newer than mock-run).
    const allRuns = [
      {
        run_id: NON_STOCK_RUN_ID,
        source_type: "artifact_storage",
        source_label: "S3 artifacts",
        bucket: "project-artifacts",
        project_id: "project-a",
        has_viewable: true,
        artifact_count: NON_STOCK_ARTIFACTS.length,
        last_modified: "2026-07-11T18:00:00Z",
      },
      {
        run_id: JSON_ONLY_RUN_ID,
        source_type: "artifact_storage",
        source_label: "S3 artifacts",
        bucket: "project-artifacts",
        project_id: "project-a",
        has_viewable: true,
        artifact_count: JSON_ONLY_ARTIFACTS.length,
        last_modified: "2026-07-10T12:00:00Z",
      },
      {
        run_id: ARTIFACT_ONLY_RUN_ID,
        source_type: "artifact_storage",
        source_label: "S3 artifacts",
        bucket: "project-artifacts",
        project_id: "project-a",
        has_viewable: true,
        artifact_count: ARTIFACT_ONLY_ARTIFACTS.length,
        last_modified: "2026-07-09T00:00:00Z",
      },
      {
        run_id: "mock-run",
        source_type: "artifact_storage",
        source_label: "S3 artifacts",
        bucket: "mock",
        project_id: "project-local",
        has_viewable: true,
        artifact_count: 1,
        last_modified: "2026-07-07T03:33:00Z",
      },
      {
        run_id: "archive-run",
        source_type: "artifact_storage",
        source_label: "S3 artifacts",
        bucket: "archive-artifacts",
        project_id: "project-a",
        has_viewable: true,
        artifact_count: 1,
        last_modified: "2026-07-06T03:33:00Z",
      },
    ];
    const url = new URL(req.url);
    const bucket = url.searchParams.get("resource_bucket") || "";
    const project = url.searchParams.get("project_id") || "";
    const query = String(url.searchParams.get("q") || "").toLowerCase();
    const runs = allRuns.filter((run) =>
      (!bucket || run.bucket === bucket) &&
      (!project || run.project_id === project) &&
      (!query || run.run_id.toLowerCase().includes(query))
    );
    req.reply(json({
      ok: true,
      runs,
      total_runs: runs.length,
      truncated: false,
      resource_scope: { project_id: project, bucket },
      access: { status: "available", scope: bucket ? "selected_resource" : "tenant" },
    }));
  }).as("artifactRuns");
  cy.intercept("GET", `/api/artifacts/run/${NON_STOCK_RUN_ID}*`, json({
    run_id: NON_STOCK_RUN_ID,
    run_ref: "npa1_mock_non_stock",
    bucket: "mock",
    project_id: "project-local",
    resolved_prefix: "",
    prefix: "sim2real-b",
    count: NON_STOCK_ARTIFACTS.length,
    artifacts: NON_STOCK_ARTIFACTS,
    preferred: NON_STOCK_ARTIFACTS[0],
  })).as("nonStockArtifactList");
  cy.intercept("GET", `/api/artifacts/run/${JSON_ONLY_RUN_ID}*`, json({
    run_id: JSON_ONLY_RUN_ID,
    bucket: "project-artifacts",
    project_id: "project-a",
    resolved_prefix: "",
    count: JSON_ONLY_ARTIFACTS.length,
    artifacts: JSON_ONLY_ARTIFACTS,
    preferred: JSON_ONLY_ARTIFACTS[0],
  })).as("jsonOnlyArtifactList");
  cy.intercept("GET", `/api/artifacts/run/${ARTIFACT_ONLY_RUN_ID}*`, json({
    run_id: ARTIFACT_ONLY_RUN_ID,
    bucket: "project-artifacts",
    project_id: "project-a",
    resolved_prefix: "tenant-runs",
    count: ARTIFACT_ONLY_ARTIFACTS.length,
    artifacts: ARTIFACT_ONLY_ARTIFACTS,
    preferred: null,
  })).as("artifactOnlyList");
  cy.intercept("GET", "/api/artifacts/run/mock-run*", json({
    run_id: "mock-run",
    prefix: "sim2real-b",
    artifacts: [
      {
        key: "mock-run/preview.png",
        s3_uri: "s3://mock/mock-run/preview.png",
        render: "image",
        size: 1234,
      },
    ],
  })).as("artifactList");
  cy.intercept("GET", `/api/artifacts/run/${DF_MOCK_RUN_ID}*`, json({
    run_id: DF_MOCK_RUN_ID,
    prefix: "physical-ai-data-factory",
    count: DF_MOCK_ARTIFACTS.length,
    artifacts: DF_MOCK_ARTIFACTS,
    preferred: DF_MOCK_ARTIFACTS.find((a) => a.key.includes("augmented_video.mp4")) || DF_MOCK_ARTIFACTS[0],
  })).as("dfArtifactList");
  cy.intercept("GET", `/api/artifacts/run/${DF_INPUT_ONLY_RUN_ID}*`, json({
    run_id: DF_INPUT_ONLY_RUN_ID,
    prefix: "physical-ai-data-factory",
    count: DF_INPUT_ONLY_ARTIFACTS.length,
    artifacts: DF_INPUT_ONLY_ARTIFACTS,
    preferred: DF_INPUT_ONLY_ARTIFACTS[0],
  })).as("dfInputOnlyArtifactList");
  cy.intercept("GET", "/api/artifacts/provenance/*", (req) => {
    if (String(req.url || "").includes(DF_MOCK_RUN_ID)) {
      req.reply(json(DF_MOCK_PROVENANCE));
      return;
    }
    // Other runs: no data-factory provenance (keeps the panel honest/empty).
    req.reply(json({ ok: true, run_id: "", components: [], summary: "", origin: {} }));
  }).as("artifactProvenance");
  cy.intercept("GET", `/api/fiftyone/dataset/${DF_MOCK_RUN_ID}`, json({
    run_id: DF_MOCK_RUN_ID,
    source: {
      source_kind: "user_supplied",
      input_origin: "operator_supplied",
      input_origin_label: "User-supplied input",
      staged_canonical_s3_uri: `s3://mock/physical-ai-data-factory/${DF_MOCK_RUN_ID}/input/`,
      asset_license: "operator-managed",
      sha256: "b".repeat(64),
    },
    review: {
      engine: "fiftyone-brain",
      real_fiftyone: true,
      label: "Real FiftyOne Brain review",
      limitation: "",
    },
    summary: {
      variant_count: 1,
      source_input_count: 1,
      original_input_count: 1,
      conditioning_count: 1,
      synthetic_augmented_count: 1,
      curation_engine: "fiftyone-brain",
      curated_kept: 1,
    },
    fields: ["lighting"],
    visualization: [],
    samples: [
      {
        id: "source.mp4",
        label: "source.mp4",
        group: "source",
        data_role: "source_input",
        data_role_label: "User-supplied input",
        video_uri: `s3://mock/${DF_MOCK_RUN_ID}/input/source.mp4`,
      },
      {
        id: "conditioning-frame-0001.png",
        label: "conditioning-frame-0001.png",
        group: "conditioning",
        data_role: "derived_conditioning",
        data_role_label: "Derived conditioning frame",
        thumbnail_uri: `s3://mock/${DF_MOCK_RUN_ID}/input/conditioning-frame-0001.png`,
      },
      {
        id: "aug0",
        label: "aug0",
        group: "augmented",
        data_role: "synthetic_augmented",
        data_role_label: "Synthetic / augmented output",
        thumbnail_uri: `s3://mock/${DF_MOCK_RUN_ID}/cosmos_augmented/aug0/frame-000000.png`,
        video_uri: `s3://mock/${DF_MOCK_RUN_ID}/cosmos_augmented/aug0/augmented_video.mp4`,
        tags: { lighting: "warm" },
      },
    ],
  })).as("dfDataset");
  cy.intercept("POST", "/api/workflows/draft", json({
    ok: true,
    yaml: WORKFLOW_YAML,
    validation: WORKFLOW_VALIDATION,
    plan: { ok: true, steps: [{ state: "draft", tool_ref: "workbench.sim2real.status" }] },
  })).as("workflowDraft");
  cy.intercept("POST", "/api/workflows/validate", (req) => {
    const yaml = String(req.body.yaml || "");
    const validation = yaml.includes("cypress-vlm-rl-loop") ? COMPLEX_WORKFLOW_VALIDATION : WORKFLOW_VALIDATION;
    req.reply(json({ ok: true, validation }));
  }).as("workflowValidate");
  cy.intercept("POST", "/api/workflows/plan", json({
    ok: true,
    plan: {
      workflow: "cypress-vlm-rl-loop",
      steps: [
        { state: "rollout", tool_ref: "workbench.sim2real.policy_rollout" },
        { state: "vlm_gate", tool_ref: "workbench.token_factory.reason" },
        { state: "finalize", tool_ref: "workbench.sim2real.status" },
      ],
    },
  })).as("workflowPlan");
  cy.intercept("POST", "/api/workflows/submit", json({
    ok: true,
    run_id: "workflow-run",
    submit_mode: "mock",
    validation: WORKFLOW_VALIDATION,
  })).as("workflowSubmitYaml");
  cy.intercept("POST", "/api/workflows/sim2real/submit", json({
    ok: true,
    run_id: "submitted-run",
    sim_viz: { ...SIM_VIZ, run_id: "submitted-run", stage: "running" },
    run: { ...RUN_DETAILS.run, run_id: "submitted-run" },
  })).as("submitSim2Real");
  cy.intercept("GET", "/api/workflows/sim2real/status", json({
    latest_submit: { run_id: "submitted-run" },
    sim_viz: { ...SIM_VIZ, run_id: "submitted-run", stage: "running" },
    run: { ...RUN_DETAILS.run, run_id: "submitted-run" },
  })).as("workflowStatus");
  cy.intercept("GET", "/api/workflows/sim2real/runs/*", (req) => {
    const runId = decodeURIComponent(req.url.split("/").pop().split("?")[0] || "mock-run");
    if (runId === "franka-demo") {
      req.reply(json({
        run: {
          run_id: "franka-demo",
          source_type: "local_demo",
          source_label: "Local demo",
          status: "completed",
          result: "rerun_ready",
          stages: [{ id: "local_demo", label: "Local Franka demo", status: "succeeded", summary: "Deterministically generated locally." }],
          logs: [{ timestamp: "2026-07-09T12:00:00Z", level: "info", message: "Local Franka demo recording regenerated." }],
        },
      }));
      return;
    }
    if (runId === NON_STOCK_RUN_ID) {
      req.reply(json(NON_STOCK_RUN_DETAILS));
      return;
    }
    if (runId === ARTIFACT_ONLY_RUN_ID) {
      req.reply(json(ARTIFACT_ONLY_RUN_DETAILS));
      return;
    }
    if (runId === JSON_ONLY_RUN_ID) {
      req.reply(json({
        run: {
          run_id: JSON_ONLY_RUN_ID,
          source_type: "artifact_storage",
          source_label: "S3 artifacts",
          status: "status_unavailable",
          status_label: "Status unavailable",
          result: "artifacts_available",
          stages: [{
            id: "evaluation",
            stage_key: "evaluation",
            label: "evaluation",
            status: "observed_output",
            status_label: "Observed output",
            artifact_count: JSON_ONLY_ARTIFACTS.length,
            evidence_type: "artifact_observation",
            evidence_source: "artifact_listing",
            authority: "observed",
          }],
          logs: [],
        },
      }));
      return;
    }
    if (runId === "cosmos-reason-run") {
      req.reply(json(GENERIC_WORKFLOW_RUN_DETAILS));
      return;
    }
    req.reply(json({ run: { ...RUN_DETAILS.run, run_id: runId } }));
  }).as("runDetails");
}

// --- MCAP substance helpers (shared by smoke + live Lichtblick specs) ---------
// An uncompressed MCAP stores channel topics/schema names and JSON messages
// verbatim, so we can assert the viewer is fed real camera data straight from the
// served bytes without a full MCAP parser.

function mcapCameraTopicCount(binaryBody) {
  return (String(binaryBody || "").match(/\/camera/g) || []).length;
}

function mcapHasCompressedImage(binaryBody) {
  return String(binaryBody || "").indexOf("foxglove.CompressedImage") >= 0;
}

function mcapHasHeldoutCamera(binaryBody) {
  return String(binaryBody || "").indexOf("/heldout/camera/") >= 0;
}

function mcapHasPointCloud(binaryBody) {
  const text = String(binaryBody || "");
  return text.indexOf("foxglove.PointCloud") >= 0 && text.indexOf("/heldout/points") >= 0;
}

function mcapHasFrameTransform(binaryBody) {
  const text = String(binaryBody || "");
  return text.indexOf("foxglove.FrameTransform") >= 0 && text.indexOf("/tf") >= 0;
}

// The 3D panel only offers its "rgba-fields" colour mode when the cloud declares
// ALL of red/green/blue/alpha; without alpha it re-colours the cloud with a fallback
// colormap instead of the captured RGB. Assert the served cloud declares the full
// set the injected default layout asks for.
function mcapPointCloudColorFields(binaryBody) {
  const text = String(binaryBody || "");
  const start = text.indexOf('"point_stride"');
  if (start < 0) return [];
  const message = text.slice(start, start + 4000);
  return ["red", "green", "blue", "alpha"].filter((name) =>
    new RegExp('"name":\\s*"' + name + '"').test(message)
  );
}

function mcapPointCloudHasRgbaFields(binaryBody) {
  return mcapPointCloudColorFields(binaryBody).length === 4;
}

function firstMcapPngPayload(binaryBody) {
  // json.dumps emits ", " / ": " separators, so allow optional whitespace.
  const match = String(binaryBody || "").match(
    /"data":\s*"([A-Za-z0-9+/=]+)"\s*,\s*"format":\s*"png"/
  );
  return match ? match[1] : null;
}

// Decode a base64 PNG payload and return {width,height,mean} using an offscreen
// canvas. Catches BOTH regressions: 32x32 solid stubs (tiny dims) and the
// PNG-row-filter corruption that turned real renders into dark noise (low mean).
function decodePngStats(base64Payload) {
  return new Cypress.Promise((resolve, reject) => {
    try {
      const binary = atob(base64Payload);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: "image/png" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let sum = 0;
        let count = 0;
        for (let i = 0; i < data.length; i += 4) {
          sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
          count += 1;
        }
        URL.revokeObjectURL(url);
        resolve({ width: img.naturalWidth, height: img.naturalHeight, mean: sum / Math.max(1, count) });
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("PNG decode failed"));
      };
      img.src = url;
    } catch (err) {
      reject(err);
    }
  });
}

Cypress.Commands.add("installAgentApiMocks", installAgentApiMocks);
Cypress.Commands.add("visitMockAgent", () => {
  installAgentApiMocks();
  cy.visit("/");
  cy.get("meta[name='npa-ui-version']").should("have.attr", "content").and("match", /^(\d+|dev)$/);
  cy.get("#statusBar").should("exist");
});

function resolveLiveAgentConfig(readValue) {
  const read = typeof readValue === "function" ? readValue : (name) => readValue && readValue[name];
  const config = {
    baseUrl: read("agentBaseUrl") || read("NPA_AGENT_BASE_URL") || "",
    username: read("agentUser") || read("NPA_AGENT_USER") || "",
    password: read("agentPassword") || read("NPA_AGENT_PASSWORD") || "",
  };
  const present = Object.values(config).filter(Boolean).length;
  if (present && present !== 3) {
    throw new Error(
      "Live Cypress configuration is incomplete; set agentBaseUrl/agentUser/agentPassword " +
      "or NPA_AGENT_BASE_URL/NPA_AGENT_USER/NPA_AGENT_PASSWORD."
    );
  }
  return present === 3 ? config : null;
}

function currentLiveAgentConfig() {
  const config = resolveLiveAgentConfig((name) => Cypress.env(name));
  if (!config) {
    throw new Error(
      "Live Cypress requires agentBaseUrl, agentUser, and agentPassword " +
      "(or their NPA_AGENT_* equivalents)."
    );
  }
  return config;
}

Cypress.Commands.add("visitLiveAgent", () => {
  const { baseUrl, username, password } = currentLiveAgentConfig();
  cy.visit({
    url: baseUrl,
    auth: { username, password },
    failOnStatusCode: true,
  });
});

export {
  ARTIFACT_ONLY_ARTIFACTS,
  ARTIFACT_ONLY_RUN_ID,
  ASSETS,
  CAMERAS,
  COMPLEX_WORKFLOW_YAML,
  decodePngStats,
  DF_INPUT_ONLY_ARTIFACTS,
  DF_INPUT_ONLY_RUN_ID,
  DF_MOCK_ARTIFACTS,
  DF_MOCK_RUN_ID,
  FIELD_IDS,
  firstMcapPngPayload,
  GENERIC_WORKFLOW_RUN_DETAILS,
  GENERIC_WORKFLOW_YAML,
  currentLiveAgentConfig,
  mcapCameraTopicCount,
  mcapHasCompressedImage,
  mcapHasFrameTransform,
  mcapHasHeldoutCamera,
  mcapHasPointCloud,
  mcapPointCloudColorFields,
  mcapPointCloudHasRgbaFields,
  NON_STOCK_ARTIFACTS,
  NON_STOCK_RUN_ID,
  resolveLiveAgentConfig,
  SIM_VIZ,
  STATIC_BUTTON_IDS,
  WORKFLOW_YAML,
};
