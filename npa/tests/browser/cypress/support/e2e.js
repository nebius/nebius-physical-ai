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

const SIM_VIZ = {
  run_id: "mock-run",
  active_run_id: "mock-run",
  stage: "demo",
  camera: "workspace",
  rrd_uri: "file:///opt/npa-agent/sim2real.rrd",
  rrd_updated_at: "2026-07-07T03:33:00Z",
  rerun_ready: true,
  rerun_iframe_url: "/rerun/?url=https://example.test/rerun/recordings/sim2real.rrd&hide_welcome_screen=1&camera=workspace",
  // Intentionally not alphabetical — UI must keep latest-first order.
  available_run_ids: ["submitted-run", "mock-run"],
  available_runs: [
    { run_id: "submitted-run", last_modified: "2026-07-08T12:00:00Z", stage: "submitted" },
    { run_id: "mock-run", last_modified: "2026-07-07T03:33:00Z", stage: "demo" },
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
  rerun_iframe_url: "/rerun/?url=https://example.test/rerun/recordings/sim2real.rrd&hide_welcome_screen=1&camera=customer-overhead",
  available_run_ids: [NON_STOCK_RUN_ID, "mock-run", "submitted-run"],
  available_runs: [
    { run_id: NON_STOCK_RUN_ID, last_modified: "2026-07-11T18:00:00Z", stage: "stage_14_rerun_viz" },
    { run_id: "submitted-run", last_modified: "2026-07-08T12:00:00Z", stage: "submitted" },
    { run_id: "mock-run", last_modified: "2026-07-07T03:33:00Z", stage: "demo" },
  ],
  artifact_render: "rerun",
  artifact_key: `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`,
  artifact_uri: `s3://mock/${NON_STOCK_RUN_ID}/reports/sim2real.rrd`,
  artifact_preview_url: "/rerun/recordings/sim2real.rrd",
  artifact_download_url: "/rerun/recordings/sim2real.rrd",
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

// A Physical AI Data Factory run whose artifacts span every pipeline stage and
// whose augment is a REAL Cosmos Transfer 2.5 GPU render — used to exercise the
// per-stage provenance panel (counts, click-to-filter, honest engine banner).
const DF_MOCK_RUN_ID = "paidf-mock-gpu-run";
const DF_MOCK_ARTIFACTS = [
  { key: `checkpoints/physical-ai-data-factory/${DF_MOCK_RUN_ID}/input/video_0.mp4`, s3_uri: `s3://mock/${DF_MOCK_RUN_ID}/input/video_0.mp4`, render: "video", size: 4096 },
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
    { stage: "Source frames", stage_key: "input", component: "Uploaded source clips", runtime: "input", artifact_count: 1 },
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
  "applySelection",
  "loadFrankaRerun",
  "submitWorkflow",
  "workflowStatus",
  "loadRunData",
  "artifactRefreshRuns",
  "artifactLoadRunArtifacts",
  "openRerun",
  "loadRerunViewer",
  "describeVisual",
];

const FIELD_IDS = [
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
  "sceneMode",
  "robotPreset",
  "cameraMode",
  "simBackend",
  "propCube",
  "assetsSummary",
  "runIdInput",
  "runIdSelect",
  "artifactPrefix",
  "artifactTypeFilter",
  "artifactSort",
  "artifactStageFilter",
  "runsArtifactsPanel",
  "artifactList",
  "simRunId",
  "simStage",
  "simCamera",
  "renderedDataSummary",
  "rerunFrame",
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
  if (String(key || "").match(/\.(mp4|webm|mov)$/)) return "video";
  if (String(key || "").match(/\.(png|jpg|jpeg|gif|webp)$/)) return "image";
  if (String(key || "").endsWith(".json")) return "json";
  if (String(key || "").match(/\.(txt|log|csv|yaml|yml|md)$/)) return "text";
  return "download";
}

function simVizForArtifact(key) {
  const render = renderForArtifactKey(key);
  const base = String(key || "").startsWith(`${NON_STOCK_RUN_ID}/`) ? NON_STOCK_SIM_VIZ : SIM_VIZ;
  const previewPath = `/api/artifacts/file/${encodeURIComponent(key.replaceAll("/", "__"))}`;
  if (render === "rerun") {
    return { ...base, artifact_render: render, artifact_key: key, artifact_uri: `s3://mock/${key}` };
  }
  return {
    ...base,
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
  cy.intercept("GET", "/api/health", json({ ok: true, tool_refs: 19 })).as("health");
  cy.intercept("GET", "/api/models", json({
    ok: true,
    model: "nvidia/Cosmos3-Super-Reasoner",
    models: ["nvidia/Cosmos3-Super-Reasoner", "mock/model"],
  })).as("models");
  cy.intercept("GET", "/api/session", json({
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
  cy.intercept("POST", "/api/sim-viz/load-franka-demo", json({ ok: true, sim_viz: SIM_VIZ })).as("loadFranka");
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
  cy.intercept("GET", "/api/artifacts/runs*", json({
    // Latest-first order from the API (non-stock newer than mock-run).
    runs: [
      {
        run_id: NON_STOCK_RUN_ID,
        has_viewable: true,
        artifact_count: NON_STOCK_ARTIFACTS.length,
        last_modified: "2026-07-11T18:00:00Z",
      },
      {
        run_id: "mock-run",
        has_viewable: true,
        artifact_count: 1,
        last_modified: "2026-07-07T03:33:00Z",
      },
    ],
    total_runs: 2,
    truncated: false,
  })).as("artifactRuns");
  cy.intercept("GET", `/api/artifacts/run/${NON_STOCK_RUN_ID}*`, json({
    run_id: NON_STOCK_RUN_ID,
    prefix: "sim2real-b",
    count: NON_STOCK_ARTIFACTS.length,
    artifacts: NON_STOCK_ARTIFACTS,
    preferred: NON_STOCK_ARTIFACTS[0],
  })).as("nonStockArtifactList");
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
    preferred: DF_MOCK_ARTIFACTS[2],
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
    if (runId === NON_STOCK_RUN_ID) {
      req.reply(json(NON_STOCK_RUN_DETAILS));
      return;
    }
    if (runId === "cosmos-reason-run") {
      req.reply(json(GENERIC_WORKFLOW_RUN_DETAILS));
      return;
    }
    req.reply(json({ run: { ...RUN_DETAILS.run, run_id: runId } }));
  }).as("runDetails");
}

Cypress.Commands.add("installAgentApiMocks", installAgentApiMocks);
Cypress.Commands.add("visitMockAgent", () => {
  installAgentApiMocks();
  cy.visit("/");
  cy.get("meta[name='npa-ui-version']").should("have.attr", "content").and("match", /^(\d+|dev)$/);
  cy.get("#statusBar").should("exist");
});

Cypress.Commands.add("visitLiveAgent", () => {
  const baseUrl = Cypress.env("agentBaseUrl") || Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl");
  const username = Cypress.env("agentUser") || Cypress.env("NPA_AGENT_USER");
  const password = Cypress.env("agentPassword") || Cypress.env("NPA_AGENT_PASSWORD");
  if (!baseUrl || !username || !password) {
    throw new Error("Set NPA_AGENT_BASE_URL, NPA_AGENT_USER, and NPA_AGENT_PASSWORD for live Cypress.");
  }
  cy.visit({
    url: baseUrl,
    auth: { username, password },
    failOnStatusCode: true,
  });
});

export {
  ASSETS,
  CAMERAS,
  COMPLEX_WORKFLOW_YAML,
  DF_INPUT_ONLY_ARTIFACTS,
  DF_INPUT_ONLY_RUN_ID,
  DF_MOCK_ARTIFACTS,
  DF_MOCK_RUN_ID,
  FIELD_IDS,
  GENERIC_WORKFLOW_RUN_DETAILS,
  GENERIC_WORKFLOW_YAML,
  NON_STOCK_ARTIFACTS,
  NON_STOCK_RUN_ID,
  SIM_VIZ,
  STATIC_BUTTON_IDS,
  WORKFLOW_YAML,
};
