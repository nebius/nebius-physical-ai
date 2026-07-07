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

const SIM_VIZ = {
  run_id: "mock-run",
  active_run_id: "mock-run",
  stage: "demo",
  camera: "workspace",
  rrd_uri: "file:///opt/npa-agent/sim2real.rrd",
  rrd_updated_at: "2026-07-07T03:33:00Z",
  rerun_ready: true,
  rerun_iframe_url: "/rerun/?url=/rerun/recordings/sim2real.rrd&camera=workspace",
  available_run_ids: ["mock-run", "submitted-run"],
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

const WORKFLOW_VALIDATION = {
  ok: true,
  status: "valid",
  name: "cypress-sim2real",
  states: ["draft"],
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
  "sceneMode",
  "robotPreset",
  "cameraMode",
  "simBackend",
  "propCube",
  "assetsSummary",
  "runIdInput",
  "runIdSelect",
  "artifactPrefix",
  "artifactRunSelect",
  "artifactList",
  "activeCameraLabel",
  "cameraCards",
  "simRunId",
  "simStage",
  "simCamera",
  "renderedDataSummary",
  "rerunFrame",
  "artifactPreviewHost",
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

function installAgentApiMocks() {
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
  cy.intercept("GET", "/api/sim-viz/status*", json(SIM_VIZ)).as("simVizStatus");
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
    req.reply(json({ ok: true, sim_viz: { ...SIM_VIZ, run_id: req.body.run_id || "mock-run" } }));
  }).as("loadRun");
  cy.intercept("POST", "/api/sim-viz/camera-preview", (req) => {
    req.reply(json({
      ok: true,
      entity_path: `world/camera_frustums/${req.body.camera || "workspace"}/frustum`,
      sim_viz: { ...SIM_VIZ, camera: req.body.camera || "workspace" },
    }));
  }).as("cameraPreview");
  cy.intercept("POST", "/api/sim-viz/load-artifact", json({
    ok: true,
    render: "image",
    sim_viz: {
      ...SIM_VIZ,
      artifact_render: "image",
      artifact_key: "mock-run/preview.png",
      artifact_uri: "s3://mock/mock-run/preview.png",
      artifact_preview_url: "/api/artifacts/preview/mock-run/preview.png",
      artifact_download_url: "/api/artifacts/download/mock-run/preview.png",
    },
  })).as("loadArtifact");
  cy.intercept("GET", "/api/artifacts/preview/mock-run/preview.png", {
    statusCode: 200,
    headers: { "content-type": "image/png" },
    body: "",
  }).as("artifactPreview");
  cy.intercept("GET", "/api/artifacts/runs*", json({
    runs: [{ run_id: "mock-run", has_viewable: true }],
    total_runs: 1,
    truncated: false,
  })).as("artifactRuns");
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
  cy.intercept("POST", "/api/workflows/draft", json({
    ok: true,
    yaml: WORKFLOW_YAML,
    validation: WORKFLOW_VALIDATION,
    plan: { ok: true, steps: [{ state: "draft", tool_ref: "workbench.sim2real.status" }] },
  })).as("workflowDraft");
  cy.intercept("POST", "/api/workflows/validate", json({
    ok: true,
    validation: WORKFLOW_VALIDATION,
  })).as("workflowValidate");
  cy.intercept("POST", "/api/workflows/plan", json({
    ok: true,
    plan: {
      workflow: "cypress-sim2real",
      steps: [{ state: "draft", tool_ref: "workbench.sim2real.status" }],
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
    req.reply(json({ run: { ...RUN_DETAILS.run, run_id: runId } }));
  }).as("runDetails");
}

Cypress.Commands.add("installAgentApiMocks", installAgentApiMocks);
Cypress.Commands.add("visitMockAgent", () => {
  installAgentApiMocks();
  cy.visit("/");
  cy.get("meta[name='npa-ui-version']").should("have.attr", "content").and("match", /^\d+|dev$/);
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
  FIELD_IDS,
  SIM_VIZ,
  STATIC_BUTTON_IDS,
  WORKFLOW_YAML,
};
