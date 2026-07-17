import {
  COMPLEX_WORKFLOW_YAML,
  FIELD_IDS,
  GENERIC_WORKFLOW_YAML,
  NON_STOCK_RUN_ID,
  STATIC_BUTTON_IDS,
  WORKFLOW_YAML,
} from "../support/e2e";

describe("NPA agent UI with mocked APIs", () => {
  beforeEach(() => {
    cy.visitMockAgent();
    cy.wait("@session");
    cy.wait("@simAssets");
  });

  it("renders a generic Stages panel (not Sim2Real-only)", () => {
    cy.get("#stagesPanel").should("exist");
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
    cy.get("#stagesPanel .hint").should("contain.text", "Timeline, result, and logs");
    cy.get("#stagesPanel .hint").should("not.contain.text", "Sim2Real-only");
    cy.get("#stageList").should("have.attr", "aria-label", "Workflow stages");
    cy.get("#stageList").should("contain.text", "Select assets");
    cy.get("#stageList").should("contain.text", "Render");
    cy.get("#stageList").should("contain.text", "Succeeded");
    cy.get("#runSummary").should("contain.text", "mock-run");
  });

  it("shows generic workflow stages for non-Sim2Real runs", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#runIdInput").clear({ force: true }).type("cosmos-reason-run", { force: true });
    cy.get("#loadRunData").click({ force: true });
    cy.wait("@loadRun");
    cy.get("#tabChat").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.get("#runSummary").should("contain.text", "cosmos-reason-run");
    cy.get("#stageList").should("contain.text", "Fetch checkpoint");
    cy.get("#stageList").should("contain.text", "Reason");
    cy.get("#stageList").should("contain.text", "Publish");
    cy.get("#stageList").should("contain.text", "Running");
    cy.get("#runLog").should("contain.text", "generic workflow stages active");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
  });

  it("keeps Stages generic after drafting a non-Sim2Real workflow YAML", () => {
    cy.get("#workflowYaml").clear().type(GENERIC_WORKFLOW_YAML, { delay: 0 });
    cy.get("#workflowValidate").click();
    cy.wait("@workflowValidate");
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.get("#stagesPanel .hint").should("contain.text", "Timeline, result, and logs");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
  });

  it("renders every static control and generated panel", () => {
    for (const id of STATIC_BUTTON_IDS) {
      cy.get(`#${id}`).should("exist");
    }
    for (const id of FIELD_IDS) {
      cy.get(`#${id}`).should("exist");
    }
    cy.get("#workflowYaml").should("contain.value", "apiVersion: npa.workflow/v0.0.1");
    cy.get("#workflowSubmitHint").should("contain.text", "plan-only");
    cy.get("#tabChat").should("have.attr", "aria-selected", "true");
    cy.get("#tabRerun").click();
    cy.get("#tabRerun").should("have.attr", "aria-selected", "true");
    cy.get("#panelRerun").should("have.class", "is-active").and("have.attr", "aria-hidden", "false");
    cy.get("#panelChat").should("have.class", "is-inactive").and("have.attr", "aria-hidden", "true");
    cy.get("#renderModeRerun").should("have.class", "is-active");
    cy.get("#renderModeVideo").should("exist");
    cy.get("#viewerPaneRerun").should("have.class", "is-active-viewer");
    cy.get("#assetsSummary").should("contain.text", "stock://robot/franka");
    cy.get("#simRunId").should("contain.text", "mock-run");
  });

  it("covers chat quick actions, sessions, model selection, submit, and copy", () => {
    cy.get("#chatActionS3").click();
    cy.get("#chatInput").should("contain.value", "configure S3");
    cy.get("#chatActionCosmos").click();
    cy.get("#chatInput").should("contain.value", "Cosmos3");
    cy.get("#chatActionWatch").click();
    cy.get("#chatInput").should("contain.value", "Rerun");
    cy.get("#chatActionWorkflow").click();
    cy.get("#chatInput").should("contain.value", "2-step sim2real workflow");

    cy.get("#chatModel").select("mock/model");
    cy.get("#chatSend").click();
    cy.wait("@chat");
    cy.get("#chatLog .msg-row.user").should("contain.text", "2-step sim2real workflow");
    cy.get("#chatLog .msg-row.assistant").should("contain.text", "Here is a 2-step workflow");
    cy.get("#workflowYaml").should("contain.value", "cypress-sim2real");

    cy.window().then((win) => {
      if (!win.navigator.clipboard) {
        Object.defineProperty(win.navigator, "clipboard", {
          value: { writeText: () => Promise.resolve() },
          configurable: true,
        });
      }
      cy.stub(win.navigator.clipboard, "writeText").resolves();
    });
    cy.get(".msg-copy-btn").contains(/^Copy/).first().click();
    cy.get("#toastHost").should("contain.text", "copied");

    cy.get("#newChatSession").click();
    cy.wait("@newChatSession");
    cy.get("#chatSessionSelect").should("have.value", "new-session");
    cy.get("#chatSessionSelect").select("session-two");
    cy.wait("@selectChatSession");
    cy.get("#chatLog").should("contain.text", "show status");
  });

  it("covers workflow draft upload, validate, plan, and submit buttons", () => {
    cy.get("#workflowYaml").clear().type(WORKFLOW_YAML, { delay: 0 });

    cy.get("#workflowUpload").click();
    cy.wait("@workflowDraft");
    cy.get("#chatLog").should("contain.text", "Uploaded workflow YAML");

    cy.get("#workflowValidate").click();
    cy.wait("@workflowValidate");
    cy.get("#workflowValidation").should("contain.text", "valid");

    cy.get("#workflowPlan").click();
    cy.wait("@workflowPlan");
    cy.get("#workflowPlanHost").should("be.visible");
    cy.get("#workflowPlanHost").should("contain.text", "workbench.sim2real.status");
    cy.get("#workflowPlanOutput").should("contain.text", "workbench.sim2real.status");
    cy.get("#workflowValidation").should("contain.text", "planned");

    cy.get("#workflowSubmitYaml").click();
    cy.wait("@workflowSubmitYaml");
    cy.get("#chatLog").should("contain.text", "Submitted npa.workflow");
    cy.get("#chatLog").should("contain.text", "plan");
  });

  it("covers Sim2Real selection, Stages panel, Rerun buttons, and run-data loading", () => {
    cy.window().then((win) => {
      cy.stub(win, "open").as("windowOpen");
    });

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");

    cy.get("#robotPreset").select("ur5e");
    cy.wait("@setSelection");
    cy.get("#robotPreset").should("have.value", "franka");

    cy.get("#simBackend").select("genesis");
    cy.get("#propCube").uncheck();
    cy.get("#applySelection").click();
    cy.wait("@setSelection");
    cy.get("#assetsSummary").should("contain.text", "stock://scene/default");

    cy.get("#loadFrankaRerun").click();
    cy.wait("@loadFranka");
    cy.get("#tabChat").click();
    cy.get("#chatLog").should("contain.text", "Loaded stock Franka");
    cy.get("#stagesPanel h3").should("have.text", "Stages");

    cy.get("#tabRerun").click();
    cy.get("#loadRerunViewer").click({ force: true });
    cy.get("#statusBar").should(($bar) => {
      expect($bar.text()).to.match(/Rerun|Reload/);
    });

    cy.get("#submitWorkflow").click();
    cy.wait("@submitSim2Real");
    cy.get("#tabChat").click();
    cy.get("#chatLog").should("contain.text", "Submitted Sim2Real run");
    cy.get("#runSummary").should("contain.text", "submitted-run");

    cy.get("#tabRerun").click();
    cy.get("#workflowStatus").click();
    cy.wait("@workflowStatus");
    cy.get("#tabChat").click();
    cy.get("#chatLog").should("contain.text", "Latest workflow status");

    cy.get("#tabRerun").click();
    cy.get("#runIdInput").clear().type("mock-run");
    cy.get("#loadRunData").click();
    cy.wait("@loadRun");
    cy.get("#tabChat").click();
    cy.get("#chatLog").should("contain.text", "Loaded run context");
    cy.get("#runLog").should("contain.text", "mock run log");
    cy.get("#stagesPanel h3").should("have.text", "Stages");

    cy.get("#tabRerun").click();
    cy.get("#openRerun").click();
    cy.get("@windowOpen").should("have.been.called");
  });

  it("covers artifact discovery, dynamic artifact load button, and camera cards", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactPrefix").type("sim2real-b");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#artifactRunSelect").select("mock-run");
    cy.wait("@artifactList");
    cy.get("#artifactList").should("contain.text", "mock-run/preview.png");

    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@artifactList");
    cy.get("#artifactList button[data-action='load-artifact']").click();
    cy.wait("@loadArtifact");
    cy.get("#chatLog").should("contain.text", "Loaded artifact");
    cy.get("#artifactPreviewHost").should("not.have.attr", "hidden");

    cy.get("#tabChat").click();
    cy.get("#panelChat").should("have.class", "is-active").and("have.attr", "aria-hidden", "false");
    cy.get("#panelRerun").should("have.class", "is-inactive");
  });

  it("discovers and interacts with non-stock Sim2Real run artifacts", () => {
    cy.window().then((win) => {
      cy.stub(win, "open").as("windowOpen");
    });

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactPrefix").clear().type("sim2real-b/custom-assets");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#artifactRunSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.wait("@loadArtifact");

    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`);
    cy.get("#artifactList").should("contain.text", "rerun");
    cy.get("#artifactList").should("contain.text", "video");
    cy.get("#artifactList").should("contain.text", "json");
    cy.get("#artifactList").should("contain.text", "text");
    cy.get("#artifactList").should("contain.text", "download");
    cy.get("#artifactList").should("contain.text", "View in Rerun");
    cy.get("#artifactList").should("contain.text", "Play");
    cy.get("#artifactTypeFilter").select("video");
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/rollouts/customer-camera.mp4`);
    cy.get("#artifactList").should("not.contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`);
    cy.get("#artifactSort").select("largest");
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", "Showing 1 of");
    cy.get("#artifactTypeFilter").select("");
    cy.wait("@nonStockArtifactList");
    cy.get("#simRunId").should("contain.text", NON_STOCK_RUN_ID);
    cy.get("#simStage").should("contain.text", "stage_14_rerun_viz");
    cy.get("#simCamera").should("contain.text", "customer-overhead");
    cy.get("#rerunFrame").should("have.attr", "src").and("include", "/rerun/");

    cy.get("#runIdInput").clear().type(NON_STOCK_RUN_ID);
    cy.get("#loadRunData").click();
    cy.wait("@loadRun");
    cy.get("#tabChat").click();
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.get("#runSummary").should("contain.text", NON_STOCK_RUN_ID);
    cy.get("#stageList").should("contain.text", "Customer assets");
    cy.get("#runLog").should("contain.text", "non-stock sim2real artifacts");

    cy.get("#tabRerun").click();
    cy.get(`#artifactList button[data-key="${NON_STOCK_RUN_ID}/rollouts/customer-camera.mp4"]`).click();
    cy.wait("@loadArtifact");
    cy.wait("@artifactFile");
    cy.get("#renderModeVideo").should("have.class", "is-active");
    cy.get("#viewerPaneMedia").should("have.class", "is-active-viewer");
    cy.get("#artifactPreviewHost video")
      .should("have.attr", "src")
      .and("match", /^blob:/);
    cy.get("#artifactPreviewHost video")
      .should("have.attr", "data-preview-url")
      .and("include", "customer-camera.mp4");
    cy.get("#renderedDataSummary").should("contain.text", "video");

    cy.get(`#artifactList button[data-key="${NON_STOCK_RUN_ID}/reports/sim2real-report.json"]`).click();
    cy.wait("@loadArtifact");
    cy.get("#renderModeData").should("have.class", "is-active");
    cy.get("#artifactPreviewHost pre").should("contain.text", "promoted");

    cy.get(`#artifactList button[data-key="${NON_STOCK_RUN_ID}/logs/orchestrator.log"]`).click();
    cy.wait("@loadArtifact");
    cy.get("#artifactPreviewHost pre").should("contain.text", "loaded customer scene mesh");

    cy.get(`#artifactList button[data-key="${NON_STOCK_RUN_ID}/raw/custom-dynamics.fooz"]`).click();
    cy.wait("@loadArtifact");
    cy.get("#artifactPreviewHost").should("contain.text", "download");
    cy.get("#artifactPreviewHost a").should("have.attr", "href").and("include", "custom-dynamics.fooz");

    cy.get(`#artifactList button[data-key="${NON_STOCK_RUN_ID}/reports/sim2real.rrd"]`).click();
    cy.wait("@loadArtifact");
    cy.get("#renderModeRerun").should("have.class", "is-active");
    cy.get("#rerunFrame").should("have.attr", "src").and("include", "/rerun/");

    cy.get("#openRerun").click();
    cy.get("@windowOpen").should("have.been.called");
  });

  it("grounds complex chat queries and complex workflow YAML drafts", () => {
    cy.get("#chatInput").type(
      "For the non-stock customer run, what can I view, which artifact should I load first, and how do I keep Rerun interactive?",
      { delay: 0 },
    );
    cy.get("#chatSend").click();
    cy.wait("@chat");
    cy.get("#chatLog").should("contain.text", "Non-stock Sim2Real artifacts");
    cy.get("#chatLog").should("contain.text", NON_STOCK_RUN_ID);
    cy.get("#chatLog").should("contain.text", "Artifact browser");

    cy.get("#chatInput").type(
      "Draft a complex VLM/RL outer loop workflow YAML for non-stock assets with a quality gate and promote or loop-back transitions.",
      { delay: 0 },
    );
    cy.get("#chatSend").click();
    cy.wait("@chat");
    cy.get("#workflowYaml").should("contain.value", "cypress-vlm-rl-loop");
    cy.get("#workflowYaml").should("contain.value", "workbench.token_factory.reason");
    cy.get("#workflowYaml").should("contain.value", "loop_back");

    cy.get("#workflowYaml").clear().type(COMPLEX_WORKFLOW_YAML, { delay: 0 });
    cy.get("#workflowValidate").click();
    cy.wait("@workflowValidate");
    cy.get("#workflowName").should("contain.text", "cypress-vlm-rl-loop");
    cy.get("#workflowStates").should("contain.text", "vlm_gate");

    cy.get("#workflowPlan").click();
    cy.wait("@workflowPlan");
    cy.get("#workflowPlanHost").should("contain.text", "workbench.token_factory.reason");
    cy.get("#workflowPlanOutput").should("contain.text", "workbench.token_factory.reason");
  });

  it("covers mobile panels toggle and mobile chat auth flow", () => {
    cy.viewport("iphone-x");
    cy.visitMockAgent();
    cy.get("body").should("have.class", "mobile-agent");

    cy.get("#mobilePanelsToggle").click();
    cy.get("body").should("have.class", "mobile-show-panels");
    cy.get("#mobilePanelsToggle").should("have.attr", "aria-expanded", "true");

    cy.get("#mobileChatPassword").type("mock-password");
    cy.get("#mobileChatAuthBtn").click();
    cy.wait("@health");
    cy.get("body").should("have.class", "mobile-auth-ready");

    cy.get("#chatInput").type("mobile hello");
    cy.get("#chatSend").click();
    cy.wait("@chat");
    cy.get("#chatLog").should("contain.text", "mobile hello");
  });
});
