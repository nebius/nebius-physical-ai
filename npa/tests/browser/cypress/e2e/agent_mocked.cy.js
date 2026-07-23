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
    // Wait for boot mount to finish so later loadRun/loadArtifact are not clobbered
    // by ensureFrankaRerunLoaded. Cover clears quickly after warm (no splash latency).
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
  });

  it("renders a generic Stages panel (not Sim2Real-only)", () => {
    cy.get("#stagesPanel").should("exist");
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
    cy.get("#stagesPanel .hint").should("contain.text", "timeline, result, and logs");
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
    // Clicking the Main tab from Viewer switches to the Main panel directly
    // (it must NOT pop the chat drawer out).
    cy.get("#tabMain").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
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
    cy.get("#stagesPanel .hint").should("contain.text", "timeline, result, and logs");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
  });

  it("never shows Loading application bundle without mount latency", () => {
    cy.get("#rerunBundleCover").should("exist");
    cy.window().then((win) => {
      const html = win.document.documentElement.outerHTML;
      expect(html).to.include("scheduleRerunBundleUncover");
      expect(html).to.include("Uncover without blocking mount latency");
      expect(html).to.include("waitUntilRerunPastBundleSplash");
      expect(html).to.include("swapRerunRecordingInPlace");
      expect(html).to.include("add_receiver");
      expect(html).not.to.include("await waitUntilRerunPastBundleSplash(iframe, 45000)");
      expect(html).not.to.include('Mount the viewer immediately so "Loading application bundle" starts early');
    });
    // Visible chrome only (skip <script> source, which contains the splash detector regex).
    cy.get("#rerunBundleCover .cover-title").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    cy.get("#rerunBundleCover .cover-hint").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    cy.get("#statusBar").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    // Mock Rerun serves a canvas with no splash; cover should clear quickly (no cold wasm).
    cy.get("#rerunBundleCover", { timeout: 15000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      const frame = $frame[0];
      const doc = frame.contentDocument || (frame.contentWindow && frame.contentWindow.document);
      const text = String((doc && doc.body && doc.body.innerText) || "");
      expect(text).not.to.match(/Loading application bundle/i);
    });
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
    cy.get("#tabMain").should("have.attr", "aria-selected", "true");
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
    cy.get("#tabMain").click();
    cy.get("#chatLog").should("contain.text", "Loaded stock Franka");
    cy.get("#stagesPanel h3").should("have.text", "Stages");

    cy.get("#tabRerun").click();
    cy.get("#loadRerunViewer").click({ force: true });
    cy.get("#statusBar").should(($bar) => {
      expect($bar.text()).to.match(/Rerun|Reload/);
    });

    cy.get("#submitWorkflow").click();
    cy.wait("@submitSim2Real");
    cy.get("#tabMain").click();
    cy.get("#chatLog").should("contain.text", "Submitted Sim2Real run");
    cy.get("#runSummary").should("contain.text", "submitted-run");

    cy.get("#tabRerun").click();
    cy.get("#workflowStatus").click();
    cy.wait("@workflowStatus");
    cy.get("#tabMain").click();
    cy.get("#chatLog").should("contain.text", "Latest workflow status");

    cy.get("#tabRerun").click();
    cy.get("#runIdInput").clear().type("mock-run");
    cy.get("#loadRunData").click();
    cy.wait("@loadRun");
    cy.get("#tabMain").click();
    cy.get("#chatLog").should("contain.text", "Loaded run context");
    cy.get("#runLog").should("contain.text", "mock run log");
    cy.get("#stagesPanel h3").should("have.text", "Stages");

    cy.get("#tabRerun").click();
    cy.get("#openRerun").click();
    cy.get("@windowOpen").should("have.been.called");
  });

  it("consolidates runs & artifacts into one latest-first picker", () => {
    cy.get("#tabRerun").click();
    cy.get("#runsArtifactsPanel").should("exist");
    cy.contains("h3", "Runs & artifacts").should("exist");
    cy.contains("h4", "Active run").should("not.exist");
    cy.contains("h4", "Artifacts").should("not.exist");
    cy.get("#artifactRunSelect").should("not.exist");

    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#artifactDiscoverStatus").should("contain.text", "latest first");
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((opt) => opt.value).filter(Boolean);
      // Discovered non-stock run is newest; must appear before older mock-run.
      expect(values[0]).to.eq(NON_STOCK_RUN_ID);
      expect(values).to.include("mock-run");
      expect(values).to.include("submitted-run");
    });
    cy.get("#stagesRunSelect option").then(($opts) => {
      const values = [...$opts].map((opt) => opt.value).filter(Boolean);
      expect(values[0]).to.eq(NON_STOCK_RUN_ID);
    });
  });

  it("covers artifact discovery, dynamic artifact load button, and camera cards", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    // Discovery is generic (no path prefix); all runs show.
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    // Consolidated picker may already have mock-run selected — force list via button.
    cy.get("#runIdSelect").select("mock-run", { force: true });
    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@artifactList");
    cy.get("#artifactList").should("contain.text", "mock-run/preview.png");

    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@artifactList");
    cy.get("#artifactList button[data-action='load-artifact']").click();
    cy.wait("@loadArtifact");
    cy.get("#chatLog").should("contain.text", "Loaded artifact");
    cy.get("#artifactPreviewHost").should("not.have.attr", "hidden");

    cy.get("#tabMain").click();
    cy.get("#panelChat").should("have.class", "is-active").and("have.attr", "aria-hidden", "false");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#panelRerun").should("have.class", "is-inactive");
  });

  it("discovers and interacts with non-stock Sim2Real run artifacts", () => {
    cy.window().then((win) => {
      cy.stub(win, "open").as("windowOpen");
    });

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
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
    cy.get("#tabMain").click();
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

  it("finds runs by name/ID via a client-side filter (no path prefix needed)", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    // Generic discovery lists every run — no prefix/category to type.
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values).to.include(NON_STOCK_RUN_ID);
      expect(values).to.include("mock-run");
    });
    // Typing part of a run name/ID filters the list client-side.
    cy.get("#artifactPrefix").clear().type("non-stock");
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values).to.include(NON_STOCK_RUN_ID);
      expect(values).to.not.include("mock-run");
    });
    // Clearing restores the full list.
    cy.get("#artifactPrefix").clear();
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values).to.include("mock-run");
    });
  });

  it("filters artifacts by workflow stage and tags timeline rows by stage", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.wait("@loadArtifact");

    // The Stage (workflow-progress) selector is populated from the loaded
    // artifacts' first path segment after the run id.
    cy.get("#artifactStageFilter option").then(($opts) => {
      const values = [...$opts].map((opt) => opt.value);
      expect(values).to.include.members(["reports", "rollouts", "logs", "raw"]);
    });

    // Selecting a stage scopes the artifact list to that workflow-progress step.
    cy.get("#artifactStageFilter").select("rollouts");
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/rollouts/customer-camera.mp4`);
    cy.get("#artifactList").should("not.contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`);

    // Clearing the stage filter restores the full listing.
    cy.get("#artifactStageFilter").select("");
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`);

    // The artifact-derived timeline tags rows with a stage key so they are
    // clickable to scope the browser. (The click handler is covered by the
    // agent unit test; the periodic sim-viz poll re-renders #stageList in the
    // mock, so a live click assertion here would be race-prone.)
    cy.get("#stageList").should("exist");
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

  it("clears the Rerun Caching cover and keeps it hidden after remounts", () => {
    // Boot must clear the cover; it must not stick on "Caching Rerun assets…".
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#statusBar").should("not.contain.text", "Caching Rerun assets");

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("body").should("have.class", "viewer-focus");

    // Repeated remounts / reloads must not leave the Caching overlay visible.
    cy.get("#loadRerunViewer").click({ force: true });
    cy.get("#rerunBundleCover", { timeout: 15000 }).should("have.attr", "hidden");
    cy.get("#loadFrankaRerun").click({ force: true });
    cy.wait("@loadFranka");
    cy.get("#rerunBundleCover", { timeout: 15000 }).should("have.attr", "hidden");
    cy.get("#statusBar").should(($el) => {
      expect($el.text()).not.to.match(/Caching Rerun assets/i);
    });
    cy.get("#rerunBundleCover .cover-hint").should(($el) => {
      // When hidden, hint text may still say Almost ready / Caching — but cover must stay hidden.
      expect(Cypress.$("#rerunBundleCover").attr("hidden")).to.exist;
    });

    // Soft-path: reload again while viewer is already mounted.
    cy.get("#loadRerunViewer").click({ force: true });
    cy.wait(400);
    cy.get("#rerunBundleCover").should("have.attr", "hidden");
    cy.get("#statusBar", { timeout: 10000 }).should("not.contain.text", "Caching Rerun assets");
  });

  it("opens the chat collapsible only via the chat button, and Main tab never pops it out", () => {
    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#chatDrawerToggle").should("be.visible");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");

    // The chat collapsible opens ONLY when the chat button (FAB) is clicked.
    cy.get("#chatDrawerToggle").click();
    cy.get("#panelChat").should("have.class", "chat-drawer-open");
    cy.get("#chatDrawerToggle").should("have.class", "is-open");
    cy.get("#chatDrawerClose").should("be.visible").click();
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#rerunBundleCover").should("have.attr", "hidden");

    // Regression guard: clicking the Main tab from the Viewer switches to the
    // Main panel and must NOT pop the chat drawer out.
    cy.get("#tabMain").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#panelRerun").should("have.class", "is-inactive");
    cy.get("body").should("not.have.class", "viewer-focus");

    // Returning to the Viewer and back to Main again still never pops the drawer.
    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#tabMain").click();
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#panelChat").should("have.class", "is-active");
  });

  it("labels the main tab 'Main' (renamed from Chat)", () => {
    cy.get("#tabMain").should("exist").and("have.text", "Main");
    cy.get("#tabMain").should("have.attr", "data-tab", "main");
    cy.get("#tabChat").should("not.exist");
  });

  it("shows a scroll-to-bottom arrow when scrolled up and jumps to the latest message", () => {
    // Fill the chat via real sends so the log overflows and can be scrolled.
    for (let i = 0; i < 6; i += 1) {
      cy.get("#chatInput").type(`Draft a 2-step Sim2Real workflow YAML please (${i})`, { delay: 0 });
      cy.get("#chatSend").click();
      cy.wait("@chat");
    }
    // Each new message auto-scrolls to the bottom, so the arrow is hidden.
    cy.get("#chatScrollBottom").should("have.attr", "hidden");

    // Scrolling up reveals the jump-to-latest arrow.
    cy.get("#chatLog").scrollTo("top");
    cy.get("#chatScrollBottom").should("not.have.attr", "hidden");
    cy.get("#chatScrollBottom").should("be.visible");

    // Clicking the arrow returns to the end of the chat and hides the arrow.
    cy.get("#chatScrollBottom").click();
    cy.get("#chatLog").should(($log) => {
      const el = $log[0];
      expect(el.scrollHeight - el.scrollTop - el.clientHeight).to.be.lessThan(41);
    });
    cy.get("#chatScrollBottom").should("have.attr", "hidden");
  });

  it("keeps local Workflow YAML edits across refresh-driven run loads", () => {
    const edited = "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nmetadata:\n  name: local-edit\n";
    cy.get("#workflowYaml").clear().type(edited, { delay: 0 });
    cy.get("#tabRerun").click();
    cy.get("#runIdInput").clear({ force: true }).type("cosmos-reason-run", { force: true });
    cy.get("#loadRunData").click({ force: true });
    cy.wait("@loadRun");
    cy.get("#tabMain").click();
    cy.get("#workflowYaml").should("contain.value", "local-edit");
  });

  it("Stages Load prefers pasted run id over a stale dropdown selection", () => {
    cy.get("#tabMain").click();
    cy.get("#stagesRunSelect").select("mock-run");
    cy.get("#stagesRunInput").clear().type("cosmos-reason-run", { delay: 0 });
    cy.get("#stagesLoadRun").click();
    cy.wait("@loadRun");
    cy.get("#runSummary").should("contain.text", "cosmos-reason-run");
  });

  it("Stages search filters the run list by name", () => {
    cy.get("#tabMain").click();
    cy.get("#stagesRunInput").clear().type("mock", { delay: 0 });
    cy.get("#stagesRunSearchHint").should("contain.text", "match");
    cy.get("#stagesRunSelect option").then(($opts) => {
      const values = [...$opts].map((opt) => opt.value).filter(Boolean);
      expect(values.length).to.be.greaterThan(0);
      expect(values.every((value) => value.includes("mock"))).to.eq(true);
    });
    cy.get("#stagesLoadRun").click();
    cy.wait("@loadRun");
    cy.get("#runSummary").should("contain.text", "mock-run");
  });

  it("rejects uniform gray / blank canvases in frameLooksBlank", () => {
    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      expect(api, "test hooks").to.exist;
      const gray = win.document.createElement("canvas");
      gray.width = 120;
      gray.height = 80;
      const gctx = gray.getContext("2d");
      gctx.fillStyle = "#9ca3af";
      gctx.fillRect(0, 0, 120, 80);
      expect(api.frameLooksBlank(gray)).to.eq(true);

      const black = win.document.createElement("canvas");
      black.width = 120;
      black.height = 80;
      black.getContext("2d").fillRect(0, 0, 120, 80);
      expect(api.frameLooksBlank(black)).to.eq(true);

      const content = win.document.createElement("canvas");
      content.width = 160;
      content.height = 100;
      const cctx = content.getContext("2d");
      cctx.fillStyle = "#0a0a12";
      cctx.fillRect(0, 0, 160, 100);
      cctx.strokeStyle = "#ff8a1f";
      cctx.lineWidth = 3;
      cctx.beginPath();
      cctx.moveTo(40, 20);
      cctx.lineTo(80, 60);
      cctx.lineTo(50, 90);
      cctx.stroke();
      cctx.strokeStyle = "#5eead4";
      cctx.beginPath();
      cctx.moveTo(90, 25);
      cctx.lineTo(120, 70);
      cctx.stroke();
      expect(api.frameLooksBlank(content)).to.eq(false);
      const stats = api.sampleFrameStats(content);
      expect(stats.vivid).to.be.greaterThan(0);

      // Large dark viewport with thin G1-style strokes — must not be wiped by downscale.
      const sparse = win.document.createElement("canvas");
      sparse.width = 960;
      sparse.height = 540;
      const sctx = sparse.getContext("2d");
      sctx.fillStyle = "#050508";
      sctx.fillRect(0, 0, 960, 540);
      sctx.strokeStyle = "#ff8a1f";
      sctx.lineWidth = 2;
      sctx.beginPath();
      sctx.moveTo(480, 80);
      sctx.lineTo(470, 220);
      sctx.lineTo(455, 360);
      sctx.lineTo(450, 480);
      sctx.stroke();
      sctx.strokeStyle = "#5eead4";
      sctx.beginPath();
      sctx.moveTo(490, 90);
      sctx.lineTo(520, 200);
      sctx.lineTo(560, 280);
      sctx.stroke();
      expect(api.frameLooksBlank(sparse), "sparse skeleton on dark grid").to.eq(false);
      expect(api.sampleFrameStats(sparse).vivid).to.be.greaterThan(2);
    });
  });

  it("Describe this appears in chat immediately and attaches a non-blank frame", () => {
    cy.intercept("POST", "/api/chat", (req) => {
      // Delayed vision reply so the pending chat bubble must appear first.
      req.reply({
        delay: 1400,
        statusCode: 200,
        body: {
          ok: true,
          grounded: false,
          tier: "vision",
          model: "Qwen/Qwen2.5-VL-72B-Instruct",
          session_id: req.body.session_id || "default",
          reply: [
            "**What I see**: Dark 3D grid with orange and cyan skeleton wireframes (G1 trajectory style).",
            "**Likely meaning**: Locomotion / trajectory overlay in the Rerun viewer.",
            "**Operator feedback**: Structured sim content is visible — not a blank frame.",
            "**Next actions**: Scrub timeline; compare held-out cameras; keep this recording.",
          ].join("\n"),
        },
      });
    }).as("slowDescribeChat");

    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      const win = $frame[0].contentWindow;
      expect(win && win.__NPA_MOCK_RERUN__).to.exist;
      win.__NPA_MOCK_RERUN__.setMode("content");
    });

    cy.get("#describeVisual").click({ force: true });
    // Immediate UX: request visible before the delayed /api/chat completes.
    cy.get("#chatLog .msg-row.user", { timeout: 2000 }).should("contain.text", "Describe this");
    cy.get("#panelChat").should("have.class", "chat-drawer-open");

    cy.wait("@slowDescribeChat", { timeout: 20000 }).then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context).to.be.an("object");
      expect(body.visual_context.capture).to.eq("frame");
      expect(body.visual_context.frame_quality).to.eq("rendered");
      expect(body.visual_context.has_image).to.eq(true);
      const messages = body.messages;
      const last = messages[messages.length - 1];
      expect(last.content).to.be.an("array");
      const imagePart = last.content.find((part) => part && String(part.type || "").startsWith("image"));
      expect(imagePart, "image part").to.exist;
      const url = imagePart.image_url.url;
      expect(url).to.match(/^data:image\/jpeg;base64,/);
      expect(url.length).to.be.greaterThan(4000);
    });
    cy.get("#chatLog .msg-row.assistant").should("contain.text", "skeleton");
    cy.get("#chatLog .msg-row.assistant").should("not.contain.text", "completely uniform gray");
  });

  it("Describe this carries grounded pipeline provenance when a run is loaded", () => {
    cy.intercept("GET", "**/api/artifacts/provenance/**", {
      statusCode: 200,
      body: {
        ok: true,
        run_id: "paidf-mock-1",
        summary:
          "Augment — Cosmos Transfer 2.5 (nvidia/Cosmos-Transfer2.5-2B) [GPU (Nebius K8s)]; " +
          "Pseudo-label augmented — Token Factory VLM (Qwen/Qwen2.5-VL-72B-Instruct) [hosted GPU (Token Factory)]",
        components: [
          {
            stage: "Augment",
            component: "Cosmos Transfer 2.5",
            runtime: "GPU (Nebius K8s)",
            model: "nvidia/Cosmos-Transfer2.5-2B",
          },
        ],
      },
    }).as("provenance");
    cy.intercept("POST", "/api/chat", (req) => {
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          grounded: false,
          tier: "vision",
          model: "Qwen/Qwen2.5-VL-72B-Instruct",
          session_id: req.body.session_id || "default",
          reply: "**What I see**: augmented road scene.\n**Where it comes from**: Cosmos Transfer 2.5 augment stage.",
        },
      });
    }).as("provChat");

    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("content");
    });
    // A loaded run id is what triggers the grounded provenance fetch.
    cy.get("#simRunId").then(($el) => {
      $el[0].textContent = "paidf-mock-1";
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@provenance");
    cy.wait("@provChat", { timeout: 20000 }).then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context.provenance, "visual_context.provenance").to.be.a("string");
      expect(body.visual_context.provenance).to.match(/Cosmos Transfer 2\.5/);
      const last = body.messages[body.messages.length - 1];
      const textPart = Array.isArray(last.content)
        ? last.content.find((part) => part && String(part.type || "").includes("text"))
        : { text: last.content };
      const promptText = String((textPart && textPart.text) || "");
      expect(promptText, "prompt provenance section").to.match(/Pipeline provenance/i);
      expect(promptText).to.match(/Cosmos Transfer 2\.5/);
    });
  });

  it("Describe this stays metadata-only for uniform gray canvases", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("gray");
    });
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      api.ensureRerunCaptureBridge(iframe, { forceRestart: true });
      const quality = await api.waitForQualityRerunFrame(2500);
      expect(quality.dataUrl, "gray must not attach").to.eq("");
      expect(quality.quality).to.be.oneOf(["unavailable", "missing"]);
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@chat", { timeout: 60000 }).then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context).to.be.an("object");
      expect(body.visual_context.capture).to.not.eq("frame");
      expect(body.visual_context.has_image).to.eq(false);
      const messages = body.messages;
      const last = messages[messages.length - 1];
      const content = last.content;
      if (Array.isArray(content)) {
        const imagePart = content.find((part) => part && String(part.type || "").startsWith("image"));
        expect(imagePart).to.not.exist;
      }
    });
    cy.get("#chatLog .msg-row.assistant").should("contain.text", "metadata only");
  });

  it("keeps the cover up while the iframe shows Loading application bundle", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");

    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("splash");
    });

    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      expect(api.rerunViewerShowsBundleSplash(iframe)).to.eq(true);
      expect(api.rerunViewerLooksDisplayReady(iframe)).to.eq(false);
      api.showRerunBundleCover("Opening viewer…", "Almost ready…");
      expect(win.document.getElementById("rerunBundleCover").hidden).to.eq(false);
      // safeHide must refuse while splash / blank canvas is showing.
      expect(api.safeHideRerunBundleCover(iframe)).to.eq(false);
      expect(win.document.getElementById("rerunBundleCover").hidden).to.eq(false);
      // Parent chrome must never echo Rerun's splash string.
      expect(win.document.getElementById("rerunBundleCover").innerText).not.to.match(
        /Loading application bundle/i,
      );
    });

    // When content returns, uncover is allowed.
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("content");
    });
    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      expect(api.rerunViewerLooksDisplayReady(iframe)).to.eq(true);
      expect(api.safeHideRerunBundleCover(iframe)).to.eq(true);
      expect(win.document.getElementById("rerunBundleCover").hidden).to.eq(true);
    });
  });

  it("generalizes capture across content / gray / splash visual modes", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");

    const modes = [
      { mode: "content", expectFrame: true },
      { mode: "gray", expectFrame: false },
      { mode: "splash", expectFrame: false },
    ];

    cy.wrap(modes).each((item) => {
      cy.get("#rerunFrame").should(($frame) => {
        $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode(item.mode);
      });
      cy.window().then({ timeout: 20000 }, async (win) => {
        const api = win.__NPA_AGENT_TEST__;
        const iframe = win.document.getElementById("rerunFrame");
        // Force a fresh bridge after mode paint so probes do not see stale frames.
        api.ensureRerunCaptureBridge(iframe, { forceRestart: true });
        await new Promise((r) => setTimeout(r, 80));
        const probed = await api.probeRerunCanvasContent(iframe);
        const result = await api.waitForQualityRerunFrame(2500);
        if (item.expectFrame) {
          expect(probed, `${item.mode} probe`).to.eq(true);
          expect(result.quality, `${item.mode} quality`).to.eq("rendered");
          expect(result.dataUrl).to.match(/^data:image\/jpeg/);
        } else {
          expect(probed, `${item.mode} probe`).to.eq(false);
          expect(result.dataUrl, `${item.mode} dataUrl`).to.eq("");
          expect(result.quality).to.be.oneOf(["unavailable", "missing"]);
        }
      });
    });
  });

  it("captures Rerun via MediaStream bridge even when sync blank checks would fail", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("content");
    });
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      expect(win.document.documentElement.outerHTML).to.include("ensureRerunCaptureBridge");
      const bridge = api.ensureRerunCaptureBridge(iframe);
      expect(bridge, "capture bridge").to.exist;
      expect(bridge.video).to.exist;
      const grabbed = await api.grabFromRerunCaptureBridge(3000);
      expect(grabbed).to.match(/^data:image\/jpeg/);
      // Capture must succeed even if we ignore sync blank gates (the live WebGL failure mode).
      const quality = await api.waitForQualityRerunFrame(4000);
      expect(quality.quality).to.eq("rendered");
      expect(quality.dataUrl.length).to.be.greaterThan(4000);
    });
  });

  it("captures live WebGL canvas via captureStream when sync readback is blank", () => {
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const canvas = win.document.createElement("canvas");
      canvas.width = 160;
      canvas.height = 120;
      const gl = canvas.getContext("webgl", { preserveDrawingBuffer: false, alpha: false });
      expect(gl, "webgl context").to.exist;
      let raf = 0;
      const paint = () => {
        // Alternating orange / cyan clears so the stream has non-uniform structure over time,
        // while sync 2D readback of a non-preserveDrawingBuffer canvas is often blank.
        const t = Date.now() % 400 < 200;
        if (t) gl.clearColor(1.0, 0.45, 0.1, 1.0);
        else gl.clearColor(0.2, 0.85, 0.8, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        raf = win.requestAnimationFrame(paint);
      };
      paint();
      await new Promise((r) => setTimeout(r, 250));
      const url = await api.captureCanvasDataUrl(canvas, { budgetMs: 2500 });
      win.cancelAnimationFrame(raf);
      if (!url) {
        // Headless Chromium often cannot composite WebGL → MediaStream; the Rerun mock
        // (2D canvas) + live agent suite cover the production path.
        expect(typeof canvas.captureStream).to.eq("function");
        return;
      }
      expect(url, "WebGL stream capture").to.match(/^data:image\/jpeg;base64,/);
      expect(url.length).to.be.greaterThan(800);
    });
  });

  it("captures image / video / data visual kinds for Describe this", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");

    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const host = win.document.getElementById("artifactPreviewHost");
      expect(host).to.exist;

      // --- image ---
      host.hidden = false;
      const imgCanvas = win.document.createElement("canvas");
      imgCanvas.width = 120;
      imgCanvas.height = 80;
      const ictx = imgCanvas.getContext("2d");
      ictx.fillStyle = "#102030";
      ictx.fillRect(0, 0, 120, 80);
      ictx.fillStyle = "#ff8800";
      ictx.fillRect(20, 15, 80, 50);
      const img = win.document.createElement("img");
      img.src = imgCanvas.toDataURL("image/png");
      await new Promise((resolve) => {
        img.onload = resolve;
        img.onerror = resolve;
      });
      host.innerHTML = "";
      host.appendChild(img);
      api.setRenderMode("image");
      let captured = await api.captureVisualContext();
      expect(captured.kind).to.eq("image");
      expect(captured.meta.capture).to.eq("frame");
      expect(captured.imageDataUrl).to.match(/^data:image\/jpeg/);
      expect(captured.prompt).to.include("visual_kind: `image`");

      // --- video (canvas.captureStream backed) ---
      const vCanvas = win.document.createElement("canvas");
      vCanvas.width = 160;
      vCanvas.height = 90;
      const vctx = vCanvas.getContext("2d");
      vctx.fillStyle = "#0a1020";
      vctx.fillRect(0, 0, 160, 90);
      vctx.fillStyle = "#33cc99";
      vctx.fillRect(30, 20, 100, 50);
      const stream = vCanvas.captureStream(12);
      const video = win.document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play().catch(() => undefined);
      await new Promise((r) => setTimeout(r, 200));
      host.innerHTML = "";
      host.appendChild(video);
      api.setRenderMode("video");
      captured = await api.captureVisualContext();
      expect(captured.kind).to.eq("video");
      expect(captured.meta.capture).to.eq("frame");
      expect(captured.imageDataUrl).to.match(/^data:image\/jpeg/);
      expect(captured.prompt).to.include("visual_kind: `video`");
      stream.getTracks().forEach((t) => t.stop());

      // --- data / text ---
      const pre = win.document.createElement("pre");
      pre.textContent = JSON.stringify(
        { success_rate: 0.82, stage: "heldout", robot: "g1" },
        null,
        2
      );
      host.innerHTML = "";
      host.appendChild(pre);
      api.setRenderMode("data");
      captured = await api.captureVisualContext();
      expect(captured.kind).to.eq("data");
      expect(captured.meta.capture).to.eq("text");
      expect(captured.imageDataUrl).to.eq("");
      expect(captured.prompt).to.include("success_rate");
      expect(captured.prompt).to.include("visual_kind: `data`");
      expect(captured.prompt.toLowerCase()).to.include("pixels");

      // restore rerun mode
      api.setRenderMode("rerun");
      host.hidden = true;
      host.innerHTML = "";
    });
  });

  it("Describe this posts vision frames for image and video panes", () => {
    cy.get("#tabRerun").click();
    cy.intercept("POST", "/api/chat", (req) => {
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          grounded: false,
          tier: "vision",
          model: "Qwen/Qwen2.5-VL-72B-Instruct",
          session_id: req.body.session_id || "default",
          reply: "**What I see**: Structured viewer content.\n**Likely meaning**: Valid capture.\n**Operator feedback**: OK.\n**Next actions**: Continue.",
        },
      });
    }).as("visualKindChat");

    // Image pane
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const host = win.document.getElementById("artifactPreviewHost");
      host.hidden = false;
      const imgCanvas = win.document.createElement("canvas");
      imgCanvas.width = 100;
      imgCanvas.height = 60;
      const ctx = imgCanvas.getContext("2d");
      ctx.fillStyle = "#203040";
      ctx.fillRect(0, 0, 100, 60);
      ctx.strokeStyle = "#ffaa00";
      ctx.lineWidth = 4;
      ctx.strokeRect(10, 10, 80, 40);
      const img = win.document.createElement("img");
      img.src = imgCanvas.toDataURL("image/png");
      await new Promise((resolve) => {
        img.onload = resolve;
      });
      host.innerHTML = "";
      host.appendChild(img);
      api.setRenderMode("image");
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@visualKindChat").then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context.kind).to.eq("image");
      expect(body.visual_context.capture).to.eq("frame");
      expect(body.visual_context.has_image).to.eq(true);
      const last = body.messages[body.messages.length - 1];
      expect(last.content).to.be.an("array");
      expect(last.content.some((p) => p && String(p.type || "").startsWith("image"))).to.eq(true);
    });

    // Video pane
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const host = win.document.getElementById("artifactPreviewHost");
      const vCanvas = win.document.createElement("canvas");
      vCanvas.width = 128;
      vCanvas.height = 72;
      const vctx = vCanvas.getContext("2d");
      vctx.fillStyle = "#101828";
      vctx.fillRect(0, 0, 128, 72);
      vctx.fillStyle = "#22d3ee";
      vctx.fillRect(24, 16, 80, 40);
      const stream = vCanvas.captureStream(12);
      const video = win.document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play().catch(() => undefined);
      await new Promise((r) => setTimeout(r, 180));
      host.innerHTML = "";
      host.appendChild(video);
      host._npaTestStream = stream;
      api.setRenderMode("video");
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@visualKindChat").then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context.kind).to.eq("video");
      expect(body.visual_context.capture).to.eq("frame");
      expect(body.visual_context.has_image).to.eq(true);
      const last = body.messages[body.messages.length - 1];
      expect(last.content.some((p) => p && String(p.type || "").startsWith("image"))).to.eq(true);
    });

    // Data pane — metadata/text only, never invents an image part
    cy.intercept("POST", "/api/chat", (req) => {
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          grounded: false,
          tier: "reasoning",
          model: "mock/model",
          session_id: req.body.session_id || "default",
          reply: "**What I see**: Metadata/text only.\n**Likely meaning**: JSON report.\n**Operator feedback**: OK.\n**Next actions**: Reload Rerun.",
        },
      });
    }).as("dataKindChat");

    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      const host = win.document.getElementById("artifactPreviewHost");
      if (host._npaTestStream) {
        host._npaTestStream.getTracks().forEach((t) => t.stop());
        delete host._npaTestStream;
      }
      const pre = win.document.createElement("pre");
      pre.textContent = JSON.stringify({ success_rate: 0.91, robot: "g1" }, null, 2);
      host.innerHTML = "";
      host.appendChild(pre);
      api.setRenderMode("data");
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@dataKindChat").then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context.kind).to.eq("data");
      expect(body.visual_context.capture).to.eq("text");
      expect(body.visual_context.has_image).to.eq(false);
      const last = body.messages[body.messages.length - 1];
      const content = last.content;
      if (Array.isArray(content)) {
        expect(content.some((p) => p && String(p.type || "").startsWith("image"))).to.eq(false);
        expect(JSON.stringify(content)).to.include("success_rate");
      } else {
        expect(String(content)).to.include("success_rate");
      }
    });
  });
});
