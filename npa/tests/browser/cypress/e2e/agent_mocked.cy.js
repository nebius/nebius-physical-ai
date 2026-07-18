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
    // From Viewer, Chat tab opens the drawer; Full chat expands Stages/Workflow.
    cy.get("#tabChat").click();
    cy.get("#panelChat").should("have.class", "chat-drawer-open");
    cy.get("#openFullChatTab").click();
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
    cy.get("#panelChat").should("have.class", "chat-drawer-open");
    cy.get("#openFullChatTab").click();
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

  it("opens bottom-right chat widget on Viewer without covering Rerun permanently", () => {
    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#chatDrawerToggle").should("be.visible");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");

    cy.get("#chatDrawerToggle").click();
    cy.get("#panelChat").should("have.class", "chat-drawer-open");
    cy.get("#chatDrawerToggle").should("have.class", "is-open");
    cy.get("#chatDrawerClose").should("be.visible").click();
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#rerunBundleCover").should("have.attr", "hidden");

    // Chat main-tab from Viewer opens the drawer instead of leaving Rerun.
    cy.get("#tabChat").click();
    cy.get("#panelChat").should("have.class", "chat-drawer-open");
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#openFullChatTab").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("body").should("not.have.class", "viewer-focus");
  });

  it("keeps local Workflow YAML edits across refresh-driven run loads", () => {
    const edited = "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nmetadata:\n  name: local-edit\n";
    cy.get("#workflowYaml").clear().type(edited, { delay: 0 });
    cy.get("#tabRerun").click();
    cy.get("#runIdInput").clear({ force: true }).type("cosmos-reason-run", { force: true });
    cy.get("#loadRunData").click({ force: true });
    cy.wait("@loadRun");
    cy.get("#tabChat").click();
    cy.get("#workflowYaml").should("contain.value", "local-edit");
  });

  it("Stages Load prefers pasted run id over a stale dropdown selection", () => {
    cy.get("#tabChat").click();
    cy.get("#stagesRunSelect").select("mock-run");
    cy.get("#stagesRunInput").clear().type("cosmos-reason-run", { delay: 0 });
    cy.get("#stagesLoadRun").click();
    cy.wait("@loadRun");
    cy.get("#runSummary").should("contain.text", "cosmos-reason-run");
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
      expect(stats.variance).to.be.greaterThan(35);
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

  it("Describe this stays metadata-only for uniform gray canvases", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("gray");
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@chat").then((interception) => {
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
      cy.window().then(async (win) => {
        const api = win.__NPA_AGENT_TEST__;
        const result = await api.waitForQualityRerunFrame(2500);
        if (item.expectFrame) {
          expect(result.quality).to.eq("rendered");
          expect(result.dataUrl).to.match(/^data:image\/jpeg/);
        } else {
          expect(result.dataUrl).to.eq("");
          expect(result.quality).to.be.oneOf(["unavailable", "missing"]);
        }
      });
    });
  });
});
