import { FIELD_IDS, STATIC_BUTTON_IDS, WORKFLOW_YAML } from "../support/e2e";

describe("NPA agent UI with mocked APIs", () => {
  beforeEach(() => {
    cy.visitMockAgent();
    cy.wait("@session");
    cy.wait("@simAssets");
    cy.wait("@cameras");
  });

  it("renders every static control and generated panel", () => {
    for (const id of STATIC_BUTTON_IDS) {
      cy.get(`#${id}`).should("exist");
    }
    for (const id of FIELD_IDS) {
      cy.get(`#${id}`).should("exist");
    }
    cy.get("#cameraCards button[data-action='select']").should("have.length.at.least", 1);
    cy.get("#cameraCards button[data-action='preview']").should("have.length.at.least", 1);
    cy.get("#workflowYaml").should("contain.value", "apiVersion: npa.workflow/v0.0.1");
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
    cy.get("#workflowPlanOutput").should("contain.text", "workbench.sim2real.status");
    cy.get("#workflowValidation").should("contain.text", "planned");

    cy.get("#workflowSubmitYaml").click();
    cy.wait("@workflowSubmitYaml");
    cy.get("#chatLog").should("contain.text", "Submitted npa.workflow YAML");
  });

  it("covers Sim2Real selection, run monitor, Rerun buttons, and run-data loading", () => {
    cy.window().then((win) => {
      cy.stub(win, "open").as("windowOpen");
    });

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
    cy.get("#chatLog").should("contain.text", "Loaded stock Franka");

    cy.get("#loadRerunViewer").click({ force: true });
    cy.get("#statusBar").should(($bar) => {
      expect($bar.text()).to.match(/Rerun|Reload/);
    });

    cy.get("#submitWorkflow").click();
    cy.wait("@submitSim2Real");
    cy.get("#chatLog").should("contain.text", "Submitted Sim2Real run");
    cy.get("#runSummary").should("contain.text", "submitted-run");

    cy.get("#workflowStatus").click();
    cy.wait("@workflowStatus");
    cy.get("#chatLog").should("contain.text", "Latest workflow status");

    cy.get("#runIdInput").clear().type("mock-run");
    cy.get("#loadRunData").click();
    cy.wait("@loadRun");
    cy.get("#chatLog").should("contain.text", "Loaded run context");
    cy.get("#runLog").should("contain.text", "mock run log");

    cy.get("#openRerun").click();
    cy.get("@windowOpen").should("have.been.called");
  });

  it("covers artifact discovery, dynamic artifact load button, and camera cards", () => {
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

    cy.get("#cameraCards button[data-action='select'][data-camera='wrist']").click({ force: true });
    cy.wait("@setCamera");
    cy.get("#activeCameraLabel").should("contain.text", "workspace");

    cy.get("#cameraCards button[data-action='preview'][data-camera='workspace']").click({ force: true });
    cy.wait("@cameraPreview");
    cy.get("#chatLog").should("contain.text", "Previewing workspace in Rerun");
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
