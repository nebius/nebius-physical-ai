import { STATIC_BUTTON_IDS } from "../support/e2e";

const requiredLiveEnv = ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"];

function liveEnvAvailable() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name) || Cypress.env(name.replace("NPA_AGENT_", "agent"))));
}

function destructiveLiveEnabled() {
  const value = Cypress.env("NPA_AGENT_CYPRESS_LIVE_DESTRUCTIVE");
  return value === true || value === 1 || value === "1";
}

describe("NPA agent UI against live infra", () => {
  before(function () {
    if (!liveEnvAvailable()) {
      this.skip();
    }
  });

  beforeEach(() => {
    cy.visitLiveAgent();
    cy.get("meta[name='npa-ui-version']").should("have.attr", "content").and("match", /^\d+$/);
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
  });

  it("loads deployed UI and every shipped button is present", () => {
    for (const id of STATIC_BUTTON_IDS) {
      cy.get(`#${id}`).should("exist");
    }
    cy.get("#chatForm").should("exist");
    cy.get("#workflowYaml").should("exist");
    cy.get("#cameraCards", { timeout: 30000 }).should("exist");
    cy.get("#rerunFrame").should("exist");
  });

  it("drives safe live controls through the browser", () => {
    cy.get("#chatActionS3").click();
    cy.get("#chatInput").should("contain.value", "configure S3");
    cy.get("#chatActionCosmos").click();
    cy.get("#chatInput").should("contain.value", "Cosmos3");
    cy.get("#chatActionWatch").click();
    cy.get("#chatInput").should("contain.value", "Rerun");
    cy.get("#chatActionWorkflow").click();
    cy.get("#chatInput").should("contain.value", "2-step sim2real workflow");

    cy.get("#workflowStatus").click();
    cy.get("#runSummary", { timeout: 30000 }).should("contain.text", "status");

    cy.get("#artifactRefreshRuns").click();
    cy.get("#artifactList", { timeout: 30000 }).should("contain.text", "Runs discovered");

    cy.get("#loadFrankaRerun").click();
    cy.get("#statusBar", { timeout: 120000 }).should(($bar) => {
      const text = $bar.text();
      expect(text).to.match(/done|Ready|SUCCESS|Rerun/i);
    });

    cy.get("#openRerun").should("be.visible");
  });

  it("submits Sim2Real from the UI when live destructive Cypress is enabled", function () {
    if (!destructiveLiveEnabled()) {
      this.skip();
    }
    cy.get("#submitWorkflow").click();
    cy.get("#chatLog", { timeout: 180000 }).should("contain.text", "Submitted Sim2Real run");
    cy.get("#runSummary", { timeout: 180000 }).should("contain.text", "run");
  });
});
