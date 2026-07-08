import { STATIC_BUTTON_IDS } from "../support/e2e";

const requiredLiveEnv = ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"];

function liveEnvAvailable() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name) || Cypress.env(name.replace("NPA_AGENT_", "agent"))));
}

function destructiveLiveEnabled() {
  const value = Cypress.env("NPA_AGENT_CYPRESS_LIVE_DESTRUCTIVE");
  return value === true || value === 1 || value === "1";
}

function liveRunId() {
  return Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || Cypress.env("NPA_AGENT_RUN_ID") || "";
}

function liveAgentRequest(path, options = {}) {
  const baseUrl = Cypress.env("agentBaseUrl") || Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl");
  const username = Cypress.env("agentUser") || Cypress.env("NPA_AGENT_USER");
  const password = Cypress.env("agentPassword") || Cypress.env("NPA_AGENT_PASSWORD");
  return cy.request({
    url: `${String(baseUrl || "").replace(/\/$/, "")}${path}`,
    auth: { username, password },
    ...options,
  });
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

  it("loads a live Sim2Real run as a Rerun artifact, not a stale JSON artifact", function () {
    const runId = liveRunId();
    if (!runId) {
      this.skip();
    }

    liveAgentRequest("/api/sim-viz/load-run", {
      method: "POST",
      body: { run_id: runId, camera: "workspace" },
    }).then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.have.property("ok", true);
      const simViz = response.body.sim_viz || {};
      expect(simViz.run_id).to.eq(runId);
      expect(simViz.artifact_render).to.eq("rerun");
      expect(simViz.artifact_key).to.match(/\/reports\/sim2real\.rrd$/);
      expect(simViz.artifact_uri).to.match(/\/reports\/sim2real\.rrd$/);
      expect(simViz.rrd_uri).to.match(/^file:\/\//);
      expect(simViz.rerun_ready).to.eq(true);
      expect(simViz.rerun_iframe_url).to.include("/rerun/recordings/sim2real.rrd");
    });

    liveAgentRequest("/api/sim-viz/status").then((response) => {
      expect(response.status).to.eq(200);
      const simViz = response.body || {};
      expect(simViz.active_run_id || simViz.run_id).to.eq(runId);
      expect(simViz.artifact_render).to.eq("rerun");
      expect(simViz.artifact_key).to.match(/\/reports\/sim2real\.rrd$/);
      expect(simViz.rerun_iframe_url).to.include("/rerun/recordings/sim2real.rrd");
    });

    cy.reload();
    cy.get("#simRunId", { timeout: 30000 }).should("contain.text", runId);
    cy.get("#rerunFrame")
      .should("have.attr", "src")
      .and("include", "/rerun/recordings/sim2real.rrd");
    cy.get("#statusBar").should("not.contain.text", "Non-RRD artifact loaded");
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
