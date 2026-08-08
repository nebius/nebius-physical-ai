const requiredLiveEnv = ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"];

function liveEnvAvailable() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name)));
}

function runId() {
  return Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || Cypress.env("NPA_AGENT_RUN_ID") || "";
}

function agentReq(path, options = {}) {
  const baseUrl = String(Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl") || "").replace(/\/$/, "");
  return cy.request({
    url: `${baseUrl}${path}`,
    auth: {
      username: Cypress.env("NPA_AGENT_USER"),
      password: Cypress.env("NPA_AGENT_PASSWORD"),
    },
    ...options,
  });
}

function artifactContentPath(activeRun, artifact) {
  const query = new URLSearchParams({
    run_id: activeRun,
    key: String(artifact.key || ""),
    bucket: String(artifact.bucket || ""),
  });
  return `/api/artifacts/content?${query.toString()}`;
}

describe("GR00T learning experience (live system)", () => {
  let activeRun = "";
  let artifacts = [];

  before(function () {
    if (!liveEnvAvailable() || !runId()) this.skip();
    activeRun = runId();
    return agentReq(`/api/artifacts/run/${encodeURIComponent(activeRun)}`).then((resp) => {
      expect(resp.status, "learning artifact inventory status").to.eq(200);
      expect(resp.body.summary.learning, "machine-readable learning summary").to.be.an("object");
      artifacts = resp.body.artifacts || [];
    });
  });

  beforeEach(() => {
    cy.visitLiveAgent();
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
    cy.get("#tabRerun").click();
    cy.get("#artifactPrefix").clear().type(activeRun, { delay: 0 });
    cy.get("#runIdSelect option", { timeout: 60000 }).should(($options) => {
      const values = [...$options].map((option) => option.value).filter(Boolean);
      expect(values).to.include(activeRun);
    });
    cy.get("#runIdInput").clear().type(activeRun, { delay: 0 });
    cy.get("#loadRunData").click();
    cy.get("#artifactRunSummary", { timeout: 120000 }).should("contain.text", activeRun);
  });

  it("shows a truthful replay-first learning summary and meaningful stages", () => {
    cy.get("#artifactRunSummary")
      .should("have.class", "learning-summary")
      .and("contain.text", "Policy learning summary")
      .and("contain.text", "Offline held-out policy evaluation")
      .and("contain.text", "Train / held-out")
      .and("contain.text", "Training coverage")
      .and("contain.text", "Baseline action_mse")
      .and("contain.text", "Post-training action_mse")
      .and("contain.text", "Before → after action error")
      .and("contain.text", "96x96 native")
      .and("contain.text", "episode-disjoint")
      .and("contain.text", "offline held-out (not rollout)");
    cy.get("#artifactRunSummary .learning-replay-actions")
      .should("contain.text", "Open in Rerun")
      .and("contain.text", "Open MCAP")
      .and("contain.text", "Play comparison video");
    cy.get("#stageList")
      .should("contain.text", "Prepare leakage-free split")
      .and("contain.text", "Baseline held-out inference")
      .and("contain.text", "Multi-GPU policy training")
      .and("contain.text", "Post-training held-out inference")
      .and("contain.text", "Compare learning")
      .and("contain.text", "Synchronized learning replay")
      .and("contain.text", "Validate and publish")
      .and("not.contain.text", "Visualize + finalize");
  });

  it("keeps raw artifacts collapsed and grouped below the replay", () => {
    cy.get("#artifactList").should("not.be.visible");
    cy.get("#rawArtifactsToggle").should("contain.text", "Show raw artifacts").click();
    cy.get("#artifactList")
      .should("be.visible")
      .and("contain.text", "grouped rows from")
      .and("contain.text", "reports/groot-learning.rrd")
      .and("contain.text", "reports/groot-learning.mcap");
    cy.get("#rawArtifactsToggle").should("contain.text", "Hide raw artifacts");
  });

  it("indexes the exact substantive RRD selected by the replay", () => {
    const rrd = artifacts.find((item) => String(item.key || "").endsWith("reports/groot-learning.rrd"));
    expect(rrd, "learning RRD artifact").to.exist;
    expect(Number(rrd.size || rrd.size_bytes || rrd.bytes || 0), "RRD inventory bytes").to.eq(600381);
    expect(String(rrd.key || ""), "run-scoped learning RRD key").to.include(activeRun);
  });

  it("loads the exact MCAP and publishes truthful Lichtblick configuration", () => {
    const mcap = artifacts.find((item) => String(item.key || "").endsWith("reports/groot-learning.mcap"));
    expect(mcap, "learning MCAP artifact").to.exist;
    expect(Number(mcap.size || mcap.size_bytes || mcap.bytes || 0), "MCAP inventory bytes").to.eq(298337);
    agentReq("/api/sim-viz/load-artifact", {
      method: "POST",
      body: { run_id: activeRun, key: mcap.key },
    }).then((resp) => {
      expect(resp.status).to.eq(200);
      expect(String(resp.body.sim_viz.artifact_key || "")).to.eq(mcap.key);
      expect(String(resp.body.sim_viz.lichtblick_iframe_url || "")).to.include("/lichtblick/");
      expect(String(resp.body.sim_viz.visualization_note || "")).to.include("Offline held-out");
      expect(String(resp.body.sim_viz.visualization_note || "")).to.include("not a rollout");
    });
  });

  it("streams the labelled comparison video with authenticated range semantics", () => {
    const video = artifacts.find((item) => String(item.key || "").endsWith("reports/offline-heldout-comparison.mp4"));
    expect(video, "offline held-out comparison video").to.exist;
    expect(Number(video.size || video.size_bytes || video.bytes || 0), "comparison video bytes").to.eq(355588);
    agentReq(artifactContentPath(activeRun, video), {
      headers: { Range: "bytes=0-4095" },
      encoding: "binary",
      failOnStatusCode: false,
    }).then((resp) => {
      expect(resp.status).to.eq(206);
      expect(resp.headers).to.have.property("content-range");
      expect(String(resp.body || "").length).to.be.greaterThan(100);
      expect(String(resp.body || "")).to.include("ftyp");
    });
  });
});
