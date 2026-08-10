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

describe("GR00T closed-loop task performance (live system)", () => {
  let activeRun = "";
  let artifacts = [];

  before(function () {
    if (!liveEnvAvailable() || !runId()) this.skip();
    activeRun = runId();
    return agentReq(`/api/artifacts/run/${encodeURIComponent(activeRun)}`).then((resp) => {
      expect(resp.status).to.eq(200);
      expect(resp.body.summary.task_performance, "task-performance summary").to.be.an("object");
      expect(resp.body.summary.task_performance.improvement_gate_passed).to.eq(true);
      artifacts = resp.body.artifacts || [];
    });
  });

  beforeEach(() => {
    cy.visitLiveAgent();
    cy.get("#tabRerun").click();
    cy.get("#artifactPrefix").clear().type(activeRun, { delay: 0 });
    cy.get("#runIdSelect option", { timeout: 60000 }).should(($options) => {
      expect([...$options].map((option) => option.value)).to.include(activeRun);
    });
    cy.get("#runIdInput").clear().type(activeRun, { delay: 0 });
    cy.get("#loadRunData").click();
    cy.get("#artifactRunSummary", { timeout: 120000 }).should("contain.text", activeRun);
  });

  it("puts honest paired task outcomes before diagnostics and raw artifacts", () => {
    cy.get("#artifactRunSummary")
      .should("have.class", "task-performance-summary")
      .and("contain.text", "Robot task performance — PushT")
      .and("contain.text", "Simulated")
      .and("contain.text", "gym_pusht/PushT-v0")
      .and("contain.text", "Paired episodes: 24")
      .and("contain.text", "Baseline success")
      .and("contain.text", "Trained success")
      .and("contain.text", "Absolute success delta")
      .and("contain.text", "Paired confidence")
      .and("contain.text", "Paired test")
      .and("contain.text", "absolute target pusher x")
      .and("contain.text", "workspace pixels")
      .and("contain.text", "coverage > 0.95")
      .and("contain.text", "PASS:");
    cy.get("#taskPerformanceSeedSelector option").should("have.length", 24);
    cy.get(".paired-rollout-videos video").should("have.length", 2);
    cy.get("#artifactRunSummary details")
      .should("contain.text", "Required offline rigor")
      .and("contain.text", "offline gate");
    cy.get("#artifactList").should("not.be.visible");
  });

  it("serves actual task RRD, MCAP, report, and labeled comparison video", () => {
    const suffixes = [
      "reports/task-performance.rrd",
      "reports/task-performance.mcap",
      "reports/task-performance-report.json",
      "reports/paired-task-performance.mp4",
    ];
    for (const suffix of suffixes) {
      const artifact = artifacts.find((item) => String(item.key || "").endsWith(suffix));
      expect(artifact, suffix).to.exist;
      expect(Number(artifact.size || artifact.size_bytes || artifact.bytes || 0), `${suffix} bytes`).to.be.greaterThan(100);
    }
    const mcap = artifacts.find((item) => String(item.key || "").endsWith("reports/task-performance.mcap"));
    agentReq("/api/sim-viz/load-artifact", {
      method: "POST",
      body: { run_id: activeRun, key: mcap.key },
    }).then((resp) => {
      expect(resp.status).to.eq(200);
      expect(String(resp.body.sim_viz.artifact_key || "")).to.eq(mcap.key);
      expect(String(resp.body.sim_viz.visualization_note || "")).to.include("closed-loop");
      expect(String(resp.body.sim_viz.visualization_note || "")).to.include("Simulated");
    });
    const video = artifacts.find((item) => String(item.key || "").endsWith("reports/paired-task-performance.mp4"));
    const query = new URLSearchParams({ run_id: activeRun, key: video.key, bucket: video.bucket || "" });
    agentReq(`/api/artifacts/content?${query.toString()}`, {
      headers: { Range: "bytes=0-4095" },
      encoding: "binary",
    }).then((resp) => {
      expect(resp.status).to.eq(206);
      expect(String(resp.body || "")).to.include("ftyp");
    });
  });

  it("selects paired outcomes and exposes failure termination evidence", () => {
    cy.get("#taskPerformanceSeedSelector").select(1);
    cy.get("#taskPerformanceSeedOutcome")
      .should("contain.text", "baseline")
      .and("contain.text", "trained");
    cy.get(".failure-gallery")
      .should("be.visible")
      .and("contain.text", "coverage=");
  });
});
