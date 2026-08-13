const requiredLiveEnv = ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"];

function liveEnvAvailable() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name)));
}

function runId() {
  return String(Cypress.env("NPA_AGENT_CYPRESS_RUN_ID") || "").trim();
}

function agentReq(path, options = {}) {
  const baseUrl = String(Cypress.env("NPA_AGENT_BASE_URL") || Cypress.config("baseUrl") || "")
    .replace(/\/$/, "");
  return cy.request({
    url: `${baseUrl}${path}`,
    auth: {
      username: Cypress.env("NPA_AGENT_USER"),
      password: Cypress.env("NPA_AGENT_PASSWORD"),
    },
    timeout: 180000,
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

function assertViewerDocument(frame, label) {
  const iframe = frame[0];
  const doc = iframe.contentDocument || (iframe.contentWindow && iframe.contentWindow.document);
  expect(doc, `${label} same-origin document`).to.exist;
  expect(doc.readyState, `${label} document loaded`).to.eq("complete");
  const text = String((doc.body && doc.body.innerText) || "");
  expect(text, `${label} has no load failure`).not.to.match(/failed to load|unable to load|application error/i);
  const canvases = [...doc.querySelectorAll("canvas")];
  expect(canvases.length, `${label} rendered canvas count`).to.be.greaterThan(0);
  expect(
    canvases.some((canvas) => canvas.width > 100 && canvas.height > 100),
    `${label} has a substantive render surface`,
  ).to.eq(true);
}

describe("GR00T operational two-GPU pipeline (live system)", { testIsolation: false }, () => {
  let activeRun = "";
  let inventory = {};
  let artifacts = [];
  let primaryCamera = "";

  before(function () {
    if (!liveEnvAvailable() || !runId()) this.skip();
    activeRun = runId();
    return agentReq(`/api/artifacts/run/${encodeURIComponent(activeRun)}`).then((resp) => {
      expect(resp.status, "artifact inventory status").to.eq(200);
      expect(resp.body.run_id, "exact run identity").to.eq(activeRun);
      inventory = resp.body;
      artifacts = resp.body.artifacts || [];
      primaryCamera = String(
        (((resp.body.summary || {}).learning || {}).artifact_contract || {}).primary_camera || "",
      ).trim();
      expect(primaryCamera, "validated report primary camera").to.match(/^[A-Za-z0-9_.-]+$/);
      expect(artifacts.length, "published artifact count").to.be.greaterThan(0);
      cy.visitLiveAgent();
      cy.get("#statusBar", { timeout: 60000 }).should("exist");
      cy.get("#tabRerun").click();
      cy.get("#runIdInput").clear().type(activeRun, { delay: 0 });
      cy.get("#loadRunData").click();
      cy.get("#statusBar", { timeout: 300000 }).should("contain.text", "Load run data done");
      cy.get("#artifactRunSummary", { timeout: 180000 }).should("contain.text", activeRun);
      cy.get("#stageList .stage-item", { timeout: 180000 }).should("have.length", 7);
    });
  });

  it("separates operational success from the learning outcome", () => {
    const learning = inventory.summary && inventory.summary.learning;
    expect(learning, "machine-readable learning summary").to.be.an("object");
    expect(learning.pipeline_status).to.eq("succeeded");
    expect(learning.learning_outcome).to.be.oneOf(["improved", "not_improved", "inconclusive"]);
    if (learning.learning_outcome !== "improved") {
      expect(learning.candidate_promoted, "unsupported candidate is not promoted").to.eq(false);
    }
    expect(learning.gpu_count, "live training GPU count").to.eq(2);
    expect(learning.optimizer_steps, "real optimizer steps").to.eq(4);
    expect(learning.closed_loop, "offline evidence is not a robot rollout").to.eq(false);

    cy.get("#artifactRunSummary")
      .should("have.class", "learning-summary")
      .and("contain.text", "Pipeline status")
      .and("contain.text", "SUCCEEDED")
      .and("contain.text", "Learning outcome")
      .and("contain.text", "Candidate promoted")
      .and("contain.text", "offline held-out (not rollout)");
  });

  it("proves all physical jobs while showing the semantic offline provenance stages", () => {
    agentReq(`/api/workflows/sim2real/runs/${encodeURIComponent(activeRun)}`).then((resp) => {
      expect(resp.status, "run details status").to.eq(200);
      const run = resp.body.run || resp.body;
      const stages = run.stages || [];
      expect(run.status, "authoritative workflow status").to.eq("succeeded");
      expect(stages, "workflow stages").to.have.length(12);
      expect(stages.every((stage) => stage.status === "succeeded"), "all stages succeeded").to.eq(true);
      expect(stages.every((stage) => String(stage.job_id || "").length > 0), "all physical job IDs").to.eq(true);
      expect(new Set(stages.map((stage) => String(stage.job_id))).size, "one physical job per serial stage")
        .to.eq(12);
    });
    cy.get("#stageList .stage-item", { timeout: 180000 }).should("have.length", 7);
    cy.get("#stageList .stage-status").each(($status) => {
      expect($status.text()).to.eq("Succeeded");
    });
    cy.get("#stageList")
      .should("contain.text", "Prepare leakage-free split")
      .and("contain.text", "Offline baseline")
      .and("contain.text", "Multi-GPU policy training")
      .and("contain.text", "Offline post-training evaluation")
      .and("contain.text", "Classify learning outcome")
      .and("contain.text", "Synchronized diagnostics")
      .and("contain.text", "Validate and publish");
  });

  it("lists and range-serves every required diagnostic artifact", () => {
    const required = [
      "reports/two-gpu-pipeline-report.json",
      "reports/groot-offline-evaluation.rrd",
      "reports/groot-offline-evaluation.mcap",
      "reports/publish-manifest.json",
      "reports/agent-ui-verification.json",
      "reports/trained-checkpoint.json",
      "checkpoints/candidate/npa_groot_finetune_manifest.json",
      "npa-workflow/manifest.json",
    ];
    for (const suffix of required) {
      const artifact = artifacts.find((item) => String(item.key || "").endsWith(suffix));
      expect(artifact, suffix).to.exist;
      expect(Number(artifact.size || 0), `${suffix} bytes`).to.be.greaterThan(0);
    }
    for (const suffix of [".rrd", ".mcap", "two-gpu-pipeline-report.json"]) {
      const artifact = artifacts.find((item) => String(item.key || "").endsWith(suffix));
      agentReq(artifactContentPath(activeRun, artifact), {
        headers: { Range: "bytes=0-255" },
        encoding: "binary",
        failOnStatusCode: false,
      }).then((resp) => {
        expect(resp.status, `${suffix} byte range`).to.eq(206);
        expect(String(resp.body || "").length, `${suffix} ranged bytes`).to.be.greaterThan(0);
      });
    }
  });

  it("opens the exact MCAP with expected topics in Lichtblick", () => {
    const mcap = artifacts.find((item) => String(item.key || "").endsWith("groot-offline-evaluation.mcap"));
    agentReq("/api/sim-viz/load-artifact", {
      method: "POST",
      body: { run_id: activeRun, run_ref: inventory.run_ref || "", key: mcap.key },
    }).then((resp) => {
      expect(resp.status).to.eq(200);
      expect(resp.body.render).to.eq("mcap");
      expect(resp.body.sim_viz.lichtblick_ready).to.eq(true);
      expect(String(resp.body.sim_viz.artifact_key || "")).to.eq(mcap.key);
    });
    agentReq(artifactContentPath(activeRun, mcap), { encoding: "binary" }).then((resp) => {
      const body = String(resp.body || "");
      expect(resp.status).to.eq(200);
      for (const topic of [
        `/camera/${primaryCamera}`,
        "/policy/predicted_action",
        "/expert/action",
        "/metrics/action_error",
        "/metrics/train_loss",
        "/log",
      ]) {
        expect(body, `MCAP topic ${topic}`).to.include(topic);
      }
    });
    cy.reload();
    cy.get("#tabRerun").click();
    cy.get("#renderModeLichtblick").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#lichtblickFrame", { timeout: 180000 })
      .should("be.visible")
      .and("have.attr", "src")
      .and("include", "/lichtblick/")
      .and("include", "ds.url")
      .and("include", "npa.layout=learning");
    cy.get("#lichtblickFrame", { timeout: 180000 }).should(($frame) => {
      assertViewerDocument($frame, "Lichtblick");
      const doc = $frame[0].contentDocument;
      const text = String((doc && doc.body && doc.body.innerText) || "");
      expect(text, "Lichtblick initializes the remote MCAP").not.to.include("Error initializing");
      expect(text, "the configured camera topic exists").not.to.include("Image topic does not exist");
      expect(text, "the camera topic is visible in the viewer").to.include(`/camera/${primaryCamera}`);
    });
    cy.get("#simvizCta")
      .should("contain.text", "OFFLINE EVALUATION")
      .and("contain.text", "NOT A ROBOT ROLLOUT");
    cy.screenshot("after-two-gpu-pipeline-ui", { capture: "viewport" });
  });

  it("paints and describes the exact RRD as offline evaluation rather than a robot rollout", () => {
    const rrd = artifacts.find((item) => String(item.key || "").endsWith("groot-offline-evaluation.rrd"));
    cy.get("#runIdInput").clear().type(activeRun, { delay: 0 });
    cy.get("#loadRunData").click();
    cy.get("#statusBar", { timeout: 300000 }).should("contain.text", "Load run data done");
    cy.window().then({ timeout: 180000 }, (win) => {
      return win.__NPA_AGENT_TEST__.loadArtifact({
        run_id: activeRun,
        run_ref: inventory.run_ref || "",
        key: rrd.key,
      });
    }).then((loaded) => {
      expect(loaded, "exact RRD becomes the active viewer artifact").to.eq(true);
    });
    cy.get("#tabRerun").click();
    cy.get("#rerunFrame", { timeout: 180000 }).should("be.visible");
    const waitForDescribePaint = (attempt) => {
      cy.window().then((win) => {
        const frame = win.document.getElementById("rerunFrame");
        return win.__NPA_AGENT_TEST__.probeRerunCanvasContent(frame);
      }).then((painted) => {
        if (!painted && attempt < 90) {
          cy.wait(1000).then(() => waitForDescribePaint(attempt + 1));
          return;
        }
        expect(painted, "Describe-this receives a painted Rerun frame").to.eq(true);
      });
    };
    waitForDescribePaint(0);
    cy.get("#rerunFrame", { timeout: 180000 }).should(($frame) => {
      assertViewerDocument($frame, "Rerun");
      const doc = $frame[0].contentDocument;
      const text = String((doc && doc.body && doc.body.innerText) || "");
      expect(text, "no Rerun view-instantiation errors")
        .not.to.match(/failed to instantiate|failed to load view|unknown component/i);
      expect(text, "optimizer-clock panels are not misleadingly empty on dataset_time")
        .not.to.include("Checkpoint validation curve")
        .and.not.to.include("Training loss");
    });
    cy.window().then({ timeout: 30000 }, (win) => {
      return win.__NPA_AGENT_TEST__.captureVisualContext();
    }).then((captured) => {
      expect(captured.meta.run_id, "Describe-this run identity").to.eq(activeRun);
      expect(String(captured.meta.artifact_key || ""), "Describe-this artifact identity")
        .to.include("groot-offline-evaluation.rrd");
    });
    cy.get("#describeVisual", { timeout: 180000 }).click({ force: true });
    cy.get("#chatLog .msg-row.assistant", { timeout: 180000 }).should(($rows) => {
      const description = String($rows.last().text() || "").toLowerCase();
      expect(description.length, "nonblank Describe-this response").to.be.greaterThan(40);
      expect(description).to.match(/offline|held-out|evaluation/);
      expect(description).to.match(/not (a )?(physical )?robot rollout|not a rollout|offline.*not.*rollout/);
    });
  });
});
