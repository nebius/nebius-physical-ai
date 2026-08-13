const RUN_ID = "groot17-8gpu-20260806T024557Z-3dfb0270";
const VIDEO_KEY = `groot-1-7-finetune/${RUN_ID}/data/videos/chunk-000/observation.image/episode_000185.mp4`;
const CHECKPOINT_KEY = `${RUN_ID}/checkpoints/checkpoint-1/global_step1/mp_rank_00_model_states.pt`;

function liveEnvAvailable() {
  return ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"].every((name) =>
    Boolean(Cypress.env(name) || Cypress.env(name.replace("NPA_AGENT_", "agent"))),
  );
}

describe("recording-independent artifacts against the live GR00T run", () => {
  before(function () {
    if (!liveEnvAvailable()) this.skip();
  });

  it("selects the run and previews JSON, YAML, log, image, video, and checkpoint metadata", () => {
    cy.visitLiveAgent();
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
    cy.get("#tabRerun").click();
    cy.get("#artifactPrefix").clear().type(RUN_ID, { delay: 0 });
    cy.get("#runIdSelect option", { timeout: 60000 }).should(($opts) => {
      const values = [...$opts].map((option) => option.value).filter(Boolean);
      expect(values).to.include(RUN_ID);
    });

    cy.intercept("POST", "/api/sim-viz/load-run").as("loadTrainingRun");
    cy.intercept("GET", `/api/artifacts/run/${RUN_ID}*`).as("liveArtifactList");
    cy.intercept("GET", "/api/artifacts/content*", (req) => {
      const key = new URL(req.url).searchParams.get("key") || "";
      if (key === `${RUN_ID}/manifest.json`) req.alias = "liveManifest";
      else if (key === `${RUN_ID}/workflow.yaml`) req.alias = "liveWorkflow";
      else if (key === `${RUN_ID}/evidence/training.log`) req.alias = "liveTrainingLog";
      else if (key === `${RUN_ID}/training-summary.png`) req.alias = "liveSummaryImage";
      req.continue();
    });
    cy.get("#runIdInput").clear().type(RUN_ID, { delay: 0 });
    cy.get("#loadRunData").click();
    cy.wait("@loadTrainingRun", { timeout: 120000 }).its("response.statusCode").should("eq", 200);
    cy.wait("@liveArtifactList", { timeout: 120000 }).its("response.statusCode").should("eq", 200);

    cy.get("#artifactRunSummary", { timeout: 60000 })
      .should("contain.text", RUN_ID)
      .and("contain.text", "Completion: completed")
      .and("contain.text", "8 × RTX PRO 6000 Blackwell Server Edition")
      .and("contain.text", "World size: 8")
      .and("contain.text", "Training steps: 1")
      .and("contain.text", "Finite loss: 1.03125")
      .and("contain.text", "1081 total · 60 output · 1020 input/staged")
      .and("contain.text", "No RRD/MCAP recording; use the artifacts below");
    cy.get("#rerunPlaceholder")
      .should("have.attr", "data-state", "no-preview-artifacts")
      .and("contain.text", "No RRD/MCAP recording; use the artifacts below");

    cy.get(`#artifactList button[data-action='preview-artifact'][data-key='${RUN_ID}/manifest.json']`).click();
    cy.wait("@liveManifest").its("response.statusCode").should("eq", 200);
    cy.get("#artifactPreviewHost pre").should("contain.text", RUN_ID);

    cy.get(`#artifactList button[data-action='preview-artifact'][data-key='${RUN_ID}/workflow.yaml']`).click();
    cy.wait("@liveWorkflow").its("response.statusCode").should("eq", 200);
    cy.get("#artifactPreviewHost pre")
      .should("contain.text", "apiVersion:")
      .and("contain.text", "groot-1-7-finetune");

    cy.get(`#artifactList button[data-action='preview-artifact'][data-key='${RUN_ID}/evidence/training.log']`).click();
    cy.wait("@liveTrainingLog").its("response.statusCode").should("eq", 200);
    cy.get("#artifactPreviewHost pre")
      .should("contain.text", "completed")
      .and("contain.text", "1.03125");

    cy.get(`#artifactList button[data-action='preview-artifact'][data-key='${RUN_ID}/training-summary.png']`).click();
    cy.wait("@liveSummaryImage").its("response.statusCode").should("eq", 200);
    cy.get("#artifactPreviewHost img", { timeout: 60000 })
      .should("have.attr", "src")
      .and("match", /^blob:/);

    cy.get("#artifactList .artifact-card[data-render='download']")
      .filter(`:contains("${CHECKPOINT_KEY.split("/").pop()}")`)
      .first()
      .should("contain.text", "checkpoint")
      .and("contain.text", "application/octet-stream")
      .within(() => {
        cy.get("button[data-action='preview-artifact']").should("not.exist");
        cy.get("button[data-action='download-artifact']").should("exist");
      });

    cy.get("#artifactRoleFilter").select("");
    cy.get(`#artifactList button[data-action='preview-artifact'][data-key='${VIDEO_KEY}']`, {
      timeout: 60000,
    }).click();
    cy.get("#artifactPreviewHost video")
      .should("have.prop", "controls", true)
      .and("have.attr", "src")
      .and("include", "/api/artifacts/content?")
      .and("include", encodeURIComponent("episode_000185.mp4"));
    cy.get("#artifactList").should("contain.text", "manifest.json");
  });
});
