import { NON_STOCK_RUN_ID, SIM_VIZ } from "../support/e2e";

describe("Operator artifact preview during status refresh", () => {
  for (const [key, alias, expected] of [
    ["reports/sim2real-report.json", "artifactContentJson", "promoted"],
    ["logs/orchestrator.log", "artifactContentText", "loaded customer scene mesh"],
  ]) {
    it(`keeps ${key} in Data when a periodic refresh reports the active recording`, () => {
      cy.visitMockAgent();
      cy.wait("@session");
      cy.wait("@artifactRuns");
      cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
      // Install the interval clock before the first user interaction arms the
      // real periodic refresh. Network promises and viewer timers stay native.
      cy.clock(Date.now(), ["setInterval", "clearInterval"]);
      cy.get("#tabRerun").click();
      cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
      cy.wait("@nonStockArtifactList");
      cy.wait("@loadArtifact");
      cy.get(`#artifactList button[data-action="preview-artifact"][data-key="${NON_STOCK_RUN_ID}/${key}"]`).click();
      cy.wait(`@${alias}`);
      cy.get("#renderModeData").should("have.class", "is-active");
      cy.get("#artifactPreviewHost pre").should("contain.text", expected);

      cy.intercept("GET", "/api/sim-viz/status*", { body: {
        ...SIM_VIZ, run_id: NON_STOCK_RUN_ID, active_run_id: NON_STOCK_RUN_ID,
        artifact_render: "rerun", stage: "preview-refresh-complete",
      } }).as("previewRefreshStatus");
      cy.get("#tabMain").click();
      cy.tick(10000);
      cy.wait("@previewRefreshStatus");
      cy.get("#simStage").should("have.text", "preview-refresh-complete");
      cy.get("#renderModeData").should("have.class", "is-active");
      cy.get("#tabRerun").click();
      cy.get("#viewerPaneMedia").should("be.visible");
      cy.get("#artifactPreviewHost pre").should("contain.text", expected);
      cy.get("#renderModeData").should("have.attr", "aria-selected", "true");
    });
  }
});
