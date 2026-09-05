import { ARTIFACT_ONLY_RUN_ID } from "../support/e2e";

describe("NPA agent local demo selection", () => {
  beforeEach(() => {
    cy.visitMockAgent();
    cy.wait("@session");
    cy.wait("@simAssets");
    cy.wait("@agentAccess");
    cy.wait("@artifactRuns");
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
  });

  for (const selectionMode of ["stages dropdown", "Viewer Load button"]) {
  it(`does not mount a local demo superseded by the ${selectionMode} after delayed details`, () => {
    if (selectionMode === "Viewer Load button") {
      cy.get("#tabRerun").click();
      cy.get("#rerunBundleCover").should("have.attr", "hidden");
    }
    let demoDetailsStarted = false;
    let releaseDemoDetails;
    const mountedRuns = [];
    cy.intercept({ method: "GET", url: "/api/workflows/sim2real/runs/franka-demo*", times: 1 }, (req) => {
      demoDetailsStarted = true;
      return new Cypress.Promise((resolve) => {
        releaseDemoDetails = () => {
          req.reply({ body: { run: { run_id: "franka-demo", stages: [], logs: [] } } });
          resolve();
        };
      });
    });
    cy.window().then((win) => {
      const frame = win.document.getElementById("rerunFrame");
      const observer = new win.MutationObserver(() => {
        mountedRuns.push(frame.dataset.rerunRunKey || "");
      });
      observer.observe(frame, { attributes: true, attributeFilter: ["data-rerun-run-key"] });
    });
    cy.get(selectionMode === "stages dropdown" ? "#stagesRunSelect" : "#runIdSelect").select("franka-demo");
    cy.wait("@loadFranka");
    cy.wrap(null).should(() => expect(demoDetailsStarted).to.eq(true));
    if (selectionMode === "stages dropdown") {
      cy.get("#stagesRunSelect").select(ARTIFACT_ONLY_RUN_ID);
    } else {
      cy.get("#runIdInput").clear().type(ARTIFACT_ONLY_RUN_ID);
      cy.get("#loadRunData").click();
    }
    cy.wait("@artifactOnlyList");
    cy.then(() => releaseDemoDetails());
    cy.get("#runSummary").should("contain.text", ARTIFACT_ONLY_RUN_ID);
    // Allow the delayed response and any erroneous mount it starts to settle.
    cy.wait(1800);
    cy.then(() => expect(mountedRuns, "viewer never reactivates the old demo").not.to.include("franka-demo"));
    cy.get("#runSummary").should("contain.text", ARTIFACT_ONLY_RUN_ID);
  });
  }

  it("does not repaint a superseded Viewer Load response over a newer dropdown selection", () => {
    cy.get("#tabRerun").click();
    let releaseDemoDetails;
    cy.intercept({ method: "GET", url: "/api/workflows/sim2real/runs/franka-demo*", times: 1 }, (req) => {
      return new Cypress.Promise((resolve) => {
        releaseDemoDetails = () => {
          req.reply({ body: { run: { run_id: "franka-demo", stages: [], logs: [] } } });
          resolve();
        };
      });
    });
    cy.get("#runIdInput").clear().type("franka-demo");
    cy.get("#loadRunData").click();
    cy.wait("@loadFranka");
    cy.wrap(null).should(() => expect(releaseDemoDetails).to.be.a("function"));
    cy.get("#loadRunData").should("have.attr", "aria-busy", "true");
    const renderedSummaries = [];
    cy.window().then((win) => {
      const summary = win.document.getElementById("renderedDataSummary");
      const observer = new win.MutationObserver(() => renderedSummaries.push(summary.textContent));
      observer.observe(summary, { childList: true, characterData: true, subtree: true });
    });
    cy.get("#runIdSelect").select(ARTIFACT_ONLY_RUN_ID);
    cy.wait("@artifactOnlyList");
    cy.get("#renderedDataSummary").should("contain.text", ARTIFACT_ONLY_RUN_ID);
    cy.then(() => releaseDemoDetails());
    // The original click's busy state ends only after its wrapper has resumed.
    cy.get("#loadRunData").should("have.attr", "aria-busy", "false");
    cy.get("#runSummary").should("contain.text", ARTIFACT_ONLY_RUN_ID);
    cy.get("#renderedDataSummary").should("contain.text", ARTIFACT_ONLY_RUN_ID)
      .and("not.contain.text", "franka-demo");
    cy.then(() => expect(renderedSummaries.join("\n"), "no transient stale summary during the new load")
      .not.to.contain("franka-demo"));
  });
});
