import { ARTIFACT_ONLY_RUN_ID } from "../support/e2e";

describe("Superseded local demo viewer mounts", () => {
  beforeEach(() => {
    cy.installAgentApiMocks();
    // A published recording carries its own capability in real status
    // responses. The generic mock omits it and exercises the blob fallback,
    // which bypasses the two awaited HEAD requests this suite must hold.
    cy.intercept({ method: "GET", url: "/api/sim-viz/status*", middleware: true }, (req) => {
      req.on("before:response", (res) => {
        res.body.artifact_preview_url = `/rerun/recordings/cap-${"A".repeat(43)}.rrd`;
      });
    });
    // Keep the replacement run's exact source provenance while giving it only
    // data/download artifacts: selecting it must not start another Rerun mount.
    cy.intercept({ method: "GET", url: `/api/artifacts/run/${ARTIFACT_ONLY_RUN_ID}*`, middleware: true }, (req) => {
      req.on("before:response", (res) => {
        res.body.artifacts = res.body.artifacts.filter((item) => ["json", "download"].includes(item.render));
        res.body.count = res.body.artifacts.length;
        res.body.preferred = null;
      });
    });
    cy.visit("/");
    cy.wait("@session");
    cy.wait("@simAssets");
    cy.wait("@agentAccess");
    cy.wait("@artifactRuns");
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
  });

  for (const boundary of ["recording capability", "authenticated blob"]) {
    it(`keeps the newer run when the demo's ${boundary} check finishes late`, () => {
      let releaseRequest;
      const mountedRuns = [];
      const path = boundary === "recording capability"
        ? /\/rerun\/recordings\/cap-[A-Za-z0-9_-]{43}\.rrd(?:\?|$)/
        : "/api/sim-viz/rrd-blob";
      cy.intercept({ method: "HEAD", url: path, times: 1 }, (req) => {
        return new Cypress.Promise((resolve) => {
          releaseRequest = () => {
            req.reply({ statusCode: 200, headers: { "content-type": "application/octet-stream" } });
            resolve();
          };
        });
      }).as("delayedDemoMount");
      cy.window().then((win) => {
        const frame = win.document.getElementById("rerunFrame");
        const observer = new win.MutationObserver(() => {
          mountedRuns.push(frame.dataset.rerunRunKey || "");
        });
        observer.observe(frame, { attributes: true, attributeFilter: ["data-rerun-run-key"] });
      });

      // Use separate native selection events. A second click on the same busy
      // Load button would be rejected by the browser and would test no race.
      cy.get("#stagesRunSelect option").contains("franka-demo").invoke("val").then((value) => {
        cy.get("#stagesRunSelect").select(value);
      });
      cy.wait("@loadFranka");
      cy.wrap(null).should(() => expect(releaseRequest, "demo reached its async mount check").to.be.a("function"));
      cy.get("#stagesRunSelect option").contains(ARTIFACT_ONLY_RUN_ID).invoke("val").then((value) => {
        cy.get("#stagesRunSelect").select(value);
      });
      cy.wait("@artifactOnlyList");
      cy.get("#runSummary").should("contain.text", ARTIFACT_ONLY_RUN_ID);
      cy.get("#openFullRerun").invoke("attr", "href").then((selectedHref) => {
        releaseRequest();
        cy.wait("@delayedDemoMount");
        // Let the resumed mount's promise chain run before asserting absence
        // of stale mutations; this is deliberately after the delayed response.
        cy.wait(350);
        cy.get("#openFullRerun").should("have.attr", "href", selectedHref);
      });
      cy.get("#rerunFrame").should("have.attr", "hidden");
      cy.get("#rerunFrame").should("not.have.attr", "src");
      cy.get("#runSummary").should("contain.text", ARTIFACT_ONLY_RUN_ID);
      cy.then(() => expect(mountedRuns, "old demo never navigated the iframe").not.to.include("franka-demo"));
    });
  }
});
