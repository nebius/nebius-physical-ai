// Deliberately use the same run basename across distinct synthetic sources.
const sourceA = {
  run_id: "cache-source-run", run_ref: "npa1_cache_source_a", project_id: "project-a",
  bucket: "cache-artifacts-a", resolved_prefix: "workflow-runs",
  source_type: "artifact_storage", summary_complete: true, has_viewable: false,
};
const sourceB = { ...sourceA, run_ref: "npa1_cache_source_b", project_id: "project-b", bucket: "cache-artifacts-b" };
const artifact = (source, name) => ({
  key: `${source.resolved_prefix}/${source.run_id}/${name}.bin`,
  render: "download", role: "output", size: 256,
});
const inventory = (source, name, cursor = "") => ({
  ok: true, ...source, artifacts: [artifact(source, name)], count: 1,
  truncated: Boolean(cursor), next_cursor: cursor, pagination_complete: !cursor,
});

function visitInventory() {
  cy.installAgentApiMocks();
  cy.intercept("GET", "/api/artifacts/runs*", { body: {
    runs: [sourceA], total_runs: 1, next_cursor: "", pagination_complete: true,
  } }).as("cacheDiscovery");
  cy.visit("/");
  cy.wait("@cacheDiscovery");
  cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
  cy.get("#tabRerun").click();
}

describe("Artifact inventory cache provenance", () => {
  it("retains a resolved source after completing a lightweight discovery row through Load next page", () => {
    visitInventory();
    cy.intercept("GET", "/api/artifacts/runs*", { body: {
      runs: [{ ...sourceA, run_ref: "" }], total_runs: 1, next_cursor: "", pagination_complete: true,
    } }).as("lightweightDiscovery");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@lightweightDiscovery");
    let inventoryRequests = 0;
    cy.intercept("GET", /\/api\/artifacts\/run\/(?:npa1_cache_source_a|cache-source-run)(?:\?|$)/, (req) => {
      inventoryRequests += 1;
      const cursor = new URL(req.url).searchParams.get("cursor");
      req.reply({ body: inventory(sourceA, cursor ? "page-two" : "page-one", cursor ? "" : "next-page") });
    }).as("resolvedInventory");
    cy.get("#runIdSelect").select(sourceA.run_id);
    cy.wait("@resolvedInventory");
    cy.get("#artifactList").should("contain.text", "page-one.bin");
    cy.then(() => expect(inventoryRequests, "selection only fetches the first page").to.eq(1));
    cy.get('#artifactList button[data-action="load-more-artifacts"]').click();
    cy.wait("@resolvedInventory").its("request.url").should("contain", sourceA.run_ref);
    cy.get("#artifactList").should("contain.text", "page-one.bin").and("contain.text", "page-two.bin");
    cy.get("#artifactLoadRunArtifacts").click();
    cy.get("#artifactLoadRunArtifacts").should("have.attr", "aria-busy", "false");
    cy.then(() => expect(inventoryRequests, "List reuses the completed exact source").to.eq(2));
  });

  for (const reference of ["", sourceA.run_ref]) {
    it(`reuses an exact inventory but fetches a changed resource with ${reference ? "a retained" : "no"} reference`, () => {
      let inventoryRequests = 0;
      visitInventory();
      cy.intercept("GET", /\/api\/artifacts\/run\/(?:npa1_cache_source_[ab]|cache-source-run)(?:\?|$)/, (req) => {
        inventoryRequests += 1;
        const requested = new URL(req.url);
        const secondSource = requested.searchParams.get("resource_bucket") === sourceB.bucket;
        req.reply({ body: inventory(secondSource ? sourceB : sourceA, secondSource ? "source-b-only" : "source-a-only") });
      }).as("sourceInventory");
      cy.get("#runIdSelect").select(sourceA.run_ref);
      cy.wait("@sourceInventory");
      cy.get("#artifactList").should("contain.text", "source-a-only.bin");
      cy.get("#artifactLoadRunArtifacts").click();
      cy.then(() => expect(inventoryRequests, "completed exact inventory reused").to.eq(1));

      // A refreshed row can lack a reference or retain a stale reference while
      // reporting another resource. Neither may inherit the loaded cache.
      cy.intercept("GET", "/api/artifacts/runs*", { body: {
        runs: [{ ...sourceB, run_ref: reference }], total_runs: 1, next_cursor: "", pagination_complete: true,
      } }).as("changedSourceDiscovery");
      cy.get("#artifactRefreshRuns").click();
      cy.wait("@changedSourceDiscovery");
      cy.get("#runIdInput").clear().type(sourceA.run_id);
      cy.get("#artifactLoadRunArtifacts").click();
      cy.wait("@sourceInventory").its("request.url").should("contain", `resource_bucket=${sourceB.bucket}`);
      cy.get("#artifactList").should("contain.text", "source-b-only.bin").and("not.contain.text", "source-a-only.bin");
      cy.then(() => expect(inventoryRequests).to.eq(2));
    });
  }

  for (const action of ["#artifactLoadRunArtifacts", '#artifactList button[data-action="load-more-artifacts"]']) {
    it(`rejects a changed source on the first continuation through ${action.startsWith("#artifactList ") ? "Load next page" : "List artifacts"}`, () => {
      visitInventory();
      let inventoryRequests = 0;
      cy.intercept("GET", /\/api\/artifacts\/run\/(?:npa1_cache_source_[ab]|cache-source-run)(?:\?|$)/, (req) => {
        inventoryRequests += 1;
        const requested = new URL(req.url);
        if (!requested.searchParams.get("cursor")) {
          req.reply({ body: inventory(sourceA, "source-a-page-one", "next-source-page") });
        } else {
          expect(requested.searchParams.get("resource_bucket")).to.eq(sourceA.bucket);
          expect(requested.searchParams.get("project_id")).to.eq(sourceA.project_id);
          expect(requested.searchParams.get("resolved_prefix")).to.eq(sourceA.resolved_prefix);
          req.reply({ body: inventory(sourceB, "must-not-merge") });
        }
      }).as("sourceInventory");
      cy.get("#runIdSelect").select(sourceA.run_ref);
      cy.wait("@sourceInventory");
      cy.get("#artifactList").should("contain.text", "source-a-page-one.bin");
      cy.then(() => expect(inventoryRequests, "first page is lazy").to.eq(1));
      cy.get(action).click();
      cy.wait("@sourceInventory");
      cy.get("#toastHost").should("contain.text", "Artifact inventory source changed during pagination");
      cy.get("#artifactList").should("contain.text", "source-a-page-one.bin").and("not.contain.text", "must-not-merge.bin");
      cy.get("#artifactList [data-bucket]").each(($button) => expect($button.attr("data-bucket")).to.eq(sourceA.bucket));
      cy.then(() => expect(inventoryRequests).to.eq(2));
    });
  }
});
