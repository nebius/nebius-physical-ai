const requiredLiveEnv = [
  "NPA_AGENT_BASE_URL",
  "NPA_AGENT_USER",
  "NPA_AGENT_PASSWORD",
  "NPA_AGENT_CYPRESS_PAGINATION_RUN_ID",
];

describe("NPA agent live artifact pagination", () => {
  before(function () {
    if (!requiredLiveEnv.every((name) => Boolean(Cypress.env(name)))) this.skip();
  });

  it("renders one native page, completes explicitly, and filters from cache", () => {
    const runId = String(Cypress.env("NPA_AGENT_CYPRESS_PAGINATION_RUN_ID"));
    let inventoryRequests = 0;
    let continuationRequests = 0;

    cy.intercept("GET", "/api/artifacts/run/**", (request) => {
      inventoryRequests += 1;
      if (new URL(request.url).searchParams.has("cursor")) continuationRequests += 1;
      request.continue();
    }).as("artifactInventoryPage");

    cy.visitLiveAgent();
    cy.get("#statusBar", { timeout: 30000 }).should("exist");
    cy.get("#tabRerun").click();
    cy.get("#runIdInput").clear().type(runId, { delay: 0 });
    cy.get("#loadRunData").click();

    cy.wait("@artifactInventoryPage", { timeout: 120000 }).then(({ response }) => {
      expect(response.statusCode).to.eq(200);
      expect(response.body.artifacts).to.have.length(1000);
      expect(response.body.next_cursor).to.be.a("string").and.not.be.empty;
    });
    cy.get("#artifactRunSummary .no-recording").should("not.exist");
    cy.get("#artifactList").should("contain.text", "1000 artifacts");

    cy.get("#artifactLoadRunArtifacts").click();
    cy.get("#statusBar", { timeout: 120000 }).should("contain.text", "List run artifacts done");
    cy.get("#artifactList", { timeout: 120000 })
      .should("contain.text", "1001 artifacts")
      .and("contain.text", "2 inventory pages merged");
    cy.get("#artifactRunSummary .no-recording").should("exist");
    cy.then(() => expect(continuationRequests, "one cursor-qualified continuation").to.eq(1));

    cy.get("#artifactTypeFilter").select("json");
    cy.get("#artifactSort").select("largest");
    cy.then(() => expect(inventoryRequests, "filter and sort reuse the completed inventory").to.eq(2));
  });
});
