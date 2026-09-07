// Run against the existing authenticated deployment. Keep all runner output
// outside Git: even assertion failures can contain selected resource identities.
const requiredLiveEnv = ["NPA_AGENT_BASE_URL", "NPA_AGENT_USER", "NPA_AGENT_PASSWORD"];

function availableBuckets(project) {
  return (project.resources || []).filter((resource) =>
    resource.name && resource.capabilities?.artifact_discovery?.status === "available"
  );
}

function expectSelectedSource(project, bucket) {
  cy.get("#agentAccessProjectSelect").should("have.value", project.id);
  cy.get(".access-project-detail").should("have.attr", "data-project-id", project.id);
  cy.get("#agentAccessBucketSelect").should("have.value", bucket.name);
  cy.get(".access-resource").should("have.attr", "data-selected-bucket", bucket.name);
  cy.get('[data-access-action="list"]')
    .should("have.attr", "data-project-id", project.id)
    .and("have.attr", "data-resource-bucket", bucket.name)
    .and(bucket.capabilities?.artifact_discovery?.status === "available" ? "not.be.disabled" : "be.disabled");
}

describe("NPA agent live access selection and operator UI configuration", () => {
  before(function () {
    if (!requiredLiveEnv.every((name) => Boolean(Cypress.env(name)))) this.skip();
  });

  it("keeps real multiple-option selections, details, and subsequent discovery synchronized", () => {
    let project;
    let buckets;
    cy.intercept("GET", "/api/access*").as("accessReport");
    cy.visitLiveAgent();
    cy.wait("@accessReport", { timeout: 120000 }).then(({ response }) => {
      expect(response.statusCode).to.eq(200);
      expect(response.body.projects.length, "real project choices").to.be.greaterThan(1);
      project = response.body.projects.find((item) => availableBuckets(item).length > 1);
      expect(Boolean(project), "a real project with multiple readable buckets").to.eq(true);
      buckets = availableBuckets(project);
      cy.get("#agentAccessProjectSelect").select(project.id, { log: false });
      cy.get("#agentAccessBucketSelect").select(buckets[0].name, { log: false });
      expectSelectedSource(project, buckets[0]);
    });
    cy.intercept("GET", "/api/artifacts/runs*", (request) => {
      const query = new URL(request.url).searchParams;
      if (query.get("project_id") === project?.id) {
        if (query.get("resource_bucket") === buckets?.[0]?.name) request.alias = "firstBucketDiscovery";
        if (query.get("resource_bucket") === buckets?.[1]?.name) request.alias = "secondBucketDiscovery";
      }
      request.continue();
    });
    cy.get('[data-access-action="list"]').click();
    cy.wait("@firstBucketDiscovery", { timeout: 120000 }).then(({ request, response }) => {
      expect(response.statusCode).to.eq(200);
      expect(new URL(request.url).searchParams.get("resource_bucket") === buckets[0].name).to.eq(true);
      cy.get("#agentAccessBucketSelect").select(buckets[1].name, { log: false });
      expectSelectedSource(project, buckets[1]);
    });
    // This independent action used to retain the previous bucket even though
    // the access card and native picker already showed the new selection.
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@secondBucketDiscovery", { timeout: 120000 }).then(({ request, response }) => {
      expect(response.statusCode).to.eq(200);
      const query = new URL(request.url).searchParams;
      expect(query.get("project_id") === project.id, "selected project reaches discovery").to.eq(true);
      expect(query.get("resource_bucket") === buckets[1].name, "new bucket reaches discovery").to.eq(true);
    });
  });

  it("applies a real single-bucket project without requiring a bucket change event", () => {
    let project;
    let bucket;
    cy.intercept("GET", "/api/access*").as("accessReport");
    cy.visitLiveAgent();
    cy.wait("@accessReport", { timeout: 120000 }).then(({ response }) => {
      expect(response.statusCode).to.eq(200);
      project = response.body.projects.find((item) =>
        (item.resources || []).length === 1 && Boolean(item.resources[0].name)
      );
      expect(Boolean(project), "a real project with one discovered bucket").to.eq(true);
      [bucket] = project.resources;
      cy.get("#agentAccessProjectSelect").select(project.id, { log: false });
      cy.get("#agentAccessBucketSelect option").should("have.length", 1);
      expectSelectedSource(project, bucket);
    });
    cy.intercept("GET", "/api/artifacts/runs*", (request) => {
      if (new URL(request.url).searchParams.get("project_id") === project?.id) {
        request.alias = "singleBucketDiscovery";
      }
      request.continue();
    });
    cy.then(() => {
      if (bucket.capabilities?.artifact_discovery?.status === "available") {
        cy.get('[data-access-action="list"]').click();
        cy.wait("@singleBucketDiscovery", { timeout: 120000 }).then(({ request, response }) => {
          expect(response.statusCode).to.eq(200);
          expect(new URL(request.url).searchParams.get("resource_bucket") === bucket.name).to.eq(true);
        });
      } else {
        // Live IAM must remain authoritative. Selecting a sole visible bucket
        // still applies its identity and explains why listing is unavailable.
        cy.get('[data-access-action="list"]').should("be.disabled");
        cy.get(".access-capability-reason").first().invoke("text").should("not.be.empty");
      }
    });
  });

  it("honors the deployed config flag and ignores a retained browser opt-in", () => {
    const enabled = Cypress.env("NPA_AGENT_EXPECT_LEISAAC") === true ||
      Cypress.env("NPA_AGENT_EXPECT_LEISAAC") === "true";
    let capabilityRequests = 0;
    cy.intercept("GET", "/api/leisaac/**", (request) => {
      capabilityRequests += 1;
      request.continue();
    }).as("leisaacStatus");
    cy.visitLiveAgent();
    cy.window().then((win) => win.localStorage.setItem("npa.agent.leisaac-ui-enabled.v1", "1"));
    cy.reload();
    cy.get("#statusBar").should("exist");
    cy.get("#tabMain").click();
    cy.get("#enableLeIsaac, #disableLeIsaac").should("not.exist");
    if (enabled) {
      cy.get("#tabLeIsaac").should("be.visible").click();
      cy.wait("@leisaacStatus", { timeout: 120000 });
      cy.get("#panelLeIsaac").should("be.visible");
      cy.then(() => expect(capabilityRequests, "enabled capability checks").to.be.greaterThan(0));
    } else {
      cy.get("#tabLeIsaac, #panelLeIsaac").should("not.exist");
      // Observe beyond two normal polling intervals, including a full reload.
      cy.wait(21000, { log: false });
      cy.then(() => expect(capabilityRequests, "disabled capability checks").to.eq(0));
    }
  });
});
