// Exercise native selects and their dependent actions against the production
// document. These synthetic API fixtures contain no operational identities.
const capability = (status = "available") => ({ status, reason: `Fixture access is ${status}.` });
const bucket = (name, readStatus = "available") => ({
  type: "object_storage_bucket",
  name,
  capabilities: { artifact_discovery: capability(), artifact_read: capability(readStatus) },
});
const project = (id, name, resources) => ({
  id, name, resources,
  deployment_project: id === "project-a",
  capabilities: {
    artifact_discovery: capability(), artifact_read: capability(),
    workflow_submission: capability(id === "project-a" ? "available" : "unavailable"),
  },
});
const report = (projects) => ({
  apiVersion: "npa.agent.access/v1",
  identity: { deployment_project_id: "project-a", deployment_project_name: "Project Alpha" },
  scope: "partial_tenant", status: "available", projects, errors: [],
});
const multipleProjects = [
  project("project-a", "Project Alpha", [bucket("alpha-artifacts"), bucket("alpha-archive", "denied")]),
  project("project-b", "Project Beta", [bucket("beta-artifacts"), bucket("beta-archive")]),
];

function visitAccess(projects, visitOptions = {}) {
  cy.installAgentApiMocks();
  cy.intercept("GET", "/api/access*", { body: report(projects) }).as("accessReport");
  cy.intercept("GET", "/api/artifacts/runs*", (req) => {
    const url = new URL(req.url);
    const projectId = url.searchParams.get("project_id");
    const name = url.searchParams.get("resource_bucket");
    req.reply({ body: {
      ok: true, pagination_complete: true, total_runs: name ? 1 : 0, next_cursor: "",
      runs: name ? [{
        run_id: `${name}-run`, run_ref: `npa1_${name.replaceAll("-", "_")}`,
        project_id: projectId, bucket: name, resolved_prefix: "workflow-runs",
        source_type: "artifact_storage", summary_complete: true, has_viewable: true,
      }] : [],
    } });
  }).as("accessRuns");
  cy.visit("/", visitOptions);
  cy.wait("@accessReport");
  cy.get("#agentAccessPanel").should("have.attr", "aria-busy", "false");
  cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
}

function assertSelection(projectId, name) {
  cy.get("#agentAccessProjectSelect").should("be.enabled").and("have.value", projectId);
  cy.get("#agentAccessBucketSelect").should("be.enabled").and("have.value", name);
  cy.get("#agentAccessProjects .access-project-detail").should("have.attr", "data-project-id", projectId);
  cy.get("#agentAccessProjects .access-resource").should("have.attr", "data-selected-bucket", name);
  cy.get('#agentAccessProjects button[data-access-action="list"]')
    .should("have.attr", "data-project-id", projectId)
    .and("have.attr", "data-resource-bucket", name);
}

describe("Agent access native selection", () => {
  for (const scopeChange of ["project", "bucket"]) {
    it(`cancels a pending demo load when the access ${scopeChange} changes`, () => {
      visitAccess(multipleProjects);
      let releaseDemoDetails;
      cy.intercept({ method: "GET", url: "/api/workflows/sim2real/runs/franka-demo*", times: 1 }, (req) => {
        return new Cypress.Promise((resolve) => {
          releaseDemoDetails = () => {
            req.reply({ body: { run: { run_id: "franka-demo", stages: [], logs: [] } } });
            resolve();
          };
        });
      });
      cy.get("#tabRerun").click();
      cy.get("#runIdInput").clear().type("franka-demo");
      cy.get("#loadRunData").click();
      cy.wait("@loadFranka");
      cy.wrap(null).should(() => expect(releaseDemoDetails, "demo details are in flight").to.be.a("function"));
      const mountedRuns = [];
      cy.window().then((win) => {
        const frame = win.document.getElementById("rerunFrame");
        const observer = new win.MutationObserver(() => mountedRuns.push(frame.dataset.rerunRunKey || ""));
        observer.observe(frame, { attributes: true, attributeFilter: ["data-rerun-run-key"] });
      });
      cy.get("#tabMain").click();
      if (scopeChange === "project") {
        cy.get("#agentAccessProjectSelect").select("project-b");
        assertSelection("project-b", "beta-artifacts");
      } else {
        cy.get("#agentAccessBucketSelect").select("alpha-archive");
        assertSelection("project-a", "alpha-archive");
      }
      cy.then(() => releaseDemoDetails());
      // Await completion of the original real click, including any resumed
      // mount/refresh, before checking that the cleared scope stayed cleared.
      cy.get("#loadRunData").should("have.attr", "aria-busy", "false");
      cy.get("#runSummary").should("have.text", "Select a run to load its result.");
      cy.get("#runIdSelect option").should("not.contain.text", "franka-demo");
      cy.get("#rerunFrame").should("have.attr", "hidden");
      cy.get("#rerunFrame").should("not.have.attr", "src");
      cy.then(() => expect(mountedRuns, "old demo never remounts in the new storage scope")
        .not.to.include("franka-demo"));
    });
  }

  it("scopes discovery on the first explicit dropdown change without a preceding List action", () => {
    visitAccess(multipleProjects);
    cy.get("#agentAccessProjectSelect").select("project-b");
    cy.get("#agentAccessBucketSelect").select("beta-archive");
    assertSelection("project-b", "beta-archive");
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.get("#runIdSelect option").should("contain.text", "beta-archive-run");
    cy.get("#artifactDiscoverStatus").should("contain.text", "project-b").and("contain.text", "beta-archive");
  });

  it("applies project and bucket changes to details and subsequent discovery", () => {
    visitAccess(multipleProjects);
    assertSelection("project-a", "alpha-artifacts");
    cy.get('#agentAccessProjects button[data-access-action="list"]').click();
    cy.get("#agentAccessActionResult").should("contain.text", "alpha-artifacts-run");

    cy.get("#agentAccessBucketSelect").select("alpha-archive");
    assertSelection("project-a", "alpha-archive");
    cy.get('#agentAccessProjects button[data-access-action="read"]').should("be.disabled");
    cy.get("#agentAccessActionResult").should("have.attr", "hidden");
    cy.get("#runIdSelect option").should("not.contain.text", "alpha-artifacts-run");

    // Do not click another access action: ordinary discovery must already use
    // the selected pair, instead of retaining the last List button's scope.
    cy.get("#agentAccessProjectSelect").focus();
    // cy.press sends native browser keyboard input, unlike invoking a handler
    // or dispatching a source-only test hook.
    cy.press(Cypress.Keyboard.Keys.DOWN);
    cy.press(Cypress.Keyboard.Keys.TAB);
    cy.get("#agentAccessBucketSelect").select("beta-archive");
    assertSelection("project-b", "beta-archive");
    cy.get('#agentAccessProjects button[data-access-action="read"]').should("be.enabled");
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.get("#runIdSelect option").should("contain.text", "beta-archive-run")
      .and("not.contain.text", "alpha-artifacts-run");
    cy.get("#artifactDiscoverStatus").should("contain.text", "project-b").and("contain.text", "beta-archive");
    cy.get("#tabMain").click();
    cy.get("#agentAccessRefresh").click();
    cy.wait("@accessReport");
    assertSelection("project-b", "beta-archive");
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.get("#runIdSelect option").should("contain.text", "beta-archive-run");
    cy.get("#artifactDiscoverStatus").should("contain.text", "project-b").and("contain.text", "beta-archive");
  });

  it("applies a sole project and bucket without requiring an impossible value change", () => {
    visitAccess([project("project-a", "Project Alpha", [bucket("alpha-artifacts")])]);
    assertSelection("project-a", "alpha-artifacts");
    cy.get("#agentAccessProjectSelect option").should("have.length", 1);
    cy.get("#agentAccessBucketSelect option").should("have.length", 1);
    // A real select whose sole option is already selected need not emit change.
    // The initial access render must have committed the pair and its actions.
    const changes = [];
    cy.get("#agentAccessProjectSelect, #agentAccessBucketSelect").then(($selects) => {
      [...$selects].forEach((select) => select.addEventListener("change", (event) => changes.push(event.target.id)));
    });
    cy.get("#agentAccessProjectSelect").focus().should("be.focused");
    cy.press(Cypress.Keyboard.Keys.DOWN);
    cy.press(Cypress.Keyboard.Keys.TAB);
    cy.get("#agentAccessBucketSelect").should("be.focused");
    cy.press(Cypress.Keyboard.Keys.DOWN);
    cy.press(Cypress.Keyboard.Keys.TAB);
    cy.then(() => expect(changes, "sole-option keyboard selection emits no change").to.deep.equal([]));
    cy.get("#agentAccessProjectHint").should("contain.text", "Only available project");
    cy.get("#agentAccessBucketHint").should("contain.text", "Only available bucket");
    cy.get('#agentAccessProjects button[data-access-action="list"]').click();
    cy.get("#agentAccessActionResult").should("contain.text", "alpha-artifacts-run");
    cy.get("#agentAccessRefresh").click();
    cy.wait("@accessReport");
    assertSelection("project-a", "alpha-artifacts");
    cy.get('#agentAccessProjects button[data-access-action="list"]').click();
    cy.get("#agentAccessActionResult").should("contain.text", "alpha-artifacts-run");
    cy.get("#agentAccessRefresh").click();
    cy.wait("@accessReport");
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.get("#runIdSelect option").should("contain.text", "alpha-artifacts-run");
  });

  it("retains project scope when a selected project has no searchable bucket", () => {
    visitAccess([...multipleProjects, project("project-c", "No Storage", [])]);
    cy.get('#agentAccessProjects button[data-access-action="list"]').click();
    cy.get("#agentAccessActionResult").should("contain.text", "alpha-artifacts-run");
    cy.get("#agentAccessProjectSelect").select("project-c");
    cy.get("#agentAccessBucketSelect").should("be.disabled").and("have.value", "");
    cy.get("#agentAccessProjects").should("contain.text", "No searchable artifact bucket");
    cy.get("#agentAccessProjects button[data-access-action]").should("not.exist");
    cy.intercept("GET", "/api/artifacts/runs*", (req) => {
      expect(new URL(req.url).searchParams.get("project_id")).to.eq("project-c");
      req.reply({ statusCode: 403, body: { detail: "selected project has no searchable artifact bucket" } });
    }).as("noBucketDiscovery");
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@noBucketDiscovery");
    cy.get("#runIdSelect option").should("not.contain.text", "alpha-artifacts-run");
    cy.get("#toastHost").should("contain.text", "selected project has no searchable artifact bucket");
  });

  it("clears scoped runs when access becomes empty and never broadens discovery", () => {
    visitAccess(multipleProjects);
    cy.get('#agentAccessProjects button[data-access-action="list"]').click();
    cy.get("#agentAccessActionResult").should("contain.text", "alpha-artifacts-run");
    // Changing the pair cancels the List action; ordinary discovery still owns
    // the selected scope when the later access refresh removes every project.
    cy.get("#agentAccessProjectSelect").select("project-b");
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.get("#runIdSelect option").should("contain.text", "beta-artifacts-run");
    cy.intercept("GET", "/api/access*", { body: report([]) }).as("emptyAccessReport");
    cy.get("#tabMain").click();
    cy.get("#agentAccessRefresh").click();
    cy.wait("@emptyAccessReport");
    cy.get("#agentAccessProjectSelect").should("be.disabled").and("have.value", "");
    cy.get("#agentAccessBucketSelect").should("be.disabled").and("have.value", "");
    cy.get("#runIdSelect option").should("not.contain.text", "beta-artifacts-run");
    let requestsAfterAccessLoss = 0;
    cy.intercept("GET", "/api/artifacts/runs*", (req) => {
      requestsAfterAccessLoss += 1;
      req.reply({ body: { runs: [], next_cursor: "", pagination_complete: true } });
    });
    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.get("#toastHost").should("contain.text", "No project is available");
    cy.get("#artifactRefreshRuns").should("have.attr", "aria-busy", "false");
    cy.then(() => expect(requestsAfterAccessLoss, "no unscoped discovery after access loss").to.eq(0));
  });
});
