describe("deployment-isolated browser identity", () => {
  beforeEach(() => {
    cy.visitMockAgent();
    cy.wait("@session");
  });

  it("renders config-driven workspace provenance", () => {
    cy.get("#workspaceIdentity").should("have.text", "Wan Workbench");
    cy.get("#deploymentProvenance")
      .should("contain.text", "wan-pr261")
      .and("contain.text", "0123456789ab");
    cy.get("html").should("have.attr", "data-deployment-id", "npa-agent-mocked-wan");
    cy.get("body").should("not.contain.text", "LeIsaac");
  });

  it("clears stale deployment-scoped state on hard refresh", () => {
    cy.window().then((win) => {
      win.localStorage.setItem("npa_agent_deployment_id", "npa-agent-other-branch");
      win.localStorage.setItem("npa_solution_state", "LeIsaac");
      win.localStorage.setItem("studio.profile-data", "other-deployment-layout");
      win.sessionStorage.setItem("npa_previous_run", "other-run");
    });
    cy.reload(true);
    cy.wait("@session");
    cy.window().then((win) => {
      expect(win.localStorage.getItem("npa_agent_deployment_id")).to.eq("npa-agent-mocked-wan");
      expect(win.localStorage.getItem("npa_solution_state")).to.eq(null);
      expect(win.localStorage.getItem("studio.profile-data")).to.eq(null);
      expect(win.sessionStorage.getItem("npa_previous_run")).to.eq(null);
    });
    cy.get("#workspaceIdentity").should("have.text", "Wan Workbench");
    cy.get("body").should("not.contain.text", "LeIsaac");
  });
});
