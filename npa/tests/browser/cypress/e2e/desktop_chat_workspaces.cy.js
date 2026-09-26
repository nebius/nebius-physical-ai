// Verify host workspace selection before creating a mobile Codex conversation.
import {mockChat} from "../support/desktop_chat";

describe("New chat workspace picker", () => {
  let chat;
  beforeEach(() => { chat = mockChat(); cy.viewport(390, 844); });

  it("highlights the NPA checkout and creates a chat in the chosen recent path", () => {
    cy.intercept("POST", "/chat/api/new", req => {
      expect(req.body.cwd).to.equal("/workspace/project");
      chat.thread = {...chat.thread, id: "new-chat", cwd: req.body.cwd};
      req.reply({thread: chat.thread});
    }).as("create");
    cy.visit("/chat/");
    cy.get("#new-chat").click();
    cy.wait("@workspaces");
    cy.get(".workspace-choice").first().should("have.class", "preferred")
      .and("contain", "nebius-physical-ai").and("contain", "NPA project");
    cy.get("#cwd").should("have.value", "/workspace/nebius-physical-ai");
    cy.get('[aria-label="Use /workspace/project"]').click().should("have.attr", "aria-pressed", "true");
    cy.get("#create-chat").click();
    cy.wait("@create");
    cy.location("hash").should("equal", "#new-chat");
    cy.get("#project").should("have.text", "/workspace/project");
  });

  it("keeps a failed custom path editable and does not submit twice", () => {
    let creates = 0;
    cy.intercept("POST", "/chat/api/new", req => {
      creates++;
      expect(req.body.cwd).to.equal("/workspace/Project with spaces α");
      req.reply({delay: 200, statusCode: 400, body: {error: "Choose an existing project directory."}});
    }).as("create");
    cy.visit("/chat/");
    cy.get("#new-chat").click();
    cy.wait("@workspaces");
    cy.get("#cwd").clear().type("/workspace/Project with spaces α");
    cy.get("#new-form").submit().submit();
    cy.wait("@create");
    cy.get("#new-dialog").should("be.visible");
    cy.get("#new-error").should("be.visible").and("contain", "existing project directory");
    cy.get("#cwd").should("be.enabled").and("have.value", "/workspace/Project with spaces α");
    cy.then(() => expect(creates).to.equal(1));
    cy.get("#cancel-new").click();
  });

  it("does not replace a path typed while recent choices load", () => {
    cy.intercept("GET", "/chat/api/workspaces", {delay: 400, body: {data: [
      {path: "/workspace/nebius-physical-ai", name: "nebius-physical-ai", preferred: true},
    ]}}).as("delayedPaths");
    cy.visit("/chat/");
    cy.get("#new-chat").click();
    cy.get("#cwd").clear().type("/workspace/custom", {delay: 0});
    cy.wait("@delayedPaths");
    cy.get("#cwd").should("have.value", "/workspace/custom");
    cy.get(".workspace-choice").should("have.attr", "aria-pressed", "false");
  });

  it("keeps custom paths usable when history cannot be loaded", () => {
    cy.intercept("GET", "/chat/api/workspaces", {statusCode: 503, body: {error: "Unavailable"}});
    cy.visit("/chat/");
    cy.get("#new-chat").click();
    cy.get("#workspace-status").should("contain", "You can still enter a path");
    cy.get("#cwd").should("be.enabled");
    cy.get("#create-chat").should("be.enabled");
  });

  it("wraps long paths without widening the phone dialog", () => {
    const path = "/workspace/" + "long-folder-name/".repeat(8) + "nebius-physical-ai";
    cy.intercept("GET", "/chat/api/workspaces", {data: [{path, name: "nebius-physical-ai", preferred: true}]});
    cy.visit("/chat/");
    cy.get("#new-chat").click();
    cy.get(".workspace-choice").should("contain", path);
    cy.get("#new-dialog").should(el => expect(el[0].scrollWidth).to.be.at.most(el[0].clientWidth));
    cy.get("#create-chat").should("be.visible");
  });
});
