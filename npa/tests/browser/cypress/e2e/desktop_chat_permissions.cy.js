// Verify permission changes and failures through the actual conversation controls.
import {mockChat, publish} from "../support/desktop_chat";

describe("Codex conversation permissions", () => {
  let chat;
  beforeEach(() => { chat = mockChat(); cy.viewport(1280, 900); });

  it("does not invent a policy when the runtime has not reported one", () => {
    cy.visit("/chat/#same-thread");
    cy.get("#permissions").should("be.enabled").and("have.prop", "value", "unknown");
    cy.get('#permissions option:selected').should("have.text", "Choose permissions");
  });

  it("updates the same conversation and reflects runtime-normalized policies", () => {
    cy.intercept("POST", "/chat/api/settings", req => {
      expect(req.body.id).to.equal("same-thread");
      expect(req.body.permissionMode).to.equal("workspace");
      expect(req.body).not.to.have.property("sandboxPolicy");
      Object.assign(chat.thread, {approvalPolicy: "on-request", sandboxPolicy: {
        type: "workspaceWrite", writableRoots: [], networkAccess: false,
        excludeTmpdirEnvVar: false, excludeSlashTmp: false,
      }});
      req.reply({ok: true});
    }).as("permissionChange");
    cy.visit("/chat/#same-thread");
    cy.get("#permissions").should("be.enabled").select("workspace");
    cy.wait("@permissionChange");
    cy.get("#permissions").should("have.value", "workspace");
    cy.then(() => publish(chat, "thread/settings/updated", {threadSettings: {
      model: "model-a", effort: "high", approvalPolicy: "never", sandboxPolicy: {type: "dangerFullAccess"},
    }}));
    cy.get("#permissions").should("have.value", "full");
  });

  it("retains a custom policy after a rejected change", () => {
    Object.assign(chat.thread, {approvalPolicy: "on-request", sandboxPolicy: {
      type: "workspaceWrite", writableRoots: ["/workspace/extra"], networkAccess: true,
    }});
    cy.intercept("POST", "/chat/api/settings", {statusCode: 400, body: {error: "Policy change denied"}});
    cy.visit("/chat/#same-thread");
    cy.get("#permissions").should("have.prop", "value", "custom").select("full");
    cy.get("#notice").should("contain", "Policy change denied");
    cy.get("#permissions").should("have.prop", "value", "custom");
  });
});
