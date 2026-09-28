// Verify phone submissions survive slow acknowledgments and interrupted requests.
import {mockChat} from "../support/desktop_chat";

describe("Shared Codex delivery receipts", () => {
  beforeEach(() => {
    mockChat();
    cy.viewport(390, 844);
    cy.visit("/chat/#same-thread");
    cy.get("#prompt").should("be.enabled");
  });

  it("waits for a slow owner without holding open or repeating the submit request", () => {
    let ready = false, sends = 0, identifier;
    cy.intercept("POST", "/chat/api/send", req => {
      sends++;
      identifier = req.body.clientUserMessageId;
      expect(req.headers.prefer).to.equal("respond-async");
      req.reply({statusCode: 202, body: {state: "pending"}});
    }).as("submit");
    cy.intercept("GET", "/chat/api/delivery?*", req => {
      expect(new URL(req.url).searchParams.get("clientUserMessageId")).to.equal(identifier);
      req.reply(ready ? {state: "complete", result: {accepted: true}} : {state: "pending"});
    }).as("receipt");
    cy.get("#prompt").type("Continue the slow task");
    cy.get("#send").click();
    cy.wait("@submit");
    cy.wait("@receipt");
    cy.get("#delivery-status").should("contain", "Checking delivery");
    cy.get("#prompt").should("have.value", "Continue the slow task");
    cy.then(() => { ready = true; });
    cy.get("#prompt").should("have.value", "");
    cy.get("#clear-send").should("not.be.visible");
    cy.then(() => expect(sends).to.equal(1));
  });

  it("recovers a lost gateway response from its receipt without resending", () => {
    let sends = 0;
    cy.intercept("POST", "/chat/api/send", req => {
      sends++;
      req.reply({statusCode: 504, body: "<html>Gateway timeout</html>", headers: {"content-type": "text/html"}});
    });
    cy.intercept("GET", "/chat/api/delivery?*", {state: "complete", result: {accepted: true}});
    cy.get("#prompt").type("Only one submission");
    cy.get("#send").click();
    cy.get("#prompt").should("have.value", "");
    cy.get("#notice").should("not.be.visible");
    cy.then(() => expect(sends).to.equal(1));
  });

  it("resumes receipt checking after reload and preserves a newer draft", () => {
    let ready = false, sends = 0;
    cy.intercept("POST", "/chat/api/send", req => {
      sends++;
      req.reply({statusCode: 202, body: {state: "pending"}});
    }).as("submit");
    cy.intercept("GET", "/chat/api/delivery?*", req =>
      req.reply(ready ? {state: "complete", result: {accepted: true}} : {state: "pending"}));
    cy.get("#prompt").type("Original message");
    cy.get("#send").click();
    cy.wait("@submit");
    cy.reload();
    cy.get("#prompt").should("be.enabled").clear().type("New unsent draft");
    cy.then(() => { ready = true; });
    cy.get("#clear-send").should("not.be.visible");
    cy.get("#prompt").should("have.value", "New unsent draft");
    cy.then(() => expect(sends).to.equal(1));
  });

  it("retains uncertain delivery and never automatically starts another turn", () => {
    let sends = 0;
    cy.intercept("POST", "/chat/api/send", req => {
      sends++;
      req.reply({state: "uncertain", error: "Owner acknowledgment timed out"});
    });
    cy.get("#prompt").type("Do not duplicate this task");
    cy.get("#send").click();
    cy.get("#notice").should("contain", "Delivery is unconfirmed");
    cy.get("#prompt").should("have.value", "Do not duplicate this task");
    cy.get("#clear-send").should("be.visible");
    cy.then(() => expect(sends).to.equal(1));
  });

  it("reuses the identity only when the original request never reached the server", () => {
    const ids = [];
    cy.intercept("POST", "/chat/api/send", req => {
      ids.push(req.body.clientUserMessageId);
      req.reply(ids.length === 1 ? {forceNetworkError: true} : {state: "complete", result: {accepted: true}});
    });
    cy.intercept("GET", "/chat/api/delivery?*", {state: "missing"});
    cy.get("#prompt").type("Recover a disconnected phone");
    cy.get("#send").click();
    cy.get("#prompt").should("have.value", "");
    cy.then(() => { expect(ids).to.have.length(2); expect(ids[1]).to.equal(ids[0]); });
  });

  it("keeps the draft when login expires instead of retrying authentication", () => {
    let sends = 0;
    cy.intercept("POST", "/chat/api/send", req => {
      sends++;
      req.reply({statusCode: 401, body: {error: "Sign in"}});
    });
    cy.get("#prompt").type("Keep my draft");
    cy.get("#send").click();
    cy.get("#notice").should("contain", "Sign-in expired");
    cy.get("#prompt").should("have.value", "Keep my draft");
    cy.then(() => expect(sends).to.equal(1));
  });
});
