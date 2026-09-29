// Verify completion indicators through cloud events, native polling, and navigation.
import {mockChat, publish} from "../support/desktop_chat";

function complete(chat, id = "reply-turn") {
  const turn = {id, status: "completed", items: [
    {id: id + "-message", type: "agentMessage", text: "Finished reply"},
  ]};
  chat.thread.status = {type: "idle"};
  chat.turns = [turn];
  publish(chat, "turn/completed", {turn});
}

function visitChat(path = "/chat/#same-thread") {
  cy.visit(path, {onBeforeLoad(win) {
    cy.stub(win.document, "hasFocus").returns(true);
  }});
  cy.get(".session").should("exist");
}

describe("Unread Codex replies", () => {
  let chat;
  beforeEach(() => { chat = mockChat(); cy.viewport(390, 844); });

  it("lists a new cloud chat as soon as its first message is accepted", () => {
    let saved = false;
    cy.intercept("GET", "/chat/api/threads*", req =>
      req.reply({data: saved ? [chat.thread] : [], nextCursor: null}));
    cy.intercept("POST", "/chat/api/send", req => {
      saved = true;
      chat.thread.status = {type: "active"};
      chat.turns = [{id: "first-turn", status: "inProgress", items: []}];
      req.reply({turn: chat.turns[0]});
    });
    cy.visit("/chat/#same-thread");
    cy.get("#prompt").should("be.enabled").type("First message");
    cy.get("#send").click();
    cy.get("#menu").click();
    cy.get(".session-spinner").should("be.visible");
    cy.then(() => complete(chat, "first-turn"));
    cy.get(".session-unread").should("be.visible");
  });

  it("replaces a sent turn's spinner with a dot until its reply is opened", () => {
    cy.intercept("POST", "/chat/api/send", req => {
      chat.thread.status = {type: "active"};
      chat.turns = [{id: "reply-turn", status: "inProgress", items: []}];
      req.reply({turn: chat.turns[0]});
    }).as("send");
    visitChat();
    cy.get("#prompt").should("be.enabled").type("Start work");
    cy.get("#send").click();
    cy.wait("@send");
    cy.get("#menu").click();
    cy.get(".session-spinner").should("be.visible");
    cy.then(() => complete(chat));
    cy.get(".session-unread").should("be.visible").and("have.attr", "aria-label", "Unread reply");
    cy.get(".session-spinner").should("not.exist");
    cy.get(".session").click();
    cy.get("#messages").should("contain", "Finished reply");
    cy.get(".session-unread").should("not.exist");
    cy.then(() => complete(chat));
    cy.get("#menu").click();
    cy.get(".session-unread").should("not.exist");
  });

  it("keeps a VS Code completion unread across reload and failed opens", () => {
    visitChat("/chat/");
    cy.then(() => complete(chat));
    cy.get(".session-unread").should("be.visible");
    cy.reload();
    cy.get(".session-unread").should("be.visible");
    cy.intercept("POST", "/chat/api/resume", {statusCode: 503, body: {error: "Owner unavailable"}});
    cy.get(".session").click();
    cy.get("#notice").should("contain", "Owner unavailable");
    cy.get("#menu").click();
    cy.get(".session-unread").should("be.visible");
  });

  it("marks a native active-to-idle refresh unread without completion events", () => {
    chat.thread.status = {type: "active"};
    visitChat("/chat/");
    cy.get(".session-spinner").should("be.visible");
    cy.then(() => {
      chat.thread.status = {type: "idle"};
      publish(chat, "native/changed", {});
    });
    cy.get(".session-unread").should("be.visible");
    cy.get(".session-spinner").should("not.exist");
  });

  it("remembers a running chat when the page closes before completion", () => {
    chat.thread.status = {type: "active"};
    visitChat("/chat/");
    cy.get(".session-spinner").should("be.visible");
    cy.then(() => { chat.thread.status = {type: "idle"}; });
    cy.reload();
    cy.get(".session-unread").should("be.visible");
  });

  it("does not mark old idle history or a rename as a new reply", () => {
    visitChat("/chat/");
    cy.get(".session-unread").should("not.exist");
    cy.then(() => {
      chat.thread.name = "Renamed in VS Code";
      chat.thread.updatedAt = Date.now() / 1000;
      publish(chat, "native/changed", {});
    });
    cy.get(".session-title").should("have.text", "Renamed in VS Code");
    cy.get(".session-unread").should("not.exist");
  });

  it("keeps a hidden selected chat unread until it is focused and refreshed", () => {
    visitChat();
    cy.get("#prompt").should("be.enabled");
    cy.document().then(doc => { cy.stub(doc, "hidden").get(() => true); });
    cy.then(() => complete(chat));
    cy.get(".session-unread").should("exist");
    cy.document().then(doc => {
      Object.defineProperty(doc, "hidden", {configurable: true, value: false});
      doc.dispatchEvent(new Event("visibilitychange"));
    });
    cy.get("#messages").should("contain", "Finished reply");
    cy.get(".session-unread").should("not.exist");
  });

  it("does not leave a dot for a finished reply already being viewed", () => {
    visitChat();
    cy.get("#prompt").should("be.enabled");
    cy.then(() => complete(chat));
    cy.get("#messages").should("contain", "Finished reply");
    cy.get(".session-unread").should("not.exist");
  });

  it("keeps a reply unread while the browser window has lost focus", () => {
    visitChat();
    cy.get("#prompt").should("be.enabled");
    cy.document().then(doc => doc.hasFocus.returns(false));
    cy.then(() => complete(chat));
    cy.get("#messages").should("contain", "Finished reply");
    cy.get(".session-unread").should("exist");
    cy.document().then(doc => doc.hasFocus.returns(true));
    cy.window().trigger("focus");
    cy.get(".session-unread").should("not.exist");
  });

  it("keeps the dot while reading older messages and clears it at the reply", () => {
    chat.turns = [{id: "old", status: "completed", items: [
      {id: "old-message", type: "agentMessage", text: "Earlier history\n".repeat(100)},
    ]}];
    visitChat();
    cy.get("#prompt").should("be.enabled");
    cy.get("#conversation").scrollTo("top");
    cy.then(() => {
      const old = chat.turns[0];
      complete(chat);
      chat.turns.push(old);
    });
    cy.get(".session-unread").should("exist");
    cy.get("#conversation").scrollTo("bottom");
    cy.get(".session-unread").should("not.exist");
  });

  it("clears a viewed native completion after concurrent metadata refreshes", () => {
    chat.thread.status = {type: "active"};
    chat.turns = [{id: "native-history", status: "inProgress", items: []}];
    visitChat();
    cy.get("#prompt").should("be.enabled");
    cy.intercept("GET", "/chat/api/turns?*", req =>
      req.reply({delay: 200, body: {data: chat.turns, nextCursor: null}}));
    cy.then(() => {
      chat.thread.status = {type: "idle"};
      chat.turns = [{id: "native-history", status: "completed", items: [
        {id: "native-reply", type: "agentMessage", text: "Native reply"},
      ]}];
      publish(chat, "native/changed", {});
    });
    cy.get("#messages").should("contain", "Native reply");
    cy.get(".session-spinner").should("not.exist");
    cy.get(".session-unread").should("not.exist");
  });

  it("does not clear a completion with a history response requested before it", () => {
    let release;
    cy.intercept("GET", "/chat/api/turns?*", req => new Promise(resolve => {
      release = () => { req.reply({data: [], nextCursor: null}); resolve(); };
    }));
    visitChat();
    cy.wrap(null).should(() => expect(release).to.be.a("function"));
    cy.then(() => {
      publish(chat, "thread/status/changed", {status: {type: "active"}});
      publish(chat, "thread/status/changed", {status: {type: "idle"}});
    });
    cy.get(".session-unread").should("exist");
    cy.then(() => release());
    cy.get("#prompt").should("be.enabled");
    cy.get(".session-unread").should("exist");
  });
});
