// Preserve composer state and delivery identity through navigation and reconnects.
import {mockChat} from "../support/desktop_chat";

const image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9RkAAAAASUVORK5CYII=";

function chooseChat(name) {
  cy.get("#menu").click();
  cy.contains(".session", name).click();
  cy.get("#prompt").should("be.enabled");
}

describe("Codex persistence under navigation", () => {
  let chat;
  beforeEach(() => {
    chat = mockChat();
    const other = {...chat.thread, id: "other-thread", name: "Another conversation"};
    cy.intercept("GET", "/chat/api/threads*", {data: [chat.thread, other]});
    cy.intercept("POST", "/chat/api/resume", req =>
      req.reply({thread: req.body.id === other.id ? other : chat.thread}));
    cy.viewport(390, 844);
    cy.visit("/chat/#same-thread");
    cy.get("#prompt").should("be.enabled");
  });

  it("retains an unsent image draft across reload and chat switches", () => {
    cy.get("#prompt").type("Keep this image draft");
    cy.get("#image-input").selectFile({contents: Cypress.Buffer.from(image.split(",")[1], "base64"),
      fileName: "draft.png", mimeType: "image/png"}, {force: true});
    cy.get(".attachment img").should("have.attr", "src", image);
    cy.reload();
    cy.get("#prompt").should("have.value", "Keep this image draft");
    cy.get(".attachment img").should("have.attr", "src", image);
    chooseChat("Another conversation");
    cy.get(".attachment").should("not.exist");
    chooseChat("Shared conversation");
    cy.get(".attachment img").should("have.attr", "src", image);
    cy.get(".attachment").click();
    cy.reload();
    cy.get(".attachment").should("not.exist");
  });

  it("clears only the sent chat's draft when its receipt arrives after switching", () => {
    let ready = false;
    cy.intercept("POST", "/chat/api/send", {statusCode: 202, body: {state: "pending"}}).as("send");
    cy.intercept("GET", "/chat/api/delivery?*", req =>
      req.reply(ready ? {state: "complete", result: {accepted: true}} : {state: "pending"}));
    cy.get("#prompt").type("Already sent message");
    cy.get("#send").click();
    cy.wait("@send");
    chooseChat("Another conversation");
    cy.get("#prompt").type("Keep the other draft");
    cy.then(() => { ready = true; });
    cy.window().should(win => expect(win.localStorage.getItem("codex-send:same-thread")).to.equal(null));
    cy.get("#prompt").should("have.value", "Keep the other draft");
    chooseChat("Shared conversation");
    cy.get("#prompt").should("have.value", "");
    cy.reload();
    cy.get("#prompt").should("have.value", "");
  });

  it("stops retrying a missing receipt after the pending send is discarded", () => {
    let sends = 0;
    cy.intercept("POST", "/chat/api/send", req => {
      sends++;
      req.reply({statusCode: 503, body: {error: "Disconnected"}});
    }).as("send");
    cy.intercept("GET", "/chat/api/delivery?*", {state: "missing"});
    cy.get("#prompt").type("Keep as an unsent draft");
    cy.get("#send").click();
    cy.wait("@send");
    cy.get("#clear-send").click();
    cy.wait(2300);
    cy.then(() => expect(sends).to.equal(1));
    cy.get("#prompt").should("have.value", "Keep as an unsent draft");
    cy.get("#send").should("be.enabled");
  });

  it("opens a bookmarked conversation without a page reload", () => {
    cy.get("#prompt").type("Keep the first draft");
    cy.window().then(win => { win.location.hash = "#other-thread"; });
    cy.get("#title").should("have.text", "Another conversation");
    cy.get("#prompt").should("have.value", "");
    cy.window().then(win => { win.location.hash = "#same-thread"; });
    cy.get("#title").should("have.text", "Shared conversation");
    cy.get("#prompt").should("have.value", "Keep the first draft");
  });

  it("lets another conversation send while the first receipt is pending", () => {
    cy.intercept("POST", "/chat/api/send", req => req.reply(req.body.id === "same-thread"
      ? {state: "pending"} : {state: "complete", result: {accepted: true}})).as("send");
    cy.intercept("GET", "/chat/api/delivery?*", {state: "pending"});
    cy.get("#prompt").type("Waiting for the first receipt");
    cy.get("#send").click();
    cy.wait("@send");
    chooseChat("Another conversation");
    cy.get("#prompt").type("Independent task");
    cy.get("#send").should("be.enabled").click();
    cy.wait("@send").its("request.body.id").should("equal", "other-thread");
    cy.get("#prompt").should("have.value", "");
    chooseChat("Shared conversation");
    cy.get("#prompt").should("have.value", "Waiting for the first receipt");
  });

  it("preserves a newly retyped identical draft when an older send completes", () => {
    let ready = false;
    cy.intercept("POST", "/chat/api/send", {state: "pending"}).as("send");
    cy.intercept("GET", "/chat/api/delivery?*", req =>
      req.reply(ready ? {state: "complete", result: {accepted: true}} : {state: "pending"}));
    cy.get("#prompt").type("Repeat this later");
    cy.get("#send").click();
    cy.wait("@send");
    cy.get("#prompt").clear().type("Repeat this later");
    cy.then(() => { ready = true; });
    cy.get("#clear-send").should("not.be.visible");
    cy.reload();
    cy.get("#prompt").should("have.value", "Repeat this later");
  });

  it("does not let a late metadata read undo a confirmed archive", () => {
    let release;
    cy.intercept({method: "GET", pathname: "/chat/api/thread"}, req => {
      if (release) return req.reply({thread: chat.thread});
      req.alias = "staleMetadata";
      return new Promise(resolve => {
        release = () => { req.reply({thread: {id: "same-thread", archived: false}}); resolve(); };
      });
    });
    cy.intercept("POST", "/chat/api/archive", req => {
      chat.thread.archived = true;
      req.reply({ok: true});
    }).as("archive");
    cy.get("#refresh").click();
    cy.wrap(null).should(() => expect(release).to.be.a("function"));
    cy.get("#manage-chat").click();
    cy.get("#archive-chat").should("be.enabled").click();
    cy.wait("@archive");
    cy.get("#run-state").should("contain", "Archived");
    cy.then(() => release());
    cy.wait("@staleMetadata");
    cy.get("#run-state").should("contain", "Archived");
    cy.get("#prompt").should("be.disabled");
  });
});
