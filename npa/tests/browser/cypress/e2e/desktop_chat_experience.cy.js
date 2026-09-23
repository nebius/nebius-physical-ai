// Keep conversation reading, mobile composing, and recovery usable during updates.
import {mockChat, publish} from "../support/desktop_chat";

function message(id, text) {
  return {id: "turn-" + id, status: "completed", items: [
    {id, type: "agentMessage", text},
  ]};
}

describe("Codex conversation experience", () => {
  let chat;
  beforeEach(() => { chat = mockChat(); cy.viewport(390, 844); });

  it("recovers a failed chat open through Refresh without losing the draft", () => {
    let attempts = 0;
    cy.intercept("POST", "/chat/api/resume", req => {
      if (++attempts === 1) req.reply({statusCode: 503, body: {error: "Temporarily unavailable"}});
      else req.reply({thread: chat.thread});
    });
    cy.visit("/chat/#same-thread", {onBeforeLoad(win) {
      win.localStorage.setItem("codex-draft:same-thread", "Preserved draft");
    }});
    cy.get("#notice").should("contain", "Temporarily unavailable");
    cy.get("#refresh").click();
    cy.get("#prompt").should("be.enabled").and("have.value", "Preserved draft");
    cy.get("#notice").should("not.be.visible");
    cy.then(() => expect(attempts).to.equal(2));
  });

  it("leaves room to read and type with a phone keyboard open", () => {
    chat.turns = [message("reading", "A message to keep visible while composing.")];
    cy.viewport(390, 430);
    cy.visit("/chat/#same-thread");
    cy.get("#prompt").should("be.enabled").type("Typing a reply");
    cy.get("#conversation").then(el => expect(el[0].clientHeight).to.be.greaterThan(150));
    cy.get("#send").then(el => expect(el[0].getBoundingClientRect().bottom).to.be.at.most(430));
    cy.get("#toggle-settings").click();
    cy.get("#model").should("be.visible");
    cy.get("#effort").select("medium");
    cy.get("#mode").should("be.enabled").select("plan");
    cy.get("#prompt").focus();
    cy.get("#model").should("not.be.visible");
    cy.get("#toggle-settings").should("contain", "Medium").and("contain", "Plan");
    cy.get("#prompt").should("have.value", "Typing a reply");
  });

  it("keeps the newest reply visible when the phone viewport shrinks", () => {
    chat.turns = [message("last", "Reading the newest reply\n".repeat(50))];
    cy.visit("/chat/#same-thread");
    cy.get('[data-item-id="last"]').should("exist");
    cy.get("#conversation").scrollTo("bottom");
    cy.viewport(390, 430);
    cy.get("#conversation").should(el => {
      const viewport = el[0];
      expect(viewport.scrollHeight-viewport.scrollTop-viewport.clientHeight).to.be.lessThan(3);
    });
    cy.get("#conversation").scrollTo("top");
    cy.viewport(390, 380);
    cy.get("#conversation").should(el => expect(el[0].scrollTop).to.be.lessThan(3));
  });

  it("keeps earlier messages and the reading position after refresh", () => {
    const latest = [message("latest", "Recent message\n".repeat(40))];
    const earlier = [message("earlier", "Earlier message\n".repeat(40))];
    cy.intercept("GET", "/chat/api/turns?*", req => req.reply(
      new URL(req.url).searchParams.has("cursor")
        ? {data: earlier, nextCursor: null} : {data: latest, nextCursor: "page-two"}));
    cy.visit("/chat/#same-thread");
    cy.get('[data-item-id="latest"]').should("exist");
    cy.get("#conversation").scrollTo("top");
    let position;
    cy.get('[data-item-id="latest"]').then(el => { position = el[0].getBoundingClientRect().top; });
    cy.get("#older").click({scrollBehavior: false});
    cy.get('[data-item-id="earlier"]').should("exist");
    cy.get('[data-item-id="latest"]').then(el => expect(Math.abs(el[0].getBoundingClientRect().top-position)).to.be.lessThan(3));
    cy.get("#refresh").click();
    cy.get('[data-item-id="earlier"]').should("exist");
    cy.get("#older").should("not.be.visible");
  });

  it("honors a complete history snapshot after an IDE rollback", () => {
    chat.turns = [message("removed", "This turn will be rolled back")];
    cy.visit("/chat/#same-thread");
    cy.get('[data-item-id="removed"]').should("exist");
    cy.then(() => { chat.turns = [message("retained", "Authoritative history")]; });
    cy.get("#refresh").click();
    cy.get('[data-item-id="retained"]').should("exist");
    cy.get('[data-item-id="removed"]').should("not.exist");
  });

  it("keeps additional session pages and scroll position during live updates", () => {
    const sessions = Array.from({length: 30}, (_,i) => ({...chat.thread, id: "session-"+i, name: "Session "+i}));
    cy.intercept("GET", "/chat/api/threads*", req => req.reply(
      new URL(req.url).searchParams.has("cursor")
        ? {data: sessions.slice(15), nextCursor: null} : {data: sessions.slice(0,15), nextCursor: "page-two"})).as("sessionPage");
    cy.visit("/chat/#same-thread");
    cy.wait("@sessionPage");
    cy.get("#menu").click();
    cy.get("#more-sessions").click();
    cy.wait("@sessionPage");
    cy.get(".session").should("have.length",30);
    cy.get("#sessions").scrollTo("bottom");
    let position;
    cy.get("#sessions").then(el => { position = el[0].scrollTop; });
    cy.then(() => publish(chat, "native/changed", {}));
    cy.wait("@sessionPage");
    cy.get(".session").should("have.length",30);
    cy.get("#sessions").then(el => expect(el[0].scrollTop).to.be.closeTo(position,2));
    cy.get("#more-sessions").should("not.be.visible");
    cy.get("#search").type("Different search");
    cy.get(".session").should("have.length",15);
  });

  it("does not send while an input method is composing text", () => {
    let sends = 0;
    cy.intercept("POST", "/chat/api/send", req => { sends++; req.reply({accepted:true}); });
    cy.visit("/chat/#same-thread");
    cy.get("#prompt").should("be.enabled").type("Unfinished composition");
    cy.window().then(win => win.document.querySelector("#prompt").dispatchEvent(
      new win.KeyboardEvent("keydown", {key:"Enter", ctrlKey:true, isComposing:true, bubbles:true, cancelable:true})));
    cy.get("#prompt").should("have.value", "Unfinished composition");
    cy.then(() => expect(sends).to.equal(0));
  });

  it("renders multiple fenced code blocks without swallowing surrounding text", () => {
    chat.turns = [message("code", "Before\n```python\nprint('first')\n```\nBetween\n```sh\necho second\n```\nAfter <img src=x>")];
    cy.visit("/chat/#same-thread");
    cy.get(".message pre").should("have.length",2);
    cy.get(".message pre").eq(0).should("have.text","print('first')\n");
    cy.get(".message pre").eq(1).should("have.text","echo second\n");
    cy.get(".message").should("contain","Before").and("contain","Between").and("contain","After <img src=x>");
    cy.get(".message img").should("not.exist");
  });
});
