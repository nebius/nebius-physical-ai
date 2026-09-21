// Exercise shared Codex controls, live activity, and message identity in a browser.
const model = (name, efforts) => ({
  model: name,
  displayName: name,
  defaultReasoningEffort: efforts[0],
  supportedReasoningEfforts: efforts.map((reasoningEffort) => ({
    reasoningEffort,
    description: reasoningEffort,
  })),
});

function serveAssets() {
  for (const [name, path, type] of [
    ["chat.html", "/chat/", "text/html"],
    ["chat.js", "/chat/chat.js", "text/javascript"],
    ["chat.css", "/chat/chat.css", "text/css"],
  ]) {
    cy.readFile("../../src/npa/tools/desktop/" + name).then((body) => {
      cy.intercept("GET", path, { body, headers: { "content-type": type } });
    });
  }
}

function mockChat() {
  const thread = {
    id: "same-thread",
    name: "Shared conversation",
    cwd: "/workspace/project",
    model: "model-a",
    reasoningEffort: "high",
    status: { type: "idle" },
  };
  const chat = { thread, turns: [], events: [], cursor: 0 };
  serveAssets();
  cy.intercept("GET", "/chat/api/state", {
    connected: true,
    instance: "runtime",
    cursor: 0,
    pending: [],
    cwd: thread.cwd,
  });
  cy.intercept("GET", "/chat/api/models", {
    data: [
      model("model-a", ["medium", "high"]),
      model("model-b", ["low", "medium"]),
    ],
  });
  mockThreadRoutes(chat);
  mockEventStream(chat);
  return chat;
}

function mockThreadRoutes(chat) {
  cy.intercept("GET", "/chat/api/threads*", (req) =>
    req.reply({ data: [chat.thread], nextCursor: null }),
  );
  cy.intercept("POST", "/chat/api/resume", (req) =>
    req.reply({ thread: chat.thread }),
  );
  cy.intercept({ method: "GET", pathname: "/chat/api/thread" }, (req) =>
    req.reply({ thread: chat.thread }),
  );
  cy.intercept("GET", "/chat/api/turns?*", (req) =>
    req.reply({ data: chat.turns, nextCursor: null }),
  );
  cy.intercept("POST", "/chat/api/settings", (req) => {
    chat.thread = {
      ...chat.thread,
      model: req.body.model,
      reasoningEffort: req.body.effort,
    };
    req.reply({ model: req.body.model, effort: req.body.effort });
  }).as("settings");
}

function mockEventStream(chat) {
  cy.intercept("GET", "/chat/api/events?*", (req) => {
    const events = chat.events.splice(0);
    chat.cursor += events.length;
    req.reply({
      delay: 100,
      body: {
        connected: true,
        instance: "runtime",
        cursor: chat.cursor,
        events,
        pending: [],
      },
    });
  });
}

function publish(chat, method, params) {
  chat.events.push({ method, params: { threadId: chat.thread.id, ...params } });
}

describe("Mobile Codex conversations", () => {
  let chat;
  beforeEach(() => {
    chat = mockChat();
    cy.viewport(390, 844);
    cy.visit("/chat/#same-thread");
    cy.get("#model").should("be.enabled").and("have.value", "model-a");
  });

  it("updates the same thread with supported model and reasoning settings", () => {
    cy.get("#model").select("model-b");
    cy.wait("@settings").its("request.body").should("deep.equal", {
      id: "same-thread",
      model: "model-b",
      effort: "low",
    });
    cy.get("#effort")
      .should("be.enabled")
      .and("have.value", "low")
      .select("medium");
    cy.wait("@settings").its("request.body").should("deep.equal", {
      id: "same-thread",
      model: "model-b",
      effort: "medium",
    });
    cy.get("#settings-status").should("contain", "Saved to this conversation");
    cy.location("hash").should("equal", "#same-thread");
    cy.document().then((doc) =>
      expect(doc.documentElement.scrollWidth).to.be.at.most(390),
    );
  });

  it("reflects model changes and working state originating in the IDE", () => {
    cy.then(() => {
      publish(chat, "thread/settings/updated", {
        threadSettings: { model: "model-b", effort: "medium" },
      });
      publish(chat, "turn/started", {
        turn: { id: "live-turn", status: "inProgress", items: [] },
      });
    });
    cy.get("#model").should("have.value", "model-b");
    cy.get("#effort").should("have.value", "medium");
    cy.title().should("contain", "Working");
    cy.get("#header-activity").should("be.visible");
    cy.get(".session-spinner").should("exist");
    cy.get("#tab-icon")
      .should("have.attr", "href")
      .and("contain", "image/svg+xml");
    cy.then(() =>
      publish(chat, "turn/completed", {
        turn: { id: "live-turn", status: "completed", items: [] },
      }),
    );
    cy.title().should("not.contain", "Working");
    cy.get("#header-activity").should("not.be.visible");
  });

  it("keeps mobile and IDE messages in the same visible conversation", () => {
    cy.intercept("POST", "/chat/api/send", (req) => {
      const user = {
        id: "mobile-message",
        type: "userMessage",
        content: [{ type: "text", text: req.body.text }],
      };
      chat.turns = [{ id: "shared-turn", status: "inProgress", items: [user] }];
      publish(chat, "turn/started", { turn: chat.turns[0] });
      req.reply({ turn: chat.turns[0] });
    }).as("send");
    cy.get("#prompt").type("Message from mobile <img src=x>");
    cy.get("#send").click();
    cy.wait("@send").its("request.body.id").should("equal", "same-thread");
    cy.get("#messages").should("contain", "Message from mobile <img src=x>");
    cy.get("#messages img").should("not.exist");
    cy.then(() => {
      const item = {
        id: "ide-message",
        type: "userMessage",
        content: [{ type: "text", text: "Follow-up from VS Code" }],
      };
      publish(chat, "item/completed", { turnId: "shared-turn", item });
    });
    cy.get("#messages").should("contain", "Follow-up from VS Code");
    cy.location("hash").should("equal", "#same-thread");
  });
});
