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
  cy.intercept("GET", "/chat/api/modes", {data: [
    {mode: "default", name: "Default"}, {mode: "plan", name: "Plan"},
  ]});
  mockThreadRoutes(chat);
  mockEventStream(chat);
  return chat;
}

function mockThreadRoutes(chat) {
  cy.intercept("GET", "/chat/api/threads*", (req) =>
    req.reply({ data: (new URL(req.url).searchParams.get("archived") === "true") === !!chat.thread.archived
      ? [chat.thread] : [], nextCursor: null }),
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
      ...(req.body.mode ? {mode: req.body.mode} : {}),
      ...(req.body.serviceTier !== undefined ? {serviceTier: req.body.serviceTier} : {}),
    };
    req.reply({ model: req.body.model, effort: req.body.effort });
  }).as("settings");
  mockManagement(chat);
}

function mockManagement(chat) {
  cy.intercept("POST", "/chat/api/rename", req => {
    chat.thread.name = req.body.name;
    publish(chat, "thread/name/updated", {threadName: chat.thread.name});
    req.reply({ok: true});
  }).as("rename");
  for (const action of ["archive", "unarchive"]) cy.intercept("POST", "/chat/api/" + action, req => {
    chat.thread.archived = action === "archive";
    publish(chat, "thread/" + action + "d", {});
    req.reply({ok: true});
  }).as(action);
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

  it("offers Home Screen guidance and an explicit desktop choice on phones", () => {
    cy.get('link[rel="manifest"]').should("have.attr", "href", "./manifest.webmanifest");
    cy.get('link[rel="apple-touch-icon"]').should("have.attr", "href", "./icon-180.png");
    cy.get("#menu").click();
    cy.get("#open-host").should("have.attr", "href", "/desktop.html?desktop=1");
    cy.get("#install-app").click();
    cy.get("#install-help").should("be.visible").and("contain.text", "Add to Home Screen");
    cy.get("#install-help button").click();
    cy.get("#install-help").should("not.be.visible");
  });

  it("renames the same chat from its phone list without interpreting markup", () => {
    cy.get("#menu").click();
    cy.get(".session-actions").click();
    cy.get("#session-name").should("be.enabled").clear().type("New <img src=x> name");
    cy.get("#rename-chat").click();
    cy.wait("@rename").its("request.body").should("deep.equal", {id: "same-thread", name: "New <img src=x> name"});
    cy.get("#title").should("have.text", "New <img src=x> name");
    cy.get(".session-title").should("have.text", "New <img src=x> name");
    cy.get(".session-title img").should("not.exist");
    cy.reload();
    cy.get("#title").should("have.text", "New <img src=x> name");
    cy.location("hash").should("equal", "#same-thread");
  });

  it("archives and restores without losing messages or a draft", () => {
    cy.then(() => { chat.turns = [{id: "saved-turn", status: "completed", items: [
      {id: "saved-message", type: "agentMessage", text: "Preserved history"},
    ]}]; });
    cy.get("#refresh").click();
    cy.get("#prompt").type("Unsent draft");
    cy.get("#manage-chat").click();
    cy.get("#archive-chat").should("be.enabled").click();
    cy.wait("@archive").its("request.body.id").should("equal", "same-thread");
    cy.get("#prompt").should("be.disabled").and("have.value", "Unsent draft");
    cy.get("#messages").should("contain", "Preserved history");
    cy.reload();
    cy.get("#run-state").should("contain", "Archived");
    cy.get("#menu").click();
    cy.get(".session").should("not.exist");
    cy.get("#archive").select("true");
    cy.get(".session-actions").click();
    cy.get("#archive-chat").should("have.text", "Restore").and("be.enabled").click();
    cy.wait("@unarchive").its("request.body.id").should("equal", "same-thread");
    cy.get("#prompt").should("be.enabled").and("have.value", "Unsent draft");
    cy.get("#messages").should("contain", "Preserved history");
    cy.location("hash").should("equal", "#same-thread");
  });

  it("keeps rename available while protecting running chats from archive", () => {
    cy.then(() => { chat.thread.status = {type: "active"}; });
    cy.get("#manage-chat").click();
    cy.get("#session-name").should("be.enabled");
    cy.get("#archive-chat").should("be.disabled");
    cy.get("#session-hint").should("contain", "Codex is working");
    cy.get("#cancel-session").click();
    cy.intercept("POST", "/chat/api/rename", {statusCode: 400, body: {error: "Owner unavailable"}});
    cy.get("#manage-chat").click();
    cy.get("#session-name").should("be.enabled").clear().type("Keep on failure");
    cy.get("#rename-chat").click();
    cy.get("#session-error").should("be.visible").and("contain", "Owner unavailable");
    cy.get("#session-name").should("have.value", "Keep on failure");
    cy.get("#title").should("have.text", "Shared conversation");
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
        threadSettings: { model: "model-b", effort: "medium", serviceTier: "default",
          collaborationMode: {mode: "plan"} },
      });
      publish(chat, "turn/started", {
        turn: { id: "live-turn", status: "inProgress", items: [] },
      });
    });
    cy.get("#model").should("have.value", "model-b");
    cy.get("#effort").should("have.value", "medium");
    cy.get("#speed").should("have.value", "default");
    cy.get("#mode").should("have.value", "plan");
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

  it("keeps drafts after reloading and supports Plan and Standard speed", () => {
    cy.get("#prompt").type("Draft stays in this browser");
    cy.reload();
    cy.get("#prompt").should("have.value", "Draft stays in this browser");
    cy.get("#mode").should("be.enabled").select("plan");
    cy.wait("@settings").its("request.body.mode").should("equal", "plan");
    cy.get("#speed").should("be.enabled").select("default");
    cy.wait("@settings").its("request.body.serviceTier").should("equal", "default");
  });

  it("keeps the final native update when history is still loading", () => {
    let delayed = false;
    cy.intercept("GET", "/chat/api/turns?*", req => {
      const snapshot = JSON.parse(JSON.stringify(chat.turns));
      if (!delayed) {
        delayed = true;
        setTimeout(() => {
          chat.thread.status = {type: "idle"};
          chat.turns[0].status = "completed";
          publish(chat, "native/changed", {});
        }, 50);
        req.reply({delay: 700, body: {data: snapshot, nextCursor: null}});
      } else req.reply({data: snapshot, nextCursor: null});
    }).as("nativeHistory");
    cy.then(() => {
      chat.thread.status = {type: "active"};
      chat.turns = [{id: "native-turn", status: "inProgress", items: []}];
      publish(chat, "native/changed", {});
    });
    cy.wait("@nativeHistory");
    cy.title().should("not.contain", "Working");
    cy.get("#header-activity").should("not.be.visible");
  });

  it("reuses a send identity after an ambiguous response and reload", () => {
    const identifiers = [];
    cy.intercept("POST", "/chat/api/send", req => {
      identifiers.push(req.body.clientUserMessageId);
      if (identifiers.length === 1) req.reply({statusCode: 503, body: {error: "Response lost"}});
      else req.reply({accepted: true});
    }).as("sendRetry");
    cy.get("#prompt").type("Submit this once");
    cy.get("#send").click();
    cy.wait("@sendRetry");
    cy.get("#notice").should("contain", "Response lost");
    cy.reload();
    cy.get("#send").should("be.enabled").click();
    cy.wait("@sendRetry");
    cy.then(() => {
      expect(identifiers).to.have.length(2);
      expect(identifiers[1]).to.equal(identifiers[0]);
    });
    cy.get("#clear-send").should("not.be.visible");
  });
});
