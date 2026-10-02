// Provide browser fixtures for shared Codex controls and live events.
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

/**
 * Serve isolated chat state through the real browser UI.
 * Args: None.
 * Returns: Mutable test conversation and event state.
 * Raises: Cypress assertion errors when fixture assets cannot be read.
 */
export function mockChat() {
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
  cy.intercept("GET", "/chat/api/workspaces", {data: [
    {path: "/workspace/nebius-physical-ai", name: "nebius-physical-ai", preferred: true, lastUsed: 10},
    {path: thread.cwd, name: "project", preferred: false, lastUsed: 20},
  ]}).as("workspaces");
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

/**
 * Queue one IDE event for the browser's next poll.
 * Args: chat — fixture state; method — event name; params — event fields.
 * Returns: None.
 * Raises: None.
 */
export function publish(chat, method, params) {
  chat.events.push({ method, params: { threadId: chat.thread.id, ...params } });
}
