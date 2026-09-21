/* Share durable Codex threads and live controls across phone and desktop browsers. */
"use strict";
const $ = (selector) => document.querySelector(selector);
const state = {
  id: null,
  thread: null,
  turns: [],
  sessions: [],
  sessionCursor: null,
  turnCursor: null,
  cursor: 0,
  pending: [],
  generation: 0,
  connected: false,
  sending: false,
  changingModel: false,
  models: [],
  statuses: new Map(),
};
const node = (tag, className, text) => {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
};
function notice(text = "") {
  $("#notice").textContent = text;
  $("#notice").hidden = !text;
}
async function api(path, body) {
  const options = { credentials: "same-origin", cache: "no-store" };
  if (body !== undefined) {
    options.method = "POST";
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch("/chat/api/" + path, options);
  if (response.status === 401)
    throw new Error("Sign-in expired. Reload this page to sign in again.");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed.");
  return data;
}
function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}
function title(thread) {
  return thread.name || thread.preview || "Untitled session";
}
function workspace(path) {
  return (path || "").replace(/\/$/, "").split("/").pop() || "VDI";
}
function relativeTime(timestamp) {
  const seconds = Math.max(0, Date.now() / 1000 - timestamp);
  if (seconds < 60) return "now";
  if (seconds < 3600) return Math.floor(seconds / 60) + "m";
  if (seconds < 86400) return Math.floor(seconds / 3600) + "h";
  return new Date(timestamp * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}
function renderSessions() {
  const root = $("#sessions");
  root.replaceChildren();
  if (!state.sessions.length)
    root.append(node("p", "loading muted", "No sessions found."));
  for (const thread of state.sessions) {
    const button = node(
      "button",
      "session" + (thread.id === state.id ? " selected" : ""),
    );
    button.setAttribute("aria-current", String(thread.id === state.id));
    button.append(node("span", "session-title", title(thread)));
    const meta = node("span", "session-meta");
    if (state.statuses.get(thread.id)?.type === "active") {
      const spinner = node("span", "spinner session-spinner");
      spinner.setAttribute("aria-label", "Codex is working");
      meta.append(spinner);
    }
    meta.append(
      node("span", "workspace", workspace(thread.cwd)),
      node("time", "", relativeTime(thread.updatedAt)),
    );
    button.append(meta);
    button.addEventListener("click", () =>
      selectThread(thread.id).catch((error) => notice(error.message)),
    );
    root.append(button);
  }
  $("#more-sessions").hidden = !state.sessionCursor;
}
async function loadSessions(append = false) {
  const query = new URLSearchParams({
    search: $("#search").value,
    archived: $("#archive").value,
  });
  if (append && state.sessionCursor) query.set("cursor", state.sessionCursor);
  const result = await api("threads?" + query);
  state.sessionCursor = result.nextCursor;
  state.sessions = append ? [...state.sessions, ...result.data] : result.data;
  for (const thread of result.data)
    state.statuses.set(thread.id, thread.status);
  renderSessions();
  updateActivity();
}
function activeTurn() {
  return state.turns.find((turn) => turn.status === "inProgress");
}
function updateControls() {
  const active = activeTurn();
  $("#prompt").disabled = !state.id || state.loading || state.externalOwner;
  $("#send").disabled =
    !state.id ||
    state.loading ||
    state.sending ||
    state.changingModel ||
    state.externalOwner ||
    !$("#prompt").value.trim() ||
    !state.connected;
  $("#stop").hidden = !active;
  $("#run-state").textContent = active
    ? "Working · send to steer"
    : state.id
      ? "Ready"
      : "Choose a session to begin";
  $("#send").setAttribute(
    "aria-label",
    active ? "Steer current turn" : "Send message",
  );
  for (const id of ["#model", "#effort"])
    $(id).disabled =
      !state.id ||
      state.loading ||
      state.externalOwner ||
      state.changingModel ||
      !state.connected ||
      !state.models.length;
  updateActivity();
}
async function selectThread(id) {
  const generation = ++state.generation;
  notice();
  state.id = id;
  state.loading = true;
  state.turns = [];
  state.thread = null;
  $("#requests").replaceChildren();
  closeSidebar();
  renderSessions();
  updateControls();
  history.replaceState(null, "", "#" + encodeURIComponent(id));
  $("#welcome").hidden = true;
  $("#messages").replaceChildren(node("p", "muted", "Opening session…"));
  $("#title").textContent = "Opening session…";
  const resumed = await api("resume", { id });
  if (generation !== state.generation) return;
  state.externalOwner = !!resumed.externalOwner;
  if (state.externalOwner)
    notice(
      "This session is open in an older Codex client. You can read it here. Close it in that client, then reopen it here to continue safely.",
    );
  state.thread = resumed.thread;
  renderModelControls();
  $("#title").textContent = title(state.thread);
  $("#project").textContent = state.thread.cwd || "VDI";
  await loadTurns(false, generation);
  if (generation !== state.generation) return;
  state.loading = false;
  renderRequests();
  updateControls();
}

function renderModelControls() {
  const current = state.thread?.model;
  const model = state.models.find((model) => model.model === current);
  const select = $("#model");
  select.replaceChildren();
  for (const option of state.models) {
    const element = node("option", "", option.displayName);
    element.value = option.model;
    select.append(element);
  }
  if (current && !model) {
    const unavailable = node("option", "", current + " (current)");
    unavailable.value = current;
    select.append(unavailable);
  }
  select.value =
    current || state.models.find((model) => model.isDefault)?.model || "";
  const selected =
    model || state.models.find((model) => model.model === select.value);
  renderReasoning(selected);
}

function renderReasoning(selected) {
  const effort = $("#effort");
  effort.replaceChildren();
  for (const option of selected?.supportedReasoningEfforts || []) {
    const element = node("option", "", effortLabel(option.reasoningEffort));
    element.value = option.reasoningEffort;
    element.title = option.description;
    effort.append(element);
  }
  effort.value =
    state.thread?.reasoningEffort || selected?.defaultReasoningEffort || "";
  $("#model-description").textContent =
    selected?.supportedReasoningEfforts.find(
      (option) => option.reasoningEffort === effort.value,
    )?.description || "";
}

function effortLabel(value) {
  return (
    { xhigh: "Extra high", max: "Maximum", ultra: "Ultra" }[value] ||
    value.charAt(0).toUpperCase() + value.slice(1)
  );
}

async function changeModel(model, effort) {
  if (!state.id || state.changingModel) return;
  const generation = state.generation;
  state.changingModel = true;
  updateControls();
  notice();
  $("#settings-status").textContent = "Updating shared Codex settings…";
  try {
    await api("settings", { id: state.id, model, effort });
    if (generation !== state.generation) return;
    await refreshThread();
    $("#settings-status").textContent =
      "Saved to this conversation · applies to the next turn";
  } catch (error) {
    notice(error.message);
    $("#settings-status").textContent =
      "Could not confirm settings. Refresh before retrying.";
  } finally {
    state.changingModel = false;
    renderModelControls();
    updateControls();
  }
}

async function refreshThread() {
  if (!state.id) return;
  const generation = state.generation;
  const result = await api("thread?id=" + encodeURIComponent(state.id));
  if (generation !== state.generation) return;
  state.thread = result.thread;
  state.statuses.set(state.id, result.thread.status);
  $("#title").textContent = title(state.thread);
  renderModelControls();
  updateControls();
}

async function loadModels() {
  state.models = (await api("models")).data;
  renderModelControls();
  updateControls();
}

let activityFrame = 0;
function updateActivity() {
  const selectedBusy =
    !!activeTurn() || state.statuses.get(state.id)?.type === "active";
  const working =
    selectedBusy ||
    [...state.statuses.values()].some((status) => status?.type === "active");
  const waiting = state.pending.some(
    (request) => request.params?.threadId === state.id,
  );
  const symbol = ["◐", "◓", "◑", "◒"][activityFrame % 4];
  const label = state.thread
    ? title(state.thread).slice(0, 70)
    : "Your desktop";
  const tabTitle =
    (!state.connected
      ? "Reconnecting · "
      : working
        ? waiting
          ? "! Needs input · "
          : symbol + " Working · "
        : "") +
    label +
    " · Codex";
  if (document.title !== tabTitle) document.title = tabTitle;
  $("#header-activity").hidden = !selectedBusy;
  renderTabIcon(working);
}

function renderTabIcon(working) {
  const angle = working ? activityFrame * 90 : 0;
  const svg =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#171717"/><circle cx="16" cy="16" r="10" fill="none" stroke="' +
    (working ? "#9edbbd" : "#ececec") +
    '" stroke-width="4" stroke-dasharray="' +
    (working ? "44 19" : "63 0") +
    '" transform="rotate(' +
    angle +
    ' 16 16)"/></svg>';
  const icon = "data:image/svg+xml," + encodeURIComponent(svg);
  if ($("#tab-icon").getAttribute("href") !== icon) $("#tab-icon").href = icon;
}

function sessionEvent(event) {
  const params = event.params || {};
  const id = params.threadId || params.thread?.id;
  if (!id) return;
  if (event.method === "thread/status/changed")
    state.statuses.set(id, params.status);
  if (event.method === "turn/started")
    state.statuses.set(id, { type: "active" });
  if (event.method === "turn/completed")
    state.statuses.set(id, { type: "idle" });
  if (event.method === "thread/closed")
    state.statuses.set(id, { type: "notLoaded" });
  if (
    event.method === "thread/settings/updated" &&
    id === state.id &&
    state.thread
  ) {
    state.thread.model = params.threadSettings.model;
    state.thread.reasoningEffort = params.threadSettings.effort;
    renderModelControls();
  }
  if (
    ["thread/status/changed", "turn/started", "turn/completed"].includes(
      event.method,
    )
  )
    renderSessions();
  updateActivity();
}
async function loadTurns(older = false, generation = state.generation) {
  if (!state.id) return;
  const query = new URLSearchParams({ id: state.id });
  if (older && state.turnCursor) query.set("cursor", state.turnCursor);
  const result = await api("turns?" + query);
  if (generation !== state.generation) return;
  const turns = result.data || result.turns || [];
  state.turnCursor = result.nextCursor;
  state.turns = older ? [...state.turns, ...turns] : turns;
  const unique = new Map(state.turns.map((turn) => [turn.id, turn]));
  state.turns = [...unique.values()];
  renderMessages(!older);
  $("#older").hidden = !state.turnCursor;
  updateControls();
}
function appendText(root, text) {
  const parts = (text || "").split(/```[^\n]*\n/);
  if (parts.length === 1) {
    root.append(node("p", "", text));
    return;
  }
  root.append(node("p", "", parts.shift()));
  for (const part of parts) {
    const end = part.indexOf("```");
    root.append(node("pre", "", end < 0 ? part : part.slice(0, end)));
    if (end >= 0 && part.slice(end + 3).trim())
      root.append(node("p", "", part.slice(end + 3).trim()));
  }
}
function textFor(item) {
  if (item.type === "userMessage")
    return (item.content || [])
      .map((entry) => entry.text || "[Attachment: " + entry.type + "]")
      .join("\n");
  if (item.type === "agentMessage") return item.text || "";
  return null;
}
function renderItem(item) {
  const text = textFor(item);
  if (text !== null) {
    const article = node(
      "article",
      "message " + (item.type === "userMessage" ? "user" : "assistant"),
    );
    article.dataset.itemId = item.id;
    if (item.type !== "userMessage")
      article.append(node("div", "role", "CODEX"));
    appendText(article, text);
    if (item.type === "agentMessage" && text) {
      const copy = node("button", "copy", "Copy");
      copy.addEventListener("click", () =>
        navigator.clipboard
          .writeText(text)
          .then(() => {
            copy.textContent = "Copied";
          })
          .catch(() => notice("Select the text to copy it.")),
      );
      article.append(copy);
    }
    return article;
  }
  if (item.type === "reasoning") return null;
  return renderTool(item);
}
function renderTool(item) {
  const details = node("details", "tool");
  details.dataset.itemId = item.id;
  const labels = {
    commandExecution: "Terminal",
    fileChange: "File changes",
    mcpToolCall: "Tool",
    webSearch: "Web search",
    plan: "Plan",
    collabAgentToolCall: "Agent",
  };
  details.append(
    node(
      "summary",
      "",
      (labels[item.type] || item.type) +
        (item.command ? " · " + item.command.slice(0, 110) : "") +
        (item.status ? " · " + item.status : ""),
    ),
  );
  const output =
    item.aggregatedOutput ||
    item.text ||
    item.query ||
    JSON.stringify(item.changes || item.result || item, null, 2);
  details.append(node("pre", "", output));
  return details;
}
let renderQueued = false;
function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderMessages();
    updateControls();
  });
}
function renderMessages(forceBottom = false) {
  const viewport = $("#conversation"),
    nearBottom =
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 140;
  const open = new Set(
    [...$("#messages").querySelectorAll("details[open]")].map(
      (el) => el.dataset.itemId,
    ),
  );
  const fragment = document.createDocumentFragment();
  for (const turn of [...state.turns].reverse()) {
    for (const item of turn.items || []) {
      const el = renderItem(item);
      if (el) {
        if (open.has(item.id)) el.open = true;
        fragment.append(el);
      }
    }
    if (turn.status === "inProgress")
      fragment.append(node("div", "working", "● Codex is working…"));
    if (turn.error)
      fragment.append(
        node("div", "working", turn.error.message || "This turn failed."),
      );
  }
  if (!state.turns.length)
    fragment.append(
      node("p", "muted", "Send a message to start this conversation."),
    );
  $("#messages").replaceChildren(fragment);
  if (nearBottom || forceBottom) viewport.scrollTop = viewport.scrollHeight;
}
function eventTurn(params) {
  let turn = state.turns.find((t) => t.id === params.turnId);
  if (!turn) {
    turn = { id: params.turnId, status: "inProgress", items: [] };
    state.turns.unshift(turn);
  }
  return turn;
}
function handleEvent(event) {
  sessionEvent(event);
  const p = event.params || {},
    method = event.method;
  if (method === "thread/started" || method === "thread/name/updated")
    loadSessions().catch(() => {});
  if (p.threadId !== state.id) return;
  if (method === "turn/started" || method === "turn/completed") {
    const old = state.turns.find((t) => t.id === p.turn.id),
      items = p.turn.items?.length ? p.turn.items : old?.items || [];
    state.turns = state.turns.filter((t) => t.id !== p.turn.id);
    state.turns.unshift({ ...p.turn, items });
    if (method === "turn/completed")
      loadTurns().catch((error) => notice(error.message));
    scheduleRender();
    return;
  }
  if (method === "item/started" || method === "item/completed") {
    const turn = eventTurn(p),
      index = turn.items.findIndex((i) => i.id === p.item.id);
    if (index < 0) turn.items.push(p.item);
    else turn.items[index] = p.item;
    scheduleRender();
    return;
  }
  if (
    method === "item/agentMessage/delta" ||
    method === "item/commandExecution/outputDelta"
  )
    renderDelta(method, p);
}
function renderDelta(method, p) {
  const turn = eventTurn(p);
  let item = turn.items.find((i) => i.id === p.itemId);
  if (!item) {
    item = {
      id: p.itemId,
      type: method.includes("agentMessage")
        ? "agentMessage"
        : "commandExecution",
      text: "",
    };
    turn.items.push(item);
  }
  const field = method.includes("agentMessage") ? "text" : "aggregatedOutput";
  item[field] = (item[field] || "") + p.delta;
  scheduleRender();
}
function renderRequests() {
  const root = $("#requests");
  root.replaceChildren();
  for (const request of state.pending.filter(
    (r) => r.params?.threadId === state.id,
  )) {
    const card = node("div", "request");
    card.append(node("h3", "", "Codex needs your input"));
    const answer = async (body) => {
      try {
        await api("answer", { requestId: request.id, ...body });
        state.pending = state.pending.filter((r) => r.id !== request.id);
        renderRequests();
      } catch (error) {
        notice(error.message);
      }
    };
    renderDecision(card, request, answer);
    root.append(card);
  }
}
function renderDecision(card, request, answer) {
  const params = request.params;
  if (request.method === "item/tool/requestUserInput")
    return renderQuestions(card, params.questions, answer);
  if (!request.method.endsWith("/requestApproval")) {
    card.append(
      node(
        "p",
        "",
        "This request needs VS Code. Open the desktop to continue.",
      ),
    );
    return;
  }
  card.append(
    node(
      "pre",
      "",
      params.reason ||
        params.command ||
        JSON.stringify(params.permissions || {}, null, 2),
    ),
  );
  for (const [decision, label] of [
    ["accept", "Allow once"],
    ["decline", "Decline"],
  ]) {
    const button = node(
      "button",
      decision === "accept" ? "primary" : "",
      label,
    );
    button.addEventListener("click", () => answer({ decision }));
    card.append(button);
  }
}
function renderQuestions(card, questions, answer) {
  const inputs = {};
  for (const question of questions) {
    const label = node(
      "label",
      "",
      question.question || question.header || "Your answer",
    );
    const input = node("input");
    input.type = question.isSecret ? "password" : "text";
    input.placeholder =
      (question.options || []).map((option) => option.label).join(" / ") ||
      "Type your answer";
    label.append(input);
    inputs[question.id] = input;
    card.append(label);
  }
  const button = node("button", "primary", "Submit answers");
  button.addEventListener("click", () =>
    answer({
      answers: Object.fromEntries(
        Object.entries(inputs).map(([id, input]) => [id, input.value]),
      ),
    }),
  );
  card.append(button);
}
async function events() {
  while (true) {
    try {
      const result = await api("events?after=" + state.cursor);
      state.connected = result.connected;
      $("#connection").textContent = result.connected
        ? "● Connected to VDI"
        : "Reconnecting…";
      if (!result.connected)
        throw new Error(
          "Codex disconnected. Reload after the service reconnects.",
        );
      state.cursor = result.cursor;
      if (state.instance !== result.instance) {
        state.instance = result.instance;
        if (state.id) await selectThread(state.id);
      }
      for (const event of result.events) handleEvent(event);
      const pendingChanged =
        JSON.stringify(state.pending) !== JSON.stringify(result.pending);
      state.pending = result.pending;
      if (pendingChanged) renderRequests();
      if (result.reset) await loadTurns();
      updateControls();
    } catch (error) {
      state.connected = false;
      $("#connection").textContent = "Reconnecting…";
      updateControls();
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
}
$("#menu").addEventListener("click", () =>
  document.body.classList.toggle("sidebar-open"),
);
$("#scrim").addEventListener("click", closeSidebar);
$("#prompt").addEventListener("input", () => {
  $("#prompt").style.height = "auto";
  $("#prompt").style.height = Math.min($("#prompt").scrollHeight, 180) + "px";
  updateControls();
});
$("#prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});
$("#composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("#prompt").value;
  const generation = state.generation;
  if ($("#send").disabled || state.sending || !text.trim()) return;
  state.sending = true;
  updateControls();
  notice();
  try {
    const active = activeTurn();
    await api("send", { id: state.id, text, turnId: active?.id });
    if (generation !== state.generation) return;
    if (state.thread && !state.thread.name && !state.thread.preview) {
      state.thread.preview = text;
      $("#title").textContent = title(state.thread);
    }
    if ($("#prompt").value === text) $("#prompt").value = "";
    $("#prompt").style.height = "auto";
    await loadTurns();
  } catch (error) {
    notice(error.message);
  } finally {
    state.sending = false;
    updateControls();
  }
});
$("#stop").addEventListener("click", async () => {
  const turn = activeTurn();
  if (!turn) return;
  try {
    await api("stop", { id: state.id, turnId: turn.id });
  } catch (error) {
    notice(error.message);
  }
});
$("#refresh").addEventListener("click", () =>
  Promise.all([loadSessions(), loadTurns(), refreshThread()]).catch((error) =>
    notice(error.message),
  ),
);
$("#older").addEventListener("click", () =>
  loadTurns(true).catch((error) => notice(error.message)),
);
$("#more-sessions").addEventListener("click", () =>
  loadSessions(true).catch((error) => notice(error.message)),
);
let searchTimer;
$("#search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(
    () => loadSessions().catch((error) => notice(error.message)),
    250,
  );
});
$("#archive").addEventListener("change", () =>
  loadSessions().catch((error) => notice(error.message)),
);
for (const id of ["#new-chat", "#welcome-new"])
  $(id).addEventListener("click", () => {
    $("#new-dialog").showModal();
  });
$("#cancel-new").addEventListener("click", () => $("#new-dialog").close());
$("#new-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const result = await api("new", { cwd: $("#cwd").value });
    $("#new-dialog").close();
    await loadSessions();
    await selectThread(result.thread.id);
  } catch (error) {
    notice(error.message);
    $("#new-dialog").close();
  }
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    loadSessions().catch(() => {});
    loadTurns().catch(() => {});
    refreshThread().catch(() => {});
  }
});
if (window.visualViewport) {
  const resize = () => {
    document.body.style.height = window.visualViewport.height + "px";
  };
  window.visualViewport.addEventListener("resize", resize);
  resize();
}
async function start() {
  try {
    const info = await api("state");
    state.cursor = info.cursor;
    state.instance = info.instance;
    state.pending = info.pending;
    state.connected = info.connected;
    loadModels().catch((error) => notice("Model controls: " + error.message));
    $("#cwd").value = info.cwd;
    $("#connection").textContent = "● Connected to VDI";
    events();
    await loadSessions();
    const id = decodeURIComponent(location.hash.slice(1));
    if (id) await selectThread(id);
    else if (innerWidth <= 760) document.body.classList.add("sidebar-open");
    setInterval(() => {
      if (!document.hidden) loadSessions().catch(() => {});
    }, 15000);
  } catch (error) {
    notice(error.message);
  }
}
start();
$("#model").addEventListener("change", () => {
  const selected = state.models.find(
    (model) => model.model === $("#model").value,
  );
  if (!selected) return;
  const current = $("#effort").value;
  const effort = selected.supportedReasoningEfforts.some(
    (option) => option.reasoningEffort === current,
  )
    ? current
    : selected.defaultReasoningEffort;
  changeModel(selected.model, effort);
});
$("#effort").addEventListener("change", () =>
  changeModel($("#model").value, $("#effort").value),
);
setInterval(() => {
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches)
    activityFrame++;
  updateActivity();
}, 750);
