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
  images: [],
  modes: [],
  hostLabel: "VDI",
  sessionGeneration: 0,
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
  const response = await fetch(new URL("./api/" + path, location.href), options);
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
function sessionButton(thread) {
  const button = node("button", "session" + (thread.id === state.id ? " selected" : ""));
  button.setAttribute("aria-current", String(thread.id === state.id));
  button.append(node("span", "session-title", title(thread)));
  const meta = node("span", "session-meta");
  if (state.statuses.get(thread.id)?.type === "active") {
    const spinner = node("span", "spinner session-spinner");
    spinner.setAttribute("aria-label", "Codex is working");
    meta.append(spinner);
  }
  meta.append(node("span", "workspace", workspace(thread.cwd)),
    node("time", "", relativeTime(thread.updatedAt)));
  button.append(meta);
  button.addEventListener("click", () =>
    selectThread(thread.id).catch((error) => notice(error.message)));
  return button;
}
function renderSessions() {
  const root = $("#sessions");
  root.replaceChildren();
  if (!state.sessions.length)
    root.append(node("p", "loading muted", "No sessions found."));
  for (const thread of state.sessions) {
    const row = node("div", "session-row");
    const actions = node("button", "session-actions", "⋯");
    actions.setAttribute("aria-label", "Manage chat: " + title(thread));
    actions.setAttribute("aria-haspopup", "dialog");
    actions.addEventListener("click", () => manageChat(thread));
    row.append(sessionButton(thread), actions);
    root.append(row);
  }
  $("#more-sessions").hidden = !state.sessionCursor;
}
async function loadSessions(append = false) {
  const generation = ++state.sessionGeneration;
  const query = new URLSearchParams({
    search: $("#search").value,
    archived: $("#archive").value,
  });
  if (append && state.sessionCursor) query.set("cursor", state.sessionCursor);
  const result = await api("threads?" + query);
  if (generation !== state.sessionGeneration) return;
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
  const readOnly = state.externalOwner || state.thread?.archived;
  $("#manage-chat").disabled = !state.thread || state.loading;
  $("#prompt").disabled = !state.id || state.loading || readOnly;
  $("#send").disabled =
    !state.id ||
    state.loading ||
    state.sending ||
    state.changingModel ||
    readOnly ||
    (!$("#prompt").value.trim() && !state.images.length) ||
    !state.connected;
  $("#stop").hidden = !active;
  $("#run-state").textContent = state.thread?.archived ? "Archived · restore to continue" : active
    ? "Working · send to steer"
    : state.id
      ? "Ready"
      : "Choose a session to begin";
  $("#send").setAttribute(
    "aria-label",
    active ? "Steer current turn" : "Send message",
  );
  updateSettingControls(readOnly);
  updateActivity();
}
function updateSettingControls(readOnly) {
  for (const id of ["#model", "#effort"])
    $(id).disabled = !state.id || state.loading || readOnly || state.changingModel ||
      !state.connected || !state.models.length;
  $("#speed").disabled = $("#model").disabled;
  $("#mode").disabled = $("#model").disabled || !state.modes.length;
}
async function selectThread(id) {
  saveDraft();
  const generation = ++state.generation;
  notice();
  state.id = id;
  $("#prompt").value = localStorage.getItem("codex-draft:" + id) || "";
  state.images = [];
  const pendingSend = JSON.parse(localStorage.getItem("codex-send:" + id) || "null");
  if (pendingSend) state.images = pendingSend.images || [];
  renderAttachments();
  $("#delivery-status").hidden = true;
  $("#clear-send").hidden = !localStorage.getItem("codex-send:" + id);
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
  if ("serviceTier" in resumed) state.thread.serviceTier = resumed.serviceTier;
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
  renderExtraControls();
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
  state.thread = {...state.thread, ...result.thread};
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
    if ("serviceTier" in params.threadSettings)
      state.thread.serviceTier = params.threadSettings.serviceTier;
    if (params.threadSettings.collaborationMode)
      state.thread.mode = params.threadSettings.collaborationMode.mode;
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
  const firstLoad = !state.turns.length;
  const query = new URLSearchParams({ id: state.id });
  if (older && state.turnCursor) query.set("cursor", state.turnCursor);
  const result = await api("turns?" + query);
  if (generation !== state.generation) return;
  const turns = result.data || result.turns || [];
  state.turnCursor = result.nextCursor;
  state.turns = older ? [...state.turns, ...turns] : turns;
  const unique = new Map(state.turns.map((turn) => [turn.id, turn]));
  state.turns = [...unique.values()];
  renderMessages(!older && firstLoad);
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
    for (const entry of item.content || []) {
      if (entry.type !== "image" || !/^data:image\/(png|jpeg|webp);base64,/.test(entry.url || "")) continue;
      const image = node("img", "message-image");
      image.src = entry.url;
      image.alt = "Attached image";
      article.append(image);
    }
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
function sessionMetadataEvent(method, p) {
  if (["thread/started", "thread/name/updated", "thread/archived", "thread/unarchived"].includes(method))
    loadSessions().catch(() => {});
  if (p.threadId !== state.id) return;
  if (method === "thread/name/updated") refreshThread().catch(() => {});
  if (method === "thread/archived" || method === "thread/unarchived") {
    if (state.thread) state.thread.archived = method === "thread/archived";
    updateControls();
  }
}
function handleEvent(event) {
  if (event.method === "native/changed") {
    refreshNative();
    return;
  }
  sessionEvent(event);
  const p = event.params || {},
    method = event.method;
  sessionMetadataEvent(method, p);
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
        ? "● Connected to " + state.hostLabel
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
  saveDraft();
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
  if ($("#send").disabled || state.sending || (!text.trim() && !state.images.length)) return;
  state.sending = true;
  updateControls();
  notice();
  try {
    const sent = await sendMessage(text);
    if (generation !== state.generation) return;
    if (state.thread && !state.thread.name && !state.thread.preview) {
      state.thread.preview = text;
      $("#title").textContent = title(state.thread);
    }
    if ($("#prompt").value === text) $("#prompt").value = "";
    state.images = state.images.filter(image => !sent.images.includes(image));
    renderAttachments();
    saveDraft();
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
    state.hostLabel = info.hostLabel || "VDI";
    state.native = !!info.native;
    configureHost();
    loadModels().catch((error) => notice("Model controls: " + error.message));
    api("modes").then(result => { state.modes = result.data; renderExtraControls(); })
      .catch(() => { state.modes = []; });
    $("#cwd").value = info.cwd;
    $("#connection").textContent = "● Connected to " + state.hostLabel;
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

function configureHost() {
  $("#host-tag").textContent = state.native ? "Mac" : "VDI";
  $("#settings-status").textContent = "Same conversations. Work stays on " + state.hostLabel + ".";
  if (!state.native) return;
  $("#open-host").textContent = "Open in VS Code ↗";
  $("#open-host").removeAttribute("href");
  $("#open-host").onclick = async () => {
    if (!state.id) return;
    try {
      const result = await api("open", {id: state.id});
      notice(result.connected ? "Open in VS Code on your Mac." : "Open request sent. Accept the VS Code link on your Mac if prompted.");
    } catch (error) { notice(error.message); }
  };
}

function saveDraft() {
  if (!state.id) return;
  try { localStorage.setItem("codex-draft:" + state.id, $("#prompt").value); }
  catch { notice("Browser storage is unavailable; keep this page open to retain your draft."); }
}

function options(select, choices, value) {
  select.replaceChildren();
  for (const [key, label] of choices) {
    const option = node("option", "", label);
    option.value = key;
    select.append(option);
  }
  select.value = value || "";
}

function renderExtraControls() {
  const selected = state.models.find(model => model.model === state.thread?.model);
  const speeds = [["", "Account default"], ["default", "Standard"],
    ...(selected?.serviceTiers || []).map(tier => [tier.id, tier.id === "priority" ? "Fast · higher usage" : tier.name || tier.id])];
  const currentMode = state.thread?.mode || state.thread?.collaborationMode?.mode;
  const speed = state.thread?.serviceTier;
  if (speed === undefined) speeds.unshift(["unknown", "Choose speed"]);
  options($("#speed"), speeds, speed === undefined ? "unknown" : speed);
  const modes = state.modes.map(mode => [mode.mode, mode.name || mode.mode]);
  if (!currentMode) modes.unshift(["unknown", "Choose mode"]);
  options($("#mode"), modes, currentMode || "unknown");
  for (const select of [$("#speed"), $("#mode")]) {
    const unknown = select.querySelector('option[value="unknown"]');
    if (unknown) unknown.disabled = true;
  }
}

async function changeExtraSettings(field, value) {
  if (!state.id || state.changingModel) return;
  const generation = state.generation;
  state.changingModel = true;
  updateControls();
  try {
    await api("settings", {id: state.id, model: state.thread.model,
      effort: state.thread.reasoningEffort, [field]: value});
    if (generation === state.generation) {
      await refreshThread();
      state.thread[field] = value;
      renderExtraControls();
    }
    $("#settings-status").textContent = "Saved to this conversation · applies to the next turn";
  } catch (error) { notice(error.message); }
  finally { state.changingModel = false; updateControls(); }
}

$("#speed").onchange = () => changeExtraSettings("serviceTier", $("#speed").value || null);
$("#mode").onchange = () => changeExtraSettings("mode", $("#mode").value);
$("#attach").onclick = () => $("#image-input").click();
$("#image-input").onchange = async () => {
  const id = state.id;
  for (const file of $("#image-input").files) {
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      notice("Attach a PNG, JPEG, or WebP image.");
      continue;
    }
    const url = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    if (state.id === id) state.images.push(url);
  }
  $("#image-input").value = "";
  renderAttachments();
  updateControls();
};

function renderAttachments() {
  $("#attachments").replaceChildren();
  state.images.forEach((url, index) => {
    const button = node("button", "attachment", "×");
    button.type = "button";
    button.setAttribute("aria-label", "Remove attached image");
    const image = node("img");
    image.src = url;
    image.alt = "Image to send";
    button.prepend(image);
    button.onclick = () => { state.images.splice(index, 1); renderAttachments(); updateControls(); };
    $("#attachments").append(button);
  });
}

async function sendMessage(text) {
  const key = "codex-send:" + state.id;
  const previous = JSON.parse(localStorage.getItem(key) || "null");
  if (previous && (previous.text !== text || JSON.stringify(previous.images) !== JSON.stringify(state.images)))
    throw new Error("Check the previous send, then discard its pending state before sending different content.");
  const body = previous || {id: state.id, text, images: [...state.images],
    turnId: activeTurn()?.id, clientUserMessageId: crypto.randomUUID()};
  localStorage.setItem(key, JSON.stringify(body));
  $("#clear-send").hidden = false;
  const result = await api("send", body);
  localStorage.removeItem(key);
  if (state.id === body.id) {
    $("#clear-send").hidden = true;
    $("#delivery-status").hidden = false;
    $("#delivery-status").textContent = result.delivery?.target === "VS Code" ?
      (result.delivery.visible ? "✓ Message visible in VS Code on your Mac" : "Sent · waiting for VS Code to display the message") : "✓ Sent to this conversation";
  }
  return body;
}

$("#clear-send").onclick = () => {
  localStorage.removeItem("codex-send:" + state.id);
  $("#clear-send").hidden = true;
  notice("Pending send cleared. Check the conversation before sending again.");
};

let nativeRefreshing = false;
let nativeRefreshQueued = false;
async function refreshNative() {
  nativeRefreshQueued = true;
  if (nativeRefreshing) return;
  nativeRefreshing = true;
  try {
    while (nativeRefreshQueued) {
      nativeRefreshQueued = false;
      await Promise.all([loadSessions(), refreshThread(), loadTurns()]);
    }
  }
  catch (error) { notice(error.message); }
  finally { nativeRefreshing = false; }
}

let managedChat;
let managementGeneration = 0;
let managingChat = false;
function managementError(message = "") {
  $("#session-error").textContent = message;
  $("#session-error").hidden = !message;
}
function managementControls(busy) {
  const archived = managedChat?.archived;
  $("#session-name").disabled = busy || archived;
  $("#rename-chat").disabled = busy || archived;
  $("#archive-chat").disabled = busy || managedChat?.status?.type === "active";
  $("#archive-chat").textContent = archived ? "Restore" : "Archive";
  $("#cancel-session").disabled = busy;
  $("#session-hint").textContent = archived
    ? "Restore this chat to continue or rename it. Messages are preserved."
    : managedChat?.status?.type === "active"
      ? "Codex is working. You can rename now; archive after the turn finishes."
      : "Renaming and archiving preserve this conversation and its messages.";
}
async function manageChat(thread) {
  const generation = ++managementGeneration;
  managedChat = thread;
  managementError();
  $("#session-name").value = title(thread);
  managementControls(true);
  $("#session-dialog").showModal();
  try {
    const result = await api("thread?id=" + encodeURIComponent(thread.id));
    if (!$("#session-dialog").open || generation !== managementGeneration) return;
    managedChat = result.thread;
    $("#session-name").value = title(managedChat);
    managementControls(false);
    if (!managedChat.archived) $("#session-name").select();
    else $("#archive-chat").focus();
  } catch (error) {
    managementError(error.message);
    $("#cancel-session").disabled = false;
  }
}
async function manageAction(action) {
  if (!managedChat || managingChat) return;
  const id = managedChat.id;
  const generation = state.generation;
  managingChat = true;
  managementControls(true);
  managementError();
  try {
    await api(action, {id, ...(action === "rename" ? {name: $("#session-name").value} : {})});
    $("#session-dialog").close();
    if (id === state.id && generation === state.generation) {
      if (action === "unarchive") await selectThread(id);
      else {
        if (action === "archive") state.thread.archived = true;
        else await refreshThread();
        updateControls();
      }
    }
    await loadSessions();
  } catch (error) {
    if ($("#session-dialog").open) managementError(error.message);
    else notice(error.message);
  } finally {
    managingChat = false;
    managementControls(false);
  }
}
$("#manage-chat").addEventListener("click", () => manageChat(state.thread));
$("#cancel-session").addEventListener("click", () => $("#session-dialog").close());
$("#session-dialog").addEventListener("cancel", event => {
  if (managingChat) event.preventDefault();
});
$("#session-form").addEventListener("submit", event => {
  event.preventDefault();
  manageAction("rename");
});
$("#archive-chat").addEventListener("click", () =>
  manageAction(managedChat.archived ? "unarchive" : "archive"));
