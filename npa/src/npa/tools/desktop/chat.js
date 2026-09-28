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
  metadataRevision: 0,
  connected: false,
  sending: new Set(),
  changingModel: false,
  models: [],
  statuses: new Map(),
  images: [],
  modes: [],
  hostLabel: "VDI",
  sessionGeneration: 0,
  activityGeneration: 0,
};
const chatActivity = new Map();
const activityKey = "codex-chat-activity:";

function activityForChat(id) {
  if (!chatActivity.has(id)) {
    let saved;
    try { saved = JSON.parse(localStorage.getItem(activityKey + id)); }
    catch { /* Browsing still works when storage is unavailable or corrupt. */ }
    chatActivity.set(id, saved && typeof saved === "object" ? saved : {});
  }
  return chatActivity.get(id);
}
function saveChatActivity(id, activity) {
  chatActivity.set(id, activity);
  state.activityGeneration++;
  try { localStorage.setItem(activityKey + id, JSON.stringify(activity)); }
  catch { /* Keep indicators in memory if the browser cannot persist them. */ }
}
function observeChatStatus(id, status, completedTurn = null) {
  const activity = activityForChat(id);
  const started = status?.type === "active" && !activity.running;
  const completed = status?.type === "idle" && (activity.running ||
    (completedTurn && completedTurn !== activity.completedTurn));
  if (!started && !completed) return;
  saveChatActivity(id, {...activity, running: started,
    unread: completed || !!activity.unread, revision: crypto.randomUUID(),
    completedTurn: completedTurn || activity.completedTurn});
}
function chatReplyVisible() {
  if (!state.id || state.loading || state.openFailed || document.hidden || !document.hasFocus()) return false;
  if (document.querySelector("dialog[open]")) return false;
  if (innerWidth <= 760 && document.body.classList.contains("sidebar-open")) return false;
  const viewport = $("#conversation");
  return viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 140;
}
function markChatRead() {
  if (!chatReplyVisible()) return;
  const activity = activityForChat(state.id);
  // Only acknowledge a history response obtained after this completion.
  if (!activity.unread || activity.running ||
      state.readableCompletion?.id !== state.id ||
      state.readableCompletion.revision !== activity.revision) return;
  saveChatActivity(state.id, {...activity, unread: false});
  renderSessions();
}
function observeLoadedTurns(turns, revision) {
  if (revision !== activityForChat(state.id).revision) {
    refreshUnreadReply();
    return;
  }
  const status = {type: turns.some(turn => turn.status === "inProgress") ? "active" : "idle"};
  observeChatStatus(state.id, status);
  state.statuses.set(state.id, status);
  state.readableCompletion = {id: state.id, revision: activityForChat(state.id).revision};
  renderSessions();
}
function refreshUnreadReply() {
  const generation = state.generation;
  clearTimeout(state.unreadRefresh);
  state.unreadRefresh = setTimeout(() => {
    if (generation === state.generation && !state.openFailed) loadTurns().catch(() => {});
  }, 100);
}
function observeSentMessage(body, result, revision) {
  if (revision !== activityForChat(body.id).revision) return;
  observeChatStatus(body.id, {type: "active"});
  const status = {type: result.turn && result.turn.status !== "inProgress" ? "idle" : "active"};
  observeChatStatus(body.id, status, status.type === "idle" ? result.turn.id : null);
  state.statuses.set(body.id, status);
  renderSessions();
}
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
async function api(path, body, headers = {}) {
  const options = { credentials: "same-origin", cache: "no-store" };
  if (body !== undefined) {
    options.method = "POST";
    options.headers = { "Content-Type": "application/json", ...headers };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(new URL("./api/" + path, location.href), options);
  if (response.status === 401)
    throw Object.assign(new Error("Sign-in expired. Reload this page to sign in again."), {status: 401});
  let data;
  try { data = await response.json(); }
  catch {
    throw Object.assign(new Error("The connection returned an unexpected response. Checking delivery again is safe."), {status: response.ok ? 502 : response.status});
  }
  if (!response.ok) throw Object.assign(new Error(data.error || "The request failed."), {status: response.status});
  return data;
}
function closeSidebar() {
  document.body.classList.remove("sidebar-open");
  markChatRead();
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
  } else if (activityForChat(thread.id).unread) {
    const dot = node("span", "session-unread");
    dot.setAttribute("role", "img");
    dot.setAttribute("aria-label", "Unread reply");
    dot.title = "Unread reply";
    meta.append(dot);
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
  const scrollTop = root.scrollTop;
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
  root.scrollTop = scrollTop;
  $("#more-sessions").hidden = !state.sessionCursor;
}
async function loadSessions(append = false, preserve = false) {
  const generation = ++state.sessionGeneration;
  const activityGeneration = state.activityGeneration;
  const query = new URLSearchParams({
    search: $("#search").value,
    archived: $("#archive").value,
  });
  const collection = query.toString();
  const keepPages = preserve && state.sessionQuery === collection && state.sessionPages;
  if (append && state.sessionCursor) query.set("cursor", state.sessionCursor);
  const result = await api("threads?" + query);
  if (generation !== state.sessionGeneration) return;
  if (!keepPages) state.sessionCursor = result.nextCursor;
  state.sessions = append ? mergeById(state.sessions, result.data)
    : keepPages ? mergeById(result.data, state.sessions) : result.data;
  state.sessionPages = append || keepPages;
  state.sessionQuery = collection;
  if (activityGeneration === state.activityGeneration) {
    for (const thread of result.data) {
      observeChatStatus(thread.id, thread.status);
      state.statuses.set(thread.id, thread.status);
    }
  }
  renderSessions();
  updateActivity();
}
function activeTurn() {
  return state.turns.find((turn) => turn.status === "inProgress");
}
function updateControls() {
  const active = activeTurn();
  const readOnly = !state.thread || state.openFailed || state.externalOwner || state.thread?.archived;
  $("#manage-chat").disabled = !state.thread || state.loading;
  $("#prompt").disabled = !state.id || state.loading || readOnly;
  $("#send").disabled =
    !state.id ||
    state.loading ||
    state.sending.has(state.id) ||
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
function prepareThread(id) {
  saveDraft();
  const generation = ++state.generation;
  notice();
  state.id = id;
  state.openFailed = false;
  state.externalOwner = false;
  $("#prompt").value = draftForChat(id);
  resizePrompt();
  const pendingSend = JSON.parse(localStorage.getItem("codex-send:" + id) || "null");
  state.images = draftImages(id, pendingSend);
  state.savedDraft = composerSnapshot();
  state.draftRevision = localStorage.getItem("codex-draft-revision:" + id);
  renderAttachments();
  $("#delivery-status").hidden = true;
  $("#clear-send").hidden = !pendingSend;
  state.loading = true;
  state.turns = [];
  state.thread = null;
  state.turnCursor = null;
  state.historyExpanded = false;
  state.olderLoading = false;
  $("#requests").replaceChildren();
  closeSidebar();
  renderSessions();
  updateControls();
  history.replaceState(null, "", "#" + encodeURIComponent(id));
  $("#welcome").hidden = true;
  $("#messages").replaceChildren(node("p", "muted", "Opening session…"));
  $("#title").textContent = "Opening session…";
  $("#project").textContent = "";
  return generation;
}
async function selectThread(id) {
  const generation = prepareThread(id);
  try {
    const resumed = await api("resume", {id});
    if (generation !== state.generation) return;
    state.externalOwner = !!resumed.externalOwner;
    if (state.externalOwner)
      notice("This session is open in an older Codex client. You can read it here. Close it in that client, then reopen it here to continue safely.");
    state.thread = resumed.thread;
    if ("serviceTier" in resumed) state.thread.serviceTier = resumed.serviceTier;
    renderModelControls();
    $("#title").textContent = title(state.thread);
    $("#project").textContent = state.thread.cwd || "VDI";
    await loadTurns(false, generation);
    recoverDelivery(id).catch(error => { if (state.id === id) notice(error.message); });
  } catch (error) {
    if (generation !== state.generation) return;
    state.openFailed = true;
    $("#title").textContent = "Could not open chat";
    $("#messages").replaceChildren(node("p", "muted", "Your draft is saved. Tap Refresh to try opening this chat again."));
    notice(error.message);
  } finally {
    if (generation === state.generation) {
      state.loading = false;
      markChatRead();
      renderRequests();
      updateControls();
    }
  }
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
  renderSettingsSummary();
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
  const metadataRevision = state.metadataRevision;
  const revision = activityForChat(state.id).revision;
  const result = await api("thread?id=" + encodeURIComponent(state.id));
  if (generation !== state.generation || metadataRevision !== state.metadataRevision) return;
  state.thread = {...state.thread, ...result.thread};
  if (revision === activityForChat(state.id).revision) {
    observeChatStatus(state.id, result.thread.status);
    state.statuses.set(state.id, result.thread.status);
    renderSessions();
  }
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
    state.sessions.some(thread => state.statuses.get(thread.id)?.type === "active");
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
  if (["thread/status/changed", "turn/started", "turn/completed"].includes(event.method))
    observeChatStatus(id, state.statuses.get(id),
      event.method === "turn/completed" ? params.turn?.id : null);
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
  if (["thread/status/changed", "turn/started", "turn/completed"].includes(event.method))
    renderSessions();
  updateActivity();
}
function mergeById(recent, previous) {
  const seen = new Set();
  return [...recent, ...previous].filter(item => {
    if (seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}
async function loadTurns(older = false, generation = state.generation) {
  if (!state.id || (older && (!state.turnCursor || state.olderLoading))) return;
  const firstLoad = !state.turns.length;
  const revision = activityForChat(state.id).revision;
  const query = new URLSearchParams({id: state.id});
  if (older) query.set("cursor", state.turnCursor);
  if (older) state.olderLoading = true;
  $("#older").disabled = !!state.olderLoading;
  try {
    const result = await api("turns?" + query);
    if (generation !== state.generation) return;
    const turns = result.data || result.turns || [];
    if (older || !state.historyExpanded) state.turnCursor = result.nextCursor;
    state.historyExpanded ||= older;
    state.turns = older ? mergeById(state.turns, turns)
      : result.nextCursor ? mergeById(turns, state.turns) : turns;
    if (!older && !result.nextCursor) {
      state.historyExpanded = false;
      state.turnCursor = null;
    }
    if (!older) observeLoadedTurns(turns, revision);
    renderMessages(!older && firstLoad, older);
    updateControls();
  } finally {
    if (generation === state.generation && older) {
      state.olderLoading = false;
      $("#older").disabled = false;
    }
  }
}
function appendText(root, text) {
  // Match opening and closing fences together so adjacent blocks stay separate.
  const fences = /^```[^\n]*\n([\s\S]*?)(?:^```[ \t]*(?:\n|$)|(?![\s\S]))/gm;
  let cursor = 0;
  for (const match of (text || "").matchAll(fences)) {
    if (match.index > cursor) root.append(node("p", "", text.slice(cursor, match.index)));
    root.append(node("pre", "", match[1]));
    cursor = match.index + match[0].length;
  }
  if (cursor < (text || "").length) root.append(node("p", "", text.slice(cursor)));
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
function renderMessages(forceBottom = false, preserveReading = false) {
  const anchor = readingAnchor();
  const viewport = $("#conversation"),
    nearBottom =
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 140;
  const open = new Set(
    [...$("#messages").querySelectorAll("details[open]")].map(
      (el) => el.dataset.itemId,
    ),
  );
  const fragment = conversationFragment(open);
  $("#older").hidden = !state.turnCursor;
  $("#messages").replaceChildren(fragment);
  if (forceBottom || (nearBottom && !preserveReading)) viewport.scrollTop = viewport.scrollHeight;
  else restoreReadingAnchor(anchor);
  markChatRead();
}
function conversationFragment(open) {
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
  return fragment;
}
function readingAnchor() {
  const top = $("#conversation").getBoundingClientRect().top;
  const item = [...$("#messages").children].find(el =>
    el.dataset.itemId && el.getBoundingClientRect().bottom > top);
  return item ? {id: item.dataset.itemId, top: item.getBoundingClientRect().top} : null;
}
function restoreReadingAnchor(anchor) {
  if (!anchor) return;
  const item = [...$("#messages").children].find(el => el.dataset.itemId === anchor.id);
  if (item) $("#conversation").scrollTop += item.getBoundingClientRect().top - anchor.top;
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
  if (["thread/archived", "thread/unarchived"].includes(method) &&
      ($("#archive").value === "true") !== (method === "thread/archived"))
    state.sessions = state.sessions.filter(thread => thread.id !== p.threadId);
  if (["thread/started", "thread/name/updated", "thread/archived", "thread/unarchived"].includes(method))
    loadSessions(false, true).catch(() => {});
  if (p.threadId !== state.id) return;
  if (method === "thread/name/updated") {
    state.metadataRevision++;
    refreshThread().catch(() => {});
  }
  if (method === "thread/archived" || method === "thread/unarchived") {
    state.metadataRevision++;
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
$("#menu").addEventListener("click", () => {
  document.body.classList.toggle("sidebar-open");
  markChatRead();
});
$("#scrim").addEventListener("click", closeSidebar);
$("#prompt").addEventListener("input", () => {
  saveDraft();
  resizePrompt();
  updateControls();
});
$("#prompt").addEventListener("keydown", (event) => {
  if (!event.isComposing && event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});
$("#composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("#prompt").value;
  const generation = state.generation;
  const id = state.id;
  if ($("#send").disabled || state.sending.has(id) || (!text.trim() && !state.images.length)) return;
  state.sending.add(id);
  updateControls();
  notice();
  try {
    const sent = await sendMessage(text);
    if (!sent || generation !== state.generation) return;
    if (state.thread && !state.thread.name && !state.thread.preview) {
      state.thread.preview = text;
      $("#title").textContent = title(state.thread);
    }
    await loadTurns();
  } catch (error) {
    if (generation === state.generation) notice(error.message);
  } finally {
    state.sending.delete(id);
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
async function refreshWorkspace() {
  if (!state.initialized) return start();
  notice();
  if (state.openFailed) return selectThread(state.id);
  await Promise.all([loadSessions(false, true), loadTurns(), refreshThread()]);
}
$("#refresh").addEventListener("click", () =>
  refreshWorkspace().catch(error => notice(error.message)));
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
function updateWorkspaceSelection() {
  for (const button of document.querySelectorAll(".workspace-choice"))
    button.setAttribute("aria-pressed", String(button.dataset.path === $("#cwd").value));
}
function renderWorkspaceChoices(choices) {
  const root = $("#workspace-options");
  root.replaceChildren();
  for (const choice of choices) {
    const button = node("button", "workspace-choice" + (choice.preferred ? " preferred" : ""));
    button.type = "button";
    button.disabled = !!state.creatingChat;
    button.dataset.path = choice.path;
    button.setAttribute("aria-label", "Use " + choice.path);
    const heading = node("strong", "", choice.name);
    if (choice.preferred) heading.append(node("span", "workspace-badge", "NPA project"));
    button.append(heading, node("span", "path", choice.path));
    button.addEventListener("click", () => {
      state.workspaceTouched = true;
      $("#cwd").value = choice.path;
      updateWorkspaceSelection();
    });
    root.append(button);
  }
  updateWorkspaceSelection();
}
async function openNewChat() {
  const generation = state.workspaceGeneration = (state.workspaceGeneration || 0) + 1;
  state.workspaceTouched = false;
  $("#cwd").value = state.lastNewCwd || state.thread?.cwd || state.defaultCwd || "";
  $("#workspace-host").textContent = "Choose a project on " + state.hostLabel + ".";
  $("#new-error").hidden = true;
  $("#workspace-status").textContent = "Loading recent paths…";
  renderWorkspaceChoices([]);
  $("#new-dialog").showModal();
  $("#new-dialog-title").focus();
  try {
    const result = await api("workspaces");
    if (generation !== state.workspaceGeneration || !$("#new-dialog").open) return;
    if (!state.workspaceTouched && !state.creatingChat)
      $("#cwd").value = state.lastNewCwd || result.data.find(choice => choice.preferred)?.path ||
        $("#cwd").value || result.data[0]?.path || "";
    renderWorkspaceChoices(result.data);
    $("#workspace-status").textContent = result.data.length ? "" : "No recent paths yet. Enter a project path below.";
  } catch (error) {
    if (generation === state.workspaceGeneration && $("#new-dialog").open)
      $("#workspace-status").textContent = "Recent paths are unavailable. You can still enter a path below.";
  }
}
for (const id of ["#new-chat", "#welcome-new"])
  $(id).addEventListener("click", openNewChat);
$("#cwd").addEventListener("input", () => {
  state.workspaceTouched = true;
  updateWorkspaceSelection();
});
$("#cancel-new").addEventListener("click", () => $("#new-dialog").close());
$("#new-dialog").addEventListener("cancel", event => {
  if (state.creatingChat) event.preventDefault();
});
$("#new-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.creatingChat) return;
  state.creatingChat = true;
  const cwd = $("#cwd").value;
  for (const control of $("#new-form").querySelectorAll("input, button")) control.disabled = true;
  $("#new-error").hidden = true;
  try {
    const result = await api("new", {cwd});
    state.lastNewCwd = result.thread.cwd || cwd;
    $("#new-dialog").close();
    await loadSessions();
    await selectThread(result.thread.id);
  } catch (error) {
    if ($("#new-dialog").open) {
      $("#new-error").textContent = error.message;
      $("#new-error").hidden = false;
    } else notice(error.message);
  } finally {
    state.creatingChat = false;
    for (const control of $("#new-form").querySelectorAll("input, button")) control.disabled = false;
  }
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    loadSessions(false, true).catch(() => {});
    loadTurns().catch(() => {});
    refreshThread().catch(() => {});
  }
});
window.addEventListener("focus", () => {
  if (state.initialized && !state.loading && !state.openFailed) loadTurns().catch(() => {});
});
window.addEventListener("hashchange", () => {
  const id = decodeURIComponent(location.hash.slice(1));
  if (state.initialized && id && id !== state.id)
    selectThread(id).catch(error => notice(error.message));
});
$("#conversation").addEventListener("scroll", markChatRead, {passive: true});
window.addEventListener("storage", event => {
  if (!event.key || event.key === "codex-draft-revision:" + state.id) syncDraft();
  if (event.key && !event.key.startsWith(activityKey)) return;
  if (event.key) chatActivity.delete(event.key.slice(activityKey.length));
  else chatActivity.clear();
  state.activityGeneration++;
  renderSessions();
});
if (window.visualViewport) {
  const resize = () => {
    const viewport = $("#conversation");
    const atBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 140;
    document.body.style.height = window.visualViewport.height + "px";
    if (atBottom) requestAnimationFrame(() => { viewport.scrollTop = viewport.scrollHeight; });
  };
  window.visualViewport.addEventListener("resize", resize);
  resize();
}
async function start() {
  try {
    const info = await api("state");
    state.initialized = true;
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
    state.defaultCwd = info.cwd;
    $("#connection").textContent = "● Connected to " + state.hostLabel;
    events();
    await loadSessions();
    const id = decodeURIComponent(location.hash.slice(1));
    if (id) await selectThread(id);
    else if (innerWidth <= 760) document.body.classList.add("sidebar-open");
    setInterval(() => {
      if (!document.hidden) loadSessions(false, true).catch(() => {});
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

function resizePrompt() {
  $("#prompt").style.height = "auto";
  $("#prompt").style.height = Math.min($("#prompt").scrollHeight, 180) + "px";
}
function setSettingsOpen(open) {
  $("#composer").classList.toggle("settings-open", open);
  $("#toggle-settings").setAttribute("aria-expanded", String(open));
}
function renderSettingsSummary() {
  const model = $("#model").selectedOptions[0]?.textContent || "Model settings";
  const effort = $("#effort").selectedOptions[0]?.textContent;
  const mode = state.thread?.mode === "plan" ? "Plan" : null;
  const speed = state.thread?.serviceTier === "priority" ? "Fast" : null;
  const text = [model, effort, mode, speed].filter(Boolean).join(" · ");
  $("#toggle-settings").textContent = text + " · Settings";
  $("#toggle-settings").title = text;
}
$("#toggle-settings").addEventListener("click", () =>
  setSettingsOpen(!$("#composer").classList.contains("settings-open")));
$("#prompt").addEventListener("focus", () => setSettingsOpen(false));

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

function draftForChat(id) {
  const key = "codex-draft:" + id;
  const current = localStorage.getItem(key);
  if (current !== null) return current;
  const legacy = localStorage.getItem("draft:" + id);
  if (legacy !== null) localStorage.setItem(key, legacy);
  return legacy || "";
}

function saveDraft() {
  if (!state.id) return;
  const snapshot = composerSnapshot();
  if (snapshot === state.savedDraft) return;
  try {
    state.draftRevision = writeDraft(state.id, $("#prompt").value, state.images);
    state.savedDraft = snapshot;
  }
  catch { notice("Browser storage is unavailable; keep this page open to retain your draft."); }
}

function composerSnapshot() {
  return JSON.stringify({text: $("#prompt").value, images: state.images});
}
function draftImages(id, pending = null) {
  const saved = localStorage.getItem("codex-draft-images:" + id);
  return saved === null ? pending?.images || [] : JSON.parse(saved);
}
function writeDraft(id, text, images) {
  const revision = crypto.randomUUID();
  localStorage.setItem("codex-draft:" + id, text);
  localStorage.setItem("codex-draft-images:" + id, JSON.stringify(images));
  localStorage.setItem("codex-draft-revision:" + id, revision);
  return revision;
}
function syncDraft() {
  if (!state.id || composerSnapshot() !== state.savedDraft) return;
  $("#prompt").value = draftForChat(state.id);
  state.images = draftImages(state.id);
  state.draftRevision = localStorage.getItem("codex-draft-revision:" + state.id);
  state.savedDraft = composerSnapshot();
  renderAttachments();
  resizePrompt();
  updateControls();
}
function clearDeliveredDraft(body) {
  const revision = localStorage.getItem("codex-draft-revision:" + body.id);
  if (body.draftRevision && body.draftRevision !== revision) return;
  const expected = JSON.stringify({text: body.text, images: body.images || []});
  const saved = JSON.stringify({text: draftForChat(body.id), images: draftImages(body.id, body)});
  if (saved !== expected) return;
  const cleared = writeDraft(body.id, "", []);
  if (state.id !== body.id || composerSnapshot() !== expected ||
      (body.draftRevision && state.draftRevision !== body.draftRevision)) return;
  $("#prompt").value = "";
  state.images = [];
  state.draftRevision = cleared;
  state.savedDraft = composerSnapshot();
  renderAttachments();
  resizePrompt();
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
  saveDraft();
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
    button.onclick = () => { state.images.splice(index, 1); saveDraft(); renderAttachments(); updateControls(); };
    $("#attachments").append(button);
  });
}

async function sendMessage(text) {
  saveDraft();
  const key = "codex-send:" + state.id;
  const previous = JSON.parse(localStorage.getItem(key) || "null");
  if (previous && (previous.text !== text || JSON.stringify(previous.images) !== JSON.stringify(state.images)))
    throw new Error("Check the previous send, then discard its pending state before sending different content.");
  const body = previous || {id: state.id, text, images: [...state.images],
    turnId: activeTurn()?.id, clientUserMessageId: crypto.randomUUID(), draftRevision: state.draftRevision};
  localStorage.setItem(key, JSON.stringify(body));
  $("#clear-send").hidden = false;
  const revision = activityForChat(body.id).revision;
  const result = await waitForDelivery(body, true);
  return result && finishDelivery(body, result, revision) ? body : null;
}

const deliveryWaits = new Map();
function waitForDelivery(body, submit = false) {
  const id = body.clientUserMessageId;
  if (!deliveryWaits.has(id))
    deliveryWaits.set(id, pollDelivery(body, submit).finally(() => deliveryWaits.delete(id)));
  return deliveryWaits.get(id);
}

async function pollDelivery(body, submit) {
  const path = "delivery?clientUserMessageId=" + encodeURIComponent(body.clientUserMessageId);
  while (pendingDelivery(body)) {
    try {
      const receipt = submit ? await api("send", body, {Prefer: "respond-async"}) : await api(path);
      if (!pendingDelivery(body)) return null;
      if (!receipt.state && submit) return receipt;
      submit = false;
      if (receipt.state === "complete") return receipt.result;
      if (receipt.state === "missing") submit = true;
      if (receipt.state === "uncertain")
        throw Object.assign(new Error("Delivery is unconfirmed. Check this conversation before discarding the pending send. " + (receipt.error || "")), {status: 409});
      if (!["pending", "missing"].includes(receipt.state))
        throw Object.assign(new Error("The service returned an invalid delivery receipt. Your draft is saved."), {status: 409});
    } catch (error) {
      submit = false;
      if (error.status && error.status < 500) throw error;
    }
    if (state.id === body.id) {
      $("#delivery-status").hidden = false;
      $("#delivery-status").textContent = "Checking delivery… You can leave this page; the message keeps its original send identity.";
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  return null;
}

function pendingDelivery(body) {
  const pending = JSON.parse(localStorage.getItem("codex-send:" + body.id) || "null");
  return pending?.clientUserMessageId === body.clientUserMessageId;
}
function finishDelivery(body, result, revision) {
  const key = "codex-send:" + body.id;
  if (!pendingDelivery(body)) return false;
  clearDeliveredDraft(body);
  observeSentMessage(body, result, revision);
  if (!state.sessions.some(thread => thread.id === body.id))
    loadSessions(false, true).catch(() => {});
  localStorage.removeItem(key);
  if (state.id === body.id) {
    $("#clear-send").hidden = true;
    $("#delivery-status").hidden = false;
    $("#delivery-status").textContent = result.delivery?.target === "VS Code" ?
      (result.delivery.visible ? "✓ Message visible in VS Code on your Mac" : "Sent · waiting for VS Code to display the message") : "✓ Sent to this conversation";
  }
  return true;
}

async function recoverDelivery(id) {
  const body = JSON.parse(localStorage.getItem("codex-send:" + id) || "null");
  if (!body || deliveryWaits.has(body.clientUserMessageId)) return;
  const revision = activityForChat(id).revision;
  const result = await waitForDelivery(body);
  if (!result || !finishDelivery(body, result, revision)) return;
  if (state.id !== id) return;
  updateControls();
  await loadTurns();
}

$("#clear-send").onclick = () => {
  localStorage.removeItem("codex-send:" + state.id);
  $("#clear-send").hidden = true;
  $("#delivery-status").hidden = true;
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
      await Promise.all([loadSessions(false, true), refreshThread(), loadTurns()]);
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
      state.metadataRevision++;
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

// Installation keeps the same chat URL and authenticated browser session.
$("#install-app").hidden = window.matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
$("#install-app").addEventListener("click", () => $("#install-help").showModal());
