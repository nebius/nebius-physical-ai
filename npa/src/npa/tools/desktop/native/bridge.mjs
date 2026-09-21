// Adapts the existing Mac Codex owner to a private JSONL control connection.
import {readFileSync, readdirSync, chmodSync} from 'node:fs';
import net from 'node:net';
import {hostname} from 'node:os';
import {join} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {randomUUID} from 'node:crypto';
import {createInterface} from 'node:readline';
import {protocolCall, pendingRequests} from './protocol.mjs';
import {CodexIPC} from './ipc.mjs';
import {AppServer} from './app-server.mjs';
import {SessionStore, liveTurns, liveMessages, itemMessage} from './store.mjs';
import {threadSettings, validateSettings, messageVisible} from './settings.mjs';
import {removeStaleSocket} from './listener.mjs';
import {requireUnowned} from './ownership.mjs';

const configPath = process.argv[2];
const config = JSON.parse(readFileSync(configPath));
const ipc = new CodexIPC(config.codexHome);
const app = new AppServer(config.binary, config.codexHome);
const store = new SessionStore(config.codexHome);
const ownThreads = new Map();
const approvals = new Map();
let changed;
let catalogCache;

function notify() {
  if (changed) return;
  changed = setTimeout(() => { changed = null; send({method: 'native/changed', params: {pending: pendingRequests(ipc, approvals)}}); }, 300);
}
const clients = new Set();
const daemon = process.argv.includes('--serve');
function send(message) {
  const data = JSON.stringify(message) + '\n';
  if (daemon) { for (const client of clients) if (!client.destroyed) client.write(data); }
  else process.stdout.write(data);
}
function followLoadedChats() {
  try {
    for (const name of readdirSync(join(config.codexHome, 'thread-writer-locks'))) {
      if (/^[a-f0-9-]{36}\.lock$/.test(name) && !ipc.following.has(name.slice(0, -5))) ipc.follow(name.slice(0, -5));
    }
  } catch { /* The directory appears when Codex first loads a chat. */ }
}
ipc.on('change', notify); ipc.on('connected', () => { followLoadedChats(); notify(); }); ipc.on('disconnected', notify);
setInterval(followLoadedChats, 5000).unref();
app.on('failure', () => { ownThreads.clear(); approvals.clear(); notify(); });
app.on('event', message => {
  const {method, params} = message; const state = ownThreads.get(params?.threadId);
  if (message.id != null) approvals.set(String(message.id), message);
  if (state && method === 'turn/started') { state.active = true; state.turnId = params.turn.id; }
  if (state && method === 'turn/completed') { state.active = false; state.turnId = null; state.live = []; }
  if (state && method === 'item/started') state.live.push(params.item);
  if (state && method === 'item/completed') {
    const index = state.live.findIndex(item => item.id === params.item.id);
    if (index < 0) state.live.push(params.item); else state.live[index] = params.item;
  }
  if (state && method === 'item/agentMessage/delta') {
    const item = state.live.find(item => item.id === params.itemId); if (item) item.text = (item.text || '') + params.delta;
  }
  notify();
});

async function list(params) {
  const rows = store.list(params.archived, params.includeAgents);
  if (!params.archived) for (const [id, own] of ownThreads) {
    if (!rows.some(row => row.id === id) && getRow(id).unsaved) rows.unshift({id, title: own.thread?.preview || 'New chat', cwd: own.thread?.cwd,
      updatedAt: own.thread?.createdAt || 0, ...threadSettings(null, own), archived: false});
  }
  return rows.map(row => ({...row,
    status: ownThreads.get(row.id)?.active ? 'active' : ipc.states.get(row.id)?.threadRuntimeStatus?.type || 'unknown'}));
}
function getRow(id) {
  try { return store.get(id); }
  catch (error) {
    const own = ownThreads.get(id);
    if (error.message !== 'Chat not found' || !own?.thread) throw error;
    return {id, title: own.thread.preview || 'New chat', cwd: own.thread.cwd, unsaved: true, archived: false};
  }
}
async function read(params) {
  const row = getRow(params.id);
  if (!ipc.following.has(params.id)) ipc.follow(params.id);
  const live = ipc.states.get(params.id); const own = ownThreads.get(params.id);
  let messages = row.unsaved ? [] : await store.history(params.id);
  if (live && liveTurns(live).length && live.turnHistory?.history?.isComplete !== false) messages = liveMessages(live);
  else if (own?.active) {
    const additions = own.live.map(itemMessage).filter(Boolean);
    const ids = new Set(messages.map(x => x.id)); messages = [...messages, ...additions.filter(x => !ids.has(x.id))];
  }
  const activeTurn = liveTurns(live).findLast(turn => turn.status === 'inProgress');
  const pending = [...approvals.values()].filter(x => x.params?.threadId === params.id);
  const visibleCount = Number.isSafeInteger(params.visibleCount) && params.visibleCount > 0 ? params.visibleCount : 150;
  const totalMessages = messages.length;
  messages = messages.slice(-visibleCount).map(message => {
    if (message.role !== 'tool' || message.text?.length <= 48000) return message;
    return {...message, text: message.text?.slice(0, 48000) + '\n\n[Output preview truncated. The full output is saved on your Mac.]'};
  });
  return {id: row.id, title: row.name || row.title || 'New chat', cwd: row.cwd,
    ...threadSettings(live, own, row), messages, totalMessages, hasMore: totalMessages > visibleCount,
    active: Boolean(own?.active || live?.threadRuntimeStatus?.type === 'active'),
    turnId: own?.turnId || activeTurn?.turnId, owner: live ? 'VS Code' : own ? 'Mobile' : 'Saved chat',
    requests: [...(live?.requests ?? []), ...pending], vscodeConnected: ipc.connected};
}
async function ensureOwn(id) {
  await app.start();
  if (!ownThreads.has(id)) {
    await requireUnowned(config.codexHome, id);
    const result = await app.request('thread/resume', {threadId: id});
    ownThreads.set(id, {active: false, live: [], thread: result.thread,
      settings: {model: result.model, effort: result.reasoningEffort, serviceTier: result.serviceTier}});
  }
}
async function prompt(params) {
  if (!params.text?.trim() && !params.images?.length) throw new Error('Enter a message');
  const input = [{type: 'text', text: params.text || '', text_elements: []},
    ...(params.images ?? []).map(url => ({type: 'image', url}))];
  const clientUserMessageId = params.clientUserMessageId || randomUUID();
  const owner = await ipc.owner(params.id);
  if (owner) return promptInVSCode(params, input, clientUserMessageId);
  await ensureOwn(params.id);
  const own = ownThreads.get(params.id);
  if (own.active) return {result: await app.request('turn/steer', {threadId: params.id, input, expectedTurnId: own.turnId, clientUserMessageId}),
    clientUserMessageId, delivery: {target: 'Mac', visible: true}};
  const result = await app.request('turn/start', {threadId: params.id, input, clientUserMessageId, turnTrigger: 'composer'});
  own.active = result.turn.status === 'inProgress'; own.turnId = result.turn.id;
  notify(); return {turnId: own.turnId, clientUserMessageId, delivery: {target: 'Mac', visible: true}};
}
async function promptInVSCode(params, input, clientUserMessageId) {
  ipc.follow(params.id);
  let result;
  const live = ipc.states.get(params.id);
  const active = live ? live.threadRuntimeStatus?.type === 'active' : Boolean(params.steer);
  if (active) {
    const cwd = store.get(params.id).cwd;
    const restoreMessage = {id: clientUserMessageId, text: params.text, cwd, createdAt: Date.now(),
      context: {prompt: params.text, turnTrigger: 'composer', addedFiles: [], fileAttachments: [],
        ideContext: null, imageAttachments: [], commentAttachments: [], workspaceRoots: [cwd]}};
    result = await ipc.control(params.id, 'thread-follower-steer-turn', {input, restoreMessage, attachments: [], clientUserMessageId});
  } else {
    result = await ipc.control(params.id, 'thread-follower-start-turn', {
      turnStart: {request: {threadId: params.id, input, clientUserMessageId, turnTrigger: 'composer'},
        context: {inheritThreadSettings: true, attachments: [], commentAttachments: []}}});
  }
  for (let attempt = 0; attempt < 20 && !messageVisible(ipc.states.get(params.id), clientUserMessageId); attempt++)
    await new Promise(resolve => setTimeout(resolve, 100));
  notify();
  return {result, clientUserMessageId, delivery: {target: 'VS Code', visible: messageVisible(ipc.states.get(params.id), clientUserMessageId)}};
}
async function stop(params) {
  const owner = await ipc.owner(params.id);
  if (owner) return ipc.control(params.id, 'thread-follower-interrupt-turn', {mode: 'user-stop', expectedTurnId: params.turnId ?? null});
  const state = ownThreads.get(params.id);
  if (!state?.turnId) throw new Error('No running turn controlled by this connection');
  return app.request('turn/interrupt', {threadId: params.id, turnId: state.turnId});
}
async function create(params) {
  await app.start();
  const result = await app.request('thread/start', {cwd: params.cwd || config.defaultCwd, model: params.model || undefined});
  ownThreads.set(result.thread.id, {active: false, live: [], thread: result.thread,
    settings: {model: result.model, effort: result.reasoningEffort, serviceTier: result.serviceTier}});
  return {id: result.thread.id};
}
async function respond(params) {
  const allowed = ['thread-follower-command-approval-decision', 'thread-follower-file-approval-decision', 'thread-follower-submit-user-input'];
  if (!allowed.includes(params.method)) throw new Error('Unsupported response');
  const owner = await ipc.owner(params.id);
  if (owner) return ipc.control(params.id, params.method, params.answer);
  const request = approvals.get(String(params.requestId));
  if (!request || request.params.threadId !== params.id) throw new Error('This request is no longer pending');
  app.respond(request.id, params.result); approvals.delete(String(request.id)); notify(); return {ok: true};
}
async function openInVSCode(params) {
  if (!/^[a-f0-9-]{36}$/.test(params.id)) throw new Error('Invalid chat identifier');
  if (getRow(params.id).unsaved) throw new Error('Send the first message before opening this chat in VS Code');
  const own = ownThreads.get(params.id);
  if (own?.active) throw new Error('Wait for this turn to finish or stop it before opening in VS Code');
  if (own) { await app.request('thread/unsubscribe', {threadId: params.id}); ownThreads.delete(params.id); }
  await promisify(execFile)('/usr/bin/open', ['-a', 'Visual Studio Code', `vscode://openai.chatgpt/local/${params.id}`]);
  for (let attempt = 0; attempt < 20; attempt++) {
    if (await ipc.owner(params.id)) { ipc.follow(params.id); return {ok: true, connected: true}; }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  return {ok: true, connected: false};
}
async function models() {
  if (catalogCache && Date.now() - catalogCache.updatedAt < 300000) return catalogCache.value;
  await app.start();
  const data = []; let cursor;
  do {
    const page = await app.request('model/list', {cursor}); data.push(...page.data); cursor = page.nextCursor;
  } while (cursor);
  const presets = await app.request('collaborationMode/list', {});
  const value = {data: data.filter(model => !model.hidden), modes: presets.data};
  catalogCache = {value, updatedAt: Date.now()}; return value;
}
async function configure(params) {
  const row = getRow(params.id);
  if (row.archived) throw new Error('Unarchive this chat in VS Code before changing its settings');
  const catalog = await models();
  const settings = validateSettings(params, catalog.data, catalog.modes);
  const owner = await ipc.owner(params.id);
  const live = ipc.states.get(params.id); const own = ownThreads.get(params.id);
  const current = threadSettings(live, own, row);
  // Preserve the current mode's instructions when only model/effort/speed changes.
  const previousMode = live?.latestThreadSettings?.collaborationMode ?? live?.latestCollaborationMode ?? own?.settings?.collaborationMode;
  settings.collaborationMode = {mode: params.mode, settings: {
    ...previousMode?.settings, model: settings.model, reasoning_effort: settings.effort,
    developer_instructions: params.mode === current.mode ? previousMode?.settings?.developer_instructions ?? null : null}};
  if (owner) {
    const result = await ipc.control(params.id, 'thread-follower-update-thread-settings', {threadSettings: settings});
    if (!result.applied) throw new Error('VS Code did not apply these settings. Refresh and try again.');
    ipc.follow(params.id);
  } else {
    await ensureOwn(params.id);
    await app.request('thread/settings/update', {threadId: params.id, ...settings});
    Object.assign(ownThreads.get(params.id).settings, settings);
  }
  notify(); return {ok: true, settings: threadSettings({latestThreadSettings: settings}, null, row)};
}
const handlers = {list, read, prompt, stop, create, respond, openInVSCode, models, configure,
  status: async () => ({hostname: hostname(), vscodeConnected: ipc.connected}),
};


const context = {handlers, ipc, app, approvals, ownThreads, mutations: 0, upgrading: false};
async function dispatch(line, output, client) {
  let message;
  let mutation = false;
  try {
    message = JSON.parse(line);
    if (message.method === 'initialized') return;
    const mutating = ['thread/start', 'thread/settings/update', 'turn/start', 'turn/steer', 'thread/open'].includes(message.method);
    if (mutating && context.upgrading) throw new Error('The local adapter is updating. Reconnect before sending.');
    mutation = mutating;
    if (mutation) context.mutations++;
    const result = await protocolCall(context, message);
    if (message.method === 'runtime/prepareUpdate') context.updateClient = client;
    if (message.id != null && message.method) output({id: message.id, result});
  } catch (error) {
    if (message?.id != null) output({id: message.id, error: {message: error.message}});
  } finally {
    if (mutation) context.mutations--;
  }
}
if (daemon) {
  await removeStaleSocket(config.native_socket);
  net.createServer(socket => {
    clients.add(socket);
    const output = message => { if (!socket.destroyed) socket.write(JSON.stringify(message) + '\n'); };
    createInterface({input: socket}).on('line', line => { void dispatch(line, output, socket); });
    socket.on('error', () => {});
    socket.on('close', () => {
      clients.delete(socket);
      if (context.updateClient === socket) {
        context.upgrading = false;
        context.updateClient = null;
      }
    });
    output({method: 'native/changed', params: {pending: pendingRequests(ipc, approvals)}});
  }).listen(config.native_socket, () => chmodSync(config.native_socket, 0o600));
} else {
  createInterface({input: process.stdin}).on('line', line => { void dispatch(line, send); });
  process.stdin.on('end', () => { app.child?.kill(); process.exit(0); });
}
