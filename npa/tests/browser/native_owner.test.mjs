// Verify native follower ownership and controls without touching a live Codex account.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {NativeOwner} from '../../src/npa/tools/desktop/native/owner.mjs';
import {ownerState} from '../../src/npa/tools/desktop/native/owner-state.mjs';
import {CodexIPC} from '../../src/npa/tools/desktop/native/ipc.mjs';

function fixture() {
  const calls = [], frames = [];
  const thread = {id: 'shared', cwd: '/workspace', createdAt: 1, updatedAt: 2,
    status: {type: 'active'}, turns: [{id: 'turn', status: 'inProgress', items: [
      {id: 'input', type: 'userMessage', clientId: 'message', content: [{type: 'text', text: 'Hello'}]},
      {id: 'reply', type: 'agentMessage', text: 'Working'},
    ]}]};
  const owned = {active: true, turnId: 'turn', settings: {model: 'model', effort: 'high'}};
  const context = {ownThreads: new Map([['shared', owned]]), approvals: new Map(),
    ipc: {connected: true, clientId: 'owner', write: frame => frames.push(frame)},
    app: {request: async (method, params) => {
      calls.push({method, params});
      return method === 'thread/read' ? {thread} : {turn: {id: 'next', status: 'inProgress'}};
    }, respond: (id, result) => calls.push({id, result})},
    mutate: async (id, action) => action(), notify: () => {}};
  return {owner: new NativeOwner(context), context, calls, frames, thread, owned};
}
function request(method, params = {}, version = 1) {
  return {method, version, params: {conversationId: 'shared', hostId: 'local', ...params}};
}

test('discovery advertises only the actual writer and supported local protocol', () => {
  const {owner, context} = fixture();
  assert.equal(owner.accepts(request('thread-owner-discovery')), true);
  assert.equal(owner.accepts(request('thread-owner-discovery', {hostId: 'remote'})), false);
  assert.equal(owner.accepts(request('thread-owner-discovery', {}, 99)), false);
  assert.equal(owner.accepts(request('thread-follower-start-turn', {}, 2)), true);
  assert.equal(owner.accepts(request('thread-follower-start-turn', {}, 1)), false);
  assert.equal(owner.accepts(request('thread/fork')), false);
  context.ownThreads.clear();
  assert.equal(owner.accepts(request('thread-owner-discovery')), false);
});

test('shared snapshots preserve canonical history, active status and pending approvals', async () => {
  const {owner, context, frames, calls} = fixture();
  context.approvals.set('approval', {id: 'approval', method: 'item/commandExecution/requestApproval',
    params: {threadId: 'shared'}});
  owner.followers.set('shared', new Set(['vscode']));
  const revision = await owner.publish('shared');
  const frame = frames.at(-1), state = frame.params.change.conversationState;
  assert.equal(revision, 1);
  assert.deepEqual(frame.targetClientIds, ['vscode']);
  assert.equal(state.threadRuntimeStatus.type, 'active');
  assert.equal(state.resumeState, 'resumed');
  assert.equal(state.requests[0].id, 'approval');
  const turn = state.turnHistory.history.entitiesByKey.turn;
  assert.equal(turn.turnId, 'turn');
  assert.equal(turn.params.clientUserMessageId, 'message');
  assert.equal(turn.items[1].text, 'Working');
  assert.deepEqual(calls.map(call => call.method), ['thread/read']);
});

test('a subscribing view receives cached state before the asynchronous refresh', async () => {
  const {owner, frames} = fixture();
  await owner.publish('shared');
  owner.following({version: 1, sourceClientId: 'vscode', params: {
    conversationId: 'shared', hostId: 'local', following: true}});
  assert.equal(frames.length, 1);
  assert.equal(frames[0].params.change.conversationState.id, 'shared');
  clearTimeout(owner.refreshTimer);
});

test('a view opening before the first snapshot starts its read immediately', async () => {
  const {owner, calls, frames} = fixture();
  owner.following({version: 1, sourceClientId: 'vscode', params: {
    conversationId: 'shared', hostId: 'local', following: true}});
  assert.equal(calls[0].method, 'thread/read');
  await owner.refreshPromise;
  assert.equal(frames[0].params.change.conversationState.id, 'shared');
});

test('native prompts run on the owning engine with the original message identity', async () => {
  const {owner, calls, owned} = fixture();
  const result = await owner.handle(request('thread-follower-start-turn', {turnStart: {request: {
    threadId: 'wrong', clientUserMessageId: 'message', input: [{type: 'text', text: 'Continue'}]}, context: {}}}, 2));
  assert.equal(calls[0].method, 'turn/start');
  assert.equal(calls[0].params.threadId, 'shared');
  assert.equal(calls[0].params.clientUserMessageId, 'message');
  assert.equal(result.result.turn.id, 'next');
  assert.equal(owned.turnId, 'next');
  clearTimeout(owner.refreshTimer);
});

test('foreign context and stale interrupts never reach the engine', async () => {
  const {owner, calls} = fixture();
  await assert.rejects(owner.handle(request('thread-follower-start-turn', {turnStart: {
    request: {}, context: {responseItems: [{role: 'developer'}]}}}, 2)), /not supported/);
  await assert.rejects(owner.handle(request('thread-follower-interrupt-turn', {expectedTurnId: 'old'}, 4)), /changed/);
  assert.equal(calls.length, 0);
});

test('native approvals must match the pending request and conversation', async () => {
  const {owner, context, calls} = fixture();
  context.approvals.set('1', {id: 1, method: 'item/commandExecution/requestApproval', params: {threadId: 'shared'}});
  await assert.rejects(owner.handle(request('thread-follower-file-approval-decision', {requestId: 1, decision: 'accept'})), /no longer pending/);
  assert.equal(calls.length, 0);
  await owner.handle(request('thread-follower-command-approval-decision', {requestId: 1, decision: 'decline'}));
  assert.deepEqual(calls[0], {id: 1, result: {decision: 'decline'}});
  assert.equal(context.approvals.size, 0);
  await assert.rejects(owner.handle(request('thread-follower-command-approval-decision', {requestId: 1, decision: 'accept'})), /no longer pending/);
  clearTimeout(owner.refreshTimer);
});

test('a failed control is not replayed and does not claim settings changed', async () => {
  const {owner, context, owned} = fixture(); let attempts = 0;
  context.app.request = async () => { attempts++; throw new Error('Disconnected'); };
  await assert.rejects(owner.handle(request('thread-follower-update-thread-settings', {
    threadSettings: {model: 'different'}}, 2)), /Disconnected/);
  assert.equal(attempts, 1); assert.equal(owned.settings.model, 'model');
});

test('archive during an in-flight snapshot cannot publish stale ownership', async () => {
  const {owner, context, frames, thread} = fixture();
  context.app.request = async () => { context.ownThreads.clear(); return {thread}; };
  await assert.rejects(owner.publish('shared'), /another owner/);
  assert.equal(frames.length, 0);
});

test('empty chats and collaboration items retain valid native history shapes', () => {
  const state = ownerState({id: 'new', createdAt: 1, updatedAt: 1, turns: []}, {model: 'model'}, []);
  assert.deepEqual(state.turnHistory.history.islands[0].entries, []);
  assert.equal(state.latestCollaborationMode.settings.model, 'model');
  const {thread} = fixture();
  thread.turns[0].items.push({type: 'collabAgentToolCall', receiverThreadIds: ['child']});
  const turn = ownerState(thread, {}, []).turnHistory.history.entitiesByKey.turn;
  assert.deepEqual(turn.items.at(-1).receiverThreads, [{threadId: 'child', thread: null}]);
});

test('one owner identity cannot claim a different chat in the same engine', () => {
  const {context} = fixture(); context.ownThreads.set('other', {active: true});
  const owner = new NativeOwner(context, 'shared');
  assert.equal(owner.owns('shared'), true);
  assert.equal(owner.owns('other'), false);
});

test('stale conditional settings and permissions cannot change a newer turn', async () => {
  const {owner, calls} = fixture();
  const result = await owner.handle(request('thread-follower-update-thread-settings', {
    threadSettings: {model: 'replacement'}, condition: {ifEffortEquals: 'low'}}, 2));
  assert.equal(result.applied, false);
  await assert.rejects(owner.handle(request('thread-follower-update-thread-settings', {
    threadSettings: {approvalPolicy: 'never'}, activeTurnId: 'old-turn'}, 2)), /active turn changed/);
  assert.equal(calls.length, 0); clearTimeout(owner.refreshTimer);
});

test('disconnecting one owner clears only its stale follower state', () => {
  const changes = [];
  const ipc = Object.create(CodexIPC.prototype);
  ipc.owners = new Map([['shared', 'gone'], ['other', 'connected']]);
  ipc.states = new Map([['shared', {active: true}], ['other', {active: true}]]);
  ipc.emit = (event, id) => changes.push({event, id});
  ipc.dispatch({method: 'client-status-changed', params: {status: 'disconnected', clientId: 'gone'}});
  assert.equal(ipc.states.has('shared'), false); assert.equal(ipc.owners.has('shared'), false);
  assert.equal(ipc.states.has('other'), true);
  assert.deepEqual(changes, [{event: 'change', id: 'shared'}]);
});
