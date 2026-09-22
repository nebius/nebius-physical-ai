// Exercise the Mac adapter without opening a real Codex conversation.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {protocolCall, pendingRequests} from '../../src/npa/tools/desktop/native/protocol.mjs';

test('native prompt carries image and stable message identity to the same owner', async () => {
  let sent;
  const context = {handlers: {
    read: async () => ({active: true, turnId: 'active-turn'}),
    prompt: async params => { sent = params; return {delivery: {visible: true}}; },
  }};
  await protocolCall(context, {method: 'turn/steer', params: {threadId: 'same-thread',
    expectedTurnId: 'active-turn', clientUserMessageId: 'stable-id',
    input: [{type: 'text', text: 'Continue'}, {type: 'image', url: 'data:image/png;base64,AAAA'}]}});
  assert.equal(sent.id, 'same-thread');
  assert.equal(sent.clientUserMessageId, 'stable-id');
  assert.deepEqual(sent.images, ['data:image/png;base64,AAAA']);
});

test('stale steering cannot accidentally start another turn', async () => {
  const context = {handlers: {read: async () => ({active: false}),
    prompt: async () => assert.fail('must not send')}};
  await assert.rejects(protocolCall(context, {method: 'turn/steer',
    params: {threadId: 'same-thread', expectedTurnId: 'finished-turn'}}), /active turn changed/);
});

test('history retains original message identities and active turn controls', async () => {
  const context = {handlers: {read: async () => ({id: 'same-thread', active: true,
    turnId: 'active-turn', messages: [{id: 'message', role: 'user', text: 'hello', clientId: 'client'}]})}};
  const result = await protocolCall(context, {method: 'thread/turns/list', params: {threadId: 'same-thread'}});
  assert.equal(result.data[0].id, 'active-turn');
  assert.equal(result.data[0].status, 'inProgress');
  assert.equal(result.data[0].items[0].clientUserMessageId, 'client');
});

test('approval identities distinguish different thread owners', () => {
  const ipc = {states: new Map([['first', {requests: [{id: 1, method: 'approval'}]}],
    ['second', {requests: [{id: 1, method: 'approval'}]}]])};
  const pending = pendingRequests(ipc, new Map());
  assert.notEqual(pending[0].id, pending[1].id);
  assert.equal(pending[0].params.threadId, 'first');
});


test('engine update refuses active or in-flight work and can be cancelled', async () => {
  const context = {handlers: {}, ownThreads: new Map([['owned', {active: true}]]), mutations: 0};
  await assert.rejects(protocolCall(context, {method: 'runtime/prepareUpdate'}), /finish/);
  context.ownThreads.clear(); context.mutations = 1;
  await assert.rejects(protocolCall(context, {method: 'runtime/prepareUpdate'}), /finish/);
  context.mutations = 0;
  assert.deepEqual(await protocolCall(context, {method: 'runtime/prepareUpdate'}), {prepared: true});
  assert.equal(context.upgrading, true);
  await protocolCall(context, {method: 'runtime/cancelUpdate'});
  assert.equal(context.upgrading, false);
});

test('engine startup preserves live sockets and refuses regular files', async () => {
  const {removeStaleSocket} = await import('../../src/npa/tools/desktop/native/listener.mjs');
  const {mkdtempSync, writeFileSync, rmSync, existsSync} = await import('node:fs');
  const {tmpdir} = await import('node:os');
  const {join} = await import('node:path');
  const {default: net} = await import('node:net');
  const directory = mkdtempSync(join(tmpdir(), 'npa-listener-'));
  const path = join(directory, 'engine.sock');
  const server = net.createServer(socket => socket.end());
  try {
    await new Promise(resolve => server.listen(path, resolve));
    await assert.rejects(removeStaleSocket(path), /already owns/);
    assert.equal(existsSync(path), true);
    await new Promise(resolve => server.close(resolve));
    writeFileSync(path, 'preserve this file');
    await assert.rejects(removeStaleSocket(path), /not a socket/);
    assert.equal(existsSync(path), true);
  } finally { server.close(); rmSync(directory, {recursive: true, force: true}); }
});

test('management targets the original native chat without resuming or sending', async () => {
  const calls = [];
  const context = {handlers: {manage: async params => { calls.push(params); return {ok: true}; }}};
  for (const method of ['thread/name/set', 'thread/archive', 'thread/unarchive'])
    await protocolCall(context, {method, params: {threadId: 'original-thread', name: 'Renamed'}});
  assert.deepEqual(calls, [
    {id: 'original-thread', action: 'rename', name: 'Renamed'},
    {id: 'original-thread', action: 'archive'}, {id: 'original-thread', action: 'unarchive'},
  ]);
});

test('archived native threads retain their identity and read-only state', async () => {
  const context = {handlers: {read: async () => ({id: 'saved-thread', title: 'Saved', archived: true})}};
  const result = await protocolCall(context, {method: 'thread/read', params: {threadId: 'saved-thread'}});
  assert.equal(result.thread.id, 'saved-thread');
  assert.equal(result.thread.archived, true);
});

test('native archive refuses active work and another VS Code owner', async () => {
  const {manageThread} = await import('../../src/npa/tools/desktop/native/management.mjs');
  const ownThreads = new Map([['same', {active: true}]]);
  const context = {getRow: () => ({archived: false}), ownThreads,
    ipc: {states: new Map(), owner: async () => 'vscode-owner'},
    app: {start: () => assert.fail('must not touch engine')}, config: {}};
  await assert.rejects(manageThread(context, {id: 'same', action: 'archive'}), /finish/);
  ownThreads.clear();
  await assert.rejects(manageThread(context, {id: 'same', action: 'archive'}), /Close this chat in VS Code/);
});

test('native archive publishes only confirmed changes and drops released ownership', async () => {
  const {manageThread} = await import('../../src/npa/tools/desktop/native/management.mjs');
  const calls = [];
  const context = {getRow: () => ({archived: false, cwd: '/workspace/project'}),
    ownThreads: new Map([['same', {active: false}]]),
    ipc: {states: new Map([['same', {}]]), archivalChanged: (...args) => calls.push(args)},
    app: {start: async () => {}, request: async () => {throw Error('Archive failed');}},
    notify: () => calls.push('notify'), config: {}};
  await assert.rejects(manageThread(context, {id: 'same', action: 'archive'}), /Archive failed/);
  assert.equal(context.ownThreads.has('same'), true);
  assert.deepEqual(calls, []);
  context.app.request = async (method, params) => calls.push([method, params]);
  await manageThread(context, {id: 'same', action: 'archive'});
  assert.equal(context.ownThreads.has('same'), false);
  assert.equal(context.ipc.states.has('same'), false);
  assert.deepEqual(calls, [['thread/archive', {threadId: 'same'}], ['same', true, '/workspace/project'], 'notify']);
});

test('native rename uses metadata API without resuming or taking ownership', async () => {
  const {manageThread} = await import('../../src/npa/tools/desktop/native/management.mjs');
  const ownThreads = new Map([['same', {active: true}]]), calls = [];
  const context = {getRow: () => ({archived: false}), ownThreads,
    ipc: {owner: () => assert.fail('rename must not take ownership')}, config: {},
    app: {start: async () => {}, request: async (...args) => calls.push(args)}, notify: () => {}};
  await manageThread(context, {id: 'same', action: 'rename', name: 'New name'});
  assert.deepEqual(calls, [['thread/name/set', {threadId: 'same', name: 'New name'}]]);
  assert.equal(ownThreads.get('same').active, true);
});
