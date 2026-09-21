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
