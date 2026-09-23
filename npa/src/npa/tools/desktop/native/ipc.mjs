// Connects to VS Code's local Codex owner without starting another writer.
import net from 'node:net';
import {randomUUID} from 'node:crypto';
import {EventEmitter} from 'node:events';
import {join} from 'node:path';
import {homedir} from 'node:os';
import {applyPatches, enablePatches} from 'immer';
enablePatches();

const versions = {'initialize': 0, 'thread-owner-discovery': 1,
  'thread-follower-start-turn': 2, 'thread-follower-steer-turn': 1,
  'thread-follower-interrupt-turn': 4, 'thread-follower-load-complete-history': 1,
  'thread-follower-update-thread-settings': 2,
  'thread-follower-command-approval-decision': 1,
  'thread-follower-file-approval-decision': 1, 'thread-follower-submit-user-input': 1};

/** Follow and control existing VS Code conversation owners over private IPC.
 * Args: codexHome is the signed-in account directory.
 * Returns: A reconnecting EventEmitter with owner state and request methods.
 * Raises: Requests reject when the owner disconnects or rejects the control.
 */
export class CodexIPC extends EventEmitter {
  constructor(codexHome = join(homedir(), '.codex')) {
    super(); this.codexHome = codexHome; this.pending = new Map(); this.states = new Map(); this.owners = new Map();
    this.following = new Set(); this.connected = false; this.connect();
  }
  /** Connect and follow known conversations after reconnecting.
   * Args: None.
   * Returns: None.
   * Raises: Requests reject if the IPC owner is unavailable.
   */
  connect() {
    this.buffer = Buffer.alloc(0);
    this.socket = net.connect(join(this.codexHome, 'ipc/ipc.sock'));
    this.socket.on('connect', async () => {
      try {
        const reply = await this.request('initialize', {clientType: 'codex-mobile'});
        this.clientId = reply.result.clientId; this.connected = true;
        for (const id of this.following) this.follow(id);
        this.emit('connected');
      } catch { this.socket.destroy(); }
    });
    this.socket.on('data', data => this.receive(data));
    this.socket.on('error', () => {});
    this.socket.on('close', () => {
      this.connected = false; this.owners.clear(); this.states.clear();
      for (const {reject, timer} of this.pending.values()) {
        clearTimeout(timer); reject(new Error('VS Code disconnected'));
      }
      this.pending.clear(); this.emit('disconnected');
      this.retry = setTimeout(() => this.connect(), 3000);
    });
  }
  /** Write one length-prefixed IPC frame.
   * Args: message is a native protocol envelope.
   * Returns: None.
   * Raises: An error if VS Code is disconnected.
   */
  write(message) {
    if (!this.socket.writable) throw new Error('VS Code is not connected');
    const body = Buffer.from(JSON.stringify(message));
    const header = Buffer.alloc(4); header.writeUInt32LE(body.length);
    this.socket.write(Buffer.concat([header, body]));
  }
  /** Accumulate complete IPC frames.
   * Args: data contains bytes from the private socket.
   * Returns: None.
   * Raises: Malformed frames emit an invalid-frame event.
   */
  receive(data) {
    this.buffer = Buffer.concat([this.buffer, data]);
    while (this.buffer.length >= 4) {
      const size = this.buffer.readUInt32LE();
      if (size > 256 * 1024 * 1024) return this.socket.destroy();
      if (this.buffer.length < size + 4) return;
      const body = this.buffer.subarray(4, size + 4);
      this.buffer = this.buffer.subarray(size + 4);
      try { this.dispatch(JSON.parse(body)); } catch { this.emit('invalid-frame'); }
    }
  }
  /** Route owner responses and conversation updates.
   * Args: message is a decoded IPC frame.
   * Returns: None.
   * Raises: A transport error when responding to discovery.
   */
  dispatch(message) {
    if (message.type === 'client-discovery-request') {
      this.write({type: 'client-discovery-response', requestId: message.requestId,
        response: {canHandle: false}}); return;
    }
    if (message.type === 'response') {
      const pending = this.pending.get(message.requestId); if (!pending) return;
      clearTimeout(pending.timer); this.pending.delete(message.requestId);
      return message.resultType === 'error' ? pending.reject(new Error(message.error)) : pending.resolve(message);
    }
    if (message.method === 'thread-stream-following-status-requested') {
      for (const id of this.following) this.follow(id); return;
    }
    if (message.method === 'thread-stream-state-changed') this.updateState(message);
  }
  /** Apply a local owner snapshot or incremental patch.
   * Args: message contains a native conversation change.
   * Returns: None.
   * Raises: None; incompatible patches request a fresh snapshot.
   */
  updateState(message) {
    const {conversationId, change, hostId} = message.params;
    if (hostId !== 'local') return;
    this.owners.set(conversationId, message.sourceClientId);
    if (change.type === 'snapshot') this.states.set(conversationId, change.conversationState);
    else if (change.type === 'patches' && this.states.has(conversationId)) {
      try { this.states.set(conversationId, applyPatches(this.states.get(conversationId), change.patches)); }
      catch { this.states.delete(conversationId); this.follow(conversationId); }
    }
    this.emit('change', conversationId);
  }
  /** Send a versioned request to a native owner.
   * Args: method and params describe the operation; targetClientId selects the owner.
   * Returns: A promise containing the owner response.
   * Raises: An owner, timeout, or transport error.
   */
  request(method, params, targetClientId) {
    const requestId = randomUUID();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId); reject(new Error('VS Code request timed out; refresh before retrying'));
      }, 30000);
      this.pending.set(requestId, {resolve, reject, timer});
      try { this.write({type: 'request', method, params, requestId, targetClientId,
        sourceClientId: this.clientId ?? 'initializing-client', version: versions[method] ?? 1,
        timeoutMs: 30000}); }
      catch (error) { clearTimeout(timer); this.pending.delete(requestId); reject(error); }
    });
  }
  /** Subscribe to the existing owner of a conversation.
   * Args: conversationId is the original thread ID.
   * Returns: None.
   * Raises: A transport error if the connected socket cannot be written.
   */
  follow(conversationId) {
    this.following.add(conversationId); if (!this.connected) return;
    this.write({type: 'broadcast', method: 'thread-stream-following-changed', version: 1,
      sourceClientId: this.clientId, params: {hostId: 'local', conversationId, following: true}});
  }
  /** Notify VS Code after Codex confirms a saved chat's archive state changed.
   * Args: conversationId identifies the chat; archived is its new state; cwd is its project.
   * Returns: None.
   * Raises: A transport error if the connected socket cannot be written.
   */
  archivalChanged(conversationId, archived, cwd) {
    if (!this.connected) return;
    this.write({type: 'broadcast', method: archived ? 'thread-archived' : 'thread-unarchived',
      version: archived ? 2 : 1, sourceClientId: this.clientId,
      params: {hostId: 'local', conversationId, cwd}});
  }
  /** Discover the VS Code owner of an existing conversation.
   * Args: conversationId is the original thread ID.
   * Returns: A promise containing the owner ID or null when none is discoverable.
   * Raises: An error for failed discovery other than an absent owner.
   */
  async owner(conversationId) {
    if (!this.connected) return null;
    try {
      const reply = await this.request('thread-owner-discovery', {hostId: 'local', conversationId});
      this.owners.set(conversationId, reply.handledByClientId); return reply.handledByClientId;
    } catch (error) { if (error.message === 'no-client-found') return null; throw error; }
  }
  /** Send a control through the current conversation owner.
   * Args: id identifies the conversation; method and params are native controls.
   * Returns: A promise containing the confirmed owner result.
   * Raises: An error if the owner is missing or rejects the request.
   */
  async control(id, method, params = {}) {
    const owner = await this.owner(id); if (!owner) throw new Error('This chat has no VS Code owner');
    return (await this.request(method, {conversationId: id, ...params}, owner)).result;
  }
}
