// Let native VS Code views follow and control the existing mobile chat writer.
import {ownerState} from './owner-state.mjs';

const versions = {'thread-owner-discovery': 1, 'thread-follower-start-turn': 2,
  'thread-follower-steer-turn': 1, 'thread-follower-interrupt-turn': 4,
  'thread-follower-load-complete-history': 1, 'thread-follower-update-thread-settings': 2,
  'thread-follower-command-approval-decision': 1, 'thread-follower-file-approval-decision': 1,
  'thread-follower-permissions-request-approval-response': 1, 'thread-follower-submit-user-input': 1};
const approvalMethods = {
  'thread-follower-command-approval-decision': 'item/commandExecution/requestApproval',
  'thread-follower-file-approval-decision': 'item/fileChange/requestApproval',
  'thread-follower-permissions-request-approval-response': 'item/permissions/requestApproval',
  'thread-follower-submit-user-input': 'item/tool/requestUserInput',
};

/** Publish mobile-owned conversations and route follower controls to their writer.
 * Args: context provides IPC, the app-server, owned threads, approvals and mutation serialization.
 * Returns: An owner handler for CodexIPC.
 * Raises: Controls reject when the thread, turn, approval, or protocol version changes.
 */
export class NativeOwner {
  constructor(context, threadId = null) {
    this.context = context; this.followers = new Map(); this.revisions = new Map();
    this.threadId = threadId;
    this.refreshing = false; this.refreshAgain = false; this.snapshots = new Map(); this.dirty = new Set();
  }
  /** Check whether this runtime owns a chat.
   * Args: id is a conversation ID.
   * Returns: Whether this engine owns its writer.
   * Raises: None.
   */
  owns(id) { return (!this.threadId || this.threadId === id) && this.context.ownThreads.has(id); }
  /** Accept only supported local controls for this writer.
   * Args: request is the native IPC request envelope.
   * Returns: Whether discovery may route this request here.
   * Raises: None.
   */
  accepts(request) {
    return (request?.hostId ?? request?.params?.hostId ?? 'local') === 'local' &&
      versions[request?.method] !== undefined && request.version === versions[request.method] &&
      this.owns(request.params?.conversationId);
  }
  /** Track subscribed views and immediately supply their current history.
   * Args: message is a native following-change broadcast.
   * Returns: None.
   * Raises: None; failed refreshes are retried on the next runtime notification.
   */
  following(message) {
    const {conversationId: id, hostId, following} = message.params;
    if (hostId !== 'local' || !this.owns(id) || message.version !== 1) return;
    const followers = this.followers.get(id) ?? new Set();
    if (following) followers.add(message.sourceClientId); else followers.delete(message.sourceClientId);
    this.followers.set(id, followers);
    if (following && !this.snapshots.has(id)) {
      this.dirty.add(id); this.refresh(); return;
    }
    if (following) this.sendSnapshot(id);
    this.changed(id);
  }
  /** Remove a disconnected follower from future snapshots.
   * Args: status describes a native IPC client connection.
   * Returns: None.
   * Raises: None.
   */
  clientStatus(status) {
    if (status.status === 'disconnected')
      for (const followers of this.followers.values()) followers.delete(status.clientId);
  }
  /** Coalesce refreshes while retaining changes arriving during an in-flight read.
   * Args: None.
   * Returns: None.
   * Raises: None; connection and ownership changes are handled by the next refresh.
   */
  refresh() {
    this.refreshAgain = true;
    if (this.refreshing) return;
    this.refreshing = true;
    this.refreshPromise = this.refreshViews().catch(error => console.error('Native chat refresh failed:', error.message))
      .finally(() => { this.refreshing = false; });
  }
  /** Schedule publication after an owned thread changes.
   * Args: id is the changed owned thread.
   * Returns: None.
   * Raises: None.
   */
  changed(id) {
    this.dirty.add(id);
    if (this.refreshTimer) return;
    this.refreshTimer = setTimeout(() => { this.refreshTimer = null; this.refresh(); }, 100);
  }
  async refreshViews() {
    while (this.refreshAgain) {
      this.refreshAgain = false;
      const dirty = [...this.dirty]; this.dirty.clear();
      for (const id of dirty) {
        if (!this.owns(id)) { this.followers.delete(id); this.snapshots.delete(id); continue; }
        if (this.context.ipc.connected) await this.publish(id);
      }
    }
  }
  /** Send a complete, revisioned snapshot without opening another writer.
   * Args: id identifies an owned conversation.
   * Returns: The published revision.
   * Raises: An app-server or IPC error if the owner disconnects.
   */
  async publish(id) {
    const owned = this.context.ownThreads.get(id);
    if (!owned) throw new Error('This chat has another owner');
    const {thread} = await this.context.app.request('thread/read', {threadId: id, includeTurns: true});
    if (this.context.ownThreads.get(id) !== owned) throw new Error('This chat has another owner');
    const requests = [...this.context.approvals.values()].filter(request => request.params?.threadId === id);
    const state = ownerState(thread, owned.settings, requests);
    const revision = (this.revisions.get(id) ?? 0) + 1;
    this.revisions.set(id, revision);
    this.snapshots.set(id, {type: 'snapshot', revision, conversationState: state});
    this.sendSnapshot(id);
    return revision;
  }
  sendSnapshot(id) {
    if (!this.followers.get(id)?.size || !this.context.ipc.connected) return;
    this.context.ipc.write({type: 'broadcast', method: 'thread-stream-state-changed', version: 11,
      sourceClientId: this.context.ipc.clientId, targetClientIds: [...(this.followers.get(id) ?? [])],
      params: {hostId: 'local', conversationId: id, change: this.snapshots.get(id)}});
  }
  /** Execute a native follower action through the single owning app-server.
   * Args: message is a validated native request.
   * Returns: The native control result.
   * Raises: A protocol or ownership error; uncertain actions are never replayed.
   */
  async handle(message) {
    if (message.method === 'thread-owner-discovery') return {supportsUntrustedAppInput: false};
    const id = message.params.conversationId;
    return this.context.mutate(id, async () => {
      if (!this.owns(id)) throw new Error('This chat has another owner');
      if (message.method === 'thread-follower-load-complete-history') return {revision: await this.publish(id)};
      const result = await this.control(message.method, id, message.params);
      this.context.notify(); this.changed(id); return result;
    });
  }
  async control(method, id, params) {
    const {app, ownThreads} = this.context;
    if (approvalMethods[method]) return this.answer(method, id, params);
    if (method === 'thread-follower-start-turn') {
      if (params.turnStart?.context?.responseItems?.length)
        throw new Error('Additional app input is not supported by this owner');
      const result = await app.request('turn/start', {...params.turnStart.request, threadId: id});
      Object.assign(ownThreads.get(id), {active: result.turn.status === 'inProgress', turnId: result.turn.id});
      return {result};
    }
    if (method === 'thread-follower-steer-turn') return {result: await app.request('turn/steer',
      {threadId: id, input: params.input, expectedTurnId: ownThreads.get(id).turnId,
        clientUserMessageId: params.clientUserMessageId})};
    if (method === 'thread-follower-interrupt-turn') {
      const turnId = ownThreads.get(id).turnId;
      if (!turnId || (params.expectedTurnId && params.expectedTurnId !== turnId))
        throw new Error('The active turn changed. Refresh before stopping it.');
      await app.request('turn/interrupt', {threadId: id, turnId});
      return {ok: true, interruptedTurnId: turnId};
    }
    if (method === 'thread-follower-update-thread-settings') return this.updateSettings(id, params);
    throw new Error('Unsupported native follower action');
  }
  async updateSettings(id, params) {
    const own = this.context.ownThreads.get(id), condition = params.condition;
    if (params.activeTurnId && params.activeTurnId !== own.turnId)
      throw new Error('The active turn changed. Refresh before changing its settings.');
    if (condition && (own.settings.effort !== condition.ifEffortEquals ||
        (condition.ifModelEquals != null && own.settings.model !== condition.ifModelEquals)))
      return {applied: false};
    await this.context.app.request('thread/settings/update', {...params.threadSettings, threadId: id});
    Object.assign(own.settings, params.threadSettings); return {applied: true};
  }
  answer(method, id, params) {
    const request = this.context.approvals.get(String(params.requestId));
    if (!request || request.params?.threadId !== id || request.method !== approvalMethods[method])
      throw new Error('This approval is no longer pending');
    const result = method.endsWith('approval-decision') ? {decision: params.decision} : params.response;
    this.context.app.respond(request.id, result);
    this.context.approvals.delete(String(request.id)); return {ok: true};
  }
}
