// Publish each mobile-owned chat through a native IPC identity.
import {CodexIPC} from './ipc.mjs';
import {NativeOwner} from './owner.mjs';

/** Coordinate native views while preserving their existing conversation writers.
 * Args: context contains the shared engine; codexHome selects the native IPC router.
 * Returns: A registry of conversation-specific owner identities.
 * Raises: None.
 */
export class SharedOwners {
  constructor(context, codexHome) { this.context = context; this.codexHome = codexHome; this.owners = new Map(); }
  /** Start or refresh only the owner whose conversation changed.
   * Args: id is the changed conversation ID.
   * Returns: None.
   * Raises: None.
   */
  changed(id) {
    if (!this.context.ownThreads.has(id)) return;
    let owner = this.owners.get(id);
    if (!owner) {
      const ipc = new CodexIPC(this.codexHome);
      owner = new NativeOwner({...this.context, ipc}, id);
      ipc.ownerHandler = owner;
      ipc.on('connected', () => {
        ipc.write({type: 'broadcast', method: 'thread-stream-following-status-requested', version: 1,
          sourceClientId: ipc.clientId, params: {hostId: 'local', conversationId: id}});
        owner.changed(id);
      });
      this.owners.set(id, owner);
    }
    owner.changed(id);
  }
  /** Check whether a native view is following this mobile-owned chat.
   * Args: id is the conversation ID.
   * Returns: Whether a follower is subscribed.
   * Raises: None.
   */
  followed(id) { return Boolean(this.owners.get(id)?.followers.get(id)?.size); }
  /** Prepare the first snapshot before launching a native view of an owned chat.
   * Args: id is the conversation about to open in VS Code.
   * Returns: A promise resolved after its shared history is available.
   * Raises: An app-server error if the owning engine is unavailable.
   */
  async prepare(id) { await this.owners.get(id)?.publish(id); }
  /** Remove a native identity after its writer has released the chat.
   * Args: id is the released conversation ID.
   * Returns: None.
   * Raises: None.
   */
  remove(id) {
    const owner = this.owners.get(id);
    if (!owner) return;
    clearTimeout(owner.refreshTimer); owner.context.ipc.dispose(); this.owners.delete(id);
  }
  /** Drop identities after the owning engine exits.
   * Args: None.
   * Returns: None.
   * Raises: None.
   */
  clear() { for (const id of this.owners.keys()) this.remove(id); }
}
