// Apply Codex metadata operations without resuming conversations or replacing owners.
import {requireUnowned} from './ownership.mjs';

/** Rename, archive, or restore one saved conversation using Codex's own API.
 * Args: context holds the connected engines; params identify the chat and action.
 * Returns: Confirmation after the operation succeeds.
 * Raises: An error for active turns, unsaved chats, existing writers, or RPC failures.
 */
export async function manageThread(context, params) {
  const {getRow, ownThreads, ipc, app, config, notify} = context;
  const row = getRow(params.id);
  if (row.unsaved) throw new Error('Send the first message before managing this chat');
  const own = ownThreads.get(params.id);
  if (params.action !== 'rename') {
    if (own?.active || ipc.states.get(params.id)?.threadRuntimeStatus?.type === 'active')
      throw new Error('Wait for Codex to finish or stop the turn before archiving');
    if (!own) {
      if (await ipc.owner(params.id)) throw new Error('Close this chat in VS Code before archiving it from mobile');
      await requireUnowned(config.codexHome, params.id);
    }
  }
  if (params.action === 'rename' && row.archived) throw new Error('Restore this chat before renaming it');
  await app.start();
  const method = params.action === 'rename' ? 'thread/name/set' : `thread/${params.action}`;
  await app.request(method, {threadId: params.id, ...(params.action === 'rename' ? {name: params.name} : {})});
  if (params.action !== 'rename') {
    ownThreads.delete(params.id);
    ipc.states.delete(params.id);
    ipc.archivalChanged(params.id, params.action === 'archive', row.cwd);
  }
  notify(); return {ok: true};
}
