// Reads local Codex metadata and renders public conversation fields for the UI.
import {DatabaseSync} from 'node:sqlite';
import {homedir} from 'node:os';
import {join} from 'node:path';
import {readFile, stat} from 'node:fs/promises';

/** Read Codex metadata and rollout logs without modifying account state.
 * Args: codexHome is the existing signed-in account directory.
 * Returns: A read-only session store.
 * Raises: An error if the installed Codex database schema is unavailable.
 */
export class SessionStore {
  constructor(codexHome = join(homedir(), '.codex')) {
    this.db = new DatabaseSync(join(codexHome, 'state_5.sqlite'), {readOnly: true});
    this.cache = new Map();
  }
  /** List saved conversations in recent activity order.
   * Args: archived selects the archive; includeAgents includes child conversations.
   * Returns: Public metadata rows.
   * Raises: A SQLite error if the schema is unsupported.
   */
  list(archived = false, includeAgents = false) {
    const rows = this.db.prepare(`SELECT id, title, name, preview, cwd, updated_at, model, source,
      reasoning_effort, archived FROM threads WHERE archived = ? ORDER BY updated_at DESC`).all(Number(archived));
    return rows.filter(row => includeAgents || !row.source.startsWith('{'))
      .map(row => ({id: row.id, title: row.name || row.title || row.preview || 'New chat',
        cwd: row.cwd, updatedAt: row.updated_at, model: row.model, source: row.source,
        effort: row.reasoning_effort, archived: Boolean(row.archived)}));
  }
  /** Read one saved conversation record.
   * Args: id is an existing conversation ID.
   * Returns: The private database row.
   * Raises: An error if the conversation is unknown.
   */
  get(id) {
    const row = this.db.prepare('SELECT * FROM threads WHERE id = ?').get(id);
    if (!row) throw new Error('Chat not found'); return row;
  }
  /** Read and cache visible messages from the saved rollout.
   * Args: id is an existing conversation ID.
   * Returns: A promise containing visible messages in log order.
   * Raises: An error if the rollout cannot be read.
   */
  async history(id) {
    const row = this.get(id); const info = await stat(row.rollout_path);
    const cached = this.cache.get(id);
    if (cached?.size === info.size && cached?.mtime === info.mtimeMs) return cached.messages;
    const messages = [];
    for (const line of (await readFile(row.rollout_path, 'utf8')).split('\n')) {
      try { const message = rolloutMessage(JSON.parse(line)); if (message) messages.push(message); } catch { /* Partial final lines are retried on the next refresh. */ }
    }
    this.cache.set(id, {size: info.size, mtime: info.mtimeMs, messages});
    if (this.cache.size > 12) this.cache.delete(this.cache.keys().next().value);
    return messages;
  }
}

function rolloutMessage(record) {
  const item = record.payload;
  if (record.type !== 'response_item' || !item) return null;
  if (item.type === 'message' && ['user', 'assistant'].includes(item.role)) {
    const text = (item.content ?? []).filter(part => part.type.includes('text')).map(part => part.text).join('\n');
    if (item.role === 'user' && (/^# AGENTS\.md instructions/.test(text) || /^<environment_context>/.test(text))) return null;
    return {id: item.id || record.timestamp, role: item.role, text, phase: item.phase,
      images: (item.content ?? []).filter(part => part.type === 'input_image').map(part => part.image_url)};
  }
  if (item.type === 'function_call') return {id: item.call_id, role: 'tool', title: item.name, text: item.arguments};
  if (item.type === 'function_call_output') return {id: item.call_id + '-output', role: 'tool', title: 'Result', text: stringify(item.output)};
  if (item.type === 'custom_tool_call') return {id: item.call_id, role: 'tool', title: item.name, text: item.input};
  if (item.type === 'custom_tool_call_output') return {id: item.call_id + '-output', role: 'tool', title: 'Result', text: stringify(item.output)};
  return null;
}

function stringify(value) { return typeof value === 'string' ? value : JSON.stringify(value, null, 2); }

/** Flatten the native history islands in display order.
 * Args: state is the followed VS Code conversation.
 * Returns: The known native turns.
 * Raises: None.
 */
export function liveTurns(state) {
  const history = state?.turnHistory?.history;
  if (!history) return state?.turns ?? [];
  return history.islands.flatMap(island => island.entries.map(entry => history.entitiesByKey[entry.key])).filter(Boolean);
}

/** Project native turns while removing duplicated steering acknowledgements.
 * Args: state is the followed VS Code conversation.
 * Returns: Visible user, assistant, and tool messages.
 * Raises: None.
 */
export function liveMessages(state) {
  return liveTurns(state).flatMap(turn => {
    const messages = [];
    const inputKey = input => JSON.stringify((input ?? []).map(part => part.type === 'text' ? ['text', part.text] : [part.type, part.url]));
    const openingMessage = turn.items?.some(item => item.type === 'userMessage' &&
      (item.clientId && turn.params?.clientUserMessageId ? item.clientId === turn.params.clientUserMessageId : inputKey(item.content) === inputKey(turn.params?.input)));
    if (turn.params?.input?.length && !openingMessage) messages.push({id: turn.turnId + '-user', role: 'user', clientId: turn.params.clientUserMessageId,
      text: turn.params.input.filter(x => x.type === 'text').map(x => x.text).join('\n'),
      images: turn.params.input.filter(x => x.type === 'image').map(x => x.url)});
    for (const item of turn.items ?? []) {
      if (item.type === 'steeringUserMessage' && turn.items.some(other => other.type === 'userMessage' &&
        (other.id === item.serverUserMessageId || (item.clientUserMessageId && other.clientId === item.clientUserMessageId)))) continue;
      const message = itemMessage(item); if (message) messages.push(message);
    }
    return messages;
  });
}

/** Project one native item into safe display fields.
 * Args: item is a native conversation item.
 * Returns: A visible message or null for unsupported internal items.
 * Raises: None.
 */
export function itemMessage(item) {
  if (item.type === 'agentMessage') return {id: item.id, role: 'assistant', text: item.text, phase: item.phase};
  if (item.type === 'userMessage') return {id: item.id, role: 'user', clientId: item.clientId,
    text: (item.content ?? []).filter(x => x.type === 'text').map(x => x.text).join('\n'),
    images: (item.content ?? []).filter(x => x.type === 'image').map(x => x.url)};
  if (item.type === 'steeringUserMessage') return {id: item.id, role: 'user', clientId: item.clientUserMessageId,
    text: item.text || (item.input ?? []).filter(x => x.type === 'text').map(x => x.text).join('\n'),
    images: (item.input ?? []).filter(x => x.type === 'image').map(x => x.url)};
  if (item.type === 'commandExecution') return {id: item.id, role: 'tool', title: item.command,
    text: item.aggregatedOutput || item.status, status: item.status};
  if (item.type === 'fileChange') return {id: item.id, role: 'tool', title: 'File changes',
    text: (item.changes ?? []).map(x => `${x.path}\n${x.diff ?? ''}`).join('\n')};
  if (item.type === 'mcpToolCall') return {id: item.id, role: 'tool', title: item.tool, text: stringify(item.result ?? item.arguments)};
  return null;
}
