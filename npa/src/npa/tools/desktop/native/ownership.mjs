// Refuse a second writer when an existing Codex process owns the session lock.
import {existsSync} from 'node:fs';
import {join} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

/** Check for a live writer before resuming an unowned conversation.
 * Args: codexHome is the account directory; id is the existing thread ID.
 * Returns: a promise resolved when no process holds the writer file open.
 * Raises: an error if a writer exists or process inspection fails.
 */
export async function requireUnowned(codexHome, id) {
  const path = join(codexHome, 'thread-writer-locks', id + '.lock');
  if (!existsSync(path)) return;
  let output;
  try { output = await promisify(execFile)('/usr/sbin/lsof', ['-t', '--', path]); }
  catch (error) {
    if (error.code === 1 && !error.stdout.trim() && !error.stderr.trim()) return;
    throw new Error('Cannot verify this session owner. Reconnect to VS Code before sending.');
  }
  if (output.stdout.trim())
    throw new Error('This session has another Codex writer. Reconnect to its owner before sending.');
}
