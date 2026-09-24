// Owns new and inactive Codex sessions through the installed app-server protocol.
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {EventEmitter} from 'node:events';

/** Run the installed Codex app-server for new or unowned conversations.
 * Args: binary is the local Codex executable; codexHome is the account directory.
 * Returns: An EventEmitter carrying app-server notifications and pending approvals.
 * Raises: Requests reject on protocol errors or process failure.
 */
export class AppServer extends EventEmitter {
  constructor(binary, codexHome) { super(); this.binary = binary; this.codexHome = codexHome; this.pending = new Map(); this.nextId = 0; }
  /** Start or reuse the account app-server.
   * Args: None.
   * Returns: The shared initialization promise.
   * Raises: A startup or protocol error.
   */
  async start() {
    if (this.ready) return this.ready;
    this.ready = this.initialize(); return this.ready;
  }
  /** Initialize the installed binary and observe process failures.
   * Args: None.
   * Returns: A promise resolved after protocol initialization.
   * Raises: A startup or protocol error.
   */
  async initialize() {
    this.child = spawn(this.binary, ['app-server'], {stdio: ['pipe', 'pipe', 'pipe'], env: {...process.env, ...(this.codexHome ? {CODEX_HOME: this.codexHome} : {})}});
    this.child.stderr.on('data', () => {});
    createInterface({input: this.child.stdout}).on('line', line => {
      try { this.dispatch(JSON.parse(line)); } catch { /* Non-protocol output is ignored. */ }
    });
    this.child.on('exit', () => this.failed(new Error('Codex stopped')));
    this.child.on('error', error => this.failed(error));
    this.child.stdin.on('error', error => this.failed(error));
    await this.request('initialize', {clientInfo: {name: 'codex_mobile', title: 'Codex Mobile', version: '1.0.0'},
      capabilities: {experimentalApi: true}});
    this.child.stdin.write(JSON.stringify({method: 'initialized'}) + '\n');
  }
  /** Reject pending requests when the app-server stops.
   * Args: error is the transport or process failure.
   * Returns: None.
   * Raises: None.
   */
  failed(error) {
    this.ready = null;
    for (const request of this.pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    this.pending.clear();
    this.emit('failure', error);
  }
  /** Deliver an app-server notification or response.
   * Args: message is a parsed protocol frame.
   * Returns: None.
   * Raises: None.
   */
  dispatch(message) {
    if (message.method) return this.emit('event', message);
    const pending = this.pending.get(message.id); if (!pending) return;
    clearTimeout(pending.timer); this.pending.delete(message.id);
    if (message.error) pending.reject(new Error(message.error.message)); else pending.resolve(message.result);
  }
  /** Send one app-server request without replaying failures.
   * Args: method and params define the Codex operation.
   * Returns: A promise containing the result.
   * Raises: A protocol or transport error.
   */
  request(method, params) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error('Codex request timed out; refresh before retrying')); }, 60000);
      this.pending.set(id, {resolve, reject, timer});
      this.child.stdin.write(JSON.stringify({id, method, params}) + '\n');
    });
  }
  /** Answer an explicitly approved app-server request.
   * Args: id is the original request ID; result is the validated answer.
   * Returns: None.
   * Raises: A transport error.
   */
  respond(id, result) { this.child.stdin.write(JSON.stringify({id, result}) + '\n'); }
}
