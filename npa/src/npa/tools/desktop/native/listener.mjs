// Keep one private engine listener without replacing another running instance.
import net from 'node:net';
import {lstatSync, unlinkSync} from 'node:fs';

/** Remove only a stale Unix socket before binding.
 * Args: path is the private socket path.
 * Returns: a promise resolved when the path is safe to bind.
 * Raises: an error for a live listener, a non-socket path, or failed inspection.
 */
export async function removeStaleSocket(path) {
  let info;
  try { info = lstatSync(path); }
  catch (error) { if (error.code === 'ENOENT') return; throw error; }
  if (!info.isSocket()) throw new Error('The engine socket path is not a socket.');
  await new Promise((resolve, reject) => {
    const probe = net.connect(path);
    probe.once('connect', () => {
      probe.destroy();
      reject(new Error('Another local chat engine already owns this socket.'));
    });
    probe.once('error', error => {
      if (error.code === 'ECONNREFUSED' || error.code === 'ENOENT') resolve();
      else reject(error);
    });
  });
  try {
    const current = lstatSync(path);
    if (current.ino !== info.ino || !current.isSocket())
      throw new Error('The engine socket changed while checking it.');
    unlinkSync(path);
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
}
