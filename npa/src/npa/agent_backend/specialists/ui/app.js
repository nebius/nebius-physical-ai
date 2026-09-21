/* Display authenticated specialist state without persisting credentials or rendering model HTML. */
const el = id => document.getElementById(id);
let credential = '', selected = '', selectedProfile = '', timer;
async function api(path, body) {
  const response = await fetch('/api/' + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: {Authorization: 'Bearer ' + credential, 'Content-Type': 'application/json'},
    ...(body === undefined ? {} : {body: JSON.stringify(body)})
  });
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.headers.get('content-type').includes('application/json') ? response.json() : response.text();
}
function node(tag, text, className = '') {
  const item = document.createElement(tag); item.textContent = text; item.className = className; return item;
}
function button(label, action) {
  const item = node('button', label, 'secondary'); item.type = 'button';
  item.onclick = () => safely(action); return item;
}
async function safely(action) {
  try { el('error').textContent = ''; await action(); }
  catch (error) { el('error').textContent = error.message; }
}
function profiles(items) {
  el('profiles').replaceChildren();
  const previous = el('specialist').value;
  el('specialist').replaceChildren(new Option('Automatic / default', 'auto'));
  for (const profile of items) {
    el('specialist').add(new Option(profile.name, profile.name));
    const card = node('article', '');
    card.append(node('h3', profile.name), node('p', profile.model, 'badge'), node('p', profile.description));
    const seen = profile.worker ? new Date(profile.worker.seen * 1000).toLocaleTimeString() : 'not observed';
    const identity = profile.worker ? 'Worker ' + profile.worker.pid : 'Worker';
    card.append(node('p', identity + ' · last seen: ' + seen, 'muted'));
    card.append(button(profile.paused ? 'Resume specialist' : 'Pause specialist', async () => {
      await api('profiles/' + encodeURIComponent(profile.name) + '/pause', {paused: !profile.paused}); await refresh();
    }));
    el('profiles').append(card);
  }
  el('specialist').value = previous;
}
async function refresh() {
  const state = await api('status'); profiles(state.profiles); el('tasks').replaceChildren();
  for (const task of state.tasks) {
    const row = node('div', '', 'task'); row.tabIndex = 0; row.setAttribute('role', 'button');
    row.append(node('span', task.goal, 'goal'), node('span', task.profile + ' · ' + (task.paused ? 'paused' : task.status), 'badge'));
    row.onclick = () => safely(() => detail(task.id));
    row.onkeydown = event => { if (event.key === 'Enter') row.click(); };
    el('tasks').append(row);
  }
  if (selected) await detail(selected);
}
async function detail(id) {
  selected = id; const task = await api('tasks/' + encodeURIComponent(id)); selectedProfile = task.profile;
  el('detail').hidden = false; el('task-title').textContent = task.profile + ' · ' + task.status;
  el('result').textContent = task.result || task.error || 'Work is in progress.';
  el('events').textContent = JSON.stringify({route: task.route, calls: task.calls, events: task.events}, null, 2);
  activity(task.events);
  el('patch').textContent = await api('tasks/' + encodeURIComponent(id) + '/patch') || 'No source edits recorded.';
  el('controls').replaceChildren();
  if (['queued', 'running'].includes(task.status)) {
    el('controls').append(button(task.paused ? 'Resume task' : 'Pause task', async () => {
      await api('tasks/' + encodeURIComponent(id) + '/pause', {paused: !task.paused}); await refresh();
    }));
  }
  el('followup').hidden = task.status !== 'completed';
  if (!['completed', 'cancelled'].includes(task.status)) {
    el('controls').append(button('Cancel task', async () => {
      await api('tasks/' + encodeURIComponent(id) + '/cancel', {}); await refresh();
    }));
  }
  const pending = task.calls.find(call => call.status === 'started');
  if (pending && !el('call-id').value) el('call-id').value = pending.call_id;
}
function activity(events) {
  el('timeline').replaceChildren();
  for (const event of events) {
    let label = event.type;
    if (event.type === 'model') {
      const cached = event.usage.cached_tokens;
      const cache = cached === undefined ? 'cache unreported' : cached + ' cached prompt tokens';
      label = event.model + ' · ' + (event.usage.total_tokens ?? 'unreported') + ' tokens · ' + cache;
      if (event.accepted === false) label += ' · response rejected · ' + (event.finish_reason || 'finish reason unreported');
    }
    if (event.type === 'tool') {
      const result = event.result;
      label = (result.operation || event.name) + ' · ' + (result.ok === false ? 'failed' : 'completed');
      if (result.returncode !== undefined) label += ' · exit ' + result.returncode;
    }
    if (event.type === 'needs_attention') label = 'Needs attention · ' + event.error;
    el('timeline').append(node('div', label, 'event'));
  }
  if (!events.length) el('timeline').append(node('p', 'No recorded activity yet.', 'muted'));
}
el('connect').onsubmit = event => { event.preventDefault(); safely(async () => {
  credential = el('token').value; el('token').value = ''; await refresh();
  el('connection').textContent = 'Connected'; clearInterval(timer);
  timer = setInterval(() => safely(refresh), 2000);
}); };
el('submit').onsubmit = event => { event.preventDefault(); safely(async () => {
  el('send').disabled = true;
  try {
    const task = await api('tasks', {goal: el('goal').value, specialist: el('specialist').value, task_id: crypto.randomUUID()});
    el('goal').value = ''; await detail(task.id); await refresh();
  } finally { el('send').disabled = false; }
}); };
el('followup').onsubmit = event => { event.preventDefault(); safely(async () => {
  const task = await api('tasks', {goal: el('followup-goal').value, specialist: selectedProfile, parent_id: selected, task_id: crypto.randomUUID()});
  el('followup-goal').value = ''; await detail(task.id); await refresh();
}); };
el('reconcile').onsubmit = event => { event.preventDefault(); safely(async () => {
  await api('tasks/' + encodeURIComponent(selected) + '/reconcile', {
    call_id: el('call-id').value, result: JSON.parse(el('verified-result').value), retry: el('retry').checked
  });
  await refresh();
}); };
