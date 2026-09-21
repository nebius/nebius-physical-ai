// Translate Mac-owned conversations into the shared browser's app-server vocabulary.
/** Project approvals with identities unique across conversation owners.
 * Args: ipc is the VS Code follower; approvals contains app-server requests.
 * Returns: A list retaining the original request IDs privately for replies.
 * Raises: None.
 */
export function pendingRequests(ipc, approvals) {
  const requests = [...approvals.values()].map(request => ({...request, native: false}));
  for (const [threadId, state] of ipc.states) {
    for (const request of state.requests ?? []) {
      requests.push({...request, native: true, params: {...request.params, threadId}});
    }
  }
  return requests.map(request => ({...request, originalId: request.id,
    id: `${request.native ? 'native' : 'app'}:${request.params.threadId}:${request.id}`}));
}

function thread(chat) {
  return {id: chat.id, name: chat.title, preview: chat.title, cwd: chat.cwd,
    model: chat.model, reasoningEffort: chat.effort, serviceTier: chat.serviceTier,
    mode: chat.mode, updatedAt: chat.updatedAt, owner: chat.owner,
    status: {type: chat.active || chat.status === 'active' ? 'active' : 'idle'}};
}

function item(message) {
  if (message.role === 'user') return {id: message.id, type: 'userMessage',
    clientUserMessageId: message.clientId,
    content: [{type: 'text', text: message.text || ''},
      ...(message.images || []).map(url => ({type: 'image', url}))]};
  if (message.role === 'assistant') return {id: message.id, type: 'agentMessage',
    text: message.text || '', phase: message.phase};
  return {id: message.id, type: 'commandExecution', command: message.title || 'Tool',
    aggregatedOutput: message.text || '', status: 'completed'};
}

function turns(chat, params) {
  const end = params.cursor ? Number(params.cursor) : chat.messages.length;
  if (!Number.isSafeInteger(end) || end < 0 || end > chat.messages.length)
    throw new Error('Invalid history cursor');
  const start = Math.max(0, end - 150);
  const latest = end === chat.messages.length;
  const messages = chat.messages.slice(start, end);
  return {data: messages.length || chat.active ? [{
    id: latest && chat.turnId ? chat.turnId : `history:${messages[0]?.id || chat.id}`,
    status: latest && chat.active ? 'inProgress' : 'completed',
    items: messages.map(item),
  }] : [], nextCursor: start ? String(start) : null};
}

async function list(context, params) {
  const rows = await context.handlers.list({archived: params.archived, includeAgents: true});
  const matching = rows.filter(row => !params.searchTerm ||
    row.title.toLowerCase().includes(params.searchTerm.toLowerCase()));
  const start = Number(params.cursor || 0);
  const limit = Number(params.limit || 50);
  if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(limit) || limit < 1)
    throw new Error('Invalid session cursor');
  return {data: matching.slice(start, start + limit).map(thread),
    nextCursor: start + limit < matching.length ? String(start + limit) : null};
}

async function configure(context, params) {
  const current = await context.handlers.read({id: params.threadId});
  const result = await context.handlers.configure({id: params.threadId,
    model: params.model ?? current.model, effort: params.effort ?? current.effort,
    serviceTier: params.serviceTier === undefined ? current.serviceTier : params.serviceTier,
    mode: params.collaborationMode?.mode ?? params.mode ?? current.mode});
  return result;
}

async function answer(context, message) {
  const request = pendingRequests(context.ipc, context.approvals).find(row => row.id === message.id);
  if (!request) throw new Error('This request is no longer pending');
  const kind = request.method;
  const method = kind.includes('requestUserInput') ? 'thread-follower-submit-user-input' :
    kind.includes('fileChange') ? 'thread-follower-file-approval-decision' :
    kind.includes('commandExecution') ? 'thread-follower-command-approval-decision' : null;
  if (request.native && !method) throw new Error('Complete this request in VS Code');
  if (!request.native) {
    context.app.respond(request.originalId, message.result);
    context.approvals.delete(String(request.originalId));
    return {};
  }
  return context.handlers.respond({id: request.params.threadId, requestId: request.originalId,
    method, result: message.result, answer: {requestId: request.originalId,
      ...(kind.includes('requestUserInput') ? {response: message.result} : message.result)}});
}

async function prompt(context, params, steer) {
  const current = await context.handlers.read({id: params.threadId});
  if (steer && (!current.active || current.turnId !== params.expectedTurnId))
    throw new Error('The active turn changed. Refresh before sending again.');
  const text = params.input.filter(part => part.type === 'text').map(part => part.text).join('\n');
  return context.handlers.prompt({id: params.threadId, text, steer,
    images: params.input.filter(part => part.type === 'image').map(part => part.url),
    clientUserMessageId: params.clientUserMessageId});
}

function runtimeOperation(context, method) {
  switch (method) {
    case 'runtime/status': return {ownedActive: [...context.ownThreads.values()].some(value => value.active)};
    case 'runtime/prepareUpdate': {
      if (context.mutations || [...context.ownThreads.values()].some(value => value.active))
        throw new Error('Wait for mobile-owned turns to finish before updating local chat.');
      context.upgrading = true;
      return {prepared: true};
    }
    case 'runtime/cancelUpdate': context.upgrading = false; return {};
    default: throw new Error('Unsupported runtime operation');
  }
}

/** Translate a shared chat RPC call to the native owner or local engine.
 * Args: context holds connected owners; message is the validated RPC envelope.
 * Returns: The shared browser protocol result.
 * Raises: An error for unsupported controls, stale turns, or owner failures.
 */
export async function protocolCall(context, message) {
  if (!message.method) return answer(context, message);
  if (message.method.startsWith('runtime/')) return runtimeOperation(context, message.method);
  const params = message.params || {};
  const handlers = context.handlers;
  switch (message.method) {
    case 'initialize': return {};
    case 'model/list': return {data: (await handlers.models()).data, nextCursor: null};
    case 'collaborationMode/list': return {data: (await handlers.models()).modes};
    case 'thread/list': return list(context, params);
    case 'thread/read':
    case 'thread/resume': return {thread: thread(await handlers.read({id: params.threadId}))};
    case 'thread/start': {
      const result = await handlers.create(params);
      return {thread: thread(await handlers.read({id: result.id}))};
    }
    case 'thread/turns/list': return turns(await handlers.read({
      id: params.threadId, visibleCount: Number.MAX_SAFE_INTEGER}), params);
    case 'thread/settings/update': return configure(context, params);
    case 'thread/open': return handlers.openInVSCode({id: params.threadId});
    case 'turn/start': return prompt(context, params, false);
    case 'turn/steer': return prompt(context, params, true);
    case 'turn/interrupt': return handlers.stop({id: params.threadId, turnId: params.turnId});
    default: throw new Error('Unsupported native control method');
  }
}
