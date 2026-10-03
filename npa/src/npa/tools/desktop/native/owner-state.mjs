// Project a mobile-owned app-server conversation into the native follower state.

/** Build a complete native snapshot from the current shared runtime.
 * Args: thread is the app-server thread; settings are its last confirmed settings;
 * requests are outstanding approvals for this thread.
 * Returns: Native conversation state with stable turn and item identities.
 * Raises: None.
 */
export function ownerState(thread, settings, requests) {
  const turns = (thread.turns ?? []).map(turn => nativeTurn(thread, settings, turn));
  return {
    id: thread.id, sessionId: thread.sessionId ?? thread.id, hostId: 'local',
    title: thread.name || thread.preview || 'New chat', cwd: thread.cwd,
    source: thread.source, originator: thread.originator, threadSource: thread.threadSource,
    ephemeral: Boolean(thread.ephemeral), forkedFromId: thread.forkedFromId ?? null,
    createdAt: thread.createdAt * 1000, updatedAt: thread.updatedAt * 1000,
    recencyAt: (thread.recencyAt ?? thread.updatedAt) * 1000,
    mode: settings.collaborationMode?.mode || thread.mode || 'default',
    threadStartKind: thread.threadStartKind || 'default', modelProvider: thread.modelProvider,
    latestModel: settings.model || thread.model || '',
    latestReasoningEffort: settings.effort ?? settings.reasoningEffort ?? thread.reasoningEffort,
    latestThreadSettings: settings,
    latestCollaborationMode: settings.collaborationMode ?? {mode: 'default', settings: {
      model: settings.model || thread.model || '', reasoning_effort: settings.effort ?? null,
      developer_instructions: null}},
    previousTurnModel: null, hasUnreadTurn: false, unreadMessageCount: 0,
    rolloutPath: thread.path || '', gitInfo: thread.gitInfo, resumeState: 'resumed',
    threadRuntimeStatus: thread.status, latestTokenUsageInfo: null,
    currentPermissions: thread.currentPermissions ?? null,
    environments: thread.environments ?? [], workspaceKind: 'project',
    turns: [], turnHistory: canonicalHistory(turns), historyMode: thread.historyMode,
    turnsPagination: {olderCursor: null, oldestLoadedTurnId: turns[0]?.turnId ?? null,
      isLoadingOlder: false, hasLoadedOldest: true}, requests,
  };
}

function nativeTurn(thread, settings, turn) {
  const opening = turn.items?.find(item => item.type === 'userMessage');
  return {turnId: turn.id, status: turn.status, error: turn.error ?? null,
    turnStartedAtMs: turn.startedAt == null ? null : turn.startedAt * 1000,
    finalAssistantStartedAtMs: turn.completedAt == null ? null : turn.completedAt * 1000,
    durationMs: turn.durationMs ?? null, diff: null, hookRuns: [],
    params: {...settings, threadId: thread.id, cwd: thread.cwd,
      input: opening?.content ?? [], clientUserMessageId: opening?.clientId,
      attachments: [], commentAttachments: []},
    items: (turn.items ?? []).map(nativeItem)};
}

function nativeItem(item) {
  if (item.type === 'collabAgentToolCall') return {...item,
    receiverThreads: (item.receiverThreadIds ?? []).map(threadId => ({threadId, thread: null}))};
  return item;
}

function canonicalHistory(turns) {
  const entitiesByKey = Object.fromEntries(turns.map(turn => [turn.turnId, turn]));
  return {kind: 'canonical', history: {entitiesByKey, generation: 0, isComplete: true,
    islands: [{id: 'shared', entries: turns.map(turn => ({key: turn.turnId, value: turn.turnId})),
      olderBoundary: {status: 'exhausted', boundaryId: 'shared:older'},
      newerBoundary: {status: 'exhausted', boundaryId: 'shared:newer'}}]}};
}
