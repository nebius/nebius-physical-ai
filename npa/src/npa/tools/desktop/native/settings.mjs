// Share the effective thread settings without exposing private instructions.
/** Project effective model, reasoning, speed, and mode without private instructions.
 * Args: live is the VS Code state; own is local engine state; row is saved metadata.
 * Returns: Public conversation settings.
 * Raises: None.
 */
export function threadSettings(live, own, row) {
  const settings = live?.latestThreadSettings ?? own?.settings ?? {};
  const collaboration = settings.collaborationMode ?? live?.latestCollaborationMode;
  return {
    model: settings.model ?? live?.latestModel ?? row?.model ?? null,
    effort: settings.effort ?? collaboration?.settings?.reasoning_effort ?? live?.latestReasoningEffort ?? row?.reasoning_effort ?? null,
    serviceTier: settings.serviceTier ?? null,
    mode: collaboration?.mode ?? 'default',
  };
}

/** Validate settings against the signed-in account catalog.
 * Args: request contains selected settings; models and modes are live catalog entries.
 * Returns: Validated model, reasoning, and speed fields.
 * Raises: An error for unavailable settings.
 */
export function validateSettings(request, models, modes) {
  const model = models.find(model => model.model === request.model && !model.hidden);
  if (!model) throw new Error('Choose a model available to your Codex account');
  if (!model.supportedReasoningEfforts.some(option => option.reasoningEffort === request.effort))
    throw new Error('This model does not support that reasoning effort');
  const serviceTier = request.serviceTier || null;
  if (serviceTier !== null && serviceTier !== 'default' && !model.serviceTiers?.some(tier => tier.id === serviceTier))
    throw new Error('This model does not support that speed');
  if (!modes.some(mode => mode.mode === request.mode)) throw new Error('Choose an available chat mode');
  return {model: model.model, effort: request.effort, serviceTier};
}

/** Find a browser send identity in the native conversation view.
 * Args: state is the followed VS Code state; clientId is the stable send UUID.
 * Returns: Whether the same message appears in VS Code.
 * Raises: None.
 */
export function messageVisible(state, clientId) {
  const history = state?.turnHistory?.history;
  const turns = history ? history.islands.flatMap(island => island.entries.map(entry => history.entitiesByKey[entry.key])).filter(Boolean) : state?.turns ?? [];
  return turns.some(turn => turn.params?.clientUserMessageId === clientId || turn.items?.some(item =>
    ['userMessage', 'steeringUserMessage'].includes(item.type) &&
    (item.clientUserMessageId === clientId || item.clientId === clientId)));
}
