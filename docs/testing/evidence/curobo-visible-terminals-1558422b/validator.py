def norm(v):
    return v.strip().casefold() if isinstance(v, str) else None

def validate(obj, expected):
    errors = []
    if not isinstance(obj, dict):
        return (False, ['response_not_object'])
    observed = obj.get('observed')
    evidence = obj.get('evidence')
    presentation = obj.get('presentation')
    if not isinstance(observed, dict):
        errors.append('observed_missing')
        observed = {}
    if not isinstance(evidence, dict):
        errors.append('evidence_missing')
        evidence = {}
    if not isinstance(presentation, dict):
        errors.append('presentation_missing')
        presentation = {}
    for k in ('hardware', 'dataset', 'mode', 'problem_id', 'status'):
        if norm(observed.get(k)) != norm(expected[k]):
            errors.append(f'{k}_mismatch')
    if presentation.get('blank') is not False:
        errors.append('blank_or_unreported')
    if presentation.get('readable') is not True:
        errors.append('unreadable_or_unreported')
    if evidence.get('goal_marker_visible') is not True:
        errors.append('goal_not_visible')
    if expected['status'] == 'success':
        label = evidence.get('path_label')
        good = isinstance(label, str) and label.strip().casefold().startswith('complete tool path (') and ('clipped' not in label.casefold()) and ('/' not in label)
        if not good:
            errors.append('complete_path_label_missing')
        if evidence.get('path_start_visible') is not True:
            errors.append('path_start_not_visible')
        if evidence.get('path_terminal_visible') is not True:
            errors.append('path_terminal_not_visible')
        if evidence.get('no_trajectory_text_visible') is not False:
            errors.append('unexpected_no_trajectory_text')
    else:
        if evidence.get('no_trajectory_text_visible') is not True:
            errors.append('no_trajectory_text_missing')
        if evidence.get('path_label') not in (None, ''):
            errors.append('unexpected_path_label')
    computed = not errors
    if obj.get('success') is not computed:
        errors.append('model_success_contradicts_fields')
    score = obj.get('score')
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        errors.append('score_invalid')
    elif (score >= threshold) != computed:
        errors.append('score_contradicts_fields')
    return (not errors, errors)
