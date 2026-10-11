"""Bounded, verified witnesses for uncovered transport predictors."""
from .local_policy import matching_boxes, can_match_box, sg


def supplement_candidates(worker, prefix, supports, status, accepted, check, limit=4):
    """Keep the original tokens; encode only a witness suffix, never the prefix."""
    text = worker._decode(prefix)
    # A structural decision on an accepted executable command must have a
    # legal observed token. Do not hide a support/checker bug with resampling.
    if '<text>' not in text and status.get(accepted) is not True:
        raise ValueError(f'Accepted structural token failed transport validation: '
                         f'prefix={text!r}, token={accepted}, continuation={worker._decode(prefix+[accepted])!r}')
    candidates = [accepted] if status.get(accepted) is True else []
    candidates += [t for t, ok in sorted(status.items()) if ok and t not in candidates]
    if candidates:
        return candidates[:limit], 'missing_teacher_scores'

    # A structural prefix already returned its verified observed token above.
    payload = text.split('<text>', 1)[1]
    begin = payload.rfind(sg.BOX_START)
    if begin < 0 or sg.BOX_END in payload[begin:]:
        return [], 'unknown_payload'
    body = payload[begin+len(sg.BOX_START):]
    if '<' in body:
        return [], 'unknown_payload'
    mapping = supports[0].executor.state.character_by_student_box
    viable = [target for target in mapping if can_match_box(body, target, .95)]
    if not viable:
        return [], 'dead_coordinate_prefix'

    def completions():
        # At most one numeric witness per registered target. This is a
        # bounded proposal search, not an exhaustive token support.
        for target in viable:
            box = next(matching_boxes(body, target, .95), None)
            if box is not None:
                complete = f'({box[0]},{box[1]}),({box[2]},{box[3]})'
                if complete.startswith(body):
                    yield text + complete[len(body):] + sg.BOX_END

    for index, complete in enumerate(completions()):
        if index >= 16:  # Bound tokenization even for a large frontier.
            break
        suffix = worker.tokenizer.encode(complete[len(text):], add_special_tokens=False)
        if not suffix:
            continue
        token = int(suffix[0])
        if token not in candidates and check(worker._decode, prefix, [token], supports).get(token) is True:
            candidates.append(token)
        if len(candidates) >= limit:
            break
    return candidates, 'supplement' if candidates else 'witness_not_found'


def earlier_coordinate_error(worker, prefix, supports):
    """Locate the original token that first made this box infeasible, for logs.

    Never roll back an already executed asynchronous trajectory or transplant
    a target to a different prefix. Earlier visited predictors keep their own
    transport labels and may receive a supplement in the same command batch.
    """
    text = worker._decode(prefix)
    begin = text.rfind(sg.BOX_START)
    mapping = supports[0].executor.state.character_by_student_box
    for i in range(len(prefix)):
        current = worker._decode(prefix[:i+1])
        if len(current) < begin + len(sg.BOX_START):
            continue
        body = current[begin+len(sg.BOX_START):].split('<',1)[0]
        if not any(can_match_box(body, target, .95) for target in mapping):
            return i
    return None
