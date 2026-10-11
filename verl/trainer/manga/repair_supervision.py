"""Align executor recovery decisions without changing predictor prefixes."""


def rejected_failure(attempt, prefix, allowed, *, rolled_back):
    """An actual sampled token violated the current executable prefix."""
    return (not rolled_back and len(prefix) < len(attempt)
            and list(attempt[:len(prefix)]) == prefix
            and attempt[len(prefix)] not in allowed)


def align_repairs(records, response_ids, start, size, *, rejected=None):
    aligned = {}
    dropped = 0
    for prefix, allowed in records:
        position = len(prefix)
        reason = None
        if not (start <= position < start + size):
            reason = 'outside_command'
        elif tuple(response_ids[:position]) != prefix:
            reason = 'prefix_changed'
        elif response_ids[position] not in allowed:
            reason = 'target_changed'
        if reason:
            dropped += 1
            if rejected is not None:
                rejected.append((prefix, allowed, reason))
            continue
        # A repaired prefix is immutable; repeated records share one target.
        aligned[position-start] = list(allowed)
    return aligned, dropped
