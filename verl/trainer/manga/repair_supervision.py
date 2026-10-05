"""Align executor recovery decisions without changing predictor prefixes."""

def align_repairs(records, response_ids, start, size):
    aligned = {}
    dropped = 0
    for prefix, allowed in records:
        position = len(prefix)
        if not (start <= position < start + size) or tuple(response_ids[:position]) != prefix:
            dropped += 1
            continue
        if response_ids[position] not in allowed:
            dropped += 1
            continue
        # An earlier repair can be revisited after rollback. Keep one target.
        aligned[position-start] = list(allowed)
    return aligned, dropped
