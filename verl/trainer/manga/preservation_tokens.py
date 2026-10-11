"""Track actual sampled tokens without relabeling edited/GT suffixes as successes."""
import re


def action_groups(tokenizer, ids):
    """Only decode the short action prefix; the rest needs no token classification."""
    groups = [3] * len(ids)
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    action = re.match(r'\s*<(enter|detect|read|link_from|ground)>', text)
    if action is None:
        raise ValueError('Preserved command is missing a recognizable action prefix')
    begin = 0
    for i in range(len(ids)):
        end = len(tokenizer.decode(ids[:i + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        if begin < action.end(1) and end > action.start(1):
            groups[i] = 1
        if end >= action.end(1):
            break
        begin = end
    return groups


def diagnostic_groups(tokenizer, ids, *, decision_only=False):
    """Diagnostic labels only: action=1, object class/coordinates=2, other=3.

    Include ground's caption references. Fixed tags/punctuation remain content;
    a token spanning a decision and punctuation belongs to the decision.
    """
    decode = lambda xs: tokenizer.decode(xs, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = decode(ids)
    spans = []
    action = re.match(r'\s*<(enter|detect|read|link_from|ground)>', text)
    if action:
        spans.append((*action.span(1), 1))
    for m in re.finditer(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>', text, re.S):
        if decision_only and '<text>' in text[:m.start()]:
            continue  # Caption mention names are free text, not object classes.
        spans.append((*m.span(1), 2))
    for m in re.finditer(r'<\|box_start\|>(.*?)<\|box_end\|>', text, re.S):
        for number in re.finditer(r'\d+', m.group(1)):
            spans.append((m.start(1) + number.start(), m.start(1) + number.end(), 2))
    groups, begin = [3] * len(ids), 0
    last = max((b for _, b, _ in spans), default=0)
    for i in range(len(ids)):
        if begin >= last:
            break
        end = len(decode(ids[:i + 1]))
        groups[i] = next((group for a, b, group in spans if begin < b and end > a), 3)
        begin = end
    return groups


def sampled_positions(start, accepted, raw, gt_suffix_start=None, repaired=()):
    raw = raw or []
    limit = min(len(accepted), len(raw))
    if gt_suffix_start is not None:
        limit = min(limit, gt_suffix_start)
    common = 0
    while common < limit and accepted[common] == raw[common]:
        common += 1
    excluded = set(repaired)
    return [start + i for i in range(common) if i not in excluded]
