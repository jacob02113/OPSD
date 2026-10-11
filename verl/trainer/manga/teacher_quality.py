"""Bounded raw object-token audit; missing sparse mass is unknown, never illegal."""
import math

KINDS = ('enter', 'detect', 'read', 'link', 'ground')
FIELDS = ('rows', 'observed_top1_checked', 'observed_top1_legal', 'checked_mass', 'legal_mass_lb',
          'illegal_mass_lb', 'unknown_mass', 'accepted_token_nll')


def audit(tokenizer, branch, positions, results, supports, budget=8):
    if not positions:
        return [0.] * len(FIELDS)
    # One reproducible position per command, spread over all object positions.
    position = positions[sum(branch) % len(positions)]
    rows = [r[position] for r in results if position in r]
    if not rows:
        return [0.] * len(FIELDS)
    probabilities = {}
    for ids, logs, _ in rows:
        for token, logp in dict(zip(ids, logs)).items():
            if math.isfinite(logp) and logp > -1e8:
                probabilities[token] = probabilities.get(token, 0.) + math.exp(logp)/len(rows)
    candidates = sorted(probabilities, key=lambda t: (-probabilities[t], t))[:budget]
    if not candidates:
        return [0.] * len(FIELDS)
    legal = []
    for token in candidates:
        text = tokenizer.decode(branch[:position]+[token], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        legal.append(None if '\ufffd' in text else any(s.viable(text) for s in supports))
    good = math.fsum(probabilities[t] for t, ok in zip(candidates, legal) if ok)
    bad = math.fsum(probabilities[t] for t, ok in zip(candidates, legal) if ok is False)
    # Accepted token is requested explicitly from every participating component.
    accepted_logs = [dict(zip(ids,logs)).get(branch[position], -math.inf) for ids,logs,_ in rows]
    maximum = max(accepted_logs)
    if not math.isfinite(maximum):
        raise ValueError('Teacher audit is missing the exact accepted-token probability')
    nll = -(maximum + math.log(math.fsum(math.exp(x-maximum) for x in accepted_logs)/len(rows)))
    return [1., float(legal[0] is not None), float(legal[0] is True), good+bad, good, bad, max(0.,1-good-bad), nll]
