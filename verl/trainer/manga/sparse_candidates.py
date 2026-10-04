"""Bounded proposal checks; no vocabulary enumeration or GPU synchronization."""
import math


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


def candidate_support(tokenizer, prefix_ids, accepted, supports, student_ids, components, budget):
    """Union student top-k, aggregate teacher top-k and the accepted continuation.

    Components contain original (unconditioned) log probabilities, including the
    exact accepted-token probability. Missing teacher entries are unknown, not
    executor-illegal. Teacher preference is explicitly conditioned on the observed
    legal entries; its unobserved mass is reported, never invented as a legal tail.
    """
    mixture = {}
    for ids, logs in components:
        for token, value in dict(zip(ids, logs)).items():
            if math.isfinite(value) and value > -1e8:
                mixture[token] = logadd(mixture.get(token, -math.inf), value)
    teacher_best = sorted(mixture, key=lambda t: (-mixture[t], t))[:budget]
    checked = sorted(set(student_ids[:budget]) | set(teacher_best) | {accepted})
    # Decode the actual token sequence, rather than concatenating decoded pieces:
    # byte tokens and BPE tokens spanning markup must retain their real semantics.
    allowed = []
    for token in checked:
        text = tokenizer.decode(prefix_ids + [token], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        if '\ufffd' not in text and any(s.viable(text) for s in supports):
            allowed.append(token)
    if accepted not in allowed:
        raise RuntimeError('Accepted continuation missing from sparse correct support')
    observed = [t for t in allowed if t in mixture]
    if not observed:
        raise RuntimeError('Teacher did not return the accepted-token probability')
    normalizer = -math.inf
    for token in observed:
        normalizer = logadd(normalizer, mixture[token])
    # Sum component probabilities before normalizing, preserving their preferences.
    logs = [mixture[t] - normalizer for t in observed]
    retained = min(1., math.exp(normalizer - math.log(len(components))))
    return allowed, checked, observed, logs, retained
