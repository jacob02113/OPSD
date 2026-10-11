"""Keep sampled token decisions; encode only explicitly edited suffixes."""


def command_tokens(tokenizer, raw, text):
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
    raw = list(raw or [])
    target = text.rstrip('\r\n') + '\n'
    if decode(raw) == target:
        result = raw
    else:
        # Preserve complete tokens before a textual intervention. Never cut a
        # token in half or normalize the preceding student's decisions.
        keep, prefix = 0, ''
        for i in range(1, len(raw) + 1):
            candidate = decode(raw[:i])
            if not target.startswith(candidate):
                break
            keep, prefix = i, candidate
        result = raw[:keep] + tokenizer.encode(target[len(prefix):], add_special_tokens=False)
        if decode(result) != target:
            raise ValueError('Edited command suffix did not preserve command text')
    # Tokens crossing the closing-tag/newline boundary get a deterministic
    # target; the full token is retained, including its closing-tag characters.
    end = len(target) - 1
    content = len(result)
    while content:
        if len(decode(result[:content])) <= end:
            break
        content -= 1
    return result, content
