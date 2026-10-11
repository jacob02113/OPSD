"""Opt-in token identity checks. No retokenization or writes when disabled."""
import json
import os
from pathlib import Path


def record(stage, expected, actual, tokenizer, **metadata):
    directory = os.environ.get('MANGA_TOKEN_AUDIT_DIR')
    if not directory:
        return
    expected, actual = list(expected), list(actual)
    common = 0
    for a, b in zip(expected, actual):
        if a != b:
            break
        common += 1
    same = expected == actual
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
    row = dict(stage=stage, **metadata, ids_equal=same,
               expected_length=len(expected), actual_length=len(actual),
               common_prefix=common, text_equal=decode(expected) == decode(actual))
    if not same:
        lo, hi = max(0, common - 8), common + 16
        row.update(expected_window=expected[lo:hi], actual_window=actual[lo:hi],
                   expected_text=decode(expected[lo:hi]), actual_text=decode(actual[lo:hi]))
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'tokens-{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
