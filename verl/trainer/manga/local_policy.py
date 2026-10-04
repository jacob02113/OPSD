"""State-local command support and sparse mixture math (no model dependencies)."""
from functools import lru_cache
import itertools
import bisect
import math
import re

from .command_teacher import sg, signature, teacher_chat, teacher_followup

COMMAND_MAX_TOKENS = 896
COMMAND_STOPS = [f'</{kind}>\n' for kind in ('enter', 'detect', 'read', 'link_to', 'ground')]

def command_boundary(text):
    return any(stop in text for stop in COMMAND_STOPS)


@lru_cache(maxsize=2)
def vocabulary_for(tokenizer):
    # AgentLoop instances are per sample; tokenizer objects are shared per worker.
    return Vocabulary(tokenizer)


def local_chat(messages, history, command, key):
    import html
    kind = key[0]
    action = {
        'enter': 'enter the panel outlined in red',
        'detect': f'detect the {key[1]} region outlined in red',
        'read': 'read the registered text region outlined in red',
        'link': ('link text region A to its speaker B' if key[1] == 'speaker'
                 else 'link registered characters A and B as the same character'),
        'ground': 'ground a description of the current panel using the highlighted registered characters',
    }[kind]
    detail = ''
    if kind == 'read':
        detail = 'The text reads: ' + repr(html.unescape(command.split('<text>', 1)[1].rsplit('</text>', 1)[0])) + '. '
    elif kind == 'ground':
        payload = command.split('<text>', 1)[1].rsplit('</text>', 1)[0]
        boxes = list(dict.fromkeys(sg._match_box(m, 0) for m in sg.BOX_RE.finditer(payload)))
        mappings = []
        pattern = re.escape(sg.REF_START) + '(.*?)' + re.escape(sg.REF_END) + '((?:' + sg.BOX_RE.pattern + ')+)'
        def plain(match):
            mention = html.unescape(match.group(1))
            labels = [str(boxes.index(sg._match_box(m, 0)) + 1) for m in sg.BOX_RE.finditer(match.group(2))]
            mappings.append(repr(mention) + ': ' + ', '.join(labels))
            return match.group(1)
        caption = html.unescape(re.sub(pattern, plain, payload, flags=re.DOTALL))
        detail = 'Description: ' + repr(caption) + '. Mention-to-box mapping: ' + '; '.join(mappings) + '. '
    followup = ('Given the steps completed so far, one executable next step is to ' + action + '. '
                'The additional thumbnail highlights the relevant regions in the original image. '
                + detail + 'Other executable next steps remain valid. Continue the sequence in the existing command format.')
    import copy
    chat = copy.deepcopy(messages)
    if history:
        chat.append({'role': 'assistant', 'content': history})
    chat.append({'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': followup}]})
    return chat


@lru_cache(maxsize=4096)
def number_options(prefix):
    return tuple(i for i in range(1001) if str(i).startswith(prefix))


@lru_cache(maxsize=8192)
def box_prefix_domains(text):
    """Return possible integer coordinates for an unfinished box, or None."""
    literals = ['(', ',', '),(', ',', ')']
    domains, cursor = [], 0
    for i, literal in enumerate(literals):
        remaining = text[cursor:]
        if len(remaining) < len(literal):
            return domains + [range(1001)] * (4-len(domains)) if literal.startswith(remaining) else None
        if not remaining.startswith(literal):
            return None
        cursor += len(literal)
        if i == 4:
            return domains if cursor == len(text) else None
        end = cursor
        while end < len(text) and text[end].isdigit() and text[end].isascii():
            end += 1
        digits = text[cursor:end]
        if end == len(text):
            possible = number_options(digits)
            return domains + [possible] + [range(1001)] * (3-len(domains)) if possible else None
        if not digits or int(digits) > 1000:
            return None
        domains.append((int(digits),))
        cursor = end
    return None


@lru_cache(maxsize=16384)
def can_match_box(text, target, threshold):
    domains = box_prefix_domains(text)
    if domains is None:
        return False
    # Discrete interval/GT breakpoints avoid enumerating all coordinate tuples.
    # IoU is piecewise linear-fractional; extrema lie at interval/GT breakpoints.
    points = []
    for i, domain in enumerate(domains):
        # Domains are ascending tuples/ranges; do not scan up to 1001
        # values for every vocabulary-trie character and candidate box.
        low, high = domain[0], domain[-1]
        axis = (target[0], target[2]) if i % 2 == 0 else (target[1], target[3])
        candidates = {low, high}
        for value in axis:
            index = bisect.bisect_left(domain, value)
            if index < len(domain): candidates.add(domain[index])
            if index: candidates.add(domain[index-1])
        points.append(sorted(candidates))
    return any(b[0] < b[2] and b[1] < b[3] and sg.iou(b, target) >= threshold
               for b in itertools.product(*points))


class Support:
    def __init__(self, executor, command, alias=None):
        self.executor, self.command = executor, command
        self.key = signature(executor, command)
        if self.key is None:
            raise ValueError('Nonexecutable teacher candidate')
        self.alias = alias if alias and signature(executor, alias) == self.key else None
        self.template = command.split('<text>', 1)[0]
        self.references = []
        for match in sg.BOX_RE.finditer(self.template):
            kind = self.template[:match.start()].rsplit(sg.REF_START, 1)[-1].split(sg.REF_END, 1)[0]
            mapping = (executor.state.text_by_student_box if kind == 'text' else
                       executor.state.character_by_student_box if kind == 'character' else executor.panel_by_box)
            self.references.append((match, sg._match_box(match, 0), mapping))
        self.targets = tuple(target for _, target, _ in self.references)
        # An instance-owned cache dies with this command's teacher contexts.
        # A decorator on the method would retain old executor snapshots globally.
        self.viable = lru_cache(maxsize=2048)(self._viable)
        self.matches_completed = lru_cache(maxsize=2048)(
            lambda text: signature(self.executor, text) == self.key)
        if self.key[0] == 'detect':
            m = sg.BOX_RE.search(command)
            self.head = command[:m.start()] + sg.BOX_START
            self.target = sg._match_box(m, 0)
        elif self.key[0] in ('read', 'ground'):
            self.head = command[:command.index('<text>')+6]

    def reference_prefix(self, prefix, start_ref=0):
        """Normalize completed reference boxes for syntax checks only.

        Return (prefix, partial_box). Payload text is never rewritten.
        """
        if self.key[0] == 'detect':
            return prefix, False
        template = self.template
        cursor = self.references[start_ref-1][0].end() if start_ref else 0
        for match, target, mapping in self.references[start_ref:]:
            start = match.start()
            # Earlier boxes have already been normalized to template lengths.
            if len(prefix) <= start:
                return prefix, False
            if prefix[cursor:start] != template[cursor:start]:
                return None, False
            close = prefix.find(sg.BOX_END, start)
            if close < 0:
                box_text = prefix[start:]
                if not box_text.startswith(sg.BOX_START):
                    return prefix, False
                value, marker, suffix = box_text[len(sg.BOX_START):].partition('<')
                if marker and not sg.BOX_END.startswith(marker + suffix):
                    return None, False
                return (prefix, True) if can_match_box(value, target, 0.95) else (None, False)
            end = close + len(sg.BOX_END)
            actual = sg.BOX_RE.fullmatch(prefix[start:end])
            if actual is None:
                return None, False
            try:
                box = sg._match_box(actual, 0)
            except ValueError:
                return None, False
            if box != target and sg.iou(box, target) < 0.95:
                return None, False
            # Match the registered identity, not just an overlapping candidate.
            if self.executor.resolve_reference(mapping, box) != mapping.get(target):
                return None, False
            prefix = prefix[:start] + match.group(0) + prefix[end:]
            cursor = match.end()
        return prefix, False

    def payload_started(self, prefix):
        normalized, partial = self.reference_prefix(prefix)
        return normalized is not None and not partial and normalized.startswith(self.head)

    def continuation(self, prefix):
        """Validate new token suffixes without rechecking already resolved objects.

        Normalization is internal only: the sampled coordinates and training
        sequence remain unchanged. Each link target still has its own predicate.
        """
        normalized, _ = self.reference_prefix(prefix)
        if normalized is None:
            return lambda suffix: False
        skip = 0
        if self.key[0] != 'detect':
            skip = min(len(self.references), normalized.split('<text>', 1)[0].count(sg.BOX_END))
        return lambda suffix: self.viable(normalized + suffix, skip)

    def _viable(self, prefix, start_ref=0):
        # Exact template prefixes need no geometry or executor parse. This also
        # makes action selection a cheap finite set of literal continuations.
        if self.command.startswith(prefix):
            return True
        prefix, partial_box = self.reference_prefix(prefix, start_ref)
        if prefix is None:
            return False
        if partial_box:
            return True
        # Newlines inside <text> are payload, including partial multiline reads.
        # Only inspect the suffix after </text> for a command separator.
        boundary_part = prefix
        if self.key[0] in ('read', 'ground') and prefix.startswith(self.head):
            end = prefix.find('</text>', len(self.head))
            boundary_part = prefix[end:] if end >= 0 else ''
        if '\n' in boundary_part:
            return prefix.endswith('\n') and self.matches_completed(prefix.rstrip('\n'))
        if self.command.startswith(prefix) or (self.alias and self.alias.startswith(prefix)):
            return True
        kind = self.key[0]
        if kind == 'detect':
            if not prefix.startswith(self.head):
                return self.head.startswith(prefix)
            rest = prefix[len(self.head):]
            end = rest.find('<')
            box_text = rest if end < 0 else rest[:end]
            if not can_match_box(box_text, self.target, self.executor.iou_threshold):
                return False
            suffix = sg.BOX_END + '</detect>'
            # IoU feasibility alone does not determine object identity: a box
            # can overlap several candidates but the executor chooses only one.
            # Reject the others as soon as all coordinates are complete, before
            # walking the next special token's characters in the vocabulary trie.
            if re.fullmatch(r'\([0-9]+,[0-9]+\),\([0-9]+,[0-9]+\)', box_text):
                if not self.matches_completed(self.head + box_text + suffix):
                    return False
            if end < 0:
                return True
            if not suffix.startswith(rest[end:]):
                return False
            # Completed coordinates must match this target, not a neighbouring one.
            completed = self.head + box_text + suffix
            return self.matches_completed(completed)
        if kind in ('read', 'ground'):
            return prefix.startswith(self.head) or self.head.startswith(prefix)
        return False


class Vocabulary:
    """Decode vocabulary once; structural prefixes prune whole token-trie branches."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.trie = {}
        self.plain_ids, self.newline_tokens = [], []
        special = set(tokenizer.all_special_ids)
        permitted = set()
        for text in (sg.BOX_START, sg.BOX_END, sg.REF_START, sg.REF_END):
            permitted.update(tokenizer.encode(text, add_special_tokens=False))
        for token in tokenizer.get_vocab().values():
            if token in special and token not in permitted:
                continue
            text = tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if '\n' in text:
                self.newline_tokens.append((token, text))
            else:
                self.plain_ids.append(token)
            if not text or '\ufffd' in text:
                # Incomplete UTF-8 byte tokens are allowed only inside free text;
                # see allowed(), where they cannot affect command structure.
                continue
            node = self.trie
            for char in text:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(token)
        self.byte_ids = [token for token in tokenizer.get_vocab().values() if token not in special and
                         '\ufffd' in tokenizer.decode([token], skip_special_tokens=False)]
        self.vocab_size = max(tokenizer.get_vocab().values()) + 1
        self.nonplain = set(range(self.vocab_size)) - set(self.plain_ids)
        self.newline_ids = {token for token, _ in self.newline_tokens}
        self.text_ids = self.plain_ids + sorted(self.newline_ids)
        self.text_excluded = self.nonplain - self.newline_ids
        self.text_mask = {'exclude': sorted(self.text_excluded), 'vocab_size': self.vocab_size}
        # Outside text payloads, only trailing newlines are permitted. Group token
        # spellings once instead of reparsing the same full command per token.
        self.newline_groups = {}
        for token, text in self.newline_tokens:
            body = text.rstrip('\n')
            if text.endswith('\n') and '\n' not in body:
                self.newline_groups.setdefault(body, []).append(token)

    def _allowed_newlines(self, support, prefix):
        if support.payload_started(prefix) and '</text>' not in prefix:
            return self.newline_ids
        result = set()
        closing = '</' + support.key[0] + '>'
        for body, tokens in self.newline_groups.items():
            # Cheap necessary condition before a full executor parse. This is
            # used only by read/ground, whose complete syntax ends with this tag.
            if (prefix + body).endswith(closing) and support.viable(prefix + body + '\n'):
                result.update(tokens)
        return result

    def union_mask(self, supports, prefix):
        """Visit the token trie once for the union, retaining candidate identity."""
        structural = []
        excluded = None
        for support in supports:
            if not support.viable(prefix):
                continue
            if support.key[0] in ('read', 'ground') and support.payload_started(prefix):
                current = (self.text_excluded if '</text>' not in prefix[len(support.head):]
                           else self.nonplain - self._allowed_newlines(support, prefix))
                excluded = current if excluded is None else excluded & current
            else:
                structural.append(support)
        if not structural and excluded == self.text_excluded:
            return self.text_mask
        result = set()
        if structural:
            predicates = [s.continuation(prefix) for s in structural]
            stack = [(self.trie, "", predicates)]
            while stack:
                node, text, active = stack.pop()
                result.update(node.get(None, ()))
                for char, child in node.items():
                    if char is None:
                        continue
                    extended = text + char
                    remaining = [predicate for predicate in active if predicate(extended)]
                    if remaining:
                        stack.append((child, extended, remaining))
        if excluded is not None:
            return {'exclude': sorted(excluded-result), 'vocab_size': self.vocab_size}
        return sorted(result)

    def mask(self, support, prefix):
        if support.key[0] in ('read','ground') and prefix.startswith(support.head):
            allowed_newlines = self._allowed_newlines(support, prefix)
            return {'exclude': sorted(self.nonplain-allowed_newlines), 'vocab_size':self.vocab_size}
        return self.allowed(support, prefix)

    def allowed(self, support, prefix):
        if support.key[0] in ('read','ground') and prefix.startswith(support.head):
            return sorted(self.plain_ids + list(self._allowed_newlines(support, prefix)))
        result = []
        stack = [(self.trie, prefix)]
        while stack:
            node, text = stack.pop()
            result.extend(node.get(None, ()))
            for char, child in node.items():
                if char is not None and support.viable(text + char):
                    stack.append((child, text + char))
        if support.key[0] in ('read','ground') and prefix.startswith(support.head):
            result.extend(self.byte_ids)
        return sorted(set(result))


def mix_sparse(components, weights, width):
    """Lower-bound mixture entries; missing top-k entries never become illegal."""
    mass = {}
    for (ids, logs), weight in zip(components, weights):
        for token, value in zip(ids, logs):
            if math.isfinite(value):
                probability = weight * math.exp(value)
                if probability:
                    mass[token] = mass.get(token, 0.) + probability
    retained = math.fsum(mass.values())
    best = sorted(mass.items(), key=lambda x: (-x[1], x[0]))[:width]
    if not best:
        raise ValueError('Empty teacher mixture support')
    # Keep true retained mass: actor top-k+tail handles the omitted probability.
    used = {i for i,p in best}
    padding = []
    i = 0
    while len(best)+len(padding) < width:
        if i not in used:
            padding.append(i)
        i += 1
    return ([i for i,p in best] + padding,
            [math.log(p) for i,p in best] + [-1e9]*len(padding), retained)
