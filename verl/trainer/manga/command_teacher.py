"""Shared state-local target selection for corrective OPSD and its diagnostic.

No model dependencies. State semantics deliberately match
the current training executor, not a claim of deployment-state equivalence.
"""
from __future__ import annotations

import copy
import html

from . import scene_graph as sg


def reference(kind, box):
    x0, y0, x1, y1 = box
    return f'{sg.REF_START}{kind}{sg.REF_END}{sg.BOX_START}({x0},{y0}),({x1},{y1}){sg.BOX_END}'


def signature(executor, command):
    """Identity of a legal operation, evaluated BEFORE executing either branch."""
    if not executor.evaluate(command, compute_reward=False).executable:
        return None
    if m := sg.ENTER_RE.fullmatch(command):
        return ('enter', executor.state.entered)
    if m := sg.DETECT_RE.fullmatch(command):
        index, _ = executor._eligible_detection(m.group(1), sg._match_box(m, 2))
        return ('detect', m.group(1), index)
    if m := sg.READ_RE.fullmatch(command):
        return ('read', executor.resolve_reference(executor.state.text_by_student_box, sg._match_box(m, 1)))
    if m := sg.LINK_RE.fullmatch(command):
        a, b = sg._match_box(m, 2), sg._match_box(m, 7)
        right = executor.resolve_reference(executor.state.character_by_student_box, b)
        if m.group(1) == 'text':
            return ('link', 'speaker', executor.resolve_reference(executor.state.text_by_student_box, a), right)
        left = executor.resolve_reference(executor.state.character_by_student_box, a)
        return ('link', 'identity', *sorted((left, right)))
    if m := sg.GROUND_RE.fullmatch(command):
        return ('ground', executor.resolve_reference(executor.panel_by_box, sg._match_box(m, 1)))
    return None


def frontier(executor):
    """GT answers with references to the objects actually registered in history."""
    t, s = executor.target, executor.state
    candidates = []
    if s.entered < len(t.panels):
        candidates.append('<enter>' + reference('panel', t.panel_boxes[s.entered]) + '</enter>')
    for kind, boxes, owners, detected in (
        ('character', t.character_boxes, t.character_panel, s.detected_characters),
        ('text', t.text_boxes, t.text_panel, s.detected_texts),
    ):
        for i, box in enumerate(boxes):
            if i not in detected and owners[i] < s.entered:
                candidates.append('<detect>' + reference(kind, box) + '</detect>')
    for i in sorted(s.detected_texts):
        box = s.detected_texts[i]
        if i not in s.reads:
            text = html.escape(str(t.texts[i].get('content', '')), quote=False)
            candidates.append('<read>' + reference('text', box) + f'<text>{text}</text></read>')
    for a, b in sorted(t.speaker_links):
        if a in s.reads and b in s.detected_characters and (a, b) not in s.speakers:
            candidates.append('<link_from>' + reference('text', s.detected_texts[a])
                              + '</link_from><link_to>' + reference('character', s.detected_characters[b]) + '</link_to>')
    for a in sorted(s.detected_characters):
        for b in sorted(s.detected_characters):
            if a < b and t.clusters[a] == t.clusters[b] and s.identities.find(a) != s.identities.find(b):
                candidates.append('<link_from>' + reference('character', s.detected_characters[a])
                                  + '</link_from><link_to>' + reference('character', s.detected_characters[b]) + '</link_to>')
    for p, g in sorted(executor.grounding_by_panel.items()):
        if p >= s.entered or p in s.groundings or not executor._ground_required(p).issubset(s.detected_characters):
            continue
        caption, cursor, pieces = str(g.get('caption', '')), 0, []
        for mention in sorted(g.get('mentions', ()), key=lambda m: (m['start'], m['end'])):
            start, end = int(mention['start']), int(mention['end'])
            if not cursor <= start < end <= len(caption):
                raise ValueError('Invalid grounding annotation')
            pieces.extend([html.escape(caption[cursor:start], quote=False), sg.REF_START,
                           html.escape(caption[start:end], quote=False), sg.REF_END])
            for box in mention.get('character_bboxes', ()):
                i = t.character_boxes.index(tuple(box))
                pieces.append(reference('character', s.detected_characters[i]).split(sg.REF_END, 1)[1])
            cursor = end
        pieces.append(html.escape(caption[cursor:], quote=False))
        candidates.append('<ground>' + reference('panel', t.panel_boxes[p]) + '<text>' + ''.join(pieces) + '</text></ground>')
    return [c for c in candidates if signature(executor, c) is not None]


def target_boxes(executor, command):
    """Only GT boxes of the selected command."""
    key = signature(executor, command)
    if key is None:
        raise ValueError('Target overlay requires a legal selected command')
    t, s = executor.target, executor.state
    if key[0] == 'detect':
        return [tuple((t.text_boxes if key[1] == 'text' else t.character_boxes)[key[2]])]
    if key[0] == 'ground':
        boxes = []
        for mention in executor.grounding_by_panel[key[1]].get('mentions', ()):
            for box in mention.get('character_bboxes', ()):
                index = t.character_boxes.index(tuple(box))
                boxes.append(tuple(t.character_boxes[index]))
        return list(dict.fromkeys(boxes))
    # ENTER/READ have one reference; LINK has exactly two endpoints.
    return [sg._match_box(m, 0) for m in sg.BOX_RE.finditer(command)]


def render_target_image(original, boxes):
    from PIL import ImageDraw
    image = original.convert('RGB').copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in boxes:
        draw.rectangle(sg._pixel_box(box, width, height), outline=(255, 0, 0),
                       width=max(2, round(min(width, height) / 300)))
    return image


def hint_for(executor, command):
    key = signature(executor, command)
    if key is None:
        raise ValueError('Hint requires a legal command')
    # Both answers and references come from GT.
    for candidate in frontier(executor):
        if signature(executor, candidate) == key:
            return candidate
    raise ValueError(f"No canonical GT command for {key}")



def dependency_graph(executor):
    """Complete each available panel stage before entering the next panel.

    Relations belong to their latest prerequisite panel, so cross-panel links
    never prevent entering the panel needed to execute them.
    """
    cached = getattr(executor, "_command_dependency_graph", None)
    if cached is not None:
        return cached
    t = executor.target
    graph = {}
    for p in range(len(t.panels)):
        graph[('enter', p)] = (('enter', p - 1),) if p else ()
    for kind, owners in (('character', t.character_panel), ('text', t.text_panel)):
        for i, owner in enumerate(owners):
            graph[('detect', kind, i)] = (('enter', owner),)
    for i in range(len(t.texts)):
        graph[('read', i)] = (('detect', 'text', i),)
    for a, b in sorted(t.speaker_links):
        graph[('link', 'speaker', a, b)] = (('read', a), ('detect', 'character', b))
    for a in range(len(t.characters)):
        for b in range(a + 1, len(t.characters)):
            if t.clusters[a] == t.clusters[b]:
                graph[('link', 'identity', a, b)] = (('detect', 'character', a), ('detect', 'character', b))
    for p in sorted(executor.grounding_by_panel):
        graph[('ground', p)] = (('enter', p), *(
            ('detect', 'character', i) for i in sorted(executor._ground_required(p))))
    stages = {}
    for node in graph:
        if node[0] == 'enter':
            continue
        stage = node_panel(executor, node)
        if node[0] == 'ground':
            stage = max([stage] + [t.character_panel[i] for i in executor._ground_required(node[1])])
        stages.setdefault(stage, []).append(node)
    for p in range(1, len(t.panels)):
        graph[('enter', p)] = (('enter', p - 1), *stages.get(p - 1, ()))
    executor._command_dependency_graph = graph
    return graph


def node_done(executor, node):
    s = executor.state
    if node[0] == 'enter':
        return node[1] < s.entered
    if node[0] == 'detect':
        return node[2] in (s.detected_texts if node[1] == 'text' else s.detected_characters)
    if node[0] == 'read':
        return node[1] in s.reads
    if node[0] == 'ground':
        return node[1] in s.groundings
    if node[1] == 'speaker':
        return (node[2], node[3]) in s.speakers
    a, b = node[2:]
    return a in s.detected_characters and b in s.detected_characters and s.identities.find(a) == s.identities.find(b)


def node_panel(executor, node):
    """A relation becomes available at the later endpoint's panel."""
    t = executor.target
    if node[0] in ('enter', 'ground'):
        return node[1]
    if node[0] == 'detect':
        return (t.text_panel if node[1] == 'text' else t.character_panel)[node[2]]
    if node[0] == 'read':
        return t.text_panel[node[1]]
    a, b = node[2:]
    return max(t.text_panel[a] if node[1] == 'speaker' else t.character_panel[a], t.character_panel[b])


def intended_node(executor, command, graph):
    """Resolve intent before legality; exact registered references take priority.

    Refuse arbitrary zero-IoU object reassignment. Malformed/unmatched actions
    use the explicit panel-local fallback instead of a pretend dependency.
    """
    t, s = executor.target, executor.state

    def resolve(kind, box):
        boxes = {'text': t.text_boxes, 'character': t.character_boxes, 'panel': t.panel_boxes}[kind]
        registered = {'text': s.text_by_student_box, 'character': s.character_by_student_box, 'panel': executor.panel_by_box}[kind]
        if box in registered:
            return registered[box]
        if not boxes:
            return None
        index = max(range(len(boxes)), key=lambda i: (sg.iou(box, boxes[i]), -i))
        return index if sg.iou(box, boxes[index]) >= executor.iou_threshold else None

    try:
        if m := sg.ENTER_RE.fullmatch(command):
            node = ('enter', resolve('panel', sg._match_box(m, 1)))
        elif m := sg.DETECT_RE.fullmatch(command):
            node = ('detect', m.group(1), resolve(m.group(1), sg._match_box(m, 2)))
        elif m := sg.READ_RE.fullmatch(command):
            node = ('read', resolve('text', sg._match_box(m, 1)))
        elif m := sg.GROUND_RE.fullmatch(command):
            node = ('ground', resolve('panel', sg._match_box(m, 1)))
        elif m := sg.LINK_RE.fullmatch(command):
            a = resolve(m.group(1), sg._match_box(m, 2))
            b = resolve('character', sg._match_box(m, 7))
            if a is None or b is None:
                return None
            node = (('link', 'speaker', a, b) if m.group(1) == 'text'
                    else ('link', 'identity', *sorted((a, b))))
        else:
            return None
    except ValueError:
        return None
    return node if node in graph else None


def ready_dependencies(executor, graph, node):
    """Visit each unmet node once, preserving prerequisite selection order."""
    result, visited = [], set()
    stack = [node] if node is not None else []
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if node_done(executor, current):
            continue
        missing = [p for p in graph[current] if not node_done(executor, p)]
        if missing:
            stack.extend(reversed(missing))
        else:
            result.append(current)
    return result



def select_target(executor, student_command):
    """Project the student's goal onto its ready prerequisite closure.

    Every returned command is canonical GT, even for an executable proposal.
    Unknown and already completed goals have explicitly labelled fallbacks.
    """
    choices = frontier(executor)
    if not choices:
        return None, None, 'terminal_or_no_supported_target'
    available = {signature(executor, c): c for c in choices}
    graph = dependency_graph(executor)
    intended = intended_node(executor, student_command, graph)
    if intended in available:
        return available[intended], intended, 'direct_target'
    if intended is not None and not node_done(executor, intended):
        # Every current-stage obligation is an ancestor of a later ENTER.
        # Preserve the linear-time shortcut for far-future goals.
        if node_panel(executor, intended) >= executor.state.entered:
            local = [n for n in available if n[0] != 'enter']
            chosen = min(local, key=lambda n: node_panel(executor, n)) if local else next(iter(available))
            return available[chosen], chosen, 'dag_prerequisite'
        for node in ready_dependencies(executor, graph, intended):
            if node in available:
                return available[node], node, 'dag_prerequisite'
        raise RuntimeError(f'Unfinished goal has no executable prerequisite: {intended}')
    # There is no unfinished goal to preserve. Never report this as intent repair.
    local = [n for n in available if n[0] != 'enter']
    chosen = min(local, key=lambda n: node_panel(executor, n)) if local else next(iter(available))
    reason = 'completed_target_fallback' if intended is not None else 'unmatched_target_fallback'
    return available[chosen], chosen, reason


def validate_candidate(executor, candidate, expected):
    key = signature(executor, candidate)
    return {'executable': key is not None, 'target_preserved': key is not None and key == expected,
            'signature': key, 'quality': executor.evaluate(candidate).reward}


def cases_from_response(target, response):
    """Keep exact raw textual prefixes, including rejected commands and whitespace."""
    executor = sg.MangaExecutor(sg.TargetGraph.parse(target))
    history = ''
    for index, line in enumerate(response.splitlines(keepends=True)):
        command = line.strip()
        if command and command not in sg.CHAT_END_TOKENS:
            selected, key, reason = select_target(executor, command)
            yield {'index': index, 'history': history, 'student_command': command,
                   'student_legal': signature(executor, command) is not None,
                   'selected': selected, 'expected': key, 'reason': reason,
                   'executor': copy.deepcopy(executor)}
            # Only the ORIGINAL student command advances the replay state.
            executor.execute(command)
        history += line


def teacher_followup(hint):
    if hint is None:
        raise ValueError('A command hint is required for teacher supervision')
    return (
        'Continue the full sequence until the graph is complete. Separate commands with newlines. '
        f'Next legal command: {hint}'
    )


def teacher_chat(messages, history, hint):
    """Keep the full raw history, then issue the same follow-up as the probe."""
    result = copy.deepcopy(messages)
    followup = teacher_followup(hint)
    if history:
        result.extend([{'role': 'assistant', 'content': history},
                       {'role': 'user', 'content': followup}])
    else:
        for message in reversed(result):
            if message['role'] == 'user':
                content = message['content']
                message['content'] = (content + '\n' + followup if isinstance(content, str)
                                      else [*content, {'type': 'text', 'text': followup}])
                break
        else:
            raise ValueError('Missing user task')
    return result


def teacher_messages(text, history, image, hint):
    return teacher_chat([{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                    {'type': 'text', 'text': text}]}], history, hint)
