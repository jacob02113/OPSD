"""Parse the bbox-referenced manga scene-graph DSL."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


REF_START, REF_END = "<|object_ref_start|>", "<|object_ref_end|>"
BOX_START, BOX_END = "<|box_start|>", "<|box_end|>"
CHAT_END_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>", "<|im_start|>", "<|end|>")
DSL_VERSION = "bbox_ref_v1"
ACTION_TAGS = ("<enter>", "<detect>", "<read>", "<link_from>", "<ground>")
ACTION_END_TAGS = ("</enter>", "</detect>", "</read>", "</link_from>", "</link_to>", "</ground>")
BOX_RE = re.compile(
    re.escape(BOX_START)
    + r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
    + re.escape(BOX_END)
)
INTERNVL_BOX_RE = re.compile(r"<box>\s*\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]\s*</box>")


def internvl_format(text: str) -> str:
    """Adapt Qwen prompt examples to InternVL's normalized ref/box notation."""
    text = text.replace(REF_START, '<ref>').replace(REF_END, '</ref>')
    return BOX_RE.sub(lambda m: '<box>[[' + ','.join(m.groups()) + ']]</box>', text)


def _parse_box(text: str, position: int) -> tuple[Box, int] | None:
    match = BOX_RE.match(text, position) or INTERNVL_BOX_RE.match(text, position)
    if not match:
        return None
    try:
        return _box(map(int, match.groups())), match.end()
    except ValueError:
        return None

Box = tuple[int, int, int, int]


@dataclass
class GraphState:
    """Executed prediction state; entity IDs are internal evaluator handles only."""

    panels: list[Box] = field(default_factory=list)
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    speakers: set[tuple[str, str]] = field(default_factory=set)
    characters: set[tuple[str, str]] = field(default_factory=set)
    grounding: set[tuple[Box, int, int, str]] = field(default_factory=set)
    grounded_captions: dict[Box, str] = field(default_factory=dict)
    read_entities: set[str] = field(default_factory=set)

    def add_entity(self, kind: str, box: Box) -> str:
        prefix = "C" if kind == "CHAR" else "T"
        index = 1 + sum(item["type"] == kind for item in self.entities.values())
        entity_id = f"{prefix}{index}"
        self.entities[entity_id] = {
            "id": entity_id,
            "type": kind,
            "bbox": list(box),
            "content": "",
            "entered_panel_count": len(self.panels),
        }
        return entity_id

    def resolve(self, kind: str, box: Box) -> str | None:
        matches = [
            entity_id
            for entity_id, entity in self.entities.items()
            if entity["type"] == kind and tuple(entity["bbox"]) == box
        ]
        if matches:
            return matches[-1]
        best_id, best_iou = None, 0.95
        for entity_id, entity in self.entities.items():
            if entity["type"] != kind:
                continue
            score = iou(entity["bbox"], box)
            if score > best_iou:
                best_id, best_iou = entity_id, score
        return best_id


def clean_response(text: str) -> str:
    """Remove chat terminators while preserving both dialects' visual tags."""

    end = len(text)
    for token in CHAT_END_TOKENS:
        position = text.find(token)
        if position >= 0:
            end = min(end, position)
    return text[:end].strip()


def unescape_text(value: str) -> str:
    """Invert prepare_data.escape_text for text and reference payloads."""

    value = html.unescape(value)
    output: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\" or index + 1 == len(value):
            output.append(value[index])
            index += 1
            continue
        escaped = value[index + 1]
        output.append({"n": "\n", "t": "\t", "\\": "\\"}.get(escaped, "\\" + escaped))
        index += 2
    return "".join(output)


def _box(value: Iterable[int]) -> Box:
    box = tuple(map(int, value))
    if len(box) != 4 or not all(0 <= coordinate <= 1000 for coordinate in box):
        raise ValueError("invalid normalized box")
    if box[0] > box[2] or box[1] > box[3]:
        raise ValueError("inverted box")
    return box  # type: ignore[return-value]


def _parse_ref(text: str, position: int) -> tuple[str, Box, int] | None:
    markers = next(((start, end) for start, end in
                    ((REF_START, REF_END), ('<ref>', '</ref>'))
                    if text.startswith(start, position)), None)
    if markers is None:
        return None
    ref_start, ref_end = markers
    value_start = position + len(ref_start)
    value_end = text.find(ref_end, value_start)
    if value_end < 0:
        return None
    parsed_box = _parse_box(text, value_end + len(ref_end))
    if not parsed_box:
        return None
    box, end = parsed_box
    return unescape_text(text[value_start:value_end]), box, end


def _parse_text(text: str, position: int) -> tuple[str, int] | None:
    if not text.startswith("<text>", position) or not text.endswith("</text>"):
        return None
    start = position + len("<text>")
    return text[start:-len("</text>")], len(text)


def _target_panel_boxes(target: dict[str, Any]) -> list[Box]:
    return [_box(panel["bbox"]) for panel in target.get("panels", [])]


def _parse_grounded_caption(raw: str, state: GraphState) -> tuple[str, list[tuple[int, int, str]]] | None:
    plain: list[str] = []
    links: list[tuple[int, int, str]] = []
    length = 0
    position = 0
    while position < len(raw):
        markers = [i for tag in (REF_START, '<ref>') if (i := raw.find(tag, position)) >= 0]
        if not markers:
            plain.append(unescape_text(raw[position:]))
            break
        marker = min(markers)
        prefix = unescape_text(raw[position:marker])
        plain.append(prefix)
        length += len(prefix)
        parsed = _parse_ref(raw, marker)
        if not parsed:
            return None
        mention, box, next_position = parsed
        boxes = [box]
        while raw.startswith((BOX_START, '<box>'), next_position):
            extra_box = _parse_box(raw, next_position)
            if extra_box is None:
                return None
            box, next_position = extra_box
            boxes.append(box)
        entity_ids = [state.resolve("CHAR", candidate) for candidate in boxes]
        # Preserve unresolved references as prediction-only IDs. The metric
        # mapper cannot match these to GT, so each distinct grounding is an FP.
        entity_ids = [entity_id if entity_id is not None else f"UNREGISTERED:{box}"
                      for entity_id, box in zip(entity_ids, boxes)]
        start = length
        plain.append(mention)
        length += len(mention)
        links.extend((start, length, entity_id) for entity_id in entity_ids if entity_id is not None)
        position = next_position
    return "".join(plain), links


def _execute(line: str, state: GraphState, target: dict[str, Any]) -> bool:
    if line.startswith("<enter>"):
        parsed = _parse_ref(line, len("<enter>"))
        if not parsed or line[parsed[2] :] != "</enter>" or parsed[0] != "panel":
            return False
        panels = _target_panel_boxes(target)
        if len(state.panels) >= len(panels) or iou(parsed[1], panels[len(state.panels)]) <= 0.95:
            return False
        state.panels.append(panels[len(state.panels)])
        return True

    if line.startswith("<detect>"):
        parsed = _parse_ref(line, len("<detect>"))
        if not parsed or line[parsed[2] :] != "</detect>" or not state.panels:
            return False
        kind = {"character": "CHAR", "text": "TEXT"}.get(parsed[0])
        if kind is None:
            return False
        state.add_entity(kind, parsed[1])
        return True

    if line.startswith("<read>"):
        parsed = _parse_ref(line, len("<read>"))
        if not parsed or parsed[0] != "text":
            return False
        if not line.endswith("</read>"):
            return False
        content = _parse_text(line[: -len("</read>")], parsed[2])
        entity_id = state.resolve("TEXT", parsed[1])
        if not content or entity_id is None or entity_id in state.read_entities:
            return False
        state.entities[entity_id]["content"] = unescape_text(content[0])
        state.read_entities.add(entity_id)
        return True

    if line.startswith("<link_from>"):
        source = _parse_ref(line, len("<link_from>"))
        link_to_start = source[2] + len("</link_from>") if source else -1
        if (
            not source
            or not line.startswith("</link_from>", source[2])
            or not line.startswith("<link_to>", link_to_start)
        ):
            return False
        target_ref = _parse_ref(line, link_to_start + len("<link_to>"))
        if not target_ref or line[target_ref[2] :] != "</link_to>":
            return False
        source_kind = {"text": "TEXT", "character": "CHAR"}.get(source[0])
        target_kind = {"character": "CHAR"}.get(target_ref[0])
        if source_kind is None or target_kind is None:
            return False
        source_id = state.resolve(source_kind, source[1])
        target_id = state.resolve(target_kind, target_ref[1])
        if source_id is None or target_id is None or source_id == target_id:
            return False
        edges = state.speakers if source_kind == "TEXT" else state.characters
        edge = (source_id, target_id)
        if edge in edges:
            return False
        edges.add(edge)
        return True

    if line.startswith("<ground>"):
        panel = _parse_ref(line, len("<ground>"))
        if not panel or panel[0] != "panel":
            return False
        panel_box = max(state.panels, key=lambda box: iou(panel[1], box), default=None)
        if panel_box is None or iou(panel[1], panel_box) <= 0.95:
            return False
        if not line.endswith("</ground>"):
            return False
        content = _parse_text(line[: -len("</ground>")], panel[2])
        if not content or panel_box in state.grounded_captions:
            return False
        parsed_caption = _parse_grounded_caption(content[0], state)
        if parsed_caption is None:
            return False
        caption, links = parsed_caption
        state.grounded_captions[panel_box] = caption
        state.grounding.update((panel_box, start, end, entity_id) for start, end, entity_id in links)
        return True

    return False


def parse_response(response: str, target: dict[str, Any]) -> tuple[GraphState, list[str], list[str]]:
    """Execute one newline-delimited action at a time."""

    state = GraphState()
    valid: list[str] = []
    invalid: list[str] = []
    for raw_line in clean_response(response).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        (valid if _execute(line, state, target) else invalid).append(line)
    return state, valid, invalid


def iou(left: Iterable[int], right: Iterable[int]) -> float:
    lx0, ly0, lx1, ly1 = map(float, left)
    rx0, ry0, rx1, ry1 = map(float, right)
    width = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    height = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    intersection = width * height
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def target_entities(target: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for prefix, kind, key in (("C", "CHAR", "characters"), ("T", "TEXT", "texts")):
        for index, item in enumerate(target.get(key, []), 1):
            entity_id = f"{prefix}{index}"
            entities[entity_id] = {"id": entity_id, "type": kind, **item}
    return entities


def match_entities(
    state: GraphState,
    target: dict[str, Any],
    iou_threshold: float,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Greedily match detections by type and IoU, respecting entered panels."""

    gt_entities = target_entities(target)
    panel_index = {tuple(panel["bbox"]): index + 1 for index, panel in enumerate(target.get("panels", []))}
    mapping: dict[str, str] = {}
    by_type: dict[str, dict[str, str]] = {"CHAR": {}, "TEXT": {}}
    for kind in by_type:
        candidates = []
        for pred_id, pred in state.entities.items():
            if pred["type"] != kind:
                continue
            for gt_id, gt in gt_entities.items():
                if gt["type"] != kind:
                    continue
                gt_panel = panel_index.get(tuple(gt["panel_bbox"]), len(panel_index) + 1)
                if gt_panel > int(pred["entered_panel_count"]):
                    continue
                candidates.append((iou(pred["bbox"], gt["bbox"]), pred_id, gt_id))
        used_pred: set[str] = set()
        used_gt: set[str] = set()
        for score, pred_id, gt_id in sorted(candidates, reverse=True):
            if score < iou_threshold:
                break
            if pred_id in used_pred or gt_id in used_gt:
                continue
            used_pred.add(pred_id)
            used_gt.add(gt_id)
            mapping[pred_id] = gt_id
            by_type[kind][pred_id] = gt_id
    return mapping, by_type


def state_to_dict(state: GraphState) -> dict[str, Any]:
    return {
        "panels": [list(box) for box in state.panels],
        "entities": list(state.entities.values()),
        "speaker_links": [list(edge) for edge in sorted(state.speakers)],
        "character_links": [list(edge) for edge in sorted(state.characters)],
        "grounding_links": [[list(panel), start, end, entity] for panel, start, end, entity in sorted(state.grounding)],
        "grounded_captions": [
            {"panel_bbox": list(panel), "caption": caption}
            for panel, caption in state.grounded_captions.items()
        ],
    }
