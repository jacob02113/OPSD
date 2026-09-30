"""Scene-graph state, command rewards, and privileged-image rendering.

The executor records only successful graph progress.  Candidate scoring is pure:
all candidates in a GRPO group are evaluated against the same pre-command state,
then one uniformly sampled anchor is executed to continue the rollout.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

from PIL import Image, ImageDraw

REF_START, REF_END = "<|object_ref_start|>", "<|object_ref_end|>"
BOX_START, BOX_END = "<|box_start|>", "<|box_end|>"
ACTION_TAGS = ("<enter>", "<detect>", "<read>", "<link_from>", "<ground>")
CHAT_END_TOKENS = ("<|im_end|>", "<|endoftext|>")

COORDS_PATTERN = r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
BOX_PATTERN = re.escape(BOX_START) + COORDS_PATTERN + re.escape(BOX_END)
BOX_RE = re.compile(BOX_PATTERN)
ENTER_RE = re.compile(
    r"^<enter>"
    + re.escape(REF_START)
    + r"panel"
    + re.escape(REF_END)
    + f"({BOX_PATTERN})</enter>$"
)
DETECT_RE = re.compile(
    r"^<detect>"
    + re.escape(REF_START)
    + r"(text|character)"
    + re.escape(REF_END)
    + f"({BOX_PATTERN})</detect>$"
)
READ_RE = re.compile(
    r"^<read>"
    + re.escape(REF_START)
    + r"text"
    + re.escape(REF_END)
    + f"({BOX_PATTERN})<text>(.*)</text></read>$",
    re.DOTALL,
)
LINK_RE = re.compile(
    r"^<link_from>"
    + re.escape(REF_START)
    + r"(text|character)"
    + re.escape(REF_END)
    + f"({BOX_PATTERN})"
    + r"</link_from><link_to>"
    + re.escape(REF_START)
    + r"character"
    + re.escape(REF_END)
    + f"({BOX_PATTERN})</link_to>$"
)
GROUND_RE = re.compile(
    r"^<ground>"
    + re.escape(REF_START)
    + r"panel"
    + re.escape(REF_END)
    + f"({BOX_PATTERN})<text>(.*)</text></ground>$",
    re.DOTALL,
)

Box = tuple[int, int, int, int]
Color = tuple[int, int, int]

DONE_COLOR: Color = (36, 166, 72)
READY_COLOR: Color = (245, 139, 0)
BLOCKED_COLOR = (128, 128, 128, 210)


def _box(values: Iterable[Any]) -> Box:
    result = tuple(map(int, values))
    if len(result) != 4 or not all(0 <= value <= 1000 for value in result):
        raise ValueError(f"Invalid normalized bbox: {result!r}")
    if result[0] > result[2] or result[1] > result[3]:
        raise ValueError(f"Inverted normalized bbox: {result!r}")
    return result  # type: ignore[return-value]


def _match_box(match: re.Match[str], group: int) -> Box:
    return _box(match.groups()[group : group + 4])


def iou(left: Box, right: Box) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _normalize_text(value: Any) -> str:
    text = html.unescape(str(value))
    text = text.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).strip())


def _edit_distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        previous = row[0]
        row[0] = i
        for j, b in enumerate(right, 1):
            current = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (a != b))
            previous = current
    return row[-1]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def add(self, item: int) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


@dataclass(frozen=True)
class TargetGraph:
    raw: dict[str, Any]
    panels: tuple[dict[str, Any], ...]
    characters: tuple[dict[str, Any], ...]
    texts: tuple[dict[str, Any], ...]
    groundings: tuple[dict[str, Any], ...]
    panel_boxes: tuple[Box, ...]
    character_boxes: tuple[Box, ...]
    text_boxes: tuple[Box, ...]
    character_panel: tuple[int, ...]
    text_panel: tuple[int, ...]
    clusters: tuple[str, ...]
    speaker_links: frozenset[tuple[int, int]]

    @classmethod
    def parse(cls, value: str | dict[str, Any]) -> "TargetGraph":
        raw = json.loads(value) if isinstance(value, str) else value
        panels = tuple(raw.get("panels", ()))
        characters = tuple(raw.get("characters", ()))
        texts = tuple(raw.get("texts", ()))
        panel_boxes = tuple(_box(item["bbox"]) for item in panels)
        character_boxes = tuple(_box(item["bbox"]) for item in characters)
        text_boxes = tuple(_box(item["bbox"]) for item in texts)
        panel_index = {box: index for index, box in enumerate(panel_boxes)}
        character_index = {box: index for index, box in enumerate(character_boxes)}
        text_index = {box: index for index, box in enumerate(text_boxes)}
        speaker_links = frozenset(
            (text_index[_box(edge["text_bbox"])], character_index[_box(edge["character_bbox"])])
            for edge in raw.get("speaker_links", ())
            if _box(edge["text_bbox"]) in text_index and _box(edge["character_bbox"]) in character_index
        )
        return cls(
            raw=raw,
            panels=panels,
            characters=characters,
            texts=texts,
            groundings=tuple(raw.get("groundings", ())),
            panel_boxes=panel_boxes,
            character_boxes=character_boxes,
            text_boxes=text_boxes,
            character_panel=tuple(panel_index[_box(item["panel_bbox"])] for item in characters),
            text_panel=tuple(panel_index[_box(item["panel_bbox"])] for item in texts),
            clusters=tuple(str(item.get("cluster", f"singleton-{i}")) for i, item in enumerate(characters)),
            speaker_links=speaker_links,
        )


@dataclass
class GraphState:
    entered: int = 0
    detected_characters: dict[int, Box] = field(default_factory=dict)
    detected_texts: dict[int, Box] = field(default_factory=dict)
    character_by_student_box: dict[Box, int] = field(default_factory=dict)
    text_by_student_box: dict[Box, int] = field(default_factory=dict)
    reads: set[int] = field(default_factory=set)
    speakers: set[tuple[int, int]] = field(default_factory=set)
    identity_links: set[tuple[int, int]] = field(default_factory=set)
    identities: UnionFind = field(default_factory=UnionFind)
    groundings: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class CandidateEvaluation:
    kind: str
    reward: float
    syntax_valid: bool
    executable: bool


class MangaExecutor:
    """Pure candidate evaluation plus mutation of one anchor GraphState."""

    def __init__(self, target: TargetGraph, iou_threshold: float = 0.5) -> None:
        self.target = target
        self.iou_threshold = iou_threshold
        self.state = GraphState()
        self.panel_by_box = {box: index for index, box in enumerate(target.panel_boxes)}
        self.grounding_by_panel = {
            self.panel_by_box[_box(item["panel_bbox"])]: item
            for item in target.groundings
            if _box(item["panel_bbox"]) in self.panel_by_box
        }

    @staticmethod
    def action_tag(command: str) -> str | None:
        return next((tag for tag in ACTION_TAGS if command.startswith(tag)), None)

    @staticmethod
    def action_kind(command: str) -> str:
        tag = MangaExecutor.action_tag(command)
        return {
            "<enter>": "enter",
            "<detect>": "detect",
            "<read>": "read",
            "<link_from>": "link",
            "<ground>": "ground",
        }.get(tag, "invalid")

    def _eligible_detection(self, kind: str, predicted: Box) -> tuple[int, float] | None:
        if kind == "character":
            boxes = self.target.character_boxes
            owners = self.target.character_panel
            detected = self.state.detected_characters
        else:
            boxes = self.target.text_boxes
            owners = self.target.text_panel
            detected = self.state.detected_texts
        candidates = [
            (index, iou(predicted, box))
            for index, box in enumerate(boxes)
            if index not in detected and owners[index] < self.state.entered
        ]
        return max(candidates, key=lambda item: (item[1], -item[0])) if candidates else None

    def _ground_required(self, panel_index: int) -> set[int]:
        grounding = self.grounding_by_panel[panel_index]
        character_index = {box: index for index, box in enumerate(self.target.character_boxes)}
        return {
            character_index[_box(box)]
            for mention in grounding.get("mentions", ())
            for box in mention.get("character_bboxes", ())
            if _box(box) in character_index
        }

    def frontier_kinds(self) -> set[str]:
        kinds: set[str] = set()
        if self.state.entered < len(self.target.panels):
            from .command_teacher import dependency_graph, node_done
            dependencies = dependency_graph(self)[('enter', self.state.entered)]
            if all(node_done(self, node) for node in dependencies):
                kinds.add("enter")
        if any(
            index not in self.state.detected_characters and owner < self.state.entered
            for index, owner in enumerate(self.target.character_panel)
        ) or any(
            index not in self.state.detected_texts and owner < self.state.entered
            for index, owner in enumerate(self.target.text_panel)
        ):
            kinds.add("detect")
        if any(index not in self.state.reads for index in self.state.detected_texts):
            kinds.add("read")
        if any(
            text in self.state.reads
            and character in self.state.detected_characters
            and (text, character) not in self.state.speakers
            for text, character in self.target.speaker_links
        ):
            kinds.add("link")
        by_cluster: dict[str, list[int]] = defaultdict(list)
        for index in self.state.detected_characters:
            by_cluster[self.target.clusters[index]].append(index)
        if any(len({self.state.identities.find(item) for item in members}) > 1 for members in by_cluster.values()):
            kinds.add("link")
        for panel_index in self.grounding_by_panel:
            if (
                panel_index < self.state.entered
                and panel_index not in self.state.groundings
                and self._ground_required(panel_index).issubset(self.state.detected_characters)
            ):
                kinds.add("ground")
        return kinds

    def complete(self) -> bool:
        return not self.frontier_kinds()

    def read_content_target(self, command: str) -> tuple[int | None, bool]:
        """Resolve one OCR target independently of candidate text/execution.

        Unregistered boxes require IoU >= 0.5. Registered references
        always retain their original identity.
        """
        prefix = CommandLayout.grpo_prefix(command)
        if not command.startswith("<read>") or prefix is None:
            return None, False
        match = BOX_RE.search(prefix)
        box = _match_box(match, 0)
        registered = self.resolve_reference(self.state.text_by_student_box, box)
        if registered is not None:
            return (registered, False) if registered not in self.state.reads else (None, False)
        ranked = sorted(((iou(box, gt), i) for i, gt in enumerate(self.target.text_boxes)), reverse=True)
        if not ranked or ranked[0][0] < 0.5:
            return None, False
        return ranked[0][1], True

    def read_content_reward(self, command: str, target_index: int) -> float:
        """Quality-only reward; never registers an object or completes READ."""
        match = READ_RE.fullmatch(command)
        if match is None:
            return 0.0
        predicted = _normalize_text(match.group(6))
        expected = _normalize_text(self.target.texts[target_index].get("content", ""))
        return 1.0 - _edit_distance(predicted, expected) / max(1, len(predicted), len(expected))

    def _ground_entity_f1(self, panel_index: int, raw_caption: str) -> float:
        grounding = self.grounding_by_panel[panel_index]
        gt = {
            (int(mention["start"]), int(mention["end"]), character_index)
            for mention in grounding.get("mentions", ())
            for raw_box in mention.get("character_bboxes", ())
            for character_index, box in enumerate(self.target.character_boxes)
            if box == _box(raw_box)
        }

        predicted: set[tuple[int, int, int]] = set()
        # Keep unresolved references as false positives instead of dropping
        # them. Deduplicate by mention span and raw box, matching the set
        # semantics used for resolved entity references.
        unresolved: set[tuple[int, int, Box]] = set()
        plain_length = 0
        cursor = 0
        while cursor < len(raw_caption):
            marker = raw_caption.find(REF_START, cursor)
            if marker < 0:
                break
            plain_length += len(html.unescape(raw_caption[cursor:marker]))
            ref_end = raw_caption.find(REF_END, marker + len(REF_START))
            if ref_end < 0:
                return 0.0
            mention = html.unescape(raw_caption[marker + len(REF_START) : ref_end])
            box_match = BOX_RE.match(raw_caption, ref_end + len(REF_END))
            if box_match is None:
                return 0.0
            start, end = plain_length, plain_length + len(mention)
            while box_match is not None:
                box = _match_box(box_match, 0)
                index = self.resolve_reference(self.state.character_by_student_box, box)
                if index is not None:
                    predicted.add((start, end, index))
                else:
                    unresolved.add((start, end, box))
                cursor = box_match.end()
                box_match = BOX_RE.match(raw_caption, cursor)
            plain_length = end
        tp = len(predicted & gt)
        fp = len(predicted - gt) + len(unresolved)
        fn = len(gt - predicted)
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0

    @staticmethod
    def resolve_reference(mapping, box):
        """Resolve a reference against registered boxes; never register an alias."""
        if box in mapping:
            return mapping[box]
        best, score = None, 0.95
        for registered, index in mapping.items():
            overlap = iou(box, registered)
            if overlap > score:
                best, score = index, overlap
        return best

    def evaluate(self, command: str, *, compute_reward: bool = True) -> CandidateEvaluation:
        """Score a student command without allowing bad arguments to abort rollout.

        Target data is validated while parsing ``TargetGraph``.  Student boxes,
        however, are sampled model outputs and may be inverted or out of range.
        Such a command is a valid GRPO negative: it receives zero reward and is
        not executable, so ``execute`` leaves ``GraphState`` unchanged.
        """
        try:
            evaluation = self._evaluate(command, compute_reward=compute_reward)
            if not compute_reward:
                return CandidateEvaluation(evaluation.kind, 0.0, evaluation.syntax_valid, evaluation.executable)
            return evaluation
        except ValueError:
            return CandidateEvaluation(self.action_kind(command.strip()), 0.0, True, False)

    def _evaluate(self, command: str, *, compute_reward: bool = True) -> CandidateEvaluation:
        command = command.strip()
        kind = self.action_kind(command)
        frontier = self.frontier_kinds()

        match = ENTER_RE.fullmatch(command)
        if match:
            box = _match_box(match, 1)
            ok = (
                "enter" in frontier
                and self.state.entered < len(self.target.panel_boxes)
                and self.resolve_reference(self.panel_by_box, box) == self.state.entered
            )
            return CandidateEvaluation("enter", float(ok), True, ok)

        match = DETECT_RE.fullmatch(command)
        if match:
            best = self._eligible_detection(match.group(1), _match_box(match, 2)) if "detect" in frontier else None
            reward = best[1] if best is not None else 0.0
            executable = best is not None and reward >= self.iou_threshold
            return CandidateEvaluation("detect", reward, True, executable)

        match = READ_RE.fullmatch(command)
        if match:
            index = self.resolve_reference(self.state.text_by_student_box, _match_box(match, 1))
            executable = "read" in frontier and index is not None and index not in self.state.reads
            if not executable:
                return CandidateEvaluation("read", 0.0, True, False)
            if not compute_reward:
                return CandidateEvaluation("read", 0.0, True, True)
            predicted = _normalize_text(match.group(6))
            expected = _normalize_text(self.target.texts[index].get("content", ""))
            distance = _edit_distance(predicted, expected)
            reward = 1.0 - distance / max(1, len(predicted), len(expected))
            return CandidateEvaluation("read", reward, True, True)

        match = LINK_RE.fullmatch(command)
        if match:
            source_kind = match.group(1)
            source_box, target_box = _match_box(match, 2), _match_box(match, 7)
            target_index = self.resolve_reference(self.state.character_by_student_box, target_box)
            executable = False
            if "link" in frontier and target_index is not None:
                if source_kind == "text":
                    source_index = self.resolve_reference(self.state.text_by_student_box, source_box)
                    edge = (source_index, target_index)
                    executable = (
                        source_index is not None
                        and source_index in self.state.reads
                        and edge in self.target.speaker_links
                        and edge not in self.state.speakers
                    )
                else:
                    source_index = self.resolve_reference(self.state.character_by_student_box, source_box)
                    executable = (
                        source_index is not None
                        and source_index != target_index
                        and self.target.clusters[source_index] == self.target.clusters[target_index]
                        and self.state.identities.find(source_index) != self.state.identities.find(target_index)
                    )
            return CandidateEvaluation("link", float(executable), True, executable)

        match = GROUND_RE.fullmatch(command)
        if match:
            panel_index = self.resolve_reference(self.panel_by_box, _match_box(match, 1))
            executable = (
                "ground" in frontier
                and panel_index is not None
                and panel_index in self.grounding_by_panel
                and panel_index not in self.state.groundings
                and panel_index < self.state.entered
                and self._ground_required(panel_index).issubset(self.state.detected_characters)
            )
            # Payload quality cannot change whether the panel node is completed.
            # Unregistered references reduce quality; malformed payload boxes may
            # also fail reward parsing, but do not invalidate the panel action.
            reward = 0.0
            if executable and compute_reward:
                try:
                    reward = self._ground_entity_f1(panel_index, match.group(6))
                except ValueError:
                    reward = 0.0
            return CandidateEvaluation("ground", reward, True, executable)

        return CandidateEvaluation(kind, 0.0, False, False)

    def execute(self, command: str) -> bool:
        evaluation = self.evaluate(command, compute_reward=False)
        if not evaluation.executable:
            return False
        command = command.strip()
        if match := ENTER_RE.fullmatch(command):
            self.state.entered += 1
        elif match := DETECT_RE.fullmatch(command):
            kind, predicted = match.group(1), _match_box(match, 2)
            best = self._eligible_detection(kind, predicted)
            assert best is not None and best[1] >= self.iou_threshold
            index = best[0]
            if kind == "character":
                self.state.detected_characters[index] = predicted
                self.state.character_by_student_box[predicted] = index
                self.state.identities.add(index)
            else:
                self.state.detected_texts[index] = predicted
                self.state.text_by_student_box[predicted] = index
        elif match := READ_RE.fullmatch(command):
            index = self.resolve_reference(self.state.text_by_student_box, _match_box(match, 1))
            self.state.reads.add(index)
        elif match := LINK_RE.fullmatch(command):
            source_box, target_box = _match_box(match, 2), _match_box(match, 7)
            target_index = self.resolve_reference(self.state.character_by_student_box, target_box)
            if match.group(1) == "text":
                self.state.speakers.add((self.resolve_reference(self.state.text_by_student_box, source_box), target_index))
            else:
                source_index = self.resolve_reference(self.state.character_by_student_box, source_box)
                self.state.identities.union(source_index, target_index)
                self.state.identity_links.add(tuple(sorted((source_index, target_index))))
        elif match := GROUND_RE.fullmatch(command):
            self.state.groundings.add(self.resolve_reference(self.panel_by_box, _match_box(match, 1)))
        return True

    def evaluate_without_mutation(self, command: str) -> CandidateEvaluation:
        return deepcopy(self).evaluate(command)


@dataclass(frozen=True)
class CommandLayout:
    """Character spans used to assign planning, structure, and GRPO masks."""

    syntax_valid: bool
    variable_spans: tuple[tuple[int, int], ...]
    planning_spans: tuple[tuple[int, int], ...]
    ignored_spans: tuple[tuple[int, int], ...] = ()

    @staticmethod
    def finite_plan_decisions(command: str) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
        """Refine finite choices; references and unknown spellings stay unchanged.

        Use the entire DSL vocabulary, never the state's legal frontier. A
        single legal action still needs planning supervision. Character branch
        points are mapped to the original generated tokens by the agent loop.
        """
        replacements = {}
        tag = MangaExecutor.action_tag(command)
        if tag is None:
            return replacements

        def branches(value: str, choices: tuple[str, ...], start: int):
            return tuple(
                (start + i, start + i + 1)
                for i in range(len(value))
                if len({choice[i:i + 1] for choice in choices if choice.startswith(value[:i])}) > 1
            )

        # The initial token also chooses continuation over EOS. EOS itself is
        # handled separately, including when attached to a completed command.
        replacements[(0, len(tag))] = ((0, 1),) + branches(tag, tuple(ACTION_TAGS), 0)
        prefix = tag + REF_START
        if tag == "<detect>" and command.startswith(prefix):
            start = len(prefix)
            end = command.find(REF_END, start)
            category = command[start:end] if end >= 0 else ""
            if category in ("text", "character"):
                replacements[(start, end)] = branches(category, ("text", "character"), start)
        return replacements

    @staticmethod
    def unclosed_content_span(command: str) -> tuple[int, int] | None:
        """Identify unfinished payload only after a valid content prefix.

        Stop before any tag fragment: generated delimiters remain OPSD targets.
        A complete inner delimiter with a bad outer tag is a structure error.
        """
        prefix = CommandLayout.grpo_prefix(command)
        if prefix is None:
            return None
        closing = BOX_END if command.startswith("<detect>") else "</text>"
        body = command[len(prefix):]
        if closing in body:
            return None
        cursor = len(prefix)
        if command.startswith("<ground>"):
            # A grounded caption contains mention references and one or more
            # boxes. Skip these complete internal structures before looking
            # for the outer content terminator; do not pair across commands.
            while True:
                marker = command.find("<", cursor)
                if marker < 0 or not command.startswith(REF_START, marker):
                    break
                ref_end = command.find(REF_END, marker + len(REF_START))
                if ref_end < 0 or "<" in command[marker + len(REF_START):ref_end]:
                    return None
                box = BOX_RE.match(command, ref_end + len(REF_END))
                if box is None:
                    return None
                while box is not None:
                    cursor = box.end()
                    box = BOX_RE.match(command, cursor)
        end = command.find("<", cursor)
        if end < 0:
            end = len(command)
        # A partial closing delimiter is allowed, but unrelated/nested tags
        # must not turn malformed DSL into a content-quality comparison.
        if end < len(command) and not closing.startswith(command[end:]):
            return None
        return (len(prefix), end)

    @staticmethod
    def zero_reward_unclosed_indices(commands: list[str], rewards: list[float]) -> list[int]:
        """Add failure credit only where relative rewards provide no signal."""
        if len(commands) < 2 or len(commands) != len(rewards) or any(r != 0.0 for r in rewards):
            return []
        return [i for i, command in enumerate(commands)
                if CommandLayout.unclosed_content_span(command) is not None]

    @staticmethod
    def all_unclosed_content(commands: list[str]) -> bool:
        return len(commands) > 1 and all(
            CommandLayout.unclosed_content_span(command) is not None for command in commands
        )

    @staticmethod
    def paired_content_spans(command: str) -> tuple[tuple[int, int], ...]:
        """Only replace content inside a complete, command-local tag pair."""
        prefix = CommandLayout.grpo_prefix(command)
        if prefix is None:
            return ()
        tag = MangaExecutor.action_tag(command)
        opening, closing = (BOX_START, BOX_END) if tag == "<detect>" else ("<text>", "</text>")
        outer_close = "</" + tag[1:]
        suffix = closing + outer_close
        if not command.endswith(suffix):
            return ()
        start, end = len(prefix), len(command) - len(suffix)
        if end < start:
            return ()
        content = command[start:end]
        # Never pair delimiters across commands or nested/repeated content tags.
        if any(marker in content for marker in (opening, closing, *ACTION_TAGS)):
            return ()
        if any(("</" + action[1:]) in content for action in ACTION_TAGS):
            return ()
        return ((start, end),) if start < end else ()

    @staticmethod
    def grpo_prefix(command: str) -> str | None:
        """Validate the fixed decision prefix before sampling answer candidates."""
        for tag, kind, tail in (
            ("detect", "(?:text|character)", re.escape(BOX_START)),
            ("read", "text", BOX_PATTERN + re.escape("<text>")),
            ("ground", "panel", BOX_PATTERN + re.escape("<text>")),
        ):
            pattern = (
                f"^<{tag}>" + re.escape(REF_START) + kind + re.escape(REF_END) + tail
            )
            match = re.match(pattern, command)
            if match:
                # A syntactically complete but inverted/out-of-range target
                # cannot provide a meaningful fixed READ/GROUND condition.
                if tag != "detect":
                    try:
                        _box(match.groups())
                    except ValueError:
                        return None
                return match.group(0)
        return None

    @classmethod
    def parse(cls, command: str) -> "CommandLayout":
        patterns = (
            # References to objects already present in GraphState are planning
            # decisions supervised by the privileged OPSD teacher.  Only newly
            # predicted outputs (detection boxes and text answers) belong to
            # command-local GRPO.
            (ENTER_RE, (), (1,)),
            (DETECT_RE, (2,), (1,)),
            (READ_RE, (6,), (1,)),
            (LINK_RE, (), (1, 2, 7)),
            (GROUND_RE, (6,), (1,)),
        )
        for pattern, variable_groups, planning_groups in patterns:
            match = pattern.fullmatch(command)
            if match is None:
                continue

            def semantic_spans(group: int) -> list[tuple[int, int]]:
                """Keep box values semantic while leaving fixed punctuation structural."""

                start, end = match.span(group)
                if not command.startswith(BOX_START, start):
                    return [(start, end)]

                # A referenced box selects an existing object (plan), whereas a
                # detected box predicts a new object (GRPO).  In either case only
                # the coordinate values carry that semantic choice.  The box
                # wrappers and ``(x,y),(x,y)`` punctuation are deterministic DSL
                # syntax and therefore remain in the structure branch.
                value_start = start + len(BOX_START)
                value_end = end - len(BOX_END)
                return [
                    (value_start + number.start(), value_start + number.end())
                    for number in re.finditer(r"\d+", command[value_start:value_end])
                ]

            tag = MangaExecutor.action_tag(command)
            if tag is None:
                raise AssertionError("A valid command must start with a known action tag.")
            spans = []
            for group in variable_groups:
                spans.extend(semantic_spans(group))
            planning_spans = [(0, len(tag))]
            for group in planning_groups:
                planning_spans.extend(semantic_spans(group))
            return cls(True, tuple(spans), tuple(planning_spans))

        tag = MangaExecutor.action_tag(command)
        # Invalid syntax receives no GRPO loss. Retain identifiable decisions
        # and let OPSD correct malformed wrappers instead of routing them to
        # an all-zero reward group. Do not distill OCR/ground answers merely
        # because their closing syntax is malformed.
        if tag is None:
            return cls(False, (), ((0, len(command)),))
        planning_spans = [(0, len(tag))]
        ignored_spans = []
        content_start = command.find("<text>", len(tag)) if tag in {"<read>", "<ground>"} else -1
        prefix_end = content_start if content_start >= 0 else len(command)
        prefix = command[:prefix_end]
        ref_pattern = re.escape(REF_START) + r"([^<]*)" + re.escape(REF_END)
        for index, match in enumerate(re.finditer(ref_pattern, prefix)):
            if tag == "<detect>" or (tag == "<link_from>" and index == 0):
                planning_spans.append(match.span(1))
        for match in BOX_RE.finditer(prefix):
            for group in range(1, 5):
                if tag == "<detect>":
                    ignored_spans.append(match.span(group))
                else:
                    planning_spans.append(match.span(group))
        if tag == "<detect>":
            # Also exclude numbers in incomplete boxes from structure KL.
            box_start = prefix.find(BOX_START)
            if box_start >= 0:
                box_end = prefix.find(BOX_END, box_start)
                box_end = box_end if box_end >= 0 else len(prefix)
                ignored_spans.extend(
                    (box_start + m.start(), box_start + m.end())
                    for m in re.finditer(r"\d+", prefix[box_start:box_end])
                )
        if content_start >= 0:
            body_start = content_start + len("<text>")
            closing = command.find("</", body_start)
            ignored_spans.append((body_start, closing if closing >= 0 else len(command)))
        return cls(False, (), tuple(planning_spans), tuple(ignored_spans))


def _pixel_box(box: Box, width: int, height: int) -> Box:
    return (
        round(box[0] * (width - 1) / 1000),
        round(box[1] * (height - 1) / 1000),
        round(box[2] * (width - 1) / 1000),
        round(box[3] * (height - 1) / 1000),
    )


def _center(box: Box) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: Color,
    width: int,
) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = max((dx * dx + dy * dy) ** 0.5, 1.0)
    dash = max(6, width * 3)
    gap = max(4, width * 2)
    position = 0.0
    while position < distance:
        finish = min(position + dash, distance)
        left, right = position / distance, finish / distance
        draw.line(
            (
                (round(start[0] + dx * left), round(start[1] + dy * left)),
                (round(start[0] + dx * right), round(start[1] + dy * right)),
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def _grounding_requirements(target: TargetGraph) -> dict[int, set[int]]:
    panel_index = {box: index for index, box in enumerate(target.panel_boxes)}
    character_index = {box: index for index, box in enumerate(target.character_boxes)}
    requirements: dict[int, set[int]] = {}
    for grounding in target.groundings:
        panel_box = _box(grounding["panel_bbox"])
        if panel_box not in panel_index:
            continue
        requirements[panel_index[panel_box]] = {
            character_index[_box(raw_box)]
            for mention in grounding.get("mentions", ())
            for raw_box in mention.get("character_bboxes", ())
            if _box(raw_box) in character_index
        }
    return requirements


def render_privileged_image(
    original: Image.Image,
    target: TargetGraph,
    state: GraphState,
    *,
    reveal_next_panel: bool = False,
    seed: int = 0,
) -> Image.Image:
    """Render only the privileged planning state and its full legal frontier.

    Green denotes already committed graph state, orange denotes operations that
    are executable now, and gray masks panels that are not yet available.  The
    overlay deliberately contains no OCR transcription or grounding caption:
    those answer-quality fields are supervised by command-local GRPO.
    """

    image = original.convert("RGB").copy()
    width, height = image.size
    line_width = max(2, round(min(width, height) / 300))
    emphasis_width = max(line_width + 2, line_width * 2)
    del seed  # Fixed semantic colors must be identical in training and inference.

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    revealed = state.entered + int(reveal_next_panel and state.entered < len(target.panels))
    for panel_index in range(revealed, len(target.panels)):
        overlay_draw.rectangle(
            _pixel_box(target.panel_boxes[panel_index], width, height),
            fill=BLOCKED_COLOR,
        )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    requirements = _grounding_requirements(target)
    # Registered references use the student's exact coordinates. Keep GT boxes
    # only for unseen objects; otherwise the overlay suggests references the
    # executor cannot resolve even when their IoU is excellent.
    character_boxes = {
        i: state.detected_characters.get(i, box) for i, box in enumerate(target.character_boxes)
    }
    text_boxes = {i: state.detected_texts.get(i, box) for i, box in enumerate(target.text_boxes)}
    for panel_index in range(state.entered):
        if panel_index in state.groundings:
            color = DONE_COLOR
        elif panel_index in requirements and requirements[panel_index].issubset(state.detected_characters):
            color = READY_COLOR
        else:
            continue
        draw.rectangle(_pixel_box(target.panel_boxes[panel_index], width, height), outline=color, width=line_width)
    if state.entered < len(target.panels):
        # ENTER is always a legal planning choice for the next panel.  Its
        # contents remain gray until the execution view explicitly reveals it.
        draw.rectangle(
            _pixel_box(target.panel_boxes[state.entered], width, height),
            outline=READY_COLOR,
            width=emphasis_width,
        )

    # Completed relations are green. Every currently executable relation is
    # drawn explicitly in orange, avoiding the ambiguity of the old scheme in
    # which an entire connected component merely shared one random color.
    for left, right in sorted(state.identity_links):
        _draw_dashed_line(
            draw,
            _center(_pixel_box(character_boxes[left], width, height)),
            _center(_pixel_box(character_boxes[right], width, height)),
            DONE_COLOR,
            line_width,
        )
    for text_index, character_index in sorted(state.speakers):
        draw.line(
            (_center(_pixel_box(text_boxes[text_index], width, height)),
             _center(_pixel_box(character_boxes[character_index], width, height))),
            fill=DONE_COLOR,
            width=line_width,
        )

    for text_index, character_index in sorted(target.speaker_links):
        edge = (text_index, character_index)
        if (
            text_index in state.reads
            and character_index in state.detected_characters
            and edge not in state.speakers
        ):
            draw.line(
                (
                    _center(_pixel_box(text_boxes[text_index], width, height)),
                    _center(_pixel_box(character_boxes[character_index], width, height)),
                ),
                fill=READY_COLOR,
                width=line_width,
            )

    detected_characters = sorted(state.detected_characters)
    for offset, left in enumerate(detected_characters):
        for right in detected_characters[offset + 1 :]:
            if target.clusters[left] != target.clusters[right]:
                continue
            if state.identities.find(left) == state.identities.find(right):
                continue
            _draw_dashed_line(
                draw,
                _center(_pixel_box(character_boxes[left], width, height)),
                _center(_pixel_box(character_boxes[right], width, height)),
                READY_COLOR,
                line_width,
            )

    for index, box in character_boxes.items():
        if target.character_panel[index] >= state.entered:
            continue
        color = DONE_COLOR if index in state.detected_characters else READY_COLOR
        draw.rectangle(_pixel_box(box, width, height), outline=color, width=line_width)
    for index, box in text_boxes.items():
        if target.text_panel[index] >= state.entered:
            continue
        color = DONE_COLOR if index in state.detected_texts and index in state.reads else READY_COLOR
        draw.rectangle(_pixel_box(box, width, height), outline=color, width=line_width)
    return image
