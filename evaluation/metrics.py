"""Metrics for the unified manga DSL output."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

if __package__:
    from .dsl import GraphState, match_entities, target_entities
else:
    from dsl import GraphState, match_entities, target_entities


def _ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text)).strip())


def _edit_distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        previous = row[0]
        row[0] = i
        for j, right in enumerate(b, 1):
            current = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (left != right))
            previous = current
    return row[-1]


def _clusters(ids: Iterable[str], links: Iterable[tuple[str, str]]) -> dict[str, int]:
    parent = {item: item for item in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in links:
        if left in parent and right in parent:
            parent[find(left)] = find(right)
    labels: dict[str, int] = {}
    roots: dict[str, int] = {}
    for item in parent:
        root = find(item)
        labels[item] = roots.setdefault(root, len(roots))
    return labels


def _cluster_pairs(labels: dict[str, int]) -> set[tuple[str, str]]:
    ids = sorted(labels)
    return {(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :] if labels[a] == labels[b]}


def _entropy(counts: Iterable[int], total: int) -> float:
    return -sum((count / total) * math.log(count / total) for count in counts if count)


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _ami(left: list[int], right: list[int]) -> float:
    """Adjusted mutual information with arithmetic-mean normalization."""

    total = len(left)
    if total <= 1:
        return 1.0
    left_counts, right_counts = Counter(left), Counter(right)
    cells = Counter(zip(left, right))
    mutual_info = sum(
        (count / total) * math.log(total * count / (left_counts[a] * right_counts[b]))
        for (a, b), count in cells.items()
        if count
    )
    expected = 0.0
    for a_count in left_counts.values():
        for b_count in right_counts.values():
            lower = max(1, a_count + b_count - total)
            upper = min(a_count, b_count)
            for overlap in range(lower, upper + 1):
                probability = math.exp(
                    _log_choose(a_count, overlap)
                    + _log_choose(total - a_count, b_count - overlap)
                    - _log_choose(total, b_count)
                )
                expected += probability * (overlap / total) * math.log(
                    total * overlap / (a_count * b_count)
                )
    normalizer = (
        _entropy(left_counts.values(), total) + _entropy(right_counts.values(), total)
    ) / 2.0
    denominator = normalizer - expected
    if abs(denominator) < 1e-12:
        same_partition = all(
            (left[i] == left[j]) == (right[i] == right[j])
            for i in range(total)
            for j in range(i + 1, total)
        )
        return 1.0 if same_partition else 0.0
    return (mutual_info - expected) / denominator


def evaluate_samples(samples: list[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
    detection = {kind: [0, 0, 0] for kind in ("CHAR", "TEXT")}
    speaker = [0, 0, 0]
    grounding = [0, 0, 0]
    grounding_span = [0, 0, 0]
    cluster_pair = [0, 0, 0]
    ami_values: list[float] = []
    character_errors = reference_characters = 0
    edit_sum = exact = text_count = 0
    panel_found = panel_total = invalid_commands = valid_commands = 0
    matched_characters = total_characters = 0

    for sample in samples:
        target, state = sample["target"], sample["state"]
        invalid_commands += len(sample.get("invalid_commands", []))
        valid_commands += len(sample.get("valid_commands", []))
        mapping, by_type = match_entities(state, target, iou_threshold)

        target_panels = {tuple(panel["bbox"]) for panel in target.get("panels", [])}
        panel_found += len(set(state.panels) & target_panels)
        panel_total += len(target["panels"])
        gt_entities = target_entities(target)
        for kind in detection:
            pred_count = sum(entity["type"] == kind for entity in state.entities.values())
            gt_count = sum(entity["type"] == kind for entity in gt_entities.values())
            tp = len(by_type[kind])
            detection[kind][0] += tp
            detection[kind][1] += pred_count - tp
            detection[kind][2] += gt_count - tp

        pred_entities = state.entities
        for pred_id, gt_id in by_type["TEXT"].items():
            pred = _normalize_text(pred_entities[pred_id].get("content", ""))
            gt = _normalize_text(gt_entities[gt_id].get("content", ""))
            distance = _edit_distance(list(pred), list(gt))
            character_errors += distance
            reference_characters += len(gt)
            edit_sum += 1.0 - distance / max(1, len(pred), len(gt))
            exact += pred == gt
            text_count += 1
        missing_ids = [
            entity_id for entity_id, item in gt_entities.items()
            if item["type"] == "TEXT" and entity_id not in by_type["TEXT"].values()
        ]
        missing_text = len(missing_ids)
        text_count += missing_text
        for entity_id in missing_ids:
            reference_length = len(_normalize_text(gt_entities[entity_id].get("content", "")))
            character_errors += reference_length
            reference_characters += reference_length

        char_by_box = {
            tuple(item["bbox"]): entity_id
            for entity_id, item in gt_entities.items() if item["type"] == "CHAR"
        }
        text_by_box = {
            tuple(item["bbox"]): entity_id
            for entity_id, item in gt_entities.items() if item["type"] == "TEXT"
        }
        gt_speaker = {
            (text_by_box[tuple(edge["text_bbox"])], char_by_box[tuple(edge["character_bbox"])])
            for edge in target.get("speaker_links", [])
            if tuple(edge["text_bbox"]) in text_by_box and tuple(edge["character_bbox"]) in char_by_box
        }
        pred_speaker = {
            (mapping.get(a, f"PRED:{a}"), mapping.get(b, f"PRED:{b}"))
            for a, b in state.speakers
        }
        speaker[0] += len(pred_speaker & gt_speaker)
        speaker[1] += len(pred_speaker - gt_speaker)
        speaker[2] += len(gt_speaker - pred_speaker)

        gt_char_ids = sorted(entity_id for entity_id, item in gt_entities.items() if item["type"] == "CHAR")
        cluster_labels: dict[str, int] = {}
        clusters: dict[str, int] = {}
        for entity_id in gt_char_ids:
            cluster = str(gt_entities[entity_id].get("cluster", entity_id))
            cluster_labels[entity_id] = clusters.setdefault(cluster, len(clusters))
        gt_labels = cluster_labels
        mapped_pred_links = [(mapping.get(a, ""), mapping.get(b, "")) for a, b in state.characters]
        pred_labels = _clusters(gt_char_ids, mapped_pred_links)
        matched_ids = sorted(by_type["CHAR"].values())
        matched_characters += len(matched_ids)
        total_characters += len(gt_char_ids)
        if len(matched_ids) > 1:
            ami_values.append(_ami(
                [gt_labels[item] for item in matched_ids], [pred_labels[item] for item in matched_ids]
            ))
        gt_pairs, pred_pairs = _cluster_pairs(gt_labels), _cluster_pairs(pred_labels)
        cluster_pair[0] += len(gt_pairs & pred_pairs)
        cluster_pair[1] += len(pred_pairs - gt_pairs)
        cluster_pair[2] += len(gt_pairs - pred_pairs)

        gt_ground = {
            (tuple(grounding["panel_bbox"]), int(mention["start"]), int(mention["end"]), char_by_box[tuple(box)])
            for grounding in target.get("groundings", [])
            for mention in grounding.get("mentions", [])
            for box in mention.get("character_bboxes", [])
            if tuple(box) in char_by_box
        }
        pred_ground = {
            (p, s, e, mapping.get(c, f"PRED:{c}")) for p, s, e, c in state.grounding
        }
        grounding[0] += len(pred_ground & gt_ground)
        grounding[1] += len(pred_ground - gt_ground)
        grounding[2] += len(gt_ground - pred_ground)
        gt_spans = {(p, s, e) for p, s, e, _ in gt_ground}
        pred_spans = {(p, s, e) for p, s, e, _ in state.grounding}
        grounding_span[0] += len(pred_spans & gt_spans)
        grounding_span[1] += len(pred_spans - gt_spans)
        grounding_span[2] += len(gt_spans - pred_spans)

    per_class = {kind.lower(): _prf(*values) for kind, values in detection.items()}
    det_total = [sum(detection[k][i] for k in detection) for i in range(3)]
    return {
        "num_samples": len(samples),
        "iou_threshold": iou_threshold,
        "invalid_commands": invalid_commands,
        "valid_commands": valid_commands,
        "command_valid_rate": _ratio(valid_commands, valid_commands + invalid_commands),
        "panel_add_recall": _ratio(panel_found, panel_total),
        "detection": {"overall": _prf(*det_total), "per_class": per_class},
        "ocr": {
            "num_text_instances": text_count,
            "1-EditDist": _ratio(edit_sum, text_count),
            "CER": _ratio(character_errors, reference_characters),
            "accuracy": _ratio(exact, text_count),
        },
        "speaker_association": _prf(*speaker),
        "character_identification": {
            "AMI": sum(ami_values) / len(ami_values) if ami_values else 0.0,
            "matched_character_coverage": _ratio(matched_characters, total_characters),
            "pairwise": _prf(*cluster_pair),
        },
        "caption_grounding": {"entity": _prf(*grounding), "span": _prf(*grounding_span)},
    }
