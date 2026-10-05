"""Manga panel reading-order sorting adapted from MagiV3."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Iterable

import networkx as nx

Rect = list[float]


def _as_rects(rects: Iterable[Iterable[float]]) -> list[Rect]:
    return [[float(value) for value in rect] for rect in rects]


def _erode_rectangle(bbox: Rect, erosion_factor: float) -> Rect:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    cx, cy = x1 + width / 2.0, y1 + height / 2.0
    if width < height:
        aspect_ratio = width / height if height else 1.0
        width_factor, height_factor = erosion_factor * aspect_ratio, erosion_factor
    else:
        aspect_ratio = height / width if width else 1.0
        width_factor, height_factor = erosion_factor, erosion_factor * aspect_ratio
    width -= width * width_factor
    height -= height * height_factor
    return [
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    ]


def _strictly_above(a: Rect, b: Rect) -> bool:
    return a[3] < b[1]


def _strictly_below(a: Rect, b: Rect) -> bool:
    return b[3] < a[1]


def _strictly_left_of(a: Rect, b: Rect) -> bool:
    return a[2] < b[0]


def _strictly_right_of(a: Rect, b: Rect) -> bool:
    return b[2] < a[0]


def _intersects(a: Rect, b: Rect) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _distance(a: Rect, b: Rect) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _merge_overlapping_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda value: value[0])
    merged: list[tuple[float, float]] = []
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            merged.append((start, end))
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    merged.append((start, end))
    return merged


def _fallback_precedes(a: int, b: int, rects: list[Rect]) -> bool:
    """Deterministic Japanese reading order: rows top-down, then right-left."""

    ra, rb = rects[a], rects[b]
    ca = ((ra[1] + ra[3]) / 2.0, -(ra[0] + ra[2]) / 2.0, a)
    cb = ((rb[1] + rb[3]) / 2.0, -(rb[0] + rb[2]) / 2.0, b)
    return ca < cb


def _use_cuts_to_determine_edge(a: int, b: int, source_rects: list[Rect]) -> bool:
    rects = deepcopy(source_rects)
    for _ in range(200):
        xmin = min(rects[a][0], rects[b][0])
        ymin = min(rects[a][1], rects[b][1])
        xmax = max(rects[a][2], rects[b][2])
        ymax = max(rects[a][3], rects[b][3])
        window = [xmin, ymin, xmax, ymax]
        indices = [idx for idx, rect in enumerate(rects) if _intersects(rect, window)]
        subset = [rects[idx] for idx in indices]

        y_ranges = _merge_overlapping_ranges([(rect[1], rect[3]) for rect in subset])
        y_split: dict[int, int] = {}
        for split_idx, (start, end) in enumerate(y_ranges):
            for idx in indices:
                rect = rects[idx]
                if start <= rect[1] <= rect[3] <= end:
                    y_split[idx] = split_idx
        if a in y_split and b in y_split and y_split[a] != y_split[b]:
            return y_split[a] < y_split[b]

        x_ranges = _merge_overlapping_ranges([(rect[0], rect[2]) for rect in subset])
        x_split: dict[int, int] = {}
        for split_idx, (start, end) in enumerate(reversed(x_ranges)):
            for idx in indices:
                rect = rects[idx]
                if start <= rect[0] <= rect[2] <= end:
                    x_split[idx] = split_idx
        if a in x_split and b in x_split and x_split[a] != x_split[b]:
            return x_split[a] < x_split[b]

        rects = [_erode_rectangle(rect, 0.05) for rect in rects]
    return _fallback_precedes(a, b, source_rects)


def _precedes(a: int, b: int, rects: list[Rect]) -> bool:
    rect_a, rect_b = rects[a], rects[b]
    center_a = ((rect_a[0] + rect_a[2]) / 2.0, (rect_a[1] + rect_a[3]) / 2.0)
    center_b = ((rect_b[0] + rect_b[2]) / 2.0, (rect_b[1] + rect_b[3]) / 2.0)
    if math.isclose(center_a[0], center_b[0]) and math.isclose(center_a[1], center_b[1]):
        area_a = max(0.0, rect_a[2] - rect_a[0]) * max(0.0, rect_a[3] - rect_a[1])
        area_b = max(0.0, rect_b[2] - rect_b[0]) * max(0.0, rect_b[3] - rect_b[1])
        return (area_a, -a) > (area_b, -b)

    copy_a, copy_b = list(rect_a), list(rect_b)
    for _ in range(200):
        if _strictly_above(copy_a, copy_b) and not _strictly_left_of(copy_a, copy_b):
            return True
        if _strictly_above(copy_b, copy_a) and not _strictly_left_of(copy_b, copy_a):
            return False
        if _strictly_right_of(copy_a, copy_b) and not _strictly_below(copy_a, copy_b):
            return True
        if _strictly_right_of(copy_b, copy_a) and not _strictly_below(copy_b, copy_a):
            return False
        if _strictly_below(copy_a, copy_b) and _strictly_right_of(copy_a, copy_b):
            return _use_cuts_to_determine_edge(a, b, rects)
        if _strictly_below(copy_b, copy_a) and _strictly_right_of(copy_b, copy_a):
            return _use_cuts_to_determine_edge(a, b, rects)
        copy_a = _erode_rectangle(copy_a, 0.05)
        copy_b = _erode_rectangle(copy_b, 0.05)
    return _fallback_precedes(a, b, rects)


def sort_panels(rects: Iterable[Iterable[float]]) -> list[int]:
    """Return indices in Japanese manga reading order.

    This follows MagiV3's pairwise spatial precedence rules and graph-based
    ordering while using a local, dependency-free cycle breaker.
    """

    original = _as_rects(rects)
    if len(original) < 2:
        return list(range(len(original)))
    eroded = [_erode_rectangle(rect, 0.05) for rect in original]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(eroded)))
    for i in range(len(eroded)):
        for j in range(len(eroded)):
            if i == j:
                continue
            if _precedes(i, j, eroded):
                graph.add_edge(i, j, weight=_distance(eroded[i], eroded[j]))
            else:
                graph.add_edge(j, i, weight=_distance(eroded[i], eroded[j]))

    while True:
        cycles = [cycle for cycle in nx.simple_cycles(graph) if len(cycle) > 1]
        if not cycles:
            break
        cycle = cycles[0]
        edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        max_edge = max(edges, key=lambda edge: graph.edges[edge]["weight"])
        graph.remove_edge(*max_edge)

    return list(nx.topological_sort(graph))
