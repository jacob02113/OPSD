"""Evaluation-data adapter for the canonical bbox-referenced preprocessor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from prepare_data import PageBuildResult, area, build_page as _build_page, overlap  # noqa: E402


def build_page(
    annotation: dict[str, Any],
    sample_id: int,
    image_root: Path,
    allow_repair: bool = False,
) -> PageBuildResult:
    """Build evaluation input with the same code path as training data."""

    if allow_repair:
        raise ValueError("REPAIR is not part of bbox_ref_v1")
    return _build_page(annotation, sample_id, image_root, allow_missing_image=True)


# Compatibility for legacy repair_engine imports. New evaluation code does not use these.
box_area = area
intersection_area = overlap


def format_box(box: Iterable[int]) -> str:
    x0, y0, x1, y1 = map(int, box)
    return f"[{x0},{y0},{x1},{y1}]"
