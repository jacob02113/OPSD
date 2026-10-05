"""Canonical prompt adapter shared with the data preprocessor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from prepare_data import PROMPT  # noqa: E402


def build_prompt(panel_records: Iterable[str], allow_repair: bool = False) -> str:
    """Build exactly the prompt used to create the training Parquet."""

    if allow_repair:
        raise ValueError("REPAIR is not part of bbox_ref_v1")
    return PROMPT.format(panel_records="\n".join(panel_records))
