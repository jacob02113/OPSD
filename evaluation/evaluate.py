#!/usr/bin/env python3
"""Evaluate closed-tag manga DSL with Qwen or InternVL ref/box notation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dsl import DSL_VERSION, parse_response, state_to_dict
from metrics import evaluate_samples


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def response_text(record: dict[str, Any]) -> str:
    for key in ("model_response", "response", "result"):
        if key in record:
            return str(record[key])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default="dataset/popmanga_unseen.parquet")
    parser.add_argument("--predictions", type=Path, default="results/qwen3-vl_4b_dsl_sft_v3.1_e1_popmanga_unseen.jsonl")
    parser.add_argument("--metrics", type=Path, default="results/qwen3-vl_4b_dsl_sft_v3.1_e1_popmanga_unseen.json")
    parser.add_argument("--processed-output", type=Path, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data, engine="pyarrow")
    targets = {int(row.sample_id): json.loads(str(row.target_json)) for row in frame.itertuples()}
    wrong_versions = {
        sample_id: target.get("dsl_version")
        for sample_id, target in targets.items()
        if target.get("dsl_version") != DSL_VERSION
    }
    if wrong_versions:
        raise ValueError(f"expected {DSL_VERSION} targets, found: {wrong_versions}")
    predictions = {int(item["sample_id"]): item for item in read_jsonl(args.predictions)}
    samples = []
    processed = []
    missing = 0
    for sample_id, target in targets.items():
        record = predictions.get(sample_id)
        if record is None:
            missing += 1
            continue
        response = response_text(record)
        state, valid, invalid = parse_response(response, target)
        samples.append({
            "target": target,
            "state": state,
            "valid_commands": valid,
            "invalid_commands": invalid,
        })
        processed.append({
            "sample_id": sample_id,
            "response": response,
            "graph": state_to_dict(state),
            "valid_commands": valid,
            "invalid_commands": invalid,
        })

    metrics = evaluate_samples(samples, iou_threshold=args.iou_threshold)
    metrics["missing_predictions"] = missing
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.processed_output:
        args.processed_output.parent.mkdir(parents=True, exist_ok=True)
        with args.processed_output.open("w", encoding="utf-8") as handle:
            for item in processed:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
