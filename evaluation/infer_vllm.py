#!/usr/bin/env python3
"""Run closed-tag manga DSL inference with the same vLLM path used by training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from dsl import ACTION_END_TAGS, DSL_VERSION, clean_response


def python_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return python_value(value.tolist())
    if isinstance(value, dict):
        return {key: python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [python_value(item) for item in value]
    return value


def batches(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def output_path_for_data(output: Path, data_path: Path, *, multiple: bool) -> Path:
    """Return one collision-resistant output path per input Parquet file."""

    output = Path(output)
    data_path = Path(data_path)
    if not multiple:
        return output
    suffix = output.suffix or ".jsonl"
    return output.with_name(f"{output.stem}_{data_path.stem}{suffix}")


def build_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand Parquet images, preserving either Qwen or InternVL DSL notation."""

    messages = copy.deepcopy(python_value(row["prompt"]))
    prompt_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )
    missing_tags = [tag for tag in ACTION_END_TAGS if tag not in prompt_text]
    if missing_tags:
        raise ValueError(f"Prompt does not use the closed-tag DSL; missing {missing_tags}")

    target = json.loads(str(row["target_json"]))
    if target.get("dsl_version") != DSL_VERSION:
        raise ValueError(
            f"Expected {DSL_VERSION} target, found {target.get('dsl_version')!r}"
        )
    images = python_value(row.get("images")) or []
    image_offset = 0

    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue

        parts: list[dict[str, Any]] = []
        for segment in filter(None, re.split("(<image>)", content)):
            if segment != "<image>":
                parts.append({"type": "text", "text": segment})
                continue

            if image_offset >= len(images):
                raise ValueError("Prompt contains more <image> placeholders than the images column")
            image = images[image_offset]
            image_part = (
                {"type": "image", **image}
                if isinstance(image, dict)
                else {"type": "image", "image": image}
            )
            parts.append(image_part)
            image_offset += 1
        message["content"] = parts

    if image_offset != len(images):
        raise ValueError(
            f"Prompt used {image_offset} image(s), but the images column contains {len(images)}"
        )
    return messages


def prepare_request(row: dict[str, Any], processor: Any) -> tuple[dict[str, Any], int]:
    """Build a Qwen multimodal vLLM request and its expanded prompt length."""

    messages = build_messages(row)
    from qwen_vl_utils import process_vision_info

    images, videos = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_metadata=True,
    )
    raw_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(
        text=[raw_prompt],
        images=images,
        videos=videos,
        return_tensors="pt",
    )
    prompt_ids = model_inputs["input_ids"][0].tolist()

    request: dict[str, Any] = {"prompt_token_ids": prompt_ids}
    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos
    if multi_modal_data:
        request["multi_modal_data"] = multi_modal_data
    return request, len(prompt_ids)


def sampling_params(args: argparse.Namespace, prompt_length: int):
    from vllm import SamplingParams

    max_tokens = min(args.max_new_tokens, args.max_model_len - prompt_length)
    if max_tokens < 1:
        raise ValueError(
            f"Prompt length {prompt_length} leaves no generation space under "
            f"--max-model-len={args.max_model_len}"
        )
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        ignore_eos=args.ignore_eos,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=["dataset/manga109_test.parquet"],
        help="One or more input Parquet files.",
    )
    parser.add_argument("--model", default="models/Qwen3-VL-4B-DSL-SFT-v3.1-e1")
    parser.add_argument("--output", type=Path, default="results/qwen3-vl_4b_dsl_sft_v3.1_e1.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=12289)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Maximum concurrent vLLM sequences; defaults to --batch-size and must fit "
            "the available Mamba cache blocks."
        ),
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    # argparse applies ``type=Path`` to command-line values, but not to
    # string defaults. Normalize both paths so default and explicit arguments
    # follow the same code path.
    args.data = [Path(path) for path in args.data]
    args.output = Path(args.output)
    if args.max_num_seqs is None:
        args.max_num_seqs = args.batch_size

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be at least 1")
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be at least 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_new_tokens < 1 or args.max_model_len < 2:
        parser.error("Token limits must be positive")

    multiple = len(args.data) > 1
    output_paths = [
        output_path_for_data(args.output, data_path, multiple=multiple)
        for data_path in args.data
    ]
    if len(set(output_paths)) != len(output_paths):
        parser.error(
            "Multiple input Parquet files would produce the same output filename; "
            "rename files with duplicate stems before running inference."
        )

    # vLLM owns worker creation; spawn avoids inheriting an initialized CUDA context.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoProcessor
    from vllm import LLM

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        generation_config="auto",
    )

    for data_index, (data_path, output_path) in enumerate(
        zip(args.data, output_paths, strict=True), start=1
    ):
        rows = [
            python_value(row)
            for row in pd.read_parquet(data_path, engine="pyarrow").to_dict("records")
        ]
        if args.num_samples >= 0:
            rows = random.Random(args.seed).sample(rows, min(args.num_samples, len(rows)))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[{data_index}/{len(args.data)}] {data_path} -> {output_path} "
            f"({len(rows)} sample(s))",
            flush=True,
        )
        with output_path.open("w", encoding="utf-8") as handle:
            for part in batches(rows, args.batch_size):
                prepared = [prepare_request(row, processor) for row in part]
                requests = [request for request, _ in prepared]
                params = [
                    sampling_params(args, prompt_length)
                    for _, prompt_length in prepared
                ]
                outputs = engine.generate(requests, sampling_params=params, use_tqdm=True)

                for row, output in zip(part, outputs, strict=True):
                    # Training consumes vLLM token IDs directly. Decode them here with
                    # the same tokenizer policy as infer.py so DSL special tokens survive.
                    text = processor.tokenizer.decode(
                        output.outputs[0].token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    text = clean_response(text)
                    handle.write(
                        json.dumps(
                            {
                                "sample_id": int(row["sample_id"]),
                                "image_rel_path": str(row["image_rel_path"]),
                                "model_response": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                handle.flush()


if __name__ == "__main__":
    main()
