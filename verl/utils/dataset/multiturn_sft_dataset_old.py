# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from tqdm import tqdm

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_tokenizer
from verl.utils.chat_template import apply_chat_template, extract_system_prompt_and_generation
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def once(func):
    """Decorator to ensure a function runs only once. Subsequent calls do nothing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, "called"):
            wrapper.called = True
            return func(*args, **kwargs)

    return wrapper


@once
def print_assembled_message(tokenizer, message_list, input_ids, loss_mask, attn_mask, tools):
    """
    Print the message after applying the chat template
    """

    tokenized = tokenizer.apply_chat_template(message_list, add_generation_prompt=False, tokenize=False, tools=tools)
    sep = "\n\n"
    str = f"tokenized entire message:\n{tokenized}"
    str += sep
    decoded_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    str += f"tokenized seperately    :\n{tokenizer.decode(decoded_ids)}"

    logger.debug(str)


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
        max_samples (int, optional): Limit the number of samples. Defaults to -1 (use all).
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        self.messages_key = config.get("messages_key", "messages")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get(
            "image_patch_size", processor.image_processor.patch_size if processor else None
        )
        self.tools_key = config.get("tools_key", "tools")
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = config.get("enable_thinking_default", None)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)
        self.filter_overlong_sequences = config.get("filter_overlong_sequences", True)
        self.filter_overlong_sequences_workers = config.get("filter_overlong_sequences_workers", 0)
        self.filter_overlong_sequences_cache = config.get("filter_overlong_sequences_cache", True)
        self.filter_overlong_sequences_cache_dir = config.get("filter_overlong_sequences_cache_dir", None)
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        self._download()
        self._read_files_and_process()
        self._filter_overlong_sequences()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # default loader loads some list as np.ndarray, which fails the tokenizer
            dataframe = pd.read_parquet(parquet_file, dtype_backend="pyarrow")
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        self._refresh_metadata()

    def _refresh_metadata(self):
        """Refresh row-aligned metadata after selecting a dataframe subset."""
        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None

        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

    def __len__(self):
        return len(self.messages)

    def _get_enable_thinking(self, item: int) -> Optional[bool]:
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if isinstance(enable_thinking, str) and enable_thinking.strip().lower() in {"", "none", "null"}:
            return None
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)
        return enable_thinking

    def get_sequence_length(self, item: int) -> int:
        """Return the exact assembled token length without applying truncation."""
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self._get_enable_thinking(item)
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        inputs = apply_chat_template(
            processor,
            messages=messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )
        return int(inputs["input_ids"].shape[-1])

    def select_indices(self, indices: list[int]) -> None:
        """Keep only the selected rows while preserving their original order."""
        self.dataframe = self.dataframe.iloc[indices].reset_index(drop=True)
        self._refresh_metadata()

    def _filter_cache_signature(self) -> str:
        sources = []
        for parquet_file in self.parquet_files:
            path = Path(parquet_file).resolve()
            stat = path.stat()
            sources.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

        processor_name = getattr(self.processor, "name_or_path", None)
        tokenizer_name = getattr(self.tokenizer, "name_or_path", None)
        payload = {
            "version": 1,
            "sources": sources,
            "max_length": self.max_length,
            "processor_class": type(self.processor).__name__ if self.processor is not None else None,
            "processor_name": processor_name,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name": tokenizer_name,
            "messages_key": self.messages_key,
            "image_key": self.image_key,
            "video_key": self.video_key,
            "enable_thinking_default": repr(self.enable_thinking_default),
            "apply_chat_template_kwargs": repr(self.apply_chat_template_kwargs),
        }
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _filter_cache_path(self) -> Optional[Path]:
        if not self.filter_overlong_sequences_cache:
            return None

        first_parquet = Path(self.parquet_files[0]).resolve()
        if self.filter_overlong_sequences_cache_dir:
            cache_dir = Path(self.filter_overlong_sequences_cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{first_parquet.stem}.maxlen-{self.max_length}.filter-cache.json"
            return cache_dir / filename
        return Path(f"{first_parquet}.maxlen-{self.max_length}.filter-cache.json")

    def _load_filter_cache(self) -> Optional[dict]:
        cache_path = self._filter_cache_path()
        if cache_path is None or not cache_path.exists():
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
            if cached.get("signature") != self._filter_cache_signature():
                return None
            result = cached.get("result")
            if not isinstance(result, dict) or result.get("total") != len(self):
                return None
            return result
        except Exception as exc:
            logger.warning("Ignoring invalid overlong-sequence cache %s: %s", cache_path, exc)
            return None

    def _save_filter_cache(self, result: dict) -> None:
        cache_path = self._filter_cache_path()
        if cache_path is None:
            return

        temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8") as cache_file:
                json.dump(
                    {"signature": self._filter_cache_signature(), "result": result},
                    cache_file,
                    ensure_ascii=True,
                )
            os.replace(temp_path, cache_path)
        except Exception as exc:
            logger.warning("Failed to save overlong-sequence cache %s: %s", cache_path, exc)
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _filter_overlong_sequences(self) -> None:
        """Filter in parallel across ranks and cache the selected indices."""
        if not self.filter_overlong_sequences:
            return

        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1

        cached_payload = [self._load_filter_cache() if rank == 0 else None]
        if distributed:
            torch.distributed.broadcast_object_list(cached_payload, src=0)
        if cached_payload[0] is not None:
            result = cached_payload[0]
            self.select_indices(result["indices"])
            if rank == 0:
                removed = result["total"] - len(result["indices"])
                print(
                    f"Loaded overlong-sequence filter cache: kept {len(result['indices'])}/{result['total']} "
                    f"samples, removed {removed}"
                )
            return

        local_indices = list(range(rank, len(self), world_size))
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1
        configured_workers = int(self.filter_overlong_sequences_workers or 0)
        workers = configured_workers or max(1, min(4, available_cpus // world_size))

        local_payload = None
        try:
            keep_indices = []
            removed_examples = []
            observed_max_length = 0

            def scan(index: int) -> tuple[int, int]:
                return index, self.get_sequence_length(index)

            with torch.inference_mode(), ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(scan, local_indices)
                for index, sequence_length in tqdm(
                    results,
                    total=len(local_indices),
                    desc=f"Filtering sequence shard {rank + 1}/{world_size} with {workers} workers",
                    disable=rank != 0,
                ):
                    observed_max_length = max(observed_max_length, sequence_length)
                    if sequence_length <= self.max_length:
                        keep_indices.append(index)
                    elif len(removed_examples) < 10:
                        removed_examples.append((index, sequence_length))

            local_payload = {
                "error": None,
                "indices": keep_indices,
                "observed_max_length": observed_max_length,
                "removed_examples": removed_examples,
            }
        except Exception as exc:
            local_payload = {"error": f"rank {rank}: {type(exc).__name__}: {exc}"}

        if distributed:
            gathered_payloads = [None] * world_size
            torch.distributed.all_gather_object(gathered_payloads, local_payload)
        else:
            gathered_payloads = [local_payload]

        errors = [payload["error"] for payload in gathered_payloads if payload["error"] is not None]
        if errors:
            raise RuntimeError(f"Failed to filter overlong sequences: {'; '.join(errors)}")

        keep_indices = sorted(index for payload in gathered_payloads for index in payload["indices"])
        removed_examples = sorted(
            example for payload in gathered_payloads for example in payload["removed_examples"]
        )[:10]
        observed_max_length = max(payload["observed_max_length"] for payload in gathered_payloads)

        if not keep_indices:
            raise ValueError(
                f"All samples exceed max_length={self.max_length}; "
                "reduce image resolution or shorten the conversations."
            )

        result = {
            "error": None,
            "indices": keep_indices,
            "total": len(self),
            "observed_max_length": observed_max_length,
            "removed_examples": removed_examples,
        }
        if rank == 0:
            self._save_filter_cache(result)

        self.select_indices(result["indices"])
        if rank == 0:
            removed = result["total"] - len(result["indices"])
            print(
                f"Filtered dataset: kept {len(result['indices'])}/{result['total']} samples, "
                f"removed {removed}, longest observed sequence={result['observed_max_length']}, "
                f"first removed (index, length)={result['removed_examples']}"
            )

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process a single message and return its tokenized representation.

        Args:
            index: turn index in the conversation
            message: A single message dictionary
            images: List of images to be used
            videos: List of videos to be used
            tools: List of tools to be used
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask, dict[str, torch.Tensor])
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        inputs = apply_chat_template(
            processor,
            messages=[message],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # remove system prompt if exists
        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            # mask out generation prompt if assistant message
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs

    def _build_messages(self, example: dict):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", "image": image}
        - <video>: {"type": "video", "video": video}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = convert_nested_value_to_list_recursive(example[self.messages_key])
        images = example[self.image_key] if self.image_key in example else []
        videos = example[self.video_key] if self.video_key in example else []

        image_offset, video_offset = 0, 0
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                continue

            if self.image_key not in example and self.video_key not in example:
                if self.processor is not None:
                    message["content"] = [{"type": "text", "text": content}]
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    image = process_image(images[image_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    video = process_video(videos[video_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "video", "video": video})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self._get_enable_thinking(item)

        # 1. tokenize each message
        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}
        for i, message in enumerate(messages):
            _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(
                index=i,
                message=message,
                full_message=messages,
                tools=tools if i == 0 else None,
                enable_thinking=enable_thinking,
            )
            input_ids.append(_input_ids)
            loss_mask.append(_loss_mask)
            attention_mask.append(_attention_mask)
            for k, v in _inputs.items():
                multi_modal_inputs.setdefault(k, []).append(v)

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        )

        print_assembled_message(self.tokenizer, messages, input_ids, loss_mask, attention_mask, tools)
        self.sanity_check(input_ids, messages, tools, enable_thinking)

        # Since the tokenizer may return user-customized results, we need to filter out inconsistent tensor shapes
        keys_to_remove = []
        for k, v in multi_modal_inputs.items():
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):
                # Check if all tensors in the list have the same shape
                first_shape = v[0].shape[1:]
                if not all(tensor.shape[1:] == first_shape for tensor in v):
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del multi_modal_inputs[k]

        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = torch.concat(v, dim=0)

        # 2. handle position_ids for Qwen-VL series models
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            image_grid_thw = multi_modal_inputs.get("image_grid_thw", None)
            video_grid_thw = multi_modal_inputs.get("video_grid_thw", None)
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts", None)

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )  # (3, seq_len)
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)

        # 3. handle padding
        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if sequence_length > self.max_length and self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            # truncate input_ids if it is longer than max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]

            # return nested tensor with out padding
            res = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], tools: list[dict], enable_thinking: bool):
        """Check concatenated input_ids of apply_chat_template to each turn equals
        apply_chat_template to whole messages.
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "
            "your tokenizer chat template settings.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "
            "input_ids as the final input_ids. "
        )

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                logger.warning_once(error_message)
            else:
                raise AssertionError(error_message)
