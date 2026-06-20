# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from pathlib import Path
from json import JSONDecodeError
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import ConcatDataset

from data.dataset_base import DataConfig, simple_custom_collate
from data.datasets_custom.validation_dataset import ValidationDataset, modality_map, sample_task_map


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _resolve_media_path(path: Any, data_root: str) -> Any:
    if not isinstance(path, str) or not path:
        return path
    if "://" in path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(data_root, path))


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = []
        for key, value in data.items():
            if isinstance(value, dict) and "interleave_array" in value:
                records.append({"index": key, "data": value})
            elif isinstance(value, dict):
                item = copy.deepcopy(value)
                item.setdefault("index", key)
                records.append(item)
            else:
                records.append({"index": key, "data": value})
    else:
        raise ValueError(f"Unsupported data file format: {path}")

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            records[i] = {"index": i, "data": record}
        else:
            record.setdefault("index", i)
    return records


def make_lance_dataset_config(raw: Dict[str, Any], model_args, training_args) -> SimpleNamespace:
    cfg = DataConfig(grouped_datasets={})

    cfg.task = raw.get("task", "t2v")
    if "_t" in cfg.task or cfg.task in {"x2t", "x2t_image", "x2t_video"}:
        cfg.target_modality = "text"
    elif "2i" in cfg.task or "image" in cfg.task:
        cfg.target_modality = "image"
    else:
        cfg.target_modality = "video"

    cfg.resolution = raw.get("resolution", "video_360p")
    cfg.text_template = _as_bool(raw.get("text_template"), False)
    cfg.system_prompt_type = raw.get("system_prompt_type", "SP0")
    cfg.max_duration = float(raw.get("max_duration", 6.0))
    cfg.enhance_prompt = _as_bool(raw.get("enhance_prompt"), False)
    cfg.num_frames = int(raw.get("num_frames", getattr(model_args, "max_num_frames", 25)))
    cfg.H = int(raw.get("height", raw.get("H", 480)))
    cfg.W = int(raw.get("width", raw.get("W", 480)))

    cfg.vit_patch_size = model_args.vit_patch_size
    cfg.vit_patch_size_temporal = model_args.vit_patch_size_temporal
    cfg.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    cfg.latent_patch_size = tuple(model_args.latent_patch_size)
    cfg.max_latent_size = model_args.max_latent_size
    cfg.max_num_frames = model_args.max_num_frames
    default_vae_downsample = (
        int(model_args.latent_patch_size[0]) * 4,
        int(model_args.latent_patch_size[1]) * 16,
        int(model_args.latent_patch_size[2]) * 16,
    )
    cfg.vae_downsample = tuple(raw.get("vae_downsample", default_vae_downsample))

    cfg.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    cfg.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    cfg.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    return cfg


class LanceSupervisedDataset(ValidationDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        data_args,
        model_args,
        training_args,
        new_token_ids,
        dataset_config,
        data_root: str = "",
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self._records_override = _load_records(data_path)
        self.data_root = data_root or str(Path(data_path).resolve().parent)
        super().__init__(
            jsonl_path=data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            new_token_ids=new_token_ids,
            dataset_config=dataset_config,
            local_rank=local_rank,
            world_size=world_size,
        )
        full_data = self._records_override
        self.data = full_data[local_rank::world_size] if world_size > 1 else full_data

    def _read_jsonl(self):
        return self._records_override

    def _normalize_interleave(self, sample: Dict[str, Any]):
        payload = sample.get("data", sample)
        if isinstance(payload, dict) and "interleave_array" in payload:
            interleave = list(payload["interleave_array"])
            dtypes = list(payload["element_dtype_array"])
            targets = list(payload["istarget_in_interleave"])
            return interleave, dtypes, targets

        task = self.data_config.task
        prompt = sample.get("prompt") or sample.get("caption") or sample.get("text") or sample.get("data")
        if isinstance(prompt, list):
            prompt = " ".join(map(str, prompt))

        target = (
            sample.get("target_video")
            or sample.get("video")
            or sample.get("target_image")
            or sample.get("image")
            or sample.get("target")
        )
        source = sample.get("source_video") or sample.get("source_image") or sample.get("input_video") or sample.get("input_image")
        if prompt is None or target is None:
            raise ValueError(
                "Generation finetune records need either Lance interleave fields or prompt + target media "
                "(video/target_video/image/target_image)."
            )

        target_dtype = "image" if ("2i" in task or "image" in task) else "video"
        if task in {"i2v", "idip", "image_edit", "video_edit"} or "edit" in task:
            if source is None:
                raise ValueError(f"Task {task} needs source_video/source_image/input_video/input_image.")
            source_dtype = "image" if str(source).lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")) else "video"
            return [source, prompt, target], [source_dtype, "text", target_dtype], [0, 0, 1]
        return [prompt, target], ["text", target_dtype], [0, 1]

    def supervised_generation_sample(self, idx: int) -> Dict[str, Any]:
        self.sample = self.set_sequence_status()
        sample = self.data[idx]
        interleave, dtypes, targets = self._normalize_interleave(sample)

        curr, curr_rope_id, sample_lens = 0, 0, 0
        curr_video_grid_thw, video_sizes, sample_modality = [], [], []
        caption_all = []
        target_count = 0

        for element, element_dtype, is_target in zip(interleave, dtypes, targets):
            if element_dtype == "text":
                text = element[-1] if isinstance(element, list) else str(element)
                caption_all.append(text)
                self.sample, curr, curr_rope_id, curr_split_len = self.process_text(
                    text,
                    curr=curr,
                    curr_rope_id=curr_rope_id,
                    curr_split_len=0,
                    item_loss=int(is_target),
                )
                sample_lens += curr_split_len
                sample_modality.extend([modality_map["text"]] * curr_split_len)
                continue

            if element_dtype not in {"image", "video"}:
                raise ValueError(f"Unsupported element dtype: {element_dtype}")

            media_path = _resolve_media_path(element, self.data_root)
            if int(is_target) == 0:
                vit_tensor = self.get_video_tensor_online(media_path, vision_stream="vit_video", element_dtype=element_dtype)
                self.sample, curr, curr_rope_id, curr_split_len, curr_video_grid_thw, _ = self.process_vit_video(
                    vit_tensor,
                    curr=curr,
                    curr_rope_id=curr_rope_id,
                    curr_split_len=0,
                    curr_video_grid_thw=curr_video_grid_thw,
                    item_loss=0,
                )
                sample_lens += curr_split_len
                sample_modality.extend([modality_map["ref_vit"]] * curr_split_len)

                vae_tensor = self.get_video_tensor_online(media_path, vision_stream="vae_video", element_dtype=element_dtype)
                self.sample, curr, curr_rope_id, curr_split_len, curr_video_grid_thw, video_sizes, _ = self.process_vae_video(
                    vae_tensor,
                    curr=curr,
                    curr_rope_id=curr_rope_id,
                    curr_split_len=0,
                    curr_video_grid_thw=curr_video_grid_thw,
                    video_sizes=video_sizes,
                    item_loss=0,
                )
                sample_lens += curr_split_len
                sample_modality.extend([modality_map["ref_source"]] * curr_split_len)
            else:
                vae_tensor = self.get_video_tensor_online(media_path, vision_stream="vae_video", element_dtype=element_dtype)
                self.sample, curr, curr_rope_id, curr_split_len, curr_video_grid_thw, video_sizes, _ = self.process_vae_video(
                    vae_tensor,
                    curr=curr,
                    curr_rope_id=curr_rope_id,
                    curr_split_len=0,
                    curr_video_grid_thw=curr_video_grid_thw,
                    video_sizes=video_sizes,
                    item_loss=1,
                )
                sample_lens += curr_split_len
                sample_modality.extend([modality_map["noise"]] * curr_split_len)
                target_count += 1

        if target_count == 0:
            raise ValueError("Generation finetune sample does not contain a target image/video.")

        task_key = "edit" if "edit" in self.data_config.task else "idip" if "idip" in self.data_config.task else "t2v"
        self.sample["sample_task"] = torch.ones(sample_lens) * sample_task_map[task_key]
        self.sample["sample_modality"] = sample_modality

        out = self._finalize_sample(
            sample_lens=sample_lens,
            curr_video_grid_thw=curr_video_grid_thw,
            sample_type="gen",
            sample=sample,
            additional_fields={"caption": " ".join(caption_all), "data_indexes": {"index": sample.get("index", idx)}},
            video_sizes=video_sizes,
        )
        out["sample_N_target"] = torch.tensor([[target_count]])
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.data_config.task in {"x2t", "x2t_image", "x2t_video"}:
            sample = self.data[idx]
            payload = sample.get("data", sample)
            for media_index, dtype in enumerate(payload.get("element_dtype_array", [])):
                if dtype in {"image", "video"}:
                    payload["interleave_array"][media_index] = _resolve_media_path(
                        payload["interleave_array"][media_index],
                        self.data_root,
                    )
            return super().__getitem__(idx)
        return self.supervised_generation_sample(idx)


def build_finetune_dataset(config_file, tokenizer, data_args, model_args, training_args, new_token_ids, local_rank, world_size):
    with open(config_file, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    datasets_meta = meta.get("datasets", meta if isinstance(meta, list) else None)
    if not datasets_meta:
        raise ValueError("Finetune config must contain a top-level `datasets` list.")

    datasets = []
    for raw in datasets_meta:
        raw = dict(raw)
        data_path = raw.get("path") or raw.get("json_path") or raw.get("jsonl_path")
        if not data_path:
            raise ValueError("Every dataset entry needs `path`.")
        data_root = raw.get("data_root") or str(Path(data_path).resolve().parent)
        dataset_config = make_lance_dataset_config(raw, model_args, training_args)
        datasets.append(
            LanceSupervisedDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                model_args=model_args,
                training_args=training_args,
                new_token_ids=new_token_ids,
                dataset_config=dataset_config,
                data_root=data_root,
                local_rank=local_rank,
                world_size=world_size,
            )
        )

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


__all__ = ["build_finetune_dataset", "simple_custom_collate"]
