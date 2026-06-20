# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import HfArgumentParser, set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_with_min_lr_schedule_with_warmup

from common.model.hacks import hack_qwen2_5_vl_config
from common.val.utils import make_padded_latent
from data.data_utils import add_special_tokens
from data.finetune_dataset import build_finetune_dataset, simple_custom_collate
from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
from train.fsdp_utils import (
    FSDPCheckpoint,
    FSDPConfig,
    fsdp_ema_setup,
    fsdp_ema_update,
    fsdp_wrapper,
    grad_checkpoint_check_fn,
)
from train.train_utils import create_logger, get_latest_ckpt

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


@dataclass
class ModelArguments:
    model_path: str = field(default="", metadata={"help": "Pretrained Lance checkpoint directory."})
    llm_path: str = field(default="", metadata={"help": "Tokenizer path; defaults to model_path."})
    vit_path: str = field(default="", metadata={"help": "Qwen2.5-VL ViT directory containing vit.safetensors."})
    llm_qk_norm: bool = True
    llm_qk_norm_und: bool = True
    llm_qk_norm_gen: bool = True
    tie_word_embeddings: bool = False
    layer_module: str = "Qwen2MoTDecoderLayer"
    max_num_frames: int = 25
    max_latent_size: int = 64
    latent_patch_size: list[int] = field(default_factory=lambda: [1, 1, 1])
    vit_patch_size: int = 14
    vit_patch_size_temporal: int = 2
    vit_max_num_patch_per_side: int = 70
    connector_act: str = "gelu_pytorch_tanh"
    interpolate_pos: bool = False
    vit_type: str = "qwen_2_5_vl_original"
    text_cond_dropout_prob: float = 0.1
    vae_cond_dropout_prob: float = 0.3
    vit_cond_dropout_prob: float = 0.3


@dataclass
class DataArguments:
    dataset_config_file: str = field(default="data/configs/lance_finetune_example.yaml")
    num_workers: int = 2
    prefetch_factor: int = 2


@dataclass
class TrainingArguments:
    visual_gen: bool = True
    visual_und: bool = True
    apply_qwen_2_5_vl_pos_emb: bool = False

    results_dir: str = "results/lance_finetune"
    checkpoint_dir: str = "results/lance_finetune/checkpoints"
    wandb_project: str = "lance"
    wandb_name: str = "finetune"
    wandb_runid: str = "0"
    wandb_resume: str = "allow"
    wandb_offline: bool = True

    global_seed: int = 2025
    auto_resume: bool = False
    resume_from: Optional[str] = None
    resume_model_only: bool = False
    finetune_from_ema: bool = True
    init_from_model_path: bool = True

    log_every: int = 10
    save_every: int = 1000
    total_steps: int = 10000
    gradient_accumulation_steps: int = 1

    lr_scheduler: str = "constant"
    warmup_steps: int = 100
    lr: float = 2e-5
    min_lr: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-15
    ema: float = 0.9999
    max_grad_norm: float = 1.0
    timestep_shift: float = 1.0
    mse_weight: float = 1.0
    ce_weight: float = 1.0
    ce_loss_reweighting: bool = False

    num_replicate: int = 1
    num_shard: int = 1
    sharding_strategy: str = "FULL_SHARD"
    backward_prefetch: str = "BACKWARD_PRE"
    cpu_offload: bool = False

    freeze_llm: bool = False
    freeze_vit: bool = False
    freeze_vae: bool = True
    freeze_und: bool = False
    copy_init_moe: bool = False


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def init_from_model_path_if_needed(model: Lance, model_args: ModelArguments, logger):
    model_file = os.path.join(model_args.model_path, "model.safetensors")
    ema_file = os.path.join(model_args.model_path, "ema.safetensors")
    ckpt_file = model_file if os.path.exists(model_file) else ema_file
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"No model.safetensors or ema.safetensors found in {model_args.model_path}")

    state_dict = load_file(ckpt_file, device="cpu")
    state_dict.pop("latent_pos_embed.pos_embed", None)
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(msg)
    del state_dict


def build_model_and_tokenizer(model_args, training_args, device, logger):
    if not model_args.model_path:
        raise ValueError("--model_path is required.")
    if not model_args.llm_path:
        model_args.llm_path = model_args.model_path

    llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.qk_norm_und = model_args.llm_qk_norm_und
    llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    llm_config.apply_qwen_2_5_vl_pos_emb = training_args.apply_qwen_2_5_vl_pos_emb
    language_model = Qwen2ForCausalLM(llm_config)

    vit_model, vit_config = None, None
    if training_args.visual_und:
        if model_args.vit_type not in ("qwen2_5_vl", "qwen_2_5_vl_original"):
            raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")
        if not model_args.vit_path:
            raise ValueError("--vit_path is required when visual_und=True.")
        vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
        vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
        vit_weights = load_file(os.path.join(model_args.vit_path, "vit.safetensors"))
        vit_model.load_state_dict(vit_weights, strict=True)
        del vit_weights

    vae_model, vae_config = None, None
    if training_args.visual_gen:
        vae_model = WanVideoVAE(device="cpu")
        vae_config = deepcopy(vae_model.vae_config)

    config = LanceConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_num_frames=model_args.max_num_frames,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Lance(
        language_model=language_model,
        vit_model=vit_model if training_args.visual_und else None,
        vit_type=model_args.vit_type,
        config=config,
        training_args=training_args,
    )

    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.init_from_model_path:
        init_from_model_path_if_needed(model, model_args, logger)

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if model_args.vit_type.lower() == "qwen2_5_vl":
        language_model = hack_qwen2_5_vl_config(language_model)
    new_token_ids.update({"image_token_id": language_model.config.video_token_id})
    model.update_tokenizer(tokenizer)

    if model_args.tie_word_embeddings:
        model.language_model.untie_lm_head()
        model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
        model_args.tie_word_embeddings = False
        llm_config.tie_word_embeddings = False

    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vae and vae_model is not None and hasattr(vae_model, "vae"):
        vae_model.vae.model.requires_grad_(False)

    model = model.to(device=device)
    logger.info(f"Model parameters: {count_parameters(model) / 1e9:.2f}B")
    return model, vae_model, tokenizer, new_token_ids


def _cat_or_none(value):
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(x, torch.Tensor) for x in value):
            return torch.cat(value, dim=0)
    if isinstance(value, torch.Tensor) and value.numel() == 0:
        return None
    return value


def _tensor_or_none(value, device, dtype=None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.to(device=device)
        return value.to(dtype=dtype) if dtype is not None else value
    if isinstance(value, list):
        if not value:
            return None
        return torch.tensor(value, device=device, dtype=dtype)
    return value


def prepare_batch(batch, vae_model, training_args, device):
    data = batch.cuda(device).to_dict()
    ce_loss_weights = data.pop("ce_loss_weights", None)
    data.pop("nested_attention_masks", None)

    data["sequence_length"] = int(sum(data["sample_lens"]))
    if "vae_latent_shapes" in data:
        data["patchified_vae_latent_shapes"] = data.pop("vae_latent_shapes")
    data["packed_latent_position_ids"] = _cat_or_none(data.get("packed_latent_position_ids"))
    data["packed_vit_position_ids"] = _cat_or_none(data.get("packed_vit_position_ids"))
    data["packed_label_ids"] = _tensor_or_none(data.get("packed_label_ids"), device, torch.long)
    data["packed_timesteps"] = _tensor_or_none(data.get("packed_timesteps"), device, torch.float32)
    ce_loss_weights = _tensor_or_none(ce_loss_weights, device, torch.float32)

    if data.get("vit_video_grid_thw") is not None and isinstance(data["vit_video_grid_thw"], torch.Tensor) and data["vit_video_grid_thw"].numel() == 0:
        data["vit_video_grid_thw"] = None
    if data.get("vae_video_grid_thw") is not None and isinstance(data["vae_video_grid_thw"], torch.Tensor) and data["vae_video_grid_thw"].numel() == 0:
        data["vae_video_grid_thw"] = None

    for key in ("packed_vit_token_indexes", "packed_vae_token_indexes", "mse_loss_indexes", "ce_loss_indexes"):
        data[key] = _tensor_or_none(data.get(key), device, torch.long)
    packed_vit_tokens = data.get("packed_vit_tokens")
    if packed_vit_tokens is None or (isinstance(packed_vit_tokens, list) and len(packed_vit_tokens) == 0):
        data["packed_vit_tokens"] = None

    if data.get("sample_task") is None or (isinstance(data.get("sample_task"), list) and not data["sample_task"]):
        data["sample_task"] = torch.zeros(data["sequence_length"], device=device)
    if data.get("sample_modality") is None or (isinstance(data.get("sample_modality"), list) and not data["sample_modality"]):
        data["sample_modality"] = torch.zeros(data["sequence_length"], device=device)

    padded_videos = data.pop("padded_videos", None)
    vae_data_mode = data.get("vae_data_mode", [])
    if training_args.visual_gen and padded_videos:
        data["padded_latent"] = make_padded_latent(padded_videos, vae_data_mode, vae_model)

    allowed = {
        "sequence_length",
        "packed_text_ids",
        "packed_text_indexes",
        "sample_lens",
        "sample_type",
        "sample_N_target",
        "packed_position_ids",
        "split_lens",
        "attn_modes",
        "ce_loss_indexes",
        "packed_label_ids",
        "packed_vit_tokens",
        "packed_vit_token_indexes",
        "packed_vit_position_ids",
        "vit_token_seqlens",
        "vit_video_grid_thw",
        "vae_video_grid_thw",
        "video_grid_thw",
        "padded_latent",
        "patchified_vae_latent_shapes",
        "packed_latent_position_ids",
        "packed_vae_token_indexes",
        "packed_timesteps",
        "mse_loss_indexes",
        "vit_data_mode",
        "sample_task",
        "sample_modality",
    }
    model_data = {key: value for key, value in data.items() if key in allowed}
    return model_data, ce_loss_weights


def reduce_mean(value):
    tensor = torch.tensor(float(value), device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / dist.get_world_size()


def main():
    assert torch.cuda.is_available(), "Lance finetuning requires CUDA."
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.results_dir, exist_ok=True)
    os.makedirs(training_args.checkpoint_dir, exist_ok=True)
    logger = create_logger(training_args.results_dir if rank == 0 else None, rank)
    if rank == 0 and wandb is not None:
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(vars(model_args))
        wandb.config.update(vars(data_args))
        wandb.config.update(vars(training_args))

    set_seed(training_args.global_seed * world_size + rank)

    model, vae_model, tokenizer, new_token_ids = build_model_and_tokenizer(model_args, training_args, device, logger)

    resume_from = None
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir) or training_args.resume_from
    else:
        resume_from = training_args.resume_from
    resume_model_only = training_args.resume_model_only
    finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False

    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )

    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema)
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=grad_checkpoint_check_fn,
    )

    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps)
    else:
        raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")

    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(resume_from, optimizer, scheduler, fsdp_config)

    dataset = build_finetune_dataset(
        data_args.dataset_config_file,
        tokenizer,
        data_args,
        model_args,
        training_args,
        new_token_ids,
        local_rank=rank,
        world_size=world_size,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=training_args.global_seed)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=simple_custom_collate,
        drop_last=True,
    )
    if data_args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = data_args.prefetch_factor
    train_loader = DataLoader(**loader_kwargs)

    if vae_model is not None:
        vae_model.to(device)
        vae_model.vae.model.eval()
    fsdp_model.train()
    ema_model.eval()

    logger.info(f"Training Lance for {training_args.total_steps} steps, starting at step {train_step}.")
    optimizer.zero_grad()
    start = time.time()
    epoch = 0
    micro_step = 0
    total_norm = torch.tensor(0.0, device=device)

    while train_step < training_args.total_steps:
        sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            if train_step >= training_args.total_steps:
                break

            data, ce_loss_weights = prepare_batch(batch, vae_model, training_args, device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                loss_dict = fsdp_model(**data)

            loss = torch.tensor(0.0, device=device)
            ce_tokens = torch.tensor(0, device=device)
            if loss_dict["ce"] is not None:
                ce = loss_dict["ce"]
                ce_tokens = torch.tensor(len(data["ce_loss_indexes"]), device=device)
                dist.all_reduce(ce_tokens, op=dist.ReduceOp.SUM)
                if training_args.ce_loss_reweighting and ce_loss_weights is not None:
                    ce = ce * ce_loss_weights
                    total_weights = ce_loss_weights.sum()
                    dist.all_reduce(total_weights, op=dist.ReduceOp.SUM)
                    ce = ce.sum() * world_size / total_weights
                else:
                    ce = ce.sum() * world_size / ce_tokens.clamp_min(1)
                loss = loss + training_args.ce_weight * ce
                loss_dict["ce"] = ce.detach()
            else:
                loss_dict["ce"] = torch.tensor(0.0, device=device)

            mse_tokens = torch.tensor(0, device=device)
            if loss_dict["mse"] is not None:
                mse_tokens = torch.tensor(len(data["mse_loss_indexes"]), device=device)
                dist.all_reduce(mse_tokens, op=dist.ReduceOp.SUM)
                mse = loss_dict["mse"].mean(dim=-1).sum() * world_size / mse_tokens.clamp_min(1)
                loss = loss + training_args.mse_weight * mse
                loss_dict["mse"] = mse.detach()
            else:
                loss_dict["mse"] = torch.tensor(0.0, device=device)

            loss = loss / training_args.gradient_accumulation_steps
            loss.backward()

            did_update = False
            if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
                total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
                optimizer.zero_grad(set_to_none=True)
                did_update = True
                train_step += 1

            if did_update and train_step % training_args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = max(time.time() - start, 1e-6)
                msg = f"(step={train_step:07d}) "
                log_payload = {"lr": optimizer.param_groups[0]["lr"], "total_norm": float(total_norm)}
                for key in ("ce", "mse"):
                    value = reduce_mean(loss_dict[key].item())
                    msg += f"Train Loss {key}: {value:.4f}, "
                    log_payload[key] = value
                log_payload["ce_tokens"] = int(ce_tokens.item())
                log_payload["mse_tokens"] = int(mse_tokens.item())
                log_payload["steps_per_sec"] = training_args.log_every / elapsed
                msg += f"Steps/Sec: {log_payload['steps_per_sec']:.2f}, LR: {log_payload['lr']:.2e}"
                logger.info(msg)
                if rank == 0:
                    print(msg, flush=True)
                    if wandb is not None:
                        wandb.log(log_payload, step=train_step)
                start = time.time()

            if did_update and train_step > 0 and training_args.save_every > 0 and train_step % training_args.save_every == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                FSDPCheckpoint.fsdp_save_ckpt(
                    ckpt_dir=training_args.checkpoint_dir,
                    train_steps=train_step,
                    model=fsdp_model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    logger=logger,
                    fsdp_config=fsdp_config,
                    data_status=data_status,
                )
                gc.collect()
                torch.cuda.empty_cache()

            micro_step += 1

    final_step = min(training_args.total_steps, train_step)
    if final_step > 0 and (training_args.save_every <= 0 or final_step % training_args.save_every != 0):
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir,
            train_steps=final_step,
            model=fsdp_model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=data_status,
        )
    if rank == 0 and wandb is not None:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
