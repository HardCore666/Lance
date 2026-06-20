# Lance 微调说明

## 支持能力

当前微调入口支持以下任务：

- 文生视频 / 文生图
- 图生视频
- 图片 / 视频编辑
- 图片 / 视频理解问答
- 理解数据和生成数据混合训练

## 主要文件

```text
train/finetune_lance.py                  # 微调入口
data/finetune_dataset.py                 # 微调数据读取与打包
data/configs/lance_finetune_example.yaml # 数据配置示例
scripts/train_lance.sh                   # 启动脚本示例
```

## 数据配置

微调数据通过 YAML 配置。示例：

```yaml
datasets:
  - path: /path/to/train.jsonl
    data_root: /path/to/media/root
    task: t2v
    resolution: video_480p
    text_template: false
    num_frames: 49
    height: 480
    width: 848
    max_duration: 6.0
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `path` | 训练数据文件，支持 JSON / JSONL |
| `data_root` | 图片、视频相对路径的根目录 |
| `task` | 任务类型，如 `t2v`、`i2v`、`video_edit`、`x2t_video` |
| `resolution` | 分辨率预设，如 `video_480p`、`image_768res` |
| `num_frames` | 生成目标视频帧数 |
| `height` / `width` | 生成目标尺寸 |
| `max_duration` | 输入视频最长采样时长 |

## 数据格式

### 视频理解

```json
{
  "0001": {
    "interleave_array": [
      "videos/example.mp4",
      [
        "Watch the video carefully and answer the question.",
        "What is happening in the video?",
        "A person is cooking in a kitchen."
      ]
    ],
    "element_dtype_array": ["video", "text"],
    "istarget_in_interleave": [0, 1]
  }
}
```

### 文生视频

JSONL 每行一个样本：

```json
{"prompt": "A red panda surfing on the sea.", "video": "videos/red_panda.mp4"}
```

### 图生视频 / 视频编辑

```json
{
  "prompt": "Make the car drive forward.",
  "source_video": "inputs/car.mp4",
  "target_video": "targets/car_drive.mp4"
}
```

也可以使用图片：

```json
{
  "prompt": "Animate this image into a short video.",
  "source_image": "inputs/cat.png",
  "target_video": "targets/cat_video.mp4"
}
```

## 启动训练

单机多卡示例：

```bash
torchrun \
  --nproc_per_node=8 \
  --master_port=29501 \
  train/finetune_lance.py \
  --model_path downloads/Lance_3B_Video \
  --llm_path downloads/Lance_3B_Video \
  --vit_path downloads/Qwen2.5-VL-ViT \
  --vit_type qwen_2_5_vl_original \
  --dataset_config_file data/configs/lance_finetune_example.yaml \
  --results_dir results/lance_finetune \
  --checkpoint_dir results/lance_finetune/checkpoints \
  --visual_gen True \
  --visual_und True \
  --max_latent_size 64 \
  --max_num_frames 49 \
  --latent_patch_size 1 1 1 \
  --sharding_strategy FULL_SHARD \
  --num_shard 8 \
  --lr 2e-5 \
  --warmup_steps 100 \
  --total_steps 10000 \
  --save_every 1000 \
  --log_every 10
```

也可以使用脚本：

```bash
MODEL_PATH=downloads/Lance_3B_Video \
VIT_PATH=downloads/Qwen2.5-VL-ViT \
DATASET_CONFIG=data/configs/lance_finetune_example.yaml \
NUM_GPUS=8 \
bash scripts/train_lance.sh
```

OpenVid1M 6 秒视频配置示例：

```bash
MODEL_PATH=downloads/Lance_3B_Video \
VIT_PATH=downloads/Qwen2.5-VL-ViT \
DATASET_CONFIG=data/configs/openvid1m_t2v.yaml \
OUTPUT_DIR=results/openvid1m_finetune \
NUM_GPUS=8 \
MAX_NUM_FRAMES=73 \
bash scripts/train_lance.sh
```

`max_duration: 6.0` 时，Lance 在线视频采样按 12 FPS 最多会到 73 帧，因此 `MAX_NUM_FRAMES` 需要设为 73；如果改成约 4 秒视频，可使用 49。

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--model_path` | Lance 模型目录 |
| `--vit_path` | Qwen2.5-VL ViT 目录 |
| `--dataset_config_file` | 数据配置文件 |
| `--max_num_frames` | 最大目标视频帧数，应不小于数据里的 `num_frames` |
| `--max_latent_size` | 最大 latent 空间尺寸，480p 视频建议设为 `64` |
| `--freeze_llm` | 是否冻结语言模型 |
| `--freeze_vit` | 是否冻结 ViT，默认不冻结 |
| `--freeze_vae` | 是否冻结 VAE，默认冻结 |

## Checkpoint

训练会保存：

```text
results/lance_finetune/checkpoints/0001000/
  model.safetensors
  ema.safetensors
  optimizer.*.pt
  scheduler.pt
```

继续训练：

```bash
torchrun --nproc_per_node=8 train/finetune_lance.py ... --auto_resume True
```

只加载模型权重继续微调：

```bash
torchrun --nproc_per_node=8 train/finetune_lance.py ... \
  --resume_from results/lance_finetune/checkpoints/0001000 \
  --resume_model_only True \
  --finetune_from_ema True
```

推理时，将 `model_path` 指向保存的 checkpoint 目录即可。
