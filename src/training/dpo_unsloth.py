"""Stage 2: Unsloth DPO on top of SFT LoRA adapter."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.training import gpu_utils

# 选择 GPU 必须在 import torch / unsloth 之前完成
gpu_utils.setup_cuda_visible_devices()

from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from trl import DPOTrainer, DPOConfig

PatchDPOTrainer()

from src.training.data_utils import load_dpo_dataset, load_system_prompt

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 DPO for Hasumi role-play LLM")
    gpu_utils.add_gpu_arg(parser)
    parser.add_argument(
        "--sft_adapter_path",
        type=str,
        default="outputs/sft_qwen25_7b_hasumi_lora",
        help="Stage 1 SFT 输出的 LoRA adapter 路径",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="基座模型名称（仅当 sft_adapter_path 不包含 base 模型信息时使用）",
    )
    parser.add_argument(
        "--dpo_data_path",
        type=str,
        default="data/dpo_train.jsonl",
        help="DPO 偏好对路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dpo_qwen25_7b_hasumi_lora",
        help="输出目录",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="最大序列长度",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="单卡 batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="训练轮数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="学习率",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="warmup 比例",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="DPO 最大 prompt 长度",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="日志步数",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="保存 checkpoint 步数",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "tensorboard", "wandb"],
        help="训练日志上报目标",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    selected_gpu, _ = gpu_utils.setup_cuda_visible_devices(args.gpu)
    logger.info("Stage 2 DPO 启动，使用 GPU %d", selected_gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt()
    logger.info("System prompt 长度：%d 字符", len(system_prompt))

    # 加载 SFT adapter（包含基座 4-bit 模型 + LoRA 权重）
    adapter_path = Path(args.sft_adapter_path)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        load_path = str(adapter_path)
        logger.info("从 SFT adapter 加载：%s", load_path)
    else:
        load_path = args.base_model_name
        logger.warning("未找到 SFT adapter，将使用基座模型：%s", load_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # 如果是基座模型，需要重新加 LoRA（正常应走上面的 adapter 路径）
    if not (adapter_path.exists() and (adapter_path / "adapter_config.json").exists()):
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            max_seq_length=args.max_seq_length,
        )

    # 加载 DPO 数据集
    logger.info("加载 DPO 数据集：%s", args.dpo_data_path)
    train_dataset = load_dpo_dataset(Path(args.dpo_data_path), tokenizer, system_prompt)

    training_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        gradient_checkpointing=True,
        report_to=args.report_to,
        remove_unused_columns=False,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info("开始 DPO 训练...")
    trainer.train(resume_from_checkpoint=False)

    logger.info("保存 DPO LoRA adapter 到 %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Stage 2 DPO 完成")


if __name__ == "__main__":
    main()
