"""Stage 1: Unsloth QLoRA SFT on Qwen2.5-7B-Instruct."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# 在 import torch 前选择 GPU
from src.training import gpu_utils

# 选择 GPU 必须在 import torch / unsloth 之前完成
gpu_utils.setup_cuda_visible_devices()

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

from src.training.data_utils import load_sft_dataset, load_system_prompt

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 SFT for Hasumi role-play LLM")
    gpu_utils.add_gpu_arg(parser)
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="基座模型名称或路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sft_train.jsonl",
        help="SFT 训练集路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sft_qwen25_7b_hasumi_lora",
        help="输出目录",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="最大序列长度",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA r",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="单卡 batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="训练轮数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="warmup 比例",
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
        default=10,
        help="日志步数",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
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

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 用户指定 GPU 时重新设置（覆盖自动选择）
    selected_gpu, _ = gpu_utils.setup_cuda_visible_devices(args.gpu)
    logger.info("Stage 1 SFT 启动，使用 GPU %d", selected_gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 system prompt
    system_prompt = load_system_prompt()
    logger.info("System prompt 长度：%d 字符", len(system_prompt))

    # 加载模型 & tokenizer
    logger.info("加载基座模型：%s", args.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    # 加载数据集
    logger.info("加载 SFT 训练集：%s", args.data_path)
    train_dataset = load_sft_dataset(Path(args.data_path), tokenizer, system_prompt)

    # 训练参数
    training_args = SFTConfig(
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
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    logger.info("开始 SFT 训练...")
    trainer.train(resume_from_checkpoint=False)

    # 保存最终 adapter
    logger.info("保存 LoRA adapter 到 %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Stage 1 SFT 完成")


if __name__ == "__main__":
    main()
