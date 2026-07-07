"""合并 LoRA 并导出 GGUF / Ollama Modelfile。"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from src.training import gpu_utils

# 选择 GPU 必须在 import torch / unsloth 之前完成
gpu_utils.setup_cuda_visible_devices()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

from src.training.data_utils import load_system_prompt

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA and export GGUF")
    gpu_utils.add_gpu_arg(parser)
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/dpo_qwen25_7b_hasumi_lora",
        help="最终 DPO adapter 路径",
    )
    parser.add_argument(
        "--base_model_full_path",
        type=str,
        default="models/Qwen2.5-7B-Instruct",
        help="用于合并的 16-bit 基座模型本地路径",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="基座模型名称（adapter 未包含基座信息或需要回退时使用）",
    )
    parser.add_argument(
        "--merged_output_dir",
        type=str,
        default="outputs/hasumi_qwen25_7b_merged",
        help="16-bit 合并模型输出目录",
    )
    parser.add_argument(
        "--gguf_output_dir",
        type=str,
        default="outputs/hasumi_qwen25_7b_gguf",
        help="GGUF 输出目录前缀",
    )
    parser.add_argument(
        "--quant_methods",
        type=str,
        default="q4_k_m,q5_k_m,q8_0",
        help="逗号分隔的量化方法",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="最大序列长度",
    )
    return parser.parse_args()


def write_modelfile(gguf_path: Path, system_prompt: str, output_path: Path) -> None:
    """生成 Ollama Modelfile。"""
    chat_template = '''{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
'''
    content = '''FROM ./{gguf_name}

TEMPLATE """{chat_template}"""

SYSTEM """{system_prompt}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
'''.replace("{gguf_name}", gguf_path.name).replace("{chat_template}", chat_template).replace("{system_prompt}", system_prompt)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Modelfile 已保存到 %s", output_path)


def merge_adapter_to_base(
    adapter_path: Path,
    base_full_path: Path,
    merged_output_dir: Path,
) -> None:
    """使用 Transformers + PEFT 把 LoRA adapter 合并到 16-bit 基座模型。"""
    logger.info("加载 16-bit 基座模型：%s", base_full_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(base_full_path),
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    logger.info("加载 LoRA adapter：%s", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path))

    logger.info("合并 LoRA 到 16-bit 基座...")
    model = model.merge_and_unload()

    merged_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(merged_output_dir),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(merged_output_dir))
    logger.info("16-bit 合并模型已保存到 %s", merged_output_dir)

    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    selected_gpu, _ = gpu_utils.setup_cuda_visible_devices(args.gpu)
    logger.info("合并与导出启动，使用 GPU %d", selected_gpu)

    adapter_path = Path(args.adapter_path)
    if not (adapter_path.exists() and (adapter_path / "adapter_config.json").exists()):
        logger.error("未找到 adapter：%s", adapter_path)
        sys.exit(1)

    base_full_path = Path(args.base_model_full_path)
    if not (base_full_path.exists() and (base_full_path / "config.json").exists()):
        logger.error(
            "未找到 16-bit 基座模型 %s，请先通过 hfd.sh 下载 Qwen/Qwen2.5-7B-Instruct",
            base_full_path,
        )
        sys.exit(1)

    merged_output_dir = Path(args.merged_output_dir)
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 合并 LoRA -> 16-bit HF 模型
    merge_adapter_to_base(adapter_path, base_full_path, merged_output_dir)

    # Step 2: 用 Unsloth 加载合并后的模型并导出 GGUF
    logger.info("加载合并后的模型用于 GGUF 导出...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_output_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    system_prompt = load_system_prompt()
    gguf_base_dir = Path(args.gguf_output_dir)
    quant_methods = [q.strip() for q in args.quant_methods.split(",") if q.strip()]

    logger.info("导出 GGUF 量化版本：%s", quant_methods)
    gguf_source_dir = Path(str(merged_output_dir) + "_gguf")
    if gguf_source_dir.exists():
        shutil.rmtree(gguf_source_dir, ignore_errors=True)

    model.save_pretrained_gguf(
        str(merged_output_dir),
        tokenizer,
        quantization_method=quant_methods,
    )

    if not gguf_source_dir.exists():
        logger.error("未找到 Unsloth 生成的 GGUF 目录：%s", gguf_source_dir)
        sys.exit(1)

    # 将生成的 .gguf 按量化级别整理到各自目录
    for q in quant_methods:
        q_dir = gguf_base_dir / q
        q_dir.mkdir(parents=True, exist_ok=True)
        q_upper = q.upper()
        candidates = [
            f for f in gguf_source_dir.glob("*.gguf") if q_upper in f.name.upper()
        ]
        if not candidates:
            logger.warning("未找到 %s 的 GGUF 文件", q)
            continue
        src = candidates[0]
        dst = q_dir / src.name
        shutil.move(str(src), str(dst))
        logger.info("移动 %s -> %s", src, dst)
        write_modelfile(dst, system_prompt, q_dir / "Modelfile")

    # 清理空的临时 GGUF 目录
    if gguf_source_dir.exists() and not any(gguf_source_dir.glob("*.gguf")):
        shutil.rmtree(gguf_source_dir, ignore_errors=True)

    # 同时在项目根目录生成一个默认 Modelfile（指向第一个量化版本）
    default_gguf_dir = gguf_base_dir / quant_methods[0]
    default_gguf_files = sorted(default_gguf_dir.glob("*.gguf"))
    if default_gguf_files:
        write_modelfile(default_gguf_files[0], system_prompt, Path("Modelfile"))

    logger.info("合并与导出完成")


if __name__ == "__main__":
    main()
