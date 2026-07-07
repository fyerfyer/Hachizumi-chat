"""SFT / DPO 数据集加载与格式清洗。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_system_prompt(path: Path = Path("data/system_prompt.txt")) -> str:
    """加载角色 system prompt（去掉 {scene_description} 占位符）。"""
    text = path.read_text(encoding="utf-8").strip()
    # 训练时不需要 "当前场景：{scene_description}" 这一行
    if "当前场景：{scene_description}" in text:
        text = text.split("当前场景：{scene_description}")[0].strip()
    return text


def merge_consecutive_assistant(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """合并连续多条 assistant 消息，保证 user/assistant 严格交替。"""
    if not messages:
        return messages

    merged: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if merged and role == "assistant" and merged[-1]["role"] == "assistant":
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})

    # 如果合并后以 assistant 开头，补一个 generic user 开头
    if merged and merged[0]["role"] == "assistant":
        merged = [{"role": "user", "content": "（和八纯聊起天来）"}] + merged

    # 如果最后一条不是 assistant，丢弃末尾无回复的 user 消息
    if merged and merged[-1]["role"] != "assistant":
        merged = merged[:-1]

    return merged


def load_sft_dataset(
    path: Path,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str | None = None,
) -> Dataset:
    """读取 messages 格式 JSONL，应用 chat template 后返回 HuggingFace Dataset。"""
    raw_samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            messages = sample.get("messages", [])
            if not messages:
                continue

            # 如果外部提供 system_prompt，替换已有的 system 消息
            if system_prompt:
                if messages[0].get("role") == "system":
                    messages[0]["content"] = system_prompt
                else:
                    messages = [{"role": "system", "content": system_prompt}] + messages

            messages = merge_consecutive_assistant(messages)
            if len(messages) < 3:
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            raw_samples.append({"text": text, "messages": messages})

    logger.info("SFT 加载完成：%d 条样本来自 %s", len(raw_samples), path)
    return Dataset.from_list(raw_samples)


def load_dpo_dataset(
    path: Path,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
) -> Dataset:
    """读取 prompt/chosen/rejected 格式 JSONL，包装成 DPOTrainer 可用格式。"""
    raw_samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            prompt = str(sample.get("prompt", "")).strip()
            chosen = str(sample.get("chosen", "")).strip()
            rejected = str(sample.get("rejected", "")).strip()
            if not prompt or not chosen or not rejected:
                continue
            if chosen == rejected:
                continue

            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            chosen_messages = [{"role": "assistant", "content": chosen}]
            rejected_messages = [{"role": "assistant", "content": rejected}]

            raw_samples.append(
                {
                    "prompt": prompt_messages,
                    "chosen": chosen_messages,
                    "rejected": rejected_messages,
                    "question_type": sample.get("question_type", "unknown"),
                }
            )

    logger.info("DPO 加载完成：%d 对样本来自 %s", len(raw_samples), path)
    return Dataset.from_list(raw_samples)
