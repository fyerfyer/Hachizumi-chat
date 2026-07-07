"""数据集增强流水线：清洗前缀、风格迁移、身份锚定、DPO 语气对、合并去重。"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from . import config
from .llm_client import LLMClient
from .style_templates import (
    IDENTITY_QUESTIONS,
    INTERJECTIONS,
    generate_negative_samples,
    generate_rule_identity_samples,
    generate_rule_style_variations,
)
from .utils import extract_json, setup_logging
from .validators import validate_dpo_pair, validate_sft_sample

logger = logging.getLogger(__name__)

USER_PREFIX_RE = re.compile(r"^([^：:]+)：\s*")


def backup_files() -> None:
    """备份原始数据文件。"""
    for src in (config.SFT_TRAIN_PATH, config.DPO_TRAIN_PATH):
        dst = Path(str(src) + ".bak")
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("已备份 %s -> %s", src, dst)


def clean_user_prefixes(samples: List[Dict]) -> List[Dict]:
    """移除 user 消息中的角色名前缀（如 绘未： / 幸：）。"""
    cleaned: List[Dict] = []
    for sample in samples:
        messages = [dict(m) for m in sample.get("messages", [])]
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                # 多行 user 内容（如合并后的其他角色）逐行清洗
                lines = content.split("\n")
                new_lines = []
                for line in lines:
                    new_lines.append(USER_PREFIX_RE.sub("", line))
                m["content"] = "\n".join(new_lines)
        sample["messages"] = messages
        cleaned.append(sample)
    return cleaned


def sample_hash(sample: Dict) -> str:
    """基于完整 messages 生成去重签名。"""
    signature = "\n".join(
        f"{m.get('role')}:{m.get('content', '').strip()}" for m in sample.get("messages", [])
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def dedup_samples(samples: List[Dict]) -> List[Dict]:
    """按完整对话内容去重。"""
    seen: Set[str] = set()
    kept: List[Dict] = []
    for s in samples:
        h = sample_hash(s)
        if h in seen:
            continue
        seen.add(h)
        kept.append(s)
    return kept


def extract_unique_assistant_replies(
    samples: List[Dict],
    min_len: int = 8,
    max_len: int = 80,
    top_k: int = 10000,
) -> List[str]:
    """提取 SFT 中不重复的 assistant 回复，用于风格迁移。"""
    replies: Set[str] = set()
    for sample in samples:
        for m in sample.get("messages", []):
            if m.get("role") == "assistant":
                text = m.get("content", "").strip()
                if min_len <= len(text) <= max_len:
                    replies.add(text)
    replies = list(replies)
    rng = random.Random(42)
    rng.shuffle(replies)
    return replies[:top_k]


def build_style_transfer_prompt(replies: List[str]) -> str:
    lines = "\n".join(f"{i}. {r}" for i, r in enumerate(replies))
    return config.STYLE_TRANSFER_PROMPT_TEMPLATE.format(n=len(replies), list=lines)


def parse_numbered_json_object(content: str, n: int) -> List[str]:
    """解析形如 {'0': '...', '1': '...'} 的 JSON 对象。"""
    data = extract_json(content)
    if not isinstance(data, dict):
        logger.warning("风格迁移返回非对象 JSON：%s", content[:200])
        return []
    out = []
    for i in range(n):
        v = data.get(str(i), "")
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def augment_style_with_api(
    client: LLMClient,
    replies: List[str],
    n_total: int,
    batch_size: int = 10,
    max_workers: int = 4,
) -> List[str]:
    """调用 API 批量改写 assistant 回复为八纯风格。"""
    n_total = min(n_total, len(replies))
    selected = replies[:n_total]
    batches = [selected[i : i + batch_size] for i in range(0, len(selected), batch_size)]
    prompts = [build_style_transfer_prompt(b) for b in batches]
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    logger.info("开始风格迁移：%d 条回复分 %d 批", len(selected), len(batches))
    results = client.batch_chat(
        messages_list,
        desc="Style transfer",
        temperature=0.85,
        max_tokens=2048,
        json_mode=True,
        max_workers=max_workers,
    )

    rewritten: List[str] = []
    parse_fail = 0
    for batch, res in zip(batches, results):
        if not res:
            continue
        outs = parse_numbered_json_object(res["content"], len(batch))
        if len(outs) != len(batch):
            parse_fail += 1
        rewritten.extend(outs)

    logger.info("风格迁移完成：成功 %d/%d，解析失败 %d 批", len(rewritten), len(selected), parse_fail)
    return rewritten


def build_identity_style_samples(
    original_replies: List[str],
    rewritten_replies: List[str],
    system_prompt: str,
) -> List[Dict]:
    """把改写后的回复包装成 SFT 样本（单轮 user->assistant）。"""
    samples: List[Dict] = []
    for orig, rew in zip(original_replies, rewritten_replies):
        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": orig},
                {"role": "assistant", "content": rew},
            ],
            "source": "style_transfer",
        })
    return samples


def build_identity_api_prompts(
    questions: List[Dict[str, str]],
    n_per_theme: int,
) -> List[Tuple[str, str]]:
    """按主题分组生成 API prompt。"""
    themes: Dict[str, List[str]] = {}
    for q in questions:
        themes.setdefault(q["type"], []).append(q["question"])

    prompts: List[Tuple[str, str]] = []
    for theme, qs in themes.items():
        # 每个 prompt 围绕一个主题生成 n_per_theme 条
        prompts.append((
            config.IDENTITY_GENERATION_PROMPT_TEMPLATE.format(
                theme=theme,
                n=n_per_theme,
            ),
            theme,
        ))
    return prompts


def generate_identity_samples_api(
    client: LLMClient,
    system_prompt: str,
    n: int = 300,
    max_workers: int = 4,
) -> List[Dict]:
    """调用 API 生成身份锚定样本。"""
    n_per_theme = max(5, n // len(IDENTITY_QUESTIONS))
    prompts_with_theme = build_identity_api_prompts(IDENTITY_QUESTIONS, n_per_theme)
    messages_list = [[{"role": "user", "content": p}] for p, _ in prompts_with_theme]

    logger.info("开始生成身份锚定样本：%d 个主题", len(prompts_with_theme))
    results = client.batch_chat(
        messages_list,
        desc="Identity samples",
        temperature=0.9,
        max_tokens=2048,
        json_mode=True,
        max_workers=max_workers,
    )

    samples: List[Dict] = []
    parse_fail = 0
    for (prompt, theme), res in zip(prompts_with_theme, results):
        if not res:
            continue
        data = extract_json(res["content"])
        if not isinstance(data, list):
            logger.warning("身份样本返回非数组 JSON：%s", res["content"][:200])
            parse_fail += 1
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if not q or not a:
                continue
            samples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source": "identity_api",
                "question_type": theme,
            })

    logger.info("身份锚定样本：API 生成 %d 条", len(samples))
    return samples


def build_style_dpo_prompts(
    questions: List[str],
    batch_size: int = 5,
) -> List[str]:
    batches = [questions[i : i + batch_size] for i in range(0, len(questions), batch_size)]
    prompts = []
    for batch in batches:
        lines = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(batch))
        prompts.append(config.STYLE_DPO_PAIR_PROMPT_TEMPLATE.format(
            n=len(batch),
            questions=lines,
        ))
    return prompts


def generate_style_dpo_pairs(
    client: LLMClient,
    character_card: Dict,
    n: int = 1500,
    batch_size: int = 5,
    max_workers: int = 4,
) -> List[Dict]:
    """生成面向语气风格的 DPO 偏好对。"""
    # 基于角色卡和固定问题生成种子
    seeds = []
    for item in IDENTITY_QUESTIONS:
        seeds.append(item["question"])
    relationships = character_card.get("relationships", {})
    for person in relationships:
        if person == config.TARGET_ROLE:
            continue
        seeds.extend([
            f"你和{person}是什么关系？",
            f"你觉得{person}怎么样？",
        ])
    for point in character_card.get("key_plot_points", []):
        seeds.extend([
            f"关于{point}，你还记得什么？",
            f"你喜不喜欢{point}？",
        ])

    rng = random.Random(42)
    rng.shuffle(seeds)
    # 重复采样直到满足 n
    questions = (seeds * ((n // len(seeds)) + 1))[:n]
    rng.shuffle(questions)

    prompts = build_style_dpo_prompts(questions, batch_size)
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    logger.info("开始生成风格 DPO 对：%d 个问题分 %d 批", len(questions), len(prompts))
    results = client.batch_chat(
        messages_list,
        desc="Style DPO pairs",
        temperature=0.9,
        max_tokens=4096,
        json_mode=True,
        max_workers=max_workers,
    )

    pairs: List[Dict] = []
    parse_fail = 0
    for batch, res in zip(
        [questions[i : i + batch_size] for i in range(0, len(questions), batch_size)],
        results,
    ):
        if not res:
            continue
        data = extract_json(res["content"])
        if not isinstance(data, list):
            logger.warning("DPO 风格对返回非数组 JSON：%s", res["content"][:200])
            parse_fail += 1
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            pair = {
                "prompt": str(item.get("prompt", "")).strip(),
                "chosen": str(item.get("chosen", "")).strip(),
                "rejected": str(item.get("rejected", "")).strip(),
                "question_type": "style",
            }
            if not validate_dpo_pair(pair):
                continue
            # chosen 必须带角色语气词或第一人称
            identity_markers = set(INTERJECTIONS) | {"我", "咱", "本小姐", "八纯"}
            if not any(w in pair["chosen"] for w in identity_markers):
                continue
            pairs.append(pair)

    logger.info("风格 DPO 对：有效 %d/%d，解析失败 %d 批", len(pairs), len(questions), parse_fail)
    return pairs


def load_samples(path: Path) -> List[Dict]:
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_samples(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def load_system_prompt_base(path: Path = config.SYSTEM_PROMPT_PATH) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if "当前场景：{scene_description}" in text:
        text = text.split("当前场景：{scene_description}")[0].strip()
    return text


def main(
    style_samples: int = config.STYLE_AUGMENT_SAMPLES,
    identity_samples: int = config.IDENTITY_SAMPLE_COUNT,
    negative_samples: int = config.NEGATIVE_SAMPLE_COUNT,
    dpo_pairs: int = config.STYLE_DPO_PAIRS,
    batch_size: int = config.STYLE_AUGMENT_BATCH_SIZE,
    dpo_batch_size: int = config.STYLE_DPO_BATCH_SIZE,
    max_workers: int = 4,
    dry_run: bool = False,
):
    setup_logging(config.LOG_DIR / "enhance_dataset.log")
    logger.info("=" * 50)
    logger.info("数据集增强流水线启动")
    logger.info("=" * 50)

    # 1. 备份
    backup_files()

    # 2. 加载原始 SFT（优先用备份，避免二次增强污染）
    src_sft_path = Path(str(config.SFT_TRAIN_PATH) + ".bak")
    if not src_sft_path.exists():
        src_sft_path = config.SFT_TRAIN_PATH
    original_sft = load_samples(src_sft_path)
    logger.info("加载原始 SFT：%d 条", len(original_sft))

    # 3. 清洗 user 前缀
    cleaned_sft = clean_user_prefixes(original_sft)
    logger.info("清洗 user 前缀完成")

    system_prompt = load_system_prompt_base()

    # 4. 风格迁移
    client: Optional[LLMClient] = None
    try:
        client = LLMClient()
        logger.info("API 客户端初始化成功：%s", client.model)
    except Exception as e:
        logger.error("API 客户端初始化失败：%s", e)
        client = None

    style_augmented: List[Dict] = []
    if style_samples > 0:
        replies = extract_unique_assistant_replies(cleaned_sft)
        logger.info("可用于风格迁移的不重复 assistant 回复：%d 条", len(replies))
        n_style = min(style_samples, len(replies))
        if client and not dry_run:
            rewritten = augment_style_with_api(
                client, replies, n_style, batch_size=batch_size, max_workers=max_workers
            )
        else:
            logger.warning("使用规则模板做风格迁移兜底")
            rewritten = generate_rule_style_variations(replies[:n_style], n_style)
        # 用原始回复作为 user prompt，改写回复作为 assistant
        style_augmented = build_identity_style_samples(
            replies[: len(rewritten)], rewritten, system_prompt
        )
        logger.info("风格迁移样本：%d 条", len(style_augmented))

    # 5. 身份锚定样本
    identity_sft: List[Dict] = []
    if identity_samples > 0:
        if client and not dry_run:
            api_identities = generate_identity_samples_api(
                client, system_prompt, n=identity_samples, max_workers=max_workers
            )
            identity_sft.extend(api_identities)
        # 规则兜底补齐
        remaining = identity_samples - len(identity_sft)
        if remaining > 0:
            identity_sft.extend(generate_rule_identity_samples(system_prompt, remaining))
        logger.info("身份锚定样本：%d 条", len(identity_sft))

    # 6. 负面样本
    negative_sft: List[Dict] = []
    if negative_samples > 0:
        negative_sft = generate_negative_samples(system_prompt, negative_samples)
        logger.info("负面样本：%d 条", negative_samples)

    # 7. 合并 SFT 并去重
    all_sft = cleaned_sft + style_augmented + identity_sft + negative_sft
    # 校验格式
    all_sft = [s for s in all_sft if validate_sft_sample(s.get("messages", []))]
    all_sft = dedup_samples(all_sft)
    logger.info("合并后 SFT：%d 条（去重前 %d 条）", len(all_sft), len(cleaned_sft) + len(style_augmented) + len(identity_sft) + len(negative_sft))

    if not dry_run:
        save_samples(config.SFT_TRAIN_PATH, all_sft)
        logger.info("已保存增强 SFT：%s", config.SFT_TRAIN_PATH)

    # 8. 风格 DPO 对
    dpo_data: List[Dict] = []
    if dpo_pairs > 0:
        card_path = config.CHARACTER_CARD_PATH
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        if client and not dry_run:
            dpo_data = generate_style_dpo_pairs(
                client, card, n=dpo_pairs, batch_size=dpo_batch_size, max_workers=max_workers
            )
        else:
            logger.warning("API 不可用或 dry_run，跳过 DPO 风格对生成")
        logger.info("风格 DPO 对：%d 条", len(dpo_data))

    if not dry_run and dpo_data:
        save_samples(config.DPO_TRAIN_PATH, dpo_data)
        logger.info("已保存增强 DPO：%s", config.DPO_TRAIN_PATH)

    # 9. 统计
    stats = {
        "original_sft": len(original_sft),
        "cleaned_sft": len(cleaned_sft),
        "style_augmented": len(style_augmented),
        "identity_samples": len(identity_sft),
        "negative_samples": len(negative_sft),
        "final_sft": len(all_sft),
        "final_dpo": len(dpo_data),
    }
    logger.info("增强统计：%s", json.dumps(stats, ensure_ascii=False, indent=2))
    print("[enhance_dataset] 增强统计：", json.dumps(stats, ensure_ascii=False, indent=2))

    if dry_run:
        logger.info("dry_run 模式，未写入实际数据文件")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="天满八纯数据集增强")
    parser.add_argument("--style-samples", type=int, default=config.STYLE_AUGMENT_SAMPLES)
    parser.add_argument("--identity-samples", type=int, default=config.IDENTITY_SAMPLE_COUNT)
    parser.add_argument("--negative-samples", type=int, default=config.NEGATIVE_SAMPLE_COUNT)
    parser.add_argument("--dpo-pairs", type=int, default=config.STYLE_DPO_PAIRS)
    parser.add_argument("--batch-size", type=int, default=config.STYLE_AUGMENT_BATCH_SIZE)
    parser.add_argument("--dpo-batch-size", type=int, default=config.STYLE_DPO_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="试运行，不写入文件")
    args = parser.parse_args()

    main(
        style_samples=args.style_samples,
        identity_samples=args.identity_samples,
        negative_samples=args.negative_samples,
        dpo_pairs=args.dpo_pairs,
        batch_size=args.batch_size,
        dpo_batch_size=args.dpo_batch_size,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
