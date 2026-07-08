"""角色风格清洗：保留八纯身份，剥离游戏角色关系与剧情。

策略：
- 只处理明确的多字角色名和带身份后缀的单字形式（如「幸君」「月酱」）。
- 单字常见汉字（如「幸」「月」「滨」）在普通词中（「万幸」「月亮」）不误伤。
- user 消息中的角色前缀（如「绘未：」「幸：」）直接剥离。
- assistant 消息中仍残留明确角色名/剧情词的样本才丢弃。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from . import config
from .utils import setup_logging

logger = logging.getLogger(__name__)

# 明确的多字角色名
MULTI_CHAR_NAMES = {
    "绘未", "绘未酱", "绘未同学",
    "新海君", "新海",
    "小月", "月酱",
    "小桃", "桃子", "桃酱",
    "咲希", "咲希酱",
    "小滨",
    "千夏", "千夏酱", "千夏同学",
    "椿", "椿酱",
    "梦乃", "吾郎", "小夏", "爱季",
    "绘未妈妈",
}

# 带身份后缀的单字形式
SUFFIXED_SINGLE = {
    "幸": ["幸君", "幸同学", "幸酱"],
    "月": ["月酱", "月同学"],
    "滨": ["滨同学", "滨酱"],
}

# user 消息前缀模式：角色名 + 全角/半角冒号
USER_PREFIX_PATTERNS = [
    r"^绘未[：:]\s*",
    r"^幸[：:]\s*",
    r"^新海[：:]\s*",
    r"^新海君[：:]\s*",
    r"^小月[：:]\s*",
    r"^小桃[：:]\s*",
    r"^咲希[：:]\s*",
    r"^小滨[：:]\s*",
    r"^千夏[：:]\s*",
    r"^椿[：:]\s*",
    r"^梦乃[：:]\s*",
    r"^吾郎[：:]\s*",
    r"^小夏[：:]\s*",
    r"^爱季[：:]\s*",
]

# 剧情/关系关键词
FORBIDDEN_TERMS = {
    "租赁恋人", "借恋", "恋爱，我借走了",
}

# 替换表：角色名 → 泛化词
NAME_REPLACEMENTS = {
    "绘未": "朋友",
    "绘未酱": "朋友",
    "绘未同学": "同学",
    "新海君": "同学",
    "新海": "同学",
    "小月": "朋友",
    "月酱": "朋友",
    "小桃": "同学",
    "桃子": "同学",
    "桃酱": "同学",
    "咲希": "同学",
    "咲希酱": "同学",
    "小滨": "同学",
    "千夏": "学妹",
    "千夏酱": "学妹",
    "千夏同学": "学妹",
    "椿": "朋友",
    "椿酱": "朋友",
    "梦乃": "朋友",
    "吾郎": "朋友",
    "小夏": "朋友",
    "爱季": "朋友",
    "绘未妈妈": "朋友的妈妈",
    "幸君": "同学",
    "幸同学": "同学",
    "幸酱": "同学",
    "月同学": "朋友",
    "滨同学": "同学",
    "滨酱": "同学",
}

TERM_REPLACEMENTS = {
    "租赁恋人": "那件事",
    "借恋": "那件事",
    "恋爱，我借走了": "那件事",
}


def build_replacement_pattern(replacements: Dict[str, str]) -> re.Pattern:
    sorted_items = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
    escaped = [re.escape(k) for k, _ in sorted_items]
    return re.compile("|".join(escaped))


NAME_PATTERN = build_replacement_pattern(NAME_REPLACEMENTS)
TERM_PATTERN = build_replacement_pattern(TERM_REPLACEMENTS)
USER_PREFIX_RE = re.compile("|".join(USER_PREFIX_PATTERNS))


def replace_names(text: str) -> str:
    def _repl(m: re.Match) -> str:
        return NAME_REPLACEMENTS[m.group(0)]
    return NAME_PATTERN.sub(_repl, text)


def replace_terms(text: str) -> str:
    def _repl(m: re.Match) -> str:
        return TERM_REPLACEMENTS[m.group(0)]
    return TERM_PATTERN.sub(_repl, text)


def strip_user_prefixes(text: str) -> str:
    """移除 user 消息中的角色名前缀，多行时逐行处理。"""
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        new_lines.append(USER_PREFIX_RE.sub("", line))
    return "\n".join(new_lines)


def contains_explicit_name(text: str) -> bool:
    """检查是否仍包含明确的角色名（单字普通词不算）。"""
    for name in MULTI_CHAR_NAMES:
        if name in text:
            return True
    for single, forms in SUFFIXED_SINGLE.items():
        for form in forms:
            if form in text:
                return True
    return False


def contains_forbidden_term(text: str) -> bool:
    for term in FORBIDDEN_TERMS:
        if term in text:
            return True
    return False


def count_substring(text: str, target: str) -> int:
    return text.count(target)


def clean_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]] | None:
    cleaned = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if not content.strip():
            return None
        # user 消息先剥离角色前缀
        if role == "user":
            content = strip_user_prefixes(content)
        content = replace_names(content)
        content = replace_terms(content)
        cleaned.append({"role": role, "content": content})

    # 格式校验
    if not cleaned or cleaned[0]["role"] != "system":
        return None
    roles = [m["role"] for m in cleaned]
    if "user" not in roles or "assistant" not in roles:
        return None
    if roles[-1] != "assistant":
        return None

    assistant_texts = [m["content"] for m in cleaned if m["role"] == "assistant"]
    final_answer = assistant_texts[-1]

    # assistant 中仍残留明确角色名或剧情词 → 丢弃
    for text in assistant_texts:
        if contains_explicit_name(text) or contains_forbidden_term(text):
            return None

    # 过滤过短 / 空泛
    if len(final_answer.strip()) < 8:
        return None

    # 控制「哟呵呵」滥用
    for text in assistant_texts:
        if count_substring(text, "哟呵呵") >= 3:
            return None
    if re.sub(r"[哟呵呵~～\s！。，？]", "", final_answer) == "":
        return None

    return cleaned


def clean_sft_samples(samples: List[Dict]) -> List[Dict]:
    kept: List[Dict] = []
    dropped_reasons = {
        "format_error": 0,
        "explicit_name": 0,
        "forbidden_term": 0,
        "too_short": 0,
        "yohoho_abuse": 0,
    }

    for sample in samples:
        messages = sample.get("messages", [])
        if not messages:
            dropped_reasons["format_error"] += 1
            continue

        assistant_texts = [m["content"] for m in messages if m.get("role") == "assistant"]
        if not assistant_texts:
            dropped_reasons["format_error"] += 1
            continue

        # 原始 assistant 若含明确角色名/剧情词，记录原因后丢弃
        raw_has_forbidden = any(
            contains_explicit_name(t) or contains_forbidden_term(t) for t in assistant_texts
        )
        if raw_has_forbidden:
            if any(contains_forbidden_term(t) for t in assistant_texts):
                dropped_reasons["forbidden_term"] += 1
            else:
                dropped_reasons["explicit_name"] += 1
            continue

        cleaned = clean_messages(messages)
        if cleaned is None:
            final = assistant_texts[-1]
            if len(final.strip()) < 8:
                dropped_reasons["too_short"] += 1
            elif any(count_substring(t, "哟呵呵") >= 3 for t in assistant_texts):
                dropped_reasons["yohoho_abuse"] += 1
            else:
                dropped_reasons["format_error"] += 1
            continue

        new_sample = dict(sample)
        new_sample["messages"] = cleaned
        new_sample["source"] = new_sample.get("source", "script") + "_cleaned"
        kept.append(new_sample)

    logger.info("SFT 清洗完成：保留 %d / %d", len(kept), len(samples))
    logger.info("丢弃原因统计：%s", json.dumps(dropped_reasons, ensure_ascii=False))
    return kept


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


def main(
    input_path: Path = config.SFT_TRAIN_PATH,
    output_path: Path = config.SFT_TRAIN_PATH,
    backup: bool = True,
):
    setup_logging(config.LOG_DIR / "style_clean.log")
    logger.info("=" * 50)
    logger.info("角色风格清洗启动")
    logger.info("输入: %s", input_path)
    logger.info("输出: %s", output_path)

    src_path = input_path
    if input_path == output_path:
        bak_path = Path(str(input_path) + ".bak")
        if bak_path.exists():
            src_path = bak_path
            logger.info("使用 .bak 作为原始数据")
        else:
            logger.warning("未找到 .bak，将直接清洗当前文件")

    if backup and input_path == output_path:
        backup_path = Path(str(output_path) + ".style_clean_bak")
        if output_path.exists():
            backup_path.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("已备份 %s -> %s", output_path, backup_path)

    samples = load_samples(src_path)
    logger.info("加载 %d 条样本", len(samples))

    cleaned = clean_sft_samples(samples)
    save_samples(output_path, cleaned)
    logger.info("已保存清洗后样本到 %s", output_path)

    # 统计残留
    name_counts: Dict[str, int] = {}
    for s in cleaned:
        text = "\n".join(m["content"] for m in s.get("messages", []))
        for name in MULTI_CHAR_NAMES:
            if name in text:
                name_counts[name] = name_counts.get(name, 0) + 1
    if name_counts:
        logger.warning("清洗后仍残留的多字角色名：%s", name_counts)
    else:
        logger.info("清洗后无多字角色名残留")

    return cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗 SFT 数据，保留八纯身份、剥离游戏关系")
    parser.add_argument("--input", type=Path, default=config.SFT_TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=config.SFT_TRAIN_PATH)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    main(input_path=args.input, output_path=args.output, backup=not args.no_backup)
