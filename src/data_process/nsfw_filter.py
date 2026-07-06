"""NSFW / R18 内容过滤。

基于关键词屏蔽从 SFT 样本中移除成人场景。过滤对象是所有 message 的 content，
包括 system 携带的场景描述。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from . import config

logger = logging.getLogger(__name__)

# 按类别组织的显式 R18 关键词。为了降低误伤，只放性器官/性行为/体液等强敏感词。
NSFW_KEYWORDS = frozenset(
    {
        # 性器官
        "阴茎",
        "阳具",
        "肉棒",
        "肉棍",
        "龟头",
        "阴道",
        "小穴",
        "阴户",
        "秘部",
        "秘裂",
        "阴唇",
        "阴核",
        "阴蒂",
        "乳头",
        "乳晕",
        "乳房",
        "胸部" "私处",
        "性器",
        "生殖器",
        "性器官",
        # 性行为/体位
        "性交",
        "做爱",
        "发生关系",
        "发生性关系",
        "同房",
        "行房",
        "交合",
        "交尾",
        "口交",
        "乳交",
        "肛交",
        "手交",
        "足交",
        "口内",
        "插入",
        "抽插",
        "内插",
        "骑乘位",
        "正常位",
        "背后位",
        "侧卧位",
        "狗爬式",
        "老汉推车",
        "座位式",
        "站立位",
        # 体液/高潮
        "精液",
        "精子",
        "射精",
        "射出",
        "中出",
        "内射",
        "颜射",
        "口爆",
        "胸射",
        "爱液",
        "淫水",
        "潮吹",
        "失禁",
        "喷出",
        "高潮",
        "绝顶",
        # 性行为动作
        "舔",
        "吸吮",
        "吸允",
        "含住",
        "吞吐",
        "缠绕",
        "摩擦",
        "搓揉",
        "揉捏",
        "勃起",
        "坚挺",
        "湿透了",
        # 性相关状态/描述
        "淫靡",
        "淫乱",
        "淫荡",
        "发情",
        "色情",
        "肉欲",
        "情欲",
        "性欲",
        "性冲动",
        "快感",
        "娇声",
        "娇喘",
        "喘息",
        "喘着",
        "呻吟",
        # 禁忌/凌辱类
        "强奸",
        "轮奸",
        "凌辱",
        "侵犯",
        "猥亵",
        "性骚扰",
        "调教",
        "肉便器",
        "性奴",
        "奴隶",
        "奴隶化",
        # 其他
        "破处",
        "处女膜",
        "初夜",
        "裸身",
        "全裸",
        "一丝不挂",
        "赤身裸体",
        "自慰",
        "手淫",
        "抚摩",
    }
)

# 预编译正则，用于把常见变体归一化（重复字符、间隔符号等）
_NORMALIZE_RE = re.compile(r"[\s·•]+|")


def _normalize(text: str) -> str:
    """简单归一化：去掉零宽空格、多余空格和分隔点。"""
    text = text.replace("\u200b", "").replace("\u200c", "")
    return _NORMALIZE_RE.sub("", text)


def is_nsfw(text: str) -> bool:
    """判断文本是否包含 NSFW 关键词。"""
    normalized = _normalize(text)
    return any(kw in normalized for kw in NSFW_KEYWORDS)


def contains_nsfw(sample: Dict) -> bool:
    """判断一条 SFT 样本是否包含 NSFW 内容。"""
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return False
    return any(is_nsfw(m.get("content", "")) for m in messages)


def filter_nsfw_samples(
    samples: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """返回 (保留样本, 被过滤样本)。"""
    kept: List[Dict] = []
    removed: List[Dict] = []
    for sample in samples:
        if contains_nsfw(sample):
            removed.append(sample)
        else:
            kept.append(sample)
    return kept, removed


def filter_sft_files(
    paths: Iterable[Path],
    dry_run: bool = False,
) -> Dict[str, dict]:
    """过滤一个或多个 SFT jsonl 文件，默认直接覆盖原文件。"""
    report: Dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            logger.warning("文件不存在：%s", path)
            continue
        samples: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        kept, removed = filter_nsfw_samples(samples)
        report[str(path)] = {
            "before": len(samples),
            "after": len(kept),
            "removed": len(removed),
        }
        logger.info(
            "NSFW 过滤 %s：%d -> %d（移除 %d）",
            path, len(samples), len(kept), len(removed)
        )
        if removed:
            scene_ids = sorted({s.get("scene_id", "n/a") for s in removed if "scene_id" in s})
            logger.info("移除样本涉及 scene_id：%s", scene_ids[:50])
        if not dry_run:
            with open(path, "w", encoding="utf-8") as f:
                for sample in kept:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="从 SFT 数据集中过滤 R18/NSFW 样本")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计不覆盖文件",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        default=[config.SFT_TRAIN_PATH, config.SFT_TEST_PATH],
        help="要过滤的 jsonl 文件路径",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = filter_sft_files(args.paths, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
