"""数据产物质量校验。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import config
from .validators import validate_dpo_pair, validate_sft_sample


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def validate():
    print("=" * 50)
    print("数据产物校验")
    print("=" * 50)

    errors = []

    # 1. 场景
    scene_count = count_lines(config.SCENES_PATH)
    print(f"scenes.jsonl: {scene_count} 场景")
    if scene_count == 0:
        errors.append("scenes.jsonl 为空")

    # 2. 角色卡
    if config.CHARACTER_CARD_PATH.exists():
        with open(config.CHARACTER_CARD_PATH, "r", encoding="utf-8") as f:
            card = json.load(f)
        print(f"角色卡: {card.get('name')}，关系 {len(card.get('relationships', {}))} 条")
    else:
        errors.append("角色卡不存在")

    # 3. SFT
    for split, path in [("train", config.SFT_TRAIN_PATH), ("test", config.SFT_TEST_PATH)]:
        n = count_lines(path)
        print(f"sft_{split}.jsonl: {n} 条")
        if n == 0:
            errors.append(f"sft_{split}.jsonl 为空")
            continue
        with open(path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        bad = [i for i, s in enumerate(samples) if not validate_sft_sample(s.get("messages", []))]
        if bad:
            errors.append(f"sft_{split} 有 {len(bad)} 条格式异常")
        assistant_turns = sum(
            1 for s in samples for m in s["messages"] if m["role"] == "assistant"
        )
        avg_turns = round(assistant_turns / max(len(samples), 1), 2)
        print(f"  - assistant 平均轮数: {avg_turns}")

    # 训练/测试场景不重叠
    train_scenes = set()
    test_scenes = set()
    if config.SFT_TRAIN_PATH.exists():
        with open(config.SFT_TRAIN_PATH, "r", encoding="utf-8") as f:
            train_scenes = {json.loads(line).get("scene_id", "") for line in f if line.strip()}
    if config.SFT_TEST_PATH.exists():
        with open(config.SFT_TEST_PATH, "r", encoding="utf-8") as f:
            test_scenes = {json.loads(line).get("scene_id", "") for line in f if line.strip()}
    overlap = train_scenes & test_scenes
    if overlap:
        errors.append(f"训练/测试场景存在重叠: {len(overlap)} 个")
    else:
        print("训练/测试场景无重叠 ✓")

    # 4. DPO
    n = count_lines(config.DPO_TRAIN_PATH)
    print(f"dpo_train.jsonl: {n} 条")
    if n == 0:
        errors.append("dpo_train.jsonl 为空")
    else:
        with open(config.DPO_TRAIN_PATH, "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        bad = [i for i, p in enumerate(pairs) if not validate_dpo_pair(p)]
        if bad:
            errors.append(f"DPO 有 {len(bad)} 条格式异常")
        type_dist = Counter(p.get("question_type", "unknown") for p in pairs)
        print(f"  - 类型分布: {dict(type_dist)}")

    print("=" * 50)
    if errors:
        print("发现异常：")
        for e in errors:
            print(f"  - {e}")
    else:
        print("全部校验通过 ✓")
    print("=" * 50)


if __name__ == "__main__":
    validate()
