"""剧本解析：将 0.txt.gz 解析为 scenes.jsonl。"""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from . import config


def normalize_speaker(raw: str) -> str:
    """角色名归一化。"""
    s = raw.strip()
    return config.SPEAKER_ALIASES.get(s, s)


def parse_line(line: str) -> Dict[str, str]:
    """单行解析成 {speaker, text}。"""
    line = line.rstrip("\n\r")
    if "：" in line:
        speaker, text = line.split("：", 1)
    else:
        speaker, text = config.NARRATOR_NAME, line
    speaker = normalize_speaker(speaker)
    return {"speaker": speaker, "text": text.strip()}


def contains_location(text: str) -> bool:
    """旁白中是否出现地点切换关键词。"""
    return any(kw in text for kw in config.LOCATION_KEYWORDS)


def split_scenes(lines: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """基于启发式规则切分场景。"""
    scenes: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    narrator_streak = 0
    gap_without_target = 0

    def flush():
        nonlocal current, narrator_streak, gap_without_target
        if len(current) >= config.MIN_SCENE_LINES:
            scenes.append(current)
        current = []
        narrator_streak = 0
        gap_without_target = 0

    for item in lines:
        speaker = item["speaker"]
        text = item["text"]

        if not current:
            current.append(item)
            narrator_streak = 1 if speaker == config.NARRATOR_NAME else 0
            gap_without_target = 0 if speaker == config.TARGET_ROLE else 1
            continue

        if speaker == config.NARRATOR_NAME:
            narrator_streak += 1
        else:
            narrator_streak = 0

        if speaker == config.TARGET_ROLE:
            gap_without_target = 0
        else:
            gap_without_target += 1

        # 触发场景边界
        boundary = False
        if narrator_streak >= config.SCENE_BOUNDARY_NARRATOR_STREAK:
            boundary = True
        if (
            gap_without_target >= config.MAX_GAP_WITHOUT_TARGET
            and speaker == config.NARRATOR_NAME
            and contains_location(text)
        ):
            boundary = True
        if len(current) >= config.MAX_SCENE_LINES:
            boundary = True

        if boundary:
            flush()
            current.append(item)
            narrator_streak = 1 if speaker == config.NARRATOR_NAME else 0
            gap_without_target = 0 if speaker == config.TARGET_ROLE else 1
        else:
            current.append(item)

    flush()
    return scenes


def build_stats(lines: List[Dict[str, str]], scenes: List[List[Dict[str, str]]]) -> Dict:
    """生成解析统计。"""
    speaker_counts = Counter(item["speaker"] for item in lines)
    total_lines = len(lines)
    hasumi_lines = speaker_counts.get(config.TARGET_ROLE, 0)
    narrator_lines = speaker_counts.get(config.NARRATOR_NAME, 0)
    protagonist_lines = speaker_counts.get(config.PROTAGONIST, 0)

    scene_lengths = [len(s) for s in scenes]
    return {
        "total_lines": total_lines,
        "total_scenes": len(scenes),
        "scene_lengths": {
            "mean": round(sum(scene_lengths) / max(len(scene_lengths), 1), 2),
            "min": min(scene_lengths) if scene_lengths else 0,
            "max": max(scene_lengths) if scene_lengths else 0,
        },
        "speaker_counts": dict(speaker_counts.most_common()),
        "target_role_lines": hasumi_lines,
        "narrator_lines": narrator_lines,
        "protagonist_lines": protagonist_lines,
    }


def main():
    raw_path = config.RAW_GZ
    print(f"[parse_scenario] 读取 {raw_path}")
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        raw_lines = f.readlines()

    parsed = [parse_line(line) for line in raw_lines if line.strip()]
    print(f"[parse_scenario] 解析行数：{len(parsed)}")

    scenes = split_scenes(parsed)
    print(f"[parse_scenario] 切分场景数：{len(scenes)}")

    # 写入 scenes.jsonl
    config.SCENES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SCENES_PATH, "w", encoding="utf-8") as f:
        for idx, scene in enumerate(scenes, 1):
            record = {"scene_id": f"scene_{idx:05d}", "lines": scene}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 写入统计
    stats = build_stats(parsed, scenes)
    with open(config.PARSE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[parse_scenario] 已保存 {config.SCENES_PATH}")
    print(f"[parse_scenario] 八纯台词数：{stats['target_role_lines']}")
    print(f"[parse_scenario] 旁白行数：{stats['narrator_lines']}")
    print(f"[parse_scenario] 幸台词数：{stats['protagonist_lines']}")
    return stats


if __name__ == "__main__":
    main()
