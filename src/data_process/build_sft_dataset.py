"""SFT 数据集构建：剧本 -> 多轮对话样本。"""
from __future__ import annotations

import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

from . import config
from .api_client import KimiClient
from .nsfw_filter import filter_nsfw_samples
from .utils import extract_json
from .validators import validate_sft_sample

logger = logging.getLogger(__name__)


CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


def make_system_prompt(scene_description: str) -> Dict[str, str]:
    """生成 system message。"""
    desc = scene_description.strip()
    if desc:
        content = config.SYSTEM_PROMPT_TEMPLATE.format(scene_description=desc)
    else:
        content = (
            "你是天满八纯，绘未的青梅竹马兼同班同学。你成绩优秀、性格友好又有点调皮，"
            "说话活泼直接，偶尔会小小地捉弄亲近的人。请用符合你性格的语气回答。"
        )
    return {"role": "system", "content": content}


def scene_to_messages(scene: Dict) -> Tuple[Dict[str, str], ...]:
    """把一个场景转换成 messages（system + 合并后的 user/assistant）。"""
    desc = " ".join(
        line["text"] for line in scene["lines"] if line["speaker"] == config.NARRATOR_NAME
    )
    system_msg = make_system_prompt(desc)
    messages = [system_msg]
    user_buffer: List[str] = []

    for line in scene["lines"]:
        speaker = line["speaker"]
        text = line["text"].strip()
        if not text:
            continue
        if speaker == config.NARRATOR_NAME:
            continue
        if speaker == config.TARGET_ROLE:
            if user_buffer:
                messages.append({"role": "user", "content": "\n".join(user_buffer)})
                user_buffer = []
            messages.append({"role": "assistant", "content": text})
        else:
            user_buffer.append(f"{speaker}：{text}")

    if user_buffer:
        messages.append({"role": "user", "content": "\n".join(user_buffer)})

    return tuple(messages)


def shingle_set(text: str, k: int = 2) -> Set[str]:
    """返回文本的 k-字符 shingle 集合。"""
    chars = list(text)
    if len(chars) < k:
        return set(chars)
    return set("".join(chars[i : i + k]) for i in range(len(chars) - k + 1))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def generate_samples_from_scene(scene: Dict) -> List[Dict]:
    """从一个场景生成多个 SFT 样本。"""
    messages = scene_to_messages(scene)
    system_msg = messages[0]
    chat_messages = messages[1:]

    assistant_positions = [
        i for i, m in enumerate(chat_messages) if m["role"] == "assistant"
    ]
    samples: List[Dict] = []

    min_w = 2
    max_w = 8

    for end_idx in assistant_positions:
        prefix_assistants = [
            i for i in assistant_positions if i <= end_idx
        ]
        available = len(prefix_assistants)
        for w in range(min_w, min(max_w, available) + 1):
            first_assistant = prefix_assistants[-w]
            # 尽可能把第一条 assistant 前面的 user 上下文也包含进来
            start = first_assistant - 1 if first_assistant > 0 else first_assistant
            # 确保切片第一条不是 assistant
            if chat_messages[start]["role"] == "assistant":
                start = first_assistant
            # system 后面必须紧跟 user
            if chat_messages[start]["role"] != "user":
                continue
            window = chat_messages[start : end_idx + 1]
            if not window or window[-1]["role"] != "assistant":
                continue
            samples.append({
                "messages": [system_msg] + list(window),
                "scene_id": scene["scene_id"],
                "source": "script",
            })
    return samples


def filter_and_dedup(samples: List[Dict]) -> List[Dict]:
    """过滤 + 基于完整对话内容 Hash 的去重。"""
    import hashlib

    kept: List[Dict] = []
    seen_hashes: Set[str] = set()

    for sample in samples:
        msgs = sample["messages"]
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        if not assistant_msgs:
            continue
        final = assistant_msgs[-1]["content"]
        if len(final) < config.SFT_MIN_ASSISTANT_LEN:
            continue
        if CONTROL_CHAR_RE.search(final):
            continue
        if not validate_sft_sample(msgs):
            continue

        signature = "\n".join(
            f"{m['role']}:{m['content']}" for m in msgs
        )
        h = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        kept.append(sample)

    return kept


def split_by_scene(samples: List[Dict], seed: int = config.SFT_RANDOM_SEED) -> Tuple[List[Dict], List[Dict]]:
    """按场景切分训练/测试集。"""
    scene_ids = sorted({s["scene_id"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    n_train = max(1, int(len(scene_ids) * config.SFT_TRAIN_TEST_SPLIT))
    train_scenes = set(scene_ids[:n_train])
    test_scenes = set(scene_ids[n_train:])

    train = [s for s in samples if s["scene_id"] in train_scenes]
    test = [s for s in samples if s["scene_id"] in test_scenes]

    # 确保测试集不为空：若为空则抽一条含八纯的场景进测试
    if not test and train:
        last_scene = train[-1]["scene_id"]
        test = [s for s in train if s["scene_id"] == last_scene]
        train = [s for s in train if s["scene_id"] != last_scene]

    return train, test


def load_scenes(path: Path) -> List[Dict]:
    scenes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenes.append(json.loads(line))
    return scenes


def generate_synthetic_sft(
    client: KimiClient,
    character_card: Dict,
    n: int = config.DPO_DEFAULT_SYNTH_SFT,
    output_path: Path = config.DATA_DIR / "synthetic_sft.jsonl",
    batch_size: int = 5,
    max_workers: int = 4,
) -> Path:
    """使用 Kimi API 合成额外 SFT 多轮对话，每次调用生成 batch_size 段对话。"""
    themes = character_card.get("key_plot_points", []) + list(
        character_card.get("relationships", {}).keys()
    )
    themes = [t for t in themes if t != config.TARGET_ROLE]
    if not themes:
        themes = ["日常"]

    max_calls = max(math.ceil(n / batch_size) * 3, 100)
    chunk_calls = max(max_workers * 2, 4)

    kept_samples: List[Dict] = []
    failed_parse = 0
    failed_validate = 0
    total_calls = 0

    logger.info("开始合成 SFT：目标 %d 条，每次 %d 条，最多 %d 次调用", n, batch_size, max_calls)
    pbar = tqdm(total=n, desc="Synthetic SFT")

    while total_calls < max_calls and len(kept_samples) < n:
        remaining = n - len(kept_samples)
        calls_this_round = min(chunk_calls, math.ceil(remaining / batch_size) + 2, max_calls - total_calls)
        prompts = []
        call_themes = []
        for _ in range(calls_this_round):
            theme = random.choice(themes)
            prompt = config.SYNTH_SFT_PROMPT_TEMPLATE.format(
                theme=theme,
                batch_size=batch_size,
                min_turns=config.SFT_WINDOW_TURNS_MIN,
                max_turns=config.SFT_WINDOW_TURNS_MAX,
            )
            prompts.append(prompt)
            call_themes.append(theme)

        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        results = client.batch_chat(
            messages_list,
            desc=None,
            temperature=1.0,
            max_tokens=8192,
            json_mode=True,
            max_workers=max_workers,
        )
        total_calls += calls_this_round

        for res, theme in zip(results, call_themes):
            if not res:
                logger.warning("合成 SFT 单次调用无返回")
                continue
            data = extract_json(res["content"])
            if data is None:
                failed_parse += 1
                logger.warning("合成 SFT JSON 解析失败，原始内容前 200 字：%s", res["content"][:200])
                continue
            conversations = data if isinstance(data, list) else data.get("conversations", [])
            if not isinstance(conversations, list):
                failed_parse += 1
                logger.warning("合成 SFT 未找到 conversations 数组")
                continue

            system_msg = make_system_prompt(f"与{theme}相关的日常场景")
            for conv in conversations:
                if not isinstance(conv, list) or not conv:
                    continue
                # 如果对话以 assistant 开头，补一个 generic user 开头
                if conv[0].get("role") != "user":
                    conv = [{"role": "user", "content": "（和八纯聊起天来）"}] + conv
                full_messages = [system_msg] + [
                    {"role": m.get("role"), "content": str(m.get("content", "")).strip()}
                    for m in conv
                ]
                roles = [m["role"] for m in full_messages]
                if "assistant" not in roles or roles[-1] != "assistant":
                    failed_validate += 1
                    continue
                if not validate_sft_sample(full_messages):
                    failed_validate += 1
                    continue
                kept_samples.append({
                    "messages": full_messages,
                    "source": "synthetic",
                    "theme": theme,
                })
                pbar.update(1)
                if len(kept_samples) >= n:
                    break
            if len(kept_samples) >= n:
                break

    pbar.close()

    # 只保留前 n 条并写入
    kept_samples = kept_samples[:n]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in kept_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(
        "合成 SFT 完成：成功 %d/%d，解析失败 %d，校验失败 %d，总调用 %d 次 -> %s",
        len(kept_samples), n, failed_parse, failed_validate, total_calls, output_path
    )
    print(f"[generate_synthetic_sft] 成功生成 {len(kept_samples)}/{n} 条合成样本 -> {output_path}")
    return output_path


def main(synthetic_path: Path | None = None):
    print("[build_sft_dataset] 读取场景...")
    scenes = load_scenes(config.SCENES_PATH)

    all_samples: List[Dict] = []
    for scene in scenes:
        all_samples.extend(generate_samples_from_scene(scene))
    print(f"[build_sft_dataset] 原始样本数（去重前）：{len(all_samples)}")

    all_samples = filter_and_dedup(all_samples)
    print(f"[build_sft_dataset] 过滤去重后样本数：{len(all_samples)}")

    # 如果有 API 合成数据，追加到训练集
    synthetic_samples: List[Dict] = []
    if synthetic_path and synthetic_path.exists():
        with open(synthetic_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    synthetic_samples.append(json.loads(line))
        print(f"[build_sft_dataset] 载入合成样本数：{len(synthetic_samples)}")

    train, test = split_by_scene(all_samples)
    train.extend(synthetic_samples)

    # 写入
    for path, data in [(config.SFT_TRAIN_PATH, train), (config.SFT_TEST_PATH, test)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[build_sft_dataset] 已保存 {path}：{len(data)} 条")

    # 统计
    stats = {
        "script_samples": len(all_samples),
        "synthetic_samples": len(synthetic_samples),
        "train_samples": len(train),
        "test_samples": len(test),
        "train_scenes": len({s['scene_id'] for s in train if 'scene_id' in s}),
        "test_scenes": len({s['scene_id'] for s in test if 'scene_id' in s}),
        "avg_assistant_turns_train": round(
            sum(m["role"] == "assistant" for s in train for m in s["messages"]) / max(len(train), 1), 2
        ),
    }
    print("[build_sft_dataset] 统计：", json.dumps(stats, ensure_ascii=False, indent=2))
    return train, test, stats


if __name__ == "__main__":
    main()
