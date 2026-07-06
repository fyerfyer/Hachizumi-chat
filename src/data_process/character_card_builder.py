"""角色卡构建：生成 character_card_hasumi.json 与 system_prompt.txt。"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import jieba

from . import config

# 允许保留的语气词 / 短表达
INTERJECTIONS = {
    "诶", "诶—", "诶——", "啊", "呀", "哦", "嘛", "呜", "哇", "呜哇",
    "诶嘿", "咕", "咕呶", "咕呶呶", "哼", "嘿嘿", "呵呵", "哈哈",
    "真是", "真是的", "没办法", "没办法呢", "没办法吧", "哟", "哟呵呵",
}

# 简单停用词表，过滤无意义高频词
STOPWORDS = {
    "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "的", "了", "在", "是", "有", "和", "与", "就", "都", "也", "而", "及",
    "吗", "呢", "吧", "啊", "哦", "嘛", "呀", "哈", "嘿", "哼",
    "不", "没", "没有", "会", "能", "要", "想", "做", "去", "来", "上", "下",
    "这", "那", "这个", "那个", "里", "个", "什么", "怎么", "为什么",
    "虽然", "但是", "因为", "所以", "如果", "不过", "而且", "或者", "然后",
    "已经", "正在", "现在", "今天", "这次", "这样", "那样", "这里", "那里",
    "自己", "还是", "就是", "真是", "可能", "需要", "开始", "结束", "已经",
    "一直", "一下", "一些", "一点", "一会儿", "一次", "一天",
    "的话", "不是", "不会", "不能", "不要", "不想", "没有", "知道", "觉得",
    "那么", "这么", "怎样", "怎么办", "就是", "也是", "还有", "而是", "地说",
}


def load_hasumi_lines(scenes_path: Path) -> List[str]:
    """读取八纯台词。"""
    lines: List[str] = []
    with open(scenes_path, "r", encoding="utf-8") as f:
        for line in f:
            scene = json.loads(line)
            for item in scene["lines"]:
                if item["speaker"] == config.TARGET_ROLE:
                    lines.append(item["text"])
    return lines


def extract_common_phrases(hasumi_lines: List[str], top_k: int = 12) -> List[str]:
    """从八纯台词中提取高频口头禅 / 短表达。"""
    # 先注入一些已知词，避免被 jieba 错误切分
    for w in INTERJECTIONS:
        jieba.add_word(w, freq=1000)

    counter: Counter = Counter()
    for text in hasumi_lines:
        for tok in jieba.lcut(text):
            tok = tok.strip()
            if not tok:
                continue
            # 保留明确语气词
            if tok in INTERJECTIONS:
                counter[tok] += 1
                continue
            # 过滤停用词、纯标点、数字
            if tok in STOPWORDS:
                continue
            if re.match(r"^[\W\d]+$", tok):
                continue
            if 2 <= len(tok) <= 4:
                counter[tok] += 1

    # 基础口头禅 + 数据驱动高频词
    base = ["诶——", "真是的", "没办法呢", "哟呵呵", "咕呶呶"]
    data_driven = [tok for tok, _ in counter.most_common(top_k * 2) if tok not in base]
    phrases = list(dict.fromkeys(base + data_driven))[:top_k]
    return phrases


def build_character_card(hasumi_lines: List[str]) -> Dict:
    """构建结构化角色卡。"""
    common_phrases = extract_common_phrases(hasumi_lines)

    card = {
        "name": "天满八纯",
        "alias": ["八纯", "てんま はすみ", "Tenma Hasumi", "天満八純"],
        "basic": {
            "gender": "女",
            "height": "159cm",
            "hair_color": "红发",
            "eye_color": "蓝瞳",
            "school": "私立木乃香坂学园",
            "class": "与主人公同班",
        },
        "personality": [
            "友好调皮",
            "好奇心旺盛",
            "待人接物让人舒服",
            "偶尔有点小恶魔/腹黑",
            "对亲近的人直率",
            "成绩优秀但不过分张扬",
        ],
        "relationships": {
            "绘未": "青梅竹马兼同班同学，关系亲密但经常互损",
            "幸": "同班同学，因「租赁恋人」委托而逐渐产生交集",
            "椿": "朋友，性格对比鲜明",
            "千夏": "学妹",
            "咲希": "同班同学",
            "桃子": "同班同学",
        },
        "speech_patterns": {
            "common_phrases": common_phrases,
            "tone": "活泼、轻快、偶尔带捉弄",
            "honorifics": "对亲近朋友不用敬语，对生人或开玩笑时会更俏皮",
            "speed": "语速偏快，情绪上来时会连续吐槽",
        },
        "forbidden": [
            "不要把绘未说成陌生人或普通同学",
            "不要把幸直接说成男朋友（应是租赁恋人/同班同学关系）",
            "不要使用现代网络流行语",
            "不要编造未在游戏中发生的剧情",
            "不要复述受版权保护的原剧本长段落",
        ],
        "key_plot_points": [
            "泳池",
            "海边",
            "租赁恋人",
            "暑假",
            "青梅竹马",
            "绘未家的便当",
            "备考",
        ],
        "source": "基于《恋爱，我借走了》剧本与设计文档整理",
    }
    return card


def main():
    print("[character_card_builder] 读取剧本场景...")
    hasumi_lines = load_hasumi_lines(config.SCENES_PATH)
    print(f"[character_card_builder] 八纯台词数：{len(hasumi_lines)}")

    card = build_character_card(hasumi_lines)

    config.CHARACTER_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHARACTER_CARD_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"[character_card_builder] 已保存 {config.CHARACTER_CARD_PATH}")

    # 生成 system prompt 模板（保留 {scene_description} 占位符）
    system_prompt = config.SYSTEM_PROMPT_TEMPLATE
    with open(config.SYSTEM_PROMPT_PATH, "w", encoding="utf-8") as f:
        f.write(system_prompt)
    print(f"[character_card_builder] 已保存 {config.SYSTEM_PROMPT_PATH}")

    print(f"[character_card_builder] 提取口头禅：{card['speech_patterns']['common_phrases']}")
    return card


if __name__ == "__main__":
    main()
