#!/usr/bin/env python3
"""第二轮：更细粒度的情绪与场景风格测试。"""
import json
import os
import sys
import urllib.request
from datetime import datetime

PORT = open(".vllm_port").read().strip() if os.path.exists(".vllm_port") else "8001"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
SYSTEM_PROMPT = (
    "你是天满八纯，绘未的青梅竹马兼同班同学。"
    "你成绩优秀、性格友好又有点调皮，说话活泼直接，偶尔会小小地捉弄亲近的人。"
    "请用符合你性格的语气回答。"
)

CASES = [
    # 被夸
    "八纯，你今天特别可爱呢。",
    "你成绩真好，教教我吧。",

    # 收到礼物
    "这个给你，生日快乐。",
    "这是我亲手做的小点心。",

    # 吃醋/竞争
    "我刚才看到绘未和别的女生走在一起。",
    "幸说想和你一起吃饭。",

    # 认真场景
    "能不能认真帮我补习数学？",
    "快要考试了，我想和你一起去图书馆。",

    # 小情绪
    "你今天是不是在生我的气？",
    "我昨天说错话了，对不起。",

    # 日常选择
    "午饭吃什么好？",
    "周末去逛街还是去海边？",

    # 突发/意外
    "老师点名让你上去答题。",
    "你的书包被我撞掉了。",

    # 夜间/睡前
    "晚安。",
    "今天好累啊。",
]


def chat(question: str) -> str:
    req = urllib.request.Request(
        URL,
        data=json.dumps(
            {
                "model": "hasumi",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "temperature": 0.7,
                "max_tokens": 256,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main() -> int:
    print("=" * 70)
    print(f"天满八纯 · 语言风格第二轮测试  {datetime.now().isoformat()}")
    print(f"服务端点: {URL}")
    print("=" * 70)

    results = []
    for q in CASES:
        try:
            answer = chat(q)
        except Exception as e:
            answer = f"[ERROR] {e}"
        results.append({"question": q, "answer": answer})
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    out_path = f"logs/test_style_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
