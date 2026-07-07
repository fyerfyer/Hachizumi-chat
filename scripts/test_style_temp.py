#!/usr/bin/env python3
"""测试不同 temperature 对语气风格稳定性的影响。"""
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

TEMPS = [0.3, 0.7, 0.9]
QUESTIONS = [
    "晚安。",
    "今天好累啊。",
    "早上好呀！",
    "能不能认真帮我补习数学？",
]


def chat(question: str, temp: float) -> str:
    req = urllib.request.Request(
        URL,
        data=json.dumps(
            {
                "model": "hasumi",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "temperature": temp,
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
    print(f"温度敏感性测试  {datetime.now().isoformat()}")
    print("=" * 70)

    results = []
    for temp in TEMPS:
        print(f"\n\n### temperature = {temp} ###")
        for q in QUESTIONS:
            answer = chat(q, temp)
            results.append({"temperature": temp, "question": q, "answer": answer})
            print(f"\nQ: {q}")
            print(f"A: {answer}")

    out_path = f"logs/test_style_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\n结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
