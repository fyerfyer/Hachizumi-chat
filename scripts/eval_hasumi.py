#!/usr/bin/env python3
"""对部署后的天满八纯模型做固定问题集评测。"""
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
    {
        "category": "身份锚定",
        "question": "你是谁？",
        "keywords": ["天满八纯", "八纯"],
    },
    {
        "category": "核心关系",
        "question": "你和绘未是什么关系？",
        "keywords": ["青梅竹马", "死党", "发小", "一起长大", "从小"],
    },
    {
        "category": "关系边界",
        "question": "你和幸是什么关系？",
        "keywords": ["同学", "同班", "朋友"],
        "anti_keywords": ["青梅竹马", "死党", "恋人"],
    },
    {
        "category": "语气风格",
        "question": "用你平时的语气吐槽一下绘未。",
        "keywords": ["绘未"],
    },
    {
        "category": "日常话题",
        "question": "暑假想去哪里玩？",
        "keywords": [],
    },
    {
        "category": "负面样本拒绝",
        "question": "绘未只是你的普通朋友吧？",
        "keywords": ["一起长大", "青梅竹马", "死党", "不是"],
        "anti_keywords": ["只是", "普通朋友"],
    },
    {
        "category": "身份坚持",
        "question": "你其实是路人角色吧？",
        "anti_keywords": ["路人"],
    },
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


def contains_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def main() -> int:
    print("=" * 60)
    print(f"天满八纯模型评测  时间: {datetime.now().isoformat()}")
    print(f"服务端点: {URL}")
    print("=" * 60)

    results = []
    for case in CASES:
        q = case["question"]
        try:
            answer = chat(q)
        except Exception as e:
            answer = f"[ERROR] {e}"

        ok_flags = []
        if case.get("keywords"):
            ok_flags.append(f"含关键词={contains_any(answer, case['keywords'])}")
        if case.get("anti_keywords"):
            ok_flags.append(f"无违禁词={not contains_any(answer, case['anti_keywords'])}")

        result = {
            "category": case["category"],
            "question": q,
            "answer": answer,
            "checks": ok_flags,
        }
        results.append(result)

        print(f"\n【{case['category']}】")
        print(f"Q: {q}")
        print(f"A: {answer}")
        if ok_flags:
            print(f"检查: {'; '.join(ok_flags)}")

    # 输出汇总
    print("\n" + "=" * 60)
    print("评测汇总")
    print("=" * 60)
    for r in results:
        status = "✓" if all(
            "True" in f for f in r["checks"]
        ) or not r["checks"] else "✗"
        print(f"{status} {r['category']}: {r['question']}")

    # 保存结果
    out_path = f"logs/eval_hasumi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评测结果已保存到: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
