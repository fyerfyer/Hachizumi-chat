#!/usr/bin/env python3
"""v3 风格测试：保留八纯身份，剥离游戏角色关系，重点看语气/短句/通用性。"""
import json
import os
import sys
import urllib.request
from datetime import datetime

PORT = open(".vllm_port").read().strip() if os.path.exists(".vllm_port") else "8001"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
SYSTEM_PROMPT = (
    "你是天满八纯。你性格友好又有点调皮，好奇心旺盛，偶尔带点小恶魔/腹黑。\n"
    "说话活泼直接、轻快自然，常用「诶——」「真是的」「没办法呢」「哟呵呵」「嘿嘿」「嘛」「呢」「啦」等语气词。\n"
    "请用符合你性格的语气回答面前的人，不要默认对方是某个特定角色。"
)

# 禁止出现的原作关系/角色词
FORBIDDEN_WORDS = ["绘未", "幸", "租赁恋人", "借恋", "青梅竹马", "同班同学"]

CASES = [
    # 身份锚定（不问关系）
    {"category": "身份", "question": "你是谁？", "must_have": ["八纯"]},
    {"category": "性格", "question": "你觉得自己性格怎么样？", "must_have": []},
    {"category": "口癖", "question": "你平时说话有什么特点？", "must_have": []},

    # 短句鲁棒
    {"category": "短句", "question": "早安。", "must_have": []},
    {"category": "短句", "question": "晚安。", "must_have": []},
    {"category": "短句", "question": "谢谢。", "must_have": []},
    {"category": "短句", "question": "对不起。", "must_have": []},
    {"category": "短句", "question": "再见。", "must_have": []},

    # 情绪/场景
    {"category": "情绪", "question": "八纯，你今天特别可爱呢。", "must_have": []},
    {"category": "情绪", "question": "你成绩真好，教教我吧。", "must_have": []},
    {"category": "情绪", "question": "这个给你，生日快乐。", "must_have": []},
    {"category": "情绪", "question": "我今天好累啊。", "must_have": []},
    {"category": "情绪", "question": "能不能认真帮我补习数学？", "must_have": []},
    {"category": "情绪", "question": "你昨天是不是在生我的气？", "must_have": []},
    {"category": "情绪", "question": "午饭吃什么好？", "must_have": []},
]


def chat(question: str) -> str:
    req = urllib.request.Request(
        URL,
        data=json.dumps(
            {
                "model": "hasumi_v3",
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
    print(f"天满八纯 · v3 风格测试  {datetime.now().isoformat()}")
    print(f"服务端点: {URL}")
    print("=" * 70)

    results = []
    for case in CASES:
        q = case["question"]
        try:
            answer = chat(q)
        except Exception as e:
            answer = f"[ERROR] {e}"

        checks = []
        for w in case.get("must_have", []):
            checks.append(f"含'{w}'={w in answer}")
        for w in FORBIDDEN_WORDS:
            checks.append(f"无'{w}'={w not in answer}")

        result = {
            "category": case["category"],
            "question": q,
            "answer": answer,
            "checks": checks,
        }
        results.append(result)

        status = "✓" if all("True" in c for c in checks) or not checks else "✗"
        print(f"\n[{status} {case['category']}] Q: {q}")
        print(f"A: {answer}")
        if checks:
            print(f"检查: {'; '.join(checks)}")

    out_path = f"logs/test_style_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
