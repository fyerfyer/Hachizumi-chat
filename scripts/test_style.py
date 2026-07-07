#!/usr/bin/env python3
"""重点评测天满八纯的语言风格是否自然、有角色感。"""
import json
import os
import sys
import urllib.request
from datetime import datetime

PORT = open(".vllm_port").read().strip() if os.path.exists(".vllm_port") else "8001"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# 这里刻意不塞太多身份/关系提示，重点看模型自己能不能保持八纯的语气。
# system prompt 保持和训练时一致即可。
SYSTEM_PROMPT = (
    "你是天满八纯，绘未的青梅竹马兼同班同学。"
    "你成绩优秀、性格友好又有点调皮，说话活泼直接，偶尔会小小地捉弄亲近的人。"
    "请用符合你性格的语气回答。"
)

# 大量覆盖日常高频场景，观察口癖、语调、情绪表达是否一致。
STYLE_CASES = [
    # 打招呼与寒暄
    "早上好呀！",
    "好久不见～",
    "今天你看起来心情不错嘛。",

    # 惊讶与感叹
    "真的假的！？",
    "诶——不会吧！",
    "这也太厉害了吧！",

    # 否认与傲娇
    "才不是呢！",
    "我哪有很在意她啊。",
    "谁、谁说我喜欢她啦！",

    # 调侃/捉弄亲近的人
    "你又迟到了哦。",
    "嘿嘿，被我抓到把柄了吧？",
    "你这副样子还真是少见呢。",

    # 开心与兴奋
    "太好了！",
    "太棒了，一起去庆祝吧！",

    # 抱怨与撒娇
    "真是的，你也太慢了。",
    "哼，不理你了。",
    "我肚子饿了啦～",

    # 安慰与关心
    "别难过了，有我在呢。",
    "没事吧？要不要我陪你？",

    # 优等生属性 + 日常吐槽
    "这次考试好简单。",
    "作业也太多了吧。",
    "要不要我借你笔记？",

    # 邀请与提议
    "一起去吃饭吧？",
    "放学后要一起回家吗？",

    # 陌生人/不熟的人
    "你是哪位？",
    "我不认识你哦。",

    # 突发场景
    "下雨了，怎么办？",
    "我摔倒了，好痛。",

    # 玩笑与吹牛
    "我可是很厉害的哦。",
    "信不信由你，我小时候还拿过奖呢。",
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
    print(f"天满八纯 · 语言风格专项测试  {datetime.now().isoformat()}")
    print(f"服务端点: {URL}")
    print("=" * 70)

    results = []
    for q in STYLE_CASES:
        try:
            answer = chat(q)
        except Exception as e:
            answer = f"[ERROR] {e}"

        results.append({"question": q, "answer": answer})
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    out_path = f"logs/test_style_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
