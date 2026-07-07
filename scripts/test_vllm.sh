#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

PORT=$(cat .vllm_port 2>/dev/null || echo "8000")
URL="http://127.0.0.1:$PORT/v1/chat/completions"
SYSTEM_PROMPT="你是天满八纯，绘未的青梅竹马兼同班同学。你成绩优秀、性格友好又有点调皮，说话活泼直接，偶尔会小小地捉弄亲近的人。请用符合你性格的语气回答。"

QUESTIONS=(
    "你是谁？"
    "你和绘未是什么关系？"
    "你和幸是什么关系？"
    "用你平时的语气吐槽一下绘未。"
    "暑假想去哪里玩？"
    "绘未只是你的普通朋友吧？"
)

echo "================================"
echo "vLLM 测试：端口 $PORT"
echo "================================"

for q in "${QUESTIONS[@]}"; do
    echo ""
    echo "Q: $q"
    RESPONSE=$(curl -s "$URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"hasumi\",
            \"messages\": [
                {\"role\": \"system\", \"content\": \"$SYSTEM_PROMPT\"},
                {\"role\": \"user\", \"content\": \"$q\"}
            ],
            \"temperature\": 0.7,
            \"max_tokens\": 256
        }" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
    echo "A: $RESPONSE"
done
