# 天满八纯角色模型

基于《恋爱，我借走了》（借恋）剧本，微调 Qwen2.5-7B-Instruct，使其能够扮演天满八纯的角色：语气活泼、友好、带点调皮，与青梅竹马绘未、同班同学幸等人物保持自然互动。

项目覆盖完整流程：剧本解析 → 角色卡构建 → SFT 训练 → DPO 偏好对齐 → 模型合并与 GGUF 导出 → Ollama / vLLM 部署测试。

## 项目结构

```
test-train/
├── dataset/                         # 原始剧本数据
│   └── 0.txt.gz                     # 游戏剧本（已繁转简）
├── data/                            # 产物目录（已 gitignore）
│   ├── raw/0.txt.gz -> ../../dataset/0.txt.gz
│   ├── processed/
│   │   ├── scenes.jsonl             # 解析后的场景
│   │   └── stats.json               # 解析统计
│   ├── character_card_hasumi.json   # 结构化角色卡
│   ├── system_prompt.txt            # system prompt 模板
│   ├── synthetic_sft.jsonl          # API 合成 SFT 样本
│   ├── sft_train.jsonl              # SFT 训练集
│   ├── sft_test.jsonl               # SFT 测试集
│   └── dpo_train.jsonl              # DPO 偏好对
├── src/
│   ├── data_process/                # 数据构建代码
│   │   ├── parse_scenario.py
│   │   ├── character_card_builder.py
│   │   ├── build_sft_dataset.py
│   │   ├── build_dpo_dataset.py
│   │   ├── llm_client.py            # 通用 LLM API 客户端
│   │   ├── validators.py
│   │   ├── utils.py
│   │   └── run_all.py
│   └── training/                    # 训练代码
│       ├── sft_unsloth.py           # Stage 1: SFT QLoRA
│       ├── dpo_unsloth.py           # Stage 2: DPO
│       ├── merge_and_export.py      # 合并 LoRA + 导出 GGUF
│       ├── data_utils.py
│       └── gpu_utils.py
├── scripts/                         # 一键脚本
│   ├── run_data_pipeline.sh
│   ├── run_data_pipeline_skip_api.sh
│   ├── run_sft.sh
│   ├── run_dpo.sh
│   └── export_gguf.sh
├── hfd.sh                           # HuggingFace 下载脚本（带镜像）
├── Modelfile                        # Ollama 部署配置
├── self-files/
│   ├── DESIGN.md                    # 原始设计文档
│   └── STYLE_TUNING_PLAN.md         # 语气调优调研方案
├── pyproject.toml
└── README.md
```

## 环境准备

使用 `uv` 管理依赖：

```bash
uv sync
source .venv/bin/activate
```

下载基座模型（使用 HuggingFace 镜像）：

```bash
./hfd.sh
```

该脚本会下载：

- `models/Qwen2.5-7B-Instruct/`：16-bit 完整模型，用于合并与导出 GGUF。
- `models/Qwen2.5-7B-Instruct-bnb-4bit/`：4-bit BNB 量化模型，用于 SFT/DPO 训练。

配置 LLM API Key（数据合成需要）：

```bash
cp .env.example .env
# 编辑 .env，填入真实 key
```

默认使用 DeepSeek：

```text
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

也兼容任意 OpenAI-compatible 服务（Kimi、OpenAI 等），通过环境变量覆盖即可。

## 数据层

### 仅使用原剧本（快速验证，不消耗 API）

```bash
./scripts/run_data_pipeline_skip_api.sh
```

### 完整数据层（剧本 + API 合成 SFT + DPO）

```bash
./scripts/run_data_pipeline.sh
```

默认参数：合成 2000 条 SFT、1500 条 DPO、并发 4。可通过环境变量调整：

```bash
SYNTH_SFT=500 DPO_PAIRS=500 MAX_WORKERS=4 ./scripts/run_data_pipeline.sh
```

### 分步运行

```bash
# 1. 解析剧本
uv run python -m src.data_process.parse_scenario

# 2. 构建角色卡
uv run python -m src.data_process.character_card_builder

# 3. 构建 SFT 数据集
uv run python -m src.data_process.build_sft_dataset

# 4. 构建 DPO 数据集
uv run python -m src.data_process.build_dpo_dataset

# 5. 校验产物
uv run python -m src.data_process.validate_outputs
```

### 产物规模（参考）

| 产物 | 仅剧本 | 完整（+API） |
|------|--------|--------------|
| `scenes.jsonl` | ~491 场景 | ~491 场景 |
| `sft_train.jsonl` | ~12,000 条 | ~14,000 条 |
| `sft_test.jsonl` | ~1,500 条 | ~1,500 条 |
| `dpo_train.jsonl` | ~51 条 | ~1,500 条 |

## 训练层

### Stage 1: SFT（Unsloth QLoRA）

```bash
./scripts/run_sft.sh
```

等价命令：

```bash
uv run python -m src.training.sft_unsloth \
  --model_name models/Qwen2.5-7B-Instruct-bnb-4bit \
  --data_path data/sft_train.jsonl \
  --output_dir outputs/sft_qwen25_7b_hasumi_lora \
  --num_train_epochs 2 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 2048
```

产物：`outputs/sft_qwen25_7b_hasumi_lora/`（LoRA adapter）。

### Stage 2: DPO（Unsloth DPO）

```bash
./scripts/run_dpo.sh
```

等价命令：

```bash
uv run python -m src.training.dpo_unsloth \
  --sft_adapter_path outputs/sft_qwen25_7b_hasumi_lora \
  --base_model_name models/Qwen2.5-7B-Instruct-bnb-4bit \
  --dpo_data_path data/dpo_train.jsonl \
  --output_dir outputs/dpo_qwen25_7b_hasumi_lora \
  --beta 0.1 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --max_seq_length 2048 \
  --max_prompt_length 1024
```

产物：`outputs/dpo_qwen25_7b_hasumi_lora/`（DPO 后的 LoRA adapter）。

### 合并与 GGUF 导出

```bash
./scripts/export_gguf.sh
```

等价命令：

```bash
uv run python -m src.training.merge_and_export \
  --adapter_path outputs/dpo_qwen25_7b_hasumi_lora \
  --base_model_full_path models/Qwen2.5-7B-Instruct \
  --merged_output_dir outputs/hasumi_qwen25_7b_merged \
  --gguf_output_dir outputs/hasumi_qwen25_7b_gguf \
  --quant_methods q4_k_m,q5_k_m,q8_0
```

产物：

- `outputs/hasumi_qwen25_7b_merged/`：合并后的 bfloat16 完整模型。
- `outputs/hasumi_qwen25_7b_gguf/*.gguf`：Q4_K_M / Q5_K_M / Q8_0 量化 GGUF。

## 部署与推理

### Ollama 部署

项目根目录已提供 `Modelfile`，指向 `outputs/hasumi_qwen25_7b_gguf/hasumi_qwen25_7b_merged.Q4_K_M.gguf`：

```bash
ollama create hasumi-qwen25-7b -f Modelfile
ollama run hasumi-qwen25-7b
```

### vLLM 推理测试

使用合并后的 bfloat16 模型：

```bash
# 启动服务
uv run python -m vllm.entrypoints.openai.api_server \
  --model outputs/hasumi_qwen25_7b_merged \
  --served-model-name hasumi \
  --dtype bfloat16 \
  --max_model_len 4096 \
  --gpu-memory-utilization 0.85

# 另开终端调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hasumi",
    "messages": [
      {"role": "system", "content": "你是天满八纯，绘未的青梅竹马兼同班同学。你成绩优秀、性格友好又有点调皮，说话活泼直接，偶尔会小小地捉弄亲近的人。"},
      {"role": "user", "content": "你是谁？"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## 语气调优参考

实际测试发现，模型已能使用 `哟呵呵`、`诶——`、`真是的`、`嘛`、`呢` 等口癖，语气整体活泼。若希望进一步强化角色语气风格，可参考 `self-files/STYLE_TUNING_PLAN.md` 中的调研方案，使用 DeepSeek API 做数据增强：清洗 user 前缀、风格迁移、生成自我介绍锚定样本、构建 DPO 语气偏好对等。

## 日志与调试

- 数据流水线日志：`logs/pipeline.log`
- SFT 训练日志：`logs/sft_train.log`
- DPO 训练日志：`logs/dpo_train.log`
- GGUF 导出日志：`logs/export_gguf.log`

## 注意事项

1. **API Key 安全**：`.env`、`.gitignore` 已排除敏感文件与产物目录，请勿将真实 key 提交到 Git。
2. **GPU 调配**：训练脚本会通过 `src/training/gpu_utils.py` 自动选择空闲 GPU；多卡环境下默认单卡运行，避免抢占其他任务资源。
3. **版权**：原始剧本与合成数据仅供个人学习本地使用，请勿公开分发。
4. **模型温度**：`kimi-for-coding` 只支持 `temperature=1`，`llm_client.py` 会自动调整；其他模型按需求设置即可。
