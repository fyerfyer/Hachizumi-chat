# 天满八纯角色模型

基于《恋爱，我借走了》（借恋）剧本，微调 Qwen2.5-7B-Instruct，使其能够扮演天满八纯的角色：语气活泼、友好、带点调皮与小恶魔/腹黑。

**当前实现重点**：保留八纯的身份与语言风格，剥离具体的游戏角色关系（绘未、幸、租赁恋人等）和剧情设定，让模型更专注于「语气性格」而非「事实关系」。

项目覆盖完整流程：剧本解析/清洗 → 风格增强 → SFT 训练 → DPO 风格偏好对齐 → 模型合并与 GGUF 导出 → Ollama / vLLM 部署测试。

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
│   │   ├── style_clean.py           # 剥离游戏角色关系/剧情
│   │   ├── style_enhance.py         # 风格增强与 DPO 生成
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
│   ├── export_gguf.sh
│   ├── test_style.py
│   ├── test_style_v2.py
│   └── test_style_v3.py             # 风格评测脚本
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

# 5. 风格清洗：剥离游戏角色关系与剧情（当前实现）
uv run python -m src.data_process.style_clean

# 6. 风格增强：补充短句/身份锚定样本与风格 DPO 对
uv run python -m src.data_process.style_enhance

# 7. 校验产物
uv run python -m src.data_process.validate_outputs
```

### 产物规模（参考）

| 产物 | 仅剧本 | 完整（+API） | 风格聚焦（当前） |
|------|--------|--------------|------------------|
| `scenes.jsonl` | ~491 场景 | ~491 场景 | ~491 场景 |
| `sft_train.jsonl` | ~12,000 条 | ~14,000 条 | ~5,600 条 |
| `sft_test.jsonl` | ~1,500 条 | ~1,500 条 | ~1,500 条 |
| `dpo_train.jsonl` | ~51 条 | ~1,500 条 | ~987 条 |

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

产物：`outputs/sft_qwen25_7b_hasumi_lora/`（LoRA adapter）；当前风格聚焦版输出到 `outputs/sft_qwen25_7b_hasumi_lora_v3/`。

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

产物：`outputs/dpo_qwen25_7b_hasumi_lora/`（DPO 后的 LoRA adapter）；当前实现输出到 `outputs/dpo_qwen25_7b_hasumi_lora_v3/`。

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

- `outputs/hasumi_qwen25_7b_merged/`：合并后的 bfloat16 完整模型；当前实现输出到 `outputs/hasumi_qwen25_7b_merged_v3/`。
- `outputs/hasumi_qwen25_7b_gguf/*.gguf`：Q4_K_M / Q5_K_M / Q8_0 量化 GGUF；当前实现输出到 `outputs/hasumi_qwen25_7b_merged_v3_gguf/*.gguf`。

## 部署与推理

### Ollama 部署

项目根目录已提供 `Modelfile`，指向当前 GGUF 输出：

```bash
ollama create hasumi-qwen25-7b -f Modelfile
ollama run hasumi-qwen25-7b
```

注意：`Modelfile` 中的 `FROM` 路径和 `SYSTEM` 提示词已随当前实现更新为去关系化版本。

### vLLM 推理测试

使用合并后的 bfloat16 模型：

```bash
# 启动服务
.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model outputs/hasumi_qwen25_7b_merged_v3 \
  --served-model-name hasumi_v3 \
  --dtype bfloat16 \
  --max_model_len 4096 \
  --gpu-memory-utilization 0.85

# 另开终端调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hasumi_v3",
    "messages": [
      {"role": "system", "content": "你是天满八纯。你性格友好又有点调皮，好奇心旺盛，偶尔带点小恶魔/腹黑。说话活泼直接、轻快自然，常用「诶——」「真是的」「没办法呢」「哟呵呵」「嘿嘿」「嘛」「呢」「啦」等语气词。请用符合你性格的语气回答面前的人，不要默认对方是某个特定角色。"},
      {"role": "user", "content": "你是谁？"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## 语气调优参考

当前实现已经完成了 `self-files/STYLE_IMPROVEMENT_PLAN.md` 和 `self-files/STYLE_TUNING_PLAN.md` 中的关键优化方向：

- 使用 `src/data_process/style_clean.py` 清洗剧本数据，剥离游戏角色关系与剧情词。
- 使用 `src/data_process/style_enhance.py` 生成去关系化的身份锚定、短句鲁棒、负面样本与风格 DPO 偏好对。
- System prompt 去关系化，聚焦语气性格而非事实关系。

评测脚本 `scripts/test_style_v3.py` 可用于验证短句稳定性、风格一致性和关系词残留情况。

## 日志与调试

- 数据流水线日志：`logs/pipeline.log`
- SFT 训练日志：`logs/sft_train.log`
- DPO 训练日志：`logs/dpo_train.log`
- GGUF 导出日志：`logs/export_gguf.log`
- 风格评测结果：`logs/test_style_v3_*.json`

## 注意事项

1. **API Key 安全**：`.env`、`.gitignore` 已排除敏感文件与产物目录，请勿将真实 key 提交到 Git。
2. **GPU 调配**：训练脚本会通过 `src/training/gpu_utils.py` 自动选择空闲 GPU；多卡环境下默认单卡运行，避免抢占其他任务资源。
3. **版权**：原始剧本与合成数据仅供个人学习本地使用，请勿公开分发。
4. **模型温度**：`kimi-for-coding` 只支持 `temperature=1`，`llm_client.py` 会自动调整；其他模型按需求设置即可。
