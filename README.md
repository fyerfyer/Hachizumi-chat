# 天满八纯角色模型 — 数据构建层

基于《恋爱，我借走了》（借恋）剧本，构建天满八纯角色扮演 LLM 的数据层。

## 项目结构

```
test-train/
├── dataset/                         # 原始数据
│   ├── 0.txt.gz                     # 游戏剧本（已繁转简）
│   └── 恋爱我借走了_ks.py           # 原始 KS 解析脚本
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
├── src/data_process/                # 数据构建代码
│   ├── parse_scenario.py
│   ├── character_card_builder.py
│   ├── build_sft_dataset.py
│   ├── build_dpo_dataset.py
│   ├── api_client.py
│   ├── validators.py
│   ├── utils.py
│   └── run_all.py
├── pyproject.toml
└── README.md
```

## 环境准备

使用 `uv` 管理环境（已安装）：

```bash
uv sync
source .venv/bin/activate
```

设置 API Key（仅运行期使用，不要写入代码）：

```bash
# 默认使用 Kimi Code API
export KIMI_API_KEY=sk-...

# 也可切换为 Deepseek 等 OpenAI-compatible 服务
export KIMI_API_KEY=sk-...
export KIMI_BASE_URL=https://api.deepseek.com/v1
export KIMI_MODEL=deepseek-chat
```

> 注意：`src/data_process/config.py` 中默认 Base URL 与模型对应 Kimi Code（`https://api.kimi.com/coding/v1`，`kimi-for-coding`）。使用其他服务时通过环境变量覆盖即可。

## 一键运行

### 仅使用原剧本（快速验证，不消耗 API）

```bash
uv run python -m src.data_process.run_all --skip-api
# 或
./scripts/run_data_pipeline_skip_api.sh
```

### 完整数据层（剧本 + API 合成 SFT + DPO）

```bash
export KIMI_API_KEY=sk-...
export KIMI_BASE_URL=https://api.deepseek.com/v1
export KIMI_MODEL=deepseek-chat

uv run python -m src.data_process.run_all \
  --synthetic-sft 2000 \
  --dpo-pairs 1500 \
  --max-workers 4

# 或
./scripts/run_data_pipeline.sh
```

参数说明：

- `--synthetic-sft N`：通过 API 合成 N 条多轮 SFT 样本。
- `--dpo-pairs N`：生成 N 条 DPO 偏好对（实际会多请求约 50% 并过滤）。
- `--max-workers N`：API 并发数，默认 4。
- `--skip-api`：跳过 API 调用，只用原剧本生成 SFT，并用规则生成少量 DPO。

## 分步运行

```bash
# 1. 解析剧本
uv run python -m src.data_process.parse_scenario

# 2. 构建角色卡
uv run python -m src.data_process.character_card_builder

# 3. 构建 SFT 数据集
uv run python -m src.data_process.build_sft_dataset

# 4. 构建 DPO 数据集
uv run python -m src.data_process.build_dpo_dataset
```

## 产物规模（参考）

| 产物 | 仅剧本 | 完整（+API） |
|------|--------|--------------|
| scenes.jsonl | ~491 场景 | ~491 场景 |
| sft_train.jsonl | ~12,000 条 | ~14,000 条 |
| sft_test.jsonl | ~1,500 条 | ~1,500 条 |
| dpo_train.jsonl | ~51 条 | ~1,500 条 |

## 产物校验

```bash
uv run python -m src.data_process.validate_outputs
```

会检查：场景/角色卡是否存在、SFT 格式与场景是否泄漏、DPO 格式与类型分布等。

## 调试日志

运行日志会写入 `logs/pipeline.log`，API 调用 token 消耗、失败原因、校验失败样本等都会记录，便于排查。

## 注意事项

1. **API Key 安全**：只通过环境变量读取，不写入代码或提交 Git；`.gitignore` 已排除 `.env`、`data/`、`logs/`。
2. **版权**：原始剧本与合成数据仅供个人学习本地使用，请勿公开分发。
3. **模型温度**：Kimi Code 的 `kimi-for-coding` 只支持 `temperature=1`，脚本会自动调整；Deepseek 等模型可按需设置。
4. **全量生成耗时**：使用 Deepseek + 并发 4 时，2000 条合成 SFT + 1500 条 DPO 约 30~60 分钟，具体取决于网络与模型响应速度。

## 后续步骤

数据层完成后，可继续实现训练层：

- SFT：`src/training/sft_unsloth.py`
- DPO：`src/training/dpo_unsloth.py`
- 导出与部署：`src/training/merge_and_export.py` + `Modelfile`
