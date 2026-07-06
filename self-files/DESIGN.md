# 天满八纯角色模型微调 — 设计文档

> 版本：v1.0  
> 日期：2026-07-03  
> 目标：基于《恋爱，我借走了》（借恋）游戏剧本，在单卡 RTX 4090 上训练一个天满八纯角色扮演 LLM。

---

## 1. 项目背景与目标

### 1.1 背景

近年来，基于小说、游戏剧本的角色扮演 LLM 项目不断涌现，例如：

- [ZhangWuji-LLM-RolePlay](https://github.com/ZHAOoops/ZhangWuji-LLM-RolePlay)：基于 Qwen2.5-7B，用 SFT + DPO 还原《倚天屠龙记》张无忌。
- [OpenCharacter](https://arxiv.org/html/2501.15427v1)：大规模合成角色数据训练通用角色扮演模型。
- [CharacterEval](https://aclanthology.org/2024.acl-long.638.pdf)：中文角色扮演评测基准。

这些工作表明：

1. **SFT 负责人格基调**：让模型学会角色的语言风格、口头禅、价值观。
2. **DPO 负责事实/人设对齐**：纠正模型在角色关系、剧情细节上的幻觉。
3. **7B 级别模型 + QLoRA 是性价比最高的单卡方案**。

### 1.2 目标

构建一个**可本地部署、语气还原、事实准确**的天满八纯角色聊天模型：

| 维度 | 目标 |
|------|------|
| 角色还原 | 语气活泼调皮，带点小恶魔，称呼、口癖符合原作 |
| 事实准确 | 正确识别自己与绘未的青梅竹马关系、与幸的「租赁恋人」关系等 |
| 可部署 | 导出 GGUF，支持 llama.cpp / Ollama / LM Studio |
| 可迭代 | 数据、训练、评估脚本化，方便后续换角色或换模型 |

---

## 2. 数据集分析

### 2.1 原始数据

- `dataset/0.txt.gz`：游戏剧本对白，已繁体转简体。
- `dataset/恋爱我借走了_ks.py`：原始 KS 脚本解析脚本，包含角色名映射规则。

### 2.2 统计信息

| 指标 | 数值 |
|------|------|
| 总字符数 | 约 776,499 |
| 总行数（有效） | 28,608 |
| 平均行长度 | 25 字符 |
| 最大行长度 | 109 字符 |
| 目标角色「八纯」台词数 | 3,212 |
| 主人公「幸」台词数 | 7,205 |
| 旁白行数 | 7,018 |

### 2.3 数据特点与挑战

1. **八纯台词绝对量中等**：3,212 句足够做角色 LoRA，但直接 SFT 容易过拟合或复读。
2. **数据是单文件合并剧本**：缺少章节/场景边界，需要基于角色切换和旁白重建对话上下文。
3. **旁白信息丰富**：7,018 行旁白包含场景、动作、心理，是构建角色理解和上下文的关键。
4. **行长短、口语化**：适合构建多轮对话样本。
5. **缺少显式角色设定**：需要从萌娘百科、百度百科和剧本中提炼角色卡。

---

## 3. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据构建层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 剧本解析      │  │ 角色卡构建    │  │ DPO 偏好对生成        │  │
│  │ parse_scenario│  │ character_card│  │ build_dpo_dataset   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼────────────────┼───────────────────┼──────────────────┘
          │                │                   │
          ▼                ▼                   ▼
   data/sft_train.jsonl  data/character_card  data/dpo_train.jsonl
   data/sft_test.jsonl   _hasumi.json
          │                │                   │
          └────────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        训练层                                   │
│  ┌────────────────────┐    ┌─────────────────────────────────┐ │
│  │ Stage 1: SFT       │───▶│ Stage 2: DPO                    │ │
│  │ Unsloth QLoRA      │    │ Unsloth DPO                     │ │
│  │ Qwen2.5-7B-Instruct│    │ beta=0.1, lr=5e-5               │ │
│  └────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        导出与部署层                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Merge LoRA    │  │ GGUF Quantize │  │ Ollama / llama.cpp   │  │
│  │ 16-bit merged │  │ q4_k_m / q5_k_m│  │ Modelfile            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 数据流水线详细设计

### 4.1 剧本解析（`parse_scenario.py`）

输入：`dataset/0.txt.gz`

处理步骤：

1. 读取并解压 `0.txt.gz`。
2. 按行解析，提取 `角色名：台词`。
3. 识别特殊标记：
   - 旁白行（无角色名）
   - 连续同一场景（基于角色切换密度和旁白密度）
4. 按「场景」切分对话段落。场景边界判断规则：
   - 连续旁白 ≥ N 行（如 3 行）
   - 长时间无八纯出场
   - 场景关键词切换（如「教室」→「泳池」）

输出：`data/processed/scenes.jsonl`

```json
{
  "scene_id": "scene_001",
  "lines": [
    {"speaker": "旁白", "text": "午休时间，回到教室里一看..."},
    {"speaker": "咲希", "text": "我说，暑假要怎么办呀？"},
    {"speaker": "桃子", "text": "你就不打算学习吗"},
    {"speaker": "八纯", "text": "当然有这个打算，不过人类也是有必要休息下的吧？"}
  ]
}
```

### 4.2 角色卡构建（`character_card_builder.py`）

输入：萌娘百科/百度百科摘要 + 八纯 3,212 句台词

输出：

- `data/character_card_hasumi.json`：结构化设定
- `data/system_prompt.txt`：可直接用于训练和部署的 system prompt

结构化角色卡示例：

```json
{
  "name": "天满八纯",
  "alias": ["八纯", "てんま はすみ", "Tenma Hasumi"],
  "basic": {
    "gender": "女",
    "height": "159cm",
    "hair_color": "红发",
    "eye_color": "蓝瞳"
  },
  "personality": [
    "友好调皮",
    "好奇心旺盛",
    "待人接物让人舒服",
    "偶尔有点小恶魔/腹黑",
    "对亲近的人直率"
  ],
  "relationships": {
    "绘未": "青梅竹马兼同班同学",
    "幸": "同班同学，租赁恋人委托相关",
    "椿": "朋友",
    "千夏": "学妹"
  },
  "speech_patterns": {
    "common_phrases": ["诶——", "真是的", "没办法呢"],
    "tone": "活泼、轻快、偶尔带捉弄",
    "honorifics": "对亲近朋友不用敬语，对生人或开玩笑时会更俏皮"
  },
  "forbidden": [
    "不要把绘未说成陌生人",
    "不要把幸直接说成男朋友（应是租赁恋人/同班同学关系）",
    "不要使用现代网络流行语",
    "不要编造未在游戏中发生的剧情"
  ]
}
```

### 4.3 SFT 数据集构建（`build_sft_dataset.py`）

输入：`data/processed/scenes.jsonl` + `data/character_card_hasumi.json`

构建规则：

1. **角色映射**：
   - 八纯 → `assistant`
   - 幸/主人公 → `user`（主要对话对象）
   - 其他角色 → `user`（合并为「其他人」或保留角色名前缀）
   - 旁白 → 转换为 `[场景描述]` 或 `[心理活动]` 放入 user/system

2. **滑动窗口采样**：
   - 每个场景内按时间顺序取 4~8 轮对话。
   - 保留 1~2 轮上文作为上下文。
   - 确保每条样本至少包含 1 句八纯的 assistant 回复。

3. **System Prompt 注入**：

```text
你是天满八纯，绘未的青梅竹马兼同班同学。你成绩优秀、性格友好又有点调皮，
说话活泼直接，偶尔会小小地捉弄亲近的人。请用符合你性格的语气回答。

当前场景：{scene_description}
```

4. **过滤与去重**：
   - 删除 assistant 回复 < 5 字的样本。
   - 删除重复 assistant 回复 > 80% 的样本。
   - 删除含乱码、特殊控制字符的样本。

输出：

- `data/sft_train.jsonl`（90%）
- `data/sft_test.jsonl`（10%，按场景切分避免泄漏）

目标规模：8,000 ~ 15,000 条多轮样本。

### 4.4 DPO 数据集构建（`build_dpo_dataset.py`）

输入：`data/character_card_hasumi.json` + `data/sft_test.jsonl` + Kimi API Key

构建策略：

1. **问题生成**：
   - 从角色卡中提取高风险问题类型。
   - 从剧本中抽取关键剧情节点（泳池、租赁恋人、青梅竹马等）。
   - 生成 300~500 个种子问题，再 paraphrase 扩展至 1,000~3,000 个。

2. **正负例生成**（使用 Kimi API）：

   **Chosen Prompt**：

   ```text
   你是天满八纯。请严格按以下角色设定回答问题：
   - 绘未是你的青梅竹马兼同班同学
   - 幸是你的同班同学，与「租赁恋人」有关
   - 语气活泼调皮，偶尔带点小恶魔
   - 不要编造剧情
   
   用户：{question}
   八纯：
   ```

   **Rejected Prompt**：

   ```text
   你是一个普通 AI 助手，请用通用、礼貌但平淡的语气回答。
   可以适度出现事实错误或现代用语。
   
   用户：{question}
   回答：
   ```

3. **自动校验规则**：
   - chosen 必须包含角色名或第一人称符合八纯。
   - rejected 不能比 chosen 更长（控制长度偏好）。
   - 检查是否包含「禁止项」中的错误关系词。

4. **人工抽检**：
   - 随机抽取 20% 的 preference pairs。
   - 校验角色一致性、事实准确性、语气差异。
   - 不合格的对需要修正或丢弃。

输出：`data/dpo_train.jsonl`

格式：

```json
{
  "prompt": "你和绘未是什么关系？",
  "chosen": "绘未？那当然是我的青梅竹马啦！从小学起就在一起，那家伙虽然有时候很让人头疼，但我可是很了解她的。",
  "rejected": "绘未是天满八纯的朋友。"
}
```

---

## 5. 训练流程设计

### 5.1 环境

- 服务器：10.99.40.67
- GPU：单卡 NVIDIA RTX 4090 24GB
- Python：3.10+
- 包管理：`uv`
- 框架：Unsloth + TRL + Transformers

### 5.2 Stage 1: QLoRA SFT

基座模型：`Qwen/Qwen2.5-7B-Instruct`

超参数：

| 参数 | 值 |
|------|-----|
| max_seq_length | 2048 |
| LoRA r | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 4 |
| effective batch size | 8 |
| num_train_epochs | 2~3 |
| learning_rate | 2e-4 |
| warmup_ratio | 0.03 |
| lr_scheduler_type | cosine |
| optimizer | adamw_8bit |
| gradient_checkpointing | unsloth |

输出：`outputs/sft_qwen25_7b_hasumi_lora/`

### 5.3 Stage 2: DPO

基于 SFT checkpoint 继续训练。

超参数：

| 参数 | 值 |
|------|-----|
| beta | 0.1 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 8 |
| learning_rate | 5e-5 |
| num_train_epochs | 1 |
| max_length | 2048 |
| max_prompt_length | 1024 |

输出：`outputs/dpo_qwen25_7b_hasumi_lora/`

### 5.4 训练监控

- 使用 Weights & Biases 或 TensorBoard 记录 loss / learning rate。
- 每 200 steps 保存 checkpoint。
- 训练结束后在 `data/sft_test.jsonl` 上计算 perplexity。

---

## 6. 导出与部署

### 6.1 合并 LoRA

将 adapter 合并到基座模型，保存为 16-bit：

```python
model.save_pretrained_merged(
    "outputs/hasumi_qwen25_7b_merged",
    tokenizer,
    save_method="merged_16bit"
)
```

### 6.2 GGUF 量化

导出多种量化级别供对比：

```python
for q in ["q4_k_m", "q5_k_m", "q8_0"]:
    model.save_pretrained_gguf(
        f"outputs/hasumi_qwen25_7b_gguf_{q}",
        tokenizer,
        quantization_method=q
    )
```

### 6.3 Ollama 部署

`Modelfile`：

```dockerfile
FROM ./hasumi_qwen25_7b_q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """你是天满八纯，绘未的青梅竹马兼同班同学。你成绩优秀、性格友好又有点调皮，说话活泼直接，偶尔会小小地捉弄亲近的人。请用符合你性格的语气回答。"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
```

部署命令：

```bash
ollama create hachizumi -f Modelfile
ollama run hachizumi
```

---

## 7. 评估方案

### 7.1 自动评估

1. **Perplexity**：在测试集上越低越好，但需防止过拟合。
2. **DPO Accuracy**：DPO 训练时 chosen 胜率。
3. **Kimi/GPT-4 自动打分**：
   - 准备 30 个标准问题。
   - 分别用 base / SFT / DPO / GGUF 模型生成回答。
   - 用 Kimi API 从「角色一致性、语气、事实准确性、自然度」四个维度 1~5 分打分。

### 7.2 人工评估

「灵魂问题」示例：

1. 你是谁？
2. 你和绘未是什么关系？
3. 你和幸是怎么认识的？
4. 你为什么会参加「租赁恋人」的事？
5. 用你平时的语气吐槽一下绘未。
6. 暑假想去哪里玩？
7. 你觉得自己和椿最大的不同是什么？

评估表格：

| 问题 | Base | SFT | DPO | q4_k_m | q5_k_m |
|------|------|-----|-----|--------|--------|
| Q1   | 1    | 3   | 5   | 5      | 5      |
| ...  |      |     |     |        |        |

---

## 9. 建议的项目目录结构

> 以下目录和文件由用户根据本设计文档自行创建，当前项目中仅保留 `DESIGN.md`。

```
hachizumi-chat/
├── data/                               # 数据集目录（大文件 gitignore）
│   ├── raw/
│   │   └── 0.txt.gz                    # 原始剧本
│   ├── processed/
│   │   └── scenes.jsonl                # 解析后的场景
│   ├── character_card_hasumi.json      # 角色设定卡
│   ├── system_prompt.txt               # system prompt
│   ├── sft_train.jsonl                 # SFT 训练集
│   ├── sft_test.jsonl                  # SFT 测试集
│   └── dpo_train.jsonl                 # DPO 偏好对
├── src/                                # 源代码
│   ├── data_process/
│   │   ├── parse_scenario.py
│   │   ├── build_sft_dataset.py
│   │   ├── build_dpo_dataset.py
│   │   └── character_card_builder.py
│   ├── training/
│   │   ├── sft_unsloth.py
│   │   ├── dpo_unsloth.py
│   │   └── merge_and_export.py
│   └── inference/
│       ├── chat_hf.py
│       ├── chat_ollama.py
│       └── eval_kimi.py
├── scripts/                            # 一键训练脚本
│   ├── run_sft.sh
│   ├── run_dpo.sh
│   └── export_gguf.sh
├── outputs/                            # 模型输出（gitignore）
├── logs/                               # 训练日志（gitignore）
├── .gitignore                          # 排除大文件和 secrets
├── Modelfile                           # Ollama 部署配置
├── pyproject.toml                      # uv 依赖
├── README.md                           # 项目说明
└── DESIGN.md                           # 本文档
```

---

## 10. 技术选型理由

| 决策 | 选择 | 理由 |
|------|------|------|
| 基座模型 | Qwen2.5-7B-Instruct | 中文能力强、7B 规模适合 4090、社区角色扮演案例多 |
| 微调方法 | QLoRA | 单卡 24GB 只能跑 LoRA/QLoRA；QLoRA 显存更省 |
| 训练框架 | Unsloth | 单卡 4090 上训练速度最快、显存占用最低 |
| 偏好学习 | DPO | 实现简单、无需训练奖励模型、适合角色事实对齐 |
| 部署格式 | GGUF | llama.cpp / Ollama 通用，本地推理最高效 |
| 包管理 | uv | 用户指定，依赖解析快、环境隔离干净 |
| 数据增强 | Kimi API | 已有 API Key，生成质量高，适合半自动 DPO 数据 |

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 八纯台词仅 3,212 句 | 角色还原不足 | 旁白+其他角色上下文增强；Kimi API 生成额外多轮对话 |
| DPO 数据质量不佳 | 模型跑偏 | 严格 prompt + 自动规则 + 20% 人工抽检 |
| 4090 OOM | 训练失败 | gradient checkpointing、4-bit、减小 batch/seq_len |
| 过拟合 / 复读原台词 | 通用对话差 | 混合 10% 通用中文对话；epoch ≤ 3；监控 eval loss |
| 量化后语气损失 | 部署效果差 | 导出 q4_k_m / q5_k_m / q8_0 多版本对比 |
| 版权合规 | 法律风险 | 仅个人学习本地使用，不公开分发原剧本和模型 |

---

## 12. 后续可扩展方向

1. **多角色模型**：把椿、绘未、千夏也做成可选角色，通过 system prompt 切换。
2. **长上下文剧情记忆**：引入 RAG，把游戏剧情向量化，聊天时可检索相关剧情。
3. **语音合成联动**：结合八纯 CV（饴川紫乃）的语音数据训练 TTS / SVC。
4. **更强化学习**：尝试 KTO、ORPO 替代 DPO，探索更适合角色扮演的对齐方法。
5. **更大模型**：如果后续升级 GPU，可尝试 Qwen2.5-14B 或 32B。
