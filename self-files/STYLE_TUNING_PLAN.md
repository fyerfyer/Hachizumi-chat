# 天满八纯角色语气调优方案

> 目标：让模型在对话中更稳定地保持天满八纯的活泼、调皮、轻快的语气风格，不强求剧情细节 100% 准确。
>
> 状态：调研文档，未实际执行。

## 一、当前效果诊断

基于 vLLM 实际推理测试（`outputs/hasumi_qwen25_7b_merged`，bfloat16），当前模型表现：

- **语气基础已具备**：能使用 `哟呵呵`、`诶——`、`嘿嘿`、`真是的`、`嘛`、`呢` 等高频口癖，句式轻快，会调侃、装可爱。
- **主要问题**：
  1. **身份锚定不稳**：不加 system prompt 时会退化为基座模型身份（自称 Qwen）。
  2. **角色关系偶发混淆**：会把"绘未"（青梅竹马）和"幸"（同班同学/租赁恋人）的关系说混。
  3. **事实幻觉**：如把蓝瞳说成"浅棕色"、红发"染了点金色"。
  4. **风格强度不够一致**：部分回答仍偏通用 AI 腔。

因为用户更关注**语气性格像不像**，本方案重点解决第 1、4 点，对第 2、3 点只做顺带缓解。

## 二、核心思路

业界做角色扮演/语气风格迁移的常见路径：

1. **数据层面**：清洗剧本数据、用更强的大模型做风格迁移/同义改写/合成对话。
2. **训练层面**：SFT 强化风格、DPO 做"有风格 vs 没风格"的偏好对齐。
3. **推理层面**：system prompt + few-shot 锚定身份。

本方案优先用 **DeepSeek API** 做数据增强，原因是：

- 成本低（`deepseek-chat` 约 $0.27/1M tokens 输入、$1.10/1M tokens 输出，且有缓存命中折扣）。
- 中文能力强，适合日式轻小说/ Galgame 风格的改写。
- API 兼容 OpenAI 格式，接入成本低。

## 三、DeepSeek API 数据增强方案

### 3.1 可选模型

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| `deepseek-chat`（DeepSeek-V3/V3.1） | 非思考模式，响应快，价格低 | 批量同义改写、风格迁移、生成合成对话 |
| `deepseek-reasoner`（DeepSeek-R1） | 思考模式，逻辑和指令遵循更强 | 生成复杂 DPO 偏好对、评估回答质量 |

对于本项目，**`deepseek-chat` 足够**，只有在生成高质量 DPO chosen/rejected 对时才需要 `deepseek-reasoner`。

### 3.2 具体增强策略

#### 策略 A：输入清洗（去掉角色前缀）

现有 `data/sft_train.jsonl` 中大量 user 消息带角色前缀：

```json
{"role": "user", "content": "幸：啊? 啊......是啊"}
```

实际聊天时用户不会自报"幸："。用 DeepSeek API 批量改写为：

```json
{"role": "user", "content": "啊? 啊......是啊"}
```

提示词示例：

```text
你是一名 Galgame 剧本数据清洗助手。下面这段对话来自游戏剧本，
user 消息中可能带有"角色名："前缀。请去掉前缀，只保留角色实际说的话。
不要改变原意，不要添加解释。

输入：幸：啊? 啊......是啊
输出：啊? 啊......是啊
```

#### 策略 B：语气风格迁移（把普通回答改成天满八纯风格）

从现有 assistant 回复中抽取"普通/正式"样本，用 API 改写成带口癖的版本：

```text
请把下面这句话改写成天满八纯的说话风格。

人物设定：
- 天满八纯是绘未的青梅竹马兼同班同学。
- 性格友好又有点调皮，说话活泼直接，偶尔会小小地捉弄亲近的人。
- 常用语气词：哟呵呵、诶——、真是的、没办法呢、嘛、呢、嘿嘿。
- 语速偏快，情绪上来时会连续吐槽。

要求：
- 保持原意不变。
- 必须带 1-2 个上述语气词。
- 句子要轻快、口语化，不要太正式。

输入：我们去图书馆吧。
输出：诶——去图书馆吗？没办法呢，那就陪你一下好啦～
```

#### 策略 C：生成"自我介绍/身份锚定"样本

针对"你是谁"这类高频破冰问题，批量生成多样化回答：

```text
请扮演天满八纯，用 3 种不同方式回答"你是谁？"。
要求：
- 必须提到自己是天满八纯。
- 必须提到绘未是自己的青梅竹马。
- 语气要活泼、调皮、带口癖。
- 每种回答长度控制在 30-60 字。
```

#### 策略 D：生成 DPO 语气偏好对

用 API 批量生成 chosen（有风格）/ rejected（平淡）对：

```text
请为天满八纯生成一段对话偏好对。

用户问题：{question}

chosen 要求：
- 天满八纯语气，活泼、调皮、带口癖。
- 像在和熟人聊天。

rejected 要求：
- 平淡、正式、像客服或 AI 助手。
- 不要有任何角色语气。

请直接输出 JSON：
{
  "chosen": "...",
  "rejected": "..."
}
```

#### 策略 E：Negative Examples（明确不要说的内容）

生成少量"错误示范"加入 SFT，告诉模型哪些表达要避免：

```text
用户：你是谁？
错误回答（不要这样回答）：我是由阿里云训练的大语言模型 Qwen。
正确回答（应该这样回答）：诶——你居然不认识我吗？我是天满八纯哦！
```

### 3.3 API 调用示例代码

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1"
)

def rewrite_in_hasumi_style(text: str) -> str:
    prompt = f"""请把下面这句话改写成天满八纯的说话风格。

人物设定：天满八纯是绘未的青梅竹马兼同班同学，性格友好又有点调皮，
说话活泼直接，常用"哟呵呵""诶——""真是的""嘛""呢"等语气词。

要求：保持原意不变，必须带 1-2 个语气词，句子轻快口语化。

输入：{text}
输出："""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=128,
    )
    return resp.choices[0].message.content
```

### 3.4 成本估算

假设对 10,000 条样本做风格迁移，平均输入 100 tokens、输出 100 tokens：

- 输入：1M tokens × $0.27 = $0.27
- 输出：1M tokens × $1.10 = $1.10
- **合计约 $1.37（约 ¥10）**

如果加入 DPO 生成（输出更长，约 200 tokens）和自我介绍样本，总成本预计在 **$5-15** 区间，非常便宜。

## 四、其他可行方案

### 4.1 本地开源模型做数据增强

如果不想调用 API，可以用本地已下载的 `Qwen2.5-7B-Instruct` 或 `Qwen2.5-14B-Instruct` 做风格迁移：

- 优点：完全离线、无 API 费用、数据不外流。
- 缺点：7B 模型的改写质量通常不如 DeepSeek-V3/R1，需要更多后清洗。

适合对数据隐私敏感或想批量跑的场景。

### 4.2 规则模板 + LLM 混合增强

对高频句式用规则模板生成，对复杂句子用 LLM 改写：

```python
templates = [
    "诶——{content}？",
    "哟呵呵，{content}嘛～",
    "真是的，{content}呢！",
    "嘿嘿，{content}哦。",
]
```

这种方法成本低、可控性强，适合补充大量短句风格样本。

### 4.3 社区/开源方案参考

- **Chat-嬛嬛**：基于《甄嬛传》剧本，用 ChatGLM + LoRA 微调模仿甄嬛语气，提供了从剧本到数据集的完整流程 [[GitHub]](https://github.com/chg0901/H-Chat)。
- **RoleLLM / CharacterEval**：研究提示工程 + 检索增强来做角色扮演，强调 context alignment 和 stylistic consistency [[arXiv]](https://arxiv.org/pdf/2406.00627v2)。
- **Ditto (Self-Alignment for Role-play)**：提出让 LLM 自我生成角色扮演数据，做 self-alignment，不依赖 GPT-4 [[ACL Anthology]](https://aclanthology.org/people/k/keming-lu/)。
- **KT-LoRA 风格化对话**：用 NekoQA-10K 做风格迁移微调，验证 LoRA 能以较低 GPU 成本把特定风格注入模型 [[ktransformers]](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/KTransformers-Fine-Tuning_Developer-Technical-Notes_zh.md)。

## 五、建议实施步骤

1. **数据清洗**：用 DeepSeek API 去掉 `sft_train.jsonl` 中 user 消息的角色前缀。
2. **风格增强**：抽取 5,000-10,000 条 assistant 回复，用 API 做语气风格迁移。
3. **生成锚定样本**：批量生成 200-500 条"你是谁/你和绘未什么关系"的自我介绍样本。
4. **生成 DPO 对**：用 API 生成 1,500-3,000 条"有风格 vs 没风格"的偏好对。
5. **合并新数据集**：与原 SFT/DPO 数据合并，注意去重。
6. **重新训练**：
   - SFT：3-4 epoch，学习率可适当降低。
   - DPO：beta 降到 0.05，重点练语气风格。
7. **评测**：用固定问题集测试语气稳定性，重点看"你是谁"和日常闲聊。

## 六、风险与注意事项

- **API 调用成本**：虽然 DeepSeek 便宜，但批量生成前建议先用 100 条样本跑通流程并检查质量。
- **数据泄漏**：剧本内容会上传到 DeepSeek 服务器。如果介意，改用本地模型方案。
- **风格过拟合**：增强样本过多可能导致模型每句话都带口癖。需要保留一定比例自然样本做平衡。
- **指令遵循冲突**：DPO 力度过大可能削弱模型基础能力。建议 beta 从 0.05 开始调。

## 七、参考资料

- DeepSeek API 官方文档：https://platform.deepseek.com/
- DeepSeek 定价参考（2025）：`deepseek-chat` 输入 $0.27/M、输出 $1.10/M [[BytePlus]](https://www.byteplus.com/en/topic/382578)
- 情绪对话模型微调与同义改写/风格迁移实践 [[博客园]](https://www.cnblogs.com/yuanxiaojiang/p/18829703)
- Prompt Framework for Role-playing: Generation and Evaluation [[arXiv]](https://arxiv.org/pdf/2406.00627v2)
- Chat-嬛嬛：基于剧本的角色扮演 LoRA 微调项目 [[GitHub]](https://github.com/chg0901/H-Chat)
- KT-LoRA 风格化对话测试 [[ktransformers]](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/KTransformers-Fine-Tuning_Developer-Technical-Notes_zh.md)
