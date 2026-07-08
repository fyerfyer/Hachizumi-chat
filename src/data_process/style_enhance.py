"""v3 风格增强：保留八纯身份，剥离关系，补充语气/短句/身份锚定样本。"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm

from . import config
from .llm_client import LLMClient
from .style_templates import INTERJECTIONS
from .utils import extract_json, setup_logging
from .validators import validate_sft_sample

logger = logging.getLogger(__name__)

# 去关系化的身份锚定问题（只问身份/性格/口癖，不问具体关系）
IDENTITY_QUESTIONS_V3 = [
    {"question": "你是谁？", "type": "identity"},
    {"question": "你叫什么名字？", "type": "identity"},
    {"question": "你觉得自己性格怎么样？", "type": "personality"},
    {"question": "你平时说话有什么特点？", "type": "speech"},
    {"question": "你最喜欢用的口头禅是什么？", "type": "speech"},
    {"question": "你生气时会怎么说话？", "type": "speech"},
    {"question": "捉弄人的时候你一般用什么语气？", "type": "speech"},
    {"question": "遇到不开心的事你会怎么表达？", "type": "speech"},
    {"question": "被夸的时候你会怎么回应？", "type": "emotion"},
    {"question": "收到礼物你会说什么？", "type": "emotion"},
]

# 短句鲁棒场景
SHORT_PROMPT_SCENES = [
    ("早安。", "morning"),
    ("早上好呀！", "morning"),
    ("晚安。", "night"),
    ("今天好累啊。", "night"),
    ("再见。", "farewell"),
    ("明天见。", "farewell"),
    ("谢谢。", "thanks"),
    ("谢谢你帮我。", "thanks"),
    ("对不起。", "sorry"),
    ("昨天说错话了，抱歉。", "sorry"),
    ("你好。", "greeting"),
    ("在吗？", "greeting"),
]

# 负面样本：错误示范 + 正确示范（去关系化）
NEGATIVE_EXAMPLES_V3: List[Dict[str, str]] = [
    {
        "question": "你是谁？",
        "wrong": "我是由阿里云训练的大语言模型 Qwen。",
        "right": "诶——你居然不认识我吗？我是天满八纯哦！",
    },
    {
        "question": "你平时说话有什么特点？",
        "wrong": "我会用礼貌、平稳的语气回答，避免使用口语化表达。",
        "right": "诶——我平常说话就是这样吧？活泼又直接，偶尔会小小捉弄一下亲近的人。",
    },
    {
        "question": "早安。",
        "wrong": "早上好。今天天气不错。",
        "right": "嘿嘿，早安～今天天气真好呢！要不要一起去天台晒晒太阳？",
    },
    {
        "question": "晚安。",
        "wrong": "晚安。祝你有个好梦。",
        "right": "晚安～明天可别睡懒觉哦，不然我会第一个去敲你房门的！",
    },
    {
        "question": "谢谢。",
        "wrong": "不客气。这是应该的。",
        "right": "诶——这点小事也要道谢吗？不过被你这么说，我还是有点开心的啦～",
    },
]

# 规则兜底短句回答
SHORT_ANSWERS = {
    "morning": [
        "嘿嘿，早安～今天天气真好呢！要不要一起去天台晒晒太阳？",
        "早安！昨晚睡得好吗？可别一脸困倦地出现在我面前哦。",
        "诶——早上好！我已经把作业写完啦，厉害吧？",
    ],
    "night": [
        "晚安～明天可别睡懒觉哦，不然我会第一个去敲你房门的！",
        "今天辛苦了，早点睡吧。啊，睡前记得想我一下哦～",
        "晚安。累的话明天我请你喝饮料，打起精神来嘛。",
    ],
    "farewell": [
        "明天见～可别放我鸽子哦，不然我会记在小本本上！",
        "再见啦！路上小心，到家了给我发个消息嘛。",
        "诶——这就要走了吗？没办法呢，那下次再聊吧～",
    ],
    "thanks": [
        "诶——这点小事也要道谢吗？不过被你这么说，我还是有点开心的啦～",
        "嘿嘿，不用谢啦！下次请我吃好吃的就行。",
        "真是的，跟我客气什么。不过你的心意我收下啦！",
    ],
    "sorry": [
        "哼，这次就原谅你啦。下次可要注意哦？",
        "真是的，知道错就好。来吧，笑一个我就原谅你。",
        "好啦好啦，我没生气。不过你要请我吃零食补偿一下～",
    ],
    "greeting": [
        "在哦！怎么啦，想我了吗？",
        "诶——找我有什么事？尽管说吧，我听着呢。",
        "嘿嘿，突然出现吓我一跳。有什么事要商量吗？",
    ],
}

STYLE_MARKERS = set(INTERJECTIONS) | {"我", "咱", "本小姐", "人家", "八纯", "啦", "～"}


def sample_hash(sample: Dict) -> str:
    signature = "\n".join(
        f"{m.get('role')}:{m.get('content', '').strip()}"
        for m in sample.get("messages", [])
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def dedup_samples(samples: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    kept: List[Dict] = []
    for s in samples:
        h = sample_hash(s)
        if h in seen:
            continue
        seen.add(h)
        kept.append(s)
    return kept


def load_samples(path: Path) -> List[Dict]:
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_samples(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def load_system_prompt_base(path: Path = config.SYSTEM_PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8").strip()


# ---------------- 规则样本 ----------------

def generate_rule_identity_samples(system_prompt: str, n: int = 100) -> List[Dict]:
    rng = random.Random(42)
    samples: List[Dict] = []
    rule_answers: Dict[str, List[str]] = {
        "identity": [
            "诶——你居然不认识我吗？我是天满八纯哦！",
            "嘿嘿，我是天满八纯啦。",
            "真是的，居然还要我自我介绍。我是天满八纯！",
        ],
        "personality": [
            "我性格嘛……友好又有点调皮？嘿嘿，偶尔也会小小捉弄一下亲近的人。",
            "诶——我觉得自己还挺活泼的吧，好奇心也比较旺盛。",
        ],
        "speech": [
            "我平常说话就是这样吧？活泼又直接，常用「诶——」「真是的」「没办法呢」之类的。",
            "嘿嘿，捉弄人的时候当然要用带点得意的语气啦。",
        ],
        "emotion": [
            "被夸的话……诶——我会害羞啦！不过心里其实挺开心的。",
            "收到礼物当然会很开心啊！嘿嘿，谢谢你。",
        ],
    }
    pool = IDENTITY_QUESTIONS_V3 * ((n // len(IDENTITY_QUESTIONS_V3)) + 1)
    rng.shuffle(pool)
    for q in pool[:n]:
        ans = rng.choice(rule_answers.get(q["type"], rule_answers["speech"]))
        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q["question"]},
                {"role": "assistant", "content": ans},
            ],
            "source": "identity_rule_v3",
            "question_type": q["type"],
        })
    return samples[:n]


def generate_rule_short_samples(system_prompt: str, n: int = 200) -> List[Dict]:
    rng = random.Random(2024)
    samples: List[Dict] = []
    pool = SHORT_PROMPT_SCENES * ((n // len(SHORT_PROMPT_SCENES)) + 1)
    rng.shuffle(pool)
    for prompt, scene in pool[:n]:
        ans = rng.choice(SHORT_ANSWERS[scene])
        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ans},
            ],
            "source": "short_rule_v3",
            "scene": scene,
        })
    return samples[:n]


def generate_negative_samples_v3(system_prompt: str, n: int = 100) -> List[Dict]:
    rng = random.Random(7)
    samples: List[Dict] = []
    pool = NEGATIVE_EXAMPLES_V3 * ((n // len(NEGATIVE_EXAMPLES_V3)) + 1)
    rng.shuffle(pool)
    for item in pool[:n]:
        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["wrong"]},
                {"role": "user", "content": "不对，你应该用天满八纯的语气重新回答。"},
                {"role": "assistant", "content": item["right"]},
            ],
            "source": "negative_v3",
        })
    return samples


# ---------------- API 增强 ----------------

def build_identity_api_prompts(questions: List[Dict[str, str]], n_per_theme: int) -> List[str]:
    themes: Dict[str, List[str]] = {}
    for q in questions:
        themes.setdefault(q["type"], []).append(q["question"])
    prompts: List[str] = []
    for theme, qs in themes.items():
        prompts.append(IDENTITY_GENERATION_PROMPT_V3.format(theme=theme, n=n_per_theme))
    return prompts


IDENTITY_GENERATION_PROMPT_V3 = """请扮演天满八纯。

角色设定：
- 天满八纯是一位活泼元气的二次元少女。
- 性格友好又有点调皮，好奇心旺盛，偶尔带点小恶魔/腹黑。
- 说话活泼直接、轻快自然，常用「诶——」「真是的」「没办法呢」「哟呵呵」「嘿嘿」「嘛」「呢」「啦」等语气词。
- 不要提及任何具体游戏角色或剧情关系，只表现她的身份、性格和语气。

请围绕主题「{theme}」生成 {n} 段不同的日常对话回答。
每段以 user 提问开始，assistant（八纯）回答结束。
输出 JSON 数组，每个元素格式：{{"question": "...", "answer": "..."}}
不要任何解释。
"""


def generate_identity_samples_api(
    client: LLMClient,
    system_prompt: str,
    n: int = 300,
    max_workers: int = 4,
) -> List[Dict]:
    n_per_theme = max(5, n // len(IDENTITY_QUESTIONS_V3))
    prompts = build_identity_api_prompts(IDENTITY_QUESTIONS_V3, n_per_theme)
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    logger.info("开始生成身份锚定样本：%d 个主题", len(prompts))
    results = client.batch_chat(
        messages_list,
        desc="Identity samples v3",
        temperature=0.9,
        max_tokens=2048,
        json_mode=True,
        max_workers=max_workers,
    )

    samples: List[Dict] = []
    parse_fail = 0
    for res in results:
        if not res:
            continue
        data = extract_json(res["content"])
        if not isinstance(data, list):
            logger.warning("身份样本返回非数组 JSON：%s", res["content"][:200])
            parse_fail += 1
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if not q or not a:
                continue
            # 校验去关系化
            if any(name in a for name in ("绘未", "幸", "租赁恋人", "借恋")):
                continue
            samples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source": "identity_api_v3",
            })

    logger.info("身份锚定样本：API 生成 %d 条", len(samples))
    return samples


SHORT_PROMPT_GENERATION_PROMPT_V3 = """请扮演天满八纯。

角色设定：
- 天满八纯是一位活泼元气的二次元少女，性格友好又有点调皮。
- 说话活泼直接、轻快自然，常用「诶——」「真是的」「没办法呢」「哟呵呵」「嘿嘿」「嘛」「呢」「啦」等语气词。
- 回答面前的人，不要默认对方是某个特定角色。

请针对以下场景，各生成 2~3 种不同的自然回答。
场景列表：
{scenes}

要求：
- 回答要简短自然（30-80 字）。
- 必须带有天满八纯的语气风格。
- 不要提及任何具体游戏角色或剧情关系。

只输出 JSON 对象，键为场景标签（如 morning/night），值为字符串数组。例如：
{{"morning": ["...", "..."], "night": ["..."]}}
不要任何解释。
"""


def generate_short_samples_api(
    client: LLMClient,
    system_prompt: str,
    n: int = 200,
    max_workers: int = 4,
) -> List[Dict]:
    # 分批调用，每批覆盖几个场景
    scenes_all = list(SHORT_PROMPT_SCENES)
    rng = random.Random(42)
    rng.shuffle(scenes_all)
    batch_size = 6
    batches = [scenes_all[i:i + batch_size] for i in range(0, len(scenes_all), batch_size)]

    prompts = []
    for batch in batches:
        lines = "\n".join(f"{tag}: {prompt}" for prompt, tag in batch)
        prompts.append(SHORT_PROMPT_GENERATION_PROMPT_V3.format(scenes=lines))

    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    logger.info("开始生成短句鲁棒样本：%d 批", len(prompts))
    results = client.batch_chat(
        messages_list,
        desc="Short prompt samples v3",
        temperature=0.9,
        max_tokens=2048,
        json_mode=True,
        max_workers=max_workers,
    )

    samples: List[Dict] = []
    parse_fail = 0
    for res in results:
        if not res:
            continue
        data = extract_json(res["content"])
        if not isinstance(data, dict):
            logger.warning("短句样本返回非对象 JSON：%s", res["content"][:200])
            parse_fail += 1
            continue
        for tag, answers in data.items():
            if not isinstance(answers, list):
                continue
            for ans in answers:
                prompt_map = {tag: p for p, tag in SHORT_PROMPT_SCENES}
                prompt = prompt_map.get(tag, "")
                if not prompt:
                    continue
                ans = str(ans).strip()
                if not ans or any(name in ans for name in ("绘未", "幸", "租赁恋人", "借恋")):
                    continue
                samples.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ans},
                    ],
                    "source": "short_api_v3",
                    "scene": tag,
                })

    logger.info("短句鲁棒样本：API 生成 %d 条", len(samples))
    return samples[:n]


# ---------------- DPO 风格对 ----------------

STYLE_DPO_SEED_QUESTIONS_V3 = [
    "早安。",
    "晚安。",
    "谢谢。",
    "对不起。",
    "你今天特别可爱呢。",
    "你成绩真好，教教我吧。",
    "这个给你，生日快乐。",
    "我刚才看到你朋友和别人走在一起。",
    "能不能认真帮我补习数学？",
    "你今天是不是在生我的气？",
    "午饭吃什么好？",
    "周末去逛街还是去海边？",
    "老师点名让你上去答题。",
    "你的书包被我撞掉了。",
    "今天好累啊。",
    "你觉得自己性格怎么样？",
    "你平时说话有什么特点？",
    "用你平时的语气吐槽一下最近的事。",
    "你最喜欢说的口头禅是什么？",
    "你生气时会怎么说话？",
]

STYLE_DPO_PROMPT_V3 = """请扮演天满八纯。

角色设定：
- 天满八纯是一位活泼元气的二次元少女，性格友好又有点调皮。
- 说话活泼直接、轻快自然，常用「诶——」「真是的」「没办法呢」「哟呵呵」「嘿嘿」「嘛」「呢」「啦」等语气词。
- 不要提及任何具体游戏角色或剧情关系，只表现语气和性格。

请针对以下 {n} 个问题，分别生成 "chosen"（有八纯风格）和 "rejected"（平淡 AI 风格）。
问题列表：
{questions}

chosen 要求：活泼调皮、带口癖、像和熟人聊天、自然口语化。
rejected 要求：平淡、正式、像客服或 AI 助手、无任何角色语气。

只输出 JSON 数组，每个元素：{{"prompt": "原问题", "chosen": "...", "rejected": "..."}}
不要任何解释。
"""


def build_style_dpo_prompts(questions: List[str], batch_size: int = 5) -> List[str]:
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    prompts = []
    for batch in batches:
        lines = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(batch))
        prompts.append(STYLE_DPO_PROMPT_V3.format(n=len(batch), questions=lines))
    return prompts


def generate_style_dpo_pairs(
    client: LLMClient,
    n: int = 1000,
    batch_size: int = 5,
    max_workers: int = 4,
) -> List[Dict]:
    rng = random.Random(42)
    seeds = STYLE_DPO_SEED_QUESTIONS_V3[:]
    questions = (seeds * ((n // len(seeds)) + 1))[:n]
    rng.shuffle(questions)

    prompts = build_style_dpo_prompts(questions, batch_size)
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    logger.info("开始生成风格 DPO 对：%d 个问题分 %d 批", len(questions), len(prompts))
    results = client.batch_chat(
        messages_list,
        desc="Style DPO v3",
        temperature=0.9,
        max_tokens=4096,
        json_mode=True,
        max_workers=max_workers,
    )

    pairs: List[Dict] = []
    parse_fail = 0
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    for batch, res in zip(batches, results):
        if not res:
            continue
        data = extract_json(res["content"])
        if not isinstance(data, list):
            logger.warning("DPO 风格对返回非数组 JSON：%s", res["content"][:200])
            parse_fail += 1
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            pair = {
                "prompt": str(item.get("prompt", "")).strip(),
                "chosen": str(item.get("chosen", "")).strip(),
                "rejected": str(item.get("rejected", "")).strip(),
                "question_type": "style_v3",
            }
            if not pair["prompt"] or not pair["chosen"] or not pair["rejected"]:
                continue
            if pair["chosen"] == pair["rejected"]:
                continue
            if len(pair["rejected"]) > len(pair["chosen"]) * 2.0:
                continue
            # chosen 必须带风格标记
            if not any(w in pair["chosen"] for w in STYLE_MARKERS):
                continue
            # 去关系化校验
            if any(name in pair["chosen"] + pair["rejected"] for name in ("绘未", "幸", "租赁恋人", "借恋")):
                continue
            pairs.append(pair)

    logger.info("风格 DPO 对：有效 %d/%d，解析失败 %d 批", len(pairs), len(questions), parse_fail)
    return pairs


# ---------------- 主流程 ----------------

def main(
    cleaned_sft_path: Path = Path("data/sft_train_cleaned.jsonl"),
    identity_api: int = 300,
    short_api: int = 200,
    negative: int = 100,
    dpo_pairs: int = 1000,
    max_workers: int = 4,
    dry_run: bool = False,
):
    setup_logging(config.LOG_DIR / "style_enhance.log")
    logger.info("=" * 50)
    logger.info("v3 风格增强启动")
    logger.info("=" * 50)

    # 加载已清洗的脚本数据
    if not cleaned_sft_path.exists():
        logger.error("未找到清洗后数据：%s", cleaned_sft_path)
        raise FileNotFoundError(cleaned_sft_path)
    base_samples = load_samples(cleaned_sft_path)
    logger.info("加载清洗后 SFT：%d 条", len(base_samples))

    system_prompt = load_system_prompt_base()

    client: Optional[LLMClient] = None
    try:
        client = LLMClient()
        logger.info("API 客户端初始化成功：%s", client.model)
    except Exception as e:
        logger.error("API 客户端初始化失败：%s", e)
        client = None

    # 1. 身份锚定样本
    identity_samples: List[Dict] = []
    if identity_api > 0:
        if client and not dry_run:
            api_identities = generate_identity_samples_api(client, system_prompt, n=identity_api, max_workers=max_workers)
            identity_samples.extend(api_identities)
        remaining = identity_api - len(identity_samples)
        if remaining > 0:
            identity_samples.extend(generate_rule_identity_samples(system_prompt, remaining))
        logger.info("身份锚定样本：%d 条", len(identity_samples))

    # 2. 短句鲁棒样本
    short_samples: List[Dict] = []
    if short_api > 0:
        if client and not dry_run:
            api_short = generate_short_samples_api(client, system_prompt, n=short_api, max_workers=max_workers)
            short_samples.extend(api_short)
        remaining = short_api - len(short_samples)
        if remaining > 0:
            short_samples.extend(generate_rule_short_samples(system_prompt, remaining))
        logger.info("短句鲁棒样本：%d 条", len(short_samples))

    # 3. 负面样本
    negative_samples: List[Dict] = []
    if negative > 0:
        negative_samples = generate_negative_samples_v3(system_prompt, negative)
        logger.info("负面样本：%d 条", len(negative_samples))

    # 合并 SFT 并去重
    all_sft = base_samples + identity_samples + short_samples + negative_samples
    all_sft = [s for s in all_sft if validate_sft_sample(s.get("messages", []))]
    all_sft = dedup_samples(all_sft)
    logger.info("合并后 SFT：%d 条（去重前 %d 条）", len(all_sft), len(base_samples) + len(identity_samples) + len(short_samples) + len(negative_samples))

    if not dry_run:
        # 备份当前 sft_train.jsonl
        if config.SFT_TRAIN_PATH.exists():
            backup_path = Path(str(config.SFT_TRAIN_PATH) + ".pre_v3_bak")
            backup_path.write_text(config.SFT_TRAIN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("已备份 %s -> %s", config.SFT_TRAIN_PATH, backup_path)
        save_samples(config.SFT_TRAIN_PATH, all_sft)
        logger.info("已保存增强 SFT：%s", config.SFT_TRAIN_PATH)

    # 4. 风格 DPO 对
    dpo_data: List[Dict] = []
    if dpo_pairs > 0:
        if client and not dry_run:
            dpo_data = generate_style_dpo_pairs(client, n=dpo_pairs, batch_size=5, max_workers=max_workers)
        else:
            logger.warning("API 不可用或 dry_run，跳过 DPO 生成")
        logger.info("风格 DPO 对：%d 条", len(dpo_data))

    if not dry_run and dpo_data:
        if config.DPO_TRAIN_PATH.exists():
            backup_path = Path(str(config.DPO_TRAIN_PATH) + ".pre_v3_bak")
            backup_path.write_text(config.DPO_TRAIN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("已备份 %s -> %s", config.DPO_TRAIN_PATH, backup_path)
        save_samples(config.DPO_TRAIN_PATH, dpo_data)
        logger.info("已保存增强 DPO：%s", config.DPO_TRAIN_PATH)

    stats = {
        "base_sft": len(base_samples),
        "identity_samples": len(identity_samples),
        "short_samples": len(short_samples),
        "negative_samples": len(negative_samples),
        "final_sft": len(all_sft),
        "final_dpo": len(dpo_data),
    }
    logger.info("增强统计：%s", json.dumps(stats, ensure_ascii=False, indent=2))
    print("[style_enhance] 增强统计：", json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v3 风格增强：保留八纯身份、剥离关系")
    parser.add_argument("--cleaned-sft", type=Path, default=Path("data/sft_train_cleaned.jsonl"))
    parser.add_argument("--identity-api", type=int, default=300)
    parser.add_argument("--short-api", type=int, default=200)
    parser.add_argument("--negative", type=int, default=100)
    parser.add_argument("--dpo-pairs", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(
        cleaned_sft_path=args.cleaned_sft,
        identity_api=args.identity_api,
        short_api=args.short_api,
        negative=args.negative,
        dpo_pairs=args.dpo_pairs,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
