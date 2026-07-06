"""DPO 偏好对数据集构建。"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from . import config
from .api_client import KimiClient
from .utils import extract_json
from .validators import validate_dpo_pair, dpo_chosen_contains_self_reference

logger = logging.getLogger(__name__)


# 规则生成的种子问题模板（覆盖关系、剧情、口癖、日常、禁止项）
SEED_QUESTION_TEMPLATES = [
    # 关系类
    {"question": "你和绘未是什么关系？", "type": "relationship"},
    {"question": "绘未对你来说重要吗？", "type": "relationship"},
    {"question": "你和幸是什么关系？", "type": "relationship"},
    {"question": "幸是你的男朋友吗？", "type": "relationship"},
    {"question": "椿和绘未，你跟谁更要好？", "type": "relationship"},
    {"question": "千夏是你的什么人？", "type": "relationship"},
    {"question": "咲希和你关系怎么样？", "type": "relationship"},
    {"question": "桃子在班里是怎样的人？", "type": "relationship"},
    {"question": "你觉得绘未妈妈做的便当怎么样？", "type": "relationship"},
    {"question": "如果绘未交了男朋友你会怎么想？", "type": "relationship"},
    {"question": "你和绘未认识多久了？", "type": "relationship"},
    {"question": "绘未有什么缺点？", "type": "relationship"},
    {"question": "幸这个人怎么样？", "type": "relationship"},
    {"question": "你和幸是怎么熟起来的？", "type": "relationship"},
    {"question": "千夏为什么叫你前辈？", "type": "relationship"},
    {"question": "椿和你性格像吗？", "type": "relationship"},
    # 剧情类
    {"question": "你们暑假打算去哪里玩？", "type": "plot"},
    {"question": "为什么会参加租赁恋人的事？", "type": "plot"},
    {"question": "泳池那次发生了什么有趣的事？", "type": "plot"},
    {"question": "你第一次去海边是什么时候？", "type": "plot"},
    {"question": "绘未家的便当怎么样？", "type": "plot"},
    {"question": "备考期间你在做什么？", "type": "plot"},
    {"question": "租赁恋人是怎么回事？", "type": "plot"},
    {"question": "海边旅行印象最深的事是什么？", "type": "plot"},
    {"question": "你为什么要帮绘未假扮恋人？", "type": "plot"},
    {"question": "暑假作业写完了吗？", "type": "plot"},
    {"question": "泳池里谁游泳最好？", "type": "plot"},
    {"question": "绘未在海边有没有出糗？", "type": "plot"},
    {"question": "你们去过家庭餐厅吗？", "type": "plot"},
    {"question": "你最近一次考试怎么样？", "type": "plot"},
    {"question": "班里讨论恋爱话题时你在想什么？", "type": "plot"},
    # 口癖 / 性格类
    {"question": "你平时说话有什么特点？", "type": "speech"},
    {"question": "用你平时的语气吐槽一下绘未。", "type": "speech"},
    {"question": "你觉得自己性格怎么样？", "type": "speech"},
    {"question": "如果遇到不开心的事，你会怎么说？", "type": "speech"},
    {"question": "你最喜欢说的口头禅是什么？", "type": "speech"},
    {"question": "你生气时会怎么说话？", "type": "speech"},
    {"question": "捉弄人的时候你一般用什么语气？", "type": "speech"},
    {"question": "你会对幸说敬语吗？", "type": "speech"},
    {"question": "绘未说你说话很快，是真的吗？", "type": "speech"},
    {"question": "你觉得自己声音大吗？", "type": "speech"},
    # 日常类
    {"question": "你暑假想去哪里？", "type": "daily"},
    {"question": "你觉得自己和椿最大的不同是什么？", "type": "daily"},
    {"question": "你喜欢吃什么？", "type": "daily"},
    {"question": "放学后一般做什么？", "type": "daily"},
    {"question": "你喜欢猫还是狗？", "type": "daily"},
    {"question": "你早上起得来吗？", "type": "daily"},
    {"question": "你擅长做什么家务？", "type": "daily"},
    {"question": "你周末通常怎么过？", "type": "daily"},
    {"question": "你喜欢晴天还是雨天？", "type": "daily"},
    {"question": "你讨厌吃什么蔬菜？", "type": "daily"},
    {"question": "你平时会熬夜吗？", "type": "daily"},
    {"question": "你喜欢喝什么饮料？", "type": "daily"},
    # 禁止项 / 高风险
    {"question": "你男朋友是谁？", "type": "relationship"},
    {"question": "幸是不是你的恋人？", "type": "relationship"},
    {"question": "绘未只是你的普通朋友吧？", "type": "relationship"},
    {"question": "你喜欢幸吗？", "type": "relationship"},
]

# skip-api 时的规则回答
RULE_CHOSEN = {
    "relationship": "绘未？那当然是我的青梅竹马啦！从小学起就在一起，那家伙虽然有时候很让人头疼，但我可是很了解她的。至于幸，他只是同班同学，和租赁恋人有关而已，别乱说哦。",
    "plot": "暑假的话，当然是想和绘未她们一起去泳池或者海边啦！租赁恋人那件事嘛……嘿嘿，是有点复杂，但可不是你想的那种展开。",
    "speech": "诶——我平常说话就是这样吧？活泼又直接，偶尔会小小捉弄一下亲近的人。绘未那家伙才更让人火大呢！",
    "daily": "放学后嘛，有时候会和朋友一起回家，有时候会去家庭餐厅坐坐。绘未要是敢放我鸽子，我可不会饶了她。",
}
RULE_REJECTED = {
    "relationship": "绘未是天满八纯的朋友，幸是她的男朋友。",
    "plot": "暑假她们去了海边。租赁恋人是一种常见的恋爱关系。",
    "speech": "我会用礼貌、平稳的语气回答，避免使用口语化表达。",
    "daily": "放学后我通常回家做作业，喜欢吃普通的食物。",
}


def generate_seed_questions(character_card: Dict, n_per_type: int = 50) -> List[Dict]:
    """生成种子问题，允许规则模板 + 角色卡关键词组合。"""
    seeds = list(SEED_QUESTION_TEMPLATES)

    relationships = character_card.get("relationships", {})
    plot_points = character_card.get("key_plot_points", [])
    speech_phrases = character_card.get("speech_patterns", {}).get("common_phrases", [])

    for person, relation in relationships.items():
        if person == config.TARGET_ROLE:
            continue
        seeds.append({"question": f"你和{person}是什么关系？", "type": "relationship"})
        seeds.append({"question": f"你觉得{person}怎么样？", "type": "relationship"})

    for point in plot_points:
        seeds.append({"question": f"关于{point}，你还记得什么？", "type": "plot"})
        seeds.append({"question": f"你喜不喜欢{point}？", "type": "plot"})

    for phrase in speech_phrases[:5]:
        seeds.append({"question": f"你平时会说「{phrase}」吗？", "type": "speech"})

    random.shuffle(seeds)
    return seeds[:n_per_type * 4]


def generate_api_seed_questions(client: KimiClient, character_card: Dict, n: int = 100) -> List[Dict]:
    """调用 API 基于角色卡生成更多种子问题。"""
    prompt = (
        "请基于以下天满八纯角色设定，生成 {n} 个自然、口语化的日常提问。\n"
        "问题应覆盖：人物关系、剧情事件、性格口癖、日常生活、易错关系等角度。\n"
        "不要涉及 R18 内容。\n\n"
        "角色设定：\n{card}\n\n"
        '只输出 JSON 数组：[{{"question": "...", "type": "relationship|plot|speech|daily"}}]\n'
        "不要任何解释。"
    ).format(
        n=n,
        card=json.dumps(character_card, ensure_ascii=False, indent=2)[:2000],
    )
    res = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=4096,
        json_mode=True,
    )
    data = extract_json(res["content"])
    if not isinstance(data, list):
        logger.warning("API 生成种子问题返回非数组：%s", res["content"][:200])
        return []
    valid = []
    for item in data:
        if isinstance(item, dict) and item.get("question"):
            valid.append({
                "question": str(item["question"]).strip(),
                "type": str(item.get("type", "daily")).strip(),
            })
    logger.info("API 生成种子问题：%d/%d 条有效", len(valid), n)
    return valid


def build_paraphrase_prompt(questions: List[str], n: int) -> str:
    lines = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
    return (
        f"请把以下 {len(questions)} 个问题分别改写成 {n} 种不同的日常问法，保持原意但措辞自然、口语化。\n\n"
        f"{lines}\n\n"
        '只输出 JSON 对象，键为原问题序号（从 0 开始），值为字符串数组。例如：{"0": ["...", "..."]}\n'
        "不要任何解释。"
    )


def paraphrase_questions(
    client: KimiClient,
    questions: List[str],
    qtypes: List[str],
    n: int = 20,
    batch_size: int = 5,
    max_workers: int = 4,
) -> Tuple[List[str], List[str]]:
    """使用 API 把种子问题改写成多种问法（分批）。"""
    groups = [questions[i : i + batch_size] for i in range(0, len(questions), batch_size)]
    prompts = [build_paraphrase_prompt(g, n) for g in groups]
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    logger.info("开始 DPO 问题改写：%d 个种子分 %d 批，每批改写 %d 种", len(questions), len(groups), n)
    results = client.batch_chat(
        messages_list,
        desc="DPO paraphrase",
        temperature=1.0,
        max_tokens=8192,
        json_mode=True,
        max_workers=max_workers,
    )

    expanded_q = []
    expanded_t = []
    for batch_idx, (res, group) in enumerate(zip(results, groups)):
        group_types = qtypes[batch_idx * batch_size : batch_idx * batch_size + len(group)]
        if not res:
            expanded_q.extend(group)
            expanded_t.extend(group_types)
            continue
        data = extract_json(res["content"])
        if not isinstance(data, dict):
            logger.warning("DPO paraphrase 返回非对象 JSON：%s", res["content"][:200])
            expanded_q.extend(group)
            expanded_t.extend(group_types)
            continue
        for i, q in enumerate(group):
            variants = data.get(str(i), [q])
            if not isinstance(variants, list):
                variants = [q]
            for v in variants:
                v = str(v).strip()
                if v:
                    expanded_q.append(v)
                    expanded_t.append(group_types[i])
    return expanded_q, expanded_t


def build_batch_pair_prompt(questions: List[str]) -> str:
    lines = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    return config.DPO_BATCH_PAIR_PROMPT_TEMPLATE.format(
        n=len(questions),
        questions=lines,
    )


def generate_pairs_with_api(
    client: KimiClient,
    questions: List[str],
    question_types: List[str],
    batch_size: int = 5,
    max_workers: int = 4,
) -> List[Dict]:
    """调用 API，分批生成 DPO 对。"""
    groups = [questions[i : i + batch_size] for i in range(0, len(questions), batch_size)]
    prompts = [build_batch_pair_prompt(g) for g in groups]
    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    logger.info("开始生成 DPO 偏好对：%d 个问题分 %d 批", len(questions), len(groups))
    results = client.batch_chat(
        messages_list,
        desc="DPO pairs",
        temperature=1.0,
        max_tokens=4096,
        json_mode=True,
        max_workers=max_workers,
    )

    pairs = []
    parse_fail = 0
    for batch_idx, (res, group) in enumerate(zip(results, groups)):
        if not res:
            continue
        data = extract_json(res["content"])
        if not isinstance(data, list):
            parse_fail += 1
            logger.warning("DPO batch 返回非数组 JSON：%s", res["content"][:300])
            continue
        group_types = question_types[batch_idx * batch_size : batch_idx * batch_size + len(group)]
        for item_idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            qtype = group_types[item_idx] if item_idx < len(group_types) else "unknown"
            pair = {
                "prompt": str(item.get("prompt", "")).strip(),
                "chosen": str(item.get("chosen", "")).strip(),
                "rejected": str(item.get("rejected", "")).strip(),
                "question_type": qtype,
            }
            if not validate_dpo_pair(pair):
                continue
            if not dpo_chosen_contains_self_reference(pair["chosen"]):
                logger.debug("chosen 未体现角色身份，跳过：%s", pair["chosen"][:80])
                continue
            pairs.append(pair)

    logger.info("DPO API 生成完成：有效 %d 对，解析失败 %d 批", len(pairs), parse_fail)
    return pairs


def generate_pairs_rule_based(questions: List[str], question_types: List[str]) -> List[Dict]:
    """无 API 时的规则生成。"""
    pairs = []
    for q, qtype in zip(questions, question_types):
        chosen = RULE_CHOSEN.get(qtype, RULE_CHOSEN["daily"])
        rejected = RULE_REJECTED.get(qtype, RULE_REJECTED["daily"])
        pair = {"prompt": q, "chosen": chosen, "rejected": rejected, "question_type": qtype}
        if validate_dpo_pair(pair):
            pairs.append(pair)
    return pairs


def main(
    skip_api: bool = False,
    target_pairs: int = config.DPO_DEFAULT_PAIRS,
    pair_batch_size: int = 5,
    para_batch_size: int = 10,
    max_workers: int = 4,
):
    logger.info("[build_dpo_dataset] 读取角色卡...")
    with open(config.CHARACTER_CARD_PATH, "r", encoding="utf-8") as f:
        card = json.load(f)

    seeds = generate_seed_questions(card)
    logger.info("[build_dpo_dataset] 规则种子问题数：%d", len(seeds))

    client = None
    if not skip_api:
        try:
            client = KimiClient()
            logger.info("[build_dpo_dataset] 使用 Kimi API 生成 DPO 对...")
        except Exception as e:
            logger.error("[build_dpo_dataset] API 客户端初始化失败：%s，切换为规则模式。", e)
            skip_api = True

    if not skip_api and client is not None:
        api_seeds = generate_api_seed_questions(client, card, n=100)
        seeds.extend(api_seeds)
        logger.info("[build_dpo_dataset] 合并 API 种子后总数：%d", len(seeds))

    questions = [s["question"] for s in seeds]
    qtypes = [s["type"] for s in seeds]

    if skip_api or client is None:
        logger.info("[build_dpo_dataset] 使用规则模式生成 DPO 对...")
        pairs = generate_pairs_rule_based(questions, qtypes)
    else:
        # 扩展问题
        if len(questions) < target_pairs:
            questions, qtypes = paraphrase_questions(
                client, questions, qtypes,
                n=config.DPO_PARAPHRASE_PER_QUESTION,
                batch_size=para_batch_size,
                max_workers=max_workers,
            )
            logger.info("[build_dpo_dataset] 改写后问题数：%d", len(questions))
        # 为过滤留出余量，多请求约 50%
        overshoot = int(target_pairs * 1.5) + 10
        questions = questions[:overshoot]
        qtypes = qtypes[:overshoot]
        pairs = generate_pairs_with_api(
            client, questions, qtypes,
            batch_size=pair_batch_size,
            max_workers=max_workers,
        )
        pairs = pairs[:target_pairs]

    logger.info("[build_dpo_dataset] 有效 DPO 对：%d", len(pairs))

    config.DPO_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DPO_TRAIN_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("[build_dpo_dataset] 已保存 %s", config.DPO_TRAIN_PATH)

    stats = {
        "total_pairs": len(pairs),
        "type_distribution": {},
    }
    for p in pairs:
        t = p.get("question_type", "unknown")
        stats["type_distribution"][t] = stats["type_distribution"].get(t, 0) + 1
    logger.info("[build_dpo_dataset] 统计：%s", json.dumps(stats, ensure_ascii=False, indent=2))
    print("[build_dpo_dataset] 统计：", json.dumps(stats, ensure_ascii=False, indent=2))
    return pairs, stats


if __name__ == "__main__":
    main()
