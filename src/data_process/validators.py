"""SFT / DPO 样本校验。"""
from __future__ import annotations

from typing import Dict, List

from . import config


def validate_sft_sample(messages: List[Dict[str, str]]) -> bool:
    """校验一条 SFT 消息链。"""
    if not messages:
        return False
    if messages[0]["role"] != "system":
        return False
    # 至少一条 user 与一条 assistant
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return False
    # 最后一条必须是 assistant
    if roles[-1] != "assistant":
        return False
    # 无空内容
    for m in messages:
        if not m.get("content", "").strip():
            return False
    # assistant 必须来自八纯（脚本数据已保证；合成数据需额外检查）
    # 这里仅做格式校验
    return True


def validate_dpo_pair(pair: Dict) -> bool:
    """校验一条 DPO 偏好对。"""
    required = {"prompt", "chosen", "rejected"}
    if not required.issubset(pair.keys()):
        return False
    prompt = str(pair.get("prompt", "")).strip()
    chosen = str(pair.get("chosen", "")).strip()
    rejected = str(pair.get("rejected", "")).strip()
    if not prompt or not chosen or not rejected:
        return False
    # chosen 与 rejected 不能相同
    if chosen == rejected:
        return False
    # 控制长度偏好：rejected 不应显著长于 chosen
    if len(rejected) > len(chosen) * 2.0:
        return False
    return True


def dpo_chosen_contains_self_reference(chosen: str) -> bool:
    """检查 chosen 是否以角色身份回答（第一人称或角色名）。"""
    markers = {"我", "咱", "本小姐", "本姑娘", "人家", "八纯"}
    return any(w in chosen for w in markers)
