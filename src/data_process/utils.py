"""通用工具：JSON 提取、日志配置。"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any


def extract_json(text: str) -> Any | None:
    """从可能包含 markdown 围栏或多余文字的文本中提取 JSON。"""
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # 尝试从第一个 [ 或 { 开始解析
    for start_char in ("[", "{"):
        idx = text.find(start_char)
        if idx == -1:
            continue
        candidate = text[idx:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # 兜底：直接尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def setup_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """配置日志同时输出到文件和控制台。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("hachizumi")
