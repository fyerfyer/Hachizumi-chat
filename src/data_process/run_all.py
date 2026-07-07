"""数据构建层一键流水线。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import config
from .llm_client import LLMClient
from .build_dpo_dataset import main as build_dpo_main
from .build_sft_dataset import generate_synthetic_sft, main as build_sft_main
from .character_card_builder import main as build_character_main
from .parse_scenario import main as parse_main
from .utils import setup_logging

logger = logging.getLogger(__name__)


def run_all(
    skip_api: bool = False,
    synthetic_sft: int = 0,
    dpo_pairs: int = 0,
    max_workers: int = 4,
):
    setup_logging(config.LOG_DIR / "pipeline.log")
    logger.info("=" * 50)
    logger.info("天满八纯角色模型 — 数据构建层流水线")
    logger.info("=" * 50)

    # 1. 解析剧本
    parse_stats = parse_main()

    # 2. 构建角色卡
    card = build_character_main()

    # 3. 合成 SFT（可选，需 API）
    synthetic_path: Path | None = None
    if not skip_api and synthetic_sft > 0:
        try:
            client = LLMClient()
            synthetic_path = generate_synthetic_sft(
                client,
                card,
                n=synthetic_sft,
                batch_size=5,
                max_workers=max_workers,
            )
        except Exception as e:
            logger.exception("合成 SFT 失败：%s", e)
            synthetic_path = None

    # 4. 构建 SFT 数据集
    build_sft_main(synthetic_path=synthetic_path)

    # 5. 构建 DPO 数据集
    target = dpo_pairs if dpo_pairs > 0 else config.DPO_DEFAULT_PAIRS
    build_dpo_main(
        skip_api=skip_api,
        target_pairs=target,
        pair_batch_size=5,
        para_batch_size=10,
        max_workers=max_workers,
    )

    logger.info("=" * 50)
    logger.info("数据构建层流水线完成。")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="天满八纯数据构建层")
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="跳过所有 Kimi API 调用，仅用原剧本生成数据",
    )
    parser.add_argument(
        "--synthetic-sft",
        type=int,
        default=0,
        help="通过 API 合成的 SFT 样本数量（默认 0）",
    )
    parser.add_argument(
        "--dpo-pairs",
        type=int,
        default=0,
        help="目标 DPO 偏好对数量（默认 1500）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="API 并发数（默认 4）",
    )
    args = parser.parse_args()

    synthetic_sft = args.synthetic_sft if args.synthetic_sft > 0 else 0
    dpo_pairs = args.dpo_pairs if args.dpo_pairs > 0 else config.DPO_DEFAULT_PAIRS

    run_all(
        skip_api=args.skip_api,
        synthetic_sft=synthetic_sft,
        dpo_pairs=dpo_pairs,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
