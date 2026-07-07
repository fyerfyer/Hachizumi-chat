"""GPU 自动选择工具，避免与服务器其他任务争抢显存。"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> list[dict]:
    """返回每张卡的显存使用信息（单位：MiB）。"""
    try:
        import pynvml
    except ImportError as e:
        raise ImportError("请安装 nvidia-ml-py 或 pynvml 以支持 GPU 自动选择") from e

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    info = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        info.append(
            {
                "id": i,
                "name": name,
                "total_mib": mem.total // (1024 * 1024),
                "used_mib": mem.used // (1024 * 1024),
                "free_mib": mem.free // (1024 * 1024),
            }
        )
    return info


def select_best_gpu(preferred_id: Optional[int] = None) -> int:
    """选择显存占用最低的 GPU；如果指定 preferred_id 且该卡可用，则优先使用。"""
    infos = get_gpu_memory_info()
    if not infos:
        raise RuntimeError("未检测到 NVIDIA GPU")

    logger.info("当前 GPU 状态：")
    for g in infos:
        logger.info(
            "  GPU %d: %s, 已用 %dMiB / 共 %dMiB",
            g["id"],
            g["name"],
            g["used_mib"],
            g["total_mib"],
        )

    if preferred_id is not None and 0 <= preferred_id < len(infos):
        logger.info("使用用户指定 GPU %d", preferred_id)
        return preferred_id

    # 优先选择空闲显存最多的卡
    best = min(infos, key=lambda g: g["used_mib"])
    logger.info(
        "自动选择 GPU %d（%s，已用 %dMiB）",
        best["id"],
        best["name"],
        best["used_mib"],
    )
    return best["id"]


def setup_cuda_visible_devices(
    gpu_id: Optional[int] = None,
) -> Tuple[int, list[dict]]:
    """设置 CUDA_VISIBLE_DEVICES 并返回选中的 GPU id 与全部 GPU 信息。"""
    infos = get_gpu_memory_info()
    selected = select_best_gpu(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected)
    # 让 torch 在多进程时只使用可见 GPU
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return selected, infos


def add_gpu_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="指定使用哪张 GPU（默认自动选择显存占用最低的卡）",
    )
