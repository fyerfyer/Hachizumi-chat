"""OpenAI-compatible LLM API 的轻量封装。

默认支持 DeepSeek、Kimi、Kimi Code 等任意兼容 OpenAI /chat/completions 的服务。
通过环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 配置。
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from . import config
from .utils import extract_json

logger = logging.getLogger(__name__)


class LLMAPIError(Exception):
    """LLM API 调用失败。"""

    pass


class LLMClient:
    """OpenAI-compatible LLM API 客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        rpm: int = config.LLM_RPM,
        timeout: int = config.LLM_TIMEOUT,
    ):
        self.api_key = api_key or config.LLM_API_KEY
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout
        self.min_interval = 60.0 / max(rpm, 1)
        self._last_call = 0.0

        if not self.api_key:
            raise LLMAPIError(
                "LLM_API_KEY 未设置。请先执行：export LLM_API_KEY=sk-..."
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _rate_limit_wait(self) -> None:
        """串行调用时的简单 RPM 限流。"""
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送一次请求，含指数退避重试。"""
        url = f"{self.base_url}/chat/completions"
        for attempt in range(config.LLM_MAX_RETRIES):
            self._rate_limit_wait()
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                self._last_call = time.time()
                if resp.status_code == 200:
                    return resp.json()
                # 限流时退避
                if resp.status_code in (429, 503):
                    wait = 2 ** attempt
                    logger.warning(f"API 限流/服务不可用，{wait}s 后重试...")
                    time.sleep(wait)
                    continue
                raise LLMAPIError(
                    f"HTTP {resp.status_code}: {resp.text[:500]}"
                )
            except requests.exceptions.Timeout:
                logger.warning(f"请求超时，第 {attempt + 1} 次重试...")
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常：{e}，第 {attempt + 1} 次重试...")
                time.sleep(2 ** attempt)
        raise LLMAPIError("超过最大重试次数，API 调用失败。")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """单次对话。"""
        # kimi-for-coding 只支持 temperature=1
        effective_temp = 1.0 if self.model == "kimi-for-coding" else temperature
        if effective_temp != temperature:
            logger.debug("当前模型只支持 temperature=1，已自动调整。")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": effective_temp,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = self._post(payload)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})
        logger.info(
            "API 调用 tokens: prompt=%s, completion=%s",
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )
        return {"content": content, "usage": usage}

    def batch_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        desc: str = "API batch",
        temperature: float = 1.0,
        max_tokens: int = 512,
        json_mode: bool = False,
        max_workers: int = 4,
    ) -> List[Optional[Dict[str, Any]]]:
        """批量调用；max_workers 控制并发，默认 4。"""
        if max_workers <= 1:
            results: List[Optional[Dict[str, Any]]] = []
            for messages in tqdm(messages_list, desc=desc):
                try:
                    res = self.chat(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
                    results.append(res)
                except LLMAPIError as e:
                    logger.error("单条 API 失败：%s", e)
                    results.append(None)
            return results

        results = [None] * len(messages_list)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self.chat,
                    messages,
                    temperature,
                    max_tokens,
                    json_mode,
                ): idx
                for idx, messages in enumerate(messages_list)
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(messages_list),
                desc=desc,
            ):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except LLMAPIError as e:
                    logger.error("单条 API 失败：%s", e)
                    results[idx] = None
        return results


# 保留旧别名，兼容历史代码
KimiAPIError = LLMAPIError
KimiClient = LLMClient


def test_one():
    """快速测试 API 连通性。"""
    client = LLMClient()
    res = client.chat(
        messages=[
            {"role": "system", "content": "你是一个 helpful 的助手。"},
            {"role": "user", "content": "用 JSON 输出 {\"hello\": \"world\"}"},
        ],
        json_mode=True,
    )
    parsed = extract_json(res["content"])
    print(parsed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_one()
