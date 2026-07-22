#!/usr/bin/env python3
"""Upload Hachizumi dataset and models to ModelScope.

Usage:
    MODELSCOPE_API_TOKEN=ms-xxxx uv run python scripts/upload_to_modelscope.py

This script creates the required ModelScope repositories (if they do not exist)
and uploads the local dataset/model folders.  Uploads use the ModelScope CLI
upload cache so they can resume if interrupted.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_OWNER = "fyerfyer"

MIT_LICENSE = """MIT License

Copyright (c) 2026 fyerfyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def dataset_readme() -> str:
    return """# Hachizumi Chat Dataset

本数据集用于训练「天满八纯（Hachizumi）」角色扮演模型。

## 内容

- `character_card_hasumi.json`：结构化角色卡
- `system_prompt.txt`：去关系化的 system prompt 模板
- `processed/scenes.jsonl`：从原始剧本解析出的场景
- `synthetic_sft.jsonl`：API 合成的 SFT 样本
- `sft_train.jsonl` / `sft_test.jsonl`：SFT 训练集与测试集
- `sft_train_cleaned.jsonl`：经风格清洗后的训练集
- `dpo_train.jsonl`：DPO 偏好对

## 许可

本项目数据以 MIT 协议发布，可公开使用。
"""


def merged_model_readme() -> str:
    return """# Hachizumi Qwen2.5-7B Merged v3

天满八纯（Hachizumi）角色扮演模型，基于 Qwen2.5-7B-Instruct 进行 SFT + DPO 后合并得到的完整 Hugging Face Transformers 模型。

## 模型信息

- Base: Qwen/Qwen2.5-7B-Instruct
- 微调: SFT (LoRA) + DPO (LoRA)，合并为完整权重
- 格式: Hugging Face Transformers (safetensors)
- 精度: bfloat16

## 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "fyerfyer/hachizumi-qwen25-7b-merged-v3",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("fyerfyer/hachizumi-qwen25-7b-merged-v3")

messages = [
    {"role": "system", "content": "你是天满八纯。你性格友好又有点调皮，好奇心旺盛，偶尔带点小恶魔/腹黑。说话活泼直接、轻快自然，常用「诶——」「真是的」「没办法呢」「哟呵呵」「嘿嘿」「嘛」「呢」「啦」等语气词。请用符合你性格的语气回答面前的人，不要默认对方是某个特定角色。"},
    {"role": "user", "content": "你是谁？"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", return_dict=True)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 许可

本项目以 MIT 协议发布，可公开使用。模型基座 Qwen2.5-7B-Instruct 遵循其原始许可。
"""


def sft_lora_readme() -> str:
    return """# Hachizumi Qwen2.5-7B SFT LoRA v3

天满八纯（Hachizumi）角色扮演模型的 Stage-1 SFT LoRA adapter。

## 模型信息

- Base: Qwen/Qwen2.5-7B-Instruct-bnb-4bit
- 任务: 全量监督微调 (SFT)
- 格式: PEFT LoRA adapter (safetensors)

## 使用方式

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "fyerfyer/hachizumi-qwen25-7b-sft-lora-v3")
tokenizer = AutoTokenizer.from_pretrained("fyerfyer/hachizumi-qwen25-7b-sft-lora-v3")
```

## 许可

本项目以 MIT 协议发布，可公开使用。模型基座 Qwen2.5-7B-Instruct 遵循其原始许可。
"""


def dpo_lora_readme() -> str:
    return """# Hachizumi Qwen2.5-7B DPO LoRA v3

天满八纯（Hachizumi）角色扮演模型的 Stage-2 DPO LoRA adapter，在 SFT LoRA v3 基础上继续 DPO 对齐。

## 模型信息

- Base: Qwen/Qwen2.5-7B-Instruct-bnb-4bit
- 任务: 直接偏好优化 (DPO)
- 格式: PEFT LoRA adapter (safetensors)

## 使用方式

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "fyerfyer/hachizumi-qwen25-7b-dpo-lora-v3")
tokenizer = AutoTokenizer.from_pretrained("fyerfyer/hachizumi-qwen25-7b-dpo-lora-v3")
```

## 许可

本项目以 MIT 协议发布，可公开使用。模型基座 Qwen2.5-7B-Instruct 遵循其原始许可。
"""


def gguf_model_readme() -> str:
    return """# Hachizumi Qwen2.5-7B GGUF v3

天满八纯（Hachizumi）角色扮演模型的 GGUF 量化版本，便于本地推理（Ollama、llama.cpp 等）。

## 模型信息

- Base: Qwen/Qwen2.5-7B-Instruct
- 来源: `fyerfyer/hachizumi-qwen25-7b-merged-v3`
- 量化版本:
  - `q4_k_m/`：Q4_K_M，平衡速度与精度
  - `q5_k_m/`：Q5_K_M，更高精度
  - `q8_0/`：Q8_0，接近原始精度

## Ollama 使用

每个量化目录下均附带 `Modelfile`，可直接：

```bash
ollama create hachizumi-qwen25-7b-q4_k_m -f q4_k_m/Modelfile
ollama run hachizumi-qwen25-7b-q4_k_m
```

## 许可

本项目以 MIT 协议发布，可公开使用。模型基座 Qwen2.5-7B-Instruct 遵循其原始许可。
"""


@dataclass(frozen=True)
class RepoDef:
    repo_id: str
    repo_type: str
    local_path: Path
    description: str
    readme_text: str
    license: str = "mit"
    exclude: tuple[str, ...] = ()


def get_repos(project_root: Path) -> list[RepoDef]:
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    return [
        RepoDef(
            repo_id=f"{REPO_OWNER}/hachizumi-chat-dataset",
            repo_type="dataset",
            local_path=data_dir,
            description="天满八纯（Hachizumi）角色扮演模型的训练数据集",
            readme_text=dataset_readme(),
            exclude=("*.bak", "raw", "synthetic"),
        ),
        RepoDef(
            repo_id=f"{REPO_OWNER}/hachizumi-qwen25-7b-sft-lora-v3",
            repo_type="model",
            local_path=outputs_dir / "sft_qwen25_7b_hasumi_lora_v3",
            description="天满八纯角色模型 Stage-1 SFT LoRA adapter",
            readme_text=sft_lora_readme(),
        ),
        RepoDef(
            repo_id=f"{REPO_OWNER}/hachizumi-qwen25-7b-dpo-lora-v3",
            repo_type="model",
            local_path=outputs_dir / "dpo_qwen25_7b_hasumi_lora_v3",
            description="天满八纯角色模型 Stage-2 DPO LoRA adapter",
            readme_text=dpo_lora_readme(),
        ),
        RepoDef(
            repo_id=f"{REPO_OWNER}/hachizumi-qwen25-7b-merged-v3",
            repo_type="model",
            local_path=outputs_dir / "hasumi_qwen25_7b_merged_v3",
            description="天满八纯角色模型完整合并版 (SFT + DPO)",
            readme_text=merged_model_readme(),
        ),
        RepoDef(
            repo_id=f"{REPO_OWNER}/hachizumi-qwen25-7b-gguf-v3",
            repo_type="model",
            local_path=outputs_dir / "hasumi_qwen25_7b_gguf_v3",
            description="天满八纯角色模型 GGUF 量化版本",
            readme_text=gguf_model_readme(),
        ),
    ]


def ensure_metadata(local_path: Path, readme_text: str, license_text: str) -> None:
    """Write README.md and LICENSE into the local folder if missing."""
    readme_path = local_path / "README.md"
    if not readme_path.exists():
        readme_path.write_text(readme_text, encoding="utf-8")
        print(f"  Wrote {readme_path}")

    license_path = local_path / "LICENSE"
    if not license_path.exists():
        license_path.write_text(license_text, encoding="utf-8")
        print(f"  Wrote {license_path}")


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and stream its output."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=full_env, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def create_repo(repo: RepoDef, token: str) -> bool:
    cmd = [
        "uv", "run", "modelscope", "create",
        "--repo-type", repo.repo_type,
        "--visibility", "public",
        "--license", repo.license,
        "--description", repo.description,
        "--exist-ok",
        repo.repo_id,
    ]
    result = run_cmd(cmd, env={"MODELSCOPE_API_TOKEN": token})
    if result.returncode != 0 and "already exists" not in (result.stdout + result.stderr):
        print(f"[ERROR] Failed to create {repo.repo_id}", file=sys.stderr)
        return False
    return True


def upload_repo(repo: RepoDef, token: str, *, max_workers: int = 4, disable_tqdm: bool = False) -> bool:
    cmd = [
        "uv", "run", "modelscope", "upload",
        "--repo-type", repo.repo_type,
        "--commit-message", "Initial upload of Hachizumi project artifacts",
        "--use-cache",
        "--max-workers", str(max_workers),
        repo.repo_id,
        str(repo.local_path),
    ]
    for pattern in repo.exclude:
        cmd.extend(["--exclude", pattern])
    if disable_tqdm:
        cmd.append("--disable-tqdm")

    result = run_cmd(cmd, env={"MODELSCOPE_API_TOKEN": token})
    return result.returncode == 0


def main() -> int:
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        print("ERROR: MODELSCOPE_API_TOKEN is not set.", file=sys.stderr)
        return 1

    project_root = Path(__file__).resolve().parent.parent
    repos = get_repos(project_root)

    print(f"Project root: {project_root}")
    print(f"Will upload {len(repos)} repositories:\n  " + "\n  ".join(r.repo_id for r in repos))
    print()

    for repo in repos:
        print(f"\n=== {repo.repo_id} ({repo.repo_type}) ===")
        if not repo.local_path.exists():
            print(f"[SKIP] Local path does not exist: {repo.local_path}")
            continue

        ensure_metadata(repo.local_path, repo.readme_text, MIT_LICENSE)

        if not create_repo(repo, token):
            print(f"[SKIP] Could not ensure repo exists: {repo.repo_id}")
            continue

        print(f"Uploading {repo.local_path} ...")
        if upload_repo(repo, token, disable_tqdm=False):
            print(f"[OK] {repo.repo_id}")
        else:
            print(f"[FAIL] {repo.repo_id}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
