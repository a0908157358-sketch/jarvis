"""讀取設定。

鐵律：缺少任何必要設定就明確報錯並請使用者補上，不使用猜測的預設值。
（唯一例外：JARVIS_MAX_STEPS 為安全上限，預設 8，可在 .env 覆寫。）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

_PLACEHOLDERS = {"", "your_api_key_here", "your_model_name_here"}
_DEFAULT_MAX_STEPS = 8


class ConfigError(Exception):
    """設定缺漏或格式錯誤。"""


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    max_steps: int


def load_config() -> Config:
    if ENV_PATH.is_file():
        # utf-8-sig：相容 Windows 記事本可能加上的 BOM
        load_dotenv(ENV_PATH, encoding="utf-8-sig")

    problems: list[str] = []

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key in _PLACEHOLDERS:
        problems.append("GEMINI_API_KEY：Gemini API 金鑰")

    model = os.getenv("GEMINI_MODEL", "").strip()
    if model in _PLACEHOLDERS:
        problems.append("GEMINI_MODEL：要使用的 Gemini 模型名稱")

    max_steps = _DEFAULT_MAX_STEPS
    raw_steps = os.getenv("JARVIS_MAX_STEPS", "").strip()
    if raw_steps:
        if raw_steps.isdigit() and int(raw_steps) >= 1:
            max_steps = int(raw_steps)
        else:
            problems.append("JARVIS_MAX_STEPS：必須是 1 以上的整數（或整行刪除以使用預設值 8）")

    if problems:
        hint = "" if ENV_PATH.is_file() else (
            f"\n（找不到 {ENV_PATH}，請先把 .env.example 複製成 .env）"
        )
        raise ConfigError(
            "設定不完整，請編輯 .env 補上下列項目；程式不會替你猜測這些值：\n"
            + "\n".join(f"  - {p}" for p in problems)
            + hint
        )

    return Config(api_key=api_key, model=model, max_steps=max_steps)
