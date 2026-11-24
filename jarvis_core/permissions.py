"""權限確認機制。

規則：
- 任何工具執行前，都必須先顯示「待允許動作說明」，使用者明確輸入/說出「允許/同意/執行」才放行。
- 輸入/說出「拒絕/取消/不要…」、按 Ctrl+C 或 Ctrl+Z+Enter，立即中止整個回合。
- 輸入內容無法判斷時，重新詢問，絕不當成同意。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

APPROVE_WORDS = {"允許", "同意", "執行", "好", "好的", "可以", "確定"}
DENY_WORDS = {"拒絕", "取消", "不允許", "不同意", "不要", "否", "中止", "算了"}


class UserAbort(Exception):
    """使用者拒絕或取消，需立即中止目前回合。"""


@dataclass(frozen=True)
class PendingAction:
    tool: str      # 工具名稱
    what: str      # 要做什麼
    target: str    # 目標是什麼
    impact: str    # 預期影響


def format_pending(action: PendingAction) -> str:
    line = "─" * 18
    return (
        f"\n{line} 待允許動作 {line}\n"
        f"工具  ：{action.tool}\n"
        f"要做什麼：{action.what}\n"
        f"目標  ：{action.target}\n"
        f"預期影響：{action.impact}\n"
        f"{line * 2}────"
    )


def _read(prompt: str, input_fn: Optional[Callable[[str], str]]) -> str:
    fn = input_fn or input
    try:
        return fn(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise UserAbort()


def ask_permission(action: PendingAction, input_fn: Optional[Callable[[str], str]] = None) -> None:
    """通過就直接返回；拒絕則丟出 UserAbort。"""
    print(format_pending(action))
    prompt = f"即將執行【{action.what}】，目標：{action.target}。請確認「允許」或「拒絕」："
    while True:
        answer = _read(prompt, input_fn)
        if any(w == answer or w in answer for w in DENY_WORDS):
            raise UserAbort()
        if any(w == answer or w in answer for w in APPROVE_WORDS):
            return
        print("無法判斷您的意思，請明確回覆「允許」「同意」「執行」或「拒絕」「取消」。")


def ask_value(
    param: str,
    description: str,
    problem: str,
    input_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """參數缺漏或有問題時，直接問使用者，不猜測。輸入「取消」則中止。"""
    print(f"\n⚠ 參數「{param}」{problem}")
    print(f"  說明：{description}")
    prompt = f"參數「{param}」{problem}（{description}），請提供明確值或說「取消」："
    while True:
        answer = _read(prompt, input_fn)
        if any(w == answer or w in answer for w in DENY_WORDS):
            raise UserAbort()
        if answer:
            return answer