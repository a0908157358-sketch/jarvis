"""安全測試工具。

兩個工具都是唯讀：
- get_current_time：讀取本機系統時間
- list_folder_preview：列出指定資料夾的檔名 / 類型 / 大小（不讀檔案內容）

工具本身不負責詢問權限；權限由 agent 在呼叫 run() 之前統一處理。
工具也不會在使用者允許前碰檔案系統，因此路徑是否存在等檢查都放在 run() 內。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from .permissions import PendingAction

Args = Dict[str, Any]
Problem = Tuple[str, str]  # (參數名稱, 問題說明)

DEFAULT_MAX_ITEMS = 50
HARD_MAX_ITEMS = 200
_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str                                   # 給模型看的說明
    params: Dict[str, Dict[str, str]]                  # {"path": {"type": "STRING", "description": "..."}}
    required: Tuple[str, ...]
    normalize: Callable[[Args], Tuple[Args, List[Problem]]]
    describe: Callable[[Args], PendingAction]
    run: Callable[[Args], Dict[str, Any]]

    def prepare(self, args: Args) -> Tuple[Args, List[Problem]]:
        """檢查必要參數 → 格式檢查與正規化。回傳 (乾淨參數, 問題清單)。"""
        missing = [(p, "缺少，必須由您提供") for p in self.required if _is_blank(args.get(p))]
        if missing:
            return args, missing
        return self.normalize(args)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


# ───────────────────────── get_current_time ─────────────────────────

def _normalize_time(args: Args) -> Tuple[Args, List[Problem]]:
    return {}, []


def _describe_time(args: Args) -> PendingAction:
    return PendingAction(
        tool="get_current_time",
        what="讀取本機系統目前的日期與時間",
        target="本機系統時鐘（不連網）",
        impact="唯讀，不修改任何設定或檔案；讀到的時間會傳送給 Gemini API 以產生回答",
    )


def _run_time(args: Args) -> Dict[str, Any]:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "weekday": _WEEKDAYS[now.weekday()],
        "utc_offset": now.strftime("%z"),
        "timezone_name": now.tzname(),
    }


# ───────────────────────── list_folder_preview ─────────────────────────

def _normalize_list_folder(args: Args) -> Tuple[Args, List[Problem]]:
    problems: List[Problem] = []
    clean: Args = {}

    path = str(args["path"]).strip().strip('"').strip("'")
    if os.path.isabs(path):
        clean["path"] = os.path.normpath(path)
    else:
        problems.append((
            "path",
            "不是完整的絕對路徑（例如 C:\\Users\\你的名稱\\Desktop\\Jarvis）；本程式不會自行推測相對路徑或 ~",
        ))

    raw = args.get("max_items")
    if _is_blank(raw):
        clean["max_items"] = DEFAULT_MAX_ITEMS
    else:
        try:
            f = float(raw)
            n = int(f) if f.is_integer() else None
        except (TypeError, ValueError):
            n = None
        if n is None or not 1 <= n <= HARD_MAX_ITEMS:
            problems.append(("max_items", f"必須是 1 到 {HARD_MAX_ITEMS} 的整數"))
        else:
            clean["max_items"] = n

    return clean, problems


def _describe_list_folder(args: Args) -> PendingAction:
    return PendingAction(
        tool="list_folder_preview",
        what="列出資料夾內容預覽（只讀取名稱、類型、大小，不讀取任何檔案內容）",
        target=args["path"],
        impact=(
            f"唯讀，不建立、修改或刪除任何檔案；最多列出 {args['max_items']} 筆；"
            "列出的名稱會傳送給 Gemini API 以產生回答"
        ),
    )


def _run_list_folder(args: Args) -> Dict[str, Any]:
    path, limit = args["path"], args["max_items"]
    if not os.path.exists(path):
        return {"error": f"路徑不存在：{path}"}
    if not os.path.isdir(path):
        return {"error": f"這不是資料夾：{path}"}

    entries = []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    is_dir = e.is_dir(follow_symlinks=False)
                except OSError:
                    is_dir = False
                size = None
                if not is_dir:
                    try:
                        size = e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
                entries.append((not is_dir, e.name.lower(), e.name, is_dir, size))
    except PermissionError:
        return {"error": f"沒有權限讀取：{path}"}

    entries.sort(key=lambda t: (t[0], t[1], t[2]))  # 資料夾在前，再依名稱
    shown = entries[:limit]
    return {
        "path": path,
        "total": len(entries),
        "shown": len(shown),
        "truncated": len(entries) > len(shown),
        "items": [
            {"name": name, "type": "folder" if is_dir else "file", "size_bytes": size}
            for _, _, name, is_dir, size in shown
        ],
    }


# ───────────────────────── 工具註冊表 ─────────────────────────

TOOLS: Dict[str, ToolSpec] = {
    "get_current_time": ToolSpec(
        name="get_current_time",
        description="取得本機電腦系統目前的日期與時間（本機時區）。",
        params={},
        required=(),
        normalize=_normalize_time,
        describe=_describe_time,
        run=_run_time,
    ),
    "list_folder_preview": ToolSpec(
        name="list_folder_preview",
        description=(
            "列出指定資料夾內的檔案與子資料夾（名稱、類型、大小），不讀取檔案內容。"
            "path 必須由使用者提供完整絕對路徑，不明確時請先詢問使用者，不可自行猜測。"
        ),
        params={
            "path": {
                "type": "STRING",
                "description": "要預覽的資料夾完整絕對路徑，例如 C:\\Users\\Name\\Desktop\\Jarvis",
            },
            "max_items": {
                "type": "INTEGER",
                "description": f"最多列出幾筆，1 到 {HARD_MAX_ITEMS}；未指定時為 {DEFAULT_MAX_ITEMS}。",
            },
        },
        required=("path",),
        normalize=_normalize_list_folder,
        describe=_describe_list_folder,
        run=_run_list_folder,
    ),
}
