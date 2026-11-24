"""全能工具註冊表（唯讀查詢 + HUD 主題控制 + 內嵌鏡頭視覺與縮放 + 本機常用程式）。"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import websockets
from google import genai
from google.genai import types

from camera_stage4.camera_controller import OnDemandCamera
from ui_stage2.theme_controller import resolve_color
from .permissions import PendingAction

Args = Dict[str, Any]
Problem = Tuple[str, str]

DEFAULT_MAX_ITEMS = 50
HARD_MAX_ITEMS = 200
_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@dataclass
class RuntimeContext:
    camera: OnDemandCamera
    ws_url: str = "ws://127.0.0.1:8766"
    gemini_client: Optional[genai.Client] = None
    gemini_model: str = ""
    on_state_change: Optional[Callable[[str, Optional[str]], None]] = None


_RUNTIME: Optional[RuntimeContext] = None


def bind_runtime(ctx: RuntimeContext) -> None:
    global _RUNTIME
    _RUNTIME = ctx


def get_runtime() -> RuntimeContext:
    if _RUNTIME is None:
        raise RuntimeError("尚未綁定執行期元件。")
    return _RUNTIME


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: Dict[str, Dict[str, str]]
    required: Tuple[str, ...]
    normalize: Callable[[Args], Tuple[Args, List[Problem]]]
    describe: Callable[[Args], PendingAction]
    run: Callable[[Args], Dict[str, Any]]

    def prepare(self, args: Args) -> Tuple[Args, List[Problem]]:
        missing = [(p, "缺少，必須由您提供") for p in self.required if _is_blank(args.get(p))]
        if missing:
            return args, missing
        return self.normalize(args)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _ws_sync_request(payload: Dict[str, Any], ws_url: str) -> Dict[str, Any]:
    async def _do() -> Dict[str, Any]:
        async with websockets.connect(ws_url, open_timeout=2.5) as ws:
            _ = await asyncio.wait_for(ws.recv(), timeout=2.0)  # 初始 sync
            await ws.send(json.dumps(payload, ensure_ascii=False))
            # 持續讀取直到收到 type == "response" 的確認封包（略過廣播封包）
            for _ in range(5):
                raw = await asyncio.wait_for(ws.recv(), timeout=2.5)
                msg = json.loads(raw)
                if msg.get("type") == "response":
                    return msg
            return {"ok": False, "message": "視覺伺服器未回傳確認封包"}

    return asyncio.run(_do())


# ───────────────────────── 1. get_current_time ─────────────────────────

def _normalize_empty(args: Args) -> Tuple[Args, List[Problem]]:
    return {}, []


def _describe_time(args: Args) -> PendingAction:
    return PendingAction(
        tool="get_current_time",
        what="讀取本機系統目前的日期與時間",
        target="本機系統時鐘",
        impact="唯讀",
    )


def _run_time(args: Args) -> Dict[str, Any]:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "weekday": _WEEKDAYS[now.weekday()],
        "utc_offset": now.strftime("%z"),
        "timezone_name": now.tzname(),
    }


# ───────────────────────── 2. list_folder_preview ─────────────────────────

def _normalize_list_folder(args: Args) -> Tuple[Args, List[Problem]]:
    problems: List[Problem] = []
    clean: Args = {}

    path = str(args["path"]).strip().strip('"').strip("'")
    if os.path.isabs(path):
        clean["path"] = os.path.normpath(path)
    else:
        problems.append((
            "path",
            "不是完整的絕對路徑（例如 C:\\Users\\你的名稱\\Desktop\\Jarvis）；不自行推測相對路徑",
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
        what="列出資料夾內容預覽",
        target=args["path"],
        impact=f"唯讀，最多列出 {args['max_items']} 筆",
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

    entries.sort(key=lambda t: (t[0], t[1], t[2]))
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


# ───────────────────────── 3. set_hud_theme ─────────────────────────

def _normalize_theme(args: Args) -> Tuple[Args, List[Problem]]:
    color = str(args["color"]).strip()
    check = resolve_color(color)
    if not check["ok"]:
        return args, [("color", check["message"])]
    return {"color": color}, []


def _describe_theme(args: Args) -> PendingAction:
    return PendingAction(
        tool="set_hud_theme",
        what=f"切換全像粒子 HUD 介面主題顏色為「{args['color']}」",
        target="本機 HUD 視覺伺服器",
        impact="更新前端粒子顏色",
    )


def _run_theme(args: Args) -> Dict[str, Any]:
    rt = get_runtime()
    try:
        res = _ws_sync_request({"action": "set_theme_color", "color_value": args["color"]}, rt.ws_url)
    except Exception as exc:
        return {"error": f"無法連線至 HUD 視覺伺服器（{exc}）"}
    if not res.get("ok"):
        return {"error": res.get("message", "主題切換失敗")}
    return {"ok": True, "message": res.get("message")}


# ───────────────────────── 4. open_camera_mode ─────────────────────────

def _describe_open_camera(args: Args) -> PendingAction:
    return PendingAction(
        tool="open_camera_mode",
        what="開啟本機攝影機並將畫面內嵌進全像 HUD 介面",
        target="本機攝影機裝置",
        impact="啟動攝影機畫面擷取",
    )


def _run_open_camera(args: Args) -> Dict[str, Any]:
    rt = get_runtime()
    ok, msg = rt.camera.open_camera()
    if rt.on_state_change:
        rt.on_state_change("CAMERA_MODE" if ok else "IDLE", msg)
    if not ok:
        return {"error": msg}
    return {"ok": True, "message": msg}


# ───────────────────────── 5. close_camera_mode ─────────────────────────

def _describe_close_camera(args: Args) -> PendingAction:
    return PendingAction(
        tool="close_camera_mode",
        what="關閉攝影機並隱藏內嵌視窗",
        target="本機攝影機裝置",
        impact="停止畫面擷取並切回 IDLE 狀態",
    )


def _run_close_camera(args: Args) -> Dict[str, Any]:
    rt = get_runtime()
    msg = rt.camera.close_camera()
    if rt.on_state_change:
        rt.on_state_change("IDLE", msg)
    return {"ok": True, "message": msg}


# ───────────────────────── 6. resize_camera_window ─────────────────────────

def _normalize_camera_scale(args: Args) -> Tuple[Args, List[Problem]]:
    raw = str(args["mode"]).strip().lower()
    mapping = {
        "larger": "larger",
        "放大": "larger",
        "大一點": "larger",
        "變大": "larger",
        "smaller": "smaller",
        "縮小": "smaller",
        "小一點": "smaller",
        "變小": "smaller",
        "reset": "reset",
        "預設": "reset",
        "恢復": "reset",
    }
    val = mapping.get(raw)
    if not val:
        return args, [("mode", "請指定放大（larger）、縮小（smaller）或恢復預設（reset）")]
    return {"mode": val}, []


def _describe_camera_scale(args: Args) -> PendingAction:
    return PendingAction(
        tool="resize_camera_window",
        what=f"調整全像介面內嵌鏡頭視窗大小（{args['mode']}）",
        target="HUD 內嵌鏡頭視窗",
        impact="變更鏡頭預覽視窗顯示比例",
    )


def _run_camera_scale(args: Args) -> Dict[str, Any]:
    rt = get_runtime()
    try:
        res = _ws_sync_request({"action": "camera_scale", "mode": args["mode"]}, rt.ws_url)
    except Exception as exc:
        return {"error": f"無法調整鏡頭視窗大小（{exc}）"}
    if not res.get("ok"):
        return {"error": res.get("message", "調整失敗")}
    return {"ok": True, "message": res.get("message")}


# ───────────────────────── 7. analyze_camera_view ─────────────────────────

def _normalize_vision(args: Args) -> Tuple[Args, List[Problem]]:
    return {"question": str(args["question"]).strip()}, []


def _describe_vision(args: Args) -> PendingAction:
    return PendingAction(
        tool="analyze_camera_view",
        what="擷取目前攝影機單張畫面並進行視覺分析",
        target="目前攝影機畫面 -> Gemini API",
        impact="傳送一張即時影像以回答問題",
    )


def _run_vision(args: Args) -> Dict[str, Any]:
    rt = get_runtime()
    if not rt.camera.active:
        return {"error": "目前攝影機尚未開啟，請先開啟鏡頭溝通模式。"}
    jpeg = rt.camera.capture_jpeg_bytes(max_size=(512, 384), quality=72)
    if jpeg is None:
        return {"error": "無法取得目前鏡頭畫面，請確認攝影機是否正常運作。"}
    if rt.gemini_client is None or not rt.gemini_model:
        return {"error": "尚未設定視覺分析模型。"}

    resp = rt.gemini_client.models.generate_content(
        model=rt.gemini_model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"請用簡短繁體中文直接回答：{args['question']}"),
                    types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                ],
            )
        ],
    )
    return {"ok": True, "vision_answer": (resp.text or "").strip()}


# ───────────────────────── 8. launch_local_app ─────────────────────────

_ALLOWED_APPS = {
    "notepad": ("notepad.exe", "記事本"),
    "calc": ("calc.exe", "小算盤"),
}


def _normalize_launch_app(args: Args) -> Tuple[Args, List[Problem]]:
    raw = str(args["app"]).strip().lower()
    alias = {
        "notepad": "notepad",
        "記事本": "notepad",
        "calc": "calc",
        "小算盤": "calc",
    }
    key = alias.get(raw)
    if not key:
        return args, [("app", "目前僅支援「記事本（notepad）」或「小算盤（calc）」，請明確指定")]
    return {"app": key}, []


def _describe_launch_app(args: Args) -> PendingAction:
    exe, label = _ALLOWED_APPS[args["app"]]
    return PendingAction(
        tool="launch_local_app",
        what=f"啟動本機桌面應用程式「{label}」",
        target=exe,
        impact="開啟應用程式視窗",
    )


def _run_launch_app(args: Args) -> Dict[str, Any]:
    exe, label = _ALLOWED_APPS[args["app"]]
    subprocess.Popen([exe], creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    return {"ok": True, "message": f"已啟動{label}（{exe}）。"}


# ───────────────────────── 工具註冊表 ─────────────────────────

TOOLS: Dict[str, ToolSpec] = {
    "get_current_time": ToolSpec(
        name="get_current_time",
        description="取得本機電腦系統目前的日期與時間（本機時區）。",
        params={},
        required=(),
        normalize=_normalize_empty,
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
    "set_hud_theme": ToolSpec(
        name="set_hud_theme",
        description="切換賈維斯全像粒子 HUD 介面的主題顏色（支援海洋藍、紅色、金色、紫色、綠色或 #RRGGBB 色碼）。",
        params={
            "color": {
                "type": "STRING",
                "description": "目標顏色名稱（海洋藍、紅色、金色、紫色、綠色）或 #RRGGBB 色碼",
            }
        },
        required=("color",),
        normalize=_normalize_theme,
        describe=_describe_theme,
        run=_run_theme,
    ),
    "open_camera_mode": ToolSpec(
        name="open_camera_mode",
        description="開啟本機攝影機並將畫面內嵌顯示於全像 HUD 視窗，進入鏡頭溝通模式。",
        params={},
        required=(),
        normalize=_normalize_empty,
        describe=_describe_open_camera,
        run=_run_open_camera,
    ),
    "close_camera_mode": ToolSpec(
        name="close_camera_mode",
        description="關閉本機攝影機並隱藏內嵌視窗，結束鏡頭溝通模式。",
        params={},
        required=(),
        normalize=_normalize_empty,
        describe=_describe_close_camera,
        run=_run_close_camera,
    ),
    "resize_camera_window": ToolSpec(
        name="resize_camera_window",
        description="將全像 HUD 介面中的內嵌鏡頭視窗放大（larger）、縮小（smaller）或恢復預設大小（reset）。",
        params={
            "mode": {
                "type": "STRING",
                "description": "縮放方向：larger（放大）、smaller（縮小）或 reset（恢復預設）",
            }
        },
        required=("mode",),
        normalize=_normalize_camera_scale,
        describe=_describe_camera_scale,
        run=_run_camera_scale,
    ),
    "analyze_camera_view": ToolSpec(
        name="analyze_camera_view",
        description="在鏡頭溝通模式開啟狀態下，擷取目前鏡頭畫面並分析畫面內容以回答使用者問題。",
        params={
            "question": {
                "type": "STRING",
                "description": "使用者針對目前鏡頭畫面想詢問的問題",
            }
        },
        required=("question",),
        normalize=_normalize_vision,
        describe=_describe_vision,
        run=_run_vision,
    ),
    "launch_local_app": ToolSpec(
        name="launch_local_app",
        description="啟動本機常用應用程式（目前支援：記事本 notepad、小算盤 calc）。",
        params={
            "app": {
                "type": "STRING",
                "description": "應用程式名稱：記事本（notepad）或小算盤（calc）",
            }
        },
        required=("app",),
        normalize=_normalize_launch_app,
        describe=_describe_launch_app,
        run=_run_launch_app,
    ),
}