from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.genai import types
import websockets

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from camera_stage4.camera_controller import OnDemandCamera
from system_stage5.system_tools import WorkspaceSystemController
from ui_stage2.theme_controller import extract_theme_command
from voice_stage3.stt_tts import ChineseVoiceEngine, create_gemini_client

END_SESSION_PHRASES = ("結束對話", "退下", "休息一下", "先這樣", "結束")
CONFIRM_YES_PHRASES = ("好", "好的", "可以", "同意", "允許", "執行", "確定", "沒問題", "好啊", "行")
CONFIRM_NO_PHRASES = ("不要", "不行", "取消", "拒絕", "算了", "先不要")

OPEN_CAMERA_PHRASES = ("開啟鏡頭溝通模式", "開始鏡頭溝通模式", "開鏡頭溝通模式", "打開鏡頭溝通模式")
CLOSE_CAMERA_PHRASES = ("結束鏡頭溝通模式", "關閉鏡頭溝通模式", "關鏡頭溝通模式")

SYSTEM_INSTRUCTION = """
你是「賈維斯」，一位使用繁體中文溝通的本機全能電腦助理。
請嚴格遵守：
1. 缺少資訊或指令不明確時，直接提問確認，嚴禁猜測或腦補。
2. 任何涉及系統、檔案、程式、視窗、攝影機或外部影響的操作，都必須先清楚說明動作、目標與預期影響，取得使用者明確允許後才執行。
3. 回覆使用自然、清楚的繁體中文，避免過多特殊符號。
"""


def normalize_text(text: str) -> str:
    return re.sub(r"[。！!？\?，,、\.…\s]+$", "", (text or "").strip())


def detect_wake_and_inline(transcript: str) -> tuple[bool, str]:
    cleaned = normalize_text(transcript)
    compact = cleaned.replace(" ", "")
    if not compact:
        return False, ""

    if "賈維斯" in compact:
        idx = compact.find("賈維斯")
        after = re.sub(r"^[，,、\s]+", "", compact[idx + 3 :])
        return True, after

    head = compact[:5]
    for start in range(len(head)):
        window = head[start : start + 3]
        if len(window) >= 2 and SequenceMatcher(None, window, "賈維斯").ratio() >= 0.55:
            after = re.sub(r"^[，,、\s]+", "", compact[start + len(window) :])
            return True, after

    return False, ""


class JarvisAssistant:
    def __init__(self, ws_url: str = "ws://127.0.0.1:8766") -> None:
        load_dotenv(ROOT_DIR / ".env")
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("缺少必填環境變數 GEMINI_API_KEY，請先在 .env 中設定。")

        self.model_name = os.getenv("GEMINI_MODEL", "").strip()
        if not self.model_name:
            raise RuntimeError("缺少必填環境變數 GEMINI_MODEL，請先在 .env 中明確指定模型名稱。")

        self.ws_url = ws_url
        self.client = create_gemini_client(api_key)

        self.voice = ChineseVoiceEngine(
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small").strip() or "small",
            device=os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu",
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8",
            tts_voice=os.getenv("TTS_VOICE", "zh-TW-YunJheNeural").strip() or "zh-TW-YunJheNeural",
        )

        ws_root_raw = os.getenv("WORKSPACE_ROOT", str(ROOT_DIR)).strip()
        self.system = WorkspaceSystemController(Path(ws_root_raw).resolve())

        camera_idx = int(os.getenv("CAMERA_INDEX", "0").strip() or "0")
        self.camera = OnDemandCamera(
            camera_index=camera_idx,
            window_width=420,
            window_height=315,
            on_external_close=self._on_camera_closed_externally,
        )

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.in_conversation: bool = False
        self.idle_turns: int = 0
        self.pending_action: Optional[Dict[str, Any]] = None
        self.history: List[types.Content] = []

    def _on_camera_closed_externally(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.push_state("IDLE", "攝影機視窗已關閉，鏡頭已釋放。"),
                self._loop,
            )

    async def connect_ws(self) -> None:
        try:
            self.ws = await websockets.connect(self.ws_url)
            print(f"[視覺連線] 已連上 {self.ws_url}")
        except Exception as exc:
            self.ws = None
            print(f"[視覺連線] 尚未連上 {self.ws_url}（{exc}）")

    async def _send_ws_cmd(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._ws_lock:
            if self.ws is None:
                await self.connect_ws()
            if self.ws is None:
                return {"ok": False, "message": "尚未連線至視覺服務。"}
            try:
                await self.ws.send(json.dumps(payload, ensure_ascii=False))
                raw = await asyncio.wait_for(self.ws.recv(), timeout=2.5)
                data = json.loads(raw)
                if data.get("type") == "response":
                    return data
                return {"ok": True, "message": "已同步。"}
            except Exception as exc:
                self.ws = None
                return {"ok": False, "message": f"視覺同步異常：{exc}"}

    async def push_state(self, state: str, note: Optional[str] = None) -> None:
        mode_label = "連續對話中" if self.in_conversation else "待機等喚醒"
        print(f"[{mode_label} | {state}]" + (f" {note}" if note else ""))
        async with self._ws_lock:
            if self.ws is None:
                await self.connect_ws()
            if self.ws is None:
                return
            try:
                await self.ws.send(json.dumps({"action": "set_state", "state": state}, ensure_ascii=False))
                _ = await asyncio.wait_for(self.ws.recv(), timeout=1.5)
                if note:
                    await self.ws.send(
                        json.dumps({"action": "command", "text": f"__NOTIFY__:{note}"}, ensure_ascii=False)
                    )
                    _ = await asyncio.wait_for(self.ws.recv(), timeout=1.5)
            except Exception:
                self.ws = None

    async def speak(self, text: str) -> None:
        await self.push_state("SPEAKING", text)
        try:
            await self.voice.speak_zh(text)
        except Exception as exc:
            print(f"[語音播報略過] {exc}")

    def _ask_gemini_sync(self, prompt: str, jpeg_bytes: Optional[bytes] = None) -> str:
        parts: List[Any] = [types.Part.from_text(text=prompt)]
        if jpeg_bytes is not None:
            parts.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))

        user_content = types.Content(role="user", parts=parts)
        contents = [*self.history, user_content]

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                    ),
                )
                reply = (resp.text or "").strip()
                if reply:
                    self.history.append(user_content)
                    self.history.append(
                        types.Content(role="model", parts=[types.Part.from_text(text=reply)])
                    )
                    if len(self.history) > 16:
                        self.history = self.history[-16:]
                return reply
            except Exception as exc:
                last_err = exc
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"Gemini 請求失敗：{last_err}")

    async def _run_confirmed_action(self, action: Dict[str, Any]) -> str:
        kind = action.get("kind")
        if kind == "open_camera":
            ok, msg = await asyncio.to_thread(self.camera.open_camera)
            await self.push_state("CAMERA_MODE" if ok else "IDLE", msg)
            return msg
        if kind == "close_camera":
            msg = await asyncio.to_thread(self.camera.close_camera)
            await self.push_state("IDLE", msg)
            return msg
        if kind == "set_theme":
            color_val = str(action.get("color", ""))
            res = await self._send_ws_cmd({"action": "set_theme_color", "color_value": color_val})
            return str(res.get("message") or "主題處理完成。")
        if kind == "vision_query":
            jpeg = await asyncio.to_thread(self.camera.capture_jpeg_bytes)
            if jpeg is None:
                return "無法取得鏡頭畫面，請確認攝影機是否開啟。"
            await self.push_state("THINKING", "正在分析鏡頭畫面...")
            return await asyncio.to_thread(self._ask_gemini_sync, str(action.get("prompt", "")), jpeg)
        if kind == "launch_app":
            return await asyncio.to_thread(self.system.launch_app, str(action.get("app_key", "")))
        return "未知的待確認動作，已取消。"

    async def handle_command(self, raw_text: str) -> None:
        cleaned = normalize_text(raw_text)
        if not cleaned:
            return
        print(f"[收到指令] {cleaned}")

        if self.pending_action is not None:
            if any(p in cleaned for p in CONFIRM_YES_PHRASES):
                act = self.pending_action
                self.pending_action = None
                msg = await self._run_confirmed_action(act)
                await self.speak(msg)
                await self.push_state("CAMERA_MODE" if self.camera.active else "IDLE", msg)
                return
            if any(p in cleaned for p in CONFIRM_NO_PHRASES):
                self.pending_action = None
                msg = "好的，已取消該操作。"
                await self.speak(msg)
                await self.push_state("CAMERA_MODE" if self.camera.active else "IDLE", msg)
                return
            await self.speak("目前有一項操作等待您明確回覆「好」或「不要」，請先確認。")
            return

        if any(cleaned == p or cleaned.endswith(p) for p in END_SESSION_PHRASES):
            self.in_conversation = False
            self.idle_turns = 0
            msg = "好的，已回到待機等喚醒。"
            await self.speak(msg)
            await self.push_state("CAMERA_MODE" if self.camera.active else "IDLE", msg)
            return

        if any(p in cleaned for p in CLOSE_CAMERA_PHRASES):
            if not self.camera.active:
                await self.speak("目前攝影機本來就是關閉的。")
                return
            self.pending_action = {"kind": "close_camera"}
            await self.speak("即將關閉攝影機並結束鏡頭溝通模式，是否允許？請說好或不要。")
            return

        if any(p in cleaned for p in OPEN_CAMERA_PHRASES):
            self.pending_action = {"kind": "open_camera"}
            await self.speak("即將開啟攝影機與可拖曳預覽視窗進入鏡頭溝通模式，是否允許？請說好或不要。")
            return

        color_val = extract_theme_command(cleaned)
        if color_val is not None:
            self.pending_action = {"kind": "set_theme", "color": color_val}
            await self.speak(f"即將把主題切換為「{color_val}」，是否允許？請說好或不要。")
            return

        if "記事本" in cleaned:
            self.pending_action = {"kind": "launch_app", "app_key": "notepad"}
            await self.speak("即將啟動本機記事本，是否允許？請說好或不要。")
            return
        if "小算盤" in cleaned:
            self.pending_action = {"kind": "launch_app", "app_key": "calc"}
            await self.speak("即將啟動本機小算盤，是否允許？請說好或不要。")
            return

        if "現在幾點" in cleaned or "現在時間" in cleaned or "幾點了" in cleaned:
            msg = self.system.get_current_time_str()
            await self.speak(msg)
            await self.push_state("CAMERA_MODE" if self.camera.active else "IDLE", msg)
            return
        if "工作區有什麼" in cleaned or "列出檔案" in cleaned or "工作區檔案" in cleaned:
            msg = self.system.list_workspace_items()
            await self.speak(msg)
            await self.push_state("CAMERA_MODE" if self.camera.active else "IDLE", msg)
            return

        if self.camera.active:
            self.pending_action = {"kind": "vision_query", "prompt": cleaned}
            await self.speak("鏡頭溝通模式已開啟，是否允許擷取目前鏡頭畫面並傳送給 Gemini 分析？請說好或不要。")
            return

        await self.push_state("THINKING", "思考中...")
        try:
            reply = await asyncio.to_thread(self._ask_gemini_sync, cleaned, None)
        except Exception as exc:
            reply = f"發生錯誤：{exc}"
        if not reply:
            reply = "我目前沒有得到明確回覆，請再說明一次。"

        await self.speak(reply)
        await self.push_state("IDLE", "待機中。")

    async def _voice_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self.in_conversation:
                    audio = await asyncio.to_thread(self.voice.record_window, 4.5, 0.006)
                    if self._stop_event.is_set() or audio is None:
                        continue
                    transcript = await asyncio.to_thread(self.voice.transcribe_zh, audio)
                    if self._stop_event.is_set() or not transcript:
                        continue
                    print(f"[待機監聽] {transcript}")
                    woken, inline_cmd = detect_wake_and_inline(transcript)
                    if not woken:
                        continue

                    self.in_conversation = True
                    self.idle_turns = 0
                    if inline_cmd:
                        await self.handle_command(inline_cmd)
                        if self.in_conversation:
                            await self.push_state("LISTENING", "已進入連續對話，請說下一句指令...")
                    else:
                        await self.push_state("LISTENING", "已喚醒，進入連續對話（可連續說指令，說「結束對話」可退回）。")
                        await self.speak("在，請說。")
                        await self.push_state("LISTENING", "請說出指令...")
                else:
                    audio = await asyncio.to_thread(self.voice.record_window, 5.5, 0.005)
                    if self._stop_event.is_set():
                        break
                    if audio is None:
                        self.idle_turns += 1
                        if self.idle_turns >= 4:
                            self.in_conversation = False
                            self.idle_turns = 0
                            await self.push_state("IDLE", "連續靜默逾時，已回到待機等喚醒。")
                        continue

                    cmd = await asyncio.to_thread(self.voice.transcribe_zh, audio)
                    if self._stop_event.is_set():
                        break
                    if not cmd:
                        await self.speak("這句話我聽得不夠清楚，不會自行猜測，請再說一次。")
                        await self.push_state("LISTENING", "請再說一次...")
                        continue

                    self.idle_turns = 0
                    await self.handle_command(cmd)
                    if self.in_conversation:
                        await self.push_state("LISTENING", "繼續聽取下一句指令...")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._stop_event.is_set():
                    print(f"[語音迴圈例外] {exc}")
                await asyncio.sleep(0.4)

    async def _stdin_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
            except Exception:
                break
            if self._stop_event.is_set():
                break
            if line == "":
                await asyncio.sleep(0.2)
                continue
            text = line.strip()
            if not text:
                continue
            self.in_conversation = True
            self.idle_turns = 0
            await self.handle_command(text)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        print(f"啟動完成 | STT: {self.voice.active_device} | Model: {self.model_name}")
        print("可直接語音喚醒進入連續對話，或在終端機直接輸入文字指令；按 Ctrl+C 結束。")
        await self.connect_ws()
        await self.push_state("IDLE", "待機中（鏡頭關閉）。")

        tasks = [
            asyncio.create_task(self._voice_loop()),
            asyncio.create_task(self._stdin_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            self._stop_event.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.camera.close_camera()
            if self.ws is not None:
                try:
                    await self.ws.close()
                except Exception:
                    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ResourceWarning)
    app = JarvisAssistant()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n已結束程式。")