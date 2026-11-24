from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
import websockets

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from camera_stage4.camera_controller import OnDemandCamera
from jarvis_core.agent import JarvisAgent
from jarvis_core.config import load_config
from jarvis_core.permissions import UserAbort
from jarvis_core.tools import RuntimeContext, bind_runtime
from voice_stage3.stt_tts import ChineseVoiceEngine

END_SESSION_PHRASES = ("結束對話", "退下", "休息一下", "修洗一下", "先這樣", "結束")


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

    head = compact[:6]
    for start in range(len(head)):
        window = head[start : start + 3]
        if len(window) >= 2 and SequenceMatcher(None, window, "賈維斯").ratio() >= 0.55:
            after = re.sub(r"^[，,、\s]+", "", compact[start + len(window) :])
            return True, after

    return False, ""


class JarvisVoiceAssistant:
    def __init__(self, ws_url: str = "ws://127.0.0.1:8766") -> None:
        self.cfg = load_config()
        self.ws_url = ws_url
        self.client = genai.Client(
            api_key=self.cfg.api_key,
            http_options=types.HttpOptions(timeout=15000),
        )

        self.voice = ChineseVoiceEngine(
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small").strip() or "small",
            device=os.getenv("WHISPER_DEVICE", "cuda").strip() or "cuda",
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16").strip() or "float16",
            tts_voice=os.getenv("TTS_VOICE", "zh-TW-YunJheNeural").strip() or "zh-TW-YunJheNeural",
        )

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_input_was_voice: bool = False

        camera_idx = int(os.getenv("CAMERA_INDEX", "0").strip() or "0")
        self.camera = OnDemandCamera(camera_index=camera_idx)

        bind_runtime(
            RuntimeContext(
                camera=self.camera,
                ws_url=self.ws_url,
                gemini_client=self.client,
                gemini_model=self.cfg.model,
                on_state_change=self._on_tool_state_change,
            )
        )

        self.agent = JarvisAgent(self.cfg, client=self.client, input_fn=self._sync_confirm_input)

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

        self.in_conversation: bool = False
        self.idle_turns: int = 0

    def _on_tool_state_change(self, state: str, note: Optional[str]) -> None:
        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.push_state(state, note), self._loop)

    def _sync_confirm_input(self, prompt: str) -> str:
        if self._last_input_was_voice:
            self.voice.speak_zh_blocking(prompt)
            for _ in range(3):
                if self._stop_event.is_set():
                    raise UserAbort()
                audio = self.voice.record_window(4.0, 0.005, silence_stop_sec=0.6)
                if audio is not None:
                    ans = self.voice.transcribe_zh(audio)
                    if ans:
                        print(f"[語音補充回覆] {ans}")
                        return ans
                self.voice.speak_zh_blocking("沒有聽清楚，請再說一次。")
            return "取消"
        return input(prompt)

    async def connect_ws(self) -> None:
        try:
            self.ws = await websockets.connect(self.ws_url, open_timeout=2.0)
            print(f"[視覺連線] 已連上 {self.ws_url}")
        except Exception as exc:
            self.ws = None
            print(f"[視覺連線] 尚未連上 {self.ws_url}（{exc}）")

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
                if note:
                    await self.ws.send(
                        json.dumps({"action": "command", "text": f"__NOTIFY__:{note}"}, ensure_ascii=False)
                    )
            except Exception:
                self.ws = None

    async def _camera_stream_loop(self) -> None:
        """當鏡頭開啟時，持續將輕量 JPEG 影格推送到 HUD 內嵌浮動視窗（約 15 FPS）。"""
        while not self._stop_event.is_set():
            try:
                if self.camera.active:
                    b64 = await asyncio.to_thread(self.camera.capture_base64_preview)
                    if b64:
                        async with self._ws_lock:
                            if self.ws is not None:
                                try:
                                    await self.ws.send(
                                        json.dumps({"action": "camera_frame", "jpeg_b64": b64}, ensure_ascii=False)
                                    )
                                except Exception:
                                    self.ws = None
                    await asyncio.sleep(0.065)
                else:
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.25)

    async def speak(self, text: str) -> None:
        await self.push_state("SPEAKING", text)
        try:
            await self.voice.speak_zh(text)
        except Exception as exc:
            print(f"[語音播報略過] {exc}")

    async def handle_turn(self, raw_text: str, from_voice: bool) -> None:
        cleaned = normalize_text(raw_text)
        if not cleaned:
            return
        self._last_input_was_voice = from_voice
        print(f"[收到指令] {cleaned}")

        if any(cleaned == p or cleaned.endswith(p) for p in END_SESSION_PHRASES):
            self.in_conversation = False
            self.idle_turns = 0
            msg = "好的，已回到待機等喚醒。"
            await self.speak(msg)
            await self.push_state("CAMERA_MODE" if self.camera.active else "IDLE", msg)
            return

        await self.push_state("THINKING", "正在處理指令...")
        # 若鏡頭正在開啟，提示模型可直接針對畫面判斷或執行控制
        prompt_for_agent = (
            f"[系統狀態：鏡頭溝通模式已開啟] {cleaned}" if self.camera.active else cleaned
        )
        try:
            reply = await asyncio.to_thread(self.agent.chat, prompt_for_agent)
        except UserAbort:
            reply = "已取消操作。"
        except Exception as exc:
            print(f"[Agent 詳細錯誤] {exc}")
            reply = "目前模型高負載無回應，請稍後再試。"

        await self.speak(reply)
        await self.push_state("CAMERA_MODE" if self.camera.active else "IDLE", "完成。")

    async def _voice_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self.in_conversation:
                    audio = await asyncio.to_thread(self.voice.record_window, 3.5, 0.006, 0.55)
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
                        await self.handle_turn(inline_cmd, from_voice=True)
                        if self.in_conversation:
                            await self.push_state("LISTENING", "已進入連續對話，請說下一句指令...")
                    else:
                        await self.push_state("LISTENING", "已喚醒，進入連續對話。")
                        await self.speak("在，請說。")
                        await self.push_state("LISTENING", "請說出指令...")
                else:
                    audio = await asyncio.to_thread(self.voice.record_window, 4.5, 0.005, 0.55)
                    if self._stop_event.is_set():
                        break
                    if audio is None:
                        self.idle_turns += 1
                        if self.idle_turns >= 5:
                            self.in_conversation = False
                            self.idle_turns = 0
                            await self.push_state("IDLE", "連續靜默逾時，已回到待機等喚醒。")
                        continue

                    cmd = await asyncio.to_thread(self.voice.transcribe_zh, audio)
                    if self._stop_event.is_set():
                        break
                    if not cmd:
                        await self.speak("沒聽清楚，請再說一次。")
                        await self.push_state("LISTENING", "請再說一次...")
                        continue

                    self.idle_turns = 0
                    await self.handle_turn(cmd, from_voice=True)
                    if self.in_conversation:
                        await self.push_state("LISTENING", "繼續聽取下一句指令...")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._stop_event.is_set():
                    print(f"[語音迴圈例外] {exc}")
                await asyncio.sleep(0.3)

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
            await self.handle_turn(text, from_voice=False)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        print(f"啟動完成 | STT: {self.voice.active_device} | Model: {self.cfg.model}")
        print("說「嗨賈維斯」喚醒進入連續對話，或直接在終端機輸入指令；按 Ctrl+C 結束。")
        await self.connect_ws()
        await self.push_state("IDLE", "待機中（鏡頭關閉）。")

        tasks = [
            asyncio.create_task(self._voice_loop()),
            asyncio.create_task(self._stdin_loop()),
            asyncio.create_task(self._camera_stream_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            self._stop_event.set()
            for t in tasks:
                t.cancel()
            self.camera.close_camera()
            if self.ws is not None:
                try:
                    await self.ws.close()
                except Exception:
                    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ResourceWarning)
    assistant = JarvisVoiceAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\n已結束程式。")
        os._exit(0)