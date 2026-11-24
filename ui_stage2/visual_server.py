from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Set

from aiohttp import web
import websockets
from websockets.exceptions import ConnectionClosed

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui_stage2.theme_controller import extract_theme_command, resolve_color

VALID_STATES = {"IDLE", "LISTENING", "THINKING", "SPEAKING", "CAMERA_MODE"}


class JarvisVisualHub:
    def __init__(self, host: str = "127.0.0.1", http_port: int = 8765, ws_port: int = 8766) -> None:
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.clients: Set[Any] = set()
        self.state: str = "IDLE"
        self.palette: Dict[str, str] = {"primary": "#00A8FF", "core": "#FF9E2C", "glow": "#0066CC"}
        self.theme_name: str = "海洋藍"
        self.html_path = Path(__file__).with_name("humanoid_hud.html")

    def current_payload(self) -> Dict[str, Any]:
        return {
            "type": "sync",
            "state": self.state,
            "theme_name": self.theme_name,
            "palette": self.palette,
        }

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        if not self.clients:
            return
        encoded = json.dumps(payload, ensure_ascii=False)
        stale = []
        for client in list(self.clients):
            try:
                await client.send(encoded)
            except Exception:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    async def set_state(self, new_state: str) -> Dict[str, Any]:
        normalized = (new_state or "").strip().upper()
        if normalized not in VALID_STATES:
            return {
                "ok": False,
                "needs_confirmation": True,
                "message": f"無效狀態「{new_state}」。合法狀態為：{', '.join(sorted(VALID_STATES))}。",
            }
        self.state = normalized
        await self.broadcast({"type": "state", "state": self.state})
        return {"ok": True, "needs_confirmation": False, "state": self.state, "message": f"狀態已設為 {self.state}。"}

    async def set_theme_color(self, color_value: str) -> Dict[str, Any]:
        result = resolve_color(color_value)
        if not result["ok"]:
            return result
        self.theme_name = result["color_name"]
        self.palette = result["palette"]
        await self.broadcast(
            {
                "type": "theme",
                "theme_name": self.theme_name,
                "palette": self.palette,
            }
        )
        return result

    async def handle_command(self, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {"ok": False, "needs_confirmation": True, "message": "空白指令，請說明明確操作。"}

        if raw.startswith("__NOTIFY__:"):
            note = raw.removeprefix("__NOTIFY__:").strip()
            await self.broadcast({"type": "notify", "message": note})
            return {"ok": True, "needs_confirmation": False, "message": note}

        color = extract_theme_command(raw)
        if color is not None:
            return await self.set_theme_color(color)

        upper = raw.upper()
        if upper in VALID_STATES:
            return await self.set_state(upper)

        return {
            "ok": False,
            "needs_confirmation": True,
            "message": f"無法辨識指令「{raw}」。請使用明確的顏色切換或狀態指令。",
        }

    async def ws_handler(self, websocket: Any) -> None:
        self.clients.add(websocket)
        try:
            await websocket.send(json.dumps(self.current_payload(), ensure_ascii=False))
            async for raw_msg in websocket:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps(
                            {"type": "response", "ok": False, "needs_confirmation": True, "message": "JSON 格式錯誤。"},
                            ensure_ascii=False,
                        )
                    )
                    continue

                action = data.get("action")
                if action == "set_theme_color":
                    res = await self.set_theme_color(str(data.get("color_value", "")))
                elif action == "set_state":
                    res = await self.set_state(str(data.get("state", "")))
                elif action == "command":
                    res = await self.handle_command(str(data.get("text", "")))
                else:
                    res = {
                        "ok": False,
                        "needs_confirmation": True,
                        "message": f"不支援的動作「{action}」。",
                    }

                await websocket.send(json.dumps({"type": "response", **res}, ensure_ascii=False))
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)

    async def index_handler(self, request: web.Request) -> web.Response:
        html = self.html_path.read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html", charset="utf-8")

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self.index_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.http_port)
        await site.start()

        ws_server = await websockets.serve(self.ws_handler, self.host, self.ws_port)
        print(f"HUD 網頁：http://{self.host}:{self.http_port}/")
        print(f"WebSocket：ws://{self.host}:{self.ws_port}")

        try:
            await asyncio.Future()
        finally:
            ws_server.close()
            await ws_server.wait_closed()
            await runner.cleanup()


if __name__ == "__main__":
    hub = JarvisVisualHub()
    try:
        asyncio.run(hub.start())
    except KeyboardInterrupt:
        print("\n視覺服務已停止。")