from __future__ import annotations

import base64
import io
import threading
import time
from typing import Optional

import cv2
from PIL import Image


class OnDemandCamera:
    """
    按需開啟的純背景攝影機控制器：
    - 預設關閉，不佔用裝置。
    - 開啟時在背景高速擷取畫面，提供即時 JPEG bytes 與 Base64 串流給 HUD 內嵌浮動視窗。
    """

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.active: bool = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set() and self.active:
            with self._lock:
                if self.cap is not None and self.cap.isOpened():
                    ret, raw = self.cap.read()
                    if ret and raw is not None:
                        self._latest_frame = raw
            time.sleep(0.03)  # 約 30 FPS 背景緩衝

    def open_camera(self) -> tuple[bool, str]:
        if self.active and self.cap is not None and self.cap.isOpened():
            return True, "鏡頭溝通模式已經開啟。"

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return False, f"無法開啟攝影機（索引 {self.camera_index}），請確認裝置與權限。"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        with self._lock:
            self.cap = cap
            self.active = True
            self._latest_frame = None

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True, "已開啟鏡頭溝通模式，畫面已內嵌至全像介面（可拖曳或說放大縮小）。"

    def close_camera(self) -> str:
        was_active = self.active
        self.active = False
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self._latest_frame = None

        return "已關閉攝影機並釋放資源。" if was_active else "目前鏡頭已關閉。"

    def capture_jpeg_bytes(self, max_size: tuple[int, int] = (512, 384), quality: int = 72) -> Optional[bytes]:
        if not self.active:
            return None
        frame = None
        with self._lock:
            if self._latest_frame is not None:
                frame = self._latest_frame.copy()
            elif self.cap is not None and self.cap.isOpened():
                ret, raw = self.cap.read()
                if ret and raw is not None:
                    frame = raw
        if frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def capture_base64_preview(self) -> Optional[str]:
        jpeg = self.capture_jpeg_bytes(max_size=(400, 300), quality=62)
        if jpeg is None:
            return None
        return base64.b64encode(jpeg).decode("ascii")