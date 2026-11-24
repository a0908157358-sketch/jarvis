from __future__ import annotations

import io
import threading
from typing import Callable, Optional

import cv2
from PIL import Image


class OnDemandCamera:
    """
    按需開啟的攝影機控制器：
    - 預設關閉，不佔用攝影機。
    - 開啟時顯示精巧視窗（420x315），使用者可自由拖曳移動。
    - 按 ESC 或關閉視窗時，會主動呼叫 on_external_close 回呼以同步釋放裝置與狀態。
    """

    def __init__(
        self,
        camera_index: int = 0,
        window_width: int = 420,
        window_height: int = 315,
        on_external_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self.camera_index = camera_index
        self.window_width = window_width
        self.window_height = window_height
        self.on_external_close = on_external_close

        self.cap: Optional[cv2.VideoCapture] = None
        self.active: bool = False
        self._closing_guard = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._window_title = "J.A.R.V.I.S. Cam (ESC/關閉可收起)"

    def _preview_loop(self) -> None:
        closed_by_user = False
        try:
            cv2.namedWindow(self._window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_title, self.window_width, self.window_height)

            while not self._stop_event.is_set() and self.active:
                frame = None
                with self._lock:
                    if self.cap is not None and self.cap.isOpened():
                        ret, raw = self.cap.read()
                        if ret and raw is not None:
                            self._latest_frame = raw
                            frame = raw

                if frame is not None:
                    cv2.imshow(self._window_title, frame)

                # 檢查使用者是否按 ESC 或點右上角 X 關閉視窗
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    closed_by_user = True
                    break
                try:
                    prop = cv2.getWindowProperty(self._window_title, cv2.WND_PROP_VISIBLE)
                    if prop < 1:
                        closed_by_user = True
                        break
                except Exception:
                    closed_by_user = True
                    break
        finally:
            try:
                cv2.destroyWindow(self._window_title)
            except Exception:
                pass
            cv2.waitKey(1)

        if closed_by_user and not self._closing_guard.is_set():
            self.close_camera()
            if self.on_external_close is not None:
                try:
                    self.on_external_close()
                except Exception:
                    pass

    def open_camera(self) -> tuple[bool, str]:
        self._closing_guard.clear()
        if self.active and self.cap is not None and self.cap.isOpened():
            return True, "鏡頭溝通模式已經開啟。"

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return (
                False,
                f"無法開啟攝影機（索引 {self.camera_index}）。請確認攝影機連線與隱私權限後再告訴我。",
            )

        with self._lock:
            self.cap = cap
            self.active = True
            self._latest_frame = None

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._thread.start()
        return True, "已開啟鏡頭溝通模式並顯示精巧預覽視窗（可拖曳移動，按 ESC 可關閉）。"

    def close_camera(self) -> str:
        self._closing_guard.set()
        was_active = self.active
        self.active = False
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.5)
        if threading.current_thread() is not self._thread:
            self._thread = None

        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self._latest_frame = None

        try:
            cv2.destroyWindow(self._window_title)
            cv2.waitKey(1)
        except Exception:
            pass

        if was_active:
            return "已關閉攝影機並釋放鏡頭資源。"
        return "目前鏡頭已關閉。"

    def capture_jpeg_bytes(self, quality: int = 75) -> Optional[bytes]:
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
        img.thumbnail((800, 600))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()