from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import edge_tts
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from google import genai

# 自動將 .venv 內的 NVIDIA CUDA 12 (cuBLAS / cuDNN 9) DLL 路徑加入程序搜尋路徑
def _register_cuda12_dlls() -> None:
    if sys.platform != "win32":
        return
    venv_site = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
    for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        dll_dir = venv_site / sub
        if dll_dir.is_dir():
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(dll_dir))
                except OSError:
                    pass


_register_cuda12_dlls()

try:
    from opencc import OpenCC
    _CC_S2T = OpenCC("s2t")
except Exception:
    _CC_S2T = None

_WAKE_FIX_PATTERN = re.compile(
    r"(?:傢伙死|家瑋死|叫維斯|假維斯|賈為斯|甲維斯|家維斯|佳維斯|加維斯|假為師|角尾絲)"
)


def to_traditional(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if _CC_S2T is not None:
        raw = _CC_S2T.convert(raw)
    raw = _WAKE_FIX_PATTERN.sub("賈維斯", raw)
    return raw


def is_meaningful(text: str) -> bool:
    stripped = re.sub(r"[。！!？\?，,、\.…\s]+", "", (text or "").strip())
    return len(stripped) >= 2


def sanitize_for_tts(text: str) -> str:
    cleaned = re.sub(r"[\r\n]+", "，", (text or "").strip())
    cleaned = re.sub(r"[{}\[\]<>\"'`\\|*_#~^]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 220:
        cleaned = cleaned[:220] + "。"
    return cleaned


class ChineseVoiceEngine:
    """
    高速高準確率中文語音引擎：
    - 優先使用 RTX 3050 CUDA 12 (float16) 加速辨識；若無 GPU 則自動無縫退回 CPU (int8)。
    - 動態語音活動偵測（講完話停頓 0.85 秒立即截斷送出，不用乾等）。
    - 全程鎖定單一台灣男聲（zh-TW-YunJheNeural）。
    """

    def __init__(
        self,
        whisper_model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        tts_voice: str = "zh-TW-YunJheNeural",
        sample_rate: int = 16000,
    ) -> None:
        self.sample_rate = sample_rate
        self.tts_voice = tts_voice
        self.whisper_model_size = whisper_model_size
        self._zh_prompt = "以下是繁體中文語音指令：嗨賈維斯，把主題換成紅色、海洋藍、金色、紫色、綠色，開啟鏡頭溝通模式，關閉鏡頭，現在幾點，打開記事本，打開小算盤。"

        try:
            self.stt_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
            self.active_device = f"{device} ({compute_type})"
        except Exception as exc:
            print(f"[STT 提示] 無法啟用 {device} ({exc})，自動改用 CPU (int8)。")
            self.stt_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
            self.active_device = "cpu (int8)"

    def _switch_to_cpu_int8(self) -> None:
        self.stt_model = WhisperModel(self.whisper_model_size, device="cpu", compute_type="int8")
        self.active_device = "cpu (int8)"

    def record_window(
        self,
        seconds: float = 5.0,
        rms_threshold: float = 0.006,
        silence_stop_sec: float = 0.85,
    ) -> Optional[np.ndarray]:
        """
        動態錄音：
        - 偵測到開口說話後，只要連續靜音超過 silence_stop_sec（預設 0.85 秒）就立刻結束錄音送辨識，大幅減少等待時間。
        """
        chunk_sec = 0.15
        chunk_frames = int(self.sample_rate * chunk_sec)
        max_chunks = max(1, int(seconds / chunk_sec))
        silence_limit = max(2, int(silence_stop_sec / chunk_sec))

        chunks = []
        heard_speech = False
        silent_chunks = 0

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                data, _ = stream.read(chunk_frames)
                mono = data.flatten().copy()
                chunks.append(mono)
                rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size > 0 else 0.0

                if rms >= rms_threshold:
                    heard_speech = True
                    silent_chunks = 0
                elif heard_speech:
                    silent_chunks += 1
                    if silent_chunks >= silence_limit:
                        break

        if not heard_speech or not chunks:
            return None
        return np.concatenate(chunks)

    def transcribe_zh(self, audio: np.ndarray) -> str:
        try:
            segments, _ = self.stt_model.transcribe(
                audio,
                language="zh",
                initial_prompt=self._zh_prompt,
                vad_filter=True,
                beam_size=3,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            raw_text = "".join(seg.text for seg in segments).strip()
        except RuntimeError as exc:
            if "cublas" in str(exc).lower() or "cuda" in str(exc).lower() or "cudnn" in str(exc).lower():
                self._switch_to_cpu_int8()
                segments, _ = self.stt_model.transcribe(
                    audio,
                    language="zh",
                    initial_prompt=self._zh_prompt,
                    vad_filter=True,
                    beam_size=3,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
                raw_text = "".join(seg.text for seg in segments).strip()
            else:
                raise

        trad = to_traditional(raw_text)
        return trad if is_meaningful(trad) else ""

    def _play_mp3_worker(self, mp3_path: str) -> bool:
        ps_script = (
            "Add-Type -AssemblyName presentationCore; "
            "$player = New-Object System.Windows.Media.MediaPlayer; "
            f"$player.Open([uri]::new('{mp3_path.replace(chr(39), chr(39)*2)}')); "
            "$player.Play(); "
            "$waited = 0; "
            "while (-not $player.NaturalDuration.HasTimeSpan -and $waited -lt 2500) { "
            "  Start-Sleep -Milliseconds 80; $waited += 80 "
            "}; "
            "if ($player.NaturalDuration.HasTimeSpan) { "
            "  $ms = [Math]::Min([int]$player.NaturalDuration.TimeSpan.TotalMilliseconds + 200, 25000); "
            "  Start-Sleep -Milliseconds $ms "
            "} else { "
            "  Start-Sleep -Milliseconds 2000 "
            "}; "
            "$player.Stop(); $player.Close()"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            timeout=30,
        )
        return res.returncode == 0

    def _speak_sapi_worker(self, clean_text: str) -> None:
        safe_text = clean_text.replace("'", "''")
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voices = $synth.GetInstalledVoices() | Where-Object { $_.Enabled }; "
            "$maleZh = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like 'zh*' -and $_.VoiceInfo.Gender -eq 'Male' } | Select-Object -First 1; "
            "$anyZh = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like 'zh*' } | Select-Object -First 1; "
            "if ($maleZh) { $synth.SelectVoice($maleZh.VoiceInfo.Name) } elseif ($anyZh) { $synth.SelectVoice($anyZh.VoiceInfo.Name) }; "
            "$synth.Rate = 2; "
            f"$synth.Speak('{safe_text}')"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            timeout=30,
        )

    def speak_zh_blocking(self, text: str) -> None:
        clean_text = sanitize_for_tts(text)
        if not clean_text:
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name

        try:
            async def _synth() -> None:
                comm = edge_tts.Communicate(clean_text, voice=self.tts_voice, rate="+12%")
                await comm.save(mp3_path)

            asyncio.run(_synth())
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                if self._play_mp3_worker(mp3_path):
                    return
        except Exception as exc:
            print(f"[TTS 備援] edge-tts 無法使用（{exc}），改用本機語音。")
        finally:
            if os.path.exists(mp3_path):
                try:
                    os.remove(mp3_path)
                except OSError:
                    pass

        self._speak_sapi_worker(clean_text)

    async def speak_zh(self, text: str) -> None:
        await asyncio.to_thread(self.speak_zh_blocking, text)


def create_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("缺少必填環境變數 GEMINI_API_KEY，請先在 .env 中設定。")
    return genai.Client(api_key=key)