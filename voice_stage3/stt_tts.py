from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from typing import Optional

import edge_tts
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from google import genai

try:
    from opencc import OpenCC
    _CC_S2T = OpenCC("s2t")
except Exception:
    _CC_S2T = None


def to_traditional(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if _CC_S2T is not None:
        return _CC_S2T.convert(raw)
    return raw


def is_meaningful(text: str) -> bool:
    stripped = re.sub(r"[。！!？\?，,、\.…\s]+", "", (text or "").strip())
    return len(stripped) >= 2


class ChineseVoiceEngine:
    """
    單一聲線語音引擎：
    - 預設使用 zh-TW-YunJheNeural（男聲）。
    - 若 edge-tts 失敗，SAPI 盡量挑選男聲或指定聲線，避免突然變女聲。
    """

    def __init__(
        self,
        whisper_model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        tts_voice: str = "zh-TW-YunJheNeural",
        sample_rate: int = 16000,
    ) -> None:
        self.sample_rate = sample_rate
        self.tts_voice = tts_voice
        self.whisper_model_size = whisper_model_size

        try:
            self.stt_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
            self.active_device = f"{device} ({compute_type})"
        except Exception:
            self.stt_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
            self.active_device = "cpu (int8)"

    def _switch_to_cpu_int8(self) -> None:
        self.stt_model = WhisperModel(self.whisper_model_size, device="cpu", compute_type="int8")
        self.active_device = "cpu (int8)"

    def record_window(self, seconds: float = 4.5, rms_threshold: float = 0.006) -> Optional[np.ndarray]:
        frames = int(seconds * self.sample_rate)
        audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        mono = audio.flatten()
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size > 0 else 0.0
        if rms < rms_threshold:
            return None
        return mono

    def transcribe_zh(self, audio: np.ndarray) -> str:
        try:
            segments, _ = self.stt_model.transcribe(
                audio,
                language="zh",
                initial_prompt=None,
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=False,
            )
            raw_text = "".join(seg.text for seg in segments).strip()
        except RuntimeError as exc:
            if "cublas" in str(exc).lower() or "cuda" in str(exc).lower():
                self._switch_to_cpu_int8()
                segments, _ = self.stt_model.transcribe(
                    audio,
                    language="zh",
                    initial_prompt=None,
                    vad_filter=True,
                    beam_size=5,
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
            "while (-not $player.NaturalDuration.HasTimeSpan -and $waited -lt 3000) { "
            "  Start-Sleep -Milliseconds 100; $waited += 100 "
            "}; "
            "if ($player.NaturalDuration.HasTimeSpan) { "
            "  $ms = [Math]::Min([int]$player.NaturalDuration.TimeSpan.TotalMilliseconds + 300, 25000); "
            "  Start-Sleep -Milliseconds $ms "
            "} else { "
            "  Start-Sleep -Milliseconds 2500 "
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
        safe_text = clean_text.replace("\r", " ").replace("\n", " ").replace("'", "''")
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voices = $synth.GetInstalledVoices() | Where-Object { $_.Enabled }; "
            "$maleZh = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like 'zh*' -and $_.VoiceInfo.Gender -eq 'Male' } | Select-Object -First 1; "
            "$anyZh = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like 'zh*' } | Select-Object -First 1; "
            "if ($maleZh) { $synth.SelectVoice($maleZh.VoiceInfo.Name) } elseif ($anyZh) { $synth.SelectVoice($anyZh.VoiceInfo.Name) }; "
            "$synth.Rate = 1; "
            f"$synth.Speak('{safe_text}')"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            timeout=30,
        )

    async def speak_zh(self, text: str) -> None:
        clean_text = re.sub(r"[\r\n]+", "，", (text or "").strip())
        if not clean_text:
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name

        try:
            comm = edge_tts.Communicate(clean_text, voice=self.tts_voice)
            await comm.save(mp3_path)
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                if await asyncio.to_thread(self._play_mp3_worker, mp3_path):
                    return
        except Exception as exc:
            print(f"[TTS 備援] edge-tts 無法使用（{exc}），改用本機語音。")
        finally:
            if os.path.exists(mp3_path):
                try:
                    os.remove(mp3_path)
                except OSError:
                    pass

        await asyncio.to_thread(self._speak_sapi_worker, clean_text)


def create_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)