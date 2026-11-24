from __future__ import annotations

import re
from typing import Any, Dict, Optional

COLOR_PRESETS: Dict[str, Dict[str, str]] = {
    "海洋藍": {"primary": "#00A8FF", "core": "#FF9E2C", "glow": "#0066CC"},
    "藍": {"primary": "#2997FF", "core": "#FF9E2C", "glow": "#0055AA"},
    "藍色": {"primary": "#2997FF", "core": "#FF9E2C", "glow": "#0055AA"},
    "紅": {"primary": "#FF3B5C", "core": "#FFD166", "glow": "#A41623"},
    "紅色": {"primary": "#FF3B5C", "core": "#FFD166", "glow": "#A41623"},
    "金": {"primary": "#FFB703", "core": "#FFF3B0", "glow": "#B56500"},
    "金色": {"primary": "#FFB703", "core": "#FFF3B0", "glow": "#B56500"},
    "紫": {"primary": "#B565FF", "core": "#FF85E4", "glow": "#5A189A"},
    "紫色": {"primary": "#B565FF", "core": "#FF85E4", "glow": "#5A189A"},
    "綠": {"primary": "#00E676", "core": "#FFE066", "glow": "#007E41"},
    "綠色": {"primary": "#00E676", "core": "#FFE066", "glow": "#007E41"},
}

HEX_PATTERN = re.compile(r"^#([0-9a-fA-F]{6})$")


def _derive_glow(hex_color: str) -> str:
    clean = hex_color.lstrip("#")
    r = int(int(clean[0:2], 16) * 0.45)
    g = int(int(clean[2:4], 16) * 0.45)
    b = int(int(clean[4:6], 16) * 0.45)
    return f"#{r:02x}{g:02x}{b:02x}"


def _derive_core(hex_color: str) -> str:
    clean = hex_color.lstrip("#")
    r = min(255, int(int(clean[0:2], 16) * 0.65 + 90))
    g = min(255, int(int(clean[2:4], 16) * 0.65 + 90))
    b = min(255, int(int(clean[4:6], 16) * 0.65 + 70))
    return f"#{r:02x}{g:02x}{b:02x}"


def resolve_color(color_value: str) -> Dict[str, Any]:
    raw = re.sub(r"[。！!？\?，,、\s]+$", "", (color_value or "").strip())
    if not raw:
        return {
            "ok": False,
            "needs_confirmation": True,
            "message": "未提供顏色名稱或色碼。請明確指定（海洋藍、紅色、金色、紫色、綠色）或 #RRGGBB。",
        }

    if raw in COLOR_PRESETS:
        palette = COLOR_PRESETS[raw]
        return {
            "ok": True,
            "needs_confirmation": False,
            "color_name": raw,
            "palette": palette,
            "message": f"已將主題切換為「{raw}」。",
        }

    match = HEX_PATTERN.match(raw)
    if match:
        primary = f"#{match.group(1).upper()}"
        return {
            "ok": True,
            "needs_confirmation": False,
            "color_name": primary,
            "palette": {
                "primary": primary,
                "core": _derive_core(primary),
                "glow": _derive_glow(primary),
            },
            "message": f"已將主題色碼切換為「{primary}」。",
        }

    supported = "、".join(["海洋藍", "紅色", "金色", "紫色", "綠色"])
    return {
        "ok": False,
        "needs_confirmation": True,
        "message": f"「{raw}」不在支援清單中，我不自行猜測。請指定（{supported}）或 #RRGGBB。",
    }


def extract_theme_command(text: str) -> Optional[str]:
    cleaned = re.sub(r"[。！!？\?，,、\s]+$", "", (text or "").strip())
    for pat in (
        r"^(?:嗨|嘿|哈囉)?\S{0,4}[，,\s]*把你的主題(?:切換成|換成)\s*(.+?)\s*$",
        r"^把你的主題(?:切換成|換成)\s*(.+?)\s*$",
        r"^把主題(?:切換成|換成)\s*(.+?)\s*$",
        r"^主題(?:切換成|換成)\s*(.+?)\s*$",
    ):
        m = re.match(pat, cleaned)
        if m:
            return m.group(1).strip()
    return None