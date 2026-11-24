"""中文賈維斯 - 第 1 階段入口。執行：python main.py"""
from __future__ import annotations

from jarvis_core.config import ConfigError, load_config
from jarvis_core.permissions import UserAbort

EXIT_WORDS = {"離開", "結束", "exit", "quit"}


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"❌ {exc}")
        return

    # 這裡才匯入 agent，讓設定錯誤時不必先載入 google-genai
    from google.genai import errors

    from jarvis_core.agent import JarvisAgent

    agent = JarvisAgent(config)
    print("中文賈維斯 第 1 階段已啟動")
    print(f"模型：{config.model}　｜　API 金鑰：已讀取（不顯示）　｜　輸入「離開」結束")
    print("提醒：任何工具執行前都會先請您允許。")

    while True:
        try:
            text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已離開。")
            return
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            print("已離開。")
            return
        try:
            print(f"\n賈維斯> {agent.chat(text)}")
        except UserAbort:
            print("\n⛔ 您已拒絕或取消，本次動作已立即中止，未執行任何後續步驟。")
        except errors.APIError as exc:
            print(f"\n❌ Gemini API 錯誤：{exc}")
        except RuntimeError as exc:
            print(f"\n❌ {exc}")


if __name__ == "__main__":
    main()
