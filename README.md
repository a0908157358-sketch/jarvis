# 中文賈維斯 — 第 1 階段

本機助理核心骨架：設定讀取、Gemini Tool Calling 迴圈、逐步權限確認、2 個唯讀測試工具。

## 專案結構

```
Jarvis\
├─ main.py                 入口（互動式對話）
├─ selftest.py             離線自我測試（不需 API 金鑰）
├─ requirements.txt
├─ .env.example            複製成 .env 後填寫
└─ jarvis_core\
   ├─ config.py            讀取 .env，缺漏就報錯，不猜預設值
   ├─ permissions.py       待允許動作說明 + 允許/同意/執行 確認
   ├─ tools.py             get_current_time、list_folder_preview
   └─ agent.py             Gemini 手動 Tool Calling 迴圈
```

## Windows 10 安裝與啟動

1. 安裝 Python 3.10 以上（安裝時勾選 **Add python.exe to PATH**），開啟「命令提示字元」確認：
   ```
   python --version
   ```
2. 切到專案資料夾。桌面若被 OneDrive 接管，路徑會是 `...\OneDrive\Desktop\Jarvis`，請以檔案總管網址列複製的實際路徑為準：
   ```
   cd /d "你的實際路徑\Desktop\Jarvis"
   ```
3. （建議，可略過）建立虛擬環境。請用命令提示字元（cmd），不要用 PowerShell，以免遇到執行原則限制：
   ```
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```
4. 安裝套件：
   ```
   pip install google-genai python-dotenv
   ```
5. 建立設定檔並用記事本填寫 `GEMINI_API_KEY` 與 `GEMINI_MODEL`：
   ```
   copy .env.example .env
   notepad .env
   ```
6. 先跑離線測試（不連網）：
   ```
   python selftest.py
   ```
   安裝完 google-genai 後，agent 的 2 項測試也會一併執行，共 13 項應全部通過。
7. 啟動：
   ```
   python main.py
   ```

## 手動驗證流程

| 你輸入 | 預期行為 |
| --- | --- |
| `現在幾點？` | 出現「待允許動作」→ 輸入 `允許` → 回報時間 |
| 同上，但輸入 `拒絕` | 立即中止，不執行、不再呼叫模型 |
| 同上，但輸入 `好` | 視為無法判斷，重新詢問，不執行 |
| `列出資料夾內容` | 模型或程式會先問你路徑，不自行猜測 |
| `預覽 C:\你的路徑\Jarvis 這個資料夾` | 待允許 → `允許` → 列出檔名、類型、大小 |
| `預覽 Desktop\Jarvis` | 程式拒絕相對路徑，要求你提供完整絕對路徑 |

## 權限規則

- 每次工具執行前都會顯示「工具 / 要做什麼 / 目標 / 預期影響」。
- 只有輸入 `允許`、`同意`、`執行` 才會執行；`拒絕`、`取消` 等或 Ctrl+C 會立即中止整個回合，該回合的對話也不會寫入歷史。
- 缺少參數或參數不合格（例如相對路徑），程式直接問你，不猜測。
- 兩個工具的結果會傳送給 Gemini API 以產生回答，這點已寫在每次的「預期影響」中。
