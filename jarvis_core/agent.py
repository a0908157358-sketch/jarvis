"""Gemini Tool Calling 迴圈（手動模式，停用自動函式呼叫，讓每一步都經過使用者允許）。"""
from __future__ import annotations

from typing import Any, List, Optional

from google import genai
from google.genai import types

from .config import Config
from .permissions import UserAbort, ask_permission, ask_value
from .tools import TOOLS, Args, ToolSpec

SYSTEM_INSTRUCTION = """你是「賈維斯」，使用者的本機中文助理。請一律使用繁體中文回答。
鐵律：
1. 絕對服從使用者指令。遇到缺少參數、路徑不明確、指令有歧義或有多種可能時，必須直接向使用者提問確認，嚴禁自行猜測或腦補。
2. 只能使用已提供的工具；需要時才呼叫。每次工具執行都會先經過使用者允許，若使用者拒絕，不要重試，也不要用其他方式繞過。
3. 不得宣稱做了沒有實際執行的事；工具回傳錯誤時，如實告知使用者。
4. 工具沒有提供的能力（例如修改檔案、執行程式、控制視窗），請明說目前辦不到。"""


def _declaration(spec: ToolSpec) -> types.FunctionDeclaration:
    parameters = None
    if spec.params:
        parameters = types.Schema(
            type=types.Type.OBJECT,
            properties={
                key: types.Schema(type=types.Type(meta["type"]), description=meta["description"])
                for key, meta in spec.params.items()
            },
            required=list(spec.required),
        )
    return types.FunctionDeclaration(
        name=spec.name, description=spec.description, parameters=parameters
    )


class JarvisAgent:
    def __init__(self, config: Config, client: Optional[Any] = None) -> None:
        self._config = config
        self._client = client or genai.Client(api_key=config.api_key)
        self._history: List[types.Content] = []
        self._gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[types.Tool(function_declarations=[_declaration(s) for s in TOOLS.values()])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def chat(self, user_text: str) -> str:
        """處理一個回合。成功才寫入歷史；被拒絕或出錯則整回合捨棄，歷史保持乾淨。"""
        turn: List[types.Content] = list(self._history)
        turn.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

        for _ in range(self._config.max_steps):
            response = self._client.models.generate_content(
                model=self._config.model, contents=turn, config=self._gen_config
            )
            if not response.candidates or response.candidates[0].content is None:
                raise RuntimeError("模型沒有回傳內容（可能被安全機制攔截或模型名稱有誤）")

            # 原樣放回模型回覆（含 thought signature 等欄位，不可改寫）
            turn.append(response.candidates[0].content)

            calls = response.function_calls or []
            if not calls:
                self._history = turn
                return response.text or "（模型沒有文字回覆）"

            parts = [self._run_call(call) for call in calls]  # 任何一個被拒絕 → UserAbort 往上拋
            turn.append(types.Content(role="user", parts=parts))

        raise RuntimeError(f"工具呼叫超過上限 {self._config.max_steps} 輪，已中止")

    def _run_call(self, call: Any) -> types.Part:
        spec = TOOLS.get(call.name)
        if spec is None:
            return types.Part.from_function_response(
                name=call.name, response={"error": f"沒有這個工具：{call.name}"}
            )

        raw_args = {k: v for k, v in dict(call.args or {}).items() if k in spec.params}
        args = self._complete_args(spec, raw_args)

        ask_permission(spec.describe(args))  # 沒有明確同意就不會往下執行

        try:
            result = spec.run(args)
        except Exception as exc:  # 工具錯誤如實回報給模型，不自行重試
            result = {"error": f"{type(exc).__name__}: {exc}"}

        status = "失敗" if "error" in result else "完成"
        print(f"✔ 已執行 {spec.name}（{status}）")
        return types.Part.from_function_response(name=spec.name, response={"result": result})

    @staticmethod
    def _complete_args(spec: ToolSpec, args: Args) -> Args:
        """參數缺漏或不合格時，直接詢問使用者，直到合格或使用者取消。"""
        while True:
            clean, problems = spec.prepare(args)
            if not problems:
                return clean
            for param, problem in problems:
                args[param] = ask_value(param, spec.params[param]["description"], problem)
