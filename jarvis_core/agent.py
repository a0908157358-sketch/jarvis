"""Gemini Tool Calling 迴圈（直接執行使用者明確下達的工具指令；缺參數時反問，絕不猜測）。"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

from google import genai
from google.genai import types

from .config import Config
from .permissions import ask_value
from .tools import TOOLS, Args, ToolSpec

SYSTEM_INSTRUCTION = """你是「賈維斯」，使用者的本機全能中文助理。請一律使用繁體中文回答，口氣自然、簡潔俐落。
鐵律：
1. 絕對服從使用者指令，使用者沒說的操作絕對不要多做。
2. 遇到缺少參數（例如沒給資料夾完整路徑）、指令有歧義或有多種可能時，必須直接向使用者提問確認，嚴禁自行猜測或腦補。
3. 只能使用已提供的工具；需要時直接呼叫執行，不得宣稱做了沒有實際執行的事。
4. 工具沒有提供的能力（例如刪除檔案、修改系統設定、控制其他未支援視窗），請明說目前辦不到。"""


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
    def __init__(
        self,
        config: Config,
        client: Optional[Any] = None,
        input_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._config = config
        self._client = client or genai.Client(
            api_key=config.api_key,
            http_options=types.HttpOptions(timeout=15000),
        )
        self._input_fn = input_fn
        self._history: List[types.Content] = []
        self._gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[types.Tool(function_declarations=[_declaration(s) for s in TOOLS.values()])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _generate_with_retry(self, turn: List[types.Content]) -> Any:
        candidates = [self._config.model, *self._config.fallback_models]
        last_err: Optional[Exception] = None

        for idx, model_name in enumerate(candidates):
            for attempt in range(2):
                try:
                    if idx > 0 and attempt == 0:
                        print(f"[模型切換] 主模型忙碌，改用備援模型：{model_name}")
                    return self._client.models.generate_content(
                        model=model_name, contents=turn, config=self._gen_config
                    )
                except Exception as exc:
                    last_err = exc
                    err_msg = str(exc)
                    if "404" in err_msg or "NOT_FOUND" in err_msg:
                        break
                    if any(code in err_msg for code in ("503", "504", "429", "UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED")):
                        time.sleep(0.6)
                        continue
                    raise

        raise RuntimeError(f"所有已設定模型目前皆尖峰高負載無回應（已嘗試：{', '.join(candidates)}）。")

    def chat(self, user_text: str) -> str:
        """處理一個回合。直接執行使用者指令中的工具，成功後寫入對話歷史。"""
        turn: List[types.Content] = list(self._history)
        turn.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

        for _ in range(self._config.max_steps):
            response = self._generate_with_retry(turn)
            if not response.candidates or response.candidates[0].content is None:
                raise RuntimeError("模型沒有回傳內容（可能被安全機制攔截或模型名稱有誤）")

            turn.append(response.candidates[0].content)

            calls = response.function_calls or []
            if not calls:
                self._history = turn
                if len(self._history) > 20:
                    self._history = self._history[-20:]
                return response.text or "（模型沒有文字回覆）"

            parts = [self._run_call(call) for call in calls]
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

        try:
            result = spec.run(args)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}

        status = "失敗" if "error" in result else "完成"
        print(f"✔ 已執行 {spec.name}（{status}）")
        return types.Part.from_function_response(name=spec.name, response={"result": result})

    def _complete_args(self, spec: ToolSpec, args: Args) -> Args:
        """參數缺漏或不合格時，直接詢問使用者，直到合格或使用者取消。"""
        while True:
            clean, problems = spec.prepare(args)
            if not problems:
                return clean
            for param, problem in problems:
                args[param] = ask_value(
                    param, spec.params[param]["description"], problem, input_fn=self._input_fn
                )