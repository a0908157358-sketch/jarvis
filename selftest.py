"""離線自我測試（不呼叫 Gemini、不需要 API 金鑰）。執行：python selftest.py

agent 的測試需要 google-genai；沒安裝時會自動略過。
"""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from jarvis_core.permissions import PendingAction, UserAbort, ask_permission, ask_value
from jarvis_core.tools import TOOLS

ACTION = PendingAction("t", "做事", "目標", "影響")


def feed(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


class PermissionTests(unittest.TestCase):
    def test_approve_words(self):
        for word in ("允許", "同意", "執行"):
            with redirect_stdout(io.StringIO()):
                ask_permission(ACTION, feed(word))

    def test_deny_words_abort(self):
        for word in ("拒絕", "取消"):
            with redirect_stdout(io.StringIO()), self.assertRaises(UserAbort):
                ask_permission(ACTION, feed(word))

    def test_ambiguous_is_not_approval(self):
        with redirect_stdout(io.StringIO()):
            ask_permission(ACTION, feed("好", "", "y", "執行"))  # 前三個都要被重問

    def test_eof_aborts(self):
        def boom(prompt):
            raise EOFError
        with redirect_stdout(io.StringIO()), self.assertRaises(UserAbort):
            ask_permission(ACTION, boom)

    def test_ask_value(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ask_value("path", "d", "缺少", feed("", "C:\\x")), "C:\\x")
            with self.assertRaises(UserAbort):
                ask_value("path", "d", "缺少", feed("取消"))


class ToolTests(unittest.TestCase):
    def test_time(self):
        spec = TOOLS["get_current_time"]
        args, problems = spec.prepare({})
        self.assertEqual(problems, [])
        result = spec.run(args)
        self.assertTrue({"iso", "weekday", "utc_offset"} <= set(result))

    def test_list_missing_path(self):
        _, problems = TOOLS["list_folder_preview"].prepare({})
        self.assertEqual(problems[0][0], "path")

    def test_list_relative_path_rejected(self):
        for p in ("Desktop\\Jarvis", "~/x", "."):
            _, problems = TOOLS["list_folder_preview"].prepare({"path": p})
            self.assertEqual([x[0] for x in problems], ["path"], p)

    def test_list_max_items(self):
        spec = TOOLS["list_folder_preview"]
        base = os.path.abspath(os.sep)
        for bad in (0, 201, "abc", 2.5):
            _, problems = spec.prepare({"path": base, "max_items": bad})
            self.assertEqual([x[0] for x in problems], ["max_items"], bad)
        clean, problems = spec.prepare({"path": base, "max_items": 5.0})  # protobuf 常把整數變 float
        self.assertEqual((problems, clean["max_items"]), ([], 5))
        clean, _ = spec.prepare({"path": base})
        self.assertEqual(clean["max_items"], 50)

    def test_list_run(self):
        spec = TOOLS["list_folder_preview"]
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "資料夾"))
            for name in ("b.txt", "a.txt", "測試.md"):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write("hello")
            args, _ = spec.prepare({"path": d, "max_items": 3})
            result = spec.run(args)
            self.assertEqual((result["total"], result["shown"], result["truncated"]), (4, 3, True))
            self.assertEqual(result["items"][0]["type"], "folder")  # 資料夾排最前
            self.assertEqual(result["items"][1]["name"], "a.txt")
            self.assertEqual(result["items"][1]["size_bytes"], 5)
            self.assertEqual(spec.describe(args).target, d)
            self.assertEqual(len(os.listdir(d)), 4)  # 沒有改動資料夾

    def test_list_not_found_and_not_dir(self):
        spec = TOOLS["list_folder_preview"]
        with tempfile.TemporaryDirectory() as d:
            missing, _ = spec.prepare({"path": os.path.join(d, "nope")})
            self.assertIn("error", spec.run(missing))
            f = os.path.join(d, "f.txt")
            open(f, "w").close()
            notdir, _ = spec.prepare({"path": f})
            self.assertIn("error", spec.run(notdir))


try:
    from google.genai import types as gtypes  # noqa: F401
    HAVE_GENAI = True
except ImportError:
    HAVE_GENAI = False


@unittest.skipUnless(HAVE_GENAI, "未安裝 google-genai，略過 agent 測試")
class AgentLoopTests(unittest.TestCase):
    def _agent(self, responses):
        from jarvis_core.agent import JarvisAgent
        from jarvis_core.config import Config

        calls = []

        def generate_content(model, contents, config):
            calls.append(list(contents))
            return responses.pop(0)

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        return JarvisAgent(Config("k", "m", 4), client=client), calls

    @staticmethod
    def _resp(text=None, call=None):
        from google.genai import types as t
        part = t.Part(function_call=t.FunctionCall(name=call, args={})) if call else t.Part(text=text)
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=t.Content(role="model", parts=[part]))],
            function_calls=[SimpleNamespace(name=call, args={})] if call else [],
            text=text,
        )

    def test_tool_call_then_answer(self):
        agent, calls = self._agent([self._resp(call="get_current_time"), self._resp(text="現在是…")])
        with redirect_stdout(io.StringIO()), mock.patch("builtins.input", side_effect=["允許"]):
            self.assertEqual(agent.chat("現在幾點"), "現在是…")
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[1][-1].parts[0].function_response)
        self.assertEqual(len(agent._history), 4)

    def test_denied_turn_is_discarded(self):
        agent, _ = self._agent([self._resp(call="get_current_time")])
        with redirect_stdout(io.StringIO()), mock.patch("builtins.input", side_effect=["拒絕"]):
            with self.assertRaises(UserAbort):
                agent.chat("現在幾點")
        self.assertEqual(agent._history, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
