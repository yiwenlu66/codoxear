import unittest
from pathlib import Path


SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"


class TestJsonBodySource(unittest.TestCase):
    def test_post_json_parsing_uses_bad_request_helper(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        helper_start = source.index("    def _read_json_body(")
        helper_end = source.index("    def _handle_voice_post", helper_start)
        post_start = source.index("    def do_POST(self) -> None:")
        post_block = source[post_start:]
        helper_block = source[helper_start:helper_end]

        self.assertIn("raise BadRequestError(\"invalid json body\")", helper_block)
        self.assertIn("raise BadRequestError(\"empty request body\")", helper_block)
        self.assertIn("except RequestPayloadTooLargeError as e:", helper_block)
        self.assertNotIn("body_text = body.decode(\"utf-8\")", post_block)
        self.assertNotIn("json.loads(body_text)", post_block)
        self.assertIn("obj = self._read_json_body()", post_block)
        self.assertIn("too_large_error=f\"file too large (max {ATTACH_UPLOAD_MAX_BYTES} bytes)\"", post_block)

    def test_500_responses_omit_trace_without_debug_flag(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        handler_start = source.index("def _handle_route_exception")
        handler_end = source.index("def _json_response", handler_start)
        handler_block = source[handler_start:handler_end]

        self.assertIn("if os.environ.get(\"CODEX_WEB_DEBUG_ERRORS\") == \"1\":", handler_block)
        self.assertIn('payload["trace"] = traceback.format_exc()', handler_block)
        self.assertNotIn('{"error": str(exc), "trace": traceback.format_exc()}', handler_block)


if __name__ == "__main__":
    unittest.main()
