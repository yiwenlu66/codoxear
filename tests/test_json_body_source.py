import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
SERVER_HANDLER_PY = ROOT / "codoxear" / "server_handler.py"
SERVER_ROUTE_DEPS_PY = ROOT / "codoxear" / "server_route_deps.py"
SERVER_HTTP_PY = ROOT / "codoxear" / "server_http.py"
AUTH_ROUTES_PY = ROOT / "codoxear" / "auth_routes.py"
CONTROL_ROUTES_PY = ROOT / "codoxear" / "control_routes.py"


class TestJsonBodySource(unittest.TestCase):
    def test_post_json_parsing_uses_bad_request_helper(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        handler_source = SERVER_HANDLER_PY.read_text(encoding="utf-8")
        route_deps_source = SERVER_ROUTE_DEPS_PY.read_text(encoding="utf-8")
        helper_start = handler_source.index("    def _read_json_body(")
        helper_end = handler_source.index("    def _handle_voice_post", helper_start)
        post_start = handler_source.index("    def do_POST(self) -> None:")
        post_block = handler_source[post_start:]
        helper_block = handler_source[helper_start:helper_end]
        auth_source = AUTH_ROUTES_PY.read_text(encoding="utf-8")
        control_source = CONTROL_ROUTES_PY.read_text(encoding="utf-8")

        self.assertIn("raise self.deps.bad_request_error(\"invalid json body\")", helper_block)
        self.assertIn("raise self.deps.bad_request_error(\"empty request body\")", helper_block)
        self.assertIn("except self.deps.request_payload_too_large_error as exc:", helper_block)
        self.assertNotIn("body_text = body.decode(\"utf-8\")", post_block)
        self.assertNotIn("json.loads(body_text)", post_block)
        self.assertIn("read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs)", route_deps_source)
        self.assertIn("obj = deps.read_json_body(handler)", auth_source)
        self.assertIn("too_large_error=f\"file too large (max {deps.attach_upload_max_bytes} bytes)\"", control_source)

    def test_500_responses_omit_trace_without_debug_flag(self) -> None:
        source = SERVER_HTTP_PY.read_text(encoding="utf-8")
        handler_start = source.index("def handle_route_exception")
        handler_end = source.index("def json_response", handler_start)
        handler_block = source[handler_start:handler_end]

        self.assertIn("if os.environ.get(\"CODEX_WEB_DEBUG_ERRORS\") == \"1\":", handler_block)
        self.assertIn('payload["trace"] = traceback.format_exc()', handler_block)
        self.assertNotIn('{"error": str(exc), "trace": traceback.format_exc()}', handler_block)


if __name__ == "__main__":
    unittest.main()
