import unittest
from io import BytesIO
from unittest.mock import patch

from codoxear import server


class _Handler:
    def __init__(self, *, cookie: str | None = None) -> None:
        self.headers: dict[str, str] = {"Cookie": cookie} if cookie is not None else {}
        self.sent: list[tuple[str, str]] = []
        self.status: int | None = None
        self.ended = False
        self.wfile = BytesIO()

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.sent.append((name, value))

    def end_headers(self) -> None:
        self.ended = True


class TestAuthCookie(unittest.TestCase):
    def test_login_cookie_has_no_ttl_and_verifies_across_time(self) -> None:
        handler = _Handler()

        server._set_auth_cookie(handler)  # type: ignore[arg-type]

        cookie = next(value for name, value in handler.sent if name == "Set-Cookie")
        self.assertIn("codoxear_auth=", cookie)
        self.assertIn("Expires=Fri, 31 Dec 9999 23:59:59 GMT", cookie)
        self.assertNotIn("Max-Age", cookie)

        token = cookie.split("codoxear_auth=", 1)[1].split(";", 1)[0]
        with patch("codoxear.server._now", return_value=10**12):
            self.assertEqual(server._verify_cookie(token), {"v": 1})

    def test_login_cookie_rejects_tampered_payload(self) -> None:
        token = server._sign_cookie({"v": 1})
        payload, sig = token.split(".", 1)
        self.assertIsNone(server._verify_cookie(f"{payload}x.{sig}"))

    def test_password_compare_wrapper_uses_constant_time_compare(self) -> None:
        with patch.dict("os.environ", {"CODEX_WEB_PASSWORD": "test-password"}):
            old_cache = server._PASSWORD_CACHE
            try:
                server._PASSWORD_CACHE = None
                self.assertTrue(server._is_same_password("test-password"))
                self.assertFalse(server._is_same_password("wrong-password"))
            finally:
                server._PASSWORD_CACHE = old_cache

    def test_legacy_exp_cookie_is_refreshed_without_time_check(self) -> None:
        token = server._sign_cookie({"exp": 1})
        handler = _Handler(cookie=f"codoxear_auth={token}")

        with patch("codoxear.server._now", return_value=10**12):
            self.assertTrue(server._require_auth(handler))  # type: ignore[arg-type]
        self.assertTrue(getattr(handler, "_codoxear_refresh_auth_cookie"))

        server._json_response(handler, 200, {"ok": True})  # type: ignore[arg-type]

        refreshed_cookie = next(value for name, value in handler.sent if name == "Set-Cookie")
        self.assertIn("Expires=Fri, 31 Dec 9999 23:59:59 GMT", refreshed_cookie)
        self.assertNotIn("Max-Age", refreshed_cookie)
        refreshed_token = refreshed_cookie.split("codoxear_auth=", 1)[1].split(";", 1)[0]
        self.assertEqual(server._verify_cookie(refreshed_token), {"v": 1})

    def test_json_response_with_etag_returns_304_for_matching_if_none_match(self) -> None:
        first = _Handler()
        payload = {"sessions": [], "recent_cwds": []}

        server._json_response_with_etag(first, payload)  # type: ignore[arg-type]

        self.assertEqual(first.status, 200)
        etag = next(value for name, value in first.sent if name == "ETag")
        self.assertTrue(first.wfile.getvalue())

        second = _Handler()
        second.headers["If-None-Match"] = etag

        server._json_response_with_etag(second, payload)  # type: ignore[arg-type]

        self.assertEqual(second.status, 304)
        self.assertEqual(second.wfile.getvalue(), b"")
        self.assertIn(("ETag", etag), second.sent)
        self.assertIn(("Content-Length", "0"), second.sent)


if __name__ == "__main__":
    unittest.main()
