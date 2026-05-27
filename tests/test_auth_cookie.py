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


if __name__ == "__main__":
    unittest.main()
