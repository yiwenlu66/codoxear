import unittest

from codoxear.server import _match_session_route


class TestSessionRouteMatcher(unittest.TestCase):
    def test_matches_exact_session_route_shape(self) -> None:
        self.assertEqual(_match_session_route("/api/sessions/abc/messages/search", "messages", "search"), "abc")
        self.assertEqual(_match_session_route("/api/sessions/a%2Fb/file/read", "file", "read"), "a%2Fb")
        self.assertEqual(_match_session_route("/api/sessions/sid/tail", "tail"), "sid")

    def test_rejects_extra_missing_or_wrong_segments(self) -> None:
        bad_paths = [
            "/api/sessions/abc/messages/search/extra",
            "/api/sessions/abc/messages",
            "/api/sessions//messages/search",
            "/api/session/abc/messages/search",
            "/v1/api/sessions/abc/messages/search",
            "/api/sessions/abc/messages/tail",
            "/api/sessions/abc/messages/search/",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                self.assertIsNone(_match_session_route(path, "messages", "search"))


if __name__ == "__main__":
    unittest.main()
