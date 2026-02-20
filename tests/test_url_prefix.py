import unittest

from codoxear.server import _normalize_url_prefix, _strip_url_prefix


class TestUrlPrefix(unittest.TestCase):
    def test_normalize_empty(self) -> None:
        self.assertEqual(_normalize_url_prefix(None), "")
        self.assertEqual(_normalize_url_prefix(""), "")
        self.assertEqual(_normalize_url_prefix("   "), "")
        self.assertEqual(_normalize_url_prefix("/"), "")

    def test_normalize_trailing_slash(self) -> None:
        self.assertEqual(_normalize_url_prefix("/codoxear"), "/codoxear")
        self.assertEqual(_normalize_url_prefix("/codoxear/"), "/codoxear")
        self.assertEqual(_normalize_url_prefix("/codoxear///"), "/codoxear")

    def test_normalize_rejects_non_path(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_url_prefix("codoxear")
        with self.assertRaises(ValueError):
            _normalize_url_prefix("https://example.com/codoxear")
        with self.assertRaises(ValueError):
            _normalize_url_prefix("/codoxear?x=1")
        with self.assertRaises(ValueError):
            _normalize_url_prefix("/codoxear#x")

    def test_strip_prefix(self) -> None:
        self.assertEqual(_strip_url_prefix("", "/api/me"), "/api/me")
        self.assertEqual(_strip_url_prefix("/codoxear", "/codoxear"), "/")
        self.assertEqual(_strip_url_prefix("/codoxear", "/codoxear/"), "/")
        self.assertEqual(_strip_url_prefix("/codoxear", "/codoxear/api/me"), "/api/me")
        self.assertIsNone(_strip_url_prefix("/codoxear", "/api/me"))


if __name__ == "__main__":
    unittest.main()

