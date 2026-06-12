import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"
APP_JS = ROOT / "codoxear" / "static" / "app.js"


class TestMobileZoomAccessibilitySource(unittest.TestCase):
    def test_viewport_does_not_disable_user_scaling(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn('name="viewport"', html)
        self.assertIn("width=device-width", html)
        self.assertIn("initial-scale=1", html)
        self.assertIn("viewport-fit=cover", html)
        self.assertNotIn("maximum-scale", html)
        self.assertNotIn("user-scalable=no", html)

    def test_app_does_not_block_pinch_zoom_gestures(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn('"gesturestart"', source)
        self.assertNotIn('"gesturechange"', source)
        self.assertNotIn('"gestureend"', source)
        self.assertNotIn("e.touches && e.touches.length > 1", source)
        self.assertNotIn("Best-effort zoom disable", source)


if __name__ == "__main__":
    unittest.main()
