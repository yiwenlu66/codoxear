import tempfile
import unittest
from pathlib import Path

from codoxear.server import STATIC_ASSET_VERSION_PLACEHOLDER
from codoxear.server import _read_static_bytes
from codoxear.server import _static_asset_version


INDEX_HTML = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "index.html"


class TestStaticAssets(unittest.TestCase):
    def test_index_html_uses_runtime_asset_version_placeholder(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn(f'window.CODOXEAR_ASSET_VERSION = "{STATIC_ASSET_VERSION_PLACEHOLDER}"', source)
        self.assertIn(f"app.css?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)

    def test_static_asset_version_changes_when_app_js_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "app.js").write_text("console.log('one');\n", encoding="utf-8")
            (root / "app.css").write_text("body { color: black; }\n", encoding="utf-8")
            before = _static_asset_version(root)
            (root / "app.js").write_text("console.log('two');\n", encoding="utf-8")
            after = _static_asset_version(root)
            self.assertNotEqual(before, after)

    def test_read_static_bytes_replaces_html_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "app.js").write_text("console.log('x');\n", encoding="utf-8")
            (root / "app.css").write_text("body { color: black; }\n", encoding="utf-8")
            index = root / "index.html"
            index.write_text(
                (
                    '<script>window.CODOXEAR_ASSET_VERSION = "__CODOXEAR_ASSET_VERSION__";</script>\n'
                    '<link rel="stylesheet" href="app.css?v=__CODOXEAR_ASSET_VERSION__" />\n'
                    '<script src="app.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                ),
                encoding="utf-8",
            )
            rendered = _read_static_bytes(index).decode("utf-8")
            version = _static_asset_version(root)
            self.assertNotIn(STATIC_ASSET_VERSION_PLACEHOLDER, rendered)
            self.assertIn(f'window.CODOXEAR_ASSET_VERSION = "{version}"', rendered)
            self.assertIn(f"app.css?v={version}", rendered)
            self.assertIn(f"app.js?v={version}", rendered)


if __name__ == "__main__":
    unittest.main()
