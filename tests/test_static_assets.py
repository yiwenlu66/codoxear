import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from codoxear.server import STATIC_ASSET_VERSION_PLACEHOLDER
from codoxear.server import _read_static_bytes
from codoxear.server import _static_asset_version
from codoxear.server import _static_cache_control_headers


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"
APP_JS = ROOT / "codoxear" / "static" / "app.js"


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

    def test_static_cache_headers_default_to_no_store(self) -> None:
        self.assertEqual(
            _static_cache_control_headers(enabled=False),
            {"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"},
        )

    def test_static_cache_headers_can_be_immutable(self) -> None:
        self.assertEqual(
            _static_cache_control_headers(enabled=True),
            {"Cache-Control": "public, max-age=31536000, immutable"},
        )

    def test_sidebar_logo_uses_url_prefix_safe_relative_path(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('src="static/codoxear-icon.png"', source)
        self.assertNotIn('src="/static/codoxear-icon.png"', source)

    def test_refresh_sessions_does_not_rebuild_backend_tabs_while_modal_is_open(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index('if (newSessionViewer.style.display === "flex") {')
        end = source.index("fileRefCandidateCache.clear();", start)
        block = source[start:end]
        self.assertNotIn("renderNewSessionBackendTabs();", block)
        self.assertNotIn("renderNewSessionProviderMenu();", block)
        self.assertIn("renderNewSessionModelMenu();", block)
        self.assertIn("renderNewSessionReasoningMenu();", block)

    def test_wheel_includes_nested_logo_assets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            outdir = root / "wheelhouse"
            src.mkdir()
            outdir.mkdir()
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(ROOT / name, src / name)
            shutil.copytree(ROOT / "codoxear", src / "codoxear", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
            subprocess.run(
                ["python3", "-m", "pip", "wheel", str(src), "-w", str(outdir), "--no-deps"],
                check=True,
                cwd=src,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            wheel = next(outdir.glob("codoxear-*.whl"))
            with zipfile.ZipFile(wheel) as zf:
                names = set(zf.namelist())
        self.assertIn("codoxear/static/logos/codex.svg", names)
        self.assertIn("codoxear/static/logos/pi.svg", names)
        self.assertIn("codoxear/static/logos/cc.svg", names)


if __name__ == "__main__":
    unittest.main()
