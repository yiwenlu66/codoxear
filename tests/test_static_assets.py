import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from codoxear.server import CONTENT_SECURITY_POLICY
from codoxear.server import FRONTEND_ASSET_FILES
from codoxear.server import STATIC_ASSET_VERSION_FILES
from codoxear.server import STATIC_ASSET_VERSION_PLACEHOLDER
from codoxear.server import TOP_LEVEL_STATIC_ASSETS
from codoxear.server import _read_static_bytes
from codoxear.server import _static_asset_version
from codoxear.server import _static_cache_control_headers
from codoxear.static_routes import SHELL_ASSET_FILES


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROUTES_PY = ROOT / "codoxear" / "static_routes.py"
INDEX_HTML = ROOT / "codoxear" / "static" / "index.html"
APP_JS = ROOT / "codoxear" / "static" / "app.js"
APP_URL_JS = ROOT / "codoxear" / "static" / "app_url.js"
APP_STORAGE_JS = ROOT / "codoxear" / "static" / "app_storage.js"
APP_PERF_JS = ROOT / "codoxear" / "static" / "app_perf.js"
APP_API_JS = ROOT / "codoxear" / "static" / "app_api.js"
APP_MARKDOWN_JS = ROOT / "codoxear" / "static" / "app_markdown.js"
APP_LAUNCH_JS = ROOT / "codoxear" / "static" / "app_launch.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_DOM_JS = ROOT / "codoxear" / "static" / "app_dom.js"
APP_FILE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
APP_VIEWPORT_JS = ROOT / "codoxear" / "static" / "app_viewport.js"
APP_POLLING_JS = ROOT / "codoxear" / "static" / "app_polling.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"
APP_MESSAGE_IDENTITY_JS = ROOT / "codoxear" / "static" / "app_message_identity.js"
APP_MESSAGE_ROWS_JS = ROOT / "codoxear" / "static" / "app_message_rows.js"
APP_CONVERSATION_COPY_JS = ROOT / "codoxear" / "static" / "app_conversation_copy.js"
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"
APP_CLIPBOARD_JS = ROOT / "codoxear" / "static" / "app_clipboard.js"
APP_VOICE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_voice_helpers.js"


class TestStaticAssets(unittest.TestCase):
    def test_index_html_uses_runtime_asset_version_placeholder(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn(f'window.CODOXEAR_ASSET_VERSION = "{STATIC_ASSET_VERSION_PLACEHOLDER}"', source)
        self.assertIn(f"app.css?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_url.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_storage.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_perf.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_api.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_markdown.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_launch.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_display.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_dom.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_file_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_session_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_viewport.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_polling.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_transcript.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_message_identity.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_message_rows.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_conversation_copy.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_modal.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_clipboard.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app_voice_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertIn(f"app.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}", source)
        self.assertLess(source.index(f"app_url.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_storage.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_storage.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_perf.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_perf.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_api.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_api.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_markdown.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_markdown.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_launch.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_launch.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_display.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_display.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_dom.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_dom.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_file_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_file_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_session_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_session_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_viewport.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_viewport.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_polling.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_polling.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_transcript.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_transcript.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_message_identity.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_message_identity.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_message_rows.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_message_rows.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_conversation_copy.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_conversation_copy.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_modal.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_modal.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_clipboard.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_clipboard.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app_voice_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))
        self.assertLess(source.index(f"app_voice_helpers.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"), source.index(f"app.js?v={STATIC_ASSET_VERSION_PLACEHOLDER}"))

    def test_app_shell_does_not_execute_third_party_assets(self) -> None:
        index = INDEX_HTML.read_text(encoding="utf-8")
        app = APP_JS.read_text(encoding="utf-8")
        app_url = APP_URL_JS.read_text(encoding="utf-8")
        app_storage = APP_STORAGE_JS.read_text(encoding="utf-8")
        app_perf = APP_PERF_JS.read_text(encoding="utf-8")
        app_api = APP_API_JS.read_text(encoding="utf-8")
        app_markdown = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        app_launch = APP_LAUNCH_JS.read_text(encoding="utf-8")
        app_display = APP_DISPLAY_JS.read_text(encoding="utf-8")
        app_dom = APP_DOM_JS.read_text(encoding="utf-8")
        app_file_helpers = APP_FILE_HELPERS_JS.read_text(encoding="utf-8")
        app_session_helpers = APP_SESSION_HELPERS_JS.read_text(encoding="utf-8")
        app_viewport = APP_VIEWPORT_JS.read_text(encoding="utf-8")
        app_polling = APP_POLLING_JS.read_text(encoding="utf-8")
        app_transcript = APP_TRANSCRIPT_JS.read_text(encoding="utf-8")
        app_message_identity = APP_MESSAGE_IDENTITY_JS.read_text(encoding="utf-8")
        app_message_rows = APP_MESSAGE_ROWS_JS.read_text(encoding="utf-8")
        app_conversation_copy = APP_CONVERSATION_COPY_JS.read_text(encoding="utf-8")
        app_modal = APP_MODAL_JS.read_text(encoding="utf-8")
        app_clipboard = APP_CLIPBOARD_JS.read_text(encoding="utf-8")
        app_voice_helpers = APP_VOICE_HELPERS_JS.read_text(encoding="utf-8")
        static_routes_source = STATIC_ROUTES_PY.read_text(encoding="utf-8")
        self.assertIn("Content-Security-Policy", index)
        self.assertIn("handler.send_header(\"Content-Security-Policy\", deps.content_security_policy)", static_routes_source)
        self.assertIn("handler.send_header(\"X-Frame-Options\", \"DENY\")", static_routes_source)
        self.assertIn("frame-ancestors 'none'", CONTENT_SECURITY_POLICY)
        for forbidden in ["fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net", "unpkg.com"]:
            self.assertNotIn(forbidden, index)
            self.assertNotIn(forbidden, app)
            self.assertNotIn(forbidden, app_url)
            self.assertNotIn(forbidden, app_storage)
            self.assertNotIn(forbidden, app_perf)
            self.assertNotIn(forbidden, app_api)
            self.assertNotIn(forbidden, app_markdown)
            self.assertNotIn(forbidden, app_launch)
            self.assertNotIn(forbidden, app_display)
            self.assertNotIn(forbidden, app_dom)
            self.assertNotIn(forbidden, app_file_helpers)
            self.assertNotIn(forbidden, app_session_helpers)
            self.assertNotIn(forbidden, app_viewport)
            self.assertNotIn(forbidden, app_polling)
            self.assertNotIn(forbidden, app_transcript)
            self.assertNotIn(forbidden, app_message_identity)
            self.assertNotIn(forbidden, app_message_rows)
            self.assertNotIn(forbidden, app_conversation_copy)
            self.assertNotIn(forbidden, app_modal)
            self.assertNotIn(forbidden, app_clipboard)
            self.assertNotIn(forbidden, app_voice_helpers)
        self.assertNotIn('src="https://', index)
        self.assertNotIn('href="https://', index)
        self.assertIn("script-src 'self' 'unsafe-inline'", index)
        self.assertIn("connect-src 'self'", index)
        self.assertIn("connect-src 'self'", CONTENT_SECURITY_POLICY)
        self.assertIn('const base = resolveAppUrl("monaco/vs");', app)
        self.assertIn('import(resolveAppUrl("pdf.mjs"))', app)

    def test_frontend_asset_manifest_drives_version_files(self) -> None:
        self.assertEqual(STATIC_ASSET_VERSION_FILES, FRONTEND_ASSET_FILES + SHELL_ASSET_FILES)

    def test_versioned_index_assets_exist_and_are_registered(self) -> None:
        source = INDEX_HTML.read_text(encoding="utf-8")
        versioned_refs = re.findall(r'(?:src|href)="([^"?]+)\?v=__CODOXEAR_ASSET_VERSION__"', source)
        self.assertEqual(set(versioned_refs), set(FRONTEND_ASSET_FILES + ("favicon.png", "manifest.webmanifest")))
        routes = dict(TOP_LEVEL_STATIC_ASSETS)
        for name in versioned_refs:
            self.assertTrue((ROOT / "codoxear" / "static" / name).is_file(), name)
            self.assertEqual(routes.get(f"/{name}"), name)

    def test_service_worker_registration_uses_asset_version(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function versionedShellAssetPath(path)", source)
        self.assertIn('return `${path}?v=${encodeURIComponent(version)}`;', source)
        self.assertIn('navigator.serviceWorker.register(resolveAppUrl(versionedShellAssetPath("/service-worker.js")), { scope: resolveAppUrl("/") });', source)
        self.assertNotIn('navigator.serviceWorker.register(resolveAppUrl("/service-worker.js")', source)

    def test_static_asset_version_changes_when_frontend_assets_change(self) -> None:
        initial_content = {
            "app_url.js": "window.CodoxearUrls = {};\n",
            "app_storage.js": "window.CodoxearStorage = {};\n",
            "app_perf.js": "window.CodoxearPerf = {};\n",
            "app_api.js": "window.CodoxearApi = {};\n",
            "app_markdown.js": "window.CodoxearMarkdown = {};\n",
            "app_launch.js": "window.CodoxearLaunch = {};\n",
            "app_display.js": "window.CodoxearDisplay = {};\n",
            "app_dom.js": "window.CodoxearDom = {};\n",
            "app_file_helpers.js": "window.CodoxearFileHelpers = {};\n",
            "app_session_helpers.js": "window.CodoxearSessionHelpers = {};\n",
            "app_viewport.js": "window.CodoxearViewport = {};\n",
            "app_polling.js": "window.CodoxearPolling = {};\n",
            "app_transcript.js": "window.CodoxearTranscript = {};\n",
            "app_message_identity.js": "window.CodoxearMessageIdentity = {};\n",
            "app_message_rows.js": "window.CodoxearMessageRows = {};\n",
            "app_conversation_copy.js": "window.CodoxearConversationCopy = {};\n",
            "app_modal.js": "window.CodoxearModal = {};\n",
            "app_clipboard.js": "window.CodoxearClipboard = {};\n",
            "app_voice_helpers.js": "window.CodoxearVoiceHelpers = {};\n",
            "app.js": "console.log('one');\n",
            "app.css": "body { color: black; }\n",
            "favicon.png": "png bytes\n",
            "manifest.webmanifest": '{"name":"one"}\n',
            "service-worker.js": "self.addEventListener('push', () => {});\n",
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, content in initial_content.items():
                (root / name).write_text(content, encoding="utf-8")
            versions = [_static_asset_version(root)]
            for name in STATIC_ASSET_VERSION_FILES:
                (root / name).write_text(initial_content[name] + "/* changed */\n", encoding="utf-8")
                versions.append(_static_asset_version(root))
            self.assertEqual(len(versions), len(set(versions)))

    def test_read_static_bytes_replaces_html_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "app.js").write_text("console.log('x');\n", encoding="utf-8")
            (root / "app.css").write_text("body { color: black; }\n", encoding="utf-8")
            index = root / "index.html"
            index.write_text(
                (
                    '<script>window.CODOXEAR_ASSET_VERSION = "__CODOXEAR_ASSET_VERSION__";</script>\n'
                    '<link rel="icon" type="image/png" href="favicon.png?v=__CODOXEAR_ASSET_VERSION__" />\n'
                    '<link rel="manifest" href="manifest.webmanifest?v=__CODOXEAR_ASSET_VERSION__" />\n'
                    '<link rel="stylesheet" href="app.css?v=__CODOXEAR_ASSET_VERSION__" />\n'
                    '<script src="app_url.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_storage.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_perf.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_api.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_markdown.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_launch.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_display.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_dom.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_file_helpers.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_session_helpers.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_viewport.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_polling.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_transcript.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_message_identity.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_message_rows.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_conversation_copy.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_modal.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_clipboard.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_voice_helpers.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                ),
                encoding="utf-8",
            )
            rendered = _read_static_bytes(index).decode("utf-8")
            version = _static_asset_version(root)
            self.assertNotIn(STATIC_ASSET_VERSION_PLACEHOLDER, rendered)
            self.assertIn(f'window.CODOXEAR_ASSET_VERSION = "{version}"', rendered)
            self.assertIn(f"favicon.png?v={version}", rendered)
            self.assertIn(f"manifest.webmanifest?v={version}", rendered)
            self.assertIn(f"app.css?v={version}", rendered)
            self.assertIn(f"app_url.js?v={version}", rendered)
            self.assertIn(f"app_storage.js?v={version}", rendered)
            self.assertIn(f"app_perf.js?v={version}", rendered)
            self.assertIn(f"app_api.js?v={version}", rendered)
            self.assertIn(f"app_markdown.js?v={version}", rendered)
            self.assertIn(f"app_launch.js?v={version}", rendered)
            self.assertIn(f"app_display.js?v={version}", rendered)
            self.assertIn(f"app_dom.js?v={version}", rendered)
            self.assertIn(f"app_file_helpers.js?v={version}", rendered)
            self.assertIn(f"app_session_helpers.js?v={version}", rendered)
            self.assertIn(f"app_viewport.js?v={version}", rendered)
            self.assertIn(f"app_polling.js?v={version}", rendered)
            self.assertIn(f"app_transcript.js?v={version}", rendered)
            self.assertIn(f"app_message_identity.js?v={version}", rendered)
            self.assertIn(f"app_message_rows.js?v={version}", rendered)
            self.assertIn(f"app_conversation_copy.js?v={version}", rendered)
            self.assertIn(f"app_modal.js?v={version}", rendered)
            self.assertIn(f"app_clipboard.js?v={version}", rendered)
            self.assertIn(f"app_voice_helpers.js?v={version}", rendered)
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

    def test_top_level_static_routes_are_registry_driven(self) -> None:
        routes = dict(TOP_LEVEL_STATIC_ASSETS)
        for name in FRONTEND_ASSET_FILES:
            self.assertEqual(routes.get(f"/{name}"), name)
        self.assertEqual(routes.get("/favicon.ico"), "favicon.png")
        self.assertEqual(routes.get("/manifest.webmanifest"), "manifest.webmanifest")
        self.assertEqual(routes.get("/service-worker.js"), "service-worker.js")
        self.assertEqual(routes.get("/favicon.png"), "favicon.png")
        self.assertEqual(routes.get("/"), "index.html")
        self.assertEqual(len(routes), len(TOP_LEVEL_STATIC_ASSETS))

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
        self.assertIn("codoxear/static/app_url.js", names)
        self.assertIn("codoxear/static/app_storage.js", names)
        self.assertIn("codoxear/static/app_perf.js", names)
        self.assertIn("codoxear/static/app_api.js", names)
        self.assertIn("codoxear/static/app_markdown.js", names)
        self.assertIn("codoxear/static/app_launch.js", names)
        self.assertIn("codoxear/static/app_display.js", names)
        self.assertIn("codoxear/static/app_dom.js", names)
        self.assertIn("codoxear/static/app_file_helpers.js", names)
        self.assertIn("codoxear/static/app_session_helpers.js", names)
        self.assertIn("codoxear/static/app_viewport.js", names)
        self.assertIn("codoxear/static/app_polling.js", names)
        self.assertIn("codoxear/static/app_transcript.js", names)
        self.assertIn("codoxear/static/app_message_identity.js", names)
        self.assertIn("codoxear/static/app_message_rows.js", names)
        self.assertIn("codoxear/static/app_conversation_copy.js", names)
        self.assertIn("codoxear/static/app_modal.js", names)
        self.assertIn("codoxear/static/app_clipboard.js", names)
        self.assertIn("codoxear/static/app_voice_helpers.js", names)
        self.assertIn("codoxear/static/logos/codex.svg", names)
        self.assertIn("codoxear/static/logos/pi.svg", names)
        self.assertIn("codoxear/static/logos/cc.svg", names)


if __name__ == "__main__":
    unittest.main()
