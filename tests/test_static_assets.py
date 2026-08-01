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
from codoxear.static_routes import UI_IMAGE_ASSET_FILES
from codoxear.static_routes import static_route_asset


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
APP_NEW_SESSION_JS = ROOT / "codoxear" / "static" / "app_new_session.js"
APP_DISPLAY_JS = ROOT / "codoxear" / "static" / "app_display.js"
APP_DOM_JS = ROOT / "codoxear" / "static" / "app_dom.js"
APP_FILE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_file_helpers.js"
APP_FILE_PICKER_JS = ROOT / "codoxear" / "static" / "app_file_picker.js"
APP_FILE_VIEWER_JS = ROOT / "codoxear" / "static" / "app_file_viewer.js"
APP_FILE_EDITOR_JS = ROOT / "codoxear" / "static" / "app_file_editor.js"
APP_SESSION_HELPERS_JS = ROOT / "codoxear" / "static" / "app_session_helpers.js"
APP_VIEWPORT_JS = ROOT / "codoxear" / "static" / "app_viewport.js"
APP_POLLING_JS = ROOT / "codoxear" / "static" / "app_polling.js"
APP_TRANSCRIPT_JS = ROOT / "codoxear" / "static" / "app_transcript.js"
APP_MESSAGE_IDENTITY_JS = ROOT / "codoxear" / "static" / "app_message_identity.js"
APP_MESSAGE_ROWS_JS = ROOT / "codoxear" / "static" / "app_message_rows.js"
APP_CONVERSATION_COPY_JS = ROOT / "codoxear" / "static" / "app_conversation_copy.js"
APP_MODAL_JS = ROOT / "codoxear" / "static" / "app_modal.js"
APP_CLIPBOARD_JS = ROOT / "codoxear" / "static" / "app_clipboard.js"
APP_CODE_COPY_JS = ROOT / "codoxear" / "static" / "app_code_copy.js"
APP_HINT_MODE_JS = ROOT / "codoxear" / "static" / "app_hint_mode.js"
APP_VOICE_HELPERS_JS = ROOT / "codoxear" / "static" / "app_voice_helpers.js"
APP_VOICE_JS = ROOT / "codoxear" / "static" / "app_voice.js"
APP_QUEUE_JS = ROOT / "codoxear" / "static" / "app_queue.js"
APP_DIAGNOSTICS_JS = ROOT / "codoxear" / "static" / "app_diagnostics.js"
APP_RECOVERY_JS = ROOT / "codoxear" / "static" / "app_recovery.js"
APP_UNATTENDED_JS = ROOT / "codoxear" / "static" / "app_unattended.js"
APP_CHAT_NAVIGATION_JS = ROOT / "codoxear" / "static" / "app_chat_navigation.js"
APP_CHAT_SEARCH_JS = ROOT / "codoxear" / "static" / "app_chat_search.js"
APP_SHELL_JS = ROOT / "codoxear" / "static" / "app_shell.js"
APP_COMPOSER_JS = ROOT / "codoxear" / "static" / "app_composer.js"


class TestStaticAssets(unittest.TestCase):
    def test_frontend_asset_manifest_drives_version_files(self) -> None:
        self.assertEqual(STATIC_ASSET_VERSION_FILES, FRONTEND_ASSET_FILES + SHELL_ASSET_FILES + UI_IMAGE_ASSET_FILES)

    def test_static_asset_version_changes_when_monaco_assets_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "app.js").write_text("console.log('x');\n", encoding="utf-8")
            monaco_loader = root / "monaco" / "vs" / "loader.js"
            monaco_worker = root / "monaco" / "vs" / "assets" / "editor.worker-test.js"
            monaco_loader.parent.mkdir(parents=True, exist_ok=True)
            monaco_worker.parent.mkdir(parents=True, exist_ok=True)
            monaco_loader.write_text("loader one\n", encoding="utf-8")
            monaco_worker.write_text("worker one\n", encoding="utf-8")
            first = _static_asset_version(root)
            _static_asset_version.cache_clear()
            monaco_worker.write_text("worker two\n", encoding="utf-8")
            second = _static_asset_version(root)
            self.assertNotEqual(first, second)

    def test_static_asset_version_changes_when_frontend_assets_change(self) -> None:
        initial_content = {
            "app_url.js": "window.CodoxearUrls = {};\n",
            "app_storage.js": "window.CodoxearStorage = {};\n",
            "app_perf.js": "window.CodoxearPerf = {};\n",
            "app_api.js": "window.CodoxearApi = {};\n",
            "app_markdown.js": "window.CodoxearMarkdown = {};\n",
            "app_launch.js": "window.CodoxearLaunch = {};\n",
            "app_new_session.js": "window.CodoxearNewSession = {};\n",
            "app_display.js": "window.CodoxearDisplay = {};\n",
            "app_dom.js": "window.CodoxearDom = {};\n",
            "app_file_helpers.js": "window.CodoxearFileHelpers = {};\n",
            "app_file_picker.js": "window.CodoxearFilePicker = {};\n",
            "app_file_viewer.js": "window.CodoxearFileViewer = {};\n",
            "app_file_editor.js": "window.CodoxearFileEditor = {};\n",
            "app_session_helpers.js": "window.CodoxearSessionHelpers = {};\n",
            "app_viewport.js": "window.CodoxearViewport = {};\n",
            "app_polling.js": "window.CodoxearPolling = {};\n",
            "app_transcript.js": "window.CodoxearTranscript = {};\n",
            "app_message_identity.js": "window.CodoxearMessageIdentity = {};\n",
            "app_message_rows.js": "window.CodoxearMessageRows = {};\n",
            "app_conversation_copy.js": "window.CodoxearConversationCopy = {};\n",
            "app_modal.js": "window.CodoxearModal = {};\n",
            "app_clipboard.js": "window.CodoxearClipboard = {};\n",
            "app_code_copy.js": "window.CodoxearCodeCopy = {};\n",
            "app_hint_mode.js": "window.CodoxearHintMode = {};\n",
            "app_voice_helpers.js": "window.CodoxearVoiceHelpers = {};\n",
            "app_voice.js": "window.CodoxearVoice = {};\n",
            "app_queue.js": "window.CodoxearQueue = {};\n",
            "app_diagnostics.js": "window.CodoxearDiagnostics = {};\n",
            "app_recovery.js": "window.CodoxearRecovery = {};\n",
            "app_unattended.js": "window.CodoxearUnattended = {};\n",
            "app_chat_navigation.js": "window.CodoxearChatNavigation = {};\n",
            "app_chat_search.js": "window.CodoxearChatSearch = {};\n",
            "app_shell.js": "window.CodoxearShell = {};\n",
            "app_composer.js": "window.CodoxearComposer = {};\n",
            "app.js": "console.log('one');\n",
            "app.css": "body { color: black; }\n",
            "favicon.png": "png bytes\n",
            "manifest.webmanifest": '{"name":"one"}\n',
            "service-worker.js": "self.addEventListener('push', () => {});\n",
            "codoxear-icon.png": "icon bytes\n",
            "logos/codex.svg": "<svg>codex</svg>\n",
            "logos/pi.svg": "<svg>pi</svg>\n",
            "logos/cc.svg": "<svg>cc</svg>\n",
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, content in initial_content.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            versions = [_static_asset_version(root)]
            for name in STATIC_ASSET_VERSION_FILES:
                target = root / name
                target.write_text(initial_content[name] + "/* changed */\n", encoding="utf-8")
                _static_asset_version.cache_clear()
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
                    '<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js" defer></script>\n'
                    '<script src="app_markdown.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_launch.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_display.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_new_session.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_dom.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_file_helpers.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_file_picker.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_file_viewer.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="monaco/vs/loader.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_file_editor.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_session_helpers.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_viewport.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_polling.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_transcript.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_message_identity.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_message_rows.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_conversation_copy.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_modal.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_clipboard.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_code_copy.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_hint_mode.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_voice_helpers.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_voice.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_queue.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_diagnostics.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_recovery.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_unattended.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_chat_navigation.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
                    '<script src="app_chat_search.js?v=__CODOXEAR_ASSET_VERSION__" defer></script>\n'
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
            self.assertIn('src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"', rendered)
            self.assertIn(f"app_markdown.js?v={version}", rendered)
            self.assertIn(f"app_launch.js?v={version}", rendered)
            self.assertIn(f"app_new_session.js?v={version}", rendered)
            self.assertIn(f"app_display.js?v={version}", rendered)
            self.assertIn(f"app_dom.js?v={version}", rendered)
            self.assertIn(f"app_file_helpers.js?v={version}", rendered)
            self.assertIn(f"app_file_picker.js?v={version}", rendered)
            self.assertIn(f"app_file_viewer.js?v={version}", rendered)
            self.assertIn(f"monaco/vs/loader.js?v={version}", rendered)
            self.assertIn(f"app_file_editor.js?v={version}", rendered)
            self.assertIn(f"app_session_helpers.js?v={version}", rendered)
            self.assertIn(f"app_viewport.js?v={version}", rendered)
            self.assertIn(f"app_polling.js?v={version}", rendered)
            self.assertIn(f"app_transcript.js?v={version}", rendered)
            self.assertIn(f"app_message_identity.js?v={version}", rendered)
            self.assertIn(f"app_message_rows.js?v={version}", rendered)
            self.assertIn(f"app_conversation_copy.js?v={version}", rendered)
            self.assertIn(f"app_modal.js?v={version}", rendered)
            self.assertIn(f"app_clipboard.js?v={version}", rendered)
            self.assertIn(f"app_code_copy.js?v={version}", rendered)
            self.assertIn(f"app_hint_mode.js?v={version}", rendered)
            self.assertIn(f"app_voice_helpers.js?v={version}", rendered)
            self.assertIn(f"app_voice.js?v={version}", rendered)
            self.assertIn(f"app_queue.js?v={version}", rendered)
            self.assertIn(f"app_diagnostics.js?v={version}", rendered)
            self.assertIn(f"app_recovery.js?v={version}", rendered)
            self.assertIn(f"app_unattended.js?v={version}", rendered)
            self.assertIn(f"app_chat_navigation.js?v={version}", rendered)
            self.assertIn(f"app_chat_search.js?v={version}", rendered)
            self.assertIn(f"app.js?v={version}", rendered)

    def test_static_asset_version_is_memoized(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            asset = root / "app.js"
            asset.write_text("one\n", encoding="utf-8")
            first = _static_asset_version(root)
            asset.write_text("two\n", encoding="utf-8")
            self.assertEqual(_static_asset_version(root), first)
            _static_asset_version.cache_clear()

    def test_static_cache_headers_default_to_no_cache(self) -> None:
        self.assertEqual(
            _static_cache_control_headers(versioned=False, is_html=True),
            {"Cache-Control": "no-cache"},
        )

    def test_static_cache_headers_can_be_immutable(self) -> None:
        self.assertEqual(
            _static_cache_control_headers(versioned=True, is_html=False),
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
        self.assertIn("codoxear/static/app_new_session.js", names)
        self.assertIn("codoxear/static/app_display.js", names)
        self.assertIn("codoxear/static/app_dom.js", names)
        self.assertIn("codoxear/static/app_file_helpers.js", names)
        self.assertIn("codoxear/static/app_file_picker.js", names)
        self.assertIn("codoxear/static/app_file_viewer.js", names)
        self.assertIn("codoxear/static/app_file_editor.js", names)
        self.assertIn("codoxear/static/monaco/LICENSE", names)
        self.assertIn("codoxear/static/monaco/ThirdPartyNotices.txt", names)
        self.assertIn("codoxear/static/monaco/vs/loader.js", names)
        self.assertTrue(any(name.startswith("codoxear/static/monaco/vs/assets/editor.worker-") and name.endswith(".js") for name in names))
        self.assertIn("codoxear/static/app_session_helpers.js", names)
        self.assertIn("codoxear/static/app_viewport.js", names)
        self.assertIn("codoxear/static/app_polling.js", names)
        self.assertIn("codoxear/static/app_transcript.js", names)
        self.assertIn("codoxear/static/app_message_identity.js", names)
        self.assertIn("codoxear/static/app_message_rows.js", names)
        self.assertIn("codoxear/static/app_conversation_copy.js", names)
        self.assertIn("codoxear/static/app_modal.js", names)
        self.assertIn("codoxear/static/app_clipboard.js", names)
        self.assertIn("codoxear/static/app_code_copy.js", names)
        self.assertIn("codoxear/static/app_hint_mode.js", names)
        self.assertIn("codoxear/static/app_voice_helpers.js", names)
        self.assertIn("codoxear/static/app_voice.js", names)
        self.assertIn("codoxear/static/app_queue.js", names)
        self.assertIn("codoxear/static/app_diagnostics.js", names)
        self.assertIn("codoxear/static/app_recovery.js", names)
        self.assertIn("codoxear/static/app_unattended.js", names)
        self.assertIn("codoxear/static/app_chat_navigation.js", names)
        self.assertIn("codoxear/static/app_chat_search.js", names)
        self.assertIn("codoxear/static/codoxear-icon.png", names)
        self.assertIn("codoxear/static/logos/codex.svg", names)
        self.assertIn("codoxear/static/logos/pi.svg", names)
        self.assertIn("codoxear/static/logos/cc.svg", names)


if __name__ == "__main__":
    unittest.main()
