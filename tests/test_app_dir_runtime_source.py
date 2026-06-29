import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import app_dir_runtime
from codoxear import util


ROOT = Path(__file__).resolve().parents[1]


class TestAppDirRuntimeSource(unittest.TestCase):
    def setUp(self) -> None:
        util._LEGACY_WARNED = False

    def tearDown(self) -> None:
        util._LEGACY_WARNED = False

    def test_app_dir_policy_has_runtime_owner_with_util_facade(self) -> None:
        util_source = (ROOT / "codoxear" / "util.py").read_text(encoding="utf-8")
        runtime_source = (ROOT / "codoxear" / "app_dir_runtime.py").read_text(encoding="utf-8")

        self.assertIn("class AppDirResolution:", runtime_source)
        self.assertIn("def resolve_default_app_dir(*, legacy_warned: bool, home: Path | None = None) -> AppDirResolution:", runtime_source)
        self.assertIn('base / "codex-web"', runtime_source)
        self.assertIn('base / "codoxear"', runtime_source)
        self.assertIn("legacy runtime dir detected", runtime_source)
        self.assertIn("from .app_dir_runtime import resolve_default_app_dir as _resolve_default_app_dir", util_source)
        self.assertIn("def default_app_dir() -> Path:", util_source)
        self.assertIn("_LEGACY_WARNED = False", util_source)
        self.assertIn("resolution = _resolve_default_app_dir(legacy_warned=_LEGACY_WARNED)", util_source)
        self.assertIn("_LEGACY_WARNED = resolution.legacy_warned", util_source)
        self.assertIn("_log_error(resolution.warning)", util_source)
        self.assertNotIn("legacy runtime dir detected", util_source)
        self.assertIn("def launch_attempts_path(app_dir: Path | None = None) -> Path:", util_source)

    def test_runtime_resolves_new_app_dir_and_warns_once_for_legacy_dir(self) -> None:
        with TemporaryDirectory() as td:
            home = Path(td) / "home"
            old = home / ".local" / "share" / "codex-web"
            old.mkdir(parents=True)
            expected = home / ".local" / "share" / "codoxear"
            first = app_dir_runtime.resolve_default_app_dir(home=home, legacy_warned=False)
            second = app_dir_runtime.resolve_default_app_dir(home=home, legacy_warned=first.legacy_warned)

        self.assertEqual(first.app_dir, expected)
        self.assertTrue(first.legacy_warned)
        self.assertIsNotNone(first.warning)
        self.assertIn("legacy runtime dir detected", first.warning or "")
        self.assertIn("codex-web", first.warning or "")
        self.assertIn("codoxear", first.warning or "")
        self.assertEqual(second.app_dir, expected)
        self.assertTrue(second.legacy_warned)
        self.assertIsNone(second.warning)

    def test_util_facade_injects_util_log_error_and_preserves_warning_state_seam(self) -> None:
        with TemporaryDirectory() as td:
            home = Path(td) / "home"
            (home / ".local" / "share" / "codex-web").mkdir(parents=True)
            messages: list[str] = []

            with patch("codoxear.app_dir_runtime.Path.home", return_value=home):
                with patch("codoxear.util._log_error", side_effect=messages.append):
                    self.assertEqual(util.default_app_dir(), home / ".local" / "share" / "codoxear")
                    self.assertEqual(util.default_app_dir(), home / ".local" / "share" / "codoxear")
                    util._LEGACY_WARNED = False
                    self.assertEqual(util.default_app_dir(), home / ".local" / "share" / "codoxear")

        self.assertEqual(len(messages), 2)
        self.assertIn("legacy runtime dir detected", messages[0])
        self.assertIn("legacy runtime dir detected", messages[1])


if __name__ == "__main__":
    unittest.main()
