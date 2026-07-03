"""Direct impl/coordinator tests for launch-defaults logic.

Previously these tests patched ``codoxear.server`` module globals (~18 sites):
``_LAUNCH_DEFAULTS_CACHE``, ``_launch_config_paths``,
``_launch_read_new_session_defaults``, ``CODEX_CONFIG_PATH``,
``MODELS_CACHE_PATH``, ``PI_SETTINGS_PATH``, ``PI_MODELS_PATH``,
``PI_AUTH_PATH`` and ``CC_SETTINGS_PATH``.

They now exercise the true seams directly:

* codex/pi/cc per-backend launch-default readers -> ``codoxear.launch_config``
  free functions (``read_codex_launch_defaults``, ``read_pi_launch_defaults``,
  ``read_cc_launch_defaults``, ``read_new_session_defaults``) called with a
  ``LaunchConfigPaths`` built from real temp config files. No module-global
  path patching; the file-system paths are the injected dependency.
* normalize/validation helpers (model_provider, service_tier,
  preferred_auth_method, pi reasoning effort) -> the pure
  ``codoxear.launch_config`` functions directly, no facade indirection.
* the cached ``_read_new_session_defaults`` facade -> the real
  ``codoxear.launch_defaults_runtime.read_new_session_defaults_cached`` with
  ``paths_provider``, ``defaults_reader``, ``cache_lock``, ``get_cache`` and
  ``set_cache`` injected, so the test owns the cache instead of mutating the
  module global.

No ``codoxear.server.*`` module-global monkeypatching remains. No file under
``codoxear/`` is modified. No ``try/except`` swallows.
"""

import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear.launch_config import LaunchConfigPaths
from codoxear.launch_config import normalize_requested_model_provider
from codoxear.launch_config import normalize_requested_pi_reasoning_effort
from codoxear.launch_config import normalize_requested_preferred_auth_method
from codoxear.launch_config import normalize_requested_service_tier
from codoxear.launch_config import read_cc_launch_defaults
from codoxear.launch_config import read_codex_launch_defaults
from codoxear.launch_config import read_new_session_defaults
from codoxear.launch_config import read_pi_launch_defaults
from codoxear.launch_defaults_runtime import read_new_session_defaults_cached

DEFAULT_AGENT_BACKEND = "codex"


def _paths_for(root: Path) -> LaunchConfigPaths:
    """Build a ``LaunchConfigPaths`` rooted at the given temp directory. Tests
    write the specific files they need; absent paths exercise the real
    "file missing" fallback branches instead of patching module globals."""
    return LaunchConfigPaths(
        codex_config_path=root / "config.toml",
        models_cache_path=root / "models.json",
        pi_settings_path=root / "pi-settings.json",
        pi_models_path=root / "pi-models.json",
        pi_auth_path=root / "pi-auth.json",
        cc_settings_path=root / "cc-settings.json",
    )


class TestLaunchDefaults(unittest.TestCase):
    def test_read_new_session_defaults_cache_uses_config_signatures(self) -> None:
        # The cached facade is exercised directly with injected paths_provider,
        # defaults_reader, cache lock and cache accessors. The test owns the
        # cache slot instead of mutating the module global
        # ``server._LAUNCH_DEFAULTS_CACHE``.
        with TemporaryDirectory() as td:
            root = Path(td)
            paths = _paths_for(root)
            paths.codex_config_path.write_text("model = 'a'\n", encoding="utf-8")
            calls = {"n": 0}

            def fake_read(_paths, *, default_agent_backend: str):
                calls["n"] += 1
                return {
                    "default_backend": default_agent_backend,
                    "call": calls["n"],
                    "backends": {},
                }

            cache_box: list = [None]
            cache_lock = threading.Lock()

            def get_cache():
                return cache_box[0]

            def set_cache(value):
                cache_box[0] = value

            kwargs = dict(
                paths_provider=lambda: paths,
                defaults_reader=fake_read,
                default_agent_backend=DEFAULT_AGENT_BACKEND,
                cache_lock=cache_lock,
                get_cache=get_cache,
                set_cache=set_cache,
            )

            first = read_new_session_defaults_cached(**kwargs)
            first["mutated"] = True  # caller mutation must not leak into cache
            second = read_new_session_defaults_cached(**kwargs)
            self.assertEqual(calls["n"], 1)
            self.assertNotIn("mutated", second)
            self.assertEqual(second["call"], 1)

            # Signature change (mtime + content) invalidates the cache.
            paths.codex_config_path.write_text("model = 'b'\nchanged = true\n", encoding="utf-8")
            future = time.time() + 2
            os.utime(paths.codex_config_path, (future, future))
            third = read_new_session_defaults_cached(**kwargs)
            self.assertEqual(calls["n"], 2)
            self.assertEqual(third["call"], 2)

    def test_read_codex_launch_defaults_includes_provider_list_and_service_tier(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.codex_config_path.write_text(
                """
model = "gpt-5.4"
model_provider = "crs"
preferred_auth_method = "apikey"
service_tier = "fast"

[model_providers.crs]
name = "CRS"

[model_providers.right]
name = "Right"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            paths.models_cache_path.write_text(
                '{"models":[{"slug":"gpt-5.4","default_reasoning_level":"medium","priority":1}]}',
                encoding="utf-8",
            )

            defaults = read_codex_launch_defaults(paths)

        self.assertEqual(defaults["model_provider"], "crs")
        self.assertEqual(defaults["preferred_auth_method"], "apikey")
        self.assertEqual(defaults["provider_choice"], "crs")
        self.assertEqual(defaults["model"], "gpt-5.4")
        self.assertEqual(defaults["model_providers"], ["chatgpt", "openai-api", "crs", "right"])
        self.assertEqual(defaults["service_tier"], "fast")
        self.assertEqual(defaults["reasoning_effort"], "medium")

    def test_read_codex_launch_defaults_falls_back_to_openai_and_flex(self) -> None:
        with TemporaryDirectory() as td:
            # Real absent-path files exercise the "config missing" fallback
            # branch instead of patching module-global path constants.
            paths = _paths_for(Path(td))
            defaults = read_codex_launch_defaults(paths)

        self.assertEqual(defaults["model_provider"], "openai")
        self.assertEqual(defaults["preferred_auth_method"], "apikey")
        self.assertEqual(defaults["provider_choice"], "openai-api")
        self.assertIsNone(defaults["model"])
        self.assertEqual(defaults["model_providers"], ["chatgpt", "openai-api"])
        self.assertEqual(defaults["service_tier"], "flex")
        self.assertIsNone(defaults["reasoning_effort"])

    def test_normalize_requested_model_provider_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_provider must be one of openai, right"):
            normalize_requested_model_provider("bytecat", allowed={"openai", "right"})

    def test_normalize_requested_service_tier_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "service_tier must be one of fast, flex"):
            normalize_requested_service_tier("slow")

    def test_normalize_requested_preferred_auth_method_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "preferred_auth_method must be one of chatgpt, apikey"):
            normalize_requested_preferred_auth_method("oauth")

    def test_read_codex_launch_defaults_maps_openai_chatgpt_choice(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.codex_config_path.write_text(
                """
model_provider = "openai"
preferred_auth_method = "chatgpt"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            paths.models_cache_path.write_text('{"models":[]}', encoding="utf-8")

            defaults = read_codex_launch_defaults(paths)

        self.assertEqual(defaults["provider_choice"], "chatgpt")

    def test_read_codex_launch_defaults_collects_provider_names_by_section_key(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.codex_config_path.write_text(
                """
service_tier = "flex"

[model_providers.crs]
name = "CRS"

[model_providers.custom]
base_url = "https://example.com/v1"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            paths.models_cache_path.write_text('{"models":[]}', encoding="utf-8")

            defaults = read_codex_launch_defaults(paths)

        self.assertEqual(defaults["model_providers"], ["chatgpt", "openai-api", "crs", "custom"])

    def test_read_pi_launch_defaults_reads_provider_model_and_thinking(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.pi_settings_path.write_text(
                """
{
  "defaultProvider": "macaron",
  "defaultModel": "gpt-5.4",
  "defaultThinkingLevel": "medium"
}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            paths.pi_models_path.write_text(
                """
{
  "providers": {
    "macaron": {
      "models": [
        {"id": "gpt-5.4"},
        {"id": "gpt-5.4-mini"}
      ]
    }
  }
}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            defaults = read_pi_launch_defaults(paths)

        self.assertEqual(defaults["provider_choice"], "macaron")
        self.assertEqual(defaults["model"], "gpt-5.4")
        self.assertEqual(defaults["reasoning_effort"], "high")
        self.assertEqual(defaults["provider_choices"], ["macaron"])
        self.assertEqual(defaults["models"], ["gpt-5.4", "gpt-5.4-mini"])
        self.assertFalse(defaults["supports_fast"])

    def test_read_pi_launch_defaults_reports_model_specific_reasoning_efforts(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.pi_settings_path.write_text(
                '{"defaultProvider":"macaron","defaultModel":"plain"}\n', encoding="utf-8"
            )
            paths.pi_models_path.write_text(
                '{"providers":{"macaron":{"models":[{"id":"plain","reasoning":false},{"id":"smart","reasoningEfforts":["low","high"]}]}}}\n',
                encoding="utf-8",
            )

            defaults = read_pi_launch_defaults(paths)

        self.assertEqual(defaults["reasoning_effort"], "off")
        self.assertEqual(defaults["reasoning_efforts"], ["off"])
        self.assertEqual(defaults["reasoning_efforts_by_model"]["macaron/plain"], ["off"])
        self.assertEqual(defaults["reasoning_efforts_by_model"]["macaron/smart"], ["low", "high"])

    def test_normalize_requested_pi_reasoning_effort_rejects_unsupported_model_effort(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.pi_models_path.write_text(
                '{"providers":{"macaron":{"models":[{"id":"plain","reasoning":false}]}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be one of off for Pi model plain"):
                normalize_requested_pi_reasoning_effort("high", model_provider="macaron", model="plain", paths=paths)
            self.assertEqual(
                normalize_requested_pi_reasoning_effort("off", model_provider="macaron", model="plain", paths=paths),
                "off",
            )

    def test_read_cc_launch_defaults_reads_settings_model_and_effort(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.cc_settings_path.write_text(
                '{"model":"claude-haiku-4-5","effortLevel":"max"}\n', encoding="utf-8"
            )

            defaults = read_cc_launch_defaults(paths)

        self.assertEqual(defaults["agent_backend"], "cc")
        self.assertEqual(defaults["model"], "claude-haiku-4-5")
        self.assertEqual(defaults["reasoning_effort"], "max")
        self.assertEqual(defaults["reasoning_efforts"], ["low", "medium", "high", "xhigh", "max"])
        self.assertEqual(defaults["provider_choices"], [])
        self.assertFalse(defaults["supports_fast"])

    def test_read_new_session_defaults_includes_registered_backends(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.pi_settings_path.write_text(
                '{"defaultProvider":"macaron","defaultModel":"gpt-5.4","defaultThinkingLevel":"medium"}\n',
                encoding="utf-8",
            )
            paths.pi_models_path.write_text(
                '{"providers":{"macaron":{"models":[{"id":"gpt-5.4"}]}}}\n', encoding="utf-8"
            )

            defaults = read_new_session_defaults(paths, default_agent_backend=DEFAULT_AGENT_BACKEND)

        self.assertEqual(defaults["default_backend"], "codex")
        self.assertIn("codex", defaults["backends"])
        self.assertIn("pi", defaults["backends"])
        self.assertIn("cc", defaults["backends"])
        self.assertEqual(defaults["backends"]["pi"]["provider_choice"], "macaron")
        self.assertNotIn("warnings", defaults)

    def test_read_new_session_defaults_fails_soft_for_malformed_backend_configs(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.codex_config_path.write_text("model = [\n", encoding="utf-8")
            paths.models_cache_path.write_text('{"models": []}\n', encoding="utf-8")
            paths.pi_settings_path.write_text("{bad json\n", encoding="utf-8")
            paths.pi_models_path.write_text('{"providers": {}}\n', encoding="utf-8")
            paths.cc_settings_path.write_text("{bad json\n", encoding="utf-8")

            defaults = read_new_session_defaults(paths, default_agent_backend=DEFAULT_AGENT_BACKEND)

        self.assertEqual(set(defaults["backends"]), {"codex", "pi", "cc"})
        self.assertEqual(set(defaults["warnings"]), {"codex", "pi", "cc"})
        self.assertEqual(defaults["backends"]["codex"]["provider_choices"], ["chatgpt", "openai-api"])
        self.assertEqual(defaults["backends"]["codex"]["reasoning_effort"], None)
        self.assertEqual(defaults["backends"]["pi"]["provider_choices"], [])
        self.assertEqual(defaults["backends"]["pi"]["reasoning_effort"], "high")
        self.assertEqual(defaults["backends"]["cc"]["provider_choices"], [])
        self.assertEqual(defaults["backends"]["cc"]["reasoning_effort"], "medium")

    def test_read_new_session_defaults_fails_soft_for_malformed_pi_models(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.pi_models_path.write_text("{bad json\n", encoding="utf-8")

            defaults = read_new_session_defaults(paths, default_agent_backend=DEFAULT_AGENT_BACKEND)

        self.assertEqual(set(defaults["backends"]), {"codex", "pi", "cc"})
        self.assertEqual(set(defaults["warnings"]), {"pi"})
        self.assertEqual(defaults["backends"]["pi"]["reasoning_efforts_by_model"], {})

    def test_read_pi_launch_defaults_includes_logged_in_oauth_providers(self) -> None:
        with TemporaryDirectory() as td:
            paths = _paths_for(Path(td))
            paths.pi_settings_path.write_text(
                '{"defaultProvider":"macaron","defaultModel":"gpt-5.4"}\n', encoding="utf-8"
            )
            paths.pi_models_path.write_text(
                '{"providers":{"macaron":{"models":[{"id":"gpt-5.4"}]}}}\n', encoding="utf-8"
            )
            paths.pi_auth_path.write_text(
                '{"openai-codex":{"type":"oauth","access":"abc","refresh":"def"},"ignore-me":{"type":"apikey"}}\n',
                encoding="utf-8",
            )

            defaults = read_pi_launch_defaults(paths)

        self.assertEqual(defaults["provider_choices"], ["macaron", "openai-codex"])


if __name__ == "__main__":
    unittest.main()
