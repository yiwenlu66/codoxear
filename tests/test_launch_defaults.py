import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear.server import _normalize_requested_model_provider
from codoxear.server import _normalize_requested_preferred_auth_method
from codoxear.server import _normalize_requested_service_tier
from codoxear.server import _read_codex_launch_defaults
from codoxear.server import _read_new_session_defaults
from codoxear.server import _read_pi_launch_defaults


class TestLaunchDefaults(unittest.TestCase):
    def test_read_codex_launch_defaults_includes_provider_list_and_service_tier(self) -> None:
        with TemporaryDirectory() as td:
            config_path = Path(td) / "config.toml"
            models_cache_path = Path(td) / "models.json"
            config_path.write_text(
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
            models_cache_path.write_text(
                '{"models":[{"slug":"gpt-5.4","default_reasoning_level":"medium","priority":1}]}',
                encoding="utf-8",
            )

            with patch("codoxear.server.CODEX_CONFIG_PATH", config_path), patch("codoxear.server.MODELS_CACHE_PATH", models_cache_path):
                defaults = _read_codex_launch_defaults()

        self.assertEqual(defaults["model_provider"], "crs")
        self.assertEqual(defaults["preferred_auth_method"], "apikey")
        self.assertEqual(defaults["provider_choice"], "crs")
        self.assertEqual(defaults["model"], "gpt-5.4")
        self.assertEqual(defaults["model_providers"], ["chatgpt", "openai-api", "crs", "right"])
        self.assertEqual(defaults["service_tier"], "fast")
        self.assertEqual(defaults["reasoning_effort"], "medium")

    def test_read_codex_launch_defaults_falls_back_to_openai_and_flex(self) -> None:
        with TemporaryDirectory() as td:
            config_path = Path(td) / "missing-config.toml"
            models_cache_path = Path(td) / "missing-models.json"
            with patch("codoxear.server.CODEX_CONFIG_PATH", config_path), patch("codoxear.server.MODELS_CACHE_PATH", models_cache_path):
                defaults = _read_codex_launch_defaults()

        self.assertEqual(defaults["model_provider"], "openai")
        self.assertEqual(defaults["preferred_auth_method"], "apikey")
        self.assertEqual(defaults["provider_choice"], "openai-api")
        self.assertIsNone(defaults["model"])
        self.assertEqual(defaults["model_providers"], ["chatgpt", "openai-api"])
        self.assertEqual(defaults["service_tier"], "flex")
        self.assertIsNone(defaults["reasoning_effort"])

    def test_normalize_requested_model_provider_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_provider must be one of openai, right"):
            _normalize_requested_model_provider("bytecat", allowed={"openai", "right"})

    def test_normalize_requested_service_tier_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "service_tier must be one of fast, flex"):
            _normalize_requested_service_tier("slow")

    def test_normalize_requested_preferred_auth_method_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "preferred_auth_method must be one of chatgpt, apikey"):
            _normalize_requested_preferred_auth_method("oauth")

    def test_read_codex_launch_defaults_maps_openai_chatgpt_choice(self) -> None:
        with TemporaryDirectory() as td:
            config_path = Path(td) / "config.toml"
            models_cache_path = Path(td) / "models.json"
            config_path.write_text(
                """
model_provider = "openai"
preferred_auth_method = "chatgpt"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            models_cache_path.write_text('{"models":[]}', encoding="utf-8")

            with patch("codoxear.server.CODEX_CONFIG_PATH", config_path), patch("codoxear.server.MODELS_CACHE_PATH", models_cache_path):
                defaults = _read_codex_launch_defaults()

        self.assertEqual(defaults["provider_choice"], "chatgpt")

    def test_read_codex_launch_defaults_collects_provider_names_by_section_key(self) -> None:
        with TemporaryDirectory() as td:
            config_path = Path(td) / "config.toml"
            models_cache_path = Path(td) / "models.json"
            config_path.write_text(
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
            models_cache_path.write_text('{"models":[]}', encoding="utf-8")

            with patch("codoxear.server.CODEX_CONFIG_PATH", config_path), patch("codoxear.server.MODELS_CACHE_PATH", models_cache_path):
                defaults = _read_codex_launch_defaults()

        self.assertEqual(defaults["model_providers"], ["chatgpt", "openai-api", "crs", "custom"])

    def test_read_pi_launch_defaults_reads_provider_model_and_thinking(self) -> None:
        with TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            models_path = Path(td) / "models.json"
            auth_path = Path(td) / "missing-auth.json"
            settings_path.write_text(
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
            models_path.write_text(
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
            with patch("codoxear.server.PI_SETTINGS_PATH", settings_path), patch("codoxear.server.PI_MODELS_PATH", models_path), patch(
                "codoxear.server.PI_AUTH_PATH", auth_path
            ):
                defaults = _read_pi_launch_defaults()

        self.assertEqual(defaults["provider_choice"], "macaron")
        self.assertEqual(defaults["model"], "gpt-5.4")
        self.assertEqual(defaults["reasoning_effort"], "high")
        self.assertEqual(defaults["provider_choices"], ["macaron"])
        self.assertEqual(defaults["models"], ["gpt-5.4", "gpt-5.4-mini"])
        self.assertFalse(defaults["supports_fast"])

    def test_read_new_session_defaults_includes_both_backends(self) -> None:
        with TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            models_path = Path(td) / "models.json"
            settings_path.write_text('{"defaultProvider":"macaron","defaultModel":"gpt-5.4","defaultThinkingLevel":"medium"}\n', encoding="utf-8")
            models_path.write_text('{"providers":{"macaron":{"models":[{"id":"gpt-5.4"}]}}}\n', encoding="utf-8")
            with patch("codoxear.server.PI_SETTINGS_PATH", settings_path), patch("codoxear.server.PI_MODELS_PATH", models_path):
                defaults = _read_new_session_defaults()

        self.assertEqual(defaults["default_backend"], "codex")
        self.assertIn("codex", defaults["backends"])
        self.assertIn("pi", defaults["backends"])
        self.assertEqual(defaults["backends"]["pi"]["provider_choice"], "macaron")

    def test_read_pi_launch_defaults_includes_logged_in_oauth_providers(self) -> None:
        with TemporaryDirectory() as td:
            settings_path = Path(td) / "settings.json"
            models_path = Path(td) / "models.json"
            auth_path = Path(td) / "auth.json"
            settings_path.write_text('{"defaultProvider":"macaron","defaultModel":"gpt-5.4"}\n', encoding="utf-8")
            models_path.write_text('{"providers":{"macaron":{"models":[{"id":"gpt-5.4"}]}}}\n', encoding="utf-8")
            auth_path.write_text(
                '{"openai-codex":{"type":"oauth","access":"abc","refresh":"def"},"ignore-me":{"type":"apikey"}}\n',
                encoding="utf-8",
            )
            with patch("codoxear.server.PI_SETTINGS_PATH", settings_path), patch("codoxear.server.PI_MODELS_PATH", models_path), patch(
                "codoxear.server.PI_AUTH_PATH", auth_path
            ):
                defaults = _read_pi_launch_defaults()

        self.assertEqual(defaults["provider_choices"], ["macaron", "openai-codex"])


if __name__ == "__main__":
    unittest.main()
