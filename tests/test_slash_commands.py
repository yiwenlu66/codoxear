from codoxear.slash_commands import (
    PI_BUILTIN_SLASH_COMMANDS,
    default_slash_commands,
    slash_commands_for_backend,
)


def names(commands):
    return [item["name"] for item in commands]


def by_name(commands, name):
    return next(item for item in commands if item["name"] == name)


def test_pi_defaults_keep_model_but_hide_interactive_builtins_and_bridge_commands():
    assert "model" in names(default_slash_commands("pi"))
    assert "new" in names(default_slash_commands("pi"))
    assert "effort" not in names(default_slash_commands("pi"))
    assert "settings" not in names(default_slash_commands("pi"))
    assert "resume" not in names(default_slash_commands("pi"))
    assert "effort" in names(default_slash_commands("pi", pi_bridge_capable=True))
    assert "thinking" in names(default_slash_commands("pi", pi_bridge_capable=True))


def test_live_pi_registry_unions_browser_safe_builtins_before_extensions():
    projected = slash_commands_for_backend(
        "pi",
        [
            {"name": "settings"},
            {"name": "resume", "description": "Resume a different session"},
            {"name": "fork"},
            {"name": "model", "description": "Extension model picker"},
            {"name": "custom", "description": "Text"},
            {"name": "custom", "description": "Duplicate"},
            {"name": "effort", "description": "Set effort"},
            {"name": "thinking", "description": "Set thinking"},
        ],
        pi_bridge_capable=True,
    )

    builtin_names = names(PI_BUILTIN_SLASH_COMMANDS)
    assert names(projected) == builtin_names + ["custom", "effort", "thinking"]
    assert all(name not in names(projected) for name in ("settings", "resume", "fork"))
    assert by_name(projected, "model") == PI_BUILTIN_SLASH_COMMANDS[0]
    assert by_name(projected, "custom")["description"] == "Text"
    assert by_name(projected, "effort")["description"] == "Set effort"
    assert by_name(projected, "thinking")["description"] == "Set thinking"


def test_pi_without_a_live_registry_keeps_existing_builtin_projection():
    assert slash_commands_for_backend("pi", None, pi_bridge_capable=True) == default_slash_commands("pi")


def test_codex_only_exposes_broker_advertised_live_controls():
    assert slash_commands_for_backend("codex") == []
    assert names(slash_commands_for_backend("codex", [{"name": "model"}, {"name": "effort"}])) == ["model", "effort"]


def test_non_pi_defaults_are_backend_specific():
    assert names(default_slash_commands("cc"))[:2] == ["model", "effort"]
    assert default_slash_commands("codex") == []
