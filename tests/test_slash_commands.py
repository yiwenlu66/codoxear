from codoxear.slash_commands import default_slash_commands, slash_commands_for_backend


def names(commands):
    return [item["name"] for item in commands]


def test_pi_defaults_keep_model_but_hide_interactive_builtins_and_bridge_commands():
    assert "model" in names(default_slash_commands("pi"))
    assert "effort" not in names(default_slash_commands("pi"))
    assert "settings" not in names(default_slash_commands("pi"))
    assert "effort" in names(default_slash_commands("pi", pi_bridge_capable=True))
    assert "thinking" in names(default_slash_commands("pi", pi_bridge_capable=True))


def test_live_pi_registry_filters_known_interactive_entries():
    projected = slash_commands_for_backend(
        "pi",
        [{"name": "settings"}, {"name": "model", "description": "Pick"}, {"name": "custom", "description": "Text"}],
        pi_bridge_capable=True,
    )
    assert names(projected) == ["model", "custom"]


def test_non_pi_defaults_are_backend_specific():
    assert names(default_slash_commands("cc"))[:2] == ["model", "effort"]
    assert default_slash_commands("codex") == []
