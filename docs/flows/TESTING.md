# Testing

Tests are unittest-based and focus on log parsing, idle heuristics, URL prefix handling, and server behavior.

## Run all tests
`python3 -m unittest discover -s tests`

## If pytest is unavailable
Pytest can still run the unittest suite if you install it, but it's optional:
`python3 -m pytest`

## Run a single test file
`python3 -m unittest tests.test_server_chat_flags`
`python3 -m unittest tests.test_server_queue`
`python3 -m unittest tests.test_file_history`
`python3 -m unittest tests.test_broker_proc_rollout`
`python3 -m unittest tests.test_broker_busy_state`
`python3 -m unittest tests.test_broker_spawn_env`
`python3 -m unittest tests.test_idle_heuristics`
`python3 -m unittest tests.test_cli_support`
`python3 -m unittest tests.test_util_gemini_offset`
`python3 -m unittest tests.test_server_spawn_cli`
`python3 -m unittest tests.test_update_check`

## Notes
- Tests rely on static fixtures and do not require a running broker.
- When adding new behavior around log parsing or idle detection, extend tests in `tests/`.
- Claude and Gemini support coverage currently lives in `test_broker_proc_rollout`, `test_broker_spawn_env`, `test_cli_support`, `test_idle_heuristics`, `test_server_chat_flags`, `test_server_spawn_cli`, and `test_util_gemini_offset`.
