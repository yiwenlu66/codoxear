#!/usr/bin/env bash
# Verify the integration-critical mechanisms without touching a running server.
# Override PYTHON_BIN when pytest lives outside the Codoxear pipx environment.

set -uo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif [[ -x "$HOME/.local/share/pipx/venvs/codoxear/bin/python" ]]; then
    python_bin="$HOME/.local/share/pipx/venvs/codoxear/bin/python"
else
    python_bin="python3"
fi

commits=(
    "dab6445|race fix: live marker wins over stale declared log"
    "685782e8|disk I/O: bounded message-path reads"
    "46fce5a3|PDF vendor: bundled PDF.js ESM pipeline"
    "1cdba92b|modal keyboard: nested target ordering"
    "49daf095|CSS tokens: paper stylesheet literals"
    "5c94a3b1|CC effort observer: live settings projection"
    "781a9421|resolveUrl fix: application URL helper"
    "5e5c8fb1|storage deps fix: unattended controller composition"
    "ba4831ac|extraction merge: message-flow controller"
    "414c5562|broker watchdog: crash recovery projection"
    "51e4a30e|offline resilience: transport failure banner"
    "4d8c4549|markdown fix: fence and long-code rendering"
    "54203bb4|send-choice: confirmed send flow"
    "6e03120b|traffic floor: continuous polling budget"
    "28dc6b94|performance floor: bounded transcript cache"
)

# Each entry is run independently so the report identifies the exact behavioral
# test module that passed or failed. The first four mechanisms form the critical
# integration set; the remaining entries are regression pins for its dependencies.
test_modules=(
    "critical|race fix|tests/test_broker_fail_closed.py"
    "critical|disk I/O|tests/test_message_index.py"
    "critical|disk I/O|tests/test_message_routes.py"
    "critical|PDF vendor|tests/test_pdf_viewer_pipeline.py"
    "critical|PDF vendor|tests/test_static_assets.py"
    "critical|modal keyboard|tests/test_frontend_modal_keyboard_module_source.py"
    "critical|modal keyboard|tests/test_modal_keyboard_nested_target_order.py"
    "pin|CSS tokens|tests/test_diagnostics_layout_css.py"
    "pin|CC effort observer|tests/test_cc_live_effort.py"
    "pin|CC effort observer|tests/test_cc_effort_live_observer.py"
    "pin|resolveUrl fix|tests/test_frontend_url_module_source.py"
    "pin|storage deps fix|tests/test_unattended_controller_deps.py"
    "pin|extraction merge|tests/test_frontend_message_flow.py"
    "pin|extraction merge|tests/test_frontend_message_flow_controller.py"
    "pin|extraction merge|tests/test_chat_transcript_runtime.py"
    "pin|broker watchdog|tests/test_broker_watchdog.py"
    "pin|offline resilience|tests/test_offline_resilience.py"
    "pin|offline resilience|tests/test_offline_banner_transport_failure.py"
    "pin|markdown fix|tests/test_markdown_edge_cases.py"
    "pin|markdown fix|tests/test_app_markdown_extended.py"
    "pin|send-choice|tests/test_send_choice.py"
    "pin|send-choice|tests/test_send_choice_steering.py"
    "pin|traffic floor|tests/test_continuous_traffic_budget.py"
    "pin|traffic floor|tests/test_frontend_polling_module_source.py"
    "pin|traffic floor|tests/test_session_polling_source.py"
    "pin|traffic floor|tests/test_traffic_floor.py"
    "pin|performance floor|tests/test_perf_floor_regression.py"
)

head="$(git rev-parse HEAD)"
commit_passed=0
commit_failed=0
test_passed=0
test_failed=0
critical_test_passed=0
critical_test_failed=0

printf '=== Codoxear critical integration verification ===\n'
printf 'Repository: %s\n' "$repo_root"
printf 'HEAD: %s\n' "$head"
printf 'Python: %s\n\n' "$python_bin"

if ! "$python_bin" -m pytest --version; then
    printf '\nRESULT: pytest is unavailable through %s\n' "$python_bin" >&2
    exit 2
fi

# Docker isolation verification (mandatory for the live deployment gate).
printf '\n-- Docker isolated browser verification --\n'
if bash "$repo_root/scripts/docker_verify.sh" HEAD; then
    printf 'PASS  Docker isolated verification\n'
else
    printf 'FAIL  Docker isolated verification\n' >&2
    exit 1
fi

printf '\n-- Commit ancestry --\n'
for entry in "${commits[@]}"; do
    IFS='|' read -r commit mechanism <<<"$entry"
    if git merge-base --is-ancestor "$commit" HEAD; then
        printf 'PASS  %s  %s\n' "$commit" "$mechanism"
        ((commit_passed += 1))
    else
        printf 'FAIL  %s  %s is not an ancestor of HEAD\n' "$commit" "$mechanism" >&2
        ((commit_failed += 1))
    fi
done

printf '\n-- Behavioral test modules --\n'
for entry in "${test_modules[@]}"; do
    IFS='|' read -r category mechanism test_file <<<"$entry"
    if [[ ! -f "$test_file" ]]; then
        printf 'FAIL  [%s] %s (%s): test file is missing\n' "$category" "$mechanism" "$test_file" >&2
        ((test_failed += 1))
        if [[ "$category" == "critical" ]]; then
            ((critical_test_failed += 1))
        fi
        continue
    fi

    printf '\n[%s] %s: %s\n' "$category" "$mechanism" "$test_file"
    if "$python_bin" -m pytest -q -- "$test_file"; then
        printf 'PASS  %s\n' "$test_file"
        ((test_passed += 1))
        if [[ "$category" == "critical" ]]; then
            ((critical_test_passed += 1))
        fi
    else
        printf 'FAIL  %s\n' "$test_file" >&2
        ((test_failed += 1))
        if [[ "$category" == "critical" ]]; then
            ((critical_test_failed += 1))
        fi
    fi
done

printf '\n=== Summary ===\n'
printf 'HEAD: %s\n' "$head"
printf 'Commit ancestry: %d passed, %d failed\n' "$commit_passed" "$commit_failed"
printf 'Critical behavioral modules: %d passed, %d failed\n' "$critical_test_passed" "$critical_test_failed"
printf 'All behavioral modules: %d passed, %d failed\n' "$test_passed" "$test_failed"

if (( commit_failed == 0 && test_failed == 0 )); then
    printf 'RESULT: PASS — all critical mechanisms and behavioral pins are verified.\n'
    exit 0
fi

printf 'RESULT: FAIL — inspect the failed rows above.\n' >&2
exit 1
