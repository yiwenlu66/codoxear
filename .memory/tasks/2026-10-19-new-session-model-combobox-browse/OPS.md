# OPS — new-session provider/model combobox browse

Append-only evidence trail. Times are +08:00.

## 2026-10-21T13:10 — read current behavior (source, not guesses)

- `app_new_session.js:831` creates `#newSessionModelInput` (role=combobox,
  `aria-controls=newSessionModelMenu`); `setBackend` (line 963) prefills it with
  the remembered pair rendered `<model> · <provider>` (`MODEL_PROVIDER_SEPARATOR`).
- `filteredNewSessionModelOptions()` used `modelInput.value` as the filter
  query, so opening the list with `"gpt-5.4 · chatgpt"` parsed to
  `chatgpt/gpt-5.4` and prefix-matched only that model (plus ids it prefixes).
- `modelInput.oninput` appended to the display string, so a typed character
  produced `"gpt-5.4 · chatgptg"` → "No matching models" while
  `parseNewSessionProviderModelInput` fell through to `model = raw`.
- 8 scattered `modelMenuOpen = false; modelMenuFocus = -1;` close sites.
- Existing option cap: `if (!query) return options.slice(0, 12);`.

## 2026-10-21T13:21 — implementation

- `browseModelOptions()`: no-query ordering = recents bucket (key `null`) first,
  then one group per provider in first-appearance order; each item carries
  `groupLabel`. Query path unchanged (exact/prefix/contains, cap 12).
- Dialog state: `browseRestoreText` (text captured at browse-clear; non-null ⇒
  browse-only) and `modelQueryDirty` (content authored by the user).
- `beginModelBrowse()` on click and on ArrowUp/Down while closed;
  `onbeforeinput` clears a committed non-empty field before the edit applies;
  `oninput` marks the field dirty; `dismissModelPicker()` is the single close
  funnel and restores only a still-empty browse; `start()` restores before
  `syncNewSessionProviderFromModelInput()`.
- Renderer groups by `groupLabel` and anchors the highlight on
  `raw || browseRestoreText || configured`.
- `#newSessionModelMenu .modelMenuGroup` sticky group label (existing tokens).
- `visualViewport` resize re-runs `applyDialogMenus()` while any dialog menu is
  open (a full list is the first menu whose tail can fall under a mobile
  keyboard).

## 2026-10-21T13:30 — VM harness (real module code in node)

`tests/test_new_session_model_options_source.py`, run under the repo venv
(`.venv/bin/python -m pytest`; the system python3 lacks py_vapid, so pytest
cannot collect there):

```
.venv/bin/python -m pytest tests/test_new_session_model_options_source.py -q
=> 8 passed
```

New cases: browse with an empty query returns all 40 configured pairs (no cap,
no duplicates, `Recent` → `chatgpt` → `openai-api` group order); a matching
query still returns 12 flat ranked rows with no `groupLabel`.

## 2026-10-21T13:38 — first Docker/browser verification run (HEAD 7f14dcc9)

`./scripts/docker_new_session_model_picker_verify.sh HEAD`
=> 15 PASS, 4 FAIL. Probe values showed `04-escape-restored.value == "g"`.
Interpretation: the verifier's step order contradicted the intended contract —
it typed `g` and then expected a dismiss to restore the pair, but a typed query
must survive a dismiss. Implementation was correct; the test was wrong.
`start_launched_the_committed_pair` failed only because the check read
`provider_choice` (null on Pi) instead of `model_provider`; the launch record
already held `model=claude-opus-4-6, model_provider=dexgem-messages`.

## 2026-10-21T13:51 — verification rerun (same commit, corrected checks)

`./scripts/docker_new_session_model_picker_verify.sh HEAD`
=> 19/19 PASS. Artifacts `/tmp/codoxear-model-combobox-verify.EFY3YR`.
Observed: browse option count 15 == configured Pi pairs, groups
`[dexgem-messages, dexgem-responses, dexgem-completions]`, highlighted row ==
committed `kimi-k3 · dexgem-messages`; click → `""`; Escape (browse-only) →
restored; type `c` → `"c"` with 9 ranked rows; Escape keeps `"c"`; select →
`claude-opus-4-6 · dexgem-messages`; outside click restores; Codex tab
repopulates and its click clears; Pi tab restores the chosen pair; browse-clear
then Start recorded `model=claude-opus-4-6, model_provider=dexgem-messages`
(`~/.local/share/codoxear/session_launches.jsonl`), not `defaults.model`.

Screenshot inspection found the highlighted row's provider hint in
`--muted` on the inverted ink fill (weak contrast). Fixed with
`#newSessionModelMenu .fileMenuItem.active .fileMenuHint { color: inherit;
opacity: 0.72; }` (tokens only, no theme change).

## 2026-10-21T13:53 — full suite (repo venv)

```
.venv/bin/python -m pytest tests -q
=> 2280 passed, 112 subtests passed, 1 warning
```

Note: the checkout has unrelated concurrent edits (another agent's undelivered-
send work); only this task's paths were staged/committed.

## 2026-10-21T13:57 — final verification of the committed state

`./scripts/docker_new_session_model_picker_verify.sh HEAD` (HEAD includes the
contrast fix committed as its own change) => 19/19 PASS.
