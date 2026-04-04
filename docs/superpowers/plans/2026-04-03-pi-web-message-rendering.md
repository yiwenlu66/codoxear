# Pi Web Message Rendering Implementation Plan

> Superseded on 2026-04-04: Codoxear reverted to a single Pi `Conversation` view and removed the `Event Stream` tab and request path.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a production-ready dual-view Pi web session experience with complete Pi event normalization and readable browser rendering.

**Architecture:** Normalize Pi session-file entries into a richer event model in `codoxear/pi_messages.py`, keep the API shape stable by returning the same `events` array, and let the browser choose between a filtered conversation timeline and a full event-stream timeline. Extend the existing Pi chat renderer with new event cards and a Pi-only view toggle.

**Tech Stack:** Python, vanilla JS, CSS, unittest, existing Codoxear server/static app architecture

---

## File Map

- Modify: `codoxear/pi_messages.py` — richer Pi event normalization
- Modify: `codoxear/static/app.js` — view toggle, render pipeline, row renderers, cache updates
- Modify: `codoxear/static/app.css` — new Pi event card/toggle styling
- Modify: `tests/test_pi_server_backend.py` — normalization regression coverage
- Modify: `tests/test_pi_chat_ui_source.py` or create focused UI source test — render pipeline coverage
- Create: `docs/superpowers/specs/2026-04-03-pi-web-message-rendering-design.md`
- Create: `docs/superpowers/plans/2026-04-03-pi-web-message-rendering.md`

### Task 1: Add failing Pi normalization tests

**Files:**
- Modify: `tests/test_pi_server_backend.py`
- Test: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Write failing tests for reasoning, todo snapshots, and system events**

```python
# Add tests asserting normalize_pi_entries emits:
# - reasoning events for assistant thinking blocks
# - todo_snapshot events for manage_todo_list tool results
# - pi_model_change / pi_thinking_level_change events
# - fallback pi_event rows for unsupported entry types
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m pytest tests/test_pi_server_backend.py -k "reasoning or todo_snapshot or model_change or fallback" -q`
Expected: FAIL because the current normalizer does not emit those events.

- [ ] **Step 3: Implement the minimal normalization changes in `codoxear/pi_messages.py`**

```python
# Extend normalize_pi_entries with specialized event emitters and fallback summaries.
```

- [ ] **Step 4: Run the same focused tests to verify they pass**

Run: `python3 -m pytest tests/test_pi_server_backend.py -k "reasoning or todo_snapshot or model_change or fallback" -q`
Expected: PASS.

### Task 2: Add failing UI render-pipeline tests

**Files:**
- Modify: `tests/test_pi_chat_ui_source.py`
- Test: `tests/test_pi_chat_ui_source.py`

- [ ] **Step 1: Write failing source-level tests for the Pi view toggle and event-type handling**

```python
# Assert app.js contains:
# - a Pi message view state or toggle
# - render handling for reasoning / todo_snapshot / pi_event-like rows
# - conversation/event-stream filtering helpers
```

- [ ] **Step 2: Run the focused UI tests to verify they fail**

Run: `python3 -m pytest tests/test_pi_chat_ui_source.py -q`
Expected: FAIL because those branches do not exist yet.

- [ ] **Step 3: Implement the minimal app.js/UI structure to satisfy the tests**

```javascript
// Add a Pi-only segmented toggle and rendering helpers keyed by event type.
```

- [ ] **Step 4: Run the focused UI tests to verify they pass**

Run: `python3 -m pytest tests/test_pi_chat_ui_source.py -q`
Expected: PASS.

### Task 3: Complete server-side Pi event normalization

**Files:**
- Modify: `codoxear/pi_messages.py`
- Test: `tests/test_pi_server_backend.py`

- [ ] **Step 1: Normalize assistant reasoning blocks**

```python
# Emit reasoning events from assistant content items with type == "thinking".
```

- [ ] **Step 2: Normalize structured tool results**

```python
# Emit tool_result rows with error/detail metadata and todo_snapshot rows for manage_todo_list.
```

- [ ] **Step 3: Normalize session/config events and generic fallback events**

```python
# Emit pi_session, pi_model_change, pi_thinking_level_change, and pi_event.
```

- [ ] **Step 4: Keep timestamps monotonic and preserve existing busy/diag semantics**

Run: `python3 -m pytest tests/test_pi_server_backend.py -q`
Expected: PASS with legacy Pi behavior preserved.

### Task 4: Implement the dual-view Pi web UI

**Files:**
- Modify: `codoxear/static/app.js`
- Modify: `codoxear/static/app.css`
- Test: `tests/test_pi_chat_ui_source.py`

- [ ] **Step 1: Add Pi view state and top-bar toggle**

```javascript
// Track piMessageView = "conversation" | "events" and show controls only for Pi sessions.
```

- [ ] **Step 2: Add row renderers for new event types**

```javascript
// Implement makeReasoningRow, makeTodoSnapshotRow, makePiEventRow.
```

- [ ] **Step 3: Route rendering through shared visibility helpers**

```javascript
// eventVisibleInCurrentView(ev), renderEventRow(ev), and rerender on toggle switch.
```

- [ ] **Step 4: Style the new cards and segmented toggle**

```css
/* Add production-ready Pi event cards, badges, error states, and toggle styles. */
```

- [ ] **Step 5: Run the UI-focused tests**

Run: `python3 -m pytest tests/test_pi_chat_ui_source.py -q`
Expected: PASS.

### Task 5: End-to-end regression verification

**Files:**
- Modify: none unless failures are found
- Test: `tests/test_pi_server_backend.py`, `tests/test_pi_chat_ui_source.py`, optionally related UI regression tests

- [ ] **Step 1: Run combined focused verification**

Run: `python3 -m pytest tests/test_pi_server_backend.py tests/test_pi_chat_ui_source.py -q`
Expected: PASS.

- [ ] **Step 2: Run broader related UI/backend regressions**

Run: `python3 -m pytest tests/test_pi_details_todo_ui.py tests/test_ui_regressions.py tests/test_message_index.py -q`
Expected: PASS.

- [ ] **Step 3: Inspect diffs for accidental non-goal changes**

Run: `git diff -- codoxear/pi_messages.py codoxear/static/app.js codoxear/static/app.css tests/test_pi_server_backend.py tests/test_pi_chat_ui_source.py`
Expected: Only planned Pi rendering changes.

## Self-Review

- Spec coverage: normalization, dual-view UI, and tests are each represented by dedicated tasks.
- Placeholder scan: all tasks point to exact files and verification commands.
- Type consistency: event names in the plan match the design (`reasoning`, `todo_snapshot`, `pi_session`, `pi_model_change`, `pi_thinking_level_change`, `pi_event`).
