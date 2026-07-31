# Code Context — UI/Product Ergonomics Reconnaissance

## Files Retrieved

1. `codoxear/static/index.html` (all 39 lines) — HTML shell: external deps (hls.js, monaco-editor, Inter font, service worker), no inline JS templates
2. `codoxear/static/app.js` (all 9402 lines) — Entire client: vanilla JS SPA, no framework, single-file monolith
3. `codoxear/static/app.css` (all 2375 lines) — All styles: responsive breakpoints at 880px, 700px, 520px
4. `codoxear/server.py` (lines 4208–4440) — `list_sessions()`: ~45-field JSON payload per session, includes files[], token{}, harness fields, git_branch, priority metadata
5. `codoxear/server.py` (lines 5489–5512) — `GET /api/sessions`: returns `sessions[]`, `recent_cwds[]`, `new_session_defaults{}`, `tmux_available`
6. `codoxear/server.py` (lines 6280–6500) — `messages/tail`, `messages/live`, `messages/history` endpoints: cursor-based pagination
7. `codoxear/server.py` (lines 5783–5840) — `file/search` endpoint: fuzzy match via git ls-files or os.walk with heap-based scoring
8. `codoxear/server.py` (lines 1553–1700) — `_file_search_score()`, `_search_git_relative_files()`: server-side fuzzy search scoring
9. `codoxear/static/app.js` (lines 3556–3650) — `appendEvent()`, `renderTranscript()`, `prependOlderEvents()`: DOM rendering for chat messages
10. `codoxear/static/app.js` (lines 2981–3070) — `makeRow()`: constructs a single message DOM node with markdown + copy button
11. `codoxear/static/app.js` (lines 2860–2980) — `rebuildDecorations()`, `trimRenderedRows()`, day-separator logic
12. `codoxear/static/app.js` (lines 3810–3940) — `pollMessages()`, `pollLoop()`: live-cursor polling at 200–900ms adaptive interval
13. `codoxear/static/app.js` (lines 9300–9402) — Boot sequence, session timer at 2500ms `setInterval`, loads voice/notification/sessions
14. `codoxear/static/app.js` (lines 5200–5370) — `renderNewSessionBackendTabs()`, `setNewSessionBackend()`, provider/model/reasoning picker logic
15. `codoxear/static/app.js` (lines 7360–7560) — `fileSearchScore()`, `scheduleSessionFileSearch()`: client-side fuzzy scoring + 120ms debounce + server API
16. `tests/test_chat_scrollback_source.py` — Source-level assertions on scrollback, tail caching, and DOM window management
17. `tests/test_sidebar_touch_mode.py` — Tests touch/desktop action mode selection
18. `tests/test_message_cursor.py` — Cursor encode/decode roundtrip tests
19. `tests/test_file_picker_session_state.py` — File picker session isolation via Node VM

## Key Code

### Polling / Network Traffic

```js
// Session list: polled every 2500ms unconditionally (app.js:9345-9360)
sessionsTimer = setInterval(async () => {
  await refreshSessions();
  await loadVoiceSettings();
  await syncNotificationState();
  if (notificationsEnabledLocally()) await pollNotificationFeed();
}, 2500);

// Message poll: adaptive interval (app.js:3930-3935)
let nextMs = 900;                      // idle default
if (now < pollFastUntilMs) nextMs = 200; // after user action
else if (turnOpen) nextMs = 250;       // agent busy
pollTimer = setTimeout(pollLoop, nextMs);
```

### Session List Payload Size

```python
# server.py:4300-4356 — Each session item includes 45+ fields:
items.append({
    "session_id", "thread_id", "pid", "broker_pid",
    "agent_backend", "owned", "transport", "cwd",
    "start_ts", "updated_ts", "log_path", "busy",
    "queue_len", "token", "thinking", "tools", "system",
    "harness_enabled", "harness_cooldown_minutes",
    "harness_remaining_injections", "alias",
    "files": list(files),      # ← full absolute paths
    "git_branch", "model_provider", "preferred_auth_method",
    "provider_choice", "model", "reasoning_effort",
    "service_tier", "tmux_session", "tmux_window",
    "launch_id", "spawn_nonce", "priority_offset",
    "snooze_until", "dependency_session_id",
    "time_priority", "base_priority", "final_priority",
    "blocked", "snoozed",
})
```

### Chat DOM Window Management

```js
// app.js:1490-1495
const INIT_PAGE_LIMIT_DESKTOP = 60;
const INIT_PAGE_LIMIT_MOBILE = 24;
const OLDER_PAGE_LIMIT = 60;
const CHAT_DOM_WINDOW = 260;
const CHAT_DOM_WINDOW_WITH_HISTORY_SLACK = 320; // 260 + 60
const OLDER_TOP_TRIGGER_PX = 1;
```

### Chat Navigation — Current State

The **only** navigation affordances in long conversations today:
1. **"Jump to latest" button** (`jumpBtn`): appears when scrolled away from bottom
2. **"Load older messages" button** (`olderBtn`): appears at top when `has_older` is true, auto-triggers at `scrollTop <= 1`
3. **Day separators** (`day-sep` class): inserted between messages on different calendar days
4. **Per-message timestamps** (`.ts` element): `HH:MM` shown in each bubble's bottom-right corner

There is **no** in-chat search, no previous/next user message navigation, no jump-to-time, and no time-range markers beyond day separators.

### New Session Provider/Model

```js
// app.js:5203-5230 — Backend tabs: "codex" and "pi" only
for (const backend of ["codex", "pi"]) { ... }

// Provider: dropdown picker (button → menu), separate from model
// Model: combobox input with dropdown of recent/configured models
// Reasoning: dropdown picker
// No combined "provider/model" selector
// Keyboard: ArrowDown/ArrowUp, Enter, Escape supported on all comboboxes
```

### File Viewer Fuzzy Search

```js
// Client-side: fileSearchScore() at app.js:7364 — mirrors server scoring
// Server-side: _file_search_score() at server.py:1553 — fuzzy char-by-char matching
// Network: scheduleSessionFileSearch() at app.js:7516 — 120ms debounce → GET /api/sessions/{id}/file/search?q=...&limit=120
// Local candidate list: fileCandidateList from /api/sessions/{id}/file/list + git changed files
// Search fallback: client scores local candidates when server results loaded; combined + sorted
```

## Architecture

### Data Flow Summary

```
Browser (app.js)
  │
  ├── GET /api/sessions                 every 2.5s (full session list + defaults + cwds)
  │     also: loadVoiceSettings, syncNotificationState, pollNotificationFeed
  │
  ├── GET /api/sessions/{id}/messages/tail    on session open (init)
  ├── GET /api/sessions/{id}/messages/live    every 200-900ms (cursor poll)
  ├── GET /api/sessions/{id}/messages/history on "load older" (backward paging)
  │
  ├── GET /api/sessions/{id}/file/search      on file picker typing (120ms debounce)
  ├── GET /api/sessions/{id}/file/list        on file viewer open (refresh)
  ├── GET /api/sessions/{id}/file/read        on file open
  │
  └── POST /api/sessions/{id}/send            user message
```

### Rendering Model

- Vanilla DOM manipulation, no virtual DOM or diffing framework
- `chatInner` is a flexbox column; messages are `.msg-row` divs
- `trimRenderedRows()` limits visible DOM to `CHAT_DOM_WINDOW = 260` rows
- `rebuildDecorations()` tears out all `.day-sep` elements and reinserts them (traverses all rows)
- Markdown rendering uses a custom hand-rolled parser (`mdToHtml()`) with a 1200-entry LRU cache (`mdCache`)
- `sessionsWrap.innerHTML = ""` rebuilds entire sidebar session list on each `refreshSessions()` call (every 2.5s)

### Mobile Layout

- Sidebar: fixed overlay at `max-width: 880px`, slides in/out with transform
- At `max-width: 520px`: larger touch targets, toast hidden, file/queue viewers go full-screen
- `isMobile()`: `window.matchMedia("(max-width: 880px)")`
- Visual viewport tracking via `updateAppHeightVar()` → CSS custom properties `--appH`, `--vvTop`, `--vvBottom`

---

## Observations, Candidates & Recommendations

### 1. Network Traffic Under Slow Mobile Networks

**Current state:**
- `GET /api/sessions` fires every 2500ms **unconditionally**, even when backgrounded, idle, or on a slow connection
- Each response includes `files[]` (absolute paths), `log_path`, `token{}` per session — payload grows linearly with session count
- Voice settings, notification state, and notification feed are fetched alongside every session refresh
- No `ETag`/`If-None-Match` or conditional polling; full payload always transferred
- No visibility-based throttling; same frequency when tab is hidden

**Candidate mechanisms:**
| Mechanism | User benefit | Complexity |
|-----------|-------------|------------|
| **Visibility-throttle**: slow session poll to 15–30s when `document.hidden` | Saves 85%+ background traffic on mobile | Low — add `visibilitychange` listener |
| **Conditional GET / ETag**: server sends `ETag`, client sends `If-None-Match`; 304 = empty body | Saves transfer bytes when nothing changed | Medium — header plumbing in `_json_response` + `api()` |
| **Separate voice/notification polls**: decouple from session refresh; poll less often (e.g. 10s) | Reduce per-tick latency for session list | Low — move to own timer |
| **Trim session payload**: omit `files[]`, `log_path`, `token{}`, `thinking/tools/system` from list; fetch on demand or via diagnostics | ~30-40% payload reduction per session | Medium — client must lazy-load some fields |
| **Adaptive poll interval**: slow session polling when no session is busy | Saves traffic during idle periods | Low |

**Likely files/functions:**
- `app.js` boot sequence (line ~9345): `setInterval(async () => { ... }, 2500)`
- `app.js` `refreshSessions()` (line ~3208): calls `loadVoiceSettings()`, `syncNotificationState()`, `pollNotificationFeed()`
- `server.py` `list_sessions()` (line ~4208): constructs the full payload
- `server.py` `_json_response()` for ETag support

**Validation plan:**
- Synthetic test: create Docker instance with 10-15 sessions. Measure `/api/sessions` response size (expect ~3-5 KB per session, 30-75 KB total).
- Use Chrome DevTools Network throttling (Slow 3G: 400 Kbps down) to observe poll latency stacking.
- Confirm `document.hidden` correctly pauses polling in backgrounded tab.
- Confirm ETag-based 304 responses reduce transfer to ~200 bytes when nothing changes.

### 2. Long-Conversation Navigation

**Current state:**
- Only navigation: "Jump to latest" (bottom), "Load older" (top), day separators, per-message HH:MM timestamps
- No search within chat
- No "previous/next user message" jump
- No time-based navigation (e.g. "jump to 14:00" or "1 hour ago")
- DOM window is capped at 260 rendered rows; scrolling through old history requires multiple "Load older" loads
- `rebuildDecorations()` iterates all rendered rows on every insert/prepend — O(n) on rendered DOM size

**Candidate mechanisms:**
| Mechanism | User benefit | Complexity |
|-----------|-------------|------------|
| **In-chat search bar** (Ctrl+F / ⌘F or dedicated button): search visible + loaded messages by text content | Core navigation for users who remember content but not position | Medium — local DOM search + highlight + scroll-to-match |
| **Previous/next user message** (keyboard shortcut, e.g. Shift+↑/↓): jump between user turns | Fast orientation; know where your prompts were | Low — collect `.msg-row.user` and navigate |
| **Time-range jump** (hour markers, or a mini-timeline along the scrollbar): tap to jump to approximate time | Orientation in 8+ hour sessions | Medium-High — needs server-side time-index or front-loaded metadata |
| **Server-side text search** (`/api/sessions/{id}/messages/search?q=...`): search the full log, return matching events with cursors | Search entire history, not just loaded DOM | High — new server endpoint, JSONL scan with byte offsets |
| **Sticky "current time" indicator**: show current visible message time in the scroll gutter or top bar | Orientation while scrolling | Low — use `firstVisibleMessageRow()` + timestamp |

**Design constraints** (from PROMPT.md):
- Navigation must not turn the chat view into a dense log/debug transcript
- Must preserve deliberate chat-detail omission
- Timestamps/roles are valid affordances; dense metadata panels are not
- Mobile and keyboard ergonomics both matter

**Likely files/functions:**
- `app.js` `rebuildDecorations()` (line ~2895): day-sep insertion logic
- `app.js` `appendEvent()` (line ~3556): event insertion
- `app.js` `renderedMessageRows()` (line ~2557): DOM query for `.msg-row`
- `app.js` `chatInner` element + `.chat` scroll container
- `app.css` `.day-sep` (line ~710), `.jumpBtn` (line ~2366)
- `server.py` message tail/live/history endpoints (lines ~6280-6500)

**Validation plan:**
- Create a synthetic JSONL log with 500+ messages spanning multiple days and hours.
- Mount in Docker test instance; verify initial load + "Load older" + DOM trimming behavior.
- Add Ctrl+F search; verify it finds text in loaded messages and scrolls to match.
- Test previous/next user message jump with keyboard shortcut; verify focus/scroll behavior.
- Measure `rebuildDecorations()` cost at 260 rows via `performance.now()`.
- Verify mobile: search bar must not permanently consume vertical space.

### 3. UI Responsiveness

**Current state:**
- `sessionsWrap.innerHTML = ""` + full DOM rebuild every 2.5s for the sidebar
- `rebuildDecorations()` removes/reinserts day separators by querying all `.day-sep` + all `.msg-row` on every event append
- Custom markdown renderer with 1200-entry cache; no lazy rendering
- No `requestIdleCallback` or `requestAnimationFrame` batching for sidebar rebuild
- `chatMarkdownHtmlCached()` runs synchronously during `makeRow()`; expensive for long messages

**Candidate mechanisms:**
| Mechanism | User benefit | Complexity |
|-----------|-------------|------------|
| **Incremental sidebar update**: diff session list and patch only changed cards | Eliminates 2.5s jank on sidebar rebuild | Medium — stable keyed DOM patching |
| **Lazy markdown rendering**: defer rendering of off-screen messages | Faster initial page load and "Load older" | Medium — IntersectionObserver + placeholder |
| **Day-separator insertion**: only insert/remove day-seps near the insertion point, not full rebuild | Reduces appendEvent() cost from O(n) to O(1) | Low-Medium |
| **Virtual scrolling**: only render visible messages + buffer | Eliminates DOM-window cap; smooth scrolling at any scale | High — framework-level change |

**Likely files/functions:**
- `app.js` `refreshSessions()` (line ~3208): `sessionsWrap.innerHTML = ""`
- `app.js` `rebuildDecorations()` (line ~2895)
- `app.js` `chatMarkdownHtmlCached()` (line ~1335)
- `app.js` `makeRow()` (line ~2981)
- `app.css` `.session` cards

**Validation plan:**
- Use Performance DevTools in Docker browser to profile a 10-session sidebar refresh cycle.
- Measure `rebuildDecorations()` with 260 DOM rows via instrumented `pushPerfSample()`.
- Compare before/after for incremental sidebar patch.
- Test on throttled CPU (4× slowdown) to simulate slow mobile device.

### 4. New-Session Provider/Model Ergonomics

**Current state:**
- Backend selection: two icon tabs (Codex logo, Pi logo) — hard-coded loop `["codex", "pi"]`
- Provider: separate dropdown button → vertical menu of provider strings
- Model: separate combobox input → dropdown of models from server defaults + recent sessions
- Reasoning: separate dropdown button
- Fast mode: checkbox toggle (Codex-only)
- Total: 4-5 separate controls to configure a session
- Recent choices remembered per-backend in `localStorage`
- Provider and model are independent selectors — no "provider/model" compound

**Candidate mechanisms:**
| Mechanism | User benefit | Complexity |
|-----------|-------------|------------|
| **Combined provider/model selector**: single combobox with "provider / model" display | 1 click/tap instead of 2; clearer mental model | Medium — combine data sources, show grouped list |
| **Most-recent-first ordering**: show recently-used provider/model combinations at top | Faster repeat launches | Low — track MRU in localStorage |
| **Keyboard-friendly quick launch**: type partial match of "provider/model" to filter | Power user speed | Low — reuse existing combobox filtering from model picker |
| **Preset buttons**: "Launch like [session X]" in sidebar context menu | Zero-config re-creation | Medium — needs launch-config extraction from session |

**Likely files/functions:**
- `app.js` `renderNewSessionBackendTabs()` (line ~5203)
- `app.js` `setNewSessionBackend()` (line ~5254)
- `app.js` `setNewSessionProvider()` (line ~5290)
- `app.js` `renderNewSessionProviderMenu()` (line ~5300)
- `app.js` `sessionModelOptions()` (line ~5330)
- `app.js` `filteredNewSessionModelOptions()` (line ~5350+)
- `server.py` `_read_new_session_defaults()` (line ~2404)
- `server.py` `_read_pi_launch_defaults()` (line ~2322)

**Validation plan:**
- Open new-session dialog on mobile (520px width); verify all controls are accessible without horizontal scroll.
- Test combined provider/model combobox with filtering; verify it correctly handles provider-specific model lists.
- Verify keyboard flow: Tab through fields, type partial model name, Enter to select.
- Check localStorage MRU persistence across page reloads.

### 5. File Viewer Combobox / Fuzzy Search

**Current state:**
- File picker: combobox input with dropdown menu
- **Local file list**: `fileCandidateList` = session files + git changed files (fetched on file viewer open)
- **Server search**: `GET /api/sessions/{id}/file/search?q=...&limit=120` with 120ms debounce
- **Hybrid scoring**: server returns scored results; client also scores local candidates and merges
- **Sorting**: `added > score > changed > alphabetical`
- **Draft file**: user can type a non-existing path to "Create new file"
- **Session isolation**: `fileViewerSessionId` + `fileSearchSessionId` tracking
- **Recent files**: session `files[]` field contains previously-opened absolute paths; displayed first when no query

**Issues observed:**
1. When typing, the menu shows "Searching files..." while the 120ms debounce + network round-trip completes — gap is noticeable on slow networks
2. Client `fileSearchScore()` duplicates server `_file_search_score()` — scoring must be kept in sync manually
3. No "recent files" section header or visual separation — recently-opened files mix with other candidates
4. `fileCandidateList` is rebuilt on every file viewer open via `refreshFileCandidates()` (network call)
5. No fuzzy-match highlighting in results — user can't see which characters matched

**Candidate mechanisms:**
| Mechanism | User benefit | Complexity |
|-----------|-------------|------------|
| **Instant local-first results**: score client-side candidates immediately, show while server search loads | Eliminates wait; server results merge in when ready | Low — already partially implemented; show local results before server returns |
| **Match highlighting**: highlight matched characters in file paths | Visual feedback for fuzzy match quality | Medium — need to track match positions from scoring |
| **Recent files header**: visual section separator "Recently opened" / "Search results" | Clearer information hierarchy | Low — CSS separator |
| **Cache file list across opens**: don't refetch if session hasn't changed | Faster file viewer open | Low — check session identity before re-fetching |

**Likely files/functions:**
- `app.js` `scheduleSessionFileSearch()` (line ~7516)
- `app.js` `visibleFilePickerEntries()` (line ~7560)
- `app.js` `renderFilePickerMenu()` (line ~7650+)
- `app.js` `fileSearchScore()` (line ~7364)
- `app.js` `refreshFileCandidates()` (line ~7789)
- `server.py` `/file/search` handler (line ~5783)
- `server.py` `_file_search_score()` (line ~1553)
- `server.py` `_search_git_relative_files()` (line ~1639)

**Validation plan:**
- Open file viewer in Docker instance with a large git repo (10K+ files).
- Type a query; verify local results appear immediately while "Searching files..." shows for server results.
- Verify recently-opened files appear at top when no query is entered.
- Test edge cases: empty query, query matching no files, file not found, session cwd that doesn't exist.
- Measure debounce + network time on throttled connection (Slow 3G).

### 6. Overall Clean Responsiveness

**Current state:**
- Clean, minimal chat UI with bubble messages, day separators, and typing indicator
- Good mobile viewport handling (`--appH`, safe-area-inset support, visual viewport tracking)
- Swipe gestures for session actions on mobile; hover-reveal on desktop
- Toast notification for transient messages (hidden at 520px width — mobile users miss feedback)
- Context chip shows token usage in topbar
- No transitions on message appearance; messages just append
- `scrollToBottom()` uses raw `chat.scrollTop = chat.scrollHeight` — no smooth scrolling

**Candidate mechanisms:**
| Mechanism | User benefit | Complexity |
|-----------|-------------|------------|
| **Smooth scroll to bottom**: `scrollBehavior: 'smooth'` or CSS `scroll-behavior: smooth` | Less jarring on new message arrival | Low — but must be careful not to conflict with auto-scroll logic |
| **Toast on mobile**: replace hidden toast with a brief in-chat indicator or snackbar at bottom | Mobile users see feedback | Low-Medium |
| **Message fade-in**: subtle opacity transition on new messages | Polish; clearer "something just appeared" signal | Low — CSS transition on `.msg-row` |
| **Loading skeleton**: show placeholder while session transcript loads | Perceived performance improvement | Low — replace blank `chatInner` with skeleton |
| **Scroll position persistence**: remember scroll position per session | Returning to a session doesn't jump to bottom | Medium — store per-session scroll offset |

**Likely files/functions:**
- `app.js` `scrollToBottom()` (line ~2867)
- `app.js` `appendEvent()` (line ~3556)
- `app.js` `renderSessionTail()` (line ~3707)
- `app.css` `.chat`, `.chatInner`, `.msg-row`
- `app.css` `.toast` (hidden at 520px)
- `app.js` `openSession()` (line ~3750+)

**Validation plan:**
- Open a session on a 520px-wide viewport; send a message and verify feedback is visible.
- Enable smooth scrolling; verify it doesn't break auto-scroll tracking (`isNearBottom()`).
- Test message fade-in with rapid message bursts to ensure no visual stutter.
- Profile DOM operations during rapid polling (turnOpen, 250ms) on throttled CPU.

---

## Start Here

**`codoxear/static/app.js` line ~9345** (the boot/timer setup block)

This is where the session poll timer, voice/notification side-fetches, and initial session load all converge. It's the origin of the 2.5s unconditional polling that dominates mobile network traffic, and any responsiveness optimization must understand and modify the timer setup here. The `pollLoop()` at line ~3911 is the second critical entry point for the message polling fast path.

## Supervisor Coordination

No blockers. All findings are read-only observations from source code. No runtime access was needed and no live state was touched.

## Summary of Highest-Value Interventions (Ranked)

1. **Visibility-throttled session polling** — Biggest impact per line of code changed. Saves 85%+ background traffic.
2. **In-chat search** (Ctrl+F / button) — Most impactful navigation feature for long conversations.
3. **Incremental sidebar update** — Eliminates DOM thrashing every 2.5s; biggest responsiveness win.
4. **Previous/next user message jump** — Lowest-complexity navigation win; high user value in long sessions.
5. **Sticky current-time indicator** — Orientation while scrolling; low complexity.
6. **Trim session list payload** — Removes `files[]`, `log_path`, `token{}` from list; reduces per-session payload ~30%.
7. **Combined provider/model selector** — Simplifies new-session flow from 4 controls to 2.
8. **Local-first file search results** — Show client-scored results instantly while server search loads.
9. **Toast on mobile** — Currently hidden at 520px; mobile users miss all feedback.
10. **Smooth scroll + message fade-in** — Polish items; low effort, visible improvement.
