# Phase 6 cohesion audit

## Audit rule and conclusion

Phase 6 permits a split only when a second responsibility is demonstrably present along an existing boundary. File size is not evidence. A `SPLITTABLE` verdict below therefore identifies:

1. a responsibility that already has a different semantic owner;
2. the exact cluster that can move;
3. every reference from that cluster back into the code that stays; and
4. an acyclic dependency direction after the move.

| Module | Verdict | Evidence-gated action |
| --- | --- | --- |
| `app_transcript.js` | **SPLITTABLE** | Move the two chat-search state runtimes to the existing chat-search owner, and move the older-load request/UI runtime to the existing message-history owner. Keep the transcript mechanics together. |
| `app_voice.js` | **SPLITTABLE** | Move browser-notification state, widget rendering, delivery/subscription transport, and its DOM construction behind a notification runtime. Keep voice settings, announcement listener, and live-audio playback. |
| `app_file_viewer_operations.js` | **SPLITTABLE** | Move the contiguous touch-editor interaction cluster into a real touch-editor runtime, replacing the present one-method touch relay. Keep media/open/save/unsaved/conflict operations. |
| `app_new_session.js` | **COHESIVE** | Do not split. Its sections are coupled parts of one launch-form transaction, and the apparently attractive seams either retain broad bidirectional plumbing or create no clearer owner. |

No `NEEDS-REDESIGN` verdict is warranted for these four modules: each module either has an evidence-backed leaf seam or is one cohesive workflow. The audit does not authorize further splits of the remaining code merely because it remains large.

---

## 1. `app_transcript.js`

### Exported factory inventory

The module exports ten factories:

1. `createPendingUserController`
2. `createTranscriptSlotRuntime`
3. `createTypingRowRuntime`
4. `createTranscriptRenderRuntime`
5. `createTranscriptDomRuntime`
6. `createTranscriptScrollRuntime`
7. `createTranscriptEventRuntime`
8. `createOlderLoadRuntime`
9. `createLoadedChatSearchRuntime`
10. `createChatSearchAllRuntime`

It also exports the pure or collection-mutating helpers `normalizeTextForPendingMatch`, `pendingMatchKey`, `eventKey`, `chatAssistantDedupeKey`, `normalizeTailEvent`, `normalizeTranscriptState`, `transcriptKey`, `historyCursorFromPayload`, `hasUsableOlderHistory`, `transcriptSnapshotFromData`, `transcriptIdentityFromData`, `tailCacheMatchesSession`, `rememberTailSnapshot`, `appendTailSnapshotEvents`, `isAgentInternalDeliveryUserEvent`, `hasHumanOriginatedUserEvent`, `startsTypingCountWindow`, `thinkingModeForTokens`, `normalizedTranscriptEvents`, and `firstUnreadMessageRow`.

### Existing section structure and dependency surfaces

| Lines | Section / responsibility | What this cluster needs | What the rest of the module needs instead |
| --- | --- | --- | --- |
| 2–69 | **Message identity and pending-row commit.** Normalize user/assistant text into stable keys and replace a local pending row when the logged user event arrives. | The key helpers need only strings and timestamps. `createPendingUserController` needs `sessionState.get`, `takePendingUserMatch`, `chatInner.querySelector`, markdown/time renderers, decoration rebuild, and `markEventSeen`. | Slot/tail identity, view rendering, scrolling, older loads, and search do not need the pending-row DOM node or markdown/time renderers. |
| 71–199 | **Tail payload normalization and cache mutation.** Normalize tail events and transcript identities, derive history availability, and update a bounded per-session tail cache. | Plain payloads, a `Map`, a session/session-index fallback, and a max-event limit. No DOM. | Rendering needs DOM and row factories; scrolling needs viewport nodes; search needs timers and request cancellation. |
| 210–360 | **Per-session transcript slot authority.** Own slot state (`bound` / `pending_bind` / `failed`), identity epochs, renewal boundaries, active live cursor, and tail caches. | `getSession` or `sessionIndex`, `maxTailEvents`, and the preceding tail helpers. | It does not need chat DOM, search controls, older-load UI, or scroll geometry. |
| 362–387 | **Turn/activity classification.** Distinguish human user events from internal delivery, decide when a typing-count window starts, and choose token/block thinking display. | Event arrays and scalar state only. | The DOM runtimes do not need backend-delivery prefix knowledge. |
| 389–570 | **Typing and idle-subagent row widget.** Own the two activity rows, their counters/gauge, insertion before the sentinel, and auto-scroll requests. | `root`, `bottomSentinel`, `el`, `shouldAutoScroll`, and `scheduleScrollToBottom`. | Tail identity, event dedupe, older loading, and search do not need these nodes. |
| 572–759 | **Transcript event-to-row rendering.** Normalize/dedupe event batches; append, replace, detach, and prepend rows while preserving view policy. | `root`, `bottomSentinel`, `document`, `safeMakeRow`, pending/dedupe/seen callbacks, `domRuntime`, `scrollRuntime`, `typingRowRuntime`, and the history slack limit. | Slot/cache logic and search request state are independent of row construction. |
| 761–840 | **Transcript DOM decoration and trimming.** Clear the render surface, rebuild day/group decorations, and trim rows without breaking scroll state. | Root/sentinel/older-wrap nodes, `el`, date label helpers, row and trim-policy callbacks, `afterDecorate`, and `scrollRuntime`. | It does not need session identities, pending-event queues, network request state, or search query state. |
| 842–1095 | **Viewport and scroll policy mechanics.** Own auto-scroll/live-tail flags, per-session scroll memory, jump/time indicators, scroll/wheel/touch reactions, and older-page edge triggers. | Chat/jump/time nodes, RAF, selection/search/first-visible/date callbacks, older-load cancel/trigger callbacks, and pixel thresholds. | Event identity and slot/cache code need none of the viewport nodes or geometry. |
| 1097–1251 | **Recent-event and local-echo authority.** Own the bounded seen-key set, pending-user queue, local echo sequence, and exact/fallback pending matching. | Four injected key/normalization functions and `maxRecentEventKeys`. No DOM or network. | Scroll, decoration, and search do not need the pending queue. |
| 1253–1355 | **Older-page request/UI state.** Own an abortable older-load request token, loading/error controls, cancel-on-scroll flag, and auto-trigger cooldown. | `olderWrap`, `olderButton`, `olderError`, `olderErrorText`, `AbortControllerCtor`, `nowMs`, and `autoCooldownMs`. | Core transcript slots/events/rendering do not need these message-history controls or request lifecycle. |
| 1357–1556 | **Chat-search state machines.** Own loaded-row match/index state and the debounced/abortable server-search request state. | `createLoadedChatSearchRuntime` is self-contained. `createChatSearchAllRuntime` needs only timeout functions, `AbortControllerCtor`, a debounce value, and the local `requireFunction` guard. | Core transcript state does not need query/match/count/truncation state or search request timers. |

### Verdict: **SPLITTABLE**

Two leaf clusters are demonstrably owned elsewhere. They should move; the remainder should not be decomposed on size alone.

#### Seam A — chat-search state belongs with `app_chat_search.js`

**Move**

- `createLoadedChatSearchRuntime` (lines 1357–1444)
- `createChatSearchAllRuntime` (lines 1446–1556)

**Stay**

- Message identity/pending reconciliation
- Slot/tail cache authority
- Typing/subagent widget
- Event/render/DOM/scroll mechanics
- Older-load runtime until Seam B is applied

**Hard reference evidence**

References from the moving search cluster into staying code are exactly:

1. `requireFunction`, used by `createChatSearchAllRuntime` to validate the injected timer/abort callback. `createLoadedChatSearchRuntime` references no staying symbol.

There are no calls from either search runtime to a transcript identity, slot, render, DOM, scroll, typing, or older-load function. The validator can become a destination-local guard, leaving no runtime edge back to `app_transcript.js`.

**Ownership ambiguity eliminated**

`app_chat_search.js` already owns search UI, query orchestration, server requests, loaded-row highlights, stepping, and disposal, but it receives its two state authorities from `app_transcript.js` through `createLoadedChatSearchRuntime` and `createChatSearchAllRuntime`. Search therefore has two owners today. Co-locating the runtimes makes `app_chat_search.js` own search state and behavior end-to-end.

**Dependency direction**

`app_chat_search.js` constructs its own local runtimes. It need not import `app_transcript.js` for search. `app_transcript.js` does not import chat search. No cycle is introduced.

#### Seam B — older-load request/UI state belongs with `app_message_history.js`

**Move**

- `createOlderLoadRuntime` (lines 1253–1355)

**Stay**

- The pure history-cursor helpers (`historyCursorFromPayload`, `hasUsableOlderHistory`) because tail caching also uses them
- All transcript identity, cache, event, render, DOM, scroll, and typing mechanics

**Hard reference evidence**

References from `createOlderLoadRuntime` into staying code are exactly:

1. `requireNode`
2. `requireFunction`

It calls no transcript-specific function and reads no transcript-owned state. Both references are generic constructor guards that can live in the destination module.

**Ownership ambiguity eliminated**

`app_message_history.js` already owns older-page loading and directly constructs `createOlderLoadRuntime`, but the state machine controlling its request token, abort controller, cooldown, error, and button is owned by `app_transcript.js`. Moving the runtime gives message history one owner for the entire older-page workflow.

**Dependency direction**

`app_message_history.js` owns/constructs the runtime and may continue importing the pure transcript cursor helpers. The direction remains `app_message_history.js` → `app_transcript.js`; `app_transcript.js` does not import message history, so there is no cycle.

#### Why the rest stays together

`createTranscriptRenderRuntime`, `createTranscriptDomRuntime`, and `createTranscriptScrollRuntime` are separate mechanics under the policy authority in `app_transcript_view.js`; they are not unrelated responsibilities. The view controller explicitly composes those mechanics to enforce the live/browsing/loading/replacing state boundary. Slot identity, event dedupe, pending echoes, row rendering, and viewport preservation are the interacting mechanisms required to maintain one transcript view. This audit found no comparably clean second owner for those clusters.

---

## 2. `app_voice.js`

### Exported factory inventory

The module exports two factories:

1. `createVoiceDom`
2. `createVoiceController`

### Existing section structure and dependency surfaces

| Lines | Section / responsibility | What this cluster needs | What the rest of the module needs instead |
| --- | --- | --- | --- |
| 1–60 | **Imports, browser-capability aliases, constants, and guards.** Establish voice/HLS, notification-device, modal, timer, and validation helpers. | `app_voice_helpers.js`, `app_modal.js`, and constants. | Each later cluster uses only a subset: HLS helpers for announcements, device/base64 helpers for notifications, modal helpers for settings. |
| 62–168 | **Voice-area DOM construction.** Build the notification button/panel, announcement button, hidden audio element, and settings dialog. | `root`, `voiceHost`, `el`, and `iconSvg`. | The controller needs the resulting nodes, while transport/state code does not need DOM construction itself. The notification nodes (71–82) are already a contiguous sub-block. |
| 170–304 | **Controller dependency binding, browser targets, state, and listener cleanup registry.** Bind DOM nodes, app effects, browser APIs/timers, and initialize all announcement, notification, settings, and dialog state. | Injected app API/effects/storage/URL helpers; browser `window`/`navigator`/`document`/Notification/crypto/AudioContext; timers. | Announcement playback does not need notification feed/panel/service-worker state; notifications do not need HLS/watchdog/listener state. |
| 305–343 | **Cross-feature accessors.** Report announcement/notification enablement and settings-open state; derive voice stream URL/credentials/audio readiness/error. | Local storage-backed flags and `voiceSettings`. | Notification transport only needs its own enablement and notification snapshot; live audio only needs announcement and audio settings. |
| 344–589 and 1049–1076 | **Announcement listener and live-audio playback.** Select native HLS vs hls.js, maintain heartbeat, retry and stall watchdogs, reset/restart playback, and describe start failures. | `liveAudio`, HLS/browser helpers, `voiceSettings.audio`, API listener endpoint, timers, storage, and the announcement button renderer. | Notification feed/subscription/panel code does not use live-audio source, watchdog, HLS object, or listener heartbeat. |
| 590–794 and 932–1047 | **Browser notifications.** Own local opt-in, desktop/push transport projection, service-worker subscription, desktop delivery/click focus, notification sound, feed cursor/items/read state, panel rendering, and device enable/toggle. | Notification DOM nodes, Notification/AudioContext/service-worker/PushManager, device classification/base64 conversion, API/auth handling, storage, focus callback, and a VAPID key/server subscription snapshot. | Announcement playback does not need the feed, unread set, service worker, push endpoint, Notification API, or notification panel. |
| 797–930 | **Settings and unattended-prompt persistence plus background refresh.** Synchronize the form, load/save unattended prompt, load/save `/api/settings/voice`, update UI, and coordinate a combined refresh. | Settings form nodes, modal-open state, API, `voiceSettings`, unattended prompt state, and both announcement and notification refresh/render entry points. | Low-level audio and notification runtimes need only their relevant slice of the returned settings snapshot. |
| 1078–1109 | **Settings-dialog visibility/focus lifecycle.** Open/close the settings dialog and restore focus. | Modal hooks, settings/backdrop nodes, document focus, RAF, settings/form loaders. | Notification delivery and live-audio playback do not need dialog focus policy. |
| 1111–1254 | **Widget and browser event wiring.** Bind announcement toggle, notification panel/enable controls, audio events, settings fields, visibility resume, and Save. | All three feature clusters because this is their current shared assembly point. | Each event family touches only its corresponding runtime except the settings Save path, which refreshes both snapshots. |
| 1256–1314 | **Combined disposal/bootstrap/public surface.** Clear timers/listeners/media, notification resources, handlers, and restore persisted announcement opt-in. | Every cluster's state because disposal is currently aggregated. | A split can delegate notification cleanup without sharing its internal state. |

### Verdict: **SPLITTABLE**

#### Seam — browser notifications are a second responsibility

**Move**

- Notification DOM creation from `createVoiceDom`: `notificationBtn`, panel/header/list/empty/clear/enable nodes, and their attachment (lines 71–82 and corresponding returned fields)
- Notification-owned state: `localNotificationEnabled`, `desktopNotificationTimers`, `deliveredDesktopNotificationIds`, `notificationFeedSinceTs`, `notificationItems`, `readNotificationIds`, `notificationPanelOpen`, `notificationAudioContext`, `notificationState`, and `swRegistration`
- Notification functions: `setNotificationEnabledLocal`; desktop/push transport projection; desktop delivery and sound; list/read/panel rendering; feed polling; service-worker/subscription sync; enable/toggle
- Notification button/panel handlers and notification-specific disposal
- The notification portion of `updateVoiceUi`, so the notification widget has one renderer

**Stay**

- Announcement opt-in, listener heartbeat, HLS/live-audio state and playback
- Voice/TTS settings persistence and settings-dialog lifecycle
- Unattended-prompt field persistence
- The announcement button and live-audio DOM
- A thin background/settings coordinator that hands the notification runtime the server snapshot

**Hard reference evidence**

With the notification cluster defined above, its references into the staying controller are exactly:

1. `updateVoiceUi`, currently called after local notification enablement changes and subscription sync. This edge disappears when notification button/panel rendering moves with the notification widget.
2. `voiceSettings.notifications.vapid_public_key`, currently read when creating a push subscription. The staying coordinator can pass the `/api/settings/voice` snapshot (or just its VAPID value) into `syncNotificationState`, so the notification runtime does not read voice state directly.
3. `isSettingsOpen` / `syncVoiceSettingsFormFromState`, reached indirectly because the shared `updateVoiceUi` also refreshes the settings form. Notification transitions do not need that side effect once notification rendering is local; the staying settings loader/save path retains form synchronization.

Everything else used by the moving cluster is either moved state/functionality or an already injected external dependency (`api`, auth-loss handling, storage, URL/version helpers, focus callback, browser targets, timers). It does not call any HLS, live-audio, announcement-heartbeat, unattended-prompt, or modal-focus function.

The resulting narrow interface from the staying coordinator to the notification runtime is enumerable: `enabledLocally()`, `syncState(snapshot)`, `pollFeed({ prime })`, and `dispose()`. Rendering happens inside notification state transitions rather than through a callback into voice.

**Ownership ambiguity eliminated**

The module currently treats voice announcements and browser notifications as one authority even though they have independent opt-ins, transports, DOM, state, and browser APIs. The ambiguity is visible in a concrete double-writer: `updateVoiceUi` toggles `notificationBtn.active` from local enablement (line 843), then calls `renderNotificationPanel`, which toggles the same class from unread count (line 728). The latter write wins, so one displayed state has two meanings and two writers. Moving the entire notification widget and consolidating its render path gives the notification button/panel one owner.

The combined `/api/settings/voice` response is a transport optimization, not evidence that audio playback and browser notifications are one responsibility. Passing its notification slice across the seam preserves that optimization.

**Dependency direction**

`app_voice.js` constructs/imports the notification runtime and passes injected infrastructure plus settings snapshots to it: `app_voice.js` → notification module. The notification module never imports `app_voice.js` and never reads voice-owned state, so no cycle is introduced.

---

## 3. `app_file_viewer_operations.js`

### Exported factory inventory

The module exports one factory:

1. `createFileViewerOperationsRuntime`

The factory returns a large method surface; there are no other module-level exports.

### Existing section structure and dependency surfaces

| Lines | Section / responsibility | What this cluster needs | What the rest of the module needs instead |
| --- | --- | --- | --- |
| 4–6 | **Runtime assembly/state declaration.** Destructure the injected viewer/editor/candidate dependencies and initialize save, unsaved, media-render, and touch-selection state. | The full constructor contract. | The breadth of this one destructure hides that later clusters use sharply different subsets. |
| 7–237 | **Media render state and load-result planning.** Own video fallback state, PDF task state/disposal, compatible-video transitions, and normalize API load results into text/diff/image/PDF/video/download plans. | `fileStatus`, time/size formatting, active-file state mutators/readers, `applyFileMode`, current-request check, and preview callbacks. | Save/unsaved logic does not need PDF tasks, content types, or video-preview tokens; touch interaction does not need media APIs. |
| 238–319 | **Mode/edit-button projection.** Derive diff/preview/download/video control availability, synchronize editor read-only state, and render the edit/save button. | Active identity/kind/view/editor capability predicates, media fallback, edit button/icon, status/unavailable state, and touch-toolbar updater. | Network open/fetch does not need button DOM; touch selection consumes only read-only/toolbar refresh hooks. |
| 320–464 | **Dirty/save transaction.** Own dirty/save-pending state and tokens, build/write requests, apply success/error, restore/discard editor text, and update UI. | Session/identity/version/text accessors, API, file-state mutators, editor/read-only/button/status effects, picker/history updates, and conflict rendering. | Media and touch selection do not need save tokens or file-write payloads. |
| 465–524 | **Unsaved-choice state machine.** Own the pending resolver, prompt plan, save/discard/cancel transitions, and plain-fallback reset. | Dirty state, prompt/dialog callbacks, discard/save operations, and mode/button/toolbar effects. | Media rendering and clipboard selection do not need the unsaved modal resolver. |
| 525–643 | **Guarded navigation and draft opening.** Guard mode changes/close/open against unsaved edits and unavailable sessions; resolve a path; prepare/open drafts; finalize successful opens. | Unsaved guard, candidate/path/identity state, inspect/open methods, status, picker, view/edit mode, and active-selection memory. | Touch interaction does not inspect paths or own open requests. |
| 644–937 | **Touch-editor interaction.** Own touch selection state, toolbar projection, diff selection visibility, directional selection, touch/delete suppression, text insertion, clipboard/manual-paste fallback, and selection copy. | Editor focus/position/selection/edit APIs; touch-mode/viewport/modal/text-entry predicates; toolbar/dialog/clipboard/toast effects; active-file kind/text/view/writability; time. It uses no API, path inspection, candidate cache, media/PDF state, save token, or open request. | The rest needs file APIs, active identity/version, candidate/diff mode, media plans, unsaved/save state, and status/picker effects; it does not need touch anchor/head/goal-column/delete-suppression state. |
| 938–1002 | **Control action adapters.** Handle diff/preview/edit/save/video-preview buttons/shortcuts and construct a download URL. | Thin calls into mode/save/video operations plus current identity/session state. | This is the UI-facing edge of the core operations. The generic shortcut gate is shared with the touch section. |
| 1003–1087 | **File-open transport.** Start an open request, fetch file/diff data, apply/render it, and handle open/draft errors. | Session/path/request authority, API, render result callback, view mode, active buffer reset/status, and finalization. | Touch selection and media fallback state do not perform network fetches. |
| 1089–1142 | **Save-conflict UI/actions.** Validate a conflict, reload or keep editing, construct conflict controls, and expose current conflict. | Current session/path/unavailable state, reload confirmation/open, editor focus, `el`, and status. | Touch and media state do not need conflict controls. |

### Verdict: **SPLITTABLE**

#### Seam — touch-editor interaction is a distinct, already split-owned feature

**Move**

- Touch-selection state: `fileTouchSelectMode`, anchor, head, goal column, and native-delete suppression deadline
- The touch interaction cluster at lines 644–937: toolbar projection, selection mode/movement, touch key handling, delete suppression, insertion, paste fallback, and copy
- Its event-facing methods should become the actual implementation behind the existing file-touch controller rather than remain methods on generic viewer operations

Keep the generic `fileEditorShortcutBlocked` gate with the core editor shortcut/action section because `handleFileEditorSaveShortcut` also uses it; inject that gate into the touch runtime.

**Stay**

- Media/PDF/video load planning and state
- Mode/edit-button projection
- Dirty/save and unsaved-choice state
- Guarded open/draft/mode/close operations
- Generic action adapters, open transport, download URL, and save-conflict handling

**Hard reference evidence**

References from the moving touch cluster into staying code reduce to exactly three named callbacks:

1. `syncFileEditorReadOnly` — called when touch selection starts/resets.
2. `setFileDirty` — called after touch paste/edit changes the editor text.
3. `fileEditorShortcutBlocked` — the general editor/modal target gate, also used by the staying save shortcut.

Its other requirements are constructor dependencies already supplied from the viewer/editor boundary, not references to another local operations cluster. They are enumerable: `isFileViewerOpen`, `isTextFileKind`, `focusEditor`, `updateFileTouchToolbar`, `useTouchFileEditorControls`, `hasActiveFileCodeEditor`, `hasBlockingFileEditorModal`, `isTextEntryTarget`, `eventTargetElement`, `normalizeFileEditorPosition`, `applyFileEditorSelection`, `isCollapsedFileSelection`, `positionAfterInsertedText`, `fileEditorEditSupportAvailable`, `updateFileDiffEditorOptions`, `showFilePasteDialog`, `hideFilePasteDialog`, `clipboardReadAvailable`, `readClipboardText`, `fileEditorDeleteCommandForKey`, `isActiveFileEditorInput`, `getActiveFileSelectionText`, `copyToClipboard`, `focusActiveFileCodeEditor`, `nowMs`, `setToast`, `getFileEditorText`, `currentFileViewMode`, `currentActiveFileKind`, `currentActiveFileText`, `activeFileEditorWritable`, `activeFileEditorIdleWritable`, `activeFileEditorIdleTextWritable`, and `blockUnavailableFileAction`.

That surface is distinct from the rest's API/path/candidate/version/save/media dependencies; it is broad because editor interactions are behaviorally rich, not because it reaches across the proposed seam.

**Ownership ambiguity eliminated**

A separate `createFileTouchController` already exists in `app_file_ops.js`, but it is only a one-method relay to `fileViewerController.handleFileTouchSelectionKeydown`. The real touch feature state and behavior live here, while touch button binding and toolbar projection live elsewhere. This leaves the named touch controller without ownership of the touch interaction it purports to control. Moving the contiguous state/behavior cluster behind a real touch-editor runtime consolidates that existing responsibility; it is not a line-count extraction.

**Dependency direction**

The file viewer/controller constructs the touch runtime with editor/capability dependencies and the three callbacks above: viewer/core operations → touch runtime. The touch runtime does not import core operations; callbacks are injected. No module cycle is introduced.

#### Adjacent correctness finding (not a cohesion verdict)

Two save-conflict paths in the audited module reference undeclared identifiers:

- `isSaveConflictCurrent` reads `activeFilePath` at line 1090, but this factory owns no such binding and does not receive it in `options`; the available authority is `currentActiveFileIdentity()`.
- `renderSaveConflict` calls `fileSaveConflictTarget` at line 1118, but the module neither defines nor imports it.

A direct ESM runtime probe of the factory produced:

- `renderSaveConflict`: `ReferenceError: fileSaveConflictTarget is not defined`
- `isSaveConflictCurrent`: `ReferenceError: activeFilePath is not defined`

This is a latent save-conflict bug, not evidence for another split. It remains unfixed because this task is analysis-only.

The neighboring `app_file_viewer_controller.js` also retains dead-looking declarations for save/media/touch state now owned by this operations runtime. That residue reinforces the touch ownership ambiguity but is outside this audit's edit scope.

---

## 4. `app_new_session.js`

### Exported factory inventory

The module exports two factories:

1. `createNewSessionController`
2. `createNewSessionDialogController`

The first implements the launch-form decisions and subflows. The second owns dialog state/DOM/events, instantiates the first, submits the launch request, and owns subscriptions/disposal.

### Existing section structure and dependency surfaces

| Lines | Section / responsibility | What this cluster needs | What the rest of the module needs instead |
| --- | --- | --- | --- |
| 7–30 | **Dependency guards.** Validate function, input, class-list, text, and present-node dependencies. | No feature state. | Shared by both exported factories. |
| 37–110 | **Decision-controller binding and subflow state.** Bind launch-config accessors/mutators, catalog/default sources, model/cwd/resume/worktree DOM and callbacks; initialize cwd/resume/debounce state. | Dialog-owned selection state via accessors, `sessionCatalog`, field/menu nodes, menu render/close callbacks, candidate fetcher, and launch-option controls. | The outer dialog owns the actual backend/provider/reasoning/menu-open/start-busy state and DOM lifecycle. |
| 112–432 | **Backend/provider/model/reasoning and launch-preset decision logic.** Parse provider/model input, derive options/capabilities/defaults, remember choices, render/select values, and apply a prior session as a launch preset. | Backend/provider/reasoning accessors and mutators; defaults/latest sessions; `CodoxearLaunch`; model/status/reasoning nodes; model/reasoning menu callbacks; backend/fast/tmux setters. | Cwd discovery needs filesystem/recent-cwd inputs and timers; resume/worktree needs cwd/git metadata. |
| 436–590 | **Working-directory suggestions and validation UI.** Merge fuzzy recent paths with debounced filesystem suggestions, own cwd error/info display, apply a suggestion, and update the name placeholder. | `recentCwds`, `CodoxearDisplay`, cwd/name/menu/hint nodes, `el`, menu focus/close/apply callbacks, `fetch`, and timers. Applying a cwd intentionally schedules resume discovery. | Provider/model parsing does not need filesystem suggestions or cwd menu state. |
| 594–716 | **Resume-candidate discovery and selection.** Format/render candidates, fetch them for `(cwd, backend)`, update cwd/git metadata, preserve/clear selection, and own the debounce/disposal sequence. | Backend and cwd values, `fetchResumeCandidates`, resume DOM, display age/ID helpers, cwd error/info state, worktree sync, and timers. | Provider/model choice does not need candidate rows; worktree availability does need the resulting git metadata and resume selection. |
| 720–735 | **Tmux/worktree projection.** Enable tmux from catalog capability and derive worktree controls/start label from git metadata and resume selection. | `tmuxAvailable`, cwd `git_repo`, `resumeSelection`, toggles/fields/branch input/start button. | This cluster deliberately consumes the preceding target-selection state. |
| 737–777 | **Decision-controller public surface.** Expose the operations used by the dialog and tests. | All preceding subflows. | No independent state. |
| 779–826 | **Dialog factory dependencies and state.** Bind root/document/window/modal/menu/spawn effects and own backend/provider/reasoning/fast/literal-input/menu/start/focus state. | `sessionCatalog`, `sessionState`, DOM helpers, modal/menu policy, resume fetcher, and spawn action. | The inner controller receives only accessors/mutators and relevant nodes. |
| 827–868 | **Dialog DOM construction.** Build every launch field, picker/menu, toggle, and action and append the overlay. | `root`, `el`, `iconSvg`. | Decision logic operates on these nodes but does not create them. |
| 870–973 | **Run-config menu/tab rendering and backend transitions.** Render reasoning/model/backend controls, derive run-config visibility, switch backend, and reset dependent selections. | Dialog state, defaults, `CodoxearLaunch`, inner controller methods, and menu nodes. | Modal lifecycle/start submission need the resulting state, not option enumeration internals. |
| 975–1034 | **Dialog/menu lifecycle, launch submission, and event wiring.** Apply menu visibility/position, open/close/focus, validate/build/submit the launch request, and bind every control. | Modal hooks, session selection/catalog, inner controller, `spawnSession`, all nodes, and RAF/document events. | Catalog refresh does not need submission internals. |
| 1036–1066 | **Live-default refresh, subscriptions, and disposal.** Refresh open-dialog options when catalog fields change and unsubscribe/clear timers. | `sessionCatalog.subscribe`, open-state predicate, render/sync methods. | No separate responsibility; it keeps the launch form current. |

### Verdict: **COHESIVE**

The module owns one user transaction: collect, validate, and submit a new-session launch specification. Provider/model/reasoning, cwd, resume, worktree, tmux, dialog lifecycle, and submission are not independent products; they are coupled constraints on that one request.

Concrete coupling makes the apparent section boundaries poor module seams:

- Backend determines provider/model/reasoning choices **and** filters resume candidates.
- A cwd lookup returns both filesystem validity and git metadata; that result controls resume choices and whether worktree creation is offered.
- Selecting a resume candidate disables worktree creation and changes the Start label.
- Applying a launch preset updates backend, provider/model, reasoning, fast tier, and tmux as one operation.
- Starting must atomically read cwd, backend/provider/model/reasoning, optional resume ID, optional worktree branch, tmux, name, and fast tier.

The provider/model section is especially unsuitable for a mechanical split: it consumes 24 controller-local dependencies or callbacks (`backend`, `provider`, `reasoningEffort`, literal-provider state, defaults/latest sessions, four state mutators, model/status/reasoning nodes, model/reasoning renderers, fast/backend/tmux setters, dialog-menu application, and model-menu close). Extracting it would preserve a large bidirectional interface rather than establish an owner.

The cwd/resume/worktree cluster has a smaller mechanical boundary, but no ownership ambiguity exists: no other module claims this launch-target state. Extracting it would add a child abstraction for fields that are validated and submitted only by this dialog. Its internal coupling through `cwdInfo`, `resumeSelection`, and the backend accessor is the launch workflow's invariant, not accidental entanglement.

The two exported factories are a legitimate layering boundary rather than two responsibilities: `createNewSessionController` supplies testable form decisions; `createNewSessionDialogController` owns the widget and transaction lifecycle that use those decisions. Keep both in this module.

One documentation detail is stale: lines 32–35 say selection state is owned by `app.js`, while `createNewSessionDialogController` now owns it locally (lines 808–819). That comment should be corrected in a future scoped edit, but it is not a reason to split.

---

## Cross-cutting duplicate-key finding: `app_transcript_render.js`

The duplicate is in the object returned by `createTranscriptRenderController`:

- first `transcriptView` shorthand property: line 863
- second `transcriptView` shorthand property: line 877

Both entries refer to the same lexical function declared at lines 175–178:

```js
function transcriptView() {
  if (!transcriptViewController) throw new Error("transcript view controller is not initialized");
  return transcriptViewController;
}
```

JavaScript object-literal semantics make the later property win, so the line-877 entry overwrites the line-863 entry. Because both shorthand entries evaluate to the exact same function binding, the returned value is unchanged.

The line-863 occurrence is therefore dead/redundant syntax, not a shadowed alternate implementation. It is accidental rather than intentional: commit `8c54deab` added the earlier entry when transcript view transitions were centralized, while the existing tail entry remained; later commit `a727e3d9` extended that tail entry with `markClickLoad` without removing the duplicate. The current effect is an esbuild warning, not a latent behavioral bug. The warning is still useful: a future duplicate with different values would silently take last-write-wins semantics, so the redundant first occurrence should be removed in a scoped implementation change.

---

## Audit boundary and residual risks

- This report is analysis only. No product source, tests, bundle, plan, git index, or refs were changed; only this requested report was created.
- The recommended splits identify ownership seams; they do not authorize behavior changes.
- The save-conflict `ReferenceError` findings in `app_file_viewer_operations.js` remain live residual defects.
- The duplicate `transcriptView` entry remains a build warning but currently returns the same function under both occurrences.
