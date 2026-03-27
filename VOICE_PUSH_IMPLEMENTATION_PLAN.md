# Voice Announcements And Mobile Push Plan

## Product Shape

- Treat each assistant message in the rollout log as an independent event.
- Distinguish three assistant message classes from rollout-log structure:
  - narration message
  - final assistant response message
  - everything else
- Fan out by message class:
  - narration message: optional voice announcement
  - final assistant response message: optional short spoken summary and optional push notification
  - everything else: no voice, no push

The core requirement is message-level fan-out, not turn-level fan-out. A single turn may emit multiple narration messages and then one final assistant response message.

## Runtime Behavior

For each assistant message parsed from the rollout log:

1. Determine whether the message is a narration message or a final assistant response message.
2. If the message is a narration message and narration announcements are enabled:
   - build spoken text as `From <session name>. <message text>`
   - choose a deterministic voice from session metadata
   - call server-side TTS
   - append resulting audio to the merged live HLS stream
3. If the message is a final assistant response message:
   - if final-response announcements are enabled:
     - summarize the message to about 20 words with an OpenAI-compatible text model
     - build spoken text as `From <session name>. <summary>`
     - choose a deterministic voice from session metadata
     - call server-side TTS
     - append resulting audio to the merged live HLS stream
   - if push notifications are enabled for any subscribed devices:
     - send a push notification to each enabled device subscription
4. If the log shape does not clearly identify message class, skip fan-out instead of guessing.

## Message Classification

Message classification should come from rollout-log structure, not text heuristics.

Required classifier outputs:

- `narration`
- `final_response`
- `other`

Implementation route:

- Reuse the existing rollout-log parsing path already used by the web UI.
- Add a dedicated extractor that maps each parsed assistant message to one of the three classes above.
- Keep the classifier strict. If the parsed event does not match a known narration or final-response shape, return `other`.
- Record the classification result in a delivery ledger so the server does not re-process the same message repeatedly.

## Audio Pipeline

### Stream Model

- Keep one merged HLS stream for the whole application.
- Do not create separate HLS streams per session.
- Use a normal browser `<audio>` element pointing at the merged HLS playlist URL.
- Rely on Safari's media playback stack for lock-screen/background playback after an explicit user gesture starts audio.
- Do not attempt to keep background JavaScript alive to feed audio chunks.

### Audio Queue

- Maintain one global server-side audio job queue.
- Every narration or final-response announcement becomes one queue item.
- Queue items must be serialized so spoken output does not overlap or reorder.

Queue item fields:

- message id
- session id
- session display name
- message class
- original text
- optional summary text
- chosen voice
- enqueue timestamp

### Voice Selection

- Do not expose voice choice in the UI.
- Derive voice deterministically from a stable session metadata hash, such as session id or alias.
- Use the same voice for the same session across time.
- Use different voices across sessions when possible so the merged stream remains audibly separable.

### TTS Inputs

Narration message:

- spoken text: `From <session name>. <narration text>`

Final assistant response message:

- first summarize to about 20 words
- spoken text: `From <session name>. <summary>`

### HLS Generation

- Server owns all HLS generation.
- TTS output audio is segmented into short rolling HLS segments.
- The merged playlist should expose a bounded live window and bounded segment retention.
- The server should retain enough recent segments for reconnects after transient network loss.
- The HLS manager should append segments only after a queue item fully completes, not while multiple jobs race in parallel.

## Summarization Path For Final Responses

- Final assistant responses should not be spoken verbatim.
- Add a server-side summarization step using an OpenAI-compatible text model before TTS.
- The summarization prompt should target a spoken mobile announcement of about 20 words.
- The summarizer should output plain speech-ready text, not markdown and not quoted text.
- If summarization fails, skip final-response announcement rather than speaking the entire original message.

## Push Notification Path

- Use iOS Web Push for an installed Home Screen PWA.
- This is browser push via service worker, not native-app APNs integration.

Browser-side route:

- register a service worker
- request notification permission from an explicit user action
- subscribe with `PushManager.subscribe()`
- send the subscription payload to the server

Server-side route:

- persist subscriptions per device
- store a per-device notifications-enabled boolean
- on each final assistant response message, send Web Push to all enabled device subscriptions
- include enough payload to route the user back into the correct session on click

Notification payload fields:

- session id
- session display name
- short text preview
- message id
- timestamp

Service worker route:

- receive `push`
- call `showNotification(...)`
- on `notificationclick`, open or focus the web app and navigate to the target session

## Settings And Controls

### Runtime Control

Voice announcement control should be directly accessible in the main UI, not buried in settings.

Recommended route:

- replace the current refresh button with a voice announcement toggle
- the primary visible control is a single announcement on/off toggle
- attach a compact popover or small menu to that control for the two underlying announcement switches:
  - announce narration messages
  - announce final assistant responses

This keeps the common path simple while still supporting the two required speech modes.

### Settings Screen

Keep the settings surface minimal.

Expose only:

- TTS base URL
- TTS API key
- per-device notification enable toggle

Do not expose:

- voice selection
- per-session voice settings
- per-session audio routing
- arbitrary announcement prompt tuning

The app is currently single-user, so TTS settings are global.

## Persistence Model

### Global Settings

One global settings record:

- `tts_enabled_for_narration`
- `tts_enabled_for_final_response`
- `tts_base_url`
- `tts_api_key`

Optional internal-only fields if needed:

- summarization model id
- TTS model id

These do not need to be user-configurable if a fixed default is acceptable.

### Device Subscriptions

One record per device subscription:

- subscription payload
- device label or user-agent metadata
- `notifications_enabled`
- created timestamp
- updated timestamp
- last success timestamp
- last failure timestamp

### Delivery Ledger

One record per processed assistant message:

- message id
- session id
- message class
- narrated status
- push status
- summary status
- timestamps

This ledger exists to prevent duplicate processing when sessions are rescanned or the server restarts.

## Server Components To Add

### Message Fan-Out Hook

- Add a hook in the server path that ingests or polls parsed rollout-log messages.
- The hook should run once per newly observed assistant message.
- It should classify the message and enqueue downstream work.

### Announcement Coordinator

- Central server component that receives classified assistant messages.
- Responsibilities:
  - check global announcement settings
  - decide whether to summarize
  - choose deterministic voice
  - enqueue audio jobs
  - decide whether to send push notifications
  - update the delivery ledger

### Summarization Client

- Server-side client for the OpenAI-compatible text endpoint.
- Takes a final assistant response message and returns a short spoken summary.
- The summarizer is used only for final assistant response announcements.

### TTS Client

- Server-side client for the OpenAI-compatible TTS endpoint.
- Uses global base URL and API key.
- Takes speech text plus selected voice and returns audio bytes.

### Global HLS Stream Manager

- Owns the merged live playlist and audio segment files.
- Appends segments from the serialized audio queue.
- Exposes a stable HLS URL for the browser player.

### Push Subscription Manager

- Persists service-worker subscriptions.
- Supports subscribe, disable, enable, and prune flows.
- Handles invalid subscription cleanup after provider send failures.

### Web Push Sender

- Sends push notifications to all enabled device subscriptions for each final assistant response message.
- Includes the session-routing payload used by the service worker.

## Browser Components To Add

### PWA Support

- web app manifest
- service worker registration
- installable Home Screen behavior

### Main UI Control

- replace refresh button with a voice announcement control
- reflect current global announcement state
- allow fast enable or disable without entering settings

### Audio Player

- a persistent `<audio>` element bound to the merged HLS playlist
- explicit user gesture to start playback
- state display for idle, buffering, live, and error

### Notification UI

- one explicit action to enable notifications on this device
- one per-device on/off toggle after subscription is registered

## API Surface To Add

Suggested server endpoints:

- `GET /api/settings/voice`
- `POST /api/settings/voice`
- `GET /api/notifications/subscription`
- `POST /api/notifications/subscription`
- `POST /api/notifications/subscription/toggle`
- `GET /api/audio/live.m3u8`
- `GET /api/audio/segments/<segment>`

The exact routes can be adjusted to match current server naming conventions, but the split should remain:

- global voice settings
- per-device notification subscription management
- merged live HLS playback

## Ordering And Deduplication Rules

- Process each assistant message at most once for audio and push.
- Narration messages may appear multiple times in one turn and should each be eligible for announcement.
- Final assistant response messages should each be eligible for:
  - one summarized voice announcement if enabled
  - one push notification per enabled device
- Deduplicate by stable message id from the parsed rollout event.
- If a stable message id is not currently available, add one at the parser boundary before implementing fan-out.

## UI Behavior Summary

- User opens the web app on iPhone and installs it to the Home Screen.
- User enables notifications on this device.
- User configures global TTS base URL and API key once in settings.
- User toggles voice announcements from the main UI.
- While using Codoxear:
  - narration messages are spoken into the merged live stream if narration announcements are enabled
  - final assistant responses are summarized, spoken into the merged live stream if final-response announcements are enabled, and pushed as notifications to enabled devices

## Implementation Notes Specific To This Repo

- The message fan-out hook belongs in the existing rollout-log parsing or session message ingestion path, not in the client.
- Voice and push decisions should be made server-side because the server remains alive when the phone screen is locked.
- The current single-user architecture means global TTS settings are sufficient.
- Device-level notification toggles should remain independent even in single-user mode.
- The merged HLS stream should use session display names already visible in the UI so spoken prefixes match what the user sees.
