## ADDED Requirements

### Requirement: A NotificationChannel abstraction SHALL dispatch push notifications to one or more channels

`VoicePushCoordinator` SHALL hold a list of `NotificationChannel` implementations and SHALL invoke each enabled channel's `send` method when a final-response notification fires. The existing Web Push pipeline SHALL be wrapped into a `WebPushChannel` that satisfies the same protocol, and a new `BarkChannel` SHALL be added.

#### Scenario: Both channels enabled both fire

- **WHEN** Web Push has at least one enabled subscription AND `bark_enabled` is true with a non-empty token
- **THEN** a final-response notification triggers exactly one Web Push send and exactly one Bark POST

#### Scenario: Disabled Bark channel does not POST

- **WHEN** `bark_enabled` is false
- **THEN** no HTTP request is issued to the Bark endpoint regardless of token state

### Requirement: Bark notifications SHALL be deliverable without HTTPS on the Codoxear server

`BarkChannel.send` SHALL POST a JSON body to `<bark_endpoint>/<bark_token>` containing `title`, `body`, `group="codoxear"`, and `url` (deep link to the session). The channel SHALL succeed when the Codoxear server is reached over plain HTTP, because the Bark API call is server-to-Bark and does not depend on the browser-side secure-context requirement.

#### Scenario: HTTP-only deployment delivers Bark notification

- **WHEN** Codoxear is reached via `http://...` (no HTTPS) AND `bark_enabled=true` with a valid token
- **THEN** a final-response message triggers a Bark POST whose response is 200 and a notification appears on the configured Bark device

#### Scenario: Notification deep link opens the source session

- **WHEN** a Bark notification is delivered with `url = <base_url>/#session=<session_id>`
- **THEN** tapping the notification opens the Codoxear UI scrolled to that session

### Requirement: Bark configuration SHALL persist in voice settings and be editable from the existing settings UI

The voice settings SHALL include `bark_enabled` (bool), `bark_endpoint` (string defaulting to `https://api.day.app`), and `bark_token` (string). The existing `/api/settings/voice` GET and POST SHALL accept and return these fields, and the settings UI SHALL render an editor for them next to the Web Push controls.

#### Scenario: Settings round-trip through the API

- **WHEN** the client POSTs `{"bark_enabled": true, "bark_token": "abc123"}` to `/api/settings/voice`
- **THEN** a subsequent GET returns the same values and `bark_endpoint` defaults to `https://api.day.app`

#### Scenario: Existing settings file without Bark keys loads with defaults

- **WHEN** the server starts with a voice settings file that does not contain any `bark_*` keys
- **THEN** `_clean_voice_settings` returns `bark_enabled=false`, `bark_endpoint="https://api.day.app"`, `bark_token=""` without raising
