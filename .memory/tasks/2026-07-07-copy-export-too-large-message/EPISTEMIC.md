# Copy export too-large messaging epistemic model

## Phenomenon
Copy Conversation has a known server-side export cap. When a transcript log exceeds the cap, the server correctly refuses full export, but the browser currently reports a generic copy failure. That message misidentifies a size/export limit as a clipboard/copy failure and gives the user no useful next action.

## Current mechanism
`copyConversation()` treats every thrown error from `/messages/export`, formatting, and clipboard write the same way: `copy failed: <message>`. Server route `handle_messages_export` returns 413 plus `max_bytes` for oversized logs, so the frontend has enough evidence to distinguish this failure from clipboard denial or network/auth errors.

## Target mechanism
The frontend recognizes the known export-too-large response shape and renders a specific, concise toast that names the conversation/export size limit. The server guard remains authoritative; the UI only changes the error projection.

## Live uncertainties
- Exact thrown error shape from `api()` for 413 responses (`status`, `obj`, `message`) should be confirmed from the app helper before implementing.
- Browser proof needs an oversized log inside Docker without committing the log.

## Current claim
This is a bounded product polish gap: the underlying protection is correct, but the user-facing failure mechanism is wrong.
