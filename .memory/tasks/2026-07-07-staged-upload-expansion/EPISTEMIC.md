# Epistemic model

## Phenomenon
Upload expansion must turn attachments from an immediate PTY paste into a server-owned staged-list workflow. The product question is not how to select more files; it is where attachment truth lives before send.

## Current mechanism causing the defect
Existing `/inject_file` stages bytes under app-dir `uploads/<session>/` and immediately injects a generated path reference through the backend PTY. Persisted session state is mostly a boolean `pending_attachment`. This model cannot truthfully support multi-file selection, per-file removal, or clear-before-send because the browser may claim it removed an attachment whose path was already written to the backend input stream.

## Target mechanism
Stage first, commit later.
- Server records staged attachment entries per session with stable ids, display names, staged filesystem paths, sizes, and created timestamps.
- Upload routes stage bytes and update the list only; they do not call `inject_attachment_keys` or write to the backend PTY.
- Browser renders the server list as attachment chips/list rows and derives visible count from list length; users can remove one entry or clear all before send.
- Send is the commit boundary: if the user explicitly sends with staged attachments, the server composes generated `Attachment N: <path>` lines with the user text and uses the existing confirmed-send path as the single backend commit acknowledgement.
- Confirmed send success clears staged entries and `pending_attachment`; commit-unknown or send failure preserves them because backend receipt is unresolved or absent.

## Design commitments
- Keep `pending_attachment` as compatibility projection, but make it reflect staged-list non-emptiness for this new path.
- Queueing remains blocked while staged attachments exist; attachments are not queued in this slice.
- Partial multi-file upload failure must return visible per-file status while preserving successful staged entries.
- Delete-session cleanup still removes staged bytes under `uploads/<session>/`; symlink/sibling safety remains.
- Drag/drop, paste, and capture producers are later producers, not required for this slice.

## Live uncertainties
- Exact persisted representation and integration with `SessionStore` lifecycle.
- Exact route shape for list/remove/clear; it should avoid overloading `pending_attachment/clear` in a way that loses identity.
- Frontend proof path for multi-file input in agent-browser.
