# Operational ledger

## 2026-07-06T18:50:00Z Task initialized
- Objective: first upload expansion slice with server-owned staged attachment list and multi-file picker.
- Success boundary: attachments are staged/removable before send; generated backend-readable path references are committed only at send boundary.

## 2026-07-06T18:54:00Z Mechanism decision before implementation
- Observed current control flow: `/inject_file` validates readiness, stages bytes, builds `Attachment N: <path>`, wraps it in bracketed paste, and calls `manager.inject_attachment_keys` immediately.
- Observed direct send flow: `SessionSendCoordinator.send()` already owns the confirmed-send boundary and clears `pending_attachment` on success.
- Mechanism decision: upload routes must become stage-only; send must compose attachment references into the confirmed send text and clear staged entries only on confirmed success.
- Rationale: separate pre-send PTY writes would recreate the wrong boundary and cannot truthfully support remove/clear before send.
