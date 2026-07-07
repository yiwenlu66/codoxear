# EPISTEMIC

## Phenomenon
The context chip is both a context-pressure status display and an action affordance. When token data exists, it displays the current context percentage and exposes the same context-detail toast that pointer users already had. The defect was DOM semantics: the chip was a clickable `span`, so keyboard and screen-reader users did not receive native control behavior.

## Accepted mechanism
The repair is deliberately frontend-local: `#ctxChip` is a native `button.status-chip` with `type="button"` and a stable accessible name. Valid token projection enables and shows the button; missing/invalid token projection clears text/title, hides it with `display:none`, and disables it. Native button semantics provide Tab focus plus Enter/Space activation for the unchanged `ctxChip.onclick` detail toast.

The CSS mechanism is also local: `.status-chip` continues to define the visual chip density, while `button.status-chip` removes user-agent button chrome that would otherwise make the native control look like a generic form button.

## Evidence basis
- Functional commit `97876db` changes only `codoxear/static/app.js`, `codoxear/static/app.css`, and `tests/test_context_chip_accessibility_source.py`; backend token extraction, context math, session APIs, uploads, and launch behavior are untouched. See OPS implementation entry.
- Proof commit `650d165` records Docker/browser artifacts under `browser-artifacts/context-chip-accessible-19394/`: visible chip was `BUTTON`, `type=button`, enabled, `aria-label="Context usage details"`, text `Ctx 18%`, and title `Context input: 150000/183616 tokens (16384 reserved; window 200000).`; click, Enter, and Space all produced `ctx 150000/200000 (18% left)`; a no-token row hid and disabled the chip and was not programmatically focusable; fake brokers recorded `send_count=0` and `key_count=0`; desktop and mobile had no horizontal overflow.
- Clean-room review `8f7a2bf` accepted the slice with no blockers.

## Ruled out scope
Token formula, context-window mapping, backend adapters, upload staging, transcript projection, launch semantics, and Monaco/editor behavior are not part of this slice. The evidence supports only the chip control semantics and the preservation of existing context-token projection behavior.

## Residual nonblockers
- The accessible name `Context usage details` is adequate because the element is announced as a button, though a future copy polish could make it more action-oriented.
- Focus ring visibility was inferred from CSS cascade and default browser behavior rather than a focused-state screenshot; no `outline:none` selector applies to the chip.
- Source tests are string-based, with runtime/browser behavior covered by the Docker proof.

## Current claim
The context chip accessibility slice is accepted. Future changes to `#ctxChip` must preserve the invariant that token truth comes from the shared token dict, the visible chip is a native/equivalent accessible control, hidden/no-token state is non-focusable, and activating the chip never crosses a backend send/key boundary.
