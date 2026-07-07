# Coarse-pointer code-copy touch-target epistemic model

## Phenomenon
Rendered markdown code-block copy buttons should remain usable on companion touch devices. The accepted code-block copy invariant says mobile code-copy controls are part of the companion-device contract: at least 44x44 CSS px and no page-level horizontal overflow.

## Current mechanism
`.code-copy-btn` is a standalone classed button with base 30x30 sizing. The phone `max-width: 520px` media rule raises it to 44x44, but the coarse-pointer media rule does not target `.code-copy-btn`. The generic coarse-pointer `button { min-height: 40px }` rule loses to the class-level `.code-copy-btn { min-height: 30px }`, and `.code-copy-btn` is not `.icon-btn`. Therefore touch devices wider than 520px can keep the 30x30 control.

## Target mechanism
A coarse-pointer media rule explicitly sizes `.code-copy-btn` to at least 44x44 and increases `.md pre` right padding so the larger control does not overlap code or create horizontal overflow. Phone behavior remains unchanged; fine-pointer desktop stays compact.

## Live risks
- Widening all copy controls would exceed the slice; `.msg-copy-btn` is out of scope.
- Enlarging `.code-copy-btn` without padding could obscure code or create overflow.
- Browser proof must emulate/measure coarse-pointer behavior, not only viewport width.

## Current claim
This is the strongest remaining bounded product defect after confirm-focus and copy-count closure: a documented touch-target contract violation on an edge but legitimate companion-device surface.
