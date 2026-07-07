# EPISTEMIC

## Phenomenon
The context chip is both status display and action affordance: when token data exists it displays context pressure and clicking it opens Details. Current evidence from the scout indicates it is constructed as a `span.status-chip`, so the action does not participate in native keyboard/screen-reader control semantics.

## Current mechanism
Context-pressure truth is already owned by backend-specific token extraction and the shared frontend `setContext(tok)` projection. Accessibility failure is likely confined to DOM control semantics and styling: the chip has `onclick` but lacks native button behavior.

## Ruled out scope
Token math, context-window mapping, backend adapters, upload staging, transcript projection, and launch semantics are not part of this slice unless implementation evidence shows the chip accessibility fix cannot be isolated.

## Current claim
A bounded frontend change should make `#ctxChip` a native/equivalent control while preserving the same text/title/show-hide behavior and the same Details action.

## Most model-changing question
Does turning the chip into a real `button.status-chip` preserve layout and CSS behavior across desktop/mobile, or do global button styles require targeted status-chip button CSS?
