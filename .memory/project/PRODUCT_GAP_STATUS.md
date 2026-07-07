# Product gap status

## Current state

`recovery/product-gaps` has no remaining bounded, mechanism-backed, product-visible defect currently justified by source evidence and accepted invariants. Further work should be treated as feature/polish unless a fresh observation contradicts this state.

Evidence basis: fresh scout after coarse-pointer code-copy closure (`/tmp/codoxear-next-slice-after-coarse-code-copy.md`, theorist run `a558de2b-d417-42ce-b9e2-c0f7ffa96c21`) inspected current source/tests/memory at `53768a3` and found prior candidates either closed or non-defects.

## Recently closed defect themes

- Code-block copy buttons: block-local accessible `.code-copy-btn`, exact nearest-code payload, mobile/coarse-pointer touch target coverage, no overflow.
- Destructive confirmations: destructive app confirmations declare `destructive:true`, focus Cancel initially, trap Tab/Shift-Tab, preserve cancel-before-mutation and no-native-confirm invariants.
- Copy Conversation count truthfulness: success toast counts formatter-owned copied sections, not raw export rows.
- Coarse-pointer code-copy targets: `pointer:coarse` + `hover:none` surfaces get 44x44 `.code-copy-btn` and 58px `<pre>` right padding; fine-pointer desktop stays compact.

## Remaining non-defects / feature territory

- Range/partial conversation copy is new functionality; per-message and per-code-block copy already satisfy the existing too-large message’s “copy a smaller range” advice.
- Code-copy opacity `0.72` on coarse-pointer tablets is visible cosmetic polish, not a target-size/overflow contract violation.
- `.msg-copy-btn` at 40px on coarse-pointer is not covered by the documented code-copy 44px contract.
- Non-`appConfirm` Tab traps and secondary-viewer Escape-chain cleanup are defensive/code-quality work; current modal isolation prevents app-control reach and no data-affecting defect is shown.

## Continuation rule

Do not keep mining the same surface for defect slices without a new observation. The next justified roadmap step should be either explicit feature work, release/promotion planning, or a fresh user-reported defect with a falsifiable mechanism.
