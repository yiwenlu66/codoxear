# Code block copy buttons epistemic model

## Phenomenon
Assistant transcript messages often contain multiple commands or snippets. The user task is copying one snippet from the message, not the whole raw answer. Before this slice, code blocks had no block-local copy control, forcing manual selection or message-level copy.

## Accepted mechanism
Rendered markdown code blocks now include a hardcoded `.code-copy-btn` inside each `<pre>` before the `<code>` element. `app_code_copy.js` handles delegated clicks by resolving the nearest `<pre>`, querying its contained `<code>`, and copying `code.textContent` through the existing clipboard helper. Because the copied value is read from DOM text content, escaped entities in rendered HTML decode back to the original code, and no code text is embedded in button attributes. `app.js` invokes the code-copy handler before file-reference delegation and stops propagation on handled clicks. The existing message-level copy button remains an independent direct row control that copies the raw markdown message.

## Evidence
- Implementation and local validation: OPS entries for executor output and main validation; functional commit `1702f63`.
- Docker/browser proof: OPS entry for `75585b9`; desktop and mobile proof artifacts show two independent code buttons, exact copied payloads, no prose leakage, message-level full markdown copy preserved, mobile 44x44 controls, and no page overflow.
- Clean-room review: OPS entry for `4b0707f`; critic accepted the slice and found no blockers.

## Ruled out mechanisms
- Attribute/data-payload copy: rejected because embedding code in attributes would create escaping and payload-size concerns. DOM `textContent` is the correct source of truth.
- Message-level copy reuse: rejected because it copies surrounding prose and all blocks, which is the user pain point.
- File-reference click fallthrough: ruled out by delegated click ordering and proof; `.code-copy-btn` clicks are handled before file-reference handling.

## Current claim
The per-code-block copy slice is accepted. Future transcript markdown changes must preserve block-local copy semantics: each code block copies exactly its own `<code>.textContent`, message-level raw markdown copy remains intact, and mobile copy controls stay at least 44x44 without creating horizontal page overflow.
