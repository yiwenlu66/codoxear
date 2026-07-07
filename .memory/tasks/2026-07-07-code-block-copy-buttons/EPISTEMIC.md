# Code block copy buttons epistemic model

## Phenomenon
Assistant transcript messages frequently contain code fences and command snippets. Codoxear renders those snippets as static code blocks without a block-local copy affordance, so users must either copy the entire message or manually select text inside the code block.

## Current mechanism
`app_markdown.js` renders code blocks as `<pre><code data-lang="...">escaped code</code></pre>`. `app_message_rows.js` adds a per-message `.msg-copy-btn` that copies the whole raw markdown message. `app.js` delegates file-reference clicks from `chatInner`, but there is no code-block copy delegation.

## Target mechanism
Each rendered code block has an accessible copy control. The control copies only its sibling/contained `<code>.textContent` through the existing clipboard helper and gives feedback without affecting message-level raw copy or file-reference clicks. CSS keeps the control usable on desktop and at least 44x44 on mobile without widening the transcript.

## Live risks
- HTML-string markdown rendering can duplicate SVG/icon concerns or break escaping if code text is embedded in attributes.
- Click delegation must stop propagation before file-reference handling or row interactions see the button click.
- Existing markdown renderer tests assert exact `<pre><code>` output and must be updated without weakening code content assertions.
- Clipboard proof needs real browser evidence; if clipboard write is blocked, instrumentation should prove the attempted payload.

## Current claim
This is a bounded product usability slice: a missing local affordance on a high-frequency artifact type, not a broad markdown redesign.
