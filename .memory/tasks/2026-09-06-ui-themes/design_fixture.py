#!/usr/bin/env python3
"""Generate a content-rich synthetic Pi session log for design-review runs.

The Docker iteration harness drops the produced file into the container HOME
at  ~/.pi/agent/sessions/--workspace--/<ts>_<uuid>.jsonl  and resumes it with
codoxear-broker, so the app renders a realistic transcript (markdown, code,
table, blockquote, day separator, token usage) without any live model.

Usage: design_fixture.py <out_dir>   -> prints the written file path
"""
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

CWD = "/workspace"
MODEL_PROVIDER = "dexgem-messages"
MODEL_ID = "claude-fable-5-1"

ASSISTANT_LONG = """Landed the token extraction. Everything visual now flows through custom properties — here's the shape of it.

## What changed

- **Geometry**: the 81 hardcoded `border-radius: 0` literals now read `--radius-control`, `--radius-card`, `--radius-bubble`, and `--radius-pill`.
- **Type**: `body` uses `--font-ui`; markdown headings use `--font-prose`.
- **Icons**: backend logos use `--icon-muted-filter`, so dark modes can invert them.

The core loop stayed small:

```python
def resolve_mode(preference: str, system_dark: bool) -> str:
    if preference in ("light", "dark"):
        return preference
    return "dark" if system_dark else "light"
```

> Every displayed state has one declared authoritative writer. The theme
> controller owns `data-theme`, `data-mode`, the stylesheet link, and the
> custom-CSS style element — nothing else touches them.

| Token | Paper | Clay | Slate |
| --- | --- | --- | --- |
| `--radius-card` | 0 | 12px | 12px |
| `--radius-bubble` | 0 | 18px | 18px |
| `--shadow-pop` | none | soft | soft |

One caveat: the `--icon-muted-filter` change is the only behavioral edit — it
exists purely so dark themes can flip the logo rendering. Paper keeps the
exact previous filter as its token value.
"""

ASSISTANT_SHORT = """Good catch. The dark paper palette keeps the ink-rule identity but drops chrome seams to `--hairline` strength:

```css
:root[data-theme="paper"][data-mode="dark"] .sidebar,
:root[data-theme="paper"][data-mode="dark"] .topbar {
  border-color: var(--hairline);
}
```

1. Cards and bubbles keep full-strength ink perimeters.
2. Chrome (sidebar, topbar, composer) drops to hairline.
3. Focus rings stay `--ink` for maximum visibility.

That keeps the e-ink discipline without the nighttime glare.
"""


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}Z"


def main() -> None:
    out_dir = sys.argv[1]
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)

    t0 = yesterday.replace(hour=16, minute=40, second=0, microsecond=0)
    t1 = t0 + timedelta(seconds=9)
    t2 = now.replace(hour=9, minute=12, second=0, microsecond=0)
    t3 = t2 + timedelta(seconds=11)

    sid = str(uuid.uuid4())
    rows = []

    def rid() -> str:
        return uuid.uuid4().hex[:8]

    rows.append({"type": "session", "version": 3, "id": sid, "timestamp": iso(t0), "cwd": CWD})
    prev = None
    r = rid(); rows.append({"type": "model_change", "id": r, "parentId": prev, "timestamp": iso(t0), "provider": MODEL_PROVIDER, "modelId": MODEL_ID}); prev = r
    r = rid(); rows.append({"type": "thinking_level_change", "id": r, "parentId": prev, "timestamp": iso(t0), "thinkingLevel": "high"}); prev = r

    def user_row(text: str, ts: datetime) -> None:
        nonlocal prev
        r = rid()
        rows.append({"type": "message", "id": r, "parentId": prev, "timestamp": iso(ts),
                     "message": {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": ms(ts)}})
        prev = r

    def assistant_row(text: str, ts: datetime, thinking: str | None, reasoning_tokens: int, total: int) -> None:
        nonlocal prev
        content = []
        if thinking:
            content.append({"type": "thinking", "thinking": thinking})
        content.append({"type": "text", "text": text})
        r = rid()
        rows.append({"type": "message", "id": r, "parentId": prev, "timestamp": iso(ts),
                     "message": {
                         "role": "assistant", "content": content,
                         "api": "anthropic-messages", "provider": MODEL_PROVIDER, "model": MODEL_ID,
                         "usage": {"input": max(total - 9000, 1000), "output": 900, "cacheRead": 8000, "cacheWrite": 0,
                                   "totalTokens": total,
                                   "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                                   "cacheWrite1h": 0, "reasoning": reasoning_tokens},
                         "stopReason": "stop", "timestamp": ms(ts), "responseId": "msg_fixture"}})
        prev = r

    user_row("Can you review the theme token extraction before I land it?", t0)
    assistant_row(ASSISTANT_LONG, t1,
                  "Let me check the radius migration covers every literal and that the icon filter token is applied at both use sites…",
                  1200, 48200)
    user_row("Looks good. One more pass on the dark palette?", t2)
    assistant_row(ASSISTANT_SHORT, t3, None, 300, 61400)

    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{iso(t0).replace(':', '-')}_{sid}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
