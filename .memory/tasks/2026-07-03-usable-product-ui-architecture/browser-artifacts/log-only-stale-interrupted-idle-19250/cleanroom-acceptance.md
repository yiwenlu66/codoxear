ACCEPT — no blockers.

The committed artifact proves the named log-only stale `interrupted_idle` boundary through a real Docker Codoxear server and real `/api/sessions`, with a controlled fake broker only for the necessary experimental condition: broker socket keeps returning `busy:false, interrupted_idle:true`.

Key evidence:
- `drive.sh` authenticates to real server port `19250`, polls real `/api/sessions`, and probes the Unix broker socket directly.
- `unified_stale_broker.py` creates a real sidecar/socket/log under the Docker app dir; product code is not stubbed.
- Phase 2 establishes the decisive condition:
  - initial interrupted log size recomputes to `351`;
  - same log grows to `465`;
  - `phase2-polls.json` records broker `interrupted_idle:true`, `busy:false`;
  - five `/api/sessions` polls all return `busy:true`;
  - `browser-dom.json` and screenshots show sidebar idle/gray before, busy/blue after, and still busy on repoll.
- Phase 3 is consistent: broker false clears suppression; later true re-arms idle; later post-baseline activity suppresses stale true again.
- Raw `.raw.txt` files are invalid JSON because embedded broker JSON was unescaped, but they are preserved as raw text and the normalized `.json` files preserve the same values. Acceptable.
- Hygiene passed: commit is evidence/memory-only, no product code, JSON valid, PNGs valid/small, no ignored runtime paths committed, no staged files.

Non-blocking boundaries:
- Browser replay is less self-contained because `browser_proof2.js` relies on an external marker-driven append step not committed as a separate coordinator; API phase already contains the decisive broker/log/poll proof.
- This closes the `/api/sessions` + sidebar busy projection boundary, not every busy-derived affordance or provider-real interrupt behavior.
