      const $ = (q) => document.querySelector(q);
      function updateAppHeightVar() {
        const vv = window.visualViewport;
        const h = vv ? vv.height : window.innerHeight;
        const top = vv ? vv.offsetTop : 0;
        const hh = Math.round(h);
        const tt = Math.round(top);
        if (updateAppHeightVar._h === hh && updateAppHeightVar._t === tt) return;
        updateAppHeightVar._h = hh;
        updateAppHeightVar._t = tt;
        document.documentElement.style.setProperty("--appH", `${hh}px`);
        document.documentElement.style.setProperty("--vvTop", `${tt}px`);
      }
      updateAppHeightVar();
      window.addEventListener("resize", updateAppHeightVar);
      if (window.visualViewport) {
        window.visualViewport.addEventListener("resize", updateAppHeightVar);
        window.visualViewport.addEventListener("scroll", updateAppHeightVar);
      }
      // Best-effort zoom disable (iOS Safari still has edge cases).
      document.addEventListener(
        "gesturestart",
        (e) => {
          e.preventDefault();
        },
        { passive: false }
      );
      document.addEventListener(
        "gesturechange",
        (e) => {
          e.preventDefault();
        },
        { passive: false }
      );
      document.addEventListener(
        "gestureend",
        (e) => {
          e.preventDefault();
        },
        { passive: false }
      );
      document.addEventListener(
        "touchstart",
        (e) => {
          if (e.touches && e.touches.length > 1) e.preventDefault();
        },
        { passive: false }
      );
      document.addEventListener(
        "touchmove",
        (e) => {
          if (e.touches && e.touches.length > 1) e.preventDefault();
        },
        { passive: false }
      );
      const el = (tag, attrs = {}, children = []) => {
        const n = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs)) {
          if (k === "class") n.className = v;
          else if (k === "text") n.textContent = v;
          else if (k === "html") n.innerHTML = v;
          else n.setAttribute(k, v);
        }
        for (const c of children) n.appendChild(c);
        return n;
      };

      const perfWindow = 200;
      const perfSamples = new Map();

      function pushPerfSample(name, valueMs) {
        if (!(valueMs >= 0)) return;
        const arr = perfSamples.get(name) || [];
        arr.push(valueMs);
        if (arr.length > perfWindow) arr.splice(0, arr.length - perfWindow);
        perfSamples.set(name, arr);
      }

      function perfPercentile(sorted, p) {
        if (!sorted.length) return 0;
        if (sorted.length === 1) return sorted[0];
        const pos = Math.max(0, Math.min(1, p)) * (sorted.length - 1);
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, sorted.length - 1);
        const frac = pos - lo;
        return sorted[lo] * (1 - frac) + sorted[hi] * frac;
      }

      function summarizePerf() {
        const out = {};
        for (const [k, arr] of perfSamples.entries()) {
          if (!arr.length) continue;
          const s = arr.slice().sort((a, b) => a - b);
          out[k] = {
            count: s.length,
            p50_ms: Math.round(perfPercentile(s, 0.5) * 100) / 100,
            p95_ms: Math.round(perfPercentile(s, 0.95) * 100) / 100,
            max_ms: Math.round(s[s.length - 1] * 100) / 100,
            last_ms: Math.round(arr[arr.length - 1] * 100) / 100,
          };
        }
        return out;
      }

      window.codoxearPerf = summarizePerf;

      const appBaseUrl = (() => {
        const here = new URL(window.location.href);
        const p0 = String(here.pathname || "/");
        if (p0.endsWith("/static/index.html")) {
          return new URL(p0.slice(0, -"/static/index.html".length) + "/", here.origin);
        }
        if (p0.endsWith("/static/")) {
          return new URL(p0.slice(0, -"/static/".length) + "/", here.origin);
        }
        return new URL(".", here);
      })();
      function resolveAppUrl(path) {
        const s = String(path ?? "");
        const rel = s.startsWith("/") ? s.slice(1) : s;
        return new URL(rel, appBaseUrl).toString();
      }

      async function api(path, { method = "GET", body } = {}) {
        const t0 = performance.now();
        const opts = { method, headers: {} };
        if (body !== undefined) {
          opts.headers["Content-Type"] = "application/json";
          opts.body = JSON.stringify(body);
        }
        const url = resolveAppUrl(path);
        const res = await fetch(url, opts);
        const txt = await res.text();
        let obj;
        try {
          obj = JSON.parse(txt);
        } catch (e) {
          console.error("api: invalid json response", { path, url, method, txt });
          throw e;
        }
        const dt = performance.now() - t0;
        const rawPath = String(path ?? "");
        if (rawPath === "/api/sessions" && method === "GET") pushPerfSample("api_sessions_ms", dt);
        else if (rawPath.includes("/messages") && method === "GET") {
          if (rawPath.includes("init=1")) pushPerfSample("api_messages_init_ms", dt);
          else pushPerfSample("api_messages_poll_ms", dt);
        }
        if (!res.ok) throw Object.assign(new Error(obj.error || "request failed"), { status: res.status, obj });
        return obj;
      }

      function fmtTs(ts) {
        try {
          const d = new Date(ts * 1000);
          const y = String(d.getFullYear()).padStart(4, "0");
          const m = String(d.getMonth() + 1).padStart(2, "0");
          const day = String(d.getDate()).padStart(2, "0");
          const hh = String(d.getHours()).padStart(2, "0");
          const mm = String(d.getMinutes()).padStart(2, "0");
          return `${y}-${m}-${day} ${hh}:${mm}`;
        } catch {
          return String(ts);
        }
      }

      function baseName(p) {
        if (!p) return "";
        const s = String(p);
        const parts = s.split("/").filter(Boolean);
        return parts.length ? parts[parts.length - 1] : s;
      }

      function shortSessionId(sid) {
        const s = sid == null ? "" : String(sid);
        const m = s.match(/^([0-9a-f]{8})[0-9a-f-]{28}-(\d+)$/i);
        if (m) return `${m[1]}-${m[2]}`;
        return s.slice(0, 8);
      }

      function escapeHtml(s) {
        return String(s)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }

      function safeUrl(u) {
        try {
          const url = new URL(String(u), location.origin);
          if (url.protocol === "http:" || url.protocol === "https:" || url.protocol === "mailto:") return url.href;
        } catch (e) {
          console.error("safeUrl: invalid url", { u, e });
        }
        return null;
      }

      function renderInlineMd(s) {
        const raw = String(s ?? "");
        const re = /`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*/g;
        let out = "";
        let last = 0;
        for (;;) {
          const m = re.exec(raw);
          if (!m) break;
          out += escapeHtml(raw.slice(last, m.index));
          if (m[1] !== undefined) {
            out += `<code>${escapeHtml(m[1])}</code>`;
          } else if (m[2] !== undefined) {
            const href = safeUrl(m[3]);
            if (!href) out += `${escapeHtml(m[2])} (${escapeHtml(m[3])})`;
            else out += `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer noopener">${escapeHtml(m[2])}</a>`;
          } else if (m[4] !== undefined) {
            out += `<strong>${escapeHtml(m[4])}</strong>`;
          } else {
            out += escapeHtml(m[0]);
          }
          last = m.index + m[0].length;
        }
        out += escapeHtml(raw.slice(last));
        return out;
      }

      function mdToHtml(src) {
        const s = String(src ?? "").replaceAll("\r\n", "\n");
        const splitByFences = (input) => {
          const chunks = [];
          const lines = String(input ?? "").split("\n");
          let textLines = [];
          let inFence = false;
          let fenceLang = "";
          let fenceLines = [];
          let fenceStart = "";

          const flushText = () => {
            const v = textLines.join("\n");
            textLines = [];
            if (v.trim()) chunks.push({ type: "text", value: v });
          };
          const flushFence = () => {
            const v = fenceLines.join("\n");
            fenceLines = [];
            chunks.push({ type: "code", lang: fenceLang, value: v });
            fenceLang = "";
            fenceStart = "";
          };

          for (const line of lines) {
            if (!inFence) {
              const m = line.match(/^\s{0,3}```\s*([a-zA-Z0-9_-]+)?\s*$/);
              if (m) {
                flushText();
                inFence = true;
                fenceLang = m[1] || "";
                fenceStart = line;
                fenceLines = [];
                continue;
              }
              textLines.push(line);
              continue;
            }
            if (line.match(/^\s{0,3}```\s*$/)) {
              inFence = false;
              flushFence();
              continue;
            }
            fenceLines.push(line);
          }

          if (inFence) {
            // Preserve prior behavior: an unclosed fence is not treated as code.
            textLines.push(fenceStart);
            for (const x of fenceLines) textLines.push(x);
          }
          flushText();
          return chunks;
        };

        const listItemInfo = (line) => {
          const l = String(line ?? "");
          let indent = 0;
          while (indent < l.length && l[indent] === " ") indent += 1;
          const t = l.trim();
          if (t.startsWith("- ") || t.startsWith("* ") || t.startsWith("\u2022 ")) {
            return { type: "ul", indent, text: t.slice(2).trimStart() };
          }
          const mOl = t.match(/^(\d+)\.\s+(.*)$/);
          if (mOl) return { type: "ol", indent, text: (mOl[2] || "").trimStart() };
          return null;
        };

        const parseList = (lines, start) => {
          const head = listItemInfo(lines[start]);
          if (!head) throw new Error("parseList called on non-list line");
          const baseIndent = head.indent;
          const listType = head.type;
          const items = [];

          let i = start;
          while (i < lines.length) {
            const info = listItemInfo(lines[i]);
            if (!info) break;
            if (info.indent < baseIndent) break;
            if (info.indent > baseIndent) {
              if (!items.length) break;
              const child = parseList(lines, i);
              items[items.length - 1].child = child.node;
              i = child.next;
              continue;
            }
            if (info.type !== listType) break;
            items.push({ text: info.text, child: null });
            i += 1;
          }
          return { node: { type: listType, items }, next: i };
        };

        const renderList = (node) => {
          const out = [];
          out.push(node.type === "ol" ? "<ol>" : "<ul>");
          for (const it of node.items) {
            out.push("<li>");
            out.push(renderInlineMd(it.text || ""));
            if (it.child) out.push(renderList(it.child));
            out.push("</li>");
          }
          out.push(node.type === "ol" ? "</ol>" : "</ul>");
          return out.join("");
        };

        const chunks = splitByFences(s);

        const out = [];
        for (const c of chunks) {
          if (c.type === "code") {
            const langAttr = c.lang ? ` data-lang="${escapeHtml(c.lang)}"` : "";
            out.push(`<pre><code${langAttr}>${escapeHtml(c.value)}</code></pre>`);
            continue;
          }
          const blocks = c.value.split(/\n{2,}/);
          for (const block of blocks) {
            const lines = block.split("\n").map((x) => x.trimEnd());
            if (!lines.length) continue;

            const head = lines[0] || "";
            const mHeading = head.match(/^(#{1,6})\s+(.*)$/);
            let startIdx = 0;
            if (mHeading) {
              const level = mHeading[1].length;
              out.push(`<h${level}>${renderInlineMd(mHeading[2])}</h${level}>`);
              startIdx = 1;
            }

            let paraLines = [];
            const flushPara = () => {
              const para = paraLines.join("\n").trim();
              paraLines = [];
              if (!para) return;
              out.push(`<p>${renderInlineMd(para).replaceAll("\n", "<br />")}</p>`);
            };

            for (let i = startIdx; i < lines.length; i++) {
              const l = lines[i] || "";
              const t = l.trim();
              if (!t) {
                flushPara();
                continue;
              }
              const info = listItemInfo(l);
              if (info) {
                flushPara();
                const parsed = parseList(lines, i);
                out.push(renderList(parsed.node));
                i = parsed.next - 1;
                continue;
              }
              paraLines.push(l);
            }
            flushPara();
          }
        }
        return out.join("");
      }

      const mdCache = new Map();
      function mdToHtmlCached(src) {
        const key = String(src ?? "");
        const hit = mdCache.get(key);
        if (hit !== undefined) return hit;
        const html = mdToHtml(key);
        mdCache.set(key, html);
        if (mdCache.size > 1200) {
          // Prevent unbounded growth; chat history is expected to be small.
          mdCache.clear();
        }
        return html;
      }

	      function iconSvg(name) {
	        if (name === "menu")
	          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>`;
	        if (name === "refresh")
	          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v6h-6"/></svg>`;
	        if (name === "harness")
	          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h3l2-4 3 8 2-4h6"/><path d="M12 21a9 9 0 1 0-9-9"/></svg>`;
	        if (name === "stop")
	          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>`;
	        if (name === "plus")
	          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>`;
	        if (name === "logout")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 17l5-5-5-5"/><path d="M15 12H3"/><path d="M21 3v18"/></svg>`;
        if (name === "send")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>`;
        if (name === "paperclip")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-8.49 8.49a5 5 0 0 1-7.07-7.07l9.19-9.19a3.5 3.5 0 0 1 4.95 4.95l-9.19 9.19a2 2 0 0 1-2.83-2.83l8.49-8.49"/></svg>`;
        if (name === "down")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/></svg>`;
        if (name === "trash")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M6 6l1 16h10l1-16"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>`;
        return "";
      }

      function renderLogin(onAuthed) {
        const root = $("#root");
        root.innerHTML = "";
        const err = el("div", { class: "err" });
        const wrap = el("div", { class: "loginWrap" });
        const box = el("div", { class: "login" }, [
          el("h1", { text: "Codoxear login" }),
          el("div", { class: "row2" }, [
            el("input", { type: "password", id: "pw", placeholder: "Password" }),
            el("button", { class: "primary", id: "loginBtn", text: "Login" }),
            err,
          ]),
        ]);
        wrap.appendChild(box);
        root.appendChild(wrap);
        $("#loginBtn").onclick = async () => {
          err.textContent = "";
          const pw = $("#pw").value;
          try {
            await api("/api/login", { method: "POST", body: { password: pw } });
            onAuthed();
          } catch (e) {
            err.textContent = e.obj?.error || e.message;
          }
        };
      }

	      function renderApp() {
	        const root = $("#root");
	        root.innerHTML = "";

	        const backdrop = el("div", { class: "backdrop", id: "backdrop" });
	        const app = el("div", { class: "app" });
        const sidebar = el("div", { class: "sidebar" });
        const sessionsWrap = el("div", { class: "sessions" });
        const sidebarFooter = el("footer", {}, [
          el("button", { id: "logoutBtnSide", type: "button", title: "Log out", "aria-label": "Log out", text: "Log out" }),
        ]);
        const main = el("div", { class: "main" });
        const chatWrap = el("div", { class: "chatWrap", id: "chatWrap" });
        const chat = el("div", { class: "chat", id: "chat" });
        const chatInner = el("div", { class: "chatInner", id: "chatInner" });
        const olderWrap = el("div", { class: "olderWrap", id: "olderWrap" });
        const olderBtn = el("button", {
          class: "olderBtn",
          id: "olderBtn",
          type: "button",
          text: "Load older messages",
        });
        olderWrap.appendChild(olderBtn);
        const bottomSentinel = el("div", { id: "bottomSentinel" });
        const jumpBtn = el("button", {
          class: "jumpBtn",
          id: "jumpBtn",
          title: "Jump to latest",
          "aria-label": "Jump to latest",
          html: iconSvg("down"),
        });
        chatInner.appendChild(olderWrap);
        chatInner.appendChild(bottomSentinel);
        chat.appendChild(chatInner);
        chatWrap.appendChild(chat);
        chatWrap.appendChild(jumpBtn);
        const composer = el("div", { class: "composer" });

	        let selected = null;
        let offset = 0;
        const INIT_PAGE_LIMIT = 80;
        const CHAT_DOM_WINDOW = 260;
        let activeLogPath = null;
        let activeThreadId = null;
        let olderBefore = 0;
        let hasOlder = false;
        let loadingOlder = false;
        let olderAutoTriggerAt = 0;
        const OLDER_AUTO_COOLDOWN_MS = 450;
        let pollTimer = null;
        let pollGen = 0;
        let pollLoopBusy = false;
        let pollKickPending = false;
	        let pollFastUntilMs = 0;
	        let turnOpen = false;
	        let sessionsTimer = null;
		        let sessionIndex = new Map(); // session_id -> session info
		        let sending = false;
			        let localEchoSeq = 0;
			        const pendingUser = [];
			        let attachedImages = 0;
		        let autoScroll = true;
			        let backfillToken = 0;
		        let backfillState = null;
			        let lastScrollTop = 0;
				        let lastToken = null;
				        let typingRow = null;
				        let attachBadgeEl = null;
                let clickLoadT0 = 0;
                let clickMetricPending = false;
					        let harnessMenuOpen = false;
					        let harnessCfg = { enabled: false, request: "" };
					        let harnessSaveTimer = null;

				        const titleLabel = el("div", { id: "threadTitle", text: "No session selected" });
				        const statusChip = el("span", { class: "status-chip", id: "statusChip", text: "Idle" });
				        const ctxChip = el("span", { class: "status-chip", id: "ctxChip", text: "" });
		        ctxChip.style.display = "none";
		        const interruptBtn = el("button", {
		          id: "interruptBtn",
		          class: "icon-btn",
		          title: "Interrupt (Esc)",
		          "aria-label": "Interrupt (Esc)",
		          type: "button",
		          html: iconSvg("stop"),
		        });
		        interruptBtn.style.display = "none";
		        const toast = el("div", { class: "muted toast", id: "toast" });
			        const toggleSidebarBtn = el("button", {
	          id: "toggleSidebarBtn",
	          class: "icon-btn",
	          title: "Toggle sidebar",
	          "aria-label": "Toggle sidebar",
	          html: iconSvg("menu"),
	        });
			        const harnessBtn = el("button", {
			          id: "harnessBtn",
			          class: "icon-btn",
			          title: "Harness mode",
			          "aria-label": "Harness mode",
				          type: "button",
				          html: iconSvg("harness"),
				        });
				        harnessBtn.disabled = true;
				        harnessBtn.classList.toggle("active", false);
				        const harnessMenu = el("div", { id: "harnessMenu", class: "harnessMenu", role: "dialog", "aria-label": "Harness mode settings" }, [
				          el("div", { class: "row" }, [
				            el("label", {}, [
				              el("input", { type: "checkbox", id: "harnessEnabled" }),
			              el("span", { text: "Harness mode" }),
			            ]),
			          ]),
			          el("div", { class: "label", text: "Additional request to append (optional; per session)" }),
			          el("textarea", { id: "harnessRequest", "aria-label": "Additional request for harness prompt" }),
			        ]);

			        const topbar = el("div", { class: "topbar" }, [
			          el("div", { class: "pill" }, [toggleSidebarBtn, el("div", {}, [titleLabel, toast])]),
			          el("div", { class: "actions" }, [
			            statusChip,
			            ctxChip,
			            interruptBtn,
			            harnessBtn,
			          ]),
			        ]);

	        const form = el("form", {}, [
	          el("button", {
	            class: "icon-btn",
	            id: "attachBtn",
	            type: "button",
	            title: "Attach image",
	            "aria-label": "Attach image",
	            html: iconSvg("paperclip"),
	          }),
          el("div", { class: "inputWrap" }, [
            el("textarea", { id: "msg", placeholder: "", "aria-label": "Enter your instructions here" }),
            el("div", { class: "ph", id: "msgPh", text: "Enter your instructions here" }),
          ]),
	          el("input", { id: "imgInput", type: "file", accept: "image/*", style: "display:none" }),
	          el("button", { class: "icon-btn primary", id: "sendBtn", type: "submit", title: "Send", "aria-label": "Send", html: iconSvg("send") }),
	        ]);
	        composer.appendChild(form);

        sidebar.appendChild(
          el("header", {}, [
            el("div", { class: "title", html: `<img class="sidebarLogo" src="/static/codoxear-icon.png" alt="" />Codoxear` }),
            el("div", { class: "actions" }, [
              el("button", { id: "newBtn", class: "icon-btn", title: "New session", "aria-label": "New session", html: iconSvg("plus") }),
              el("button", { id: "refreshBtn", class: "icon-btn", title: "Refresh", "aria-label": "Refresh", html: iconSvg("refresh") }),
            ]),
          ])
        );
        sidebar.appendChild(sessionsWrap);
        sidebar.appendChild(sidebarFooter);
	        main.appendChild(topbar);
	        main.appendChild(chatWrap);
	        main.appendChild(composer);
		        app.appendChild(sidebar);
		        app.appendChild(main);
		        app.appendChild(backdrop);
		        root.appendChild(app);
		        root.appendChild(harnessMenu);

        function setToast(text) {
          toast.textContent = text || "";
          if (!text) return;
          setTimeout(() => {
            if (toast.textContent === text) toast.textContent = "";
          }, 2200);
        }

			        function setStatus({ running, queueLen }) {
			          const q = Number(queueLen || 0);
			          const mobile = isMobile();
			          if (running) {
			            statusChip.style.display = "none";
			            statusChip.classList.remove("running");
			          } else {
			            statusChip.style.display = "inline-flex";
			            if (q) statusChip.textContent = mobile ? `Q ${q}` : `Queue ${q}`;
			            else statusChip.textContent = "Idle";
			          }
			          interruptBtn.style.display = running && selected ? "inline-flex" : "none";
			          interruptBtn.disabled = !(running && selected);
			        }

	        function setContext(tok) {
	          if (!tok || typeof tok !== "object") {
	            lastToken = null;
	            ctxChip.style.display = "none";
	            ctxChip.textContent = "";
	            ctxChip.title = "";
	            return;
	          }
	          const ctx = Number(tok.context_window);
	          const used = Number(tok.tokens_in_context);
	          const pct = Number(tok.percent_remaining);
	          if (!Number.isFinite(ctx) || !Number.isFinite(used) || ctx <= 0 || used < 0) {
	            lastToken = null;
	            ctxChip.style.display = "none";
	            return;
	          }
	          const p = Number.isFinite(pct) ? Math.max(0, Math.min(100, Math.round(pct))) : null;
	          lastToken = { ctx, used, pct: p, remaining: Math.max(ctx - used, 0), baseline: Number(tok.baseline_tokens) || 0, asOf: tok.as_of || "" };
	          ctxChip.style.display = "inline-flex";
	          ctxChip.textContent = p === null ? "Ctx" : `Ctx ${p}%`;
	          ctxChip.title = `Context: ${used}/${ctx} tokens (baseline ${lastToken.baseline}).`;
	        }
	        ctxChip.onclick = () => {
	          if (!lastToken) return;
	          setToast(`ctx ${lastToken.used}/${lastToken.ctx} (${lastToken.pct ?? "?"}% left)`);
	        };

		        function resetChatRenderState() {
		          autoScroll = true;
              pendingUser.length = 0;
              sending = false;
              localEchoSeq = 0;
              olderBefore = 0;
              hasOlder = false;
              loadingOlder = false;
              olderAutoTriggerAt = 0;
              clickMetricPending = false;
		          chatInner.innerHTML = "";
	          chatInner.appendChild(olderWrap);
	          chatInner.appendChild(bottomSentinel);
              setOlderState({ hasMore: false, isLoading: false });
	          typingRow = null;
	          jumpBtn.style.display = "none";
	          backfillState = null;
	          backfillToken += 1;
	          lastScrollTop = 0;
	          chat.scrollTop = 0;
	        }

          function setOlderState({ hasMore, isLoading }) {
            hasOlder = Boolean(hasMore);
            loadingOlder = Boolean(isLoading);
            olderWrap.style.display = hasOlder ? "flex" : "none";
            olderBtn.disabled = loadingOlder;
            olderBtn.textContent = loadingOlder ? "Loading..." : "Load older messages";
          }

          function markClickFirstPaint() {
            if (!clickMetricPending) return;
            clickMetricPending = false;
            const dt = performance.now() - clickLoadT0;
            pushPerfSample("click_to_first_message_ms", dt);
          }

	        function ensureTypingRow() {
	          if (typingRow && typingRow.isConnected) return typingRow;
	          const row = el("div", { class: "msg-row assistant typing-row" });
	          row.dataset.role = "assistant";
	          const bubble = el("div", { class: "msg assistant typing" });
	          const dots = el("div", { class: "typingDots", "aria-label": "Running", title: "Running" }, [
	            el("span", { class: "typingDot" }),
	            el("span", { class: "typingDot" }),
	            el("span", { class: "typingDot" }),
	          ]);
	          bubble.appendChild(dots);
	          row.appendChild(bubble);
	          typingRow = row;
	          return row;
	        }

	        function setTyping(show) {
	          if (!show) {
	            if (typingRow && typingRow.isConnected) typingRow.remove();
	            return;
	          }
	          const row = ensureTypingRow();
	          if (!row.isConnected) {
	            chatInner.insertBefore(row, bottomSentinel);
	          } else if (row.nextSibling !== bottomSentinel) {
	            chatInner.insertBefore(row, bottomSentinel);
	          }
	          if (autoScroll) requestAnimationFrame(() => scrollToBottom());
	        }

        function isNearBottom() {
          const thresholdPx = 80;
          return chat.scrollHeight - (chat.scrollTop + chat.clientHeight) <= thresholdPx;
        }

        function scrollToBottom() {
          bottomSentinel.scrollIntoView({ block: "end" });
          lastScrollTop = chat.scrollTop;
        }

        function ymd(d) {
          const y = String(d.getFullYear()).padStart(4, "0");
          const m = String(d.getMonth() + 1).padStart(2, "0");
          const day = String(d.getDate()).padStart(2, "0");
          return `${y}-${m}-${day}`;
        }

        function dayLabel(d) {
          const today = new Date();
          const a = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
          const b = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
          const diffDays = Math.round((a - b) / 86400000);
          const date = ymd(d);
          if (diffDays === 0) return `Today (${date})`;
          if (diffDays === 1) return `Yesterday (${date})`;
          return date;
        }

        function time24(d) {
          const hh = String(d.getHours()).padStart(2, "0");
          const mm = String(d.getMinutes()).padStart(2, "0");
          return `${hh}:${mm}`;
        }

        function rebuildDecorations({ preserveScroll }) {
          const oldTop = chat.scrollTop;
          const oldH = chat.scrollHeight;

          for (const n of Array.from(chatInner.querySelectorAll(".day-sep"))) n.remove();

          const rows = Array.from(chatInner.querySelectorAll(".msg-row"));
          let prevRole = null;
          let prevDay = null;
          let lastDay = null;

          for (const row of rows) {
            const role = row.classList.contains("user") ? "user" : "assistant";
            const ts = Number(row.dataset.ts || "0");
            const day = ts ? ymd(new Date(ts * 1000)) : null;

            row.classList.remove("grouped");
            if (prevRole === role && prevDay && day && prevDay === day) row.classList.add("grouped");
            prevRole = role;
            prevDay = day;

            if (day && day !== lastDay) {
              const d = new Date(ts * 1000);
              const sep = el("div", { class: "day-sep", text: dayLabel(d) });
              sep.dataset.day = day;
              chatInner.insertBefore(sep, row);
              lastDay = day;
            }
          }

          if (preserveScroll) {
            chat.scrollTop = oldTop + (chat.scrollHeight - oldH);
          }
          if (autoScroll) {
            requestAnimationFrame(() => scrollToBottom());
            jumpBtn.style.display = "none";
          } else {
            jumpBtn.style.display = "inline-flex";
          }
        }

        function trimRenderedRows({ fromTop }) {
          const rows = Array.from(chatInner.querySelectorAll(".msg-row")).filter((x) => !x.classList.contains("typing-row"));
          if (rows.length <= CHAT_DOM_WINDOW) return;
          const extra = rows.length - CHAT_DOM_WINDOW;
          if (fromTop) {
            for (const row of rows.slice(0, extra)) row.remove();
          } else {
            for (const row of rows.slice(rows.length - extra)) row.remove();
          }
        }

        function makeRow(ev, { ts, pending }) {
          const role = ev.role === "user" ? "user" : "assistant";
          const row = el("div", { class: `msg-row ${role}` });
          row.dataset.role = role;
          if (typeof ts === "number" && Number.isFinite(ts)) row.dataset.ts = String(ts);

          const bubble = el("div", { class: role === "user" ? "msg user" : "msg assistant" });
          bubble.appendChild(el("div", { class: "md", html: mdToHtmlCached(ev.text) }));
          if (typeof ts === "number" && Number.isFinite(ts)) bubble.appendChild(el("div", { class: "ts", text: time24(new Date(ts * 1000)) }));

          if (pending) {
            bubble.style.opacity = "0.72";
            bubble.setAttribute("data-pending", "1");
            if (ev.localId) bubble.setAttribute("data-local-id", String(ev.localId));
          }

          row.appendChild(bubble);
          return { row, bubble };
        }

        function normalizeTextForPendingMatch(s) {
          // Normalize common platform newline differences to improve pending->ack reconciliation.
          return String(s || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
        }

        function pendingMatchKey(s) {
          // Codex log serialization can trim trailing whitespace/newlines; match on a slightly
          // normalized key to avoid duplicating the optimistic local echo bubble.
          const t = normalizeTextForPendingMatch(s);
          return t.replace(/[ \t]+$/gm, "").replace(/\s+$/, "");
        }

        function consumePendingUserIfMatches(ev) {
          if (ev.role !== "user" || ev.pending) return false;
          const key = pendingMatchKey(ev.text);
          const loose = normalizeTextForPendingMatch(ev.text);
          const evTs = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
          const candidates = [];
          for (let i = 0; i < pendingUser.length; i++) {
            const x = pendingUser[i];
            if (x.key === key || x.loose === loose) candidates.push({ i, x });
          }
          if (!candidates.length) return false;
          let best = candidates[0];
          if (evTs !== null) {
            let bestD = Math.abs(evTs - (best.x.t0 || evTs));
            for (const c of candidates.slice(1)) {
              const d = Math.abs(evTs - (c.x.t0 || evTs));
              if (d < bestD) {
                best = c;
                bestD = d;
              }
            }
          }
          const idx = best.i;
          if (idx < 0) return false;
          const { id } = pendingUser[idx];
          pendingUser.splice(idx, 1);
          const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${id}"]`);
          if (!pendingEl) return true;

          pendingEl.style.opacity = "1";
          pendingEl.removeAttribute("data-local-id");
          pendingEl.removeAttribute("data-pending");

          const mdEl = pendingEl.querySelector(".md");
          if (mdEl && typeof ev.text === "string") mdEl.innerHTML = mdToHtmlCached(ev.text);

          const row = pendingEl.closest(".msg-row");
          if (row && typeof ev.ts === "number" && Number.isFinite(ev.ts)) row.dataset.ts = String(ev.ts);
          const tsEl = pendingEl.querySelector(".ts");
          if (tsEl && typeof ev.ts === "number" && Number.isFinite(ev.ts)) tsEl.textContent = time24(new Date(ev.ts * 1000));
          rebuildDecorations({ preserveScroll: true });
          return true;
        }

        function isMobile() {
          return window.matchMedia && window.matchMedia("(max-width: 880px)").matches;
        }

        function setSidebarOpen(open) {
          if (open) {
            document.body.classList.add("sidebar-open");
            localStorage.setItem("codexweb.sidebarOpen", "1");
          } else {
            document.body.classList.remove("sidebar-open");
            localStorage.removeItem("codexweb.sidebarOpen");
          }
        }

        function setSidebarCollapsed(collapsed) {
          if (collapsed) {
            document.body.classList.add("sidebar-collapsed");
            localStorage.setItem("codexweb.sidebarCollapsed", "1");
          } else {
            document.body.classList.remove("sidebar-collapsed");
            localStorage.removeItem("codexweb.sidebarCollapsed");
          }
        }

	        async function refreshSessions() {
	          const data = await api("/api/sessions");
	          sessionsWrap.innerHTML = "";
	          sessionIndex = new Map();
	          const sessions = (data.sessions || [])
              .slice()
              .sort((a, b) => (b.updated_ts || b.start_ts || 0) - (a.updated_ts || a.start_ts || 0));
		          for (const s of sessions) {
		            sessionIndex.set(s.session_id, s);
		            const badge = el("span", { class: "badge" + (s.busy ? " busy" : ""), text: s.busy ? "busy" : "idle" });
		            const q = s.queue_len ? el("span", { class: "badge queue", text: `queue ${s.queue_len}` }) : null;
		            const card = el("div", { class: "session" + (selected === s.session_id ? " active" : "") });

		            const title = baseName(s.cwd) || s.session_id.slice(0, 12);
		            const badges = [];
		            if (s.harness_enabled) badges.push(el("span", { class: "badge harness", text: "harness", title: "Harness mode enabled" }));
		            badges.push(badge);
		            if (q) badges.push(q);
		            let delBtn = null;
	            if (s.owned) {
	              delBtn = el("button", {
	                class: "icon-btn danger sessionDel",
	                title: "Delete session",
	                "aria-label": "Delete session",
	                type: "button",
	                html: iconSvg("trash"),
	              });
              delBtn.onclick = async (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (!confirm("Delete this session?")) return;
                try {
	                  await api(`/api/sessions/${s.session_id}/delete`, { method: "POST", body: {} });
		                  if (selected === s.session_id) {
		                    selected = null;
	                      offset = 0;
	                      activeLogPath = null;
	                      activeThreadId = null;
	                      turnOpen = false;
		                    localStorage.removeItem("codexweb.selected");
		                    titleLabel.textContent = "No session selected";
	                      setStatus({ running: false, queueLen: 0 });
	                      setContext(null);
	                      setTyping(false);
	                      setAttachCount(0);
		                    resetChatRenderState();
	                      if (harnessMenuOpen) hideHarnessMenu();
	                      updateHarnessBtnState();
		                  }
	                  await refreshSessions();
	                } catch (err) {
                  setToast(`delete error: ${err.message}`);
	                }
	              };
	            }
		            const top = el("div", { class: "row" }, [
		              el("div", { class: "titleLine", text: title, title: s.cwd || "" }),
		              el("div", {}, badges),
		            ]);
	            const pid = s.pid ? String(s.pid) : "?";
              const updatedTs = typeof s.updated_ts === "number" && Number.isFinite(s.updated_ts) ? s.updated_ts : s.start_ts;
	            const meta = el("div", { class: "muted subLine", text: `id ${shortSessionId(s.session_id)}  pid ${pid}  last ${fmtTs(updatedTs)}` });
              const mainCol = el("div", { class: "sessionMain" }, [top, meta]);
	            card.appendChild(mainCol);
              if (delBtn) card.appendChild(el("div", { class: "sessionAction" }, [delBtn]));
	            card.onclick = () => {
	              if (isMobile()) setSidebarOpen(false);
	              selectSession(s.session_id);
	            };
	            sessionsWrap.appendChild(card);
	          }
	          if (selected && !sessionIndex.has(selected)) {
	            selected = null;
	            offset = 0;
	            activeLogPath = null;
	            activeThreadId = null;
            pollGen += 1;
            if (pollTimer) clearTimeout(pollTimer);
            pollTimer = null;
            pollKickPending = false;
            localStorage.removeItem("codexweb.selected");
	            titleLabel.textContent = "No session selected";
	            setStatus({ running: false, queueLen: 0 });
	            setTyping(false);
	            resetChatRenderState();
	            turnOpen = false;
	            if (harnessMenuOpen) hideHarnessMenu();
	            updateHarnessBtnState();
	          } else if (selected) {
	            const s = sessionIndex.get(selected);
	            if (s) titleLabel.textContent = `${baseName(s.cwd) || s.session_id} (${String(s.session_id).slice(0, 8)})`;
	          }
	          updateHarnessBtnState();
	          return sessions;
	        }

	        function appendEvent(ev) {
	          if (consumePendingUserIfMatches(ev)) return;

	          const stick = autoScroll || isNearBottom();
	          const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : ev.pending ? Date.now() / 1000 : null;
			          const { row } = makeRow(ev, { ts, pending: Boolean(ev.pending) });
			          const anchor = typingRow && typingRow.isConnected ? typingRow : bottomSentinel;
			          chatInner.insertBefore(row, anchor);
            trimRenderedRows({ fromTop: true });
	          rebuildDecorations({ preserveScroll: false });
            if (!ev.pending) markClickFirstPaint();

          if (stick) {
            requestAnimationFrame(() => scrollToBottom());
            jumpBtn.style.display = "none";
          } else {
            jumpBtn.style.display = "inline-flex";
          }
        }

        function prependOlderEvents(allEvents) {
          const msgs = [];
          for (const ev of allEvents) {
            if (!ev || (ev.role !== "user" && ev.role !== "assistant")) continue;
            msgs.push(ev);
          }
          if (!msgs.length) return;
          const oldTop = chat.scrollTop;
          const oldH = chat.scrollHeight;
          const frag = document.createDocumentFragment();
          for (const ev of msgs) {
            const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
            frag.appendChild(makeRow(ev, { ts, pending: false }).row);
          }
          const firstMsg = chatInner.querySelector(".msg-row:not(.typing-row)");
          const anchor = firstMsg || (typingRow && typingRow.isConnected ? typingRow : bottomSentinel);
          chatInner.insertBefore(frag, anchor);
          trimRenderedRows({ fromTop: false });
          rebuildDecorations({ preserveScroll: true });
          chat.scrollTop = oldTop + (chat.scrollHeight - oldH);
        }

        async function loadOlderMessages({ auto = false } = {}) {
          if (!selected || !hasOlder || loadingOlder) return;
          if (auto) {
            const now = performance.now();
            if (now - olderAutoTriggerAt < OLDER_AUTO_COOLDOWN_MS) return;
            olderAutoTriggerAt = now;
          }
          const sid = selected;
          const gen = pollGen;
          setOlderState({ hasMore: hasOlder, isLoading: true });
          try {
            const reqBefore = Math.max(0, Number(olderBefore) || 0);
            const data = await api(`/api/sessions/${sid}/messages?offset=0&init=1&limit=${INIT_PAGE_LIMIT}&before=${reqBefore}`);
            if (selected !== sid || pollGen !== gen) return;
            const evs = Array.isArray(data.events) ? data.events : [];
            if (evs.length) prependOlderEvents(evs);
            olderBefore = Number.isFinite(Number(data.next_before)) ? Number(data.next_before) : reqBefore;
            setOlderState({ hasMore: Boolean(data.has_older), isLoading: false });
          } catch {
            if (selected !== sid || pollGen !== gen) return;
            setOlderState({ hasMore: hasOlder, isLoading: false });
          }
        }

        function maybeAutoLoadOlder() {
          if (chat.scrollTop > 1) return;
          void loadOlderMessages({ auto: true });
        }

		        function startInitialRender(allEvents) {
		          backfillToken += 1;
		          const myToken = backfillToken;

		          const msgs = [];
		          for (const ev of allEvents) {
		            if (!ev || (ev.role !== "user" && ev.role !== "assistant")) continue;
		            if (consumePendingUserIfMatches(ev)) continue;
		            msgs.push(ev);
		          }
		          if (!msgs.length) return;
			          const frag = document.createDocumentFragment();
		          for (const ev of msgs) {
		            const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
		            frag.appendChild(makeRow(ev, { ts, pending: false }).row);
	          }
	          const anchor = typingRow && typingRow.isConnected ? typingRow : bottomSentinel;
	          chatInner.insertBefore(frag, anchor);
            trimRenderedRows({ fromTop: true });
	          rebuildDecorations({ preserveScroll: false });
            markClickFirstPaint();
	          // Ensure scroll-to-bottom happens after layout.
	          requestAnimationFrame(() => {
	            if (myToken !== backfillToken) return;
	            scrollToBottom();
	            requestAnimationFrame(() => {
	              if (myToken !== backfillToken) return;
	              scrollToBottom();
	            });
	          });
	          backfillState = null;
	        }

			        async function pollMessages(sid = selected, gen = pollGen) {
			          if (!sid) return;
			          try {
	            const prevOffset = offset;
	            const reqOffset = offset;
		            const data = await api(`/api/sessions/${sid}/messages?offset=${reqOffset}`);
	            if (gen !== pollGen || sid !== selected) return;
	            const lp = data && typeof data.log_path === "string" ? data.log_path : null;
	            const tid = data && typeof data.thread_id === "string" ? data.thread_id : null;
	            const nowBusy = Boolean(data.busy);
	            if (!activeLogPath && lp) activeLogPath = lp;
	            if (activeLogPath && !lp) {
	              activeLogPath = null;
	              activeThreadId = tid;
	              offset = 0;
	              resetChatRenderState();
	              setAttachCount(0);
	              setTyping(false);
	              turnOpen = false;
	              setOlderState({ hasMore: false, isLoading: false });
	              olderBefore = 0;
	              setStatus({ running: Boolean(nowBusy), queueLen: data.queue_len });
	              setContext(data.token);
	              setTyping(Boolean(nowBusy));
	              return;
	            }
	            if (activeLogPath && lp && lp !== activeLogPath) {
	              activeLogPath = lp;
	              activeThreadId = tid;
	              offset = 0;
	              resetChatRenderState();
              setAttachCount(0);
              setTyping(false);
              turnOpen = false;
              setStatus({ running: false, queueLen: 0 });
              try {
                const d2 = await api(`/api/sessions/${sid}/messages?offset=0&init=1&limit=${INIT_PAGE_LIMIT}&before=0`);
                if (gen !== pollGen || sid !== selected) return;
                if (d2 && typeof d2.log_path === "string") activeLogPath = d2.log_path;
                if (d2 && typeof d2.thread_id === "string") activeThreadId = d2.thread_id;
                offset = d2.offset;
                const evs2 = Array.isArray(d2.events) ? d2.events : [];
                if (evs2.length) startInitialRender(evs2);
                olderBefore = Number.isFinite(Number(d2.next_before)) ? Number(d2.next_before) : 0;
                setOlderState({ hasMore: Boolean(d2.has_older), isLoading: false });
	                const nowBusy2 = Boolean(d2.busy);
	                const turnStart2 = Boolean(d2.turn_start);
	                const turnEnd2 = Boolean(d2.turn_end);
	                const turnAborted2 = Boolean(d2.turn_aborted);
	                if (turnStart2 || nowBusy2) turnOpen = true;
	                if (turnEnd2 || turnAborted2 || !nowBusy2) turnOpen = false;
                setStatus({ running: Boolean(turnOpen || nowBusy2), queueLen: d2.queue_len });
                setContext(d2.token);
                setTyping(Boolean(turnOpen || nowBusy2));
                } catch (e2) {
                  console.error("poll init reload failed", e2);
                  throw e2;
                }
                return;
             }
	
		            offset = data.offset;
	            const evs = Array.isArray(data.events) ? data.events : [];
	            if (prevOffset === 0 && !chatInner.querySelector(".msg-row:not(.typing-row)") && evs.length) {
	              startInitialRender(evs);
            } else {
              for (const ev of evs) appendEvent(ev);
            }

            const turnStart = Boolean(data.turn_start);
            const turnEnd = Boolean(data.turn_end);
            const turnAborted = Boolean(data.turn_aborted);
            if (turnStart) {
              turnOpen = true;
            }
            if (!turnOpen && nowBusy) {
              turnOpen = true;
            }

            if ((turnEnd || turnAborted) && turnOpen) {
              turnOpen = false;
            }
		            if (turnOpen && !nowBusy) {
		              turnOpen = false;
		            }

				            setStatus({ running: Boolean(turnOpen || nowBusy), queueLen: data.queue_len });
				            setContext(data.token);
				            setTyping(Boolean(turnOpen || nowBusy));
	            const s = sessionIndex.get(sid);
	            if (s) titleLabel.textContent = `${baseName(s.cwd) || s.session_id} (${s.session_id.slice(0, 8)})`;
		          } catch (e) {
            if (gen !== pollGen || sid !== selected) return;
            if (e && e.status === 404) {
              selected = null;
              offset = 0;
              activeLogPath = null;
              activeThreadId = null;
              pollGen += 1;
              if (pollTimer) clearTimeout(pollTimer);
              pollTimer = null;
              pollKickPending = false;
              turnOpen = false;
		              localStorage.removeItem("codexweb.selected");
		              titleLabel.textContent = "No session selected";
		              setStatus({ running: false, queueLen: 0 });
		              setTyping(false);
		              resetChatRenderState();
                try {
                  await refreshSessions();
                } catch (e2) {
                  console.error("refreshSessions failed after session disappeared", e2);
                  toast.textContent = `refresh error: ${e2 && e2.message ? e2.message : "unknown error"}`;
                }
                return;
            }
            toast.textContent = `error: ${e.message}`;
          }
        }

        async function pollLoop() {
          if (!selected) return;
          if (pollLoopBusy) {
            pollKickPending = true;
            return;
          }
          pollLoopBusy = true;
          const mySid = selected;
          const myGen = pollGen;
          try {
            await pollMessages(mySid, myGen);
          } finally {
            pollLoopBusy = false;
          }
          if (pollKickPending) {
            pollKickPending = false;
            kickPoll(0);
            return;
          }
          if (selected !== mySid || pollGen !== myGen) return;
          const now = Date.now();
          let nextMs = 900;
          if (now < pollFastUntilMs) nextMs = 200;
          else if (turnOpen) nextMs = 250;
          pollTimer = setTimeout(pollLoop, nextMs);
        }

        function kickPoll(ms = 0) {
          if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
          }
          if (pollLoopBusy) {
            pollKickPending = true;
            return;
          }
          pollTimer = setTimeout(pollLoop, ms);
        }

		        async function selectSession(id) {
	          pollGen += 1;
	          const myGen = pollGen;
	          if (pollTimer) {
	            clearTimeout(pollTimer);
	            pollTimer = null;
	          }
		          pollKickPending = false;
		          const sid = id;
			          selected = sid;
			          offset = 0;
		              activeLogPath = null;
		              activeThreadId = null;
				          resetChatRenderState();
				          setAttachCount(0);
				          localStorage.setItem("codexweb.selected", sid);
				          setStatus({ running: false, queueLen: 0 });
			          setContext(null);
		          setTyping(false);
		          turnOpen = false;
		          {
		            const s = sessionIndex.get(sid);
		            if (s) titleLabel.textContent = `${baseName(s.cwd) || s.session_id} (${String(s.session_id).slice(0, 8)})`;
		            else titleLabel.textContent = sid ? String(sid) : "No session selected";
		          }
                clickLoadT0 = performance.now();
                clickMetricPending = true;
		          if (pollGen !== myGen || selected !== sid) return;
			          const s0 = sessionIndex.get(sid);
			          if (s0 && s0.token) setContext(s0.token);
					          try {
						            const data = await api(`/api/sessions/${sid}/messages?offset=0&init=1&limit=${INIT_PAGE_LIMIT}&before=0`);
					            if (pollGen !== myGen || selected !== sid) return;
                    if (data && typeof data.log_path === "string") activeLogPath = data.log_path;
                    if (data && typeof data.thread_id === "string") activeThreadId = data.thread_id;
				            offset = data.offset;
				            const evs = Array.isArray(data.events) ? data.events : [];
					            if (evs.length) startInitialRender(evs);
                    olderBefore = Number.isFinite(Number(data.next_before)) ? Number(data.next_before) : 0;
                    setOlderState({ hasMore: Boolean(data.has_older), isLoading: false });
				            const nowBusy = Boolean(data.busy);
	            const turnStart = Boolean(data.turn_start);
	            const turnEnd = Boolean(data.turn_end);
	            const turnAborted = Boolean(data.turn_aborted);
				            if (turnStart || nowBusy) turnOpen = true;
				            if (turnEnd || turnAborted || !nowBusy) turnOpen = false;
				            setStatus({ running: Boolean(turnOpen || nowBusy), queueLen: data.queue_len });
				            setContext(data.token);
				            setTyping(Boolean(turnOpen || nowBusy));
			          } catch {
			            await pollMessages(sid, myGen);
			            if (pollGen !== myGen || selected !== sid) return;
			          }
            refreshSessions().catch((e) => console.error("refreshSessions failed", e));
           kickPoll(900);
           if (isMobile()) setSidebarOpen(false);
           updateHarnessBtnState();
         }

			        $("#refreshBtn").onclick = refreshSessions;
			        function updateHarnessBtnState() {
			          const s = selected ? sessionIndex.get(selected) : null;
			          const on = Boolean(s && s.harness_enabled);
			          harnessBtn.disabled = !selected;
			          harnessBtn.classList.toggle("active", Boolean(selected && on));
			        }
           async function loadHarnessCfgForSelected() {
             if (!selected) return;
             const sid = selected;
              const d = await api(`/api/sessions/${sid}/harness`);
              if (selected !== sid) return;
              if (!d || typeof d !== "object") throw new Error("invalid harness response");
              if (typeof d.enabled !== "boolean") throw new Error("invalid harness.enabled");
              if (typeof d.request !== "string") throw new Error("invalid harness.request");
              harnessCfg = { enabled: d.enabled, request: d.request };
             const enabledEl = $("#harnessEnabled");
             const requestEl = $("#harnessRequest");
             if (enabledEl) enabledEl.checked = harnessCfg.enabled;
             if (requestEl) requestEl.value = harnessCfg.request;
           }
			        function scheduleHarnessSave() {
			          if (!selected) return;
			          const sid = selected;
			          if (harnessSaveTimer) clearTimeout(harnessSaveTimer);
			          harnessSaveTimer = setTimeout(async () => {
			            if (selected !== sid) return;
               try {
                 await api(`/api/sessions/${sid}/harness`, { method: "POST", body: { enabled: harnessCfg.enabled, request: harnessCfg.request } });
                 await refreshSessions();
               } catch (e) {
                 console.error("save harness failed", e);
                 setToast(`harness save error: ${e && e.message ? e.message : "unknown error"}`);
               }
               updateHarnessBtnState();
             }, 450);
           }
			        function hideHarnessMenu() {
			          harnessMenuOpen = false;
			          harnessMenu.style.display = "none";
			        }
			        async function showHarnessMenu() {
			          if (!selected) return;
			          harnessMenuOpen = true;
			          harnessMenu.style.display = "block";
			          const rect = harnessBtn.getBoundingClientRect();
			          const top = Math.min(window.innerHeight - 12, rect.bottom + 8);
			          harnessMenu.style.top = `${top}px`;
			          harnessMenu.style.left = "12px";
			          harnessMenu.style.right = "auto";
             const w = harnessMenu.offsetWidth || 320;
             const left = Math.max(12, Math.min(window.innerWidth - 12 - w, rect.right - w));
             harnessMenu.style.left = `${left}px`;
             try {
               await loadHarnessCfgForSelected();
             } catch (e) {
               console.error("load harness failed", e);
               setToast(`harness load error: ${e && e.message ? e.message : "unknown error"}`);
               hideHarnessMenu();
             }
           }
			        function toggleHarnessMenu() {
			          if (harnessMenuOpen) hideHarnessMenu();
			          else showHarnessMenu();
			        }

			        harnessBtn.onclick = (e) => {
			          e.preventDefault();
			          e.stopPropagation();
		          toggleHarnessMenu();
		        };
		        harnessMenu.onclick = (e) => e.stopPropagation();
		        if (window.__codexwebHarnessGlobalHandlers) {
		          const h = window.__codexwebHarnessGlobalHandlers;
		          if (h.docClick) document.removeEventListener("click", h.docClick);
		          if (h.resize) window.removeEventListener("resize", h.resize);
		        }
		        const onDocClick = () => {
		          if (harnessMenuOpen) hideHarnessMenu();
		        };
		        const onResize = () => {
		          if (harnessMenuOpen) hideHarnessMenu();
		        };
			        window.__codexwebHarnessGlobalHandlers = { docClick: onDocClick, resize: onResize };
			        document.addEventListener("click", onDocClick);
			        window.addEventListener("resize", onResize);
			        const harnessEnabledEl = $("#harnessEnabled");
			        const harnessRequestEl = $("#harnessRequest");
			        if (harnessEnabledEl)
			          harnessEnabledEl.onchange = (e) => {
			            if (!selected) return;
			            harnessCfg.enabled = Boolean(e.target.checked);
			            const s = sessionIndex.get(selected);
			            if (s) s.harness_enabled = harnessCfg.enabled;
			            updateHarnessBtnState();
			            scheduleHarnessSave();
			          };
			        if (harnessRequestEl)
			          harnessRequestEl.oninput = (e) => {
			            if (!selected) return;
			            harnessCfg.request = String(e.target.value ?? "");
			            scheduleHarnessSave();
			          };
	        $("#newBtn").onclick = async () => {
	          const cur = selected ? sessionIndex.get(selected) : null;
	          const def = cur && cur.cwd && cur.cwd !== "?" ? cur.cwd : "";
	          const cwd = prompt("New session cwd:", def);
          if (!cwd) return;
          try {
            setToast("starting...");
            const res = await api("/api/sessions", { method: "POST", body: { cwd } });
            const brokerPid = res && res.broker_pid ? Number(res.broker_pid) : null;
            if (!brokerPid) {
              setToast("start failed");
              return;
            }
            setToast(`started (broker ${brokerPid})`);
            for (let i = 0; i < 60; i++) {
              const sessions = await refreshSessions();
              const found = (sessions || []).find((x) => Number(x.broker_pid || 0) === brokerPid);
              if (found) {
                selectSession(found.session_id);
                return;
              }
              await new Promise((r) => setTimeout(r, 250));
            }
            setToast("session will appear once Codex creates a rollout log");
          } catch (e) {
            setToast(`start error: ${e.message}`);
          }
        };
	        interruptBtn.onclick = async () => {
	          if (!selected) return;
	          try {
	            setToast("interrupting...");
            await api(`/api/sessions/${selected}/interrupt`, { method: "POST" });
            pollFastUntilMs = Date.now() + 2500;
            kickPoll(0);
          } catch (e) {
            setToast(`interrupt error: ${e.message}`);
          }
        };

        $("#logoutBtnSide").onclick = async () => {
          await api("/api/logout", { method: "POST" });
          renderLogin(renderApp);
        };

        toggleSidebarBtn.onclick = () => {
          if (isMobile()) {
            setSidebarOpen(!document.body.classList.contains("sidebar-open"));
            return;
          }
          setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
        };
	        backdrop.onclick = () => setSidebarOpen(false);

	        chat.addEventListener("scroll", () => {
	          const cur = chat.scrollTop;
	          const d = cur - lastScrollTop;
	          lastScrollTop = cur;
	          if (d < 0) autoScroll = false;
          else if (isNearBottom()) autoScroll = true;
          jumpBtn.style.display = autoScroll ? "none" : "inline-flex";
        });
        chat.addEventListener(
          "wheel",
          (e) => {
            if (e.deltaY < 0) {
              autoScroll = false;
              jumpBtn.style.display = "inline-flex";
              maybeAutoLoadOlder();
            }
          },
          { passive: true }
        );
        let touchY = null;
        chat.addEventListener(
          "touchstart",
          (e) => {
            const t = e.touches && e.touches[0];
            touchY = t ? t.clientY : null;
          },
          { passive: true }
        );
        chat.addEventListener(
          "touchmove",
          (e) => {
            const t = e.touches && e.touches[0];
            if (!t || touchY === null) return;
            const dy = t.clientY - touchY;
            touchY = t.clientY;
            // Finger moves down -> content scrolls up.
            if (dy > 0) {
              autoScroll = false;
              jumpBtn.style.display = "inline-flex";
              maybeAutoLoadOlder();
            }
          },
          { passive: true }
        );
        jumpBtn.onclick = () => {
          autoScroll = true;
          jumpBtn.style.display = "none";
          scrollToBottom();
        };
        olderBtn.onclick = () => {
          void loadOlderMessages({ auto: false });
        };

	        const textarea = $("#msg");
	        const msgPh = $("#msgPh");
	        const imgInput = $("#imgInput");
	        const attachBtn = $("#attachBtn");
	        if (!attachBadgeEl) {
	          attachBadgeEl = el("span", { class: "attachBadge", id: "attachBadge" });
	          attachBtn.appendChild(attachBadgeEl);
	        }
	        const setAttachCount = (n) => {
	          const next = Math.max(0, Number(n) || 0);
	          attachedImages = next;
	          if (!attachBadgeEl) return;
	          if (next > 0) {
	            attachBadgeEl.textContent = String(next);
	            attachBadgeEl.style.display = "inline-flex";
	          } else {
	            attachBadgeEl.textContent = "";
	            attachBadgeEl.style.display = "none";
	          }
	        };
	        setAttachCount(0);
	        function autoGrow() {
	          const basePx = parseFloat(getComputedStyle(textarea).minHeight || "0") || 32;
	          const maxPx = 180;
	          const hasNewline = textarea.value.includes("\n");
	          if (msgPh) msgPh.style.display = textarea.value ? "none" : "flex";
	          textarea.style.height = `${basePx}px`;
	          let h = textarea.scrollHeight;
	          const needsMultiline = hasNewline || h > basePx + 1;
	          form.classList.toggle("multiline", needsMultiline);
	          textarea.style.height = needsMultiline ? "auto" : `${basePx}px`;
	          h = textarea.scrollHeight;
	          const next = needsMultiline ? Math.min(h, maxPx) : basePx;
	          textarea.style.height = `${next}px`;
	          textarea.style.overflowY = h > maxPx ? "auto" : "hidden";
	          if (autoScroll) requestAnimationFrame(() => scrollToBottom());
	        }
	        textarea.addEventListener("input", autoGrow);
	        textarea.addEventListener(
	          "focus",
	          () => {
	            autoScroll = true;
	            jumpBtn.style.display = "none";
	            const tick = () => {
	              updateAppHeightVar();
	              scrollToBottom();
	            };
	            requestAnimationFrame(tick);
	            setTimeout(tick, 120);
	            setTimeout(tick, 350);
	          },
	          { passive: true }
	        );
	        textarea.addEventListener(
	          "blur",
	          () => {
	            setTimeout(updateAppHeightVar, 0);
	          },
	          { passive: true }
	        );
        textarea.addEventListener("keydown", (e) => {
          if (e.key !== "Enter") return;
          if (e.isComposing) return;
          if (!(e.ctrlKey || e.metaKey)) return;
          e.preventDefault();
          form.requestSubmit();
        });
        window.addEventListener("resize", () => {
          if (autoScroll) requestAnimationFrame(() => scrollToBottom());
        });

	        attachBtn.onclick = () => {
	          if (!selected) return;
	          imgInput.value = "";
	          imgInput.click();
	        };
		        imgInput.addEventListener("change", async () => {
		          if (!selected) return;
		          const f = imgInput.files && imgInput.files[0];
		          if (!f) return;
		          if (sending) return;
		          try {
	            function safeStem(name) {
	              const s = String(name || "image");
	              const base = s.split("/").pop() || s;
	              const dot = base.lastIndexOf(".");
	              return (dot > 0 ? base.slice(0, dot) : base).replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 80) || "image";
	            }
	            function extLower(name) {
	              const s = String(name || "");
	              const dot = s.lastIndexOf(".");
	              return dot >= 0 ? s.slice(dot + 1).toLowerCase() : "";
	            }
	            function isLikelyHeic(file) {
	              const t = String(file.type || "").toLowerCase();
	              const e = extLower(file.name);
	              return t.includes("heic") || t.includes("heif") || e === "heic" || e === "heif";
	            }
	            function isSupportedMime(type) {
	              const t = String(type || "").toLowerCase();
	              return t === "image/png" || t === "image/jpeg" || t === "image/jpg" || t === "image/webp";
	            }
	            function b64FromBytes(bytes) {
	              let bin = "";
	              const chunk = 0x8000;
	              for (let i = 0; i < bytes.length; i += chunk) {
	                bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
	              }
	              return btoa(bin);
	            }
	            async function toJpegBlob(file, { maxDim = 2048, quality = 0.86 } = {}) {
	              const url = URL.createObjectURL(file);
	              try {
	                const img = new Image();
	                img.decoding = "async";
	                img.src = url;
	                if (img.decode) await img.decode();
	                else
	                  await new Promise((resolve, reject) => {
	                    img.onload = resolve;
	                    img.onerror = () => reject(new Error("decode failed"));
	                  });
	                const w0 = img.naturalWidth || img.width || 0;
	                const h0 = img.naturalHeight || img.height || 0;
	                if (!w0 || !h0) throw new Error("invalid image dimensions");
	                const scale = Math.min(1, maxDim / Math.max(w0, h0));
	                const w = Math.max(1, Math.round(w0 * scale));
	                const h = Math.max(1, Math.round(h0 * scale));
	                const canvas = document.createElement("canvas");
	                canvas.width = w;
	                canvas.height = h;
	                const ctx = canvas.getContext("2d", { alpha: false });
	                if (!ctx) throw new Error("no canvas");
	                ctx.drawImage(img, 0, 0, w, h);
	                const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
	                if (!blob) throw new Error("jpeg encode failed");
	                return blob;
	              } finally {
	                URL.revokeObjectURL(url);
	              }
	            }

	            setToast("attaching image...");
	            const maxBytes = 10 * 1024 * 1024;
	            let uploadBlob = f;
	            let uploadName = f.name || "image";
	            if (f.size > maxBytes || isLikelyHeic(f) || !isSupportedMime(f.type)) {
	              setToast("converting image...");
	              const stem = safeStem(f.name);
	              uploadName = `${stem}.jpg`;
	              // Try a few (dim, quality) pairs until it fits.
	              const tries = [
	                { maxDim: 2048, quality: 0.86 },
	                { maxDim: 1600, quality: 0.82 },
	                { maxDim: 1600, quality: 0.72 },
	                { maxDim: 1280, quality: 0.68 },
	                { maxDim: 1280, quality: 0.58 },
	              ];
	              let blob = null;
	              for (const t of tries) {
	                blob = await toJpegBlob(f, t);
	                if (blob.size <= maxBytes) break;
	              }
	              if (!blob || blob.size > maxBytes) throw new Error("image too large");
	              uploadBlob = blob;
	            }

	            const ab = await uploadBlob.arrayBuffer();
	            if (ab.byteLength > maxBytes) throw new Error("image too large");
	            const b64 = b64FromBytes(new Uint8Array(ab));
		            const res = await api(`/api/sessions/${selected}/inject_image`, {
		              method: "POST",
		              body: { filename: uploadName, data_b64: b64 },
		            });
		            if (res && res.ok) {
		              setToast("image attached");
		              setAttachCount(attachedImages + 1);
		            }
		            pollFastUntilMs = Date.now() + 4000;
		            kickPoll(0);
		          } catch (e) {
	            setToast(`attach error: ${e.message}`);
	          }
	        });

        form.onsubmit = async (e) => {
          e.preventDefault();
          if (!selected) return;
          const raw = $("#msg").value;
          if (!raw || !raw.trim()) return;
          if (sending) return;
          sending = true;
          $("#sendBtn").disabled = true;
          setToast("sending...");
          $("#msg").value = "";
          autoGrow();

          const localId = ++localEchoSeq;
          const t0 = Date.now() / 1000;
          pendingUser.push({ id: localId, key: pendingMatchKey(raw), loose: normalizeTextForPendingMatch(raw), t0, text: raw });
          appendEvent({ role: "user", text: raw, pending: true, localId, ts: t0 });
          turnOpen = true;
	          try {
	            const res = await api(`/api/sessions/${selected}/send`, { method: "POST", body: { text: raw } });
	            if (res.queued) setToast(`queued (queue ${res.queue_len})`);
	            else setToast("sent");
	            setAttachCount(0);
	            pollFastUntilMs = Date.now() + 5000;
	            kickPoll(0);
	            await refreshSessions();
	          } catch (e2) {
	            setToast(`send error: ${e2.message}`);
            const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${localId}"]`);
            if (pendingEl) {
              pendingEl.style.opacity = "1";
              pendingEl.style.borderColor = "rgba(185, 28, 28, 0.7)";
              pendingEl.style.boxShadow = "0 0 0 2px rgba(185, 28, 28, 0.12)";
            }
          } finally {
            sending = false;
            $("#sendBtn").disabled = false;
          }
        };

	        (async () => {
	          if (localStorage.getItem("codexweb.sidebarCollapsed") === "1") setSidebarCollapsed(true);
	          if (localStorage.getItem("codexweb.sidebarOpen") === "1") setSidebarOpen(true);

	          try {
	            const sessions = await refreshSessions();
	            const remembered = localStorage.getItem("codexweb.selected");
	            const first = sessions && sessions.length ? sessions[0].session_id : null;
	            const pick = remembered && sessionIndex.has(remembered) ? remembered : first;
	            if (pick) selectSession(pick);
	          } catch (e) {
	            if (e && e.status === 401) {
	              renderLogin(renderApp);
	              return;
	            }
	            console.error("initial refreshSessions failed", e);
	            setToast(`sessions error: ${e && e.message ? e.message : "unknown error"}`);
	          } finally {
	            if (msgPh) msgPh.style.display = textarea.value ? "none" : "flex";
	            autoGrow();

	            if (sessionsTimer) clearInterval(sessionsTimer);
	            sessionsTimer = setInterval(async () => {
	              try {
	                await refreshSessions();
	              } catch (e2) {
	                if (e2 && e2.status === 401) {
	                  if (sessionsTimer) clearInterval(sessionsTimer);
	                  sessionsTimer = null;
	                  renderLogin(renderApp);
	                  return;
	                }
	                console.error("refreshSessions timer failed", e2);
	              }
	            }, 2500);
	          }
	        })();
      }

      (async function boot() {
        try {
          await api("/api/me");
          renderApp();
        } catch (e) {
          if (e && e.status === 401) {
            renderLogin(renderApp);
            return;
          }
          console.error("boot auth check failed", e);
          const err = document.createElement("pre");
          err.textContent = `error: unable to contact server (${e && e.message ? e.message : "unknown error"})`;
          document.body.innerHTML = "";
          document.body.appendChild(err);
        }
      })();
