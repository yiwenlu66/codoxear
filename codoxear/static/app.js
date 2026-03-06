	      const $ = (q) => document.querySelector(q);
	      const UI_VERSION = "20260306.6";
	      function updateAppHeightVar() {
	        const vv = window.visualViewport;
	        const layoutH = Math.round(window.innerHeight);
	        const visualH = Math.round(vv ? vv.height : window.innerHeight);
	        const visualTop = Math.max(0, Math.round(vv ? vv.offsetTop : 0));
	        if (updateAppHeightVar._h === visualH && updateAppHeightVar._l === layoutH && updateAppHeightVar._t === visualTop) return;
	        updateAppHeightVar._h = visualH;
	        updateAppHeightVar._l = layoutH;
	        updateAppHeightVar._t = visualTop;
	        document.documentElement.style.setProperty("--appH", `${visualH}px`);
	        document.documentElement.style.setProperty("--layoutH", `${layoutH}px`);
	        document.documentElement.style.setProperty("--vvTop", `${visualTop}px`);
	      }
	      updateAppHeightVar();
	      window.addEventListener("resize", updateAppHeightVar);
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

      function fmtBytes(n) {
        const v = Number(n);
        if (!Number.isFinite(v)) return String(n ?? "");
        if (v < 1024) return `${v} B`;
        const units = ["B", "KB", "MB", "GB", "TB"];
        let val = v;
        let u = 0;
        while (val >= 1024 && u < units.length - 1) {
          val /= 1024;
          u += 1;
        }
        const dec = val >= 100 ? 0 : val >= 10 ? 1 : 2;
        return `${val.toFixed(dec)} ${units[u]}`;
      }

      function listFromFilesField(val) {
        if (!Array.isArray(val)) return [];
        const out = [];
        for (const v of val) {
          if (typeof v !== "string") continue;
          const p = v.trim();
          if (!p || out.includes(p)) continue;
          out.push(p);
        }
        return out;
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

      function sessionDisplayName(s) {
        if (!s || typeof s !== "object") return "";
        const alias = typeof s.alias === "string" ? s.alias.trim() : "";
        if (alias) return alias;
        const cwdName = baseName(s.cwd);
        if (cwdName) return cwdName;
        const ts = typeof s.updated_ts === "number" && Number.isFinite(s.updated_ts)
          ? s.updated_ts
          : typeof s.start_ts === "number" && Number.isFinite(s.start_ts)
            ? s.start_ts
            : 0;
        return ts ? `Session ${fmtTs(ts)}` : "Session";
      }

	      function fmtIdleAge(seconds) {
	        const s = Number(seconds);
	        if (!(s >= 0)) return "";
	        if (s < 60) return "just now";
	        if (s < 3600) return `${Math.max(1, Math.floor(s / 60))}m`;
	        if (s < 86400) return `${Math.max(1, Math.floor(s / 3600))}h`;
	        return `${Math.max(1, Math.floor(s / 86400))}d`;
	      }

	      function fmtRelativeAge(seconds) {
	        const base = fmtIdleAge(seconds);
	        if (!base || base === "just now") return base;
	        return `${base} ago`;
	      }

      function sessionTitleWithId(s) {
        if (!s || typeof s !== "object") return "No session selected";
        const name = sessionDisplayName(s);
        return name || "No session selected";
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

      function stripPathLocationSuffix(rawPath) {
        const s = String(rawPath || "").trim();
        return s.replace(/:\d+(?::\d+)?$/, "");
      }

      function localPathFromRef(u) {
        const raw = String(u ?? "").trim();
        if (!raw) return null;
        if (raw.startsWith("/") && !raw.startsWith("//")) return stripPathLocationSuffix(raw);
        try {
          const url = new URL(raw, location.href);
          if (url.origin !== location.origin) return null;
          const pathname = stripPathLocationSuffix(decodeURIComponent(url.pathname || ""));
          if (/^\/(?:home|tmp|mnt|var|opt|usr|etc|private|Users|Volumes)\//.test(pathname)) return pathname;
        } catch {}
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
            const localPath = localPathFromRef(m[3]);
            if (localPath) {
              out += `<a href="#" class="inlineFileLink" data-local-path="${escapeHtml(localPath)}">${escapeHtml(m[2])}</a>`;
            } else {
              const href = safeUrl(m[3]);
              if (!href) out += `${escapeHtml(m[2])} (${escapeHtml(m[3])})`;
              else out += `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer noopener">${escapeHtml(m[2])}</a>`;
            }
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
        if (name === "edit")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>`;
        if (name === "file")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/></svg>`;
        if (name === "x")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18"/><path d="M6 6l12 12"/></svg>`;
        if (name === "queue")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h10"/></svg>`;
        if (name === "duplicate")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="8" y="8" width="11" height="11" rx="2"/><rect x="5" y="5" width="11" height="11" rx="2"/></svg>`;
        if (name === "help")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2-3 4"/><path d="M12 17h.01"/></svg>`;
        if (name === "info")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>`;
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
          el("button", { id: "helpBtnSide", type: "button", title: "Help", "aria-label": "Help", html: iconSvg("help") + "Help" }),
          el("button", { id: "logoutBtnSide", type: "button", title: "Log out", "aria-label": "Log out", html: iconSvg("logout") + "Log out" }),
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
        const INIT_PAGE_LIMIT_DESKTOP = 60;
        const INIT_PAGE_LIMIT_MOBILE = 24;
        const OLDER_PAGE_LIMIT = 60;
        const CACHE_LIMIT = 40;
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
         let currentRunning = false;
         let openSwipeContent = null;
        const cacheBySession = new Map();
        const cacheLoaded = new Set();
        const cacheSaveTimers = new Map();
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
        let queueBadgeEl = null;
         const recentEventKeys = [];
         const recentEventKeySet = new Set();
         const RECENT_EVENT_KEYS_MAX = 320;
         let lastAssistantKey = "";
                 let clickLoadT0 = 0;
                 let clickMetricPending = false;
              let harnessMenuOpen = false;
              let harnessCfg = { enabled: false, request: "" };
              let harnessSaveTimer = null;

            const titleLabel = el("div", { id: "threadTitle", text: "No session selected" });
            titleLabel.style.cursor = "pointer";
            titleLabel.title = "Click to rename";
            titleLabel.onclick = () => {
              if (!selected) return;
              void renameSessionId(selected);
            };
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
        const diagBtn = el("button", {
          id: "diagBtn",
          class: "icon-btn",
          title: "Details",
          "aria-label": "Details",
          type: "button",
          html: iconSvg("info"),
        });
        diagBtn.disabled = true;
        const fileBtn = el("button", {
          id: "fileBtn",
          class: "icon-btn",
          title: "View file",
          "aria-label": "View file",
          type: "button",
          html: iconSvg("file"),
        });
        fileBtn.disabled = true;
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

        const topMeta = el("div", { class: "topMeta" }, [ctxChip]);
        const titleRow = el("div", { class: "titleRow" }, [titleLabel, topMeta]);
        const titleWrap = el("div", { class: "titleWrap" }, [titleRow]);
        const topbar = el("div", { class: "topbar" }, [
          el("div", { class: "pill" }, [toggleSidebarBtn, titleWrap]),
          el("div", { class: "actions topActions" }, [
            fileBtn,
            diagBtn,
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
          el("button", { class: "icon-btn", id: "queueBtn", type: "button", title: "Queued messages", "aria-label": "Queued messages", html: iconSvg("queue") }),
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
        main.appendChild(toast);
        main.appendChild(chatWrap);
        main.appendChild(composer);
        app.appendChild(sidebar);
        app.appendChild(main);
        app.appendChild(backdrop);
        root.appendChild(app);
        root.appendChild(harnessMenu);

        const fileBackdrop = el("div", { class: "modalBackdrop", id: "fileBackdrop" });
        const fileCloseBtn = el("button", {
          id: "fileCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const fileStatus = el("div", { class: "muted fileStatus", id: "fileStatus", text: "" });
        const filePickerBtn = el("button", {
          id: "filePickerBtn",
          class: "filePickerBtn",
          type: "button",
          title: "Choose file",
          "aria-label": "Choose file",
        });
        const filePickerMenu = el("div", { id: "filePickerMenu", class: "filePickerMenu" });
        const fileModeDiffBtn = el("button", {
          id: "fileModeDiffBtn",
          class: "icon-btn text-btn",
          type: "button",
          title: "Toggle diff",
          "aria-label": "Toggle diff",
          text: "Diff",
        });
        const fileAddBtn = el("button", {
          id: "fileAddBtn",
          class: "icon-btn text-btn",
          type: "button",
          title: "Add file",
          "aria-label": "Add file",
          text: "Add",
        });
        const fileDiff = el("div", { class: "fileDiff", id: "fileDiff" });
        const fileViewer = el("div", { class: "fileViewer", id: "fileViewer", role: "dialog", "aria-label": "File viewer" }, [
          el("div", { class: "fileViewerHeader" }, [
            el("div", { class: "title", text: "View file" }),
            el("div", { class: "actions" }, [fileModeDiffBtn, fileAddBtn, fileCloseBtn]),
          ]),
          el("div", { class: "fileCandRow", id: "fileCandRow" }, [filePickerBtn, filePickerMenu]),
          fileStatus,
          fileDiff,
        ]);
        root.appendChild(fileBackdrop);
        root.appendChild(fileViewer);

        const sendChoiceBackdrop = el("div", { class: "modalBackdrop", id: "sendChoiceBackdrop" });
        const sendChoice = el("div", { class: "sendChoice", id: "sendChoice", role: "dialog", "aria-label": "Send options" }, [
          el("div", { class: "title", text: "Current response is running" }),
          el("div", { class: "muted", text: "Choose how to handle your next message." }),
          el("div", { class: "sendChoiceActions" }, [
            el("button", { class: "primary", id: "sendChoiceNow", type: "button", text: "Send now" }),
            el("button", { id: "sendChoiceLater", type: "button", text: "Send after current" }),
            el("button", { id: "sendChoiceCancel", type: "button", text: "Cancel" }),
          ]),
        ]);
        root.appendChild(sendChoiceBackdrop);
        root.appendChild(sendChoice);

        const queueBackdrop = el("div", { class: "modalBackdrop", id: "queueBackdrop" });
        const queueCloseBtn = el("button", {
          id: "queueCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const queueList = el("div", { class: "queueList", id: "queueList" });
        const queueEmpty = el("div", { class: "muted", id: "queueEmpty", text: "No queued messages." });
        const queueViewer = el("div", { class: "queueViewer", id: "queueViewer", role: "dialog", "aria-label": "Queued messages" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Queued messages" }),
            el("div", { class: "actions" }, [queueCloseBtn]),
          ]),
          queueEmpty,
          queueList,
        ]);
        root.appendChild(queueBackdrop);
        root.appendChild(queueViewer);

        const helpBackdrop = el("div", { class: "modalBackdrop", id: "helpBackdrop" });
        const helpCloseBtn = el("button", {
          id: "helpCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const helpViewer = el("div", { class: "helpViewer", id: "helpViewer", role: "dialog", "aria-label": "Help" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Help" }),
            el("div", { class: "actions" }, [helpCloseBtn]),
          ]),
          el("div", {
            class: "helpBody",
            html: `<div class="muted">Sessions list</div>
<ul class="md">
  <li>On mobile: swipe left on a session to reveal <b>Rename</b> and <b>Duplicate</b>.</li>
  <li>On mobile: swipe right on a web-owned session to reveal <b>Delete</b>.</li>
  <li>On desktop: session actions are shown on the right.</li>
  <li>The dot indicates state: <b>blue</b> = busy, <b>gray</b> = idle.</li>
  <li>The status line starts with a boxed <b>W</b> (web-owned) or boxed <b>T</b> (terminal-owned).</li>
</ul>
<div class="muted">Queue</div>
<ul class="md">
  <li>When a response is running, choose <b>Send after current</b> to enqueue the message for that session.</li>
  <li>Queued messages are stored with the session and sent when it becomes idle.</li>
</ul>`,
          }),
        ]);
        root.appendChild(helpBackdrop);
        root.appendChild(helpViewer);

        const diagBackdrop = el("div", { class: "modalBackdrop", id: "diagBackdrop" });
        const diagCloseBtn = el("button", {
          id: "diagCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const diagStatus = el("div", { class: "muted", id: "diagStatus", text: "" });
        const diagContent = el("div", { class: "detailsGrid", id: "diagContent" });
        const diagViewer = el("div", { class: "diagViewer", id: "diagViewer", role: "dialog", "aria-label": "Details" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Details" }),
            el("div", { class: "actions" }, [diagCloseBtn]),
          ]),
          diagStatus,
          diagContent,
        ]);
        root.appendChild(diagBackdrop);
        root.appendChild(diagViewer);

        function setToast(text) {
          toast.textContent = text || "";
          if (!text) return;
          setTimeout(() => {
            if (toast.textContent === text) toast.textContent = "";
          }, 2200);
        }

        let currentQueueLen = 0;
        function setStatus({ running, queueLen }) {
          const q = Math.max(0, Number(queueLen) || 0);
          const mobile = isMobile();
          const wasRunning = currentRunning;
          currentRunning = Boolean(running);
          currentQueueLen = q;
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
          if (wasRunning && !currentRunning) {
            // no-op placeholder; keep transition boundary for future UI behavior
          }
          updateQueueBadge();
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
          recentEventKeys.length = 0;
          recentEventKeySet.clear();
          lastAssistantKey = "";
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

        function initPageLimit() {
          return isMobile() ? INIT_PAGE_LIMIT_MOBILE : INIT_PAGE_LIMIT_DESKTOP;
        }

        function olderPageLimit() {
          return OLDER_PAGE_LIMIT;
        }

        function cacheStorageKey(sid) {
          return `codexweb.cache.v4.${sid}`;
        }

        function normalizeCacheEvent(ev) {
          if (!ev || (ev.role !== "user" && ev.role !== "assistant")) return null;
          if (typeof ev.text !== "string" || !ev.text.trim()) return null;
          const out = { role: ev.role, text: ev.text };
          if (typeof ev.ts === "number" && Number.isFinite(ev.ts)) out.ts = ev.ts;
          return out;
        }

        function assistantTextKey(ev) {
          if (!ev || ev.role !== "assistant") return "";
          if (typeof ev.text !== "string") return "";
          return pendingMatchKey(ev.text);
        }

        function isAdjacentAssistantDuplicate(prev, ev) {
          if (!prev || !ev) return false;
          if (prev.role !== "assistant" || ev.role !== "assistant") return false;
          const a = assistantTextKey(prev);
          if (!a) return false;
          return a === assistantTextKey(ev);
        }

        function dedupeAdjacentAssistantDuplicates(events) {
          const out = [];
          for (const ev of events || []) {
            const prev = out.length ? out[out.length - 1] : null;
            if (isAdjacentAssistantDuplicate(prev, ev)) continue;
            out.push(ev);
          }
          return out;
        }

        function loadCacheFromStorage(sid) {
          if (!sid || cacheLoaded.has(sid)) return;
          cacheLoaded.add(sid);
          try {
            const raw = localStorage.getItem(cacheStorageKey(sid));
            if (!raw) return;
            const obj = JSON.parse(raw);
            if (!obj || typeof obj !== "object") return;
            const eventsIn = Array.isArray(obj.events) ? obj.events : [];
            const events = [];
            for (const ev of eventsIn) {
              const norm = normalizeCacheEvent(ev);
              if (norm) events.push(norm);
            }
            if (!events.length) return;
            const deduped = dedupeAdjacentAssistantDuplicates(events);
            const cacheChanged = deduped.length !== events.length;
            if (cacheChanged) {
              events.length = 0;
              events.push(...deduped);
            }
            if (events.length > CACHE_LIMIT) events.splice(0, events.length - CACHE_LIMIT);
            const cache = {
              log_path: typeof obj.log_path === "string" ? obj.log_path : null,
              offset: Number(obj.offset) || 0,
              older_before: Number(obj.older_before) || 0,
              has_older: Boolean(obj.has_older),
              events,
            };
            cacheBySession.set(sid, cache);
            if (cacheChanged) scheduleCacheSave(sid);
          } catch {
            // ignore corrupted cache
          }
        }

        function getCache(sid) {
          if (!sid) return null;
          loadCacheFromStorage(sid);
          return cacheBySession.get(sid) || null;
        }

        function saveCacheNow(sid) {
          if (!sid) return;
          const cache = cacheBySession.get(sid);
          if (!cache) {
            localStorage.removeItem(cacheStorageKey(sid));
            return;
          }
          const payload = {
            log_path: cache.log_path || null,
            offset: Number(cache.offset) || 0,
            older_before: Number(cache.older_before) || 0,
            has_older: Boolean(cache.has_older),
            events: Array.isArray(cache.events) ? cache.events : [],
          };
          try {
            localStorage.setItem(cacheStorageKey(sid), JSON.stringify(payload));
          } catch {
            // ignore quota issues
          }
        }

        function scheduleCacheSave(sid) {
          if (!sid) return;
          const existing = cacheSaveTimers.get(sid);
          if (existing) clearTimeout(existing);
          const t = setTimeout(() => {
            cacheSaveTimers.delete(sid);
            saveCacheNow(sid);
          }, 400);
          cacheSaveTimers.set(sid, t);
        }

        function setCacheMeta(sid, { logPath, offset: off, olderBefore, hasOlder } = {}) {
          if (!sid) return;
          const cache =
            getCache(sid) || { log_path: null, offset: 0, older_before: 0, has_older: false, events: [] };
          if (logPath !== undefined) cache.log_path = logPath || null;
          if (typeof off === "number" && Number.isFinite(off)) cache.offset = off;
          if (typeof olderBefore === "number" && Number.isFinite(olderBefore)) cache.older_before = olderBefore;
          if (typeof hasOlder === "boolean") cache.has_older = hasOlder;
          cacheBySession.set(sid, cache);
          scheduleCacheSave(sid);
        }

        function replaceCacheEvents(sid, events) {
          if (!sid) return;
          const cache =
            getCache(sid) || { log_path: null, offset: 0, older_before: 0, has_older: false, events: [] };
          const out = [];
          for (const ev of events || []) {
            const norm = normalizeCacheEvent(ev);
            if (!norm) continue;
            const prev = out.length ? out[out.length - 1] : null;
            if (isAdjacentAssistantDuplicate(prev, norm)) continue;
            out.push(norm);
          }
          if (out.length > CACHE_LIMIT) out.splice(0, out.length - CACHE_LIMIT);
          cache.events = out;
          cacheBySession.set(sid, cache);
          scheduleCacheSave(sid);
        }

        function appendCacheEvents(sid, events) {
          if (!sid || !events || !events.length) return;
          const cache =
            getCache(sid) || { log_path: null, offset: 0, older_before: 0, has_older: false, events: [] };
          const list = Array.isArray(cache.events) ? cache.events : [];
          let prev = list.length ? list[list.length - 1] : null;
          for (const ev of events) {
            const norm = normalizeCacheEvent(ev);
            if (!norm) continue;
            if (isAdjacentAssistantDuplicate(prev, norm)) continue;
            list.push(norm);
            prev = norm;
          }
          if (list.length > CACHE_LIMIT) list.splice(0, list.length - CACHE_LIMIT);
          cache.events = list;
          cacheBySession.set(sid, cache);
          scheduleCacheSave(sid);
        }

        function clearCache(sid) {
          if (!sid) return;
          cacheBySession.delete(sid);
          cacheLoaded.delete(sid);
          cacheSaveTimers.delete(sid);
          localStorage.removeItem(cacheStorageKey(sid));
        }

        function updateQueueBadge() {
          if (!queueBadgeEl) return;
          if (!selected) {
            queueBadgeEl.textContent = "";
            queueBadgeEl.style.display = "none";
            return;
          }
          const n = Math.max(0, Number(currentQueueLen) || 0);
          if (n > 0) {
            queueBadgeEl.textContent = String(n);
            queueBadgeEl.style.display = "inline-flex";
          } else {
            queueBadgeEl.textContent = "";
            queueBadgeEl.style.display = "none";
          }
          if (queueViewer.style.display === "flex") {
            void refreshQueueViewer();
          }
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
          // Avoid scrollIntoView() on mobile Safari, which can scroll the whole page when the
          // on-screen keyboard opens/closes.
          chat.scrollTop = chat.scrollHeight;
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

      function eventKey(ev) {
        if (!ev || (ev.role !== "user" && ev.role !== "assistant")) return "";
        const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : null;
        if (ts === null) return "";
        const tsMs = Math.round(ts * 1000);
        const text = typeof ev.text === "string" ? pendingMatchKey(ev.text) : "";
        return `${ev.role}|${tsMs}|${text}`;
      }

        function markEventSeen(ev) {
          const key = eventKey(ev);
          if (!key) return;
          if (recentEventKeySet.has(key)) return;
          recentEventKeySet.add(key);
          recentEventKeys.push(key);
          if (recentEventKeys.length > RECENT_EVENT_KEYS_MAX) {
            const drop = recentEventKeys.splice(0, recentEventKeys.length - RECENT_EVENT_KEYS_MAX);
            for (const k of drop) recentEventKeySet.delete(k);
          }
        }

        function isDuplicateEvent(ev) {
          const key = eventKey(ev);
          if (!key) return false;
          return recentEventKeySet.has(key);
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
          if (!pendingEl) return false;

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
          if (selected) {
            appendCacheEvents(selected, [ev]);
          }
          markEventSeen(ev);
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
           openSwipeContent = null;
           sessionIndex = new Map();
           const mobile = isMobile();
            const sessions = (data.sessions || [])
               .slice()
               .sort((a, b) => (b.updated_ts || b.start_ts || 0) - (a.updated_ts || a.start_ts || 0));
		          for (const s of sessions) {
		            sessionIndex.set(s.session_id, s);
		            const card = el("div", { class: "session" + (selected === s.session_id ? " active" : "") });

             const title = sessionDisplayName(s);
             const badges = [];
             if (s.harness_enabled) badges.push(el("span", { class: "badge harness", text: "harness", title: "Harness mode enabled" }));
             if (s.queue_len) badges.push(el("span", { class: "badge queue", text: `queue ${s.queue_len}` }));

	             const updatedTs = typeof s.updated_ts === "number" && Number.isFinite(s.updated_ts) ? s.updated_ts : s.start_ts;
	             const ageS = updatedTs ? Math.max(0, Date.now() / 1000 - updatedTs) : 0;
	             const ownerTxt = s.owned ? "W" : "T";
	             const stateTxt = fmtRelativeAge(ageS);
	             const cwdBase = baseName(s.cwd);

            function closeOpenSwipe() {
              if (!openSwipeContent) return;
              openSwipeContent.style.transform = "translate3d(0px, 0, 0)";
              openSwipeContent.dataset.swipeX = "0";
              openSwipeContent = null;
            }

             async function doDelete(e) {
               if (e) {
                 e.preventDefault();
                 e.stopPropagation();
               }
               closeOpenSwipe();
               if (!confirm("Delete this session?")) return;
               try {
                 await api(`/api/sessions/${s.session_id}/delete`, { method: "POST", body: {} });
                 clearCache(s.session_id);
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
                   updateQueueBadge();
                   if (harnessMenuOpen) hideHarnessMenu();
                   updateHarnessBtnState();
                 }
                 await refreshSessions();
               } catch (err) {
                 setToast(`delete error: ${err.message}`);
               }
             }

             const renameBtn = el("button", {
               class: "icon-btn",
               title: "Rename session",
               "aria-label": "Rename session",
               type: "button",
               html: iconSvg("edit"),
             });
             renameBtn.onclick = (e) => {
               e.preventDefault();
               e.stopPropagation();
               closeOpenSwipe();
               void renameSessionId(s.session_id);
             };
             const dupBtn = el("button", {
               class: "icon-btn",
               title: "Duplicate session",
               "aria-label": "Duplicate session",
               type: "button",
               html: iconSvg("duplicate"),
             });
             dupBtn.onclick = async (e) => {
               e.preventDefault();
               e.stopPropagation();
               closeOpenSwipe();
               const cwd = s && s.cwd && s.cwd !== "?" ? s.cwd : "";
               if (!cwd) {
                 setToast("cwd unavailable");
                 return;
               }
               await spawnSessionWithCwd(cwd);
             };
             const delBtn = s.owned
               ? el("button", {
                   class: "icon-btn danger sessionDel",
                   title: "Delete session",
                   "aria-label": "Delete session",
                   type: "button",
                   html: iconSvg("trash"),
                 })
               : null;
             if (delBtn) delBtn.onclick = (e) => void doDelete(e);

             const stateDot = el("span", { class: "stateDot" + (s.busy ? " busy" : " idle") });
             const titleRow = el("div", { class: "sessionTitleRow" }, [
               stateDot,
               el("div", { class: "titleLine", text: title, title: s.cwd || "" }),
             ]);
	             const badgesWrap = el("div", { class: "sessionBadges" }, badges);
	             const meta = el("div", { class: "muted subLine sessionMetaLine" }, [
	               el("span", { class: "ownerBadge", text: ownerTxt, title: s.owned ? "web-owned session" : "terminal-owned session" }),
	               el("span", { class: "metaText", text: `${stateTxt}${cwdBase ? ` | ${cwdBase}` : ""}` }),
	             ]);

             if (mobile) {
               const leftActions = el("div", { class: "sessionActions left" }, delBtn ? [delBtn] : []);
               const rightActions = el("div", { class: "sessionActions right" }, [renameBtn, dupBtn]);
               const top = el("div", { class: "row" }, [titleRow, badgesWrap]);
               const inner = el("div", { class: "sessionInner" }, [top, meta]);
               const content = el("div", { class: "sessionContent" }, [inner]);
               content.dataset.swipeX = "0";
               const swipe = el("div", { class: "sessionSwipe" }, [leftActions, rightActions, content]);
               card.appendChild(swipe);

	               const leftMax = s.owned ? 72 : 0;
	               const rightMax = 104;
	               let startX = null;
	               let startY = 0;
	               let startSwipe = 0;
	               let lastMoveTs = 0;
	               let lastMoveX = 0;
	               let swipeVelocity = 0;
	               let dragging = false;
	                content.addEventListener("pointerdown", (e) => {
	                  if (e.pointerType === "mouse" && e.button !== 0) return;
	                  startX = e.clientX;
	                  startY = e.clientY;
	                  startSwipe = Number(content.dataset.swipeX || 0);
	                  lastMoveTs = performance.now();
	                  lastMoveX = e.clientX;
	                  swipeVelocity = 0;
	                  dragging = false;
	                  if (openSwipeContent && openSwipeContent !== content) closeOpenSwipe();
	                  try {
	                    content.setPointerCapture(e.pointerId);
	                  } catch (_) {}
                });
	                 content.addEventListener("pointermove", (e) => {
	                   if (startX === null) return;
	                   const dx = e.clientX - startX;
	                   const dy = e.clientY - startY;
	                  const now = performance.now();
	                  const dt = Math.max(now - lastMoveTs, 1);
	                  swipeVelocity = ((e.clientX - lastMoveX) / dt) * 1000;
	                  lastMoveTs = now;
	                  lastMoveX = e.clientX;
	                  if (!dragging) {
	                    if (Math.abs(dx) < 4) return;
	                    if (Math.abs(dx) < Math.abs(dy) * 0.7) return;
	                    dragging = true;
	                    content.style.transition = "none";
	                  }
	                  if (dragging) e.preventDefault();
	                  let x = startSwipe + dx;
                  x = Math.min(leftMax, Math.max(-rightMax, x));
                   content.style.transform = `translate3d(${x}px, 0, 0)`;
                   content.dataset.swipeX = String(x);
                 });
                function finishSwipe(e) {
                  if (startX === null) return;
                  try {
                    if (e && e.pointerId != null) content.releasePointerCapture(e.pointerId);
                  } catch (_) {}
                  startX = null;
                  if (!dragging) return;
	                  dragging = false;
	                  content.style.transition = "";
	                  const x = Number(content.dataset.swipeX || 0);
	                 let target = 0;
	                  const commitLeft = leftMax > 0 && (x > leftMax * 0.28 || swipeVelocity > 420);
	                  const commitRight = rightMax > 0 && (-x > rightMax * 0.28 || swipeVelocity < -420);
	                  if (commitLeft) target = leftMax;
	                  else if (commitRight) target = -rightMax;
	                  content.style.transform = `translate3d(${target}px, 0, 0)`;
	                  content.dataset.swipeX = String(target);
	                  if (target !== 0) openSwipeContent = content;
	                  else if (openSwipeContent === content) openSwipeContent = null;
	                }
               content.addEventListener("pointerup", finishSwipe);
               content.addEventListener("pointercancel", finishSwipe);

               card.onclick = () => {
                 const x = Number(content.dataset.swipeX || 0);
                 if (Math.abs(x) > 2) {
                   closeOpenSwipe();
                   return;
                 }
                 setSidebarOpen(false);
                 selectSession(s.session_id);
               };
	             } else {
	               card.classList.add("desktop");
	               const actions = el("div", { class: "sessionActionsInline" }, delBtn ? [renameBtn, dupBtn, delBtn] : [renameBtn, dupBtn]);
	               const titleWithBadges = el("div", { class: "sessionTitleWithBadges" }, [titleRow, badgesWrap]);
	               const main = el("div", { class: "sessionMain" }, [titleWithBadges, meta]);
	               const inner = el("div", { class: "sessionInner sessionDesktopLayout" }, [main, actions]);
	               card.appendChild(inner);
	               card.onclick = () => selectSession(s.session_id);
	             }

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
            updateQueueBadge();
          } else if (selected) {
            const s = sessionIndex.get(selected);
            if (s) titleLabel.textContent = sessionTitleWithId(s);
          }
          updateHarnessBtnState();
          updateQueueBadge();
          return sessions;
        }

        function appendEvent(ev) {
          if (!ev || (ev.role !== "user" && ev.role !== "assistant")) return;
          if (consumePendingUserIfMatches(ev)) return;
          if (!ev.pending && ev.role === "assistant") {
            const k = assistantTextKey(ev);
            if (k && k === lastAssistantKey) return;
          }
          if (isDuplicateEvent(ev)) return;

          const stick = autoScroll || isNearBottom();
          const ts = typeof ev.ts === "number" && Number.isFinite(ev.ts) ? ev.ts : ev.pending ? Date.now() / 1000 : null;
           const { row } = makeRow(ev, { ts, pending: Boolean(ev.pending) });
	          const anchor = typingRow && typingRow.isConnected ? typingRow : bottomSentinel;
	          chatInner.insertBefore(row, anchor);
            trimRenderedRows({ fromTop: true });
          rebuildDecorations({ preserveScroll: false });
            if (!ev.pending) markClickFirstPaint();
            if (!ev.pending && selected) {
              appendCacheEvents(selected, [ev]);
            }
          markEventSeen(ev);

          if (stick) {
            requestAnimationFrame(() => scrollToBottom());
            jumpBtn.style.display = "none";
          } else {
            jumpBtn.style.display = "inline-flex";
          }
          if (ev.role === "user") lastAssistantKey = "";
          else if (!ev.pending && ev.role === "assistant") lastAssistantKey = assistantTextKey(ev) || "";
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
            const data = await api(`/api/sessions/${sid}/messages?offset=0&init=1&limit=${olderPageLimit()}&before=${reqBefore}`);
            if (selected !== sid || pollGen !== gen) return;
            const evs = Array.isArray(data.events) ? data.events : [];
            if (evs.length) prependOlderEvents(evs);
            olderBefore = Number.isFinite(Number(data.next_before)) ? Number(data.next_before) : reqBefore;
            setOlderState({ hasMore: Boolean(data.has_older), isLoading: false });
            setCacheMeta(sid, { olderBefore, hasOlder: Boolean(data.has_older) });
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

         // Guardrail: startInitialRender is intended to backfill an empty chat view.
         // If it is invoked again (e.g., due to a race), clear existing rendered rows
         // to avoid duplicating the last assistant message.
         if (chatInner.querySelector(".msg-row:not(.typing-row)")) {
           for (const n of Array.from(chatInner.querySelectorAll(".day-sep"))) n.remove();
           for (const n of Array.from(chatInner.querySelectorAll(".msg-row:not(.typing-row)"))) n.remove();
         }

         const msgs = [];
         const initDedup = new Set();
         for (const ev of allEvents) {
               if (!ev || (ev.role !== "user" && ev.role !== "assistant")) continue;
               if (consumePendingUserIfMatches(ev)) continue;
             const k = eventKey(ev);
             if (k && initDedup.has(k)) continue;
             if (k) initDedup.add(k);
             const prev = msgs.length ? msgs[msgs.length - 1] : null;
             if (isAdjacentAssistantDuplicate(prev, ev)) continue;
               msgs.push(ev);
         }
           if (!msgs.length) return;
           lastAssistantKey = "";
           for (let i = msgs.length - 1; i >= 0; i--) {
             const ev = msgs[i];
             if (!ev) continue;
             if (ev.role === "user") break;
             if (ev.role === "assistant") {
               lastAssistantKey = assistantTextKey(ev) || "";
               break;
             }
           }
           if (selected) replaceCacheEvents(selected, msgs);
           recentEventKeys.length = 0;
           recentEventKeySet.clear();
           for (const ev of msgs) {
             markEventSeen(ev);
          }
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
                const d2 = await api(`/api/sessions/${sid}/messages?offset=0&init=1&limit=${initPageLimit()}&before=0`);
                if (gen !== pollGen || sid !== selected) return;
                if (d2 && typeof d2.log_path === "string") activeLogPath = d2.log_path;
                if (d2 && typeof d2.thread_id === "string") activeThreadId = d2.thread_id;
                offset = d2.offset;
                const evs2 = Array.isArray(d2.events) ? d2.events : [];
                if (evs2.length) startInitialRender(evs2);
                olderBefore = Number.isFinite(Number(d2.next_before)) ? Number(d2.next_before) : 0;
                setOlderState({ hasMore: Boolean(d2.has_older), isLoading: false });
                setCacheMeta(sid, {
                  logPath: activeLogPath,
                  offset,
                  olderBefore,
                  hasOlder: Boolean(d2.has_older),
                });
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
              setCacheMeta(sid, { logPath: activeLogPath || lp || null, offset });
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
            if (s) titleLabel.textContent = sessionTitleWithId(s);
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
              updateQueueBadge();
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
            updateQueueBadge();
            setStatus({ running: false, queueLen: 0 });
            setContext(null);
            setTyping(false);
            turnOpen = false;
		          {
		            const s = sessionIndex.get(sid);
            if (s) titleLabel.textContent = sessionTitleWithId(s);
            else titleLabel.textContent = sid ? String(sid) : "No session selected";
		          }
                clickLoadT0 = performance.now();
                clickMetricPending = true;
          if (pollGen !== myGen || selected !== sid) return;
			          const s0 = sessionIndex.get(sid);
			          if (s0 && s0.token) setContext(s0.token);
                const cached = getCache(sid);
                const hasCached = Boolean(
                  cached &&
                    Array.isArray(cached.events) &&
                    cached.events.length &&
                    Number(cached.offset) > 0
                );
                if (hasCached) {
                  activeLogPath = typeof cached.log_path === "string" ? cached.log_path : null;
                  offset = Number(cached.offset) || 0;
                  olderBefore = Number(cached.older_before) || 0;
                  setOlderState({ hasMore: Boolean(cached.has_older), isLoading: false });
                  startInitialRender(cached.events);
                  try {
                    await pollMessages(sid, myGen);
                  } catch {
                    // ignore and rely on next poll
                  }
                  if (pollGen !== myGen || selected !== sid) return;
                } else {
				            try {
						            const data = await api(`/api/sessions/${sid}/messages?offset=0&init=1&limit=${initPageLimit()}&before=0`);
					            if (pollGen !== myGen || selected !== sid) return;
                    if (data && typeof data.log_path === "string") activeLogPath = data.log_path;
                    if (data && typeof data.thread_id === "string") activeThreadId = data.thread_id;
				            offset = data.offset;
				            const evs = Array.isArray(data.events) ? data.events : [];
					            if (evs.length) startInitialRender(evs);
                    olderBefore = Number.isFinite(Number(data.next_before)) ? Number(data.next_before) : 0;
                    setOlderState({ hasMore: Boolean(data.has_older), isLoading: false });
                    setCacheMeta(sid, {
                      logPath: activeLogPath,
                      offset,
                      olderBefore,
                      hasOlder: Boolean(data.has_older),
                    });
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
          fileBtn.disabled = !selected;
          diagBtn.disabled = !selected;
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
        async function renameSessionId(sid) {
          if (!sid) return;
          const s = sessionIndex.get(sid);
          const currentAlias = s && typeof s.alias === "string" ? s.alias : "";
          const fallback = sessionDisplayName(s);
          const def = currentAlias || fallback || "";
          const next = prompt("Rename session (blank to clear):", def);
          if (next === null) return;
          try {
            const res = await api(`/api/sessions/${sid}/rename`, { method: "POST", body: { name: String(next) } });
            const alias = res && typeof res.alias === "string" ? res.alias : "";
            if (s) s.alias = alias;
            await refreshSessions();
            if (selected === sid) {
              const s2 = sessionIndex.get(sid);
              if (s2) titleLabel.textContent = sessionTitleWithId(s2);
            }
            setToast(alias ? "renamed" : "alias cleared");
          } catch (e) {
            setToast(`rename error: ${e && e.message ? e.message : "unknown error"}`);
          }
        }
        let fileViewMode = localStorage.getItem("codexweb.fileViewMode") || "diff"; // "diff" | "file"
        let fileCandidateList = [];
        let fileEntryMap = new Map();
        let activeFilePath = "";
        let fileMenuOpen = false;

        function extToPrismLang(p) {
          const ext = String(p || "").split(".").pop().toLowerCase();
          if (ext === "js") return "javascript";
          if (ext === "ts") return "typescript";
          if (ext === "json") return "json";
          if (ext === "py") return "python";
          if (ext === "sh" || ext === "bash" || ext === "zsh") return "bash";
          if (ext === "md") return "markdown";
          if (ext === "html" || ext === "htm") return "markup";
          if (ext === "css") return "css";
          if (ext === "yml" || ext === "yaml") return "yaml";
          if (ext === "toml") return "toml";
          if (ext === "rs") return "rust";
          if (ext === "go") return "go";
          if (ext === "java") return "java";
          if (ext === "c" || ext === "h") return "c";
          if (ext === "cpp" || ext === "cc" || ext === "hpp") return "cpp";
          return "";
        }

        function renderCodeHtml(line, lang) {
          const raw = String(line ?? "");
          if (raw === "") return "&nbsp;";
          const grammar = window.Prism && lang ? window.Prism.languages[lang] : null;
          if (grammar && window.Prism && typeof window.Prism.highlight === "function") {
            try {
              return window.Prism.highlight(raw, grammar, lang);
            } catch (_) {}
          }
          return escapeHtml(raw);
        }

        function renderTableHtml(rows) {
          const body = rows
            .map((row) => {
              if (row.kind === "Hunk") {
                return `<tr class="fileRow fileRowHunk"><td class="fileLineNo"></td><td class="fileLineNo"></td><td class="fileLineCell"><code class="fileLineCode">${escapeHtml(
                  row.text
                )}</code></td></tr>`;
              }
              const oldNo = row.oldNo == null ? "" : String(row.oldNo);
              const newNo = row.newNo == null ? "" : String(row.newNo);
              return `<tr class="fileRow fileRow${row.kind}"><td class="fileLineNo">${oldNo}</td><td class="fileLineNo">${newNo}</td><td class="fileLineCell"><code class="fileLineCode language-${escapeHtml(
                row.lang || "plain"
              )}">${row.html}</code></td></tr>`;
            })
            .join("");
          return `<table class="fileTable"><tbody>${body}</tbody></table>`;
        }

        function renderFullFileHtml(rel, text) {
          const source = String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
          const lines = source.split("\n");
          const lang = extToPrismLang(rel);
          const rows = lines
            .map((line, idx) => ({ kind: "Context", oldNo: idx + 1, newNo: idx + 1, html: renderCodeHtml(line, lang), lang }));
          return renderTableHtml(rows);
        }

        function renderDiffHtml(rel, diffText) {
          const lang = extToPrismLang(rel);
          const rows = [];
          let oldLine = 0;
          let newLine = 0;
          for (const raw of String(diffText || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n")) {
            if (raw.startsWith("@@")) {
              const m = raw.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
              oldLine = m ? Number(m[1]) : oldLine;
              newLine = m ? Number(m[2]) : newLine;
              rows.push({ kind: "Hunk", text: raw });
              continue;
            }
            if (raw.startsWith("diff --git") || raw.startsWith("index ") || raw.startsWith("--- ") || raw.startsWith("+++ ") || raw.startsWith("\\ No newline")) {
              continue;
            }
            if (raw.startsWith("+")) {
              rows.push({ kind: "Add", oldNo: null, newNo: newLine++, html: renderCodeHtml(raw.slice(1), lang), lang });
              continue;
            }
            if (raw.startsWith("-")) {
              rows.push({ kind: "Del", oldNo: oldLine++, newNo: null, html: renderCodeHtml(raw.slice(1), lang), lang });
              continue;
            }
            if (raw.startsWith(" ")) {
              rows.push({ kind: "Context", oldNo: oldLine++, newNo: newLine++, html: renderCodeHtml(raw.slice(1), lang), lang });
            }
          }
          if (!rows.length) {
            return `<div class="fileEmpty">No diff for ${escapeHtml(rel)}.</div>`;
          }
          return renderTableHtml(rows);
        }

        function applyFileMode() {
          const isDiff = fileViewMode === "diff";
          fileModeDiffBtn.classList.toggle("active", isDiff);
          fileDiff.style.display = "block";
        }

        function applyFileMenuState() {
          filePickerBtn.classList.toggle("active", fileMenuOpen);
          filePickerMenu.classList.toggle("open", fileMenuOpen);
        }

        function setFilePath(rel) {
          const next = String(rel || "").trim();
          activeFilePath = next;
          const entry = fileEntryMap.get(next);
          filePickerBtn.textContent = entry ? formatFileChoice(entry) : next || "Choose file";
          fileMenuOpen = false;
          applyFileMenuState();
        }

        function formatFileChoice(entry) {
          const rel = String(entry && entry.path ? entry.path : "");
          const changed = Boolean(entry && entry.changed);
          const add = entry && Object.prototype.hasOwnProperty.call(entry, "additions") ? entry.additions : null;
          const del = entry && Object.prototype.hasOwnProperty.call(entry, "deletions") ? entry.deletions : null;
          const stat = changed
            ? `  ${add == null ? "+?" : `+${add}`} ${del == null ? "-?" : `-${del}`}`
            : "  manual";
          return `${rel}${stat}`;
        }

        function upsertFileEntry(entry) {
          if (!entry || typeof entry.path !== "string" || !entry.path.trim()) return;
          const path = entry.path.trim();
          const merged = {
            path,
            additions: entry.additions ?? null,
            deletions: entry.deletions ?? null,
            changed: Boolean(entry.changed),
          };
          if (!fileEntryMap.has(path)) {
            fileCandidateList.push(path);
          }
          fileEntryMap.set(path, merged);
        }

        function renderFilePickerMenu() {
          filePickerMenu.innerHTML = "";
          if (!fileCandidateList.length) return;
          for (const path of fileCandidateList) {
            const entry = fileEntryMap.get(path) || { path, changed: false, additions: null, deletions: null };
            const btn = el("button", {
              class: "fileMenuItem" + (activeFilePath === path ? " active" : ""),
              type: "button",
              title: path,
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: path }));
            const statText = entry.changed
              ? `${entry.additions == null ? "+?" : `+${entry.additions}`} ${entry.deletions == null ? "-?" : `-${entry.deletions}`}`
              : "manual";
            const statCls = entry.changed ? "fileMenuStat changed" : "fileMenuStat manual";
            const stat = el("span", { class: statCls });
            if (entry.changed) {
              stat.appendChild(el("span", { class: "fileMenuAdd", text: entry.additions == null ? "+?" : `+${entry.additions}` }));
              stat.appendChild(el("span", { class: "fileMenuDel", text: entry.deletions == null ? "-?" : `-${entry.deletions}` }));
            } else {
              stat.textContent = statText;
            }
            btn.appendChild(stat);
            btn.onclick = () => {
              setFilePath(path);
              const selectedEntry = fileEntryMap.get(path);
              if (!selectedEntry || selectedEntry.changed) {
                fileViewMode = "diff";
              } else {
                fileViewMode = "file";
              }
              localStorage.setItem("codexweb.fileViewMode", fileViewMode);
              applyFileMode();
              renderFilePickerMenu();
              void openFilePath(path);
            };
            filePickerMenu.appendChild(btn);
          }
        }

        function sessionRelativePath(rawPath) {
          const s = selected ? sessionIndex.get(selected) : null;
          if (!s || !s.cwd) return null;
          const abs = stripPathLocationSuffix(rawPath);
          const cwd = String(s.cwd || "").replace(/\/+$/, "");
          if (!abs) return null;
          if (abs === cwd) return ".";
          if (abs.startsWith(cwd + "/")) return abs.slice(cwd.length + 1);
          return null;
        }

        async function refreshFileCandidates() {
          fileCandidateList = [];
          fileEntryMap = new Map();
          if (!selected) return;
          try {
            const res = await api(`/api/sessions/${selected}/git/changed_files`);
            const entriesIn = Array.isArray(res.entries) ? res.entries : [];
            const changedEntries = entriesIn
              .filter((entry) => entry && typeof entry.path === "string" && String(entry.path).trim())
              .map((entry) => ({
                path: String(entry.path).trim(),
                additions:
                  typeof entry.additions === "number" && Number.isFinite(entry.additions) ? entry.additions : entry.additions == null ? null : null,
                deletions:
                  typeof entry.deletions === "number" && Number.isFinite(entry.deletions) ? entry.deletions : entry.deletions == null ? null : null,
                changed: true,
              }));
            const s = selected ? sessionIndex.get(selected) : null;
            const manualEntries = listFromFilesField(s && s.files)
              .map((abs) => sessionRelativePath(abs))
              .filter((rel) => typeof rel === "string" && rel && rel !== ".")
              .map((path) => ({ path, additions: null, deletions: null, changed: false }));
            const merged = [];
            const seen = new Set();
            for (const entry of [...changedEntries, ...manualEntries]) {
              if (!entry || !entry.path || seen.has(entry.path)) continue;
              seen.add(entry.path);
              merged.push(entry);
            }
            fileCandidateList = merged.map((entry) => entry.path);
            for (const entry of merged) {
              fileEntryMap.set(entry.path, entry);
            }
          } catch (e) {}
          renderFilePickerMenu();
          setFilePath(activeFilePath || fileCandidateList[0] || "");
        }

        async function showFileViewer({ path = "", mode = "", manual = false } = {}) {
          fileBackdrop.style.display = "block";
          fileViewer.style.display = "flex";
          if (mode === "file" || mode === "diff") {
            fileViewMode = mode;
            localStorage.setItem("codexweb.fileViewMode", fileViewMode);
          }
          applyFileMode();
          await refreshFileCandidates();
          const preferred = String(path || "").trim() || activeFilePath || localStorage.getItem("codexweb.filePath") || "";
          if (preferred) {
            setFilePath(preferred);
            void openFilePath(preferred);
            return;
          }
          const first = fileCandidateList.length ? fileCandidateList[0] : "";
          if (first) {
            fileViewMode = "diff";
            localStorage.setItem("codexweb.fileViewMode", fileViewMode);
            applyFileMode();
            setFilePath(first);
            void openFilePath(first);
            return;
          }
          filePickerBtn.textContent = "No files";
          fileStatus.textContent = "No changed files. Use Add to open a file.";
        }
        function hideFileViewer() {
          fileBackdrop.style.display = "none";
          fileViewer.style.display = "none";
        }
        async function openFilePath(nextPath = null) {
          if (!selected) return;
          const rel = String(nextPath == null ? activeFilePath : nextPath).trim();
          if (!rel) {
            fileStatus.textContent = "Choose a file first.";
            return;
          }
          activeFilePath = rel;
          fileStatus.textContent = "Loading...";
          fileDiff.innerHTML = "";
          try {
            if (fileViewMode === "diff") {
              const res = await api(`/api/sessions/${selected}/git/diff?path=${encodeURIComponent(rel)}`);
              const diff = res && typeof res.diff === "string" ? res.diff : "";
              fileDiff.innerHTML = renderDiffHtml(rel, diff);
              fileStatus.textContent = diff ? rel : `${rel} - no diff`;
            } else {
              const res = await api(`/api/sessions/${selected}/file/read?path=${encodeURIComponent(rel)}`);
              if (!res || typeof res.text !== "string") throw new Error("invalid response");
              fileDiff.innerHTML = renderFullFileHtml(rel, res.text);
              const size = typeof res.size === "number" ? res.size : res.text.length;
              fileStatus.textContent = `${rel} - ${fmtBytes(size)}`;
              if (!fileEntryMap.has(rel)) upsertFileEntry({ path: rel, additions: null, deletions: null, changed: false });
              renderFilePickerMenu();
            }
            localStorage.setItem("codexweb.filePath", rel);
          } catch (e) {
            fileStatus.textContent = `error: ${e && e.message ? e.message : "unknown error"}`;
          }
        }
        fileBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void showFileViewer();
        };
        filePickerBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          fileMenuOpen = !fileMenuOpen;
          renderFilePickerMenu();
          applyFileMenuState();
        };
        fileModeDiffBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          fileViewMode = fileViewMode === "diff" ? "file" : "diff";
          localStorage.setItem("codexweb.fileViewMode", fileViewMode);
          applyFileMode();
          renderFilePickerMenu();
          void openFilePath(activeFilePath);
        };
        fileAddBtn.onclick = async (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!selected) return;
          const raw = window.prompt("Open file", activeFilePath || "");
          const path = String(raw || "").trim();
          if (!path) return;
          try {
            const res = await api("/api/files/read", { method: "POST", body: { session_id: selected, path } });
            const abs = typeof res.path === "string" ? res.path : path;
            const rel = sessionRelativePath(abs) || path;
            upsertFileEntry({ path: rel, additions: null, deletions: null, changed: false });
            renderFilePickerMenu();
            setFilePath(rel);
            fileViewMode = "file";
            localStorage.setItem("codexweb.fileViewMode", fileViewMode);
            applyFileMode();
            void openFilePath(rel);
          } catch (err) {
            setToast(`file open error: ${err && err.message ? err.message : "unknown error"}`);
          }
        };
        fileCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideFileViewer();
        };
        fileBackdrop.onclick = () => hideFileViewer();
        chatInner.addEventListener("click", async (e) => {
          const target = e.target instanceof Element ? e.target.closest("a[data-local-path]") : null;
          if (!target) return;
          e.preventDefault();
          const abs = String(target.getAttribute("data-local-path") || "").trim();
          const rel = sessionRelativePath(abs);
          if (rel) {
            void showFileViewer({ path: rel, mode: "file", manual: false });
            return;
          }
          const match = [...sessionIndex.values()].find((s) => {
            const cwd = String(s && s.cwd ? s.cwd : "").replace(/\/+$/, "");
            return cwd && (abs === cwd || abs.startsWith(cwd + "/"));
          });
          if (!match) {
            setToast("file is outside the known session roots");
            return;
          }
          await selectSession(match.session_id);
          const matchRoot = String(match.cwd || "").replace(/\/+$/, "");
          const rel2 = abs === matchRoot ? "." : abs.slice(matchRoot.length + 1);
          void showFileViewer({ path: rel2, mode: "file", manual: false });
        });
        document.addEventListener("click", (e) => {
          if (fileViewer.style.display !== "flex" || !fileMenuOpen) return;
          const t = e.target instanceof Element ? e.target : null;
          if (!t) return;
          if (t.closest("#fileCandRow")) return;
          fileMenuOpen = false;
          applyFileMenuState();
        });
        document.addEventListener("keydown", (e) => {
          if (e.key !== "Escape") return;
          if (fileViewer.style.display === "flex") hideFileViewer();
          if (sendChoice.style.display === "flex") hideSendChoice();
          if (queueViewer.style.display === "flex") hideQueueViewer();
          if (helpViewer.style.display === "flex") hideHelpViewer();
          if (diagViewer.style.display === "flex") hideDiagViewer();
        });

        let sendChoicePending = null;
        function showSendChoice(raw) {
          sendChoicePending = { sid: selected, text: raw };
          sendChoiceBackdrop.style.display = "block";
          sendChoice.style.display = "flex";
        }
        function hideSendChoice() {
          sendChoicePending = null;
          sendChoiceBackdrop.style.display = "none";
          sendChoice.style.display = "none";
        }
        const sendChoiceNowBtn = $("#sendChoiceNow");
        const sendChoiceLaterBtn = $("#sendChoiceLater");
        const sendChoiceCancelBtn = $("#sendChoiceCancel");
        if (sendChoiceNowBtn)
          sendChoiceNowBtn.onclick = async () => {
            const raw = sendChoicePending && sendChoicePending.text;
            const sid = sendChoicePending && sendChoicePending.sid;
            hideSendChoice();
            if (!raw || !sid) return;
            clearComposer();
            await sendText(raw, { sid });
          };
        if (sendChoiceLaterBtn)
          sendChoiceLaterBtn.onclick = async () => {
            const raw = sendChoicePending && sendChoicePending.text;
            const sid = sendChoicePending && sendChoicePending.sid;
            hideSendChoice();
            if (!raw || !sid) return;
              clearComposer();
              try {
                const res = await api(`/api/sessions/${sid}/enqueue`, { method: "POST", body: { text: raw } });
              const qn = res && typeof res.queue_len_total === "number" ? res.queue_len_total : res && typeof res.queue_len === "number" ? res.queue_len : null;
                if (res && res.queued) setToast(`queued (${qn ?? "?"})`);
                else setToast("sent");
                pollFastUntilMs = Date.now() + 5000;
                kickPoll(0);
                await refreshSessions();
              updateQueueBadge();
            } catch (e) {
              setToast(`queue error: ${e && e.message ? e.message : "unknown error"}`);
            }
          };
        if (sendChoiceCancelBtn)
          sendChoiceCancelBtn.onclick = () => {
            hideSendChoice();
          };
        sendChoiceBackdrop.onclick = () => hideSendChoice();

        const queueUpdateTimers = new Map();
        let queueLastEditMs = 0;
        let queueViewerSid = null;
        let queueViewerItems = [];

        function scheduleQueueUpdate(sid, idx, text) {
          if (!sid) return;
          if (!String(text || "").trim()) {
            const key0 = `${sid}:${idx}`;
            const existing0 = queueUpdateTimers.get(key0);
            if (existing0) clearTimeout(existing0);
            queueUpdateTimers.delete(key0);
            return;
          }
          const key = `${sid}:${idx}`;
          const existing = queueUpdateTimers.get(key);
          if (existing) clearTimeout(existing);
          const t = setTimeout(async () => {
            queueUpdateTimers.delete(key);
            try {
              await api(`/api/sessions/${sid}/queue/update`, { method: "POST", body: { index: idx, text } });
              await refreshSessions();
              updateQueueBadge();
            } catch (e) {
              setToast(`queue update error: ${e && e.message ? e.message : "unknown error"}`);
            }
          }, 350);
          queueUpdateTimers.set(key, t);
        }

        function renderQueueList() {
          queueList.innerHTML = "";
          const sid = queueViewerSid || selected;
          if (!sid) {
            queueEmpty.style.display = "block";
            return;
          }
          const q = Array.isArray(queueViewerItems) ? queueViewerItems : [];
          queueEmpty.style.display = q.length ? "none" : "block";
          if (!q.length) return;
          q.forEach((text, idx) => {
            const row = el("div", { class: "queueItem" });
            const ta = el("textarea", { class: "queueText", "aria-label": `Queued message ${idx + 1}` });
            ta.value = text;
            ta.oninput = () => {
              queueLastEditMs = Date.now();
              scheduleQueueUpdate(sid, idx, String(ta.value || ""));
            };
            const del = el("button", { class: "icon-btn danger", title: "Delete", "aria-label": "Delete", type: "button", html: iconSvg("trash") });
            del.onclick = async (e) => {
              e.preventDefault();
              e.stopPropagation();
              try {
                await api(`/api/sessions/${sid}/queue/delete`, { method: "POST", body: { index: idx } });
                await refreshQueueViewer();
                await refreshSessions();
                updateQueueBadge();
              } catch (e2) {
                setToast(`queue delete error: ${e2 && e2.message ? e2.message : "unknown error"}`);
              }
            };
            row.appendChild(ta);
            row.appendChild(del);
            queueList.appendChild(row);
          });
        }

        async function refreshQueueViewer() {
          const sid = queueViewerSid || selected;
          if (!sid) return;
          if (queueViewer.style.display === "flex" && Date.now() - queueLastEditMs < 900) return;
          queueEmpty.textContent = "Loading...";
          try {
            const data = await api(`/api/sessions/${sid}/queue`);
            if (queueViewerSid && queueViewerSid !== sid) return;
            const q = data && Array.isArray(data.queue) ? data.queue.filter((x) => typeof x === "string") : [];
            queueViewerSid = sid;
            queueViewerItems = q;
            queueEmpty.textContent = "No queued messages.";
            renderQueueList();
          } catch (e) {
            if (queueViewerSid && queueViewerSid !== sid) return;
            queueViewerSid = sid;
            queueViewerItems = [];
            queueEmpty.textContent = `Queue unavailable: ${e && e.message ? e.message : "unknown error"}`;
            setToast(`queue load error: ${e && e.message ? e.message : "unknown error"}`);
            renderQueueList();
          }
        }

        function showQueueViewer() {
          if (!selected) return;
          queueViewerSid = selected;
          queueBackdrop.style.display = "block";
          queueViewer.style.display = "flex";
          void refreshQueueViewer();
        }

        function hideQueueViewer() {
          queueBackdrop.style.display = "none";
          queueViewer.style.display = "none";
          queueViewerSid = null;
          queueViewerItems = [];
        }

        function showHelpViewer() {
          helpBackdrop.style.display = "block";
          helpViewer.style.display = "flex";
        }
        function hideHelpViewer() {
          helpBackdrop.style.display = "none";
          helpViewer.style.display = "none";
        }

        async function showDiagViewer() {
          if (!selected) return;
          diagContent.innerHTML = "";
          diagStatus.textContent = "Loading...";
          diagBackdrop.style.display = "block";
          diagViewer.style.display = "flex";
          try {
            const d = await api(`/api/sessions/${selected}/diagnostics`);
            diagStatus.textContent = "";
            const now = Date.now() / 1000;
            const addRow = (label, value, { mono = false } = {}) => {
              const v = value == null || value === "" ? "-" : String(value);
              const row = el("div", { class: "detailsRow" });
              row.appendChild(el("div", { class: "detailsLabel", text: String(label || "") }));
              row.appendChild(el("div", { class: mono ? "detailsValue mono" : "detailsValue", text: v }));
              diagContent.appendChild(row);
            };
            const age = (ts) => {
              const t = Number(ts);
              if (!Number.isFinite(t) || t <= 0) return "";
              const s = Math.max(0, Math.floor(now - t));
              const a = fmtIdleAge(s);
              return a ? `${a} ago` : "";
            };
            addRow("Session", d && d.session_id ? d.session_id : "-");
            addRow("Thread", d && d.thread_id ? d.thread_id : "-");
            addRow("Owned", d && typeof d.owned === "boolean" ? (d.owned ? "web" : "terminal") : "-");
            addRow("Busy", d && typeof d.busy === "boolean" ? (d.busy ? "busy" : "idle") : "-");
            addRow("Queue", d && typeof d.queue_len === "number" ? String(d.queue_len) : "-");
            addRow("CWD", d && d.cwd ? d.cwd : "-", { mono: true });
            addRow("Started", d && typeof d.start_ts === "number" ? `${fmtTs(d.start_ts)}${age(d.start_ts) ? " (" + age(d.start_ts) + ")" : ""}` : "-");
            addRow(
              "Updated",
              d && typeof d.updated_ts === "number" ? `${fmtTs(d.updated_ts)}${age(d.updated_ts) ? " (" + age(d.updated_ts) + ")" : ""}` : "-"
            );
            addRow("Broker PID", d && typeof d.broker_pid === "number" ? String(d.broker_pid) : "-");
            addRow("Codex PID", d && typeof d.codex_pid === "number" ? String(d.codex_pid) : "-");
            addRow("Log", d && d.log_path ? d.log_path : "-", { mono: true });
	            addRow("Provider", d && d.model_provider ? d.model_provider : "-");
	            addRow("Model", d && d.model ? d.model : "-");
	            addRow("Reasoning", d && d.reasoning_effort ? d.reasoning_effort : "-");
	            addRow("UI", UI_VERSION);
	            const tok = d && d.token && typeof d.token === "object" ? d.token : null;
	            if (tok) {
	              const ctx = Number(tok.context_window);
	              const used = Number(tok.tokens_in_context);
	              const pct = Number(tok.percent_remaining);
              if (Number.isFinite(ctx) && Number.isFinite(used) && ctx > 0 && used >= 0) {
                const p = Number.isFinite(pct) ? Math.max(0, Math.min(100, Math.round(pct))) : null;
                const txt = p === null ? `${used}/${ctx}` : `${used}/${ctx} (${p}% left)`;
                addRow("Context", txt);
              }
            }
          } catch (e) {
            diagStatus.textContent = `error: ${e && e.message ? e.message : "unknown error"}`;
          }
        }
        function hideDiagViewer() {
          diagBackdrop.style.display = "none";
          diagViewer.style.display = "none";
        }

        const queueBtn = $("#queueBtn");
        if (queueBtn) {
          queueBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const raw = $("#msg") ? $("#msg").value : "";
            if (raw && raw.trim()) {
              if (!selected) return;
              const sid = selected;
              clearComposer();
              void (async () => {
                try {
                  const res = await api(`/api/sessions/${sid}/enqueue`, { method: "POST", body: { text: raw } });
                  const qn =
                    res && typeof res.queue_len_total === "number"
                      ? res.queue_len_total
                      : res && typeof res.queue_len === "number"
                        ? res.queue_len
                        : null;
                  if (res && res.queued) setToast(`queued (${qn ?? "?"})`);
                  else setToast("sent");
                  pollFastUntilMs = Date.now() + 5000;
                  kickPoll(0);
                  await refreshSessions();
                  updateQueueBadge();
                } catch (e2) {
                  setToast(`queue error: ${e2 && e2.message ? e2.message : "unknown error"}`);
                }
              })();
              return;
            }
            showQueueViewer();
          };
        }
        queueCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideQueueViewer();
        };
        queueBackdrop.onclick = () => hideQueueViewer();

        $("#helpBtnSide").onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          showHelpViewer();
        };
        helpCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideHelpViewer();
        };
        helpBackdrop.onclick = () => hideHelpViewer();

        diagBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          void showDiagViewer();
        };
        diagCloseBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideDiagViewer();
        };
        diagBackdrop.onclick = () => hideDiagViewer();
        async function spawnSessionWithCwd(cwd) {
          if (!cwd || !String(cwd).trim()) {
            setToast("cwd unavailable");
            return null;
          }
          try {
            setToast("starting...");
            const res = await api("/api/sessions", { method: "POST", body: { cwd: String(cwd) } });
            const brokerPid = res && res.broker_pid ? Number(res.broker_pid) : null;
            if (!brokerPid) {
              setToast("start failed");
              return null;
            }
            setToast(`started (broker ${brokerPid})`);
            for (let i = 0; i < 60; i++) {
              const sessions = await refreshSessions();
              const found = (sessions || []).find((x) => Number(x.broker_pid || 0) === brokerPid);
              if (found) {
                selectSession(found.session_id);
                return brokerPid;
              }
              await new Promise((r) => setTimeout(r, 250));
            }
            setToast("session will appear once Codex creates a rollout log");
            return brokerPid;
          } catch (e) {
            setToast(`start error: ${e.message}`);
            return null;
          }
        }
        $("#newBtn").onclick = async () => {
          const cur = selected ? sessionIndex.get(selected) : null;
          const def = cur && cur.cwd && cur.cwd !== "?" ? cur.cwd : "";
          const cwd = prompt("New session cwd:", def);
          if (!cwd) return;
          await spawnSessionWithCwd(cwd);
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
         const isIOS =
           /iP(hone|od|ad)/.test(navigator.userAgent || "") ||
           (navigator.platform === "MacIntel" && navigator.maxTouchPoints && navigator.maxTouchPoints > 1);
	         let iosViewportGuardTimer = null;
	         let iosViewportGuardUntil = 0;
	         function normalizePageScroll() {
	           if (!isIOS) return;
	           const y = window.scrollY || document.documentElement.scrollTop || document.body.scrollTop || 0;
	           if (y <= 0) return;
	           window.scrollTo(0, 0);
	           document.documentElement.scrollTop = 0;
	           document.body.scrollTop = 0;
	         }
	         function stopIOSViewportGuard() {
	           if (iosViewportGuardTimer) clearTimeout(iosViewportGuardTimer);
	           iosViewportGuardTimer = null;
	           iosViewportGuardUntil = 0;
	         }
	         function isIOSViewportGuardActive() {
	           return isIOS && Date.now() < iosViewportGuardUntil;
	         }
	         function runIOSViewportGuard({ preserveChatBottom, durationMs = 1400 } = {}) {
	           if (!isIOS) return;
	           stopIOSViewportGuard();
	           iosViewportGuardUntil = Date.now() + Math.max(0, Number(durationMs) || 0);
	           const tick = () => {
	             updateAppHeightVar();
	             normalizePageScroll();
	             if (preserveChatBottom && (autoScroll || isNearBottom())) scrollToBottom();
	             if (!isIOSViewportGuardActive()) {
	               iosViewportGuardTimer = null;
	               return;
	             }
	             iosViewportGuardTimer = setTimeout(tick, 50);
	           };
	           tick();
	         }
	         if (window.visualViewport) {
	           const onViewportShift = () => {
	             updateAppHeightVar();
	             if (!isIOS) return;
	             if (document.activeElement === textarea || isIOSViewportGuardActive()) {
	               normalizePageScroll();
	               if (autoScroll || isNearBottom()) requestAnimationFrame(() => scrollToBottom());
	             }
	           };
	           window.visualViewport.addEventListener("resize", onViewportShift);
	           window.visualViewport.addEventListener("scroll", onViewportShift);
	         }
         const attachBtn = $("#attachBtn");
         if (!attachBadgeEl) {
           attachBadgeEl = el("span", { class: "attachBadge", id: "attachBadge" });
           attachBtn.appendChild(attachBadgeEl);
         }
        if (!queueBadgeEl && queueBtn) {
          queueBadgeEl = el("span", { class: "attachBadge queueBadge", id: "queueBadge" });
          queueBtn.appendChild(queueBadgeEl);
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
        updateQueueBadge();
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
	              const wasNear = isNearBottom();
              if (wasNear) {
                autoScroll = true;
                jumpBtn.style.display = "none";
              }
	              if (isIOS) runIOSViewportGuard({ preserveChatBottom: wasNear, durationMs: 1800 });
	              else {
	                const tick = () => {
	                  updateAppHeightVar();
	                  if (wasNear) scrollToBottom();
	                };
	                requestAnimationFrame(tick);
	                setTimeout(tick, 120);
	              }
	            },
	            { passive: true }
	          );
	          textarea.addEventListener(
	            "blur",
	            () => {
	              if (isIOS) runIOSViewportGuard({ preserveChatBottom: false, durationMs: 900 });
	              else setTimeout(updateAppHeightVar, 0);
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

        function clearComposer() {
          $("#msg").value = "";
          autoGrow();
        }

        async function sendText(raw, { sid = null } = {}) {
          const sessionId = sid || selected;
          if (!sessionId) return;
          if (!raw || !raw.trim()) return;
          if (sending) return;
          sending = true;
          $("#sendBtn").disabled = true;
          setToast("sending...");

          const localId = ++localEchoSeq;
          const t0 = Date.now() / 1000;
          const renderHere = sessionId === selected;
          if (renderHere) {
            pendingUser.push({ id: localId, key: pendingMatchKey(raw), loose: normalizeTextForPendingMatch(raw), t0, text: raw });
            appendEvent({ role: "user", text: raw, pending: true, localId, ts: t0 });
            turnOpen = true;
            currentRunning = true;
          }
          try {
            const res = await api(`/api/sessions/${sessionId}/send`, { method: "POST", body: { text: raw } });
            if (res.queued) setToast(`queued (queue ${res.queue_len})`);
            else setToast("sent");
            setAttachCount(0);
            pollFastUntilMs = Date.now() + 5000;
            kickPoll(0);
            await refreshSessions();
          } catch (e2) {
            setToast(`send error: ${e2.message}`);
            if (renderHere) {
              const pendingEl = chatInner.querySelector(`.msg.user[data-local-id="${localId}"]`);
              if (pendingEl) {
                pendingEl.style.opacity = "1";
                pendingEl.style.borderColor = "rgba(185, 28, 28, 0.7)";
                pendingEl.style.boxShadow = "0 0 0 2px rgba(185, 28, 28, 0.12)";
              }
            }
          } finally {
            sending = false;
            $("#sendBtn").disabled = false;
          }
        }

        form.onsubmit = async (e) => {
          e.preventDefault();
          if (!selected) return;
          const raw = $("#msg").value;
          if (!raw || !raw.trim()) return;
          if (sending) return;
          if (currentRunning) {
            showSendChoice(raw);
            return;
          }
          clearComposer();
          await sendText(raw);
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
