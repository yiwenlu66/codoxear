	      const $ = (q) => document.querySelector(q);
	      const UI_VERSION = "20260309.9";
	      function isTextEntryElement(target) {
	        const el = target instanceof Element ? target.closest("textarea, input, [contenteditable], [contenteditable=''], [contenteditable='true']") : null;
	        if (!(el instanceof HTMLElement)) return false;
	        if (el.tagName !== "INPUT") return true;
	        const type = String(el.getAttribute("type") || "text").toLowerCase();
	        return !["button", "checkbox", "color", "file", "hidden", "image", "radio", "range", "reset", "submit"].includes(type);
	      }
	      function updateAppHeightVar() {
	        const vv = window.visualViewport;
	        const layoutH = Math.round(window.innerHeight);
	        const visualH = Math.round(vv ? vv.height : window.innerHeight);
	        const visualTop = Math.max(0, Math.round(vv ? vv.offsetTop : 0));
	        const visualBottom = Math.max(0, layoutH - visualH - visualTop);
	        if (updateAppHeightVar._h === visualH && updateAppHeightVar._l === layoutH && updateAppHeightVar._t === visualTop && updateAppHeightVar._b === visualBottom) return;
	        updateAppHeightVar._h = visualH;
	        updateAppHeightVar._l = layoutH;
	        updateAppHeightVar._t = visualTop;
	        updateAppHeightVar._b = visualBottom;
	        document.documentElement.style.setProperty("--appH", `${visualH}px`);
	        document.documentElement.style.setProperty("--layoutH", `${layoutH}px`);
	        document.documentElement.style.setProperty("--vvTop", `${visualTop}px`);
	        document.documentElement.style.setProperty("--vvBottom", `${visualBottom}px`);
	      }
	      updateAppHeightVar();
	      window.addEventListener("resize", updateAppHeightVar);
      // Best-effort zoom disable (iOS Safari still has edge cases).
      document.addEventListener(
        "gesturestart",
        (e) => {
          if (isTextEntryElement(e.target)) return;
          e.preventDefault();
        },
        { passive: false }
      );
      document.addEventListener(
        "gesturechange",
        (e) => {
          if (isTextEntryElement(e.target)) return;
          e.preventDefault();
        },
        { passive: false }
      );
      document.addEventListener(
        "gestureend",
        (e) => {
          if (isTextEntryElement(e.target)) return;
          e.preventDefault();
        },
        { passive: false }
      );
      document.addEventListener(
        "touchstart",
        (e) => {
          if (isTextEntryElement(e.target)) return;
          if (e.touches && e.touches.length > 1) e.preventDefault();
        },
        { passive: false }
      );
      document.addEventListener(
        "touchmove",
        (e) => {
          if (isTextEntryElement(e.target)) return;
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

      const CLICKABLE_FILE_EXTENSIONS = new Set([
        "bash",
        "c",
        "cc",
        "cfg",
        "conf",
        "cpp",
        "css",
        "csv",
        "gif",
        "go",
        "h",
        "hpp",
        "html",
        "htm",
        "ico",
        "ini",
        "java",
        "jpeg",
        "jpg",
        "js",
        "json",
        "jsonl",
        "log",
        "md",
        "patch",
        "png",
        "py",
        "rs",
        "scss",
        "sh",
        "sql",
        "svg",
        "toml",
        "ts",
        "tsx",
        "txt",
        "webp",
        "xml",
        "yaml",
        "yml",
        "zsh",
      ]);

      function filePathExtension(path) {
        const last = String(path || "").split("/").pop() || "";
        const idx = last.lastIndexOf(".");
        if (idx <= 0 || idx === last.length - 1) return "";
        return last.slice(idx + 1).toLowerCase();
      }

      function hasClickableFileExtension(path) {
        const ext = filePathExtension(path);
        return ext ? CLICKABLE_FILE_EXTENSIONS.has(ext) : false;
      }

      function normalizeLineNumber(value) {
        const n = Number(value);
        if (!Number.isFinite(n) || n <= 0) return null;
        return Math.max(1, Math.floor(n));
      }

      function formatPriorityOffset(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return "0.00";
        return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
      }

      function parseFileLocation(rawValue) {
        const raw = String(rawValue || "").trim();
        if (!raw) return null;
        let path = raw;
        let line = null;
        let m = path.match(/^(.*)#L(\d+)(?:-\d+)?$/);
        if (m) {
          path = m[1];
          line = normalizeLineNumber(m[2]);
        } else {
          m = path.match(/^(.*):(\d+)(?::\d+)?$/);
          if (m && !/^[A-Za-z]:$/.test(m[1])) {
            path = m[1];
            line = normalizeLineNumber(m[2]);
          }
        }
        path = String(path || "").trim();
        if (!path) return null;
        return { path, line };
      }

      function stripPathLocationSuffix(rawPath) {
        const parsed = parseFileLocation(rawPath);
        return parsed ? parsed.path : String(rawPath || "").trim();
      }

      function parseLocalFileRef(rawValue) {
        const parsed = parseFileLocation(rawValue);
        if (!parsed) return null;
        const path = parsed.path;
        if (!path) return null;
        if (path.includes("://") || path.startsWith("mailto:")) return null;
        if (path.startsWith("//")) return null;
        const looksAbsolute = path.startsWith("/");
        const looksRelative = path.startsWith("./") || path.startsWith("../") || path.includes("/");
        const looksBareFile = !looksAbsolute && !looksRelative && hasClickableFileExtension(path);
        if (!looksAbsolute && !looksRelative && !looksBareFile) return null;
        return { path, line: parsed.line };
      }

      function normalizePathLike(rawPath) {
        const raw = String(rawPath || "").trim();
        if (!raw) return "";
        const absolute = raw.startsWith("/");
        const parts = raw.split("/");
        const out = [];
        for (const part of parts) {
          if (!part || part === ".") continue;
          if (part === "..") {
            if (out.length && out[out.length - 1] !== "..") {
              out.pop();
              continue;
            }
            if (!absolute) out.push("..");
            continue;
          }
          out.push(part);
        }
        const joined = out.join("/");
        if (absolute) return joined ? `/${joined}` : "/";
        return joined || ".";
      }

      function pathDirname(rawPath) {
        const raw = normalizePathLike(rawPath);
        if (!raw || raw === ".") return ".";
        if (raw === "/") return "/";
        const idx = raw.lastIndexOf("/");
        if (idx < 0) return ".";
        if (idx === 0) return "/";
        return raw.slice(0, idx);
      }

      function resolveRelativePath(basePath, rawPath) {
        const raw = String(rawPath || "").trim();
        if (!raw) return "";
        if (raw.startsWith("/")) return normalizePathLike(raw);
        const baseDir = pathDirname(basePath);
        if (!baseDir || baseDir === ".") return normalizePathLike(raw);
        if (baseDir === "/") return normalizePathLike(`/${raw}`);
        return normalizePathLike(`${baseDir}/${raw}`);
      }

      function resolveLocalRefWithOptions(ref, options) {
        if (!ref || !options || typeof options.resolveLocalRef !== "function") return ref;
        const next = options.resolveLocalRef({ path: ref.path, line: ref.line });
        if (!next || typeof next.path !== "string" || !next.path.trim()) return ref;
        return { path: next.path.trim(), line: normalizeLineNumber(next.line ?? ref.line) };
      }

      function localFileRefFromRef(u, options = null) {
        const raw = String(u ?? "").trim();
        if (!raw) return null;
        const direct = parseLocalFileRef(raw);
        if (direct) return resolveLocalRefWithOptions(direct, options);
        try {
          const url = new URL(raw, location.href);
          if (url.origin !== location.origin) return null;
          const combined = `${decodeURIComponent(url.pathname || "")}${url.hash || ""}`;
          const parsed = parseLocalFileRef(combined);
          if (!parsed) return null;
          if (parsed.path.startsWith("/") && /^\/(?:home|tmp|mnt|var|opt|usr|etc|private|Users|Volumes)\//.test(parsed.path)) {
            return resolveLocalRefWithOptions(parsed, options);
          }
        } catch {}
        return null;
      }

      function fileLocationDisplaySuffix(rawRef, lineNumber) {
        const line = normalizeLineNumber(lineNumber);
        if (!line) return "";
        const raw = String(rawRef ?? "").trim();
        if (/#L\d+(?:-\d+)?$/i.test(raw)) return `#L${line}`;
        if (/:\d+(?::\d+)?$/.test(raw)) return `:${line}`;
        return `#L${line}`;
      }

      function formatLocalFileLinkLabel(label, rawRef, localRef) {
        const text = String(label ?? "");
        if (!localRef || !localRef.line) return text;
        const parsedLabel = parseFileLocation(text);
        if (parsedLabel && parsedLabel.line) return text;
        if (!parseLocalFileRef(text)) return text;
        return `${text}${fileLocationDisplaySuffix(rawRef, localRef.line)}`;
      }

      function renderInlineText(rawText, options = null) {
        const raw = String(rawText ?? "");
        const re =
          /(^|[\s([{"'])((?:\/[A-Za-z0-9._~@%+=:,/-]+|(?:\.{1,2}\/)?[A-Za-z0-9._~@-]+(?:\/[A-Za-z0-9._~@-]+)+|[A-Za-z0-9._~@-]+\.[A-Za-z0-9._-]+)(?:#L\d+(?:-\d+)?)?(?::\d+(?::\d+)?)?)(?=$|[\s)\]}:;"',!?])/g;
        let out = "";
        let last = 0;
        for (;;) {
          const m = re.exec(raw);
          if (!m) break;
          const wholeStart = m.index;
          const tokenStart = wholeStart + m[1].length;
          const token = m[2];
          out += escapeHtml(raw.slice(last, tokenStart));
          const ref = resolveLocalRefWithOptions(parseLocalFileRef(token), options);
          if (ref) {
            out += `<span data-candidate-file-path="${escapeHtml(ref.path)}"${ref.line ? ` data-candidate-file-line="${ref.line}"` : ""}>${escapeHtml(token)}</span>`;
          } else {
            out += escapeHtml(token);
          }
          last = tokenStart + token.length;
        }
        out += escapeHtml(raw.slice(last));
        return out;
      }

      function renderInlineMd(s, options = null) {
        const raw = String(s ?? "");
        const re = /!\[([^\]]*)\]\(([^)]+)\)|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*/g;
        let out = "";
        let last = 0;
        for (;;) {
          const m = re.exec(raw);
          if (!m) break;
          out += renderInlineText(raw.slice(last, m.index), options);
          if (m[1] !== undefined) {
            const imageAlt = m[1];
            const imageRef = m[2];
            const localImageRef = localFileRefFromRef(imageRef, options);
            const imageSrc =
              options && typeof options.resolveImageSrc === "function"
                ? options.resolveImageSrc(imageRef, localImageRef)
                : safeUrl(imageRef);
            if (!imageSrc) out += `![${escapeHtml(imageAlt)}](${escapeHtml(imageRef)})`;
            else out += `<img src="${escapeHtml(imageSrc)}" alt="${escapeHtml(imageAlt)}" loading="lazy" />`;
          } else if (m[3] !== undefined) {
            const inlineRef = resolveLocalRefWithOptions(parseLocalFileRef(m[3]), options);
            if (inlineRef) {
              out += `<code><span data-candidate-file-path="${escapeHtml(inlineRef.path)}"${inlineRef.line ? ` data-candidate-file-line="${inlineRef.line}"` : ""}>${escapeHtml(m[3])}</span></code>`;
            } else {
              out += `<code>${escapeHtml(m[3])}</code>`;
            }
          } else if (m[4] !== undefined) {
            const localRef = localFileRefFromRef(m[5], options);
            if (localRef) {
              out += `<span data-candidate-file-path="${escapeHtml(localRef.path)}"${localRef.line ? ` data-candidate-file-line="${localRef.line}"` : ""}>${escapeHtml(formatLocalFileLinkLabel(m[4], m[5], localRef))}</span>`;
            } else {
              const href = safeUrl(m[5]);
              if (!href) out += `${escapeHtml(m[4])} (${escapeHtml(m[5])})`;
              else out += `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer noopener">${escapeHtml(m[4])}</a>`;
            }
          } else if (m[6] !== undefined) {
            out += `<strong>${escapeHtml(m[6])}</strong>`;
          } else {
            out += escapeHtml(m[0]);
          }
          last = m.index + m[0].length;
        }
        out += renderInlineText(raw.slice(last), options);
        return out;
      }

      function mdToHtml(src, options = null) {
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
            out.push(renderInlineMd(it.text || "", options));
            if (it.child) out.push(renderList(it.child));
            out.push("</li>");
          }
          out.push(node.type === "ol" ? "</ol>" : "</ul>");
          return out.join("");
        };

        const splitTableCells = (line) => {
          let text = String(line ?? "").trim();
          if (!text.includes("|")) return [];
          if (text.startsWith("|")) text = text.slice(1);
          if (text.endsWith("|")) text = text.slice(0, -1);
          const cells = [];
          let cell = "";
          let escaped = false;
          for (const ch of text) {
            if (escaped) {
              cell += ch;
              escaped = false;
              continue;
            }
            if (ch === "\\") {
              escaped = true;
              continue;
            }
            if (ch === "|") {
              cells.push(cell.trim());
              cell = "";
              continue;
            }
            cell += ch;
          }
          if (escaped) cell += "\\";
          cells.push(cell.trim());
          return cells;
        };

        const parseTableAlignmentRow = (line) => {
          const cells = splitTableCells(line);
          if (!cells.length) return null;
          const alignments = [];
          for (const cell of cells) {
            const compact = String(cell ?? "").replace(/\s+/g, "");
            if (!/^:?-{3,}:?$/.test(compact)) return null;
            if (compact.startsWith(":") && compact.endsWith(":")) alignments.push("center");
            else if (compact.endsWith(":")) alignments.push("right");
            else if (compact.startsWith(":")) alignments.push("left");
            else alignments.push("");
          }
          return alignments;
        };

        const parseTable = (lines, start) => {
          if (start + 1 >= lines.length) return null;
          const headerLine = lines[start] || "";
          const separatorLine = lines[start + 1] || "";
          if (!headerLine.includes("|") || !separatorLine.includes("|")) return null;
          const headers = splitTableCells(headerLine);
          const alignments = parseTableAlignmentRow(separatorLine);
          if (!headers.length || !alignments || headers.length !== alignments.length) return null;
          const rows = [];
          let i = start + 2;
          while (i < lines.length) {
            const line = lines[i] || "";
            if (!line.trim() || !line.includes("|")) break;
            if (parseTableAlignmentRow(line)) break;
            const cells = splitTableCells(line);
            if (cells.length !== headers.length) break;
            rows.push(cells);
            i += 1;
          }
          return { node: { headers, alignments, rows }, next: i };
        };

        const renderTableCell = (tag, text, alignment) => {
          const alignAttr = alignment ? ` style="text-align:${alignment}"` : "";
          return `<${tag}${alignAttr}>${renderInlineMd(text || "", options)}</${tag}>`;
        };

        const renderTable = (node) => {
          const out = [];
          out.push('<div class="md-table-wrap"><table>');
          out.push("<thead><tr>");
          for (let i = 0; i < node.headers.length; i++) {
            out.push(renderTableCell("th", node.headers[i], node.alignments[i]));
          }
          out.push("</tr></thead>");
          out.push("<tbody>");
          for (const row of node.rows) {
            out.push("<tr>");
            for (let i = 0; i < row.length; i++) {
              out.push(renderTableCell("td", row[i], node.alignments[i]));
            }
            out.push("</tr>");
          }
          out.push("</tbody></table></div>");
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
              out.push(`<h${level}>${renderInlineMd(mHeading[2], options)}</h${level}>`);
              startIdx = 1;
            }

            let paraLines = [];
            const flushPara = () => {
              const para = paraLines.join("\n").trim();
              paraLines = [];
              if (!para) return;
              out.push(`<p>${renderInlineMd(para, options).replaceAll("\n", "<br />")}</p>`);
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
              const table = parseTable(lines, i);
              if (table) {
                flushPara();
                out.push(renderTable(table.node));
                i = table.next - 1;
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

      function isMarkdownPreviewable(path) {
        const ext = filePathExtension(path);
        return ext === "md" || ext === "markdown" || ext === "mdown" || ext === "mkd";
      }

      function previewImageUrlForRef(rawRef, localRef, { filePath, sessionId } = {}) {
        if (localRef && localRef.path) {
          if (sessionId) return resolveAppUrl(`/api/sessions/${sessionId}/file/blob?path=${encodeURIComponent(localRef.path)}`);
          if (localRef.path.startsWith("/")) return resolveAppUrl(`/api/files/blob?path=${encodeURIComponent(localRef.path)}`);
        }
        const safe = safeUrl(rawRef);
        return safe || null;
      }

      function markdownPreviewHtml(src, { filePath = "", sessionId = "" } = {}) {
        const basePath = String(filePath || "").trim();
        const sid = String(sessionId || "").trim();
        return mdToHtml(src, {
          resolveLocalRef(ref) {
            if (!ref || typeof ref.path !== "string") return ref;
            return { path: resolveRelativePath(basePath, ref.path), line: ref.line };
          },
          resolveImageSrc(rawRef, localRef) {
            return previewImageUrlForRef(rawRef, localRef, { filePath: basePath, sessionId: sid });
          },
        });
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
        if (name === "download")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>`;
        if (name === "preview")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z"/><circle cx="12" cy="12" r="2.5"/></svg>`;
        if (name === "diff")
          return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 4v16"/><path d="M17 4v16"/><path d="M4 7h6"/><path d="M14 17h6"/><path d="M14 7h6"/><path d="M4 17h6"/></svg>`;
        if (name === "chevronDown")
          return `<svg class="icon pickerChevronIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>`;
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
        let activeFileLine = null;
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
	         let openSwipeSessionId = null;
	         let openSwipeTargetX = 0;
	         let swipeRefreshDeferred = false;
        const cacheBySession = new Map();
        const cacheLoaded = new Set();
        const cacheSaveTimers = new Map();
	        let sessionIndex = new Map(); // session_id -> session info
	        let sending = false;
	        let localEchoSeq = 0;
	        const pendingUser = [];
	        let attachedFiles = 0;
		        let autoScroll = true;
			        let backfillToken = 0;
        let backfillState = null;
			    let lastScrollTop = 0;
				    let lastToken = null;
				    let typingRow = null;
        let attachBadgeEl = null;
        let queueBadgeEl = null;
        const fileRefValidationCache = new Map();
        const fileRefValidationPending = new Map();
        const fileRefCandidateCache = new Map();
        let editDependencyMenuOpen = false;
        let newSessionRecentMenuOpen = false;
        let newSessionResumeMenuOpen = false;
        let newSessionResumeCandidates = [];
        let newSessionResumeSelection = null;
        let newSessionResumeLoadSeq = 0;
        let newSessionResumeLoadTimer = null;
        let newSessionCwdInfo = { git_repo: false, git_root: "", git_branch: "" };
        const recentEventKeys = [];
         const recentEventKeySet = new Set();
         const RECENT_EVENT_KEYS_MAX = 320;
         let lastAssistantKey = "";
                 let clickLoadT0 = 0;
                 let clickMetricPending = false;
              let harnessMenuOpen = false;
              let harnessCfg = { enabled: false, request: "" };
              let harnessSaveTimer = null;
              let editSessionId = null;

            const titleLabel = el("div", { id: "threadTitle", text: "No session selected" });
            titleLabel.style.cursor = "pointer";
            titleLabel.title = "Edit conversation";
            titleLabel.onclick = () => {
              if (!selected) return;
              openEditSession(selected);
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
            title: "Attach file",
            "aria-label": "Attach file",
            html: iconSvg("paperclip"),
          }),
          el("div", { class: "inputWrap" }, [
            el("textarea", { id: "msg", placeholder: "", "aria-label": "Enter your instructions here" }),
            el("div", { class: "ph", id: "msgPh", text: "Enter your instructions here" }),
          ]),
          el("input", { id: "imgInput", type: "file", style: "display:none" }),
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
          class: "icon-btn",
          type: "button",
          title: "Toggle diff",
          "aria-label": "Toggle diff",
          html: iconSvg("diff"),
        });
        const fileModePreviewBtn = el("button", {
          id: "fileModePreviewBtn",
          class: "icon-btn",
          type: "button",
          title: "Toggle markdown preview",
          "aria-label": "Toggle markdown preview",
          html: iconSvg("preview"),
        });
        const fileDownloadBtn = el("button", {
          id: "fileDownloadBtn",
          class: "icon-btn",
          type: "button",
          title: "Download file",
          "aria-label": "Download file",
          html: iconSvg("download"),
        });
        const fileAddBtn = el("button", {
          id: "fileAddBtn",
          class: "icon-btn",
          type: "button",
          title: "Add file",
          "aria-label": "Add file",
          html: iconSvg("plus"),
        });
        const fileDiff = el("div", { class: "fileDiff", id: "fileDiff" });
        const fileImage = el("img", { id: "fileImage", class: "fileImage", alt: "" });
        const fileViewer = el("div", { class: "fileViewer", id: "fileViewer", role: "dialog", "aria-label": "File viewer" }, [
          el("div", { class: "fileViewerHeader" }, [
            el("div", { class: "title", text: "View file" }),
            el("div", { class: "actions" }, [fileModeDiffBtn, fileModePreviewBtn, fileDownloadBtn, fileAddBtn, fileCloseBtn]),
          ]),
          el("div", { class: "fileCandRow", id: "fileCandRow" }, [filePickerBtn, filePickerMenu]),
          fileStatus,
          fileDiff,
          fileImage,
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
  <li>On mobile: swipe left on a session to reveal <b>Edit</b> and <b>Duplicate</b>.</li>
  <li>On mobile: swipe right on any session to reveal <b>Delete</b>.</li>
  <li>On desktop: session actions are shown on the right.</li>
  <li>The dot indicates state: <b>blue</b> = busy, <b>gray</b> = idle, <b>orange</b> = snoozed or blocked.</li>
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

        const editCloseBtn = el("button", {
          id: "editCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const editStatus = el("div", { class: "muted", id: "editStatus", text: "" });
        const editNameInput = el("input", {
          id: "editNameInput",
          type: "text",
          placeholder: "Conversation title",
          maxlength: "80",
          autocomplete: "off",
        });
        const editPriorityRange = el("input", {
          id: "editPriorityRange",
          type: "range",
          min: "-1",
          max: "1",
          step: "0.05",
          value: "0",
        });
        const editPriorityValue = el("span", { class: "rangeValue", id: "editPriorityValue", text: "+0.00" });
        const editPriorityResetBtn = el("button", {
          id: "editPriorityResetBtn",
          class: "icon-btn text-btn subtleBtn",
          type: "button",
          text: "Reset",
        });
        const editSnoozeModeButtons = new Map();
        let editSnoozeMode = "none";
        const editSnoozeButtons = el("div", { class: "choiceChips", id: "editSnoozeButtons" });
        for (const [value, label] of [
          ["none", "No snooze"],
          ["4h", "4 hours"],
          ["tomorrow", "Tomorrow"],
          ["custom", "Custom"],
        ]) {
          const btn = el("button", {
            type: "button",
            class: "choiceChip",
            "data-snooze-mode": value,
            text: label,
          });
          editSnoozeModeButtons.set(value, btn);
          editSnoozeButtons.appendChild(btn);
        }
        const editSnoozeCustomDate = el("input", { id: "editSnoozeCustomDate", type: "date" });
        const editSnoozeCustomTime = el("input", { id: "editSnoozeCustomTime", type: "time", step: "60" });
        const editSnoozeCustomRow = el("div", { class: "customSnoozeRow", id: "editSnoozeCustomRow" }, [
          editSnoozeCustomDate,
          editSnoozeCustomTime,
        ]);
        const editDependencyBtn = el("button", {
          id: "editDependencyBtn",
          class: "filePickerBtn dialogPickerBtn",
          type: "button",
          "aria-label": "Choose dependency",
        });
        const editDependencyMenu = el("div", { id: "editDependencyMenu", class: "filePickerMenu dialogPickerMenu" });
        const editDependencyField = el("div", { class: "pickerField" }, [editDependencyBtn]);
        const editSaveBtn = el("button", { class: "primary", id: "editSaveBtn", type: "button", text: "Save" });
        const editViewer = el("dialog", { class: "formViewer formDialog", id: "editViewer", "aria-label": "Edit conversation" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "Edit conversation" }),
            el("div", { class: "actions" }, [editCloseBtn]),
          ]),
          editStatus,
          el("div", { class: "formBody" }, [
            el("div", { class: "formGrid twoCol" }, [
              el("label", { class: "field" }, [
                el("span", { class: "fieldLabel", text: "Conversation name" }),
                editNameInput,
              ]),
              el("label", { class: "field" }, [
                el("span", { class: "fieldLabel", text: "Priority offset" }),
                el("div", { class: "sliderRow" }, [editPriorityRange, editPriorityValue, editPriorityResetBtn]),
              ]),
            ]),
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Snooze" }),
              editSnoozeButtons,
              editSnoozeCustomRow,
            ]),
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Depends on" }),
              editDependencyField,
            ]),
          ]),
          el("div", { class: "formActions" }, [
            el("button", { id: "editCancelBtn", type: "button", text: "Cancel" }),
            editSaveBtn,
          ]),
        ]);
        root.appendChild(editViewer);
        editViewer.appendChild(editDependencyMenu);

        const newSessionBackdrop = el("div", { class: "modalBackdrop", id: "newSessionBackdrop" });
        const newSessionCloseBtn = el("button", {
          id: "newSessionCloseBtn",
          class: "icon-btn",
          title: "Close",
          "aria-label": "Close",
          type: "button",
          html: iconSvg("x"),
        });
        const newSessionStatus = el("div", { class: "muted", id: "newSessionStatus", text: "" });
        const newSessionCwdInput = el("input", {
          id: "newSessionCwdInput",
          type: "text",
          placeholder: "/path/to/project",
          autocomplete: "off",
          spellcheck: "false",
        });
        const newSessionRecentBtn = el("button", {
          id: "newSessionRecentBtn",
          class: "filePickerBtn dialogPickerBtn sidePickerBtn",
          type: "button",
          "aria-label": "Choose a recent working directory",
          "aria-haspopup": "menu",
          "aria-expanded": "false",
        });
        const newSessionRecentMenu = el("div", { id: "newSessionRecentMenu", class: "filePickerMenu dialogPickerMenu" });
        const newSessionCwdField = el("div", { class: "pickerInputRow pickerField" }, [
          newSessionCwdInput,
          newSessionRecentBtn,
        ]);
        const newSessionResumeBtn = el("button", {
          id: "newSessionResumeBtn",
          class: "filePickerBtn dialogPickerBtn sidePickerBtn",
          type: "button",
          "aria-label": "Choose a conversation to resume",
          "aria-haspopup": "menu",
          "aria-expanded": "false",
        });
        const newSessionResumeMenu = el("div", { id: "newSessionResumeMenu", class: "filePickerMenu dialogPickerMenu" });
        const newSessionResumeHint = el("div", { class: "muted", id: "newSessionResumeHint", text: "Default is Start fresh. Open the menu to pick a matching recent conversation." });
        const newSessionWorktreeToggle = el("input", {
          id: "newSessionWorktreeToggle",
          type: "checkbox",
        });
        const newSessionWorktreeInput = el("input", {
          id: "newSessionWorktreeBranchInput",
          type: "text",
          placeholder: "feature/my-branch",
          autocomplete: "off",
          spellcheck: "false",
          disabled: true,
        });
        const newSessionWorktreeHint = el("div", { class: "fieldHint", id: "newSessionWorktreeHint", text: "" });
        const newSessionWorktreeField = el("div", { class: "field", id: "newSessionWorktreeField" }, [
          el("span", { class: "fieldLabel", text: "Git worktree branch" }),
          el("label", { class: "checkField" }, [
            newSessionWorktreeToggle,
            el("span", { text: "Create a new worktree for this session" }),
          ]),
          newSessionWorktreeInput,
          newSessionWorktreeHint,
        ]);
        const newSessionStartBtn = el("button", { class: "primary", id: "newSessionStartBtn", type: "button", text: "Start session" });
        const newSessionViewer = el("div", { class: "formViewer newSessionViewer", id: "newSessionViewer", role: "dialog", "aria-label": "New session" }, [
          el("div", { class: "queueHeader" }, [
            el("div", { class: "title", text: "New session" }),
            el("div", { class: "actions" }, [newSessionCloseBtn]),
          ]),
          newSessionStatus,
          el("div", { class: "formBody" }, [
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Working directory" }),
              newSessionCwdField,
              el("span", { class: "fieldHint", text: "Type a path or open Recent directories." }),
            ]),
            el("label", { class: "field" }, [
              el("span", { class: "fieldLabel", text: "Resume conversation" }),
              newSessionResumeBtn,
              newSessionResumeHint,
            ]),
            newSessionWorktreeField,
          ]),
          el("div", { class: "formActions" }, [
            el("button", { id: "newSessionCancelBtn", type: "button", text: "Cancel" }),
            newSessionStartBtn,
          ]),
        ]);
        root.appendChild(newSessionBackdrop);
        root.appendChild(newSessionViewer);
        newSessionViewer.appendChild(newSessionRecentMenu);
        newSessionViewer.appendChild(newSessionResumeMenu);

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
          const md = el("div", { class: "md", html: mdToHtmlCached(ev.text) });
          bubble.appendChild(md);
          void upgradeCandidateFileRefs(md);
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
          fileRefCandidateCache.clear();
          const mobile = isMobile();
            const sessions = (data.sessions || [])
               .slice()
               .sort((a, b) => {
                 const p = Number(b.final_priority || 0) - Number(a.final_priority || 0);
                 if (p) return p;
                 const u = Number(b.updated_ts || b.start_ts || 0) - Number(a.updated_ts || a.start_ts || 0);
                 if (u) return u;
                 const s0 = Number(b.start_ts || 0) - Number(a.start_ts || 0);
                 if (s0) return s0;
                 return String(a.session_id || "").localeCompare(String(b.session_id || ""));
               });
	           if (mobile && openSwipeSessionId && sessionsWrap.childElementCount > 0) {
	             sessionIndex = new Map();
	             for (const s of sessions) sessionIndex.set(s.session_id, s);
	             swipeRefreshDeferred = true;
	             return sessions;
	           }
	           sessionsWrap.innerHTML = "";
	           openSwipeContent = null;
	           sessionIndex = new Map();
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
	             const branchTxt = typeof s.git_branch === "string" ? s.git_branch.trim() : "";

	            function closeOpenSwipe() {
	              if (!openSwipeContent) return;
	              openSwipeContent.style.transform = "translate3d(0px, 0, 0)";
	              openSwipeContent.dataset.swipeX = "0";
	              openSwipeContent = null;
	              openSwipeSessionId = null;
	              openSwipeTargetX = 0;
	              if (swipeRefreshDeferred) {
	                swipeRefreshDeferred = false;
	                void refreshSessions().catch((e) => console.error("refreshSessions failed after swipe close", e));
	              }
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
               title: "Edit conversation",
               "aria-label": "Edit conversation",
               type: "button",
               html: iconSvg("edit"),
             });
             renameBtn.onclick = (e) => {
               e.preventDefault();
               e.stopPropagation();
               closeOpenSwipe();
               openEditSession(s.session_id);
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
             const delBtn = el("button", {
               class: "icon-btn danger sessionDel",
               title: "Delete session",
               "aria-label": "Delete session",
               type: "button",
               html: iconSvg("trash"),
             });
             delBtn.onclick = (e) => void doDelete(e);

             const stateDot = el("span", {
               class:
                 "stateDot" +
                 (s.snoozed || s.blocked ? " suppressed" : s.busy ? " busy" : " idle"),
             });
             const titleRow = el("div", { class: "sessionTitleRow" }, [
               stateDot,
               el("div", { class: "titleLine", text: title, title: s.cwd || "" }),
             ]);
	             const badgesWrap = el("div", { class: "sessionBadges" }, badges);
	             const meta = el("div", { class: "muted subLine sessionMetaLine" }, [
	               el("span", { class: "ownerBadge", text: ownerTxt, title: s.owned ? "web-owned session" : "terminal-owned session" }),
	               el("span", { class: "metaText", text: `${stateTxt}${cwdBase ? ` | ${cwdBase}` : ""}${branchTxt ? ` | ${branchTxt}` : ""}` }),
	             ]);

             if (mobile) {
               const leftActions = el("div", { class: "sessionActions left" }, [delBtn]);
               const rightActions = el("div", { class: "sessionActions right" }, [renameBtn, dupBtn]);
               const top = el("div", { class: "row" }, [titleRow, badgesWrap]);
               const inner = el("div", { class: "sessionInner" }, [top, meta]);
	               const content = el("div", { class: "sessionContent" }, [inner]);
	               content.dataset.swipeX = "0";
	               const swipe = el("div", { class: "sessionSwipe" }, [leftActions, rightActions, content]);
	               card.appendChild(swipe);
	               if (openSwipeSessionId === s.session_id && openSwipeTargetX !== 0) {
	                  content.style.transform = `translate3d(${openSwipeTargetX}px, 0, 0)`;
	                  content.dataset.swipeX = String(openSwipeTargetX);
	                  openSwipeContent = content;
	               }

		               const leftMax = 72;
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
	                  if (target !== 0) {
	                    openSwipeContent = content;
	                    openSwipeSessionId = s.session_id;
	                    openSwipeTargetX = target;
	                  } else if (openSwipeContent === content) {
	                    openSwipeContent = null;
	                    openSwipeSessionId = null;
	                    openSwipeTargetX = 0;
	                  }
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
	               const actions = el("div", { class: "sessionActionsInline" }, [renameBtn, dupBtn, delBtn]);
	               const titleWithBadges = el("div", { class: "sessionTitleWithBadges" }, [titleRow, badgesWrap]);
	               const main = el("div", { class: "sessionMain" }, [titleWithBadges, meta]);
	               const inner = el("div", { class: "sessionInner sessionDesktopLayout" }, [main, actions]);
	               card.appendChild(inner);
	               card.onclick = () => selectSession(s.session_id);
	             }

	             sessionsWrap.appendChild(card);
	            }
	          if (openSwipeSessionId && !sessionIndex.has(openSwipeSessionId)) {
	            openSwipeSessionId = null;
	            openSwipeTargetX = 0;
	            openSwipeContent = null;
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
        function renderRecentCwdOptions() {
          const out = [];
          const seen = new Set();
          for (const s of sessionIndex.values()) {
            const cwd = typeof s.cwd === "string" ? s.cwd.trim() : "";
            if (!cwd || seen.has(cwd)) continue;
            seen.add(cwd);
            out.push(cwd);
          }
          return out;
        }

        function hideEditSession() {
          editSessionId = null;
          editStatus.textContent = "";
          editDependencyMenuOpen = false;
          applyDialogMenus();
          if (editViewer.open) editViewer.close();
        }

        function syncEditPriorityLabel() {
          editPriorityValue.textContent = formatPriorityOffset(editPriorityRange.value);
        }

        function setEditSnoozeMode(mode) {
          editSnoozeMode = ["none", "4h", "tomorrow", "custom"].includes(mode) ? mode : "none";
          for (const [value, btn] of editSnoozeModeButtons.entries()) {
            btn.classList.toggle("active", value === editSnoozeMode);
          }
          editSnoozeCustomRow.style.display = editSnoozeMode === "custom" ? "grid" : "none";
        }

        function tomorrowSnoozeSeconds() {
          const d = new Date();
          d.setDate(d.getDate() + 1);
          d.setHours(9, 0, 0, 0);
          return Math.floor(d.getTime() / 1000);
        }

        function fillCustomSnoozeInputs(tsSeconds) {
          const ts = Number(tsSeconds);
          const d = Number.isFinite(ts) && ts > 0 ? new Date(ts * 1000) : new Date(Date.now() + 24 * 3600 * 1000);
          const yyyy = String(d.getFullYear()).padStart(4, "0");
          const mm = String(d.getMonth() + 1).padStart(2, "0");
          const dd = String(d.getDate()).padStart(2, "0");
          const hh = String(d.getHours()).padStart(2, "0");
          const mi = String(d.getMinutes()).padStart(2, "0");
          editSnoozeCustomDate.value = `${yyyy}-${mm}-${dd}`;
          editSnoozeCustomTime.value = `${hh}:${mi}`;
        }

        function fillDependencyOptions(currentSid, currentDependencySid) {
          editDependencyMenu.innerHTML = "";
          const addItem = (value, label, active) => {
            const btn = el("button", {
              class: "fileMenuItem" + (active ? " active" : ""),
              type: "button",
              title: label,
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: label }));
            btn.onclick = () => {
              editDependencyBtn.dataset.value = value || "";
              setDependencyButtonContent();
              editDependencyMenuOpen = false;
              applyDialogMenus();
            };
            editDependencyMenu.appendChild(btn);
          };
          addItem("", "No dependency", !currentDependencySid);
          for (const s of sessionIndex.values()) {
            if (!s || s.session_id === currentSid) continue;
            const label = `${sessionDisplayName(s)}${s.cwd ? ` | ${baseName(s.cwd)}` : ""}`;
            addItem(s.session_id, label, currentDependencySid === s.session_id);
          }
          editDependencyBtn.dataset.value = currentDependencySid || "";
          setDependencyButtonContent();
        }

        function setDependencyButtonContent() {
          const value = String(editDependencyBtn.dataset.value || "");
          let label = "No dependency";
          if (value) {
            const s = sessionIndex.get(value);
            if (s) label = `${sessionDisplayName(s)}${s.cwd ? ` | ${baseName(s.cwd)}` : ""}`;
          }
          setPickerButtonContent(editDependencyBtn, label);
        }

        function renderRecentCwdMenu() {
          newSessionRecentMenu.innerHTML = "";
          const items = renderRecentCwdOptions();
          if (!items.length) {
            newSessionRecentMenu.appendChild(el("div", { class: "pickerEmpty", text: "No recent directories" }));
            return;
          }
          for (const cwd of items) {
            const btn = el("button", {
              class: "fileMenuItem" + (newSessionCwdInput.value.trim() === cwd ? " active" : ""),
              type: "button",
              title: cwd,
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: cwd }));
            btn.onclick = () => {
              newSessionCwdInput.value = cwd;
              newSessionRecentMenuOpen = false;
              applyDialogMenus();
              scheduleNewSessionResumeLoad();
              newSessionCwdInput.focus();
            };
            newSessionRecentMenu.appendChild(btn);
          }
        }

        function newSessionResumeLabel(item) {
          if (!item || typeof item !== "object") return "Start fresh";
          const alias = typeof item.alias === "string" ? item.alias.trim() : "";
          const firstUser = typeof item.first_user_message === "string" ? item.first_user_message.trim() : "";
          const primary = alias || firstUser || shortSessionId(item.session_id);
          const ts = Number(item.updated_ts || 0);
          const age = ts > 0 ? fmtRelativeAge(Math.max(0, Date.now() / 1000 - ts)) : "";
          return `${age ? `${age} | ` : ""}${primary}`;
        }

        function setPickerButtonContent(button, primaryText, secondaryText = "", placeholder = false) {
          if (!button) return;
          button.innerHTML = "";
          const textWrap = el("span", { class: `pickerButtonText${placeholder ? " placeholder" : ""}` });
          textWrap.appendChild(el("span", { class: "pickerButtonPrimary", text: String(primaryText || "") }));
          if (secondaryText) textWrap.appendChild(el("span", { class: "pickerButtonSecondary", text: String(secondaryText) }));
          button.appendChild(textWrap);
          button.appendChild(el("span", { class: "pickerButtonChevron", html: iconSvg("chevronDown") }));
        }

        function setNewSessionResumeSelection(item) {
          newSessionResumeSelection = item && typeof item === "object" ? item : null;
          setPickerButtonContent(
            newSessionResumeBtn,
            newSessionResumeSelection ? newSessionResumeLabel(newSessionResumeSelection) : "Start fresh",
            newSessionResumeSelection ? "Matching recent conversation" : "Open to pick a matching recent conversation",
            !newSessionResumeSelection
          );
          syncNewSessionWorktreeUi();
        }

        function worktreePathSlug(branch) {
          return String(branch || "")
            .trim()
            .replace(/[^A-Za-z0-9._-]+/g, "-")
            .replace(/^[.-]+|[.-]+$/g, "") || "worktree";
        }

        function defaultWorktreePath(cwd, branch) {
          const rawCwd = String(cwd || "").trim().replace(/\/+$/, "");
          const base = baseName(rawCwd) || "worktree";
          const slash = rawCwd.lastIndexOf("/");
          const parent = slash > 0 ? rawCwd.slice(0, slash) : slash === 0 ? "/" : "";
          const slug = worktreePathSlug(branch);
          if (parent === "/") return `/${base}-${slug}`;
          return `${parent || "."}/${base}-${slug}`;
        }

        function newSessionWorktreeBranchHint() {
          const branch = String(newSessionWorktreeInput.value || "").trim();
          const cwd = String(newSessionCwdInput.value || "").trim();
          if (!branch) return "Enter a branch name. The worktree will be created as a sibling of the selected directory.";
          const path = defaultWorktreePath(cwd, branch);
          const baseInfo = newSessionCwdInfo && newSessionCwdInfo.git_branch ? ` from ${newSessionCwdInfo.git_branch}` : "";
          return `Creates branch ${branch}${baseInfo} at ${path}.`;
        }

        function clearNewSessionCwdInfo() {
          newSessionCwdInfo = { git_repo: false, git_root: "", git_branch: "" };
        }

        function syncNewSessionWorktreeUi() {
          const canOffer = !!(newSessionCwdInfo && newSessionCwdInfo.git_repo) && !newSessionResumeSelection;
          if (!canOffer) newSessionWorktreeToggle.checked = false;
          const enabled = canOffer && !!newSessionWorktreeToggle.checked;
          newSessionWorktreeField.style.display = canOffer ? "" : "none";
          newSessionWorktreeInput.disabled = !enabled;
          newSessionWorktreeInput.style.display = enabled ? "" : "none";
          newSessionWorktreeHint.textContent = canOffer ? newSessionWorktreeBranchHint() : "";
          if (newSessionResumeSelection) newSessionStartBtn.textContent = "Resume session";
          else if (enabled) newSessionStartBtn.textContent = "Create worktree session";
          else newSessionStartBtn.textContent = "Start session";
        }

        function renderNewSessionResumeMenu() {
          newSessionResumeMenu.innerHTML = "";
          const freshBtn = el("button", {
            class: "fileMenuItem" + (!newSessionResumeSelection ? " active" : ""),
            type: "button",
            title: "Start a new conversation",
          });
          freshBtn.appendChild(el("span", { class: "fileMenuPath", text: "Start fresh" }));
          freshBtn.onclick = () => {
            setNewSessionResumeSelection(null);
            newSessionResumeMenuOpen = false;
            applyDialogMenus();
          };
          newSessionResumeMenu.appendChild(freshBtn);
          if (!newSessionResumeCandidates.length) {
            newSessionResumeMenu.appendChild(el("div", { class: "pickerEmpty", text: "No matching sessions" }));
            return;
          }
          for (const item of newSessionResumeCandidates) {
            const btn = el("button", {
              class: "fileMenuItem" + (newSessionResumeSelection && newSessionResumeSelection.session_id === item.session_id ? " active" : ""),
              type: "button",
              title: newSessionResumeLabel(item),
            });
            btn.appendChild(el("span", { class: "fileMenuPath", text: newSessionResumeLabel(item) }));
            btn.onclick = () => {
              setNewSessionResumeSelection(item);
              newSessionResumeMenuOpen = false;
              applyDialogMenus();
            };
            newSessionResumeMenu.appendChild(btn);
          }
        }

        async function loadNewSessionResumeCandidates(cwd) {
          const raw = String(cwd || "").trim();
          const seq = ++newSessionResumeLoadSeq;
          if (!raw) {
            newSessionResumeCandidates = [];
            setNewSessionResumeSelection(null);
            clearNewSessionCwdInfo();
            newSessionResumeHint.textContent = "Specify a working directory to load prior sessions.";
            renderNewSessionResumeMenu();
            syncNewSessionWorktreeUi();
            return;
          }
          newSessionResumeHint.textContent = "Loading matching sessions...";
          try {
            const res = await api(`/api/session_resume_candidates?cwd=${encodeURIComponent(raw)}`);
            if (seq !== newSessionResumeLoadSeq) return;
            newSessionCwdInfo = {
              git_repo: !!(res && res.git_repo),
              git_root: res && typeof res.git_root === "string" ? res.git_root : "",
              git_branch: res && typeof res.git_branch === "string" ? res.git_branch : "",
            };
            const items = Array.isArray(res && res.sessions) ? res.sessions.filter((item) => item && typeof item === "object" && typeof item.session_id === "string") : [];
            newSessionResumeCandidates = items;
            const currentId = newSessionResumeSelection && typeof newSessionResumeSelection.session_id === "string" ? newSessionResumeSelection.session_id : "";
            const next = currentId ? items.find((item) => item.session_id === currentId) || null : null;
            setNewSessionResumeSelection(next);
            newSessionResumeHint.textContent = items.length ? `${items.length} matching session${items.length === 1 ? "" : "s"} found.` : "No matching sessions for this directory.";
            renderNewSessionResumeMenu();
            syncNewSessionWorktreeUi();
          } catch (e) {
            if (seq !== newSessionResumeLoadSeq) return;
            newSessionResumeCandidates = [];
            setNewSessionResumeSelection(null);
            clearNewSessionCwdInfo();
            newSessionResumeHint.textContent = e && e.message ? e.message : "Unable to load prior sessions.";
            renderNewSessionResumeMenu();
            syncNewSessionWorktreeUi();
          }
        }

        function scheduleNewSessionResumeLoad() {
          if (newSessionResumeLoadTimer) clearTimeout(newSessionResumeLoadTimer);
          const cwd = String(newSessionCwdInput.value || "").trim();
          newSessionResumeLoadTimer = setTimeout(() => {
            newSessionResumeLoadTimer = null;
            void loadNewSessionResumeCandidates(cwd);
          }, 180);
        }

        function applyDialogMenus() {
          editDependencyMenu.classList.toggle("open", editDependencyMenuOpen);
          newSessionRecentMenu.classList.toggle("open", newSessionRecentMenuOpen);
          newSessionResumeMenu.classList.toggle("open", newSessionResumeMenuOpen);
          editDependencyBtn.setAttribute("aria-expanded", editDependencyMenuOpen ? "true" : "false");
          newSessionRecentBtn.setAttribute("aria-expanded", newSessionRecentMenuOpen ? "true" : "false");
          newSessionResumeBtn.setAttribute("aria-expanded", newSessionResumeMenuOpen ? "true" : "false");
          if (editDependencyMenuOpen) positionDialogMenu(editDependencyMenu, editDependencyBtn);
          if (newSessionRecentMenuOpen) positionDialogMenu(newSessionRecentMenu, newSessionRecentBtn);
          if (newSessionResumeMenuOpen) positionDialogMenu(newSessionResumeMenu, newSessionResumeBtn);
        }

        function positionDialogMenu(menu, anchorBtn) {
          if (!menu || !anchorBtn) return;
          const host = menu.parentElement;
          if (!host) return;
          const vv = window.visualViewport;
          const rect = anchorBtn.getBoundingClientRect();
          const hostRect = host.getBoundingClientRect();
          const viewportW = hostRect.width;
          const viewportTop = vv ? vv.offsetTop : 0;
          const viewportBottom = viewportTop + (vv ? vv.height : window.innerHeight);
          const margin = 12;
          const desiredWidth = Math.min(Math.max(rect.width, 280), viewportW - margin * 2);
          menu.style.position = "absolute";
          const left = Math.max(margin, Math.min(viewportW - margin - desiredWidth, rect.left - hostRect.left));
          menu.style.left = `${left}px`;
          menu.style.width = `${desiredWidth}px`;
          menu.style.right = "auto";
          menu.style.bottom = "auto";
          menu.style.maxHeight = "";
          const menuHeight = Math.min(menu.scrollHeight || 260, Math.floor((viewportBottom - viewportTop) * 0.5));
          const spaceBelow = viewportBottom - rect.bottom - margin;
          const spaceAbove = rect.top - viewportTop - margin;
          const openAbove = spaceBelow < Math.min(220, menuHeight) && spaceAbove > spaceBelow;
          if (openAbove) {
            const maxHeight = Math.max(120, spaceAbove - 8);
            menu.style.maxHeight = `${maxHeight}px`;
            const top = Math.max(viewportTop + margin - hostRect.top, rect.top - hostRect.top - Math.min(menuHeight, maxHeight) - 8);
            menu.style.top = `${top}px`;
          } else {
            const maxHeight = Math.max(120, spaceBelow - 8);
            menu.style.maxHeight = `${maxHeight}px`;
            const top = Math.min(viewportBottom - margin - hostRect.top - Math.min(menuHeight, maxHeight), rect.bottom - hostRect.top + 8);
            menu.style.top = `${top}px`;
          }
        }

        function openEditSession(sid) {
          if (!sid) return;
          const s = sessionIndex.get(sid);
          if (!s) return;
          editSessionId = sid;
          editStatus.textContent = "";
          editNameInput.value = typeof s.alias === "string" ? s.alias : "";
          editNameInput.placeholder = sessionDisplayName(s) || "Conversation title";
          editPriorityRange.value = String(Number(s.priority_offset || 0));
          syncEditPriorityLabel();
          const snoozeUntil = Number(s.snooze_until || 0);
          if (snoozeUntil > Date.now() / 1000) {
            setEditSnoozeMode("custom");
            fillCustomSnoozeInputs(snoozeUntil);
          } else {
            setEditSnoozeMode("none");
            fillCustomSnoozeInputs(tomorrowSnoozeSeconds());
          }
          fillDependencyOptions(sid, s.dependency_session_id || "");
          if (!editViewer.open) editViewer.showModal();
        }

        function hideNewSessionDialog() {
          newSessionStatus.textContent = "";
          newSessionRecentMenuOpen = false;
          newSessionResumeMenuOpen = false;
          applyDialogMenus();
          newSessionBackdrop.style.display = "none";
          newSessionViewer.style.display = "none";
        }

        function openNewSessionDialog({ cwd = null, statusText = "" } = {}) {
          const cur = selected ? sessionIndex.get(selected) : null;
          const initialCwd = typeof cwd === "string" && cwd.trim() ? cwd.trim() : cur && cur.cwd && cur.cwd !== "?" ? cur.cwd : "";
          newSessionStatus.textContent = String(statusText || "");
          newSessionCwdInput.value = initialCwd;
          newSessionResumeCandidates = [];
          setNewSessionResumeSelection(null);
          clearNewSessionCwdInfo();
          newSessionWorktreeToggle.checked = false;
          newSessionWorktreeInput.value = "";
          newSessionWorktreeInput.disabled = true;
          newSessionWorktreeInput.style.display = "none";
          newSessionWorktreeField.style.display = "none";
          newSessionWorktreeHint.textContent = "";
          newSessionResumeHint.textContent = "Loading matching sessions...";
          setPickerButtonContent(newSessionRecentBtn, "Recent directories", "Reuse a path from another session", true);
          renderRecentCwdMenu();
          renderNewSessionResumeMenu();
          newSessionBackdrop.style.display = "block";
          newSessionViewer.style.display = "flex";
          scheduleNewSessionResumeLoad();
          syncNewSessionWorktreeUi();
          if (isMobile()) return;
          requestAnimationFrame(() => {
            newSessionCwdInput.focus({ preventScroll: true });
            const end = newSessionCwdInput.value.length;
            try {
              newSessionCwdInput.setSelectionRange(end, end);
            } catch {}
          });
        }

        editPriorityRange.oninput = syncEditPriorityLabel;
        editPriorityResetBtn.onclick = () => {
          editPriorityRange.value = "0";
          syncEditPriorityLabel();
        };
        for (const [mode, btn] of editSnoozeModeButtons.entries()) {
          btn.onclick = () => {
            setEditSnoozeMode(mode);
            if (mode === "tomorrow") fillCustomSnoozeInputs(tomorrowSnoozeSeconds());
            else if (mode === "4h") fillCustomSnoozeInputs(Math.floor(Date.now() / 1000) + 4 * 3600);
          };
        }
        editDependencyBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          editDependencyMenuOpen = !editDependencyMenuOpen;
          newSessionRecentMenuOpen = false;
          newSessionResumeMenuOpen = false;
          applyDialogMenus();
        };
        newSessionRecentBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          renderRecentCwdMenu();
          newSessionRecentMenuOpen = !newSessionRecentMenuOpen;
          editDependencyMenuOpen = false;
          newSessionResumeMenuOpen = false;
          applyDialogMenus();
        };
        newSessionResumeBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          renderNewSessionResumeMenu();
          newSessionResumeMenuOpen = !newSessionResumeMenuOpen;
          editDependencyMenuOpen = false;
          newSessionRecentMenuOpen = false;
          applyDialogMenus();
        };
        editCloseBtn.onclick = () => hideEditSession();
        $("#editCancelBtn").onclick = () => hideEditSession();
        editViewer.addEventListener("cancel", (e) => {
          e.preventDefault();
          hideEditSession();
        });
        editViewer.onclick = (e) => {
          if (e.target === editViewer) hideEditSession();
        };
        editSaveBtn.onclick = async () => {
          if (!editSessionId) return;
          let snoozeUntil = null;
          const snoozeMode = editSnoozeMode;
          if (snoozeMode === "4h") {
            snoozeUntil = Math.floor(Date.now() / 1000) + 4 * 3600;
          } else if (snoozeMode === "tomorrow") {
            snoozeUntil = tomorrowSnoozeSeconds();
          } else if (snoozeMode === "custom") {
            const dateRaw = String(editSnoozeCustomDate.value || "").trim();
            const timeRaw = String(editSnoozeCustomTime.value || "").trim();
            if (!dateRaw || !timeRaw) {
              editStatus.textContent = "Choose both a custom date and time.";
              return;
            }
            const parsed = Date.parse(`${dateRaw}T${timeRaw}`);
            if (!Number.isFinite(parsed)) {
              editStatus.textContent = "Invalid snooze time.";
              return;
            }
            snoozeUntil = Math.floor(parsed / 1000);
          }
          try {
            editStatus.textContent = "Saving...";
            await api(`/api/sessions/${editSessionId}/edit`, {
              method: "POST",
              body: {
                name: String(editNameInput.value || ""),
                priority_offset: Number(editPriorityRange.value || 0),
                snooze_until: snoozeUntil,
                dependency_session_id: String(editDependencyBtn.dataset.value || "") || null,
              },
            });
            hideEditSession();
            await refreshSessions();
            if (selected === editSessionId) {
              const s2 = sessionIndex.get(editSessionId);
              if (s2) titleLabel.textContent = sessionTitleWithId(s2);
            }
            setToast("conversation updated");
          } catch (e) {
            editStatus.textContent = e && e.message ? e.message : "Save failed";
          }
        };

        newSessionCloseBtn.onclick = () => hideNewSessionDialog();
        $("#newSessionCancelBtn").onclick = () => hideNewSessionDialog();
        newSessionBackdrop.onclick = () => hideNewSessionDialog();
        newSessionViewer.onclick = (e) => e.stopPropagation();
        newSessionCwdInput.oninput = () => {
          renderRecentCwdMenu();
          scheduleNewSessionResumeLoad();
        };
        newSessionWorktreeToggle.onchange = () => {
          syncNewSessionWorktreeUi();
          if (newSessionWorktreeToggle.checked) newSessionWorktreeInput.focus();
        };
        newSessionWorktreeInput.oninput = () => syncNewSessionWorktreeUi();
        newSessionStartBtn.onclick = async () => {
          const cwd = String(newSessionCwdInput.value || "").trim();
          if (!cwd) {
            newSessionStatus.textContent = "Working directory is required.";
            return;
          }
          const resumeSessionId = newSessionResumeSelection && newSessionResumeSelection.session_id ? newSessionResumeSelection.session_id : null;
          const worktreeBranch = !resumeSessionId && newSessionWorktreeToggle.checked ? String(newSessionWorktreeInput.value || "").trim() : null;
          if (newSessionWorktreeToggle.checked && !worktreeBranch) {
            newSessionStatus.textContent = "Branch name is required.";
            return;
          }
          newSessionStatus.textContent = resumeSessionId ? "Resuming..." : worktreeBranch ? "Creating worktree..." : "Starting...";
          const brokerPid = await spawnSessionWithCwd(cwd, resumeSessionId, worktreeBranch);
          if (brokerPid) hideNewSessionDialog();
          else newSessionStatus.textContent = "Start failed.";
        };
        let fileViewMode = localStorage.getItem("codexweb.fileViewMode") || "diff"; // "diff" | "file" | "preview"
        let fileNonDiffMode = localStorage.getItem("codexweb.fileNonDiffMode") === "preview" ? "preview" : "file";
        let fileCandidateList = [];
        let fileEntryMap = new Map();
        let activeFilePath = "";
        let fileMenuOpen = false;
        let monacoReadyPromise = null;
        let monacoNs = null;
        let monacoThemeReady = false;
        let fileEditor = null;
        let fileEditorKind = "";
        let fileEditorModels = [];

        function extToEditorLang(p) {
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

        function disposeFileEditor() {
          fileDiff.innerHTML = "";
          for (const model of fileEditorModels) {
            try {
              model.dispose();
            } catch (_) {}
          }
          fileEditorModels = [];
          if (fileEditor) {
            try {
              fileEditor.dispose();
            } catch (_) {}
            fileEditor = null;
          }
          fileEditorKind = "";
        }

        function ensureMonaco() {
          if (monacoReadyPromise) return monacoReadyPromise;
          monacoReadyPromise = new Promise((resolve, reject) => {
            const finish = () => {
              if (!(window.require && window.require.config)) {
                reject(new Error("monaco loader unavailable"));
                return;
              }
              const base = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs";
              window.MonacoEnvironment = {
                getWorkerUrl(_moduleId, _label) {
                  const src = `
self.MonacoEnvironment={baseUrl:${JSON.stringify(base + "/")}};
importScripts(${JSON.stringify(base + "/base/worker/workerMain.js")});
`;
                  return `data:text/javascript;charset=utf-8,${encodeURIComponent(src)}`;
                },
              };
              window.require.config({ paths: { vs: base } });
              window.require(["vs/editor/editor.main"], () => {
                monacoNs = window.monaco;
                if (!monacoNs) {
                  reject(new Error("monaco failed to initialize"));
                  return;
                }
                if (!monacoThemeReady) {
                  monacoNs.editor.defineTheme("codoxear-github-light", {
                    base: "vs",
                    inherit: true,
                    rules: [],
                    colors: {
                      "editor.background": "#ffffff",
                      "editor.lineHighlightBackground": "#f6f8fa",
                      "editorGutter.background": "#ffffff",
                      "editorLineNumber.foreground": "#8c959f",
                      "editorLineNumber.activeForeground": "#57606a",
                      "diffEditor.insertedTextBackground": "#dafbe1",
                      "diffEditor.removedTextBackground": "#ffebe9",
                      "diffEditor.insertedLineBackground": "#f0fff4",
                      "diffEditor.removedLineBackground": "#fff5f5",
                    },
                  });
                  monacoThemeReady = true;
                }
                resolve(monacoNs);
              }, reject);
            };
            if (window.monaco && window.monaco.editor) {
              monacoNs = window.monaco;
              finish();
              return;
            }
            if (window.require && window.require.config) {
              finish();
              return;
            }
            const waitForLoader = () => {
              if (window.require && window.require.config) {
                finish();
                return;
              }
              setTimeout(waitForLoader, 25);
            };
            waitForLoader();
          });
          return monacoReadyPromise;
        }

        function applyEditorLineFocus(lineNumber) {
          const line = normalizeLineNumber(lineNumber);
          if (!fileEditor || !line) return;
          if (fileEditorKind === "diff" && fileEditor.getModifiedEditor) {
            const editor = fileEditor.getModifiedEditor();
            editor.setPosition({ lineNumber: line, column: 1 });
            editor.revealLineInCenter(line);
            editor.focus();
            return;
          }
          if (fileEditor.setPosition) {
            fileEditor.setPosition({ lineNumber: line, column: 1 });
            fileEditor.revealLineInCenter(line);
            fileEditor.focus();
          }
        }

        async function renderMonacoFile(rel, text, lineNumber = null, langOverride = "") {
          const monaco = await ensureMonaco();
          const host = fileDiff;
          const lang = langOverride || extToEditorLang(rel);
          if (fileEditorKind !== "file") {
            disposeFileEditor();
            fileEditor = monaco.editor.create(host, {
              language: lang || "plaintext",
              value: String(text || ""),
              readOnly: true,
              theme: "codoxear-github-light",
              lineNumbers: "on",
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              wordWrap: "on",
              folding: false,
              renderLineHighlight: "none",
              glyphMargin: false,
              overviewRulerBorder: false,
              stickyScroll: { enabled: false },
              automaticLayout: true,
            });
            fileEditorKind = "file";
            fileEditorModels = [fileEditor.getModel()].filter(Boolean);
          } else {
            const model = fileEditor.getModel();
            monaco.editor.setModelLanguage(model, lang || "plaintext");
            model.setValue(String(text || ""));
          }
          const targetLine = normalizeLineNumber(lineNumber) || 1;
          fileEditor.setScrollPosition({ scrollTop: 0, scrollLeft: 0 });
          fileEditor.setPosition({ lineNumber: targetLine, column: 1 });
          fileEditor.revealPositionInCenter({ lineNumber: targetLine, column: 1 });
          fileEditor.layout();
          requestAnimationFrame(() => {
            if (!fileEditor) return;
            fileEditor.layout();
            applyEditorLineFocus(targetLine);
          });
          setTimeout(() => {
            if (!fileEditor) return;
            fileEditor.layout();
            applyEditorLineFocus(targetLine);
          }, 60);
        }

        async function renderMonacoDiff(rel, originalText, modifiedText, lineNumber = null) {
          const monaco = await ensureMonaco();
          const host = fileDiff;
          const lang = extToEditorLang(rel);
          disposeFileEditor();
          const originalModel = monaco.editor.createModel(String(originalText || ""), lang || "plaintext");
          const modifiedModel = monaco.editor.createModel(String(modifiedText || ""), lang || "plaintext");
          fileEditor = monaco.editor.createDiffEditor(host, {
            readOnly: true,
            theme: "codoxear-github-light",
            renderSideBySide: false,
            useInlineViewWhenSpaceIsLimited: true,
            lineNumbers: "on",
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            wordWrap: "on",
            diffWordWrap: "on",
            folding: false,
            renderLineHighlight: "none",
            glyphMargin: false,
            overviewRulerBorder: false,
            stickyScroll: { enabled: false },
            automaticLayout: true,
            hideUnchangedRegions: {
              enabled: true,
              contextLineCount: 4,
              minimumLineCount: 1,
              revealLineCount: 2,
            },
          });
          fileEditor.setModel({ original: originalModel, modified: modifiedModel });
          fileEditorKind = "diff";
          fileEditorModels = [originalModel, modifiedModel];
          const originalEditor = fileEditor.getOriginalEditor();
          const modifiedEditor = fileEditor.getModifiedEditor();
          originalEditor.updateOptions({
            wordWrap: "on",
            lineNumbers: "off",
            glyphMargin: false,
            lineDecorationsWidth: 0,
            lineNumbersMinChars: 0,
          });
          modifiedEditor.updateOptions({
            wordWrap: "on",
            lineNumbers: "on",
            glyphMargin: false,
            lineDecorationsWidth: 0,
            lineNumbersMinChars: 3,
          });
          const targetLine = normalizeLineNumber(lineNumber) || 1;
          originalEditor.setScrollPosition({ scrollTop: 0, scrollLeft: 0 });
          modifiedEditor.setScrollPosition({ scrollTop: 0, scrollLeft: 0 });
          originalEditor.setPosition({ lineNumber: targetLine, column: 1 });
          modifiedEditor.setPosition({ lineNumber: targetLine, column: 1 });
          modifiedEditor.revealPositionInCenter({ lineNumber: targetLine, column: 1 });
          fileEditor.layout();
          requestAnimationFrame(() => {
            if (!fileEditor) return;
            fileEditor.layout();
            applyEditorLineFocus(targetLine);
          });
          setTimeout(() => {
            if (!fileEditor) return;
            fileEditor.layout();
            applyEditorLineFocus(targetLine);
          }, 60);
        }

        function renderMarkdownPreview(rel, text) {
          disposeFileEditor();
          fileDiff.innerHTML = "";
          const preview = el("div", {
            class: "md fileMarkdownPreview",
            html: markdownPreviewHtml(String(text || ""), { filePath: rel, sessionId: selected || "" }),
          });
          fileDiff.appendChild(preview);
          void upgradeCandidateFileRefs(preview);
        }

        function setFileViewMode(mode) {
          const next = mode === "preview" ? "preview" : mode === "file" ? "file" : "diff";
          fileViewMode = next;
          localStorage.setItem("codexweb.fileViewMode", fileViewMode);
          if (next !== "diff") {
            fileNonDiffMode = next;
            localStorage.setItem("codexweb.fileNonDiffMode", fileNonDiffMode);
          }
          applyFileMode();
        }

        function applyFileMode() {
          const hasPath = Boolean(activeFilePath);
          const isDiff = fileViewMode === "diff";
          const isPreview = fileViewMode === "preview";
          const previewable = isMarkdownPreviewable(activeFilePath);
          fileModeDiffBtn.classList.toggle("active", hasPath && isDiff);
          fileModePreviewBtn.classList.toggle("active", hasPath && isPreview);
          fileModeDiffBtn.disabled = !hasPath;
          fileModePreviewBtn.disabled = !hasPath;
          fileDownloadBtn.disabled = !hasPath;
          fileModePreviewBtn.style.display = previewable ? "" : "none";
        }

        function applyFileMenuState() {
          filePickerBtn.classList.toggle("active", fileMenuOpen);
          filePickerMenu.classList.toggle("open", fileMenuOpen);
        }

        function setFilePath(rel, { line = null } = {}) {
          const next = String(rel || "").trim();
          activeFilePath = next;
          activeFileLine = normalizeLineNumber(line);
          const entry = fileEntryMap.get(next);
          setFilePickerButtonContent(entry || (next ? { path: next, changed: false, additions: null, deletions: null } : null), next || "Choose file");
          fileMenuOpen = false;
          applyFileMenuState();
          applyFileMode();
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

        function setFilePickerButtonContent(entry, fallbackText = "Choose file") {
          filePickerBtn.innerHTML = "";
          if (!entry || typeof entry.path !== "string" || !entry.path.trim()) {
            filePickerBtn.textContent = fallbackText;
            return;
          }
          filePickerBtn.appendChild(el("span", { class: "fileMenuPath", text: entry.path }));
          const changed = Boolean(entry.changed);
          if (changed) {
            const stat = el("span", { class: "fileMenuStat changed" });
            stat.appendChild(el("span", { class: "fileMenuAdd", text: entry.additions == null ? "+?" : `+${entry.additions}` }));
            stat.appendChild(el("span", { class: "fileMenuDel", text: entry.deletions == null ? "-?" : `-${entry.deletions}` }));
            filePickerBtn.appendChild(stat);
          }
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

        function rememberOpenedFile(relPath, absPath = null) {
          const rel = String(relPath || "").trim();
          if (!rel) return;
          upsertFileEntry({ path: rel, additions: null, deletions: null, changed: false });
          const s = selected ? sessionIndex.get(selected) : null;
          if (!s) return;
          const files = listFromFilesField(s.files);
          const abs = typeof absPath === "string" && absPath.trim()
            ? absPath.trim()
            : s.cwd && rel !== "."
              ? `${String(s.cwd).replace(/\/+$/, "")}/${rel.replace(/^\.?\//, "")}`
              : "";
          if (!abs) return;
          const nextFiles = [abs, ...files.filter((x) => x !== abs)];
          s.files = nextFiles;
        }

        async function getKnownFileRefCandidates() {
          if (!selected) return [];
          const sid = selected;
          const hit = fileRefCandidateCache.get(sid);
          if (hit) return hit;
          const task = (async () => {
            const out = new Set();
            const s = sessionIndex.get(sid);
            for (const abs of listFromFilesField(s && s.files)) {
              const rel = sessionRelativePath(abs);
              if (typeof rel === "string" && rel && rel !== ".") out.add(rel);
            }
            try {
              const res = await api(`/api/sessions/${sid}/git/changed_files`);
              const entries = Array.isArray(res.entries) ? res.entries : [];
              for (const entry of entries) {
                if (!entry || typeof entry.path !== "string") continue;
                const path = String(entry.path).trim();
                if (path) out.add(path);
              }
            } catch {}
            return [...out];
          })();
          fileRefCandidateCache.set(sid, task);
          const resolved = await task;
          fileRefCandidateCache.set(sid, resolved);
          return resolved;
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
            if (entry.changed) {
              const stat = el("span", { class: "fileMenuStat changed" });
              stat.appendChild(el("span", { class: "fileMenuAdd", text: entry.additions == null ? "+?" : `+${entry.additions}` }));
              stat.appendChild(el("span", { class: "fileMenuDel", text: entry.deletions == null ? "-?" : `-${entry.deletions}` }));
              btn.appendChild(stat);
            }
            btn.onclick = () => {
              setFilePath(path, { line: null });
              const selectedEntry = fileEntryMap.get(path);
              if (!selectedEntry || selectedEntry.changed) {
                setFileViewMode("diff");
              } else {
                setFileViewMode(isMarkdownPreviewable(path) && fileNonDiffMode === "preview" ? "preview" : "file");
              }
              renderFilePickerMenu();
              void openFilePath(path, { line: null });
            };
            filePickerMenu.appendChild(btn);
          }
        }

        function fileRefValidationKey(path) {
          return `${selected || ""}|${String(path || "").trim()}`;
        }

        async function inspectFileRefPath(path) {
          const rawPath = String(path || "").trim();
          if (!rawPath) return { ok: false };
          let inspectPath = rawPath;
          if (!rawPath.includes("/") && selected) {
            const candidates = await getKnownFileRefCandidates();
            const matches = candidates.filter((candidate) => {
              const tail = String(candidate || "").split("/").pop() || "";
              return tail === rawPath;
            });
            if (matches.length === 1) inspectPath = matches[0];
            else if (matches.length > 1) return { ok: false, ambiguous: true, path: rawPath };
          }
          const key = fileRefValidationKey(inspectPath);
          if (fileRefValidationCache.has(key)) return fileRefValidationCache.get(key);
          const pending = fileRefValidationPending.get(key);
          if (pending) return pending;
          const task = (async () => {
            try {
              const body = { path: inspectPath };
              if (selected) body.session_id = selected;
              const res = await api("/api/files/inspect", { method: "POST", body });
              return { ok: true, path: rawPath, inspectPath, kind: res.kind, resolvedPath: res.path };
            } catch {
              return { ok: false, path: rawPath, inspectPath };
            }
          })();
          fileRefValidationPending.set(key, task);
          const result = await task;
          fileRefValidationPending.delete(key);
          fileRefValidationCache.set(key, result);
          return result;
        }

        async function upgradeCandidateFileRefs(root) {
          if (!root) return;
          const nodes = Array.from(root.querySelectorAll("[data-candidate-file-path]"));
          for (const node of nodes) {
            const path = String(node.getAttribute("data-candidate-file-path") || "").trim();
            const line = normalizeLineNumber(node.getAttribute("data-candidate-file-line"));
            if (!path) continue;
            const result = await inspectFileRefPath(path);
            if (!result || !result.ok) continue;
            const link = el("a", {
              href: "#",
              class: "inlineFileLink",
              "data-file-path": result.kind === "directory" ? result.resolvedPath || result.inspectPath || path : result.inspectPath || path,
              "data-file-kind": result.kind || "text",
            });
            if (line && result.kind !== "directory") link.setAttribute("data-file-line", String(line));
            link.textContent = node.textContent || path;
            node.replaceWith(link);
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
          setFilePath(activeFilePath || fileCandidateList[0] || "", { line: activeFileLine });
        }

        async function showFileViewer({ path = "", mode = "", manual = false, line = null } = {}) {
          fileBackdrop.style.display = "block";
          fileViewer.style.display = "flex";
          if (mode === "file" || mode === "diff" || mode === "preview") setFileViewMode(mode);
          else applyFileMode();
          await refreshFileCandidates();
          const preferred = String(path || "").trim() || activeFilePath || localStorage.getItem("codexweb.filePath") || "";
          if (preferred) {
            setFilePath(preferred, { line });
            void openFilePath(preferred, { line });
            return;
          }
          const first = fileCandidateList.length ? fileCandidateList[0] : "";
          if (first) {
            setFileViewMode("diff");
            setFilePath(first, { line: null });
            void openFilePath(first, { line: null });
            return;
          }
          setFilePickerButtonContent(null, "No files");
          fileStatus.textContent = "No changed files. Use Add to open a file.";
        }
        function hideFileViewer() {
          disposeFileEditor();
          fileImage.removeAttribute("src");
          fileImage.style.display = "none";
          fileDiff.style.display = "block";
          fileBackdrop.style.display = "none";
          fileViewer.style.display = "none";
          activeFileLine = null;
        }
        async function openFilePath(nextPath = null, { line = undefined } = {}) {
          if (!selected) return;
          const rel = String(nextPath == null ? activeFilePath : nextPath).trim();
          if (!rel) {
            fileStatus.textContent = "Choose a file first.";
            return;
          }
          activeFilePath = rel;
          activeFileLine = line === undefined ? activeFileLine : normalizeLineNumber(line);
          fileStatus.textContent = "Loading...";
          disposeFileEditor();
          fileImage.removeAttribute("src");
          fileImage.style.display = "none";
          fileDiff.style.display = "block";
          try {
            const viewMode = fileViewMode === "preview" && !isMarkdownPreviewable(rel) ? "file" : fileViewMode;
            if (viewMode !== fileViewMode) setFileViewMode(viewMode);
            if (viewMode === "diff") {
              const res = await api(`/api/sessions/${selected}/git/file_versions?path=${encodeURIComponent(rel)}`);
              const baseText = res && typeof res.base_text === "string" ? res.base_text : "";
              const currentText = res && typeof res.current_text === "string" ? res.current_text : "";
              if (!res.base_exists && !res.current_exists) {
                disposeFileEditor();
                fileStatus.textContent = `${rel} - no diff`;
              } else {
                await renderMonacoDiff(rel, baseText, currentText, activeFileLine);
                fileStatus.textContent = `${rel} - diff`;
              }
              rememberOpenedFile(rel, res && typeof res.abs_path === "string" ? res.abs_path : null);
            } else {
              const res = await api(`/api/sessions/${selected}/file/read?path=${encodeURIComponent(rel)}`);
              if (!res || typeof res.kind !== "string") throw new Error("invalid response");
              if (res.kind === "image") {
                if (typeof res.image_url !== "string" || !res.image_url) throw new Error("invalid image response");
                fileDiff.style.display = "none";
                fileImage.src = resolveAppUrl(res.image_url);
                fileImage.alt = rel;
                fileImage.style.display = "block";
                const size = typeof res.size === "number" ? res.size : 0;
                fileStatus.textContent = `${rel} - ${fmtBytes(size)}`;
              } else {
                if (typeof res.text !== "string") throw new Error("invalid response");
                if (viewMode === "preview" && isMarkdownPreviewable(rel)) {
                  renderMarkdownPreview(rel, res.text);
                } else {
                  await renderMonacoFile(rel, res.text, activeFileLine);
                }
                const size = typeof res.size === "number" ? res.size : res.text.length;
                fileStatus.textContent = viewMode === "preview" && isMarkdownPreviewable(rel) ? `${rel} - preview - ${fmtBytes(size)}` : `${rel} - ${fmtBytes(size)}`;
              }
              rememberOpenedFile(rel, typeof res.path === "string" ? res.path : null);
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
          setFileViewMode(fileViewMode === "diff" ? fileNonDiffMode : "diff");
          renderFilePickerMenu();
          void openFilePath(activeFilePath, { line: activeFileLine });
        };
        fileModePreviewBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!isMarkdownPreviewable(activeFilePath)) return;
          setFileViewMode(fileViewMode === "preview" ? "file" : "preview");
          renderFilePickerMenu();
          void openFilePath(activeFilePath, { line: activeFileLine });
        };
        fileDownloadBtn.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!selected || !activeFilePath) return;
          const url = resolveAppUrl(`/api/sessions/${selected}/file/download?path=${encodeURIComponent(activeFilePath)}`);
          const link = document.createElement("a");
          link.href = url;
          link.rel = "noopener";
          link.style.display = "none";
          document.body.appendChild(link);
          link.click();
          link.remove();
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
            setFilePath(rel, { line: null });
            setFileViewMode(isMarkdownPreviewable(rel) && fileNonDiffMode === "preview" ? "preview" : "file");
            void openFilePath(rel, { line: null });
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
        async function openFileReference(ref) {
          if (!ref || typeof ref.path !== "string") return;
          const rawPath = String(ref.path || "").trim();
          const line = normalizeLineNumber(ref.line);
          if (!rawPath) return;
          const parsed = parseLocalFileRef(rawPath);
          if (!parsed) {
            setToast("unsupported file reference");
            return;
          }
          if (!parsed.path.startsWith("/")) {
            if (!selected) {
              setToast("select a session first");
              return;
            }
            void showFileViewer({ path: parsed.path, mode: "file", manual: false, line });
            return;
          }
          if (selected) {
            void showFileViewer({ path: parsed.path, mode: "file", manual: false, line });
            return;
          }
          const currentRel = sessionRelativePath(parsed.path);
          if (currentRel) {
            void showFileViewer({ path: currentRel, mode: "file", manual: false, line });
            return;
          }
          const match = [...sessionIndex.values()].find((s) => {
            const cwd = String(s && s.cwd ? s.cwd : "").replace(/\/+$/, "");
            return cwd && (parsed.path === cwd || parsed.path.startsWith(cwd + "/"));
          });
          if (!match) {
            setToast("file is outside the known session roots");
            return;
          }
          await selectSession(match.session_id);
          const matchRoot = String(match.cwd || "").replace(/\/+$/, "");
          const rel2 = parsed.path === matchRoot ? "." : parsed.path.slice(matchRoot.length + 1);
          void showFileViewer({ path: rel2, mode: "file", manual: false, line });
        }

        async function confirmDirectorySession(rawPath) {
          const cwd = String(rawPath || "").trim();
          if (!cwd) return;
          openNewSessionDialog({
            cwd,
            statusText: "Review resume or worktree options, then start the session.",
          });
        }

        chatInner.addEventListener("click", async (e) => {
          const target = e.target instanceof Element ? e.target.closest("a[data-file-path]") : null;
          if (!target) return;
          e.preventDefault();
          const path = String(target.getAttribute("data-file-path") || "").trim();
          const kind = String(target.getAttribute("data-file-kind") || "").trim();
          const line = normalizeLineNumber(target.getAttribute("data-file-line"));
          if (kind === "directory") {
            await confirmDirectorySession(path);
            return;
          }
          await openFileReference({ path, line });
        });
        document.addEventListener("click", (e) => {
          const t = e.target instanceof Element ? e.target : null;
          if (!t) return;
          if (fileViewer.style.display === "flex" && fileMenuOpen && !t.closest("#fileCandRow")) {
            fileMenuOpen = false;
            applyFileMenuState();
          }
          if (editDependencyMenuOpen && !t.closest("#editDependencyBtn") && !t.closest("#editDependencyMenu")) {
            editDependencyMenuOpen = false;
            applyDialogMenus();
          }
          if (newSessionRecentMenuOpen && !t.closest("#newSessionRecentBtn") && !t.closest("#newSessionRecentMenu")) {
            newSessionRecentMenuOpen = false;
            applyDialogMenus();
          }
          if (newSessionResumeMenuOpen && !t.closest("#newSessionResumeBtn") && !t.closest("#newSessionResumeMenu")) {
            newSessionResumeMenuOpen = false;
            applyDialogMenus();
          }
        });
        document.addEventListener("keydown", (e) => {
          if (e.key !== "Escape") return;
          if (fileViewer.style.display === "flex") hideFileViewer();
          if (sendChoice.style.display === "flex") hideSendChoice();
          if (queueViewer.style.display === "flex") hideQueueViewer();
          if (helpViewer.style.display === "flex") hideHelpViewer();
          if (diagViewer.style.display === "flex") hideDiagViewer();
          if (editViewer.style.display === "flex") hideEditSession();
          if (newSessionViewer.style.display === "flex") hideNewSessionDialog();
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
	            addRow("Branch", d && d.git_branch ? d.git_branch : "-");
	            addRow("Provider", d && d.model_provider ? d.model_provider : "-");
	            addRow("Model", d && d.model ? d.model : "-");
	            addRow("Reasoning", d && d.reasoning_effort ? d.reasoning_effort : "-");
	            addRow("Priority", d && typeof d.final_priority === "number" ? Number(d.final_priority).toFixed(4) : "-");
	            addRow("Priority offset", d && typeof d.priority_offset === "number" ? formatPriorityOffset(d.priority_offset) : "-");
	            addRow("Snooze", d && typeof d.snooze_until === "number" ? fmtTs(d.snooze_until) : "-");
	            addRow("Depends on", d && d.dependency_session_id ? d.dependency_session_id : "-");
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
        async function spawnSessionWithCwd(cwd, resumeSessionId = null, worktreeBranch = null) {
          if (!cwd || !String(cwd).trim()) {
            setToast("cwd unavailable");
            return null;
          }
          try {
            const modeLabel = resumeSessionId ? "resuming..." : worktreeBranch ? "creating worktree..." : "starting...";
            setToast(modeLabel);
            const body = { cwd: String(cwd) };
            if (resumeSessionId) body.resume_session_id = String(resumeSessionId);
            if (worktreeBranch) body.worktree_branch = String(worktreeBranch);
            const res = await api("/api/sessions", { method: "POST", body });
            const brokerPid = res && res.broker_pid ? Number(res.broker_pid) : null;
            if (!brokerPid) {
              setToast("start failed");
              return null;
            }
            const doneLabel = resumeSessionId ? "resumed" : worktreeBranch ? "worktree started" : "started";
            setToast(`${doneLabel} (broker ${brokerPid})`);
            for (let i = 0; i < 60; i++) {
              const sessions = await refreshSessions();
              const found = (sessions || []).find((x) => Number(x.broker_pid || 0) === brokerPid);
              if (found) {
                selectSession(found.session_id);
                return brokerPid;
              }
              await new Promise((r) => setTimeout(r, 250));
            }
            setToast(`${doneLabel} session will appear once Codex creates a rollout log`);
            return brokerPid;
          } catch (e) {
            const errLabel = resumeSessionId ? "resume" : worktreeBranch ? "worktree start" : "start";
            setToast(`${errLabel} error: ${e.message}`);
            return null;
          }
        }
        $("#newBtn").onclick = async () => {
          openNewSessionDialog();
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
	         function activeTextEntryElement() {
	           const active = document.activeElement;
	           return isTextEntryElement(active) ? active : null;
	         }
	         let iosViewportGuardTimer = null;
	         let iosViewportGuardUntil = 0;
	         function normalizePageScroll() {
	           if (!isIOS) return;
	           const activeEntry = activeTextEntryElement();
	           if (activeEntry && activeEntry !== textarea) return;
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
	             const activeEntry = activeTextEntryElement();
	             if (activeEntry && activeEntry !== textarea) {
	               stopIOSViewportGuard();
	               return;
	             }
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
	             const activeEntry = activeTextEntryElement();
	             if (activeEntry && activeEntry !== textarea) {
	               stopIOSViewportGuard();
	               return;
	             }
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
          attachedFiles = next;
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
	              setTimeout(() => {
	                if (isIOS) {
	                  const activeEntry = activeTextEntryElement();
	                  if (activeEntry && activeEntry !== textarea) {
	                    stopIOSViewportGuard();
	                    updateAppHeightVar();
	                    return;
	                  }
	                  runIOSViewportGuard({ preserveChatBottom: false, durationMs: 900 });
	                  return;
	                }
	                updateAppHeightVar();
	              }, 0);
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
	              const s = String(name || "file");
	              const base = s.split("/").pop() || s;
	              const dot = base.lastIndexOf(".");
	              return (dot > 0 ? base.slice(0, dot) : base).replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 80) || "file";
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
	            function looksLikeImage(file) {
	              const t = String(file && file.type ? file.type : "").toLowerCase();
	              if (t.startsWith("image/")) return true;
	              const e = extLower(file && file.name ? file.name : "");
	              return ["png", "jpg", "jpeg", "webp", "gif", "bmp", "svg", "avif", "heic", "heif"].includes(e);
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

	            setToast("uploading file...");
	            const maxBytes = 10 * 1024 * 1024;
	            let uploadBlob = f;
	            let uploadName = f.name || "file";
	            if (looksLikeImage(f) && (f.size > maxBytes || isLikelyHeic(f))) {
	              setToast("compressing image...");
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
	            if (ab.byteLength > maxBytes) throw new Error("file too large");
		            const b64 = b64FromBytes(new Uint8Array(ab));
			            const res = await api(`/api/sessions/${selected}/inject_image`, {
		              method: "POST",
		              body: { filename: uploadName, data_b64: b64, attachment_index: attachedFiles + 1 },
		            });
		            if (res && res.ok) {
		              setToast("file attached");
		              setAttachCount(attachedFiles + 1);
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
