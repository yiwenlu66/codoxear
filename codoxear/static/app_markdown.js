(function () {
  "use strict";

  // Dynamically load KaTeX from CDN for math rendering. The script/style are
  // injected at runtime so index.html stays CDN-free and CSP loads are explicit.
  (function loadKatex() {
    if (window.__codoxearKatexLoading) return;
    window.__codoxearKatexLoading = true;
    if (typeof document === "undefined" || !document.createElement) return;
    const base = "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/";
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = base + "katex.min.css";
    link.crossOrigin = "anonymous";
    document.head.appendChild(link);
    const script = document.createElement("script");
    script.src = base + "katex.min.js";
    script.defer = true;
    document.head.appendChild(script);
  })();

  const codoxearUrls = window.CodoxearUrls;
  if (!codoxearUrls || typeof codoxearUrls.resolveAppUrl !== "function") throw new Error("Codoxear URL helpers failed to load");
  function resolveAppUrl(path) {
    return codoxearUrls.resolveAppUrl(path);
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
    "7z",
    "3gp",
    "avi",
    "bash",
    "bin",
    "bz2",
    "c",
    "cc",
    "cfg",
    "conf",
    "cpp",
    "css",
    "csv",
    "flv",
    "gif",
    "go",
    "gz",
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
    "m4v",
    "md",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ogv",
    "pdf",
    "patch",
    "png",
    "py",
    "rs",
    "scss",
    "sh",
    "sql",
    "svg",
    "tar",
    "tgz",
    "toml",
    "ts",
    "tsx",
    "txt",
    "webm",
    "webp",
    "wmv",
    "xml",
    "xz",
    "yaml",
    "yml",
    "zip",
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
    const raw = String(rawPath ?? "");
    const trimmed = raw.trim();
    let m = trimmed.match(/^(.*)#L(\d+)(?:-\d+)?$/);
    if (m) return m[1];
    m = trimmed.match(/^(.*):(\d+)(?::\d+)?$/);
    if (m && !/^[A-Za-z]:$/.test(m[1])) return m[1];
    return raw;
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

  function rewriteOaiMemCitations(rawText) {
    const raw = String(rawText ?? "");
    if (!raw.includes("<oai-mem-citation>")) return raw;
    const blockRe =
      /<oai-mem-citation>\s*<citation_entries>\s*([\s\S]*?)\s*<\/citation_entries>\s*<rollout_ids>[\s\S]*?<\/rollout_ids>\s*<\/oai-mem-citation>/g;
    return raw.replace(blockRe, (whole, body) => {
      const lines = String(body || "")
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
      if (!lines.length) return whole;
      const items = [];
      for (const line of lines) {
        const m = line.match(/^(.*?):(\d+)(?:-(\d+))?\|note=\[(.*)\]$/);
        if (!m) return whole;
        const relPath = String(m[1] || "").trim().replace(/^\.?\//, "");
        const startLine = normalizeLineNumber(m[2]);
        const endLine = normalizeLineNumber(m[3]);
        const note = String(m[4] || "").trim();
        if (!relPath || !startLine || !note) return whole;
        const rangeSuffix = endLine && endLine >= startLine ? `#L${startLine}-${endLine}` : `#L${startLine}`;
        const target = `~/.codex/memories/${relPath}${rangeSuffix}`;
        items.push(`[${note}](${target})`);
      }
      return `\n---\n\nMemory citations:\n${items.map((item, idx) => `${idx + 1}. ${item}`).join("\n")}`;
    });
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

  // Math is extracted from non-code text before markdown rendering and replaced
  // with opaque tokens. The ASCII tokens survive escaping and nested mdToHtml
  // calls; KaTeX output is substituted only after the HTML is assembled.
  const MATH_TOKEN_PREFIX = "@@MATH";
  const MATH_TOKEN_SUFFIX = "@@";

  function mathToken(id) {
    return `${MATH_TOKEN_PREFIX}${id}${MATH_TOKEN_SUFFIX}`;
  }

  function renderMath(latex, displayMode) {
    const src = String(latex ?? "");
    if (typeof katex !== "undefined" && katex && typeof katex.renderToString === "function") {
      try {
        return String(
          katex.renderToString(src, {
            displayMode: !!displayMode,
            throwOnError: false,
            strict: "ignore",
          })
        );
      } catch (e) {
        // Fall through to the readable source fallback when KaTeX rejects input.
      }
    }
    const cls = displayMode ? "md-math-fallback md-math-display" : "md-math-fallback md-math-inline";
    const open = displayMode ? "\\[" : "\\(";
    const close = displayMode ? "\\]" : "\\)";
    return `<span class="${cls}">${open}${escapeHtml(src)}${close}</span>`;
  }

  function extractMathFromText(input, store) {
    let text = String(input ?? "");
    const push = (latex, display) => {
      const id = store.length;
      store.push({ latex: String(latex).trim(), display: !!display });
      return mathToken(id);
    };
    // Match displays first so their delimiters cannot be consumed by inline rules.
    text = text.replace(/\\\[([\s\S]+?)\\\]/g, (_m, body) => push(body, true));
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_m, body) => push(body, true));
    // Single-$ inline math remains deliberately unsupported: in prose it is
    // ambiguous with currency, shell variables, and code. Explicit \(...\) is
    // unambiguous and is the form emitted by well-formed math output.
    text = text.replace(/\\\(([\s\S]+?)\\\)/g, (_m, body) => push(body, false));
    return text;
  }

  function substituteMath(html, store) {
    if (!store.length) return html;
    let out = html;
    for (let i = 0; i < store.length; i++) {
      const entry = store[i];
      out = out.split(mathToken(i)).join(renderMath(entry.latex, entry.display));
    }
    return out;
  }

  function mdToHtml(src, options = null) {
    const s = rewriteOaiMemCitations(String(src ?? "").replaceAll("\r\n", "\n"));
    const listItemInfo = (line) => {
      const l = String(line ?? "");
      const mUl = l.match(/^(\s*)([-*\u2022])(\s+)(.*)$/);
      if (mUl) {
        return {
          type: "ul",
          indent: mUl[1].length,
          contentIndent: mUl[1].length + mUl[2].length + mUl[3].length,
          text: (mUl[4] || "").trimStart(),
        };
      }
      const mOl = l.match(/^(\s*)(\d+\.)(\s+)(.*)$/);
      if (mOl) {
        return {
          type: "ol",
          indent: mOl[1].length,
          contentIndent: mOl[1].length + mOl[2].length + mOl[3].length,
          marker: mOl[2],
          text: (mOl[4] || "").trimStart(),
        };
      }
      return null;
    };

    const leadingSpaceCount = (line) => {
      const raw = String(line ?? "");
      let i = 0;
      while (i < raw.length && raw[i] === " ") i += 1;
      return i;
    };

    const stripContinuationIndent = (line, width) => {
      const raw = String(line ?? "");
      let i = 0;
      while (i < raw.length && i < width && raw[i] === " ") i += 1;
      return raw.slice(i);
    };

    const fenceOpenInfo = (line) => {
      const m = String(line ?? "").match(/^\s{0,3}```\s*([a-zA-Z0-9_-]+)?\s*$/);
      return m ? { lang: m[1] || "" } : null;
    };

    const isFenceClose = (line) => /^\s{0,3}```\s*$/.test(String(line ?? ""));

    const pendingListFenceIndent = (priorLines, line) => {
      const indent = leadingSpaceCount(line);
      if (!indent) return null;
      for (let j = priorLines.length - 1; j >= 0; j--) {
        const prev = priorLines[j] || "";
        if (!prev.trim()) continue;
        const info = listItemInfo(prev);
        if (info) {
          const contentIndent = info.contentIndent || 0;
          return indent >= contentIndent && fenceOpenInfo(stripContinuationIndent(line, contentIndent)) ? contentIndent : null;
        }
        if (leadingSpaceCount(prev) < indent) return null;
      }
      return null;
    };

    const continuesPriorList = (priorLines, line) => {
      const next = listItemInfo(line);
      if (!next) return false;
      for (let j = priorLines.length - 1; j >= 0; j--) {
        const prev = priorLines[j] || "";
        if (!prev.trim()) continue;
        const info = listItemInfo(prev);
        if (info) return next.indent >= info.indent;
        if (leadingSpaceCount(prev) < next.indent) return false;
      }
      return false;
    };

    const splitByFences = (input) => {
      const chunks = [];
      const lines = String(input ?? "").split("\n");
      let textLines = [];
      let inFence = false;
      let fenceLang = "";
      let fenceLines = [];
      let fenceStart = "";
      let deferredFenceIndent = null;

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
        if (deferredFenceIndent !== null) {
          textLines.push(line);
          if (isFenceClose(stripContinuationIndent(line, deferredFenceIndent))) deferredFenceIndent = null;
          continue;
        }
        if (!inFence) {
          const m = fenceOpenInfo(line);
          const nestedIndent = pendingListFenceIndent(textLines, line);
          if (m || nestedIndent !== null) {
            if (nestedIndent !== null) {
              deferredFenceIndent = nestedIndent;
              textLines.push(line);
              continue;
            }
            flushText();
            inFence = true;
            fenceLang = m.lang || "";
            fenceStart = line;
            fenceLines = [];
            continue;
          }
          textLines.push(line);
          continue;
        }
        if (isFenceClose(line)) {
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

    const parseIndentedFence = (lines, start, contentIndent) => {
      const open = fenceOpenInfo(stripContinuationIndent(lines[start], contentIndent));
      if (!open) return null;
      const codeLines = [];
      let i = start + 1;
      while (i < lines.length) {
        const stripped = stripContinuationIndent(lines[i], contentIndent);
        if (isFenceClose(stripped)) {
          return { node: { type: "code", lang: open.lang, value: codeLines.join("\n") }, next: i + 1 };
        }
        codeLines.push(stripped);
        i += 1;
      }
      return null;
    };

    const renderCodeBlock = (value, lang) => {
      const langAttr = lang ? ` data-lang="${escapeHtml(lang)}"` : "";
      const copyButton = '<button class="code-copy-btn" type="button" aria-label="Copy code" title="Copy code"></button>';
      return `<pre>${copyButton}<code${langAttr}>${escapeHtml(value)}</code></pre>`;
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
        if (!info) {
          const last = items[items.length - 1];
          if (last && !String(lines[i] || "").trim()) {
            let j = i + 1;
            while (j < lines.length && !String(lines[j] || "").trim()) j += 1;
            const nextFence = j < lines.length ? parseIndentedFence(lines, j, last.contentIndent || baseIndent) : null;
            if (nextFence) {
              last.blocks.push(nextFence.node);
              i = nextFence.next;
              continue;
            }
            const nextInfo = j < lines.length ? listItemInfo(lines[j]) : null;
            if (nextInfo && nextInfo.indent >= baseIndent) {
              i = j;
              continue;
            }
          }
          const fence = last ? parseIndentedFence(lines, i, last.contentIndent || baseIndent) : null;
          if (!fence) break;
          last.blocks.push(fence.node);
          i = fence.next;
          continue;
        }
        if (info.indent < baseIndent) break;
        if (info.indent > baseIndent) {
          if (!items.length) break;
          const child = parseList(lines, i);
          items[items.length - 1].child = child.node;
          i = child.next;
          continue;
        }
        if (info.type !== listType) break;
        items.push({ text: info.text, marker: info.marker || "", contentIndent: info.contentIndent, child: null, blocks: [] });
        i += 1;
      }
      return { node: { type: listType, items }, next: i };
    };

    const renderList = (node) => {
      const out = [];
      out.push(node.type === "ol" ? '<ol class="md-literal-ol">' : "<ul>");
      for (const it of node.items) {
        out.push("<li>");
        if (node.type === "ol") {
          out.push('<span class="md-list-line">');
          out.push(`<span class="md-list-marker">${escapeHtml(it.marker || "")}</span>`);
          out.push(`<span class="md-list-body">${renderInlineMd(it.text || "", options)}</span>`);
          out.push("</span>");
        } else {
          out.push(renderInlineMd(it.text || "", options));
        }
        for (const block of it.blocks || []) {
          if (block.type === "code") out.push(renderCodeBlock(block.value, block.lang));
        }
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

    const blockquoteInfo = (line) => {
      const m = String(line ?? "").match(/^\s{0,3}>(?:[ \t]?)(.*)$/);
      return m ? { text: m[1] || "" } : null;
    };

    const parseBlockquote = (lines, start) => {
      const quoteLines = [];
      let i = start;
      while (i < lines.length) {
        const line = lines[i] || "";
        const info = blockquoteInfo(line);
        if (info) {
          quoteLines.push(info.text);
          i += 1;
          continue;
        }
        // CommonMark allows lazy continuation lines inside a block quote paragraph.
        if (quoteLines.length && line.trim()) {
          quoteLines.push(line);
          i += 1;
          continue;
        }
        break;
      }
      return { node: { type: "blockquote", value: quoteLines.join("\n") }, next: i };
    };

    const renderBlockquote = (node) => `<blockquote>${mdToHtml(node.value || "", options)}</blockquote>`;

    const splitTextBlocks = (input) => {
      const blocks = [];
      const lines = String(input ?? "").split("\n");
      let current = [];
      let inFence = false;
      let currentFenceIndent = 0;
      const flush = () => {
        const block = current.join("\n");
        current = [];
        if (block.trim()) blocks.push(block);
      };
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx];
        const stripped = line.trim();
        if (!inFence && !stripped) {
          let j = idx + 1;
          while (j < lines.length && !String(lines[j] || "").trim()) j += 1;
          if (
            j < lines.length &&
            (pendingListFenceIndent(current, lines[j]) !== null || continuesPriorList(current, lines[j]))
          ) {
            current.push(line);
            continue;
          }
          flush();
          continue;
        }
        const nestedIndent = pendingListFenceIndent(current, line);
        const open = fenceOpenInfo(line);
        if (!inFence && open) {
          inFence = true;
          currentFenceIndent = 0;
          current.push(line);
          continue;
        }
        if (!inFence && nestedIndent !== null) {
          inFence = true;
          currentFenceIndent = nestedIndent;
          current.push(line);
          continue;
        }
        if (inFence && isFenceClose(stripContinuationIndent(line, currentFenceIndent))) {
          inFence = false;
          currentFenceIndent = 0;
          current.push(line);
          continue;
        }
        current.push(line);
      }
      flush();
      return blocks;
    };

    const chunks = splitByFences(s);
    const mathStore = [];
    for (const chunk of chunks) {
      if (chunk.type === "text") chunk.value = extractMathFromText(chunk.value, mathStore);
    }

    const out = [];
    for (const c of chunks) {
      if (c.type === "code") {
        out.push(renderCodeBlock(c.value, c.lang));
        continue;
      }
      const blocks = splitTextBlocks(c.value);
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
          if (/^(?:-{3,}|\*{3,}|_{3,})$/.test(t.replace(/\s+/g, ""))) {
            flushPara();
            out.push("<hr />");
            continue;
          }
          if (blockquoteInfo(l)) {
            flushPara();
            const parsed = parseBlockquote(lines, i);
            out.push(renderBlockquote(parsed.node));
            i = parsed.next - 1;
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
    return substituteMath(out.join(""), mathStore);
  }

  const mdCache = new Map();
  function mdToHtmlCached(src, options = null) {
    const text = String(src ?? "");
    const scope = options && typeof options.cacheKey === "string" ? options.cacheKey : "";
    const key = `${scope}\0${text}`;
    const hit = mdCache.get(key);
    if (hit !== undefined) return hit;
    const html = mdToHtml(text, options);
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

  function chatMarkdownHtmlCached(src, sessionId) {
    const sid = String(sessionId || "").trim();
    return mdToHtmlCached(src, {
      cacheKey: sid ? `chat:${sid}` : "chat",
      resolveImageSrc(rawRef, localRef) {
        return previewImageUrlForRef(rawRef, localRef, { sessionId: sid });
      },
    });
  }

  window.CodoxearMarkdown = Object.freeze({
    escapeHtml,
    mdToHtml,
    mdToHtmlCached,
    normalizeLineNumber,
    parseLocalFileRef,
    isMarkdownPreviewable,
    markdownPreviewHtml,
    chatMarkdownHtmlCached,
  });
})();
