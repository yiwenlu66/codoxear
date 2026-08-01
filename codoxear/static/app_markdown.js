(function () {
  "use strict";

  // KaTeX loads independently because rendered transcript rows can arrive before
  // the library does. Until then, renderMath leaves readable source in place.
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
    "7z", "3gp", "avi", "bash", "bin", "bz2", "c", "cc", "cfg", "conf", "cpp", "css", "csv", "flv", "gif", "go", "gz", "h", "hpp", "html", "htm", "ico", "ini", "java", "jpeg", "jpg", "js", "json", "jsonl", "log", "m4v", "md", "mkv", "mov", "mp4", "mpeg", "mpg", "ogv", "pdf", "patch", "png", "py", "rs", "scss", "sh", "sql", "svg", "tar", "tgz", "toml", "ts", "tsx", "txt", "webm", "webp", "wmv", "xml", "xz", "yaml", "yml", "zip", "zsh",
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
    return path ? { path, line } : null;
  }

  function parseLocalFileRef(rawValue) {
    const parsed = parseFileLocation(rawValue);
    if (!parsed) return null;
    const path = parsed.path;
    if (path.includes("://") || path.startsWith("mailto:") || path.startsWith("//")) return null;
    const looksAbsolute = path.startsWith("/");
    const looksRelative = path.startsWith("./") || path.startsWith("../") || path.includes("/");
    const looksBareFile = !looksAbsolute && !looksRelative && hasClickableFileExtension(path);
    return looksAbsolute || looksRelative || looksBareFile ? { path, line: parsed.line } : null;
  }

  function normalizePathLike(rawPath) {
    const raw = String(rawPath || "").trim();
    if (!raw) return "";
    const absolute = raw.startsWith("/");
    const out = [];
    for (const part of raw.split("/")) {
      if (!part || part === ".") continue;
      if (part === "..") {
        if (out.length && out[out.length - 1] !== "..") out.pop();
        else if (!absolute) out.push("..");
      } else out.push(part);
    }
    const joined = out.join("/");
    return absolute ? (joined ? `/${joined}` : "/") : joined || ".";
  }

  function pathDirname(rawPath) {
    const raw = normalizePathLike(rawPath);
    if (!raw || raw === ".") return ".";
    if (raw === "/") return "/";
    const idx = raw.lastIndexOf("/");
    return idx < 0 ? "." : idx === 0 ? "/" : raw.slice(0, idx);
  }

  function resolveRelativePath(basePath, rawPath) {
    const raw = String(rawPath || "").trim();
    if (!raw) return "";
    if (raw.startsWith("/")) return normalizePathLike(raw);
    const baseDir = pathDirname(basePath);
    if (!baseDir || baseDir === ".") return normalizePathLike(raw);
    return normalizePathLike(baseDir === "/" ? `/${raw}` : `${baseDir}/${raw}`);
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
    const blockRe = /<oai-mem-citation>\s*<citation_entries>\s*([\s\S]*?)\s*<\/citation_entries>\s*<rollout_ids>[\s\S]*?<\/rollout_ids>\s*<\/oai-mem-citation>/g;
    return raw.replace(blockRe, (whole, body) => {
      const lines = String(body || "").split("\n").map((line) => line.trim()).filter(Boolean);
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
        const range = endLine && endLine >= startLine ? `#L${startLine}-${endLine}` : `#L${startLine}`;
        items.push(`[${note}](~/.codex/memories/${relPath}${range})`);
      }
      return `\n---\n\nMemory citations:\n${items.map((item, index) => `${index + 1}. ${item}`).join("\n")}`;
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
      const parsed = parseLocalFileRef(`${decodeURIComponent(url.pathname || "")}${url.hash || ""}`);
      if (parsed && parsed.path.startsWith("/") && /^\/(?:home|tmp|mnt|var|opt|usr|etc|private|Users|Volumes)\//.test(parsed.path)) {
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

  const FILE_REF_TEXT_RE = /(^|[\s([{"'])((?:\/[A-Za-z0-9._~@%+=:,/-]+|(?:\.{1,2}\/)?[A-Za-z0-9._~@-]+(?:\/[A-Za-z0-9._~@-]+)+|[A-Za-z0-9._~@-]+\.[A-Za-z0-9._-]+)(?:#L\d+(?:-\d+)?)?(?::\d+(?::\d+)?)?)(?=$|[\s)\]}:;"',!?])/g;

  function createFileRefSpan(doc, ref, label) {
    const span = doc.createElement("span");
    span.dataset.candidateFilePath = ref.path;
    if (ref.line) span.dataset.candidateFileLine = String(ref.line);
    span.textContent = label;
    return span;
  }

  function nodeHasAncestor(node, tagName) {
    for (let current = node.parentElement; current; current = current.parentElement) {
      if (current.tagName === tagName) return true;
    }
    return false;
  }

  function rewriteTextFileRefs(root, doc, options) {
    const walker = doc.createTreeWalker(root, 4);
    const textNodes = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (!nodeHasAncestor(node, "A") && !nodeHasAncestor(node, "CODE") && !nodeHasAncestor(node, "SCRIPT") && !nodeHasAncestor(node, "STYLE")) textNodes.push(node);
    }
    for (const textNode of textNodes) {
      const raw = textNode.nodeValue || "";
      FILE_REF_TEXT_RE.lastIndex = 0;
      let last = 0;
      let changed = false;
      const fragment = doc.createDocumentFragment();
      for (let match = FILE_REF_TEXT_RE.exec(raw); match; match = FILE_REF_TEXT_RE.exec(raw)) {
        const tokenStart = match.index + match[1].length;
        const token = match[2];
        const ref = resolveLocalRefWithOptions(parseLocalFileRef(token), options);
        if (!ref) continue;
        changed = true;
        fragment.append(raw.slice(last, tokenStart));
        fragment.append(createFileRefSpan(doc, ref, token));
        last = tokenStart + token.length;
      }
      if (changed) {
        fragment.append(raw.slice(last));
        textNode.replaceWith(fragment);
      }
    }
  }

  function rewriteMarkedLinks(root, doc, options) {
    for (const link of root.querySelectorAll("a[href]")) {
      const rawHref = link.getAttribute("href") || "";
      const localRef = localFileRefFromRef(rawHref, options);
      if (localRef) {
        link.replaceWith(createFileRefSpan(doc, localRef, formatLocalFileLinkLabel(link.textContent, rawHref, localRef)));
        continue;
      }
      const href = safeUrl(rawHref);
      if (!href) {
        link.replaceWith(doc.createTextNode(`${link.textContent} (${rawHref})`));
        continue;
      }
      link.href = href;
      link.target = "_blank";
      link.rel = "noreferrer noopener";
    }
  }

  function rewriteMarkedImages(root, doc, options) {
    for (const image of root.querySelectorAll("img[src]")) {
      const rawSrc = image.getAttribute("src") || "";
      const localRef = localFileRefFromRef(rawSrc, options);
      const src = options && typeof options.resolveImageSrc === "function" ? options.resolveImageSrc(rawSrc, localRef) : safeUrl(rawSrc);
      if (!src) image.replaceWith(doc.createTextNode(image.alt || ""));
      else image.src = src;
      image.loading = "lazy";
    }
  }

  function decorateCodeBlocks(root, doc) {
    for (const pre of root.querySelectorAll("pre")) {
      const code = pre.querySelector(":scope > code");
      if (!code) continue;
      const languageClass = Array.from(code.classList).find((value) => value.startsWith("language-"));
      if (languageClass) code.dataset.lang = languageClass.slice("language-".length);
      if (!pre.querySelector(":scope > .code-copy-btn")) {
        const button = doc.createElement("button");
        button.className = "code-copy-btn";
        button.type = "button";
        button.setAttribute("aria-label", "Copy code");
        button.title = "Copy code";
        pre.prepend(button);
      }
    }
  }

  function rewriteInlineCodeFileRefs(root, doc, options) {
    for (const code of root.querySelectorAll("code")) {
      if (nodeHasAncestor(code, "PRE")) continue;
      const ref = resolveLocalRefWithOptions(parseLocalFileRef(code.textContent), options);
      if (!ref) continue;
      code.replaceChildren(createFileRefSpan(doc, ref, code.textContent));
    }
  }

  function wrapMarkedTables(root, doc) {
    for (const table of root.querySelectorAll("table")) {
      if (table.parentElement && table.parentElement.classList.contains("md-table-wrap")) continue;
      const wrapper = doc.createElement("div");
      wrapper.className = "md-table-wrap";
      table.replaceWith(wrapper);
      wrapper.append(table);
    }
  }

  function postProcessMarkedHtml(html, options) {
    if (typeof document === "undefined" || !document.createElement) return html;
    const template = document.createElement("template");
    template.innerHTML = html;
    const root = template.content;
    rewriteMarkedLinks(root, document, options);
    rewriteMarkedImages(root, document, options);
    rewriteInlineCodeFileRefs(root, document, options);
    rewriteTextFileRefs(root, document, options);
    decorateCodeBlocks(root, document);
    wrapMarkedTables(root, document);
    return template.innerHTML;
  }

  // Math tokens are opaque to marked and are substituted only after its HTML has
  // been passed through the Codoxear DOM post-processors. Fence/code-span
  // protection prevents source code from becoming mathematical markup.
  const MATH_TOKEN_PREFIX = "@@MATH";
  const MATH_TOKEN_SUFFIX = "@@";
  function mathToken(id) {
    return `${MATH_TOKEN_PREFIX}${id}${MATH_TOKEN_SUFFIX}`;
  }

  function renderMath(latex, displayMode) {
    const src = String(latex ?? "");
    if (typeof katex !== "undefined" && katex && typeof katex.renderToString === "function") {
      try {
        return String(katex.renderToString(src, { displayMode: !!displayMode, throwOnError: false, strict: "ignore" }));
      } catch {}
    }
    const cls = displayMode ? "md-math-fallback md-math-display" : "md-math-fallback md-math-inline";
    return `<span class="${cls}">${displayMode ? "\\[" : "\\("}${escapeHtml(src)}${displayMode ? "\\]" : "\\)"}</span>`;
  }

  function extractMathFromPlainText(text, store) {
    const push = (latex, display) => {
      const id = store.length;
      store.push({ latex: String(latex).trim(), display: !!display });
      return mathToken(id);
    };
    return String(text)
      .replace(/\\\[([\s\S]+?)\\\]/g, (_m, body) => push(body, true))
      .replace(/\$\$([\s\S]+?)\$\$/g, (_m, body) => push(body, true))
      .replace(/\\\(([\s\S]+?)\\\)/g, (_m, body) => push(body, false))
      .replace(/\$(?!\$|\s)([^$\n]*?\S)\$(?!\$)/g, (_m, body) => push(body, false));
  }

  function extractMathFromText(input, store) {
    const lines = String(input ?? "").split("\n");
    const out = [];
    let fence = null;
    for (const line of lines) {
      const openOrClose = line.match(/^\s{0,3}(`{3,}|~{3,})/);
      if (fence) {
        out.push(line);
        if (openOrClose && openOrClose[1][0] === fence) fence = null;
        continue;
      }
      if (openOrClose) {
        fence = openOrClose[1][0];
        out.push(line);
        continue;
      }
      const parts = line.split(/(`+[^`]*`+)/g);
      out.push(parts.map((part, index) => (index % 2 ? part : extractMathFromPlainText(part, store))).join(""));
    }
    return out.join("\n");
  }

  function substituteMath(html, store) {
    let out = html;
    for (let i = 0; i < store.length; i++) out = out.split(mathToken(i)).join(renderMath(store[i].latex, store[i].display));
    return out;
  }

  function mdToHtml(src, options = null) {
    if (!window.marked || typeof window.marked.parse !== "function") throw new Error("marked failed to load");
    const mathStore = [];
    const prepared = extractMathFromText(rewriteOaiMemCitations(String(src ?? "").replaceAll("\r\n", "\n")), mathStore);
    return substituteMath(postProcessMarkedHtml(window.marked.parse(prepared), options), mathStore);
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
    if (mdCache.size > 1200) mdCache.clear();
    return html;
  }

  function isMarkdownPreviewable(path) {
    return ["md", "markdown", "mdown", "mkd"].includes(filePathExtension(path));
  }

  function previewImageUrlForRef(rawRef, localRef, { filePath, sessionId } = {}) {
    if (localRef && localRef.path) {
      if (sessionId) return resolveAppUrl(`/api/sessions/${sessionId}/file/blob?path=${encodeURIComponent(localRef.path)}`);
      if (localRef.path.startsWith("/")) return resolveAppUrl(`/api/files/blob?path=${encodeURIComponent(localRef.path)}`);
    }
    return safeUrl(rawRef);
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
