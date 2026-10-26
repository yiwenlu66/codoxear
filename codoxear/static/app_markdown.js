import * as CodoxearUrls from "./app_application.js";


// KaTeX is loaded via index.html <script> BEFORE Monaco's loader.js.
  // This ordering ensures KaTeX's UMD wrapper takes the browser global
  // path (window.katex) instead of the AMD path (define).

  function resolveAppUrl(path) {
    return CodoxearUrls.resolveAppUrl(path);
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
    "7z", "3gp", "avi", "bash", "bin", "bz2", "c", "cc", "cfg", "conf", "cpp", "css", "csv", "doc", "docx", "flv", "gif", "go", "gz", "h", "hpp", "html", "htm", "ico", "ini", "java", "jpeg", "jpg", "js", "json", "jsonl", "log", "m4v", "md", "mkv", "mov", "mp4", "mpeg", "mpg", "ogv", "pdf", "patch", "ppt", "pptx", "png", "py", "rs", "scss", "sh", "sql", "svg", "tar", "tgz", "toml", "ts", "tsx", "txt", "webm", "webp", "wmv", "xls", "xlsx", "xml", "xz", "yaml", "yml", "zip", "zsh",
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
    // A leading `@` is a mention marker, never path syntax: the message keeps
    // "@src/a.js" while the link resolves "src/a.js". `./@foo/x.md` still
    // addresses a directory literally named `@foo`.
    const path = parsed.path.startsWith("@") ? parsed.path.slice(1) : parsed.path;
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

  const FILE_REF_TEXT_RE = /(^|[\s([{"'@])((?:\/[A-Za-z0-9._~@%+=:,/-]+|(?:\.{1,2}\/)?[A-Za-z0-9._~@-]+(?:\/[A-Za-z0-9._~@-]+)+|[A-Za-z0-9._~@-]+\.[A-Za-z0-9._-]+)(?:#L\d+(?:-\d+)?)?(?::\d+(?::\d+)?)?)(?=$|[\s)\]}:;"',!?])/g;

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
      if (!nodeHasAncestor(node, "A") && !nodeHasAncestor(node, "CODE") && !nodeHasAncestor(node, "SCRIPT") && !nodeHasAncestor(node, "STYLE") && !nodeHasAncestor(node, "VIDEO")) textNodes.push(node);
    }
    for (const textNode of textNodes) {
      const raw = textNode.nodeValue || "";
      FILE_REF_TEXT_RE.lastIndex = 0;
      let last = 0;
      let changed = false;
      const fragment = doc.createDocumentFragment();
      for (let match = FILE_REF_TEXT_RE.exec(raw); match; match = FILE_REF_TEXT_RE.exec(raw)) {
        let tokenStart = match.index + match[1].length;
        let token = match[2];
        if (match[1] === "@") {
          // The `@` is itself the boundary (CJK prose has no spaces), so absorb
          // it into the span: the whole mention stays clickable and its label
          // keeps the `@`. This branch cannot reach an email local part, because
          // `foo@bar.com` matches as one token from its own start and is never
          // stripped.
          tokenStart -= 1;
          token = `@${token}`;
        }
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

  const IMAGE_DIMENSION_CACHE_KEY = "codoxear.image-dimensions.v1";
  const IMAGE_DIMENSION_CACHE_LIMIT = 500;
  const DEFAULT_IMAGE_DIMENSIONS = Object.freeze({ width: 16, height: 9 });
  const imageDimensionRequests = new Map();
  let imageHydrationScheduled = false;

  function imageDimensionStorage() {
    try {
      const storage = window.localStorage;
      return storage && typeof storage.getItem === "function" && typeof storage.setItem === "function" ? storage : null;
    } catch {
      return null;
    }
  }

  function readImageDimensionCache() {
    const storage = imageDimensionStorage();
    if (!storage) return {};
    try {
      const parsed = JSON.parse(storage.getItem(IMAGE_DIMENSION_CACHE_KEY) || "{}");
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }

  function writeImageDimensionCache(entries) {
    const storage = imageDimensionStorage();
    if (!storage) return;
    const valid = Object.entries(entries)
      .filter(([, value]) => {
        const width = Number(value && value.width);
        const height = Number(value && value.height);
        return Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0;
      })
      .sort(([, a], [, b]) => Number(b.usedAt || 0) - Number(a.usedAt || 0))
      .slice(0, IMAGE_DIMENSION_CACHE_LIMIT);
    try {
      storage.setItem(IMAGE_DIMENSION_CACHE_KEY, JSON.stringify(Object.fromEntries(valid)));
    } catch {}
  }

  function nextImageUse(entries) {
    const latest = Object.values(entries).reduce((max, entry) => Math.max(max, Number(entry && entry.usedAt) || 0), 0);
    return Math.max(Date.now(), latest + 1);
  }

  function cachedImageDimensions(source) {
    const entries = readImageDimensionCache();
    const entry = entries[source];
    const width = Number(entry && entry.width);
    const height = Number(entry && entry.height);
    if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) return null;
    entries[source] = { width, height, usedAt: nextImageUse(entries) };
    writeImageDimensionCache(entries);
    return { width, height };
  }

  function applyImageDimensions(image, source, dimensions) {
    image.style.aspectRatio = `${dimensions.width} / ${dimensions.height}`;
    image.style.width = "100%";
    image.style.height = "auto";
    image.setAttribute("width", String(dimensions.width));
    image.setAttribute("height", String(dimensions.height));
    image.dataset.codoxearImageKey = source;
  }

  function prepareImageForDisplay(image, source) {
    const dimensions = cachedImageDimensions(source) || DEFAULT_IMAGE_DIMENSIONS;
    applyImageDimensions(image, source, dimensions);
  }

  function validImageDimensions(value) {
    const width = Number(value && value.width);
    const height = Number(value && value.height);
    return Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0 ? { width, height } : null;
  }

  function requestImageDimensions(url) {
    const existing = imageDimensionRequests.get(url);
    if (existing) return existing;
    if (typeof fetch !== "function") return Promise.reject(new Error("image dimension fetch unavailable"));
    const request = fetch(url, { credentials: "same-origin" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`image dimensions request failed: ${response.status}`);
        const dimensions = validImageDimensions(await response.json());
        if (!dimensions) throw new Error("image dimensions response is invalid");
        return dimensions;
      })
      .finally(() => imageDimensionRequests.delete(url));
    imageDimensionRequests.set(url, request);
    return request;
  }

  function assignImageSource(image, source, dimensions) {
    applyImageDimensions(image, source, dimensions);
    image.removeAttribute("data-codoxear-image-src");
    image.removeAttribute("data-codoxear-image-dimensions-url");
    image.src = source;
  }

  function hydrateMarkedImage(image) {
    const source = image && image.dataset ? image.dataset.codoxearImageSrc : "";
    const dimensionsUrl = image && image.dataset ? image.dataset.codoxearImageDimensionsUrl : "";
    if (!source || !dimensionsUrl) return;
    const cached = cachedImageDimensions(source);
    if (cached) {
      assignImageSource(image, source, cached);
      return;
    }
    requestImageDimensions(dimensionsUrl).then(
      (dimensions) => assignImageSource(image, source, dimensions),
      () => assignImageSource(image, source, DEFAULT_IMAGE_DIMENSIONS)
    );
  }

  function hydrateMarkedImages(root = document) {
    if (!root || typeof root.querySelectorAll !== "function") return;
    for (const image of root.querySelectorAll("img[data-codoxear-image-src][data-codoxear-image-dimensions-url]")) hydrateMarkedImage(image);
  }

  function scheduleMarkedImageHydration() {
    if (imageHydrationScheduled || typeof Promise === "undefined") return;
    imageHydrationScheduled = true;
    Promise.resolve().then(() => {
      imageHydrationScheduled = false;
      hydrateMarkedImages();
    });
  }

  function rememberImageDimensions(image) {
    const source = image && image.dataset ? image.dataset.codoxearImageKey : "";
    const width = Number(image && image.naturalWidth);
    const height = Number(image && image.naturalHeight);
    if (!source || !Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) return;
    const entries = readImageDimensionCache();
    entries[source] = { width, height, usedAt: nextImageUse(entries) };
    writeImageDimensionCache(entries);
    applyImageDimensions(image, source, { width, height });
  }

  if (typeof document !== "undefined" && typeof document.addEventListener === "function") {
    document.addEventListener("load", (event) => {
      const image = event.target;
      if (image && typeof image.matches === "function" && image.matches("img[data-codoxear-image-key]")) rememberImageDimensions(image);
    }, true);
  }

  function rewriteMarkedImages(root, doc, options) {
    for (const image of root.querySelectorAll("img[src]")) {
      const rawSrc = image.getAttribute("src") || "";
      const localRef = localFileRefFromRef(rawSrc, options);
      const src = options && typeof options.resolveImageSrc === "function" ? options.resolveImageSrc(rawSrc, localRef) : safeUrl(rawSrc);
      const dimensionsUrl = options && typeof options.resolveImageDimensionsUrl === "function" ? options.resolveImageDimensionsUrl(rawSrc, localRef) : null;
      if (!src) image.replaceWith(doc.createTextNode(image.alt || ""));
      else if (dimensionsUrl && !cachedImageDimensions(src)) {
        // A server-readable file stays source-less until its header dimensions
        // arrive. This prevents the browser from beginning the image request
        // with a guessed 16:9 layout box.
        image.removeAttribute("src");
        image.dataset.codoxearImageSrc = src;
        image.dataset.codoxearImageDimensionsUrl = dimensionsUrl;
      } else {
        prepareImageForDisplay(image, src);
        image.src = src;
      }
      image.loading = "lazy";
    }
    scheduleMarkedImageHydration();
  }

  // Browser-decodable video containers, mirroring the file viewer's
  // BROWSER_SAFE_VIDEO_TYPES (mp4/webm/ogg). Anything else (mkv, avi, mov,
  // ...) resolves to null so the player collapses to a file link whose
  // viewer offers the transcoded compatible preview instead of a dead
  // inline player.
  const INLINE_VIDEO_EXTENSIONS = new Set(["mp4", "m4v", "webm", "ogv", "ogg"]);

  function previewVideoUrlForRef(rawRef, localRef, { sessionId } = {}) {
    if (localRef && localRef.path) {
      if (!INLINE_VIDEO_EXTENSIONS.has(filePathExtension(localRef.path))) return null;
      if (sessionId) return resolveAppUrl(`/api/sessions/${sessionId}/file/blob?path=${encodeURIComponent(localRef.path)}`);
      if (localRef.path.startsWith("/")) return resolveAppUrl(`/api/files/blob?path=${encodeURIComponent(localRef.path)}`);
      return null;
    }
    return safeUrl(rawRef);
  }

  function resolveVideoSourceUrl(rawSrc, localRef, options) {
    if (options && typeof options.resolveVideoSrc === "function") return options.resolveVideoSrc(rawSrc, localRef);
    return localRef ? null : safeUrl(rawSrc);
  }

  // marked passes raw <video> HTML through verbatim; without this pass the
  // browser resolves a local-path src against the app origin, the fetch 404s,
  // and the player shows the browser's "no supported format/MIME" fallback.
  // Images already route through resolveImageSrc; this gives video the same
  // treatment (blob URL for server-readable files, safeUrl for remote).
  function rewriteMarkedVideos(root, doc, options) {
    for (const video of root.querySelectorAll("video")) {
      let firstRaw = "";
      const trackRaw = (raw) => {
        if (!firstRaw && raw) firstRaw = raw;
        return localFileRefFromRef(raw, options);
      };
      const ownSrc = String(video.getAttribute("src") || "").trim();
      const ownUrl = ownSrc ? resolveVideoSourceUrl(ownSrc, trackRaw(ownSrc), options) || "" : "";
      if (ownUrl) {
        // A video element's own src wins over <source> children.
        for (const source of video.querySelectorAll("source")) source.remove();
        video.setAttribute("src", ownUrl);
      } else {
        video.removeAttribute("src");
        let kept = 0;
        for (const source of video.querySelectorAll("source")) {
          const raw = String(source.getAttribute("src") || "").trim();
          const url = raw ? resolveVideoSourceUrl(raw, trackRaw(raw), options) || "" : "";
          if (url) {
            source.setAttribute("src", url);
            kept += 1;
          } else source.remove();
        }
        if (!kept) {
          // No playable source survives resolution. Drop the dead player and
          // leave the first candidate as text; the file-ref text pass below
          // turns a local path into a clickable viewer link.
          video.replaceWith(doc.createTextNode(firstRaw || String(video.textContent || "")));
          continue;
        }
      }
      if (!video.hasAttribute("controls")) video.setAttribute("controls", "");
      if (!video.hasAttribute("preload")) video.setAttribute("preload", "metadata");
      video.setAttribute("playsinline", "");
      const poster = String(video.getAttribute("poster") || "").trim();
      if (poster) {
        const posterRef = localFileRefFromRef(poster, options);
        const posterUrl = options && typeof options.resolveImageSrc === "function"
          ? options.resolveImageSrc(poster, posterRef)
          : posterRef ? null : safeUrl(poster);
        if (posterUrl) video.setAttribute("poster", posterUrl);
        else video.removeAttribute("poster");
      }
    }
  }

  function decorateCodeBlocks(root, doc) {
    for (const pre of root.querySelectorAll("pre")) {
      const code = pre.querySelector(":scope > code");
      if (!code) continue;
      const languageClass = Array.from(code.classList).find((value) => value.startsWith("language-"));
      if (languageClass) code.dataset.lang = languageClass.slice("language-".length);
      if (!pre.parentElement || !pre.parentElement.classList.contains("codeBlockWrap")) {
        const wrap = doc.createElement("div");
        wrap.className = "codeBlockWrap";
        pre.parentNode.insertBefore(wrap, pre);
        wrap.appendChild(pre);
      }
      const wrap = pre.parentElement;
      if (wrap && !wrap.querySelector(":scope > .code-copy-btn")) {
        const button = doc.createElement("button");
        button.className = "code-copy-btn";
        button.type = "button";
        button.setAttribute("aria-label", "Copy code");
        button.title = "Copy code";
        wrap.appendChild(button);
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
    rewriteMarkedVideos(root, document, options);
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
      .replace(/\\\(([\s\S]+?)\\\)/g, (_m, body) => push(body, false));
  }

  function extractMathFromText(input, store) {
    // First pass: extract \[...\] and \(...\) from the FULL text, including
    // multi-line display math. Code fence detection runs on the result.
    let text = String(input ?? "").replaceAll("\r\n", "\n");
    text = extractMathFromPlainText(text, store);
    // Second pass: code fence detection on the math-stripped text.
    // Only $$...$$ and remaining $...$ need per-line code-fence protection.
    const lines = text.split("\n");
    const out = [];
    let fence = null;
    for (const line of lines) {
      const openOrClose = line.match(/^\s{0,3}(`{3,}|~{3,})/);
      if (fence) {
        out.push(line);
        if (openOrClose && openOrClose[1][0] === fence.delimiter && openOrClose[1].length >= fence.length) fence = null;
        continue;
      }
      if (openOrClose) {
        fence = { delimiter: openOrClose[1][0], length: openOrClose[1].length };
        out.push(line);
        continue;
      }
      const parts = line.split(/(`+[^`]*`+)/g);
      out.push(parts.map((part, index) => (index % 2 ? part : extractDollarMath(part, store))).join(""));
    }
    return out.join("\n");
  }

  function extractDollarMath(text, store) {
    const push = (latex, display) => {
      const id = store.length;
      store.push({ latex: String(latex).trim(), display: !!display });
      return mathToken(id);
    };
    return String(text)
      .replace(/\$\$([\s\S]+?)\$\$/g, (_m, body) => push(body, true))
      // Pandoc tex_math_dollars: delimiters flank non-space, and a closing
      // dollar cannot be followed by a digit so ordinary dollar amounts remain literal.
      .replace(/\$(?!\$|\s)([^$\n]*?\S)\$(?!\$|\d)/g, (_m, body) => push(body, false));
  }

  function substituteMath(html, store) {
    let out = html;
    for (let i = 0; i < store.length; i++) out = out.split(mathToken(i)).join(renderMath(store[i].latex, store[i].display));
    return out;
  }

  // CommonMark delimiter flanking treats CJK brackets 「」『』《》 and fullwidth
  // punctuation （）：，！？。 as Unicode punctuation, and its flanking rules
  // encode a whitespace-delimited-language assumption: an opener followed by
  // punctuation may only open when preceded by whitespace/punctuation, and a
  // closer preceded by punctuation may only close when followed by
  // whitespace/punctuation. CJK prose has no inter-word spaces, so both
  // 首推是**「标题」** (opener rejected) and **结论。**后续 (closer rejected)
  // are ordinary text, not edge cases; marked then leaves the `**` literal or
  // pairs the opener with a far-away delimiter, bolding the prose between two
  // intended spans.
  //
  // The inline extension below runs ahead of marked's emStrong and resolves
  // these spans by matching, the same way CommonMark's delimiter stack does:
  // a run's role comes from pairing, not from guessing at its local context.
  // It fires at an open-capable run, scans forward for the NEAREST closer
  // under CJK-relaxed flanking, and claims the span only when CommonMark
  // would resolve it differently — the opener or that nearest closer needed
  // the relaxed rule. When neither does, the built-in pairs the very same
  // delimiters (relaxed capability is a superset of CommonMark's, so the
  // nearest relaxed closer is also the nearest CommonMark closer whenever
  // that run is CommonMark-valid) and the extension defers, keeping every
  // non-CJK span on the built-in path byte-for-byte.
  const CJK_PUNCT_RANGE = [[0x3001, 0x303f], [0xff00, 0xff65], [0x2018, 0x201d]];
  // Flanking approximation of Unicode punctuation/symbols (marked itself uses
  // \p{P}\p{S}). Where the approximation disagrees with marked, the extension
  // may claim a span the built-in would have rendered — but it pairs the same
  // nearest delimiters, so the output is unchanged.
  const PUNCTISH_RANGES = [[0x21, 0x2f], [0x3a, 0x40], [0x5b, 0x60], [0x7b, 0x7e], [0xa1, 0xbf], [0x2010, 0x2027], [0x2030, 0x205e]];

  function codeInRanges(code, ranges) {
    if (!Number.isFinite(code)) return false;
    for (const [lo, hi] of ranges) if (code >= lo && code <= hi) return true;
    return false;
  }

  function isCjkPunctCode(code) {
    return codeInRanges(code, CJK_PUNCT_RANGE);
  }

  function isPunctishCode(code) {
    return isCjkPunctCode(code) || codeInRanges(code, PUNCTISH_RANGES);
  }

  function isFlankWsCode(code) {
    if (!Number.isFinite(code)) return true; // end of content flanks like whitespace
    return code === 0x20 || (code >= 0x09 && code <= 0x0d) || code === 0xa0 || code === 0x1680 || (code >= 0x2000 && code <= 0x200a) || code === 0x2028 || code === 0x2029 || code === 0x202f || code === 0x205f || code === 0x3000;
  }

  // Mirrors marked's own blockSkip masking: code spans, inline HTML tags and
  // [label](dest) links become filler "a"s so a rescue closer is never chosen
  // from inside a span the built-in would keep atomic. Length-preserving, so
  // scan indices double as indices into the real source.
  const CJK_DELIM_SCAN_MASK_RE = /\[[^[\]]*?\]\((?:\\.|[^\\()]|\((?:\\.|[^\\()])*\))*\)|`[^`]*?`|<[^<>]*?>/g;

  function maskDelimScanSource(src) {
    if (!/[`<[\]]/.test(src)) return src;
    return src.replace(CJK_DELIM_SCAN_MASK_RE, (match) => "a".repeat(match.length));
  }

  function lastInlineSourceCharCode(tokens) {
    if (!Array.isArray(tokens) || !tokens.length) return null;
    const raw = tokens[tokens.length - 1] && tokens[tokens.length - 1].raw;
    return typeof raw === "string" && raw ? raw.charCodeAt(raw.length - 1) : null;
  }

  // CommonMark left-flanking for `*`: not followed by whitespace, and either
  // not followed by punctuation or preceded by whitespace/punctuation.
  function commonmarkRunCanOpen(beforeCode, afterCode) {
    if (isFlankWsCode(afterCode)) return false;
    if (!isPunctishCode(afterCode)) return true;
    return beforeCode === null || isFlankWsCode(beforeCode) || isPunctishCode(beforeCode);
  }

  // CommonMark right-flanking for `*`: not preceded by whitespace, and either
  // not preceded by punctuation or followed by whitespace/punctuation.
  function commonmarkRunCanClose(prevCode, nextCode) {
    if (isFlankWsCode(prevCode)) return false;
    if (!isPunctishCode(prevCode)) return true;
    return isFlankWsCode(nextCode) || isPunctishCode(nextCode);
  }

  // CJK-relaxed flanking: additionally, a run may open when immediately
  // followed by CJK punctuation and may close when immediately preceded by it.
  function rescueRunCanOpen(beforeCode, afterCode) {
    return isCjkPunctCode(afterCode) || commonmarkRunCanOpen(beforeCode, afterCode);
  }

  function rescueRunCanClose(prevCode, nextCode) {
    return isCjkPunctCode(prevCode) || commonmarkRunCanClose(prevCode, nextCode);
  }

  function tokenizeCjkEmphasis(src, tokens) {
    if (src.charCodeAt(0) !== 0x2a) return undefined;
    let runEnd = 1;
    while (src.charCodeAt(runEnd) === 0x2a) runEnd += 1;
    const runLen = runEnd;
    if (runLen > 3) return undefined;
    const beforeCode = lastInlineSourceCharCode(tokens);
    const afterCode = src.charCodeAt(runEnd);
    if (!rescueRunCanOpen(beforeCode, afterCode)) return undefined;
    // Claim the span only when CommonMark would resolve it differently: the
    // opener needed the relaxed rule, or the nearest acceptable closer did.
    // Otherwise defer so the built-in emStrong produces its byte-for-byte
    // result for the identical span.
    const openerNeedsRescue = !commonmarkRunCanOpen(beforeCode, afterCode);
    const masked = maskDelimScanSource(src);
    for (let i = runLen; i < masked.length;) {
      if (masked.charCodeAt(i) !== 0x2a) {
        i += 1;
        continue;
      }
      let end = i;
      while (masked.charCodeAt(end) === 0x2a) end += 1;
      if (i > runLen && end - i >= runLen && rescueRunCanClose(src.charCodeAt(i - 1), src.charCodeAt(end))) {
        if (!openerNeedsRescue && commonmarkRunCanClose(src.charCodeAt(i - 1), src.charCodeAt(end))) return undefined;
        const raw = src.slice(0, i + runLen);
        // ***both*** nests as em > strong in marked; re-lexing raw minus one
        // delimiter on each side reproduces exactly that structure (and lets
        // the inner strong itself go through this rescue when CJK-adjacent).
        const inner = runLen === 3 ? raw.slice(1, -1) : src.slice(runLen, i);
        return { type: "cjkEmphasis", raw, level: runLen, tokens: this.lexer.inlineTokens(inner) };
      }
      i = end;
    }
    return undefined;
  }

  let cjkEmphasisRegistered = false;
  function ensureCjkEmphasisExtension() {
    if (cjkEmphasisRegistered) return;
    if (!window.marked || typeof window.marked.use !== "function") return;
    window.marked.use({
      extensions: [
        {
          name: "cjkEmphasis",
          level: "inline",
          tokenizer: tokenizeCjkEmphasis,
          renderer(token) {
            const inner = this.parser.parseInline(token.tokens);
            // level 3 re-lexed raw minus one delimiter per side, so the inner
            // strong token already exists — mirroring marked's em renderer.
            return token.level === 2 ? `<strong>${inner}</strong>` : `<em>${inner}</em>`;
          },
        },
      ],
    });
    cjkEmphasisRegistered = true;
  }

  // marked's GFM extended autolink scans a URL candidate to the first
  // whitespace or `<` and trims only ASCII trailing punctuation, so a bare URL
  // followed directly by prose punctuation becomes one anchor with the prose:
  // `…/dexgem-strategy（私有，jackw1997 邀请已发）` links every character up to
  // the next space. CJK prose has no inter-word spaces, so the scan never sees a
  // boundary — cmark-gfm/GitHub renders the same HTML, and this is the common
  // case for agent output, not an edge case. An autolink instead ends at the
  // first non-ASCII punctuation or symbol. Non-ASCII letters and digits never
  // terminate, so CJK paths and query values (`…/wiki/中文`, `?wd=萧山`) keep
  // working, percent-encoded forms (`%EF%BC%88`) are untouched, and ASCII URL
  // punctuation stays governed by marked's own rules.
  //
  // This is an inline *extension*, not a `use({tokenizer})` override: marked
  // tries extension tokenizers ahead of the built-ins and reads their
  // `undefined` as "defer", while a `use({tokenizer})` override returning
  // `undefined` suppresses the built-in entirely (`false` is its fall-through
  // signal), which would force reimplementing marked's URL body. The truncated
  // candidate goes back through that body, so GFM's trailing-punctuation
  // trimming stays marked's.
  const PROSE_BOUNDARY_RE = /[\p{P}\p{S}]/u;

  function firstProseBoundaryIndex(source) {
    for (let i = 0; i < source.length;) {
      const code = source.codePointAt(i);
      const size = code > 0xffff ? 2 : 1;
      if (code > 0x7f && PROSE_BOUNDARY_RE.test(source.slice(i, i + size))) return i;
      i += size;
    }
    return -1;
  }

  function tokenizeCjkUrlBoundary(src) {
    const tokenizer = this.lexer && this.lexer.tokenizer;
    const rule = tokenizer && tokenizer.rules && tokenizer.rules.inline && tokenizer.rules.inline.url;
    if (typeof rule?.exec !== "function") return undefined;
    const match = rule.exec(src);
    // `url` also owns marked's email branch (the `@` capture); an email
    // autolink already stops at CJK punctuation, so defer that branch as-is.
    if (!match || match[2] === "@") return undefined;
    const cut = firstProseBoundaryIndex(match[0]);
    if (cut < 0) return undefined;
    return tokenizer.url(match[0].slice(0, cut)) || undefined;
  }

  let cjkUrlBoundaryRegistered = false;
  function ensureCjkUrlBoundary() {
    if (cjkUrlBoundaryRegistered) return;
    if (!window.marked || typeof window.marked.use !== "function") return;
    window.marked.use({
      extensions: [
        {
          name: "cjkUrlBoundary",
          level: "inline",
          tokenizer: tokenizeCjkUrlBoundary,
        },
      ],
    });
    cjkUrlBoundaryRegistered = true;
  }

  // marked v15's del rule accepts a run of one OR two tildes per side
  // (`~~?...~~?` with a backreference), so a sentence like
  // `100~300°/s ... **~50 像素**` pairs two unrelated approximation/range
  // tildes into one strikethrough span — deleting the prose between them and
  // swallowing the `**`. Agent output is dense with `100~300` ranges and
  // `~50` approximations, so this false-positive class is frequent. GFM
  // defines strikethrough as exactly `~~`; a single `~` is never markup.
  // This override runs ahead of the builtin (marked.use tokenizer wrapping)
  // and reproduces the builtin rule with the run pinned to exactly two
  // tildes; anything else returns undefined — never `false` — so the
  // single-tilde builtin never gets a fallback chance and the tildes render
  // as literal text.
  const DOUBLE_TILDE_DEL_RE = /^(~~)(?=[^\s~])((?:\\.|[^\\])*?(?:\\.|[^\s~\\]))\1(?=[^~]|$)/;

  let gfmStrikethroughRegistered = false;
  function ensureGfmStrikethrough() {
    if (gfmStrikethroughRegistered) return;
    if (!window.marked || typeof window.marked.use !== "function") return;
    window.marked.use({
      tokenizer: {
        del(src) {
          const match = DOUBLE_TILDE_DEL_RE.exec(src);
          if (!match) return undefined;
          return { type: "del", raw: match[0], text: match[2], tokens: this.lexer.inlineTokens(match[2]) };
        },
      },
    });
    gfmStrikethroughRegistered = true;
  }

  function mdToHtml(src, options = null) {
    if (!window.marked || typeof window.marked.parse !== "function") throw new Error("marked failed to load");
    ensureCjkEmphasisExtension();
    ensureCjkUrlBoundary();
    ensureGfmStrikethrough();
    const mathStore = [];
    const prepared = extractMathFromText(rewriteOaiMemCitations(String(src ?? "").replaceAll("\r\n", "\n")), mathStore);
    // breaks: true preserves single newlines from the raw text as line
    // breaks — agent messages are plain-text-first, not print-style prose.
    // The CJK emphasis rescue, the CJK autolink boundary, and the GFM-strict
    // strikethrough override are registered lazily and idempotently above.
    return substituteMath(postProcessMarkedHtml(window.marked.parse(prepared, { breaks: true }), options), mathStore);
  }

  const mdCache = new Map();
  function mdToHtmlCached(src, options = null) {
    const text = String(src ?? "");
    const scope = options && typeof options.cacheKey === "string" ? options.cacheKey : "";
    const key = `${scope}\0${text}`;
    const hit = mdCache.get(key);
    if (hit !== undefined) {
      scheduleMarkedImageHydration();
      return hit;
    }
    const html = mdToHtml(text, options);
    mdCache.set(key, html);
    if (mdCache.size > 1200) mdCache.clear();
    return html;
  }

  function previewImageUrlForRef(rawRef, localRef, { filePath, sessionId } = {}) {
    if (localRef && localRef.path) {
      if (sessionId) return resolveAppUrl(`/api/sessions/${sessionId}/file/blob?path=${encodeURIComponent(localRef.path)}`);
      if (localRef.path.startsWith("/")) return resolveAppUrl(`/api/files/blob?path=${encodeURIComponent(localRef.path)}`);
    }
    return safeUrl(rawRef);
  }

  function previewImageDimensionsUrlForRef(rawRef, localRef, { filePath, sessionId } = {}) {
    if (!localRef || !localRef.path) return null;
    if (sessionId) return resolveAppUrl(`/api/sessions/${sessionId}/file/image-dimensions?path=${encodeURIComponent(localRef.path)}`);
    if (localRef.path.startsWith("/")) return resolveAppUrl(`/api/files/image-dimensions?path=${encodeURIComponent(localRef.path)}`);
    return null;
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
      resolveImageDimensionsUrl(rawRef, localRef) {
        return previewImageDimensionsUrlForRef(rawRef, localRef, { filePath: basePath, sessionId: sid });
      },
      resolveVideoSrc(rawRef, localRef) {
        return previewVideoUrlForRef(rawRef, localRef, { sessionId: sid });
      },
    });
  }

  // Display projection for the server-generated attachment prefix block
  // (file_upload.attachment_inject_text): a run of leading lines with the
  // exact shape "Attachment N: /abs/path". The committed text stays
  // backend-truthful plain text (agent prompt, terminal echo, copy, export);
  // only chat rendering rewrites it. Image attachments render inline via
  // markdown image syntax; anything else keeps its plain line, whose path the
  // candidate-ref upgrade turns into a file link. The matching rule mirrors
  // session_listing._redact_generated_attachment_prefix_paths.
  const ATTACHMENT_PREFIX_LINE_RE = /^Attachment \d+: (\/.+)$/;
  const INLINE_IMAGE_EXTENSIONS = new Set(["avif", "bmp", "gif", "ico", "jpeg", "jpg", "png", "svg", "webp"]);

  function attachmentDisplayMarkdown(src) {
    const text = String(src ?? "");
    if (!text.startsWith("Attachment ")) return text;
    const lines = text.split("\n");
    let end = 0;
    while (end < lines.length && ATTACHMENT_PREFIX_LINE_RE.test(lines[end])) end += 1;
    if (end === 0) return text;
    const rewritten = lines.slice(0, end).map((line) => {
      const path = ATTACHMENT_PREFIX_LINE_RE.exec(line)[1];
      if (!INLINE_IMAGE_EXTENSIONS.has(filePathExtension(path))) return line;
      // Markdown image destinations cannot carry whitespace or parentheses;
      // leave such lines as plain file-ref text rather than emit broken syntax.
      if (/[\s()]/.test(path)) return line;
      return `![${path.split("/").pop()}](${path})`;
    });
    return [...rewritten, ...lines.slice(end)].join("\n");
  }

  function chatMarkdownHtmlCached(src, sessionId) {
    const sid = String(sessionId || "").trim();
    return mdToHtmlCached(attachmentDisplayMarkdown(src), {
      cacheKey: sid ? `chat:${sid}` : "chat",
      resolveImageSrc(rawRef, localRef) {
        return previewImageUrlForRef(rawRef, localRef, { sessionId: sid });
      },
      resolveImageDimensionsUrl(rawRef, localRef) {
        return previewImageDimensionsUrlForRef(rawRef, localRef, { sessionId: sid });
      },
      resolveVideoSrc(rawRef, localRef) {
        return previewVideoUrlForRef(rawRef, localRef, { sessionId: sid });
      },
    });
  }

export { escapeHtml, mdToHtml, mdToHtmlCached, normalizeLineNumber, parseLocalFileRef, rewriteInlineCodeFileRefs, markdownPreviewHtml, chatMarkdownHtmlCached, attachmentDisplayMarkdown, prepareImageForDisplay, rememberImageDimensions, hydrateMarkedImages, rewriteMarkedVideos, rewriteTextFileRefs, previewVideoUrlForRef };
