

  function requireFunction(value, name) {
    if (typeof value !== "function") throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requireStatusNode(value) {
    if (!value || typeof value.replaceChildren !== "function") throw new TypeError("file viewer dependency missing: fileStatus");
    return value;
  }

  function requireEditButtonNode(value) {
    if (!value || !value.classList || typeof value.classList.toggle !== "function" || typeof value.setAttribute !== "function") {
      throw new TypeError("file viewer dependency missing: fileEditButton");
    }
    return value;
  }

  const BROWSER_SAFE_VIDEO_TYPES = new Set(["video/mp4", "video/webm", "video/ogg"]);
  const FILE_EDITOR_UNAVAILABLE_MESSAGE = "Editing is unavailable because the code editor failed to load. Read-only preview remains available.";

  function bindFileTouchPress(button, handler, options = {}) {
    if (!button || typeof button.addEventListener !== "function" || typeof handler !== "function") return false;
    const nowMs = typeof options.nowMs === "function" ? options.nowMs : () => Date.now();
    let suppressClickUntil = 0;
    let sawPointerTouchAt = 0;
    const run = (event) => {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      suppressClickUntil = nowMs() + 700;
      handler();
    };
    button.addEventListener("pointerdown", (event) => {
      if (event && event.pointerType === "touch") sawPointerTouchAt = nowMs();
      run(event);
    });
    button.addEventListener(
      "touchstart",
      (event) => {
        if (nowMs() - sawPointerTouchAt < 700) {
          event.preventDefault();
          event.stopPropagation();
          return;
        }
        run(event);
      },
      { passive: false }
    );
    button.addEventListener("click", (event) => {
      if (nowMs() < suppressClickUntil) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      run(event);
    });
    return true;
  }

  function bindFileTouchClick(button, handler) {
    if (!button || typeof button.addEventListener !== "function" || typeof handler !== "function") return false;
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      handler();
    });
    return true;
  }

  function requireStyledNode(value, name) {
    if (!value || !value.style) throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requireRenderHostNode(value, name) {
    if (!value || !("innerHTML" in value) || typeof value.appendChild !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requirePasteInput(value) {
    if (!value || !("value" in value)) throw new TypeError("file viewer dependency missing: filePasteInput");
    return value;
  }

  function requireVideoNode(value) {
    if (!value || !value.style || typeof value.removeAttribute !== "function" || typeof value.load !== "function") {
      throw new TypeError("file viewer dependency missing: fileVideo");
    }
    return value;
  }

  function requireImageNode(value) {
    if (!value || !value.style || typeof value.removeAttribute !== "function") {
      throw new TypeError("file viewer dependency missing: fileImage");
    }
    return value;
  }

  function requireToggleClassNode(value, name) {
    if (!value || !value.classList || typeof value.classList.toggle !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireModeControlButton(value, name) {
    if (
      !value ||
      !value.classList ||
      typeof value.classList.toggle !== "function" ||
      !value.style ||
      typeof value.setAttribute !== "function" ||
      !("disabled" in value)
    ) {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireModalHostNode(value, name) {
    if (!value || typeof value.setAttribute !== "function" || typeof value.removeAttribute !== "function") {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }

  function requireTextNode(value, name) {
    if (!value || !("textContent" in value)) throw new TypeError(`file viewer dependency missing: ${name}`);
    return value;
  }

  function requireUnsavedButtonNode(value, name) {
    if (!value || !("hidden" in value) || !("disabled" in value) || !("textContent" in value)) {
      throw new TypeError(`file viewer dependency missing: ${name}`);
    }
    return value;
  }


  const DOWNLOAD_BUFFER_LIMIT_BYTES = 32 * 1024 * 1024;
  const DOWNLOAD_OBJECT_URL_TTL_MS = 60 * 1000;
  // The page owns the filename, so the blob type stays inert on purpose: a
  // *known* type that disagrees with the name's extension is the trigger that
  // makes Safari add an extension of its own.
  const DOWNLOAD_INERT_TYPE = "application/octet-stream";

  // A download name is a filesystem identifier, not a path: keep the basename,
  // strip separators and control characters, and replace lone UTF-16 surrogates
  // (a raw-byte rel path can carry them) so the attribute never receives a
  // half-pair. Empty input stays empty, which leaves the browser its own
  // derivation instead of inventing a name.
  function downloadFileName(path) {
    const raw = String(path ?? "");
    const base = raw.split(/[/\\]/).pop() || "";
    const wellFormed = typeof base.toWellFormed === "function"
      ? base.toWellFormed()
      : base.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, "\uFFFD");
    const cleaned = wellFormed.replace(/[\u0000-\u001f\u007f]/g, "").trim();
    if (!cleaned || cleaned === "." || cleaned === "..") return "";
    return cleaned;
  }

  function createFileDownloadRuntime(options = {}) {
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const documentRef = options.document || null;
    if (!documentRef || typeof documentRef.createElement !== "function" || !documentRef.body || typeof documentRef.body.appendChild !== "function") {
      throw new TypeError("file viewer dependency missing: document");
    }
    const resolveDownloadFileName = typeof options.resolveDownloadFileName === "function" ? options.resolveDownloadFileName : null;
    const fetchFn = typeof options.fetchFn === "function" ? options.fetchFn : null;
    const createObjectUrl = typeof options.createObjectUrl === "function" ? options.createObjectUrl : null;
    const revokeObjectUrl = typeof options.revokeObjectUrl === "function" ? options.revokeObjectUrl : null;
    const BlobCtor = typeof options.BlobCtor === "function" ? options.BlobCtor : null;
    const shouldBuffer = typeof options.shouldBufferDownload === "function" ? options.shouldBufferDownload : null;
    const setTimeoutFn = typeof options.setTimeoutFn === "function" ? options.setTimeoutFn : (callback, delay) => setTimeout(callback, delay);
    const onError = typeof options.onError === "function" ? options.onError : () => {};
    const bufferLimitBytes = Number.isFinite(options.bufferLimitBytes) && options.bufferLimitBytes > 0
      ? options.bufferLimitBytes
      : DOWNLOAD_BUFFER_LIMIT_BYTES;
    let inFlight = false;

    function clickDownload(href, filename) {
      const link = documentRef.createElement("a");
      link.href = href;
      link.download = filename || "";
      link.rel = "noopener";
      link.style.display = "none";
      documentRef.body.appendChild(link);
      link.click();
      link.remove();
    }

    // Streaming transfer: the browser owns the bytes and the response headers.
    // Used wherever the platform's own download flow works, and as the fallback
    // for a file too large to hold in page memory.
    function download(apiPath, filename) {
      const path = String(apiPath || "");
      if (!path) return false;
      clickDownload(resolveAppUrl(path), downloadFileName(filename));
      return true;
    }

    function bufferingAvailable() {
      return Boolean(fetchFn && createObjectUrl && revokeObjectUrl && BlobCtor);
    }

    // `Content-Length` is a hint, so the cap is enforced while reading: a server
    // that under-reports must not be able to exhaust the page.
    async function readBoundedBody(response) {
      const body = response && response.body;
      if (!body || typeof body.getReader !== "function") {
        const blob = await response.blob();
        return blob.size > bufferLimitBytes ? null : [blob];
      }
      const reader = body.getReader();
      const parts = [];
      let total = 0;
      try {
        for (;;) {
          const step = await reader.read();
          if (!step || step.done) break;
          if (!step.value || !step.value.byteLength) continue;
          total += step.value.byteLength;
          if (total > bufferLimitBytes) {
            try { await reader.cancel(); } catch (_error) {}
            return null;
          }
          parts.push(step.value);
        }
      } finally {
        try { reader.releaseLock(); } catch (_error) {}
      }
      return parts;
    }

    // Buffered transfer: the bytes become a page-owned blob, so nothing in the
    // response (disposition, type, URL) participates in naming it. `no-store`
    // also keeps the body out of the service worker's caches.
    async function bufferedDownload(apiPath, filename) {
      const response = await fetchFn(resolveAppUrl(apiPath), { credentials: "same-origin", cache: "no-store" });
      if (response.status === 401) {
        throw Object.assign(new Error("authentication required"), { codoxearDownloadReason: "auth" });
      }
      if (!response.ok) throw new Error(`download failed (${response.status})`);
      const parts = await readBoundedBody(response);
      if (!parts) {
        return download(apiPath, filename) ? { ok: true, mode: "stream" } : { ok: false, reason: "no-file" };
      }
      const blob = BlobCtor(parts, { type: DOWNLOAD_INERT_TYPE });
      const objectUrl = createObjectUrl(blob);
      clickDownload(objectUrl, downloadFileName(filename));
      // iOS may not finish the save until after the click returns, so the object
      // URL outlives this turn.
      setTimeoutFn(() => {
        try { revokeObjectUrl(objectUrl); } catch (_error) {}
      }, DOWNLOAD_OBJECT_URL_TTL_MS);
      return { ok: true, mode: "buffer", size: blob.size };
    }

    async function downloadActive(apiPath) {
      const path = String(apiPath || "");
      if (!path) return { ok: false, reason: "no-file" };
      // The caller owns the API path (it already builds it); the viewer owns the
      // name; this runtime owns only the transfer strategy. The name is resolved
      // per click so a session switch cannot hand over a stale one.
      const filename = resolveDownloadFileName ? resolveDownloadFileName() : "";
      if (!(shouldBuffer && shouldBuffer()) || !bufferingAvailable()) {
        return download(path, filename) ? { ok: true, mode: "stream" } : { ok: false, reason: "no-file" };
      }
      if (inFlight) return { ok: false, reason: "busy" };
      inFlight = true;
      try {
        return await bufferedDownload(path, filename);
      } catch (error) {
        const reason = error && error.codoxearDownloadReason ? error.codoxearDownloadReason : "transport";
        onError(reason === "auth"
          ? "download failed: sign in again"
          : `download failed: ${error && error.message ? error.message : "network error"}`);
        return { ok: false, reason };
      } finally {
        inFlight = false;
      }
    }

    return Object.freeze({ download, downloadActive });
  }

export { createFileDownloadRuntime, downloadFileName };
