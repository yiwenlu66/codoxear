

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


  function createFileMediaPreviewRuntime(options = {}) {
    const controller = options.controller;
    if (!controller || typeof controller.loadCompatibleMediaPreview !== "function" || typeof controller.handleFileMediaPreviewButtonPress !== "function" || typeof controller.currentActiveMediaPreviewToken !== "function") {
      throw new TypeError("file viewer dependency missing: controller");
    }
    const fetchPreview = requireFunction(options.fetchPreview, "fetchPreview");
    const resolveAppUrl = requireFunction(options.resolveAppUrl, "resolveAppUrl");
    const handleAuthLoss = requireFunction(options.handleAuthLoss, "handleAuthLoss");
    const errorText = requireFunction(options.errorText, "fileMediaPreviewErrorText");
    const video = options.video;
    if (!video || typeof video.load !== "function") throw new TypeError("file viewer dependency missing: video");
    const audio = options.audio;
    if (!audio || typeof audio.load !== "function") throw new TypeError("file viewer dependency missing: audio");

    async function prepareCompatibleMediaPreview(previewUrl) {
      const res = await fetchPreview(resolveAppUrl(previewUrl), { headers: { Range: "bytes=0-0" } });
      if (res.status === 401) {
        handleAuthLoss();
        throw new Error("authentication required");
      }
      if (!res.ok) {
        let detail = "";
        try {
          const obj = await res.clone().json();
          if (obj && typeof obj.error === "string") detail = obj.error;
        } catch (_) {
          try {
            detail = await res.text();
          } catch (_) {}
        }
        throw new Error(detail || `media preview failed (${res.status})`);
      }
      return true;
    }

    function loadCompatibleMediaPreviewDom(kind, previewUrl) {
      const media = kind === "audio" ? audio : video;
      media.src = resolveAppUrl(previewUrl);
      media.load();
      return true;
    }

    async function loadCompatibleMediaPreview(expectedToken = "", { explicit = false } = {}) {
      return await controller.loadCompatibleMediaPreview(expectedToken, {
        explicit,
        preparePreview: (previewUrl) => prepareCompatibleMediaPreview(previewUrl),
        loadPreviewDom: (kind, previewUrl) => loadCompatibleMediaPreviewDom(kind, previewUrl),
        errorText,
      });
    }

    async function handleButtonPress() {
      const token = controller.currentActiveMediaPreviewToken();
      return await controller.handleFileMediaPreviewButtonPress(token, (nextToken, options) => loadCompatibleMediaPreview(nextToken, options));
    }

    return Object.freeze({ handleButtonPress, loadCompatibleMediaPreview, loadCompatibleMediaPreviewDom, prepareCompatibleMediaPreview });
  }

export { createFileMediaPreviewRuntime };
