  "use strict";


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


  function createFileRenderSurfaceRuntime(options = {}) {
    const diff = requireStyledNode(options.diff, "fileDiff");
    const image = requireImageNode(options.image);
    const video = requireVideoNode(options.video);
    const videoPreviewButton = requireStyledNode(options.videoPreviewButton, "fileVideoPreviewButton");
    const clearActiveVideoFallback = requireFunction(options.clearActiveVideoFallback, "clearActiveVideoFallback");

    function setSurface(surface) {
      const next = String(surface || "");
      if (next !== "diff" && next !== "image" && next !== "video") throw new Error("invalid file render surface");
      diff.style.display = next === "diff" ? "block" : "none";
      image.style.display = next === "image" ? "block" : "none";
      video.style.display = next === "video" ? "block" : "none";
      return next;
    }

    function clearImage() {
      image.removeAttribute("src");
      return true;
    }

    function clearVideo() {
      clearActiveVideoFallback();
      videoPreviewButton.style.display = "none";
      videoPreviewButton.disabled = true;
      video.onerror = null;
      video.onloadedmetadata = null;
      if (typeof video.pause === "function") video.pause();
      video.removeAttribute("src");
      video.load();
      video.style.display = "none";
      return true;
    }

    function reset() {
      clearImage();
      clearVideo();
      setSurface("diff");
      return true;
    }

    function showImage(src, alt = "") {
      clearVideo();
      image.src = String(src || "");
      image.alt = String(alt || "");
      setSurface("image");
      return true;
    }

    function clearVideoHandlers() {
      video.onerror = null;
      video.onloadedmetadata = null;
      return true;
    }

    function showVideo(loadPlan = {}, callbacks = {}) {
      const resolveAppUrl = requireFunction(callbacks.resolveAppUrl, "resolveAppUrl");
      const setStatus = requireFunction(callbacks.setStatus, "setStatus");
      const loadPreview = requireFunction(callbacks.loadPreview, "loadPreview");
      const handleError = requireFunction(callbacks.handleError, "handleError");
      const handleLoadedMetadata = requireFunction(callbacks.handleLoadedMetadata, "handleLoadedMetadata");
      const token = String(loadPlan.token || "");
      video.onerror = () => {
        handleError(loadPlan, { clearVideoHandlers, loadPreview });
      };
      video.onloadedmetadata = () => {
        handleLoadedMetadata(loadPlan);
      };
      setSurface("video");
      if (loadPlan.shouldPreviewFirst) {
        void loadPreview(token, { explicit: false });
      } else {
        video.src = resolveAppUrl(loadPlan.videoUrl);
        setStatus(loadPlan.initialStatus);
      }
      return true;
    }

    return Object.freeze({ clearImage, clearVideo, clearVideoHandlers, reset, setSurface, showImage, showVideo });
  }

export { createFileRenderSurfaceRuntime };
