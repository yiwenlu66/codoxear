(function () {
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


  function createFilePdfRenderRuntime(options = {}) {
    const host = requireRenderHostNode(options.host, "filePdfHost");
    const el = requireFunction(options.el, "el");
    const ensurePdfJs = requireFunction(options.ensurePdfJs, "ensurePdfJs");
    const createCanvas = requireFunction(options.createCanvas, "createCanvas");
    const devicePixelRatio = requireFunction(options.devicePixelRatio, "devicePixelRatio");
    const disposeFileEditor = requireFunction(options.disposeFileEditor, "disposeFileEditor");
    const disposePdfRender = requireFunction(options.disposePdfRender, "disposePdfRender");
    const clearFileVideo = requireFunction(options.clearFileVideo, "clearFileVideo");
    const setFileRenderSurface = requireFunction(options.setFileRenderSurface, "setFileRenderSurface");
    const renderDownloadFallback = requireFunction(options.renderDownloadFallback, "renderDownloadFallback");
    const isCurrentFileOpenRequest = requireFunction(options.isCurrentFileOpenRequest, "isCurrentFileOpenRequest");
    const setActivePdfRenderState = requireFunction(options.setActivePdfRenderState, "setActivePdfRenderState");
    const isActivePdfRenderState = requireFunction(options.isActivePdfRenderState, "isActivePdfRenderState");
    const updateFileTouchToolbar = requireFunction(options.updateFileTouchToolbar, "updateFileTouchToolbar");
    const IntersectionObserverCtor = options.IntersectionObserverCtor;

    function preparePdfSurface(rel) {
      disposeFileEditor();
      disposePdfRender();
      clearFileVideo();
      host.innerHTML = "";
      setFileRenderSurface("diff");
      host.scrollTop = 0;
      const container = el("div", { class: "filePdfPages", role: "document", "aria-label": `${rel} PDF preview` });
      host.appendChild(container);
      return container;
    }

    async function renderPageIntoSlot(state, pdf, firstPage, scale, slot, request) {
      const pageNumber = Number(slot.dataset.pageNumber || "0");
      if (!pageNumber || state.rendered.has(pageNumber) || state.rendering.has(pageNumber)) return;
      state.rendering.add(pageNumber);
      slot.classList.add("rendering");
      try {
        const page = pageNumber === 1 ? firstPage : await pdf.getPage(pageNumber);
        if (!isActivePdfRenderState(state) || !isCurrentFileOpenRequest(request)) return;
        const viewport = page.getViewport({ scale });
        const outputScale = devicePixelRatio() || 1;
        const canvas = createCanvas();
        const context = canvas.getContext("2d", { alpha: false });
        if (!context) throw new Error("PDF canvas unavailable");
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;
        const transform = outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null;
        const task = page.render({
          canvasContext: context,
          viewport,
          transform,
          background: "rgb(255, 255, 255)",
        });
        state.renderTasks.add(task);
        await task.promise;
        state.renderTasks.delete(task);
        if (!isActivePdfRenderState(state) || !isCurrentFileOpenRequest(request)) return;
        slot.replaceChildren(canvas);
        slot.classList.remove("rendering");
        state.rendered.add(pageNumber);
      } catch (e) {
        if (e && e.name === "RenderingCancelledException") return;
        if (!isActivePdfRenderState(state) || !isCurrentFileOpenRequest(request)) return;
        slot.textContent = `Page ${pageNumber} failed to render`;
        slot.classList.add("failed");
      } finally {
        state.rendering.delete(pageNumber);
      }
    }

    async function render(rel, url, request) {
      const container = preparePdfSurface(rel);
      if (typeof IntersectionObserverCtor !== "function") {
        if (!isCurrentFileOpenRequest(request)) return false;
        renderDownloadFallback(rel, url, "PDF lazy renderer unavailable");
        return true;
      }
      let pdfjs;
      try {
        pdfjs = await ensurePdfJs();
      } catch (e) {
        if (!isCurrentFileOpenRequest(request)) return false;
        renderDownloadFallback(rel, url, e && e.message ? e.message : "PDF renderer unavailable");
        return true;
      }
      if (!isCurrentFileOpenRequest(request)) return false;
      const loadingTask = pdfjs.getDocument({ url, withCredentials: true });
      const state = {
        request,
        loadingTask,
        observer: null,
        renderTasks: new Set(),
        rendered: new Set(),
        rendering: new Set(),
        pdf: null,
      };
      setActivePdfRenderState(state);
      const pdf = await loadingTask.promise;
      if (!isActivePdfRenderState(state) || !isCurrentFileOpenRequest(request)) return false;
      state.pdf = pdf;
      const firstPage = await pdf.getPage(1);
      if (!isActivePdfRenderState(state) || !isCurrentFileOpenRequest(request)) return false;
      const unitViewport = firstPage.getViewport({ scale: 1 });
      const maxWidth = Math.max(240, container.clientWidth - 24);
      const scale = maxWidth / Math.max(1, unitViewport.width);
      const pageWidth = Math.floor(unitViewport.width * scale);
      const pageHeight = Math.floor(unitViewport.height * scale);
      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const slot = el("div", {
          class: "filePdfPage",
          "data-page-number": String(pageNumber),
          style: `width: ${pageWidth}px; min-height: ${pageHeight}px;`,
        }, [
          el("span", { class: "filePdfPageLabel", text: `Page ${pageNumber}` }),
        ]);
        container.appendChild(slot);
      }
      state.observer = new IntersectionObserverCtor(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting) void renderPageIntoSlot(state, pdf, firstPage, scale, entry.target, request);
          }
        },
        { root: host, rootMargin: "900px 0px" }
      );
      container.querySelectorAll(".filePdfPage").forEach((slot) => state.observer.observe(slot));
      updateFileTouchToolbar();
      return true;
    }

    return Object.freeze({ render });
  }


  window.CodoxearFilePdf = Object.freeze({ createFilePdfRenderRuntime });
})();
