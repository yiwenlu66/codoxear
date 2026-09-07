import * as CodoxearModal from "./app_modal.js";

function requireFunction(value, name) {
  if (typeof value !== "function") throw new TypeError(`help dependency missing: ${name}`);
  return value;
}

function requireNode(value, name) {
  if (!value || typeof value !== "object" || !value.style) throw new TypeError(`help dependency missing: ${name}`);
  return value;
}

// Help owns its complete modal lifecycle. The shared modal policy remains
// injected because opening Help must coordinate with every other overlay.
function createHelpController(options = {}) {
  if (!options || typeof options !== "object") throw new TypeError("help dependency missing: options");
  const backdrop = requireNode(options.backdrop, "backdrop");
  const viewer = requireNode(options.viewer, "viewer");
  const closeButton = requireNode(options.closeButton, "closeButton");
  const openButton = requireNode(options.openButton, "openButton");
  const documentTarget = options.documentTarget;
  const ElementCtor = options.ElementCtor;
  if (!documentTarget || typeof documentTarget !== "object") throw new TypeError("help dependency missing: documentTarget");
  if (typeof ElementCtor !== "function") throw new TypeError("help dependency missing: ElementCtor");
  const prepareModalOpen = requireFunction(options.prepareModalOpen, "prepareModalOpen");
  const afterModalVisibilityChanged = requireFunction(options.afterModalVisibilityChanged, "afterModalVisibilityChanged");
  const addEvent = requireFunction(options.addEvent, "addEvent");
  const focusModalSurface = options.focusModalSurface || CodoxearModal.focusModalSurface;
  const isModalTargetOpen = options.isModalTargetOpen || CodoxearModal.isModalTargetOpen;
  const restoreModalFocus = options.restoreModalFocus || CodoxearModal.restoreModalFocus;

  let returnFocusElement = null;

  function show({ opener = null } = {}) {
    returnFocusElement = opener instanceof ElementCtor
      ? opener
      : documentTarget.activeElement instanceof ElementCtor
        ? documentTarget.activeElement
        : null;
    prepareModalOpen();
    backdrop.style.display = "block";
    viewer.style.display = "flex";
    afterModalVisibilityChanged();
    focusModalSurface(viewer);
  }

  function hide() {
    const wasOpen = isModalTargetOpen(viewer);
    const focusTarget = returnFocusElement;
    returnFocusElement = null;
    backdrop.style.display = "none";
    viewer.style.display = "none";
    afterModalVisibilityChanged();
    if (wasOpen) restoreModalFocus(focusTarget, () => isModalTargetOpen(viewer));
  }

  addEvent(openButton, "click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    show({ opener: event.currentTarget });
  });
  addEvent(closeButton, "click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    hide();
  });
  addEvent(backdrop, "click", hide);

  return Object.freeze({ hide, isOpen: () => isModalTargetOpen(viewer), show });
}

export { createHelpController };
