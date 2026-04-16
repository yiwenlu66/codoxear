if (typeof globalThis.requestAnimationFrame !== "function") {
  globalThis.requestAnimationFrame = (callback: FrameRequestCallback): number =>
    window.setTimeout(() => callback(Date.now()), 16);
}

if (typeof globalThis.cancelAnimationFrame !== "function") {
  globalThis.cancelAnimationFrame = (handle: number): void => {
    window.clearTimeout(handle);
  };
}

if (typeof document.queryCommandSupported !== "function") {
  document.queryCommandSupported = () => false;
}

export {};
