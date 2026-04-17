import { beforeEach } from "vitest";

class MemoryStorage implements Storage {
  private data = new Map<string, string>();

  get length(): number {
    return this.data.size;
  }

  clear(): void {
    this.data.clear();
  }

  getItem(key: string): string | null {
    return this.data.has(key) ? this.data.get(key) || null : null;
  }

  key(index: number): string | null {
    return Array.from(this.data.keys())[index] || null;
  }

  removeItem(key: string): void {
    this.data.delete(String(key));
  }

  setItem(key: string, value: string): void {
    this.data.set(String(key), String(value));
  }
}

const localStorageShim = new MemoryStorage();
const sessionStorageShim = new MemoryStorage();

function installStorage(name: "localStorage" | "sessionStorage", storage: Storage) {
  Object.defineProperty(window, name, {
    configurable: true,
    value: storage,
    writable: true,
  });
  Object.defineProperty(globalThis, name, {
    configurable: true,
    value: storage,
    writable: true,
  });
}

function ensureStorage(name: "localStorage" | "sessionStorage", storage: Storage) {
  const current = (window as typeof window & Record<string, unknown>)[name] as Partial<Storage> | undefined;
  if (
    !current
    || typeof current.getItem !== "function"
    || typeof current.setItem !== "function"
    || typeof current.removeItem !== "function"
    || typeof current.clear !== "function"
  ) {
    installStorage(name, storage);
  }
}

function ensureBrowserShims() {
  ensureStorage("localStorage", localStorageShim);
  ensureStorage("sessionStorage", sessionStorageShim);

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
}

ensureBrowserShims();
beforeEach(() => {
  ensureBrowserShims();
});

export {};
