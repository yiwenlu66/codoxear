import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

const apiMock = vi.hoisted(() => ({
  me: vi.fn(),
  login: vi.fn(),
}));

const unauthorizedSubscription = vi.hoisted(() => ({
  callback: null as null | (() => void),
  unsubscribe: vi.fn(),
}));

vi.mock("../lib/api", () => ({
  api: apiMock,
}));

vi.mock("../lib/http", async () => {
  const actual = await vi.importActual<typeof import("../lib/http")>("../lib/http");
  return {
    ...actual,
    subscribeUnauthorized: vi.fn((callback: () => void) => {
      unauthorizedSubscription.callback = callback;
      return unauthorizedSubscription.unsubscribe;
    }),
  };
});

vi.mock("./AppShell", () => ({
  AppShell: () => <div data-testid="app-shell-stub">Shell ready</div>,
}));

import App from "./App";
import { HttpError } from "../lib/http";

let root: HTMLDivElement | null = null;

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise((resolve) => window.setTimeout(resolve, 0));
}

async function settleAuthRequest() {
  const result = apiMock.me.mock.results[0];
  if (result && result.value && typeof result.value.then === "function") {
    try {
      await result.value;
    } catch {
      // Ignore expected failures in auth-gate tests.
    }
  }
  await flush();
}

async function renderApp() {
  act(() => {
    render(<App />, root!);
  });
  await act(async () => {
    await settleAuthRequest();
  });
}

describe("App", () => {
  afterEach(() => {
    if (root) {
      render(null, root);
      root.remove();
      root = null;
    }
    apiMock.me.mockReset();
    apiMock.login.mockReset();
    unauthorizedSubscription.callback = null;
    unauthorizedSubscription.unsubscribe.mockReset();
    document.body.innerHTML = "";
  });

  it("shows the password screen when the auth check returns 401", async () => {
    apiMock.me.mockRejectedValue(new HttpError("Unauthorized", 401));
    root = document.createElement("div");
    document.body.appendChild(root);

    await renderApp();

    expect(root.textContent).toContain("Sign in");
    expect(root.querySelector('input[type="password"]')).not.toBeNull();
    expect(root.querySelector('[data-testid="app-shell-stub"]')).toBeNull();
  });

  it("renders the app shell after a successful auth check", async () => {
    apiMock.me.mockResolvedValue({ ok: true });
    root = document.createElement("div");
    document.body.appendChild(root);

    await renderApp();

    expect(root.querySelector('[data-testid="app-shell-stub"]')).not.toBeNull();
    expect(root.textContent).toContain("Shell ready");
  });

  it("returns to the password screen when a global unauthorized callback fires", async () => {
    apiMock.me.mockResolvedValue({ ok: true });
    root = document.createElement("div");
    document.body.appendChild(root);

    await renderApp();

    expect(root.querySelector('[data-testid="app-shell-stub"]')).not.toBeNull();
    expect(unauthorizedSubscription.callback).not.toBeNull();

    await act(async () => {
      unauthorizedSubscription.callback?.();
      await flush();
    });

    expect(root.textContent).toContain("Sign in");
    expect(root.querySelector('[data-testid="app-shell-stub"]')).toBeNull();
  });

  it("logs in from the password screen and then shows the app shell", async () => {
    apiMock.me.mockRejectedValue(new HttpError("Unauthorized", 401));
    apiMock.login.mockResolvedValue({ ok: true });
    root = document.createElement("div");
    document.body.appendChild(root);

    await renderApp();

    const input = root.querySelector<HTMLInputElement>('input[type="password"]');
    const submit = Array.from(root.querySelectorAll("button")).find((button) => button.textContent?.includes("Sign in"));
    expect(input).not.toBeNull();
    expect(submit).not.toBeNull();

    await act(async () => {
      input!.value = "123456";
      input!.dispatchEvent(new Event("input", { bubbles: true }));
    });

    await act(async () => {
      submit!.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      await flush();
    });

    expect(apiMock.login).toHaveBeenCalledWith("123456");
    expect(root.querySelector('[data-testid="app-shell-stub"]')).not.toBeNull();
  });
});
