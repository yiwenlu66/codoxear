interface ErrorPayload {
  error?: unknown;
}

export class HttpError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

const unauthorizedListeners = new Set<() => void>();

export function subscribeUnauthorized(listener: () => void) {
  unauthorizedListeners.add(listener);
  return () => {
    unauthorizedListeners.delete(listener);
  };
}

function notifyUnauthorized() {
  for (const listener of unauthorizedListeners) {
    listener();
  }
}

function parseJsonBody(text: string): unknown {
  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    const invalidJsonError = new Error("Invalid JSON response");
    (invalidJsonError as Error & { cause?: unknown }).cause = error;
    throw invalidJsonError;
  }
}

function resolveApiPath(path: string): string {
  if (!path.startsWith("/")) {
    return path;
  }
  return path.slice(1);
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(resolveApiPath(path), {
    signal,
    headers: {
      Accept: "application/json",
    },
  });
  const text = await response.text();
  const data = parseJsonBody(text);

  if (!response.ok) {
    const errorMessage = (data as ErrorPayload).error;
    const message = typeof errorMessage === "string" ? errorMessage : `Request failed: ${response.status}`;
    if (response.status === 401) {
      notifyUnauthorized();
    }
    throw new HttpError(message, response.status);
  }

  return data as T;
}

export async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(resolveApiPath(path), {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  const data = parseJsonBody(text);

  if (!response.ok) {
    const errorMessage = (data as ErrorPayload).error;
    const message = typeof errorMessage === "string" ? errorMessage : `Request failed: ${response.status}`;
    if (response.status === 401) {
      notifyUnauthorized();
    }
    throw new HttpError(message, response.status);
  }

  return data as T;
}
