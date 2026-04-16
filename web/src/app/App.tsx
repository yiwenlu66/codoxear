import { useEffect, useMemo, useState } from "preact/hooks";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api } from "../lib/api";
import { HttpError, subscribeUnauthorized } from "../lib/http";
import { AppProviders } from "./providers";
import { AppShell } from "./AppShell";

type AuthState = "checking" | "authenticated" | "unauthenticated";

function LoginScreen({
  error,
  loading,
  onSubmit,
}: {
  error: string;
  loading: boolean;
  onSubmit: (password: string) => Promise<void>;
}) {
  const [password, setPassword] = useState("");

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6 py-12">
        <div className="rounded-3xl border border-border bg-card p-8 shadow-xl">
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Codoxear</p>
          <h1 className="text-3xl font-semibold tracking-tight">Sign in</h1>
          <p className="mt-2 text-sm text-muted-foreground">Enter the Codoxear password to continue to your sessions.</p>
          <form
            className="mt-6 flex flex-col gap-4"
            onSubmit={(event) => {
              event.preventDefault();
              void onSubmit(password);
            }}
          >
            <label className="flex flex-col gap-2 text-sm font-medium">
              <span>Password</span>
              <Input
                autoComplete="current-password"
                autoFocus
                disabled={loading}
                name="password"
                placeholder="Password"
                type="password"
                value={password}
                onInput={(event) => setPassword(event.currentTarget.value)}
              />
            </label>
            {error ? <p className="text-sm font-medium text-red-600">{error}</p> : null}
            <Button disabled={loading || !password.trim()} type="submit">
              {loading ? "Signing in..." : "Sign in"}
            </Button>
          </form>
        </div>
      </div>
    </main>
  );
}

export default function App() {
  const [authState, setAuthState] = useState<AuthState>("checking");
  const [loginError, setLoginError] = useState("");
  const [loginPending, setLoginPending] = useState(false);

  useEffect(() => {
    const unsubscribe = subscribeUnauthorized(() => {
      setLoginPending(false);
      setLoginError("");
      setAuthState("unauthenticated");
    });

    let cancelled = false;
    api.me()
      .then(() => {
        if (cancelled) return;
        setLoginError("");
        setAuthState("authenticated");
      })
      .catch((error) => {
        if (cancelled) return;
        if (error instanceof HttpError && error.status === 401) {
          setAuthState("unauthenticated");
          return;
        }
        setLoginError(error instanceof Error ? error.message : "Unable to reach the server.");
        setAuthState("unauthenticated");
      });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const content = useMemo(() => {
    if (authState === "authenticated") {
      return (
        <AppProviders>
          <AppShell />
        </AppProviders>
      );
    }

    if (authState === "checking") {
      return (
        <main className="flex min-h-screen items-center justify-center bg-background text-foreground">
          <p className="text-sm text-muted-foreground">Checking session…</p>
        </main>
      );
    }

    return (
      <LoginScreen
        error={loginError}
        loading={loginPending}
        onSubmit={async (password) => {
          const trimmed = password.trim();
          if (!trimmed) {
            setLoginError("Password required");
            return;
          }

          setLoginPending(true);
          setLoginError("");
          try {
            await api.login(trimmed);
            setAuthState("authenticated");
          } catch (error) {
            setLoginError(error instanceof Error ? error.message : "Sign in failed");
          } finally {
            setLoginPending(false);
          }
        }}
      />
    );
  }, [authState, loginError, loginPending]);

  return content;
}
