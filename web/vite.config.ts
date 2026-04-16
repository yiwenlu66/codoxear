import preact from "@preact/preset-vite";
import { resolve } from "node:path";
import { defineConfig, loadEnv } from "vite";
import { fileURLToPath } from "node:url";

const webRoot = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, repoRoot, "");
  const rawPort = String(env.CODEX_WEB_PORT || "8743").trim();
  const serverPort = Number(rawPort);
  if (!Number.isInteger(serverPort) || serverPort <= 0) {
    throw new Error(`Invalid CODEX_WEB_PORT: ${rawPort || "<empty>"}`);
  }

  const rawPrefix = String(env.CODEX_WEB_URL_PREFIX || "").trim();
  if (rawPrefix && rawPrefix !== "/" && !rawPrefix.startsWith("/")) {
    throw new Error(`Invalid CODEX_WEB_URL_PREFIX: ${rawPrefix}`);
  }

  const urlPrefix = rawPrefix && rawPrefix !== "/" ? rawPrefix.replace(/\/+$/, "") : "";
  const proxy = {
    "/api": {
      target: `http://127.0.0.1:${serverPort}`,
      changeOrigin: true,
    },
  } as Record<string, { target: string; changeOrigin: boolean }>;

  if (urlPrefix) {
    proxy[`${urlPrefix}/api`] = {
      target: `http://127.0.0.1:${serverPort}`,
      changeOrigin: true,
    };
  }

  return {
    base: urlPrefix ? `${urlPrefix}/` : "/",
    plugins: [preact()],
    resolve: {
      alias: {
        "@": resolve(webRoot, "src"),
        "@ui": resolve(webRoot, "src/components/ui"),
      },
    },
    server: {
      port: 5173,
      proxy,
    },
    build: {
      outDir: "dist",
      emptyOutDir: true,
      manifest: true,
      sourcemap: true,
    },
    test: {
      environment: "jsdom",
      maxWorkers: 1,
      minWorkers: 1,
      setupFiles: ["./src/test/setup.ts"],
    },
  };
});
