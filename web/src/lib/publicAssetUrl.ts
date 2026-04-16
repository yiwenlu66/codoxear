const trimLeadingSlashes = (value: string) => value.replace(/^\/+/, "");
const trimTrailingSlashes = (value: string) => value.replace(/\/+$/, "");

function getPublicAssetResolutionBase(): string {
  if (typeof document !== "undefined" && document.baseURI) {
    return document.baseURI;
  }

  return typeof window === "undefined" ? "http://localhost/" : window.location.href;
}

function toPublicAssetBasePath(baseUrl: string): string {
  const normalizedBaseUrl = String(baseUrl || "/").trim() || "/";
  const pathname = new URL(normalizedBaseUrl, getPublicAssetResolutionBase()).pathname;
  return trimTrailingSlashes(pathname);
}

export function toPublicAssetUrl(path: string, baseUrl = import.meta.env.BASE_URL): string {
  const assetPath = trimLeadingSlashes(path);
  const normalizedBase = toPublicAssetBasePath(baseUrl);
  return normalizedBase ? `${normalizedBase}/${assetPath}` : `/${assetPath}`;
}
