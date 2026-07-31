(function () {
  "use strict";

  const perfWindow = 200;
  const perfSamples = new Map();

  function pushSample(name, valueMs) {
    if (!(valueMs >= 0)) return;
    const arr = perfSamples.get(name) || [];
    arr.push(valueMs);
    if (arr.length > perfWindow) arr.splice(0, arr.length - perfWindow);
    perfSamples.set(name, arr);
  }

  function percentile(sorted, p) {
    if (!sorted.length) return 0;
    if (sorted.length === 1) return sorted[0];
    const pos = Math.max(0, Math.min(1, p)) * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, sorted.length - 1);
    const frac = pos - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
  }

  function summarize() {
    const out = {};
    for (const [k, arr] of perfSamples.entries()) {
      if (!arr.length) continue;
      const s = arr.slice().sort((a, b) => a - b);
      out[k] = {
        count: s.length,
        p50_ms: Math.round(percentile(s, 0.5) * 100) / 100,
        p95_ms: Math.round(percentile(s, 0.95) * 100) / 100,
        max_ms: Math.round(s[s.length - 1] * 100) / 100,
        last_ms: Math.round(arr[arr.length - 1] * 100) / 100,
      };
    }
    return out;
  }

  window.CodoxearPerf = Object.freeze({
    pushSample,
    summarize,
  });
})();
