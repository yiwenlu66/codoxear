import { afterEach, describe, expect, it } from "vitest";

import { clearRememberedFileSelections, preferredFileSelectionForSession, rememberFileSelection } from "./fileSelectionState";

describe("fileSelectionState", () => {
  afterEach(() => {
    clearRememberedFileSelections();
  });

  it("remembers the last file selection per session", () => {
    rememberFileSelection("sess-a", "src/first.tsx", 7);
    rememberFileSelection("sess-a", "src/second.tsx", 18);

    expect(preferredFileSelectionForSession("sess-a")).toEqual({
      path: "src/second.tsx",
      line: 18,
    });
  });

  it("keeps selections isolated by session and normalizes invalid lines", () => {
    rememberFileSelection("sess-a", "src/first.tsx", 0);
    rememberFileSelection("sess-b", "src/second.tsx", 3.8);

    expect(preferredFileSelectionForSession("sess-a")).toEqual({
      path: "src/first.tsx",
      line: null,
    });
    expect(preferredFileSelectionForSession("sess-b")).toEqual({
      path: "src/second.tsx",
      line: 3,
    });
  });
});
