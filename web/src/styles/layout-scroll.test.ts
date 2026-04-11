import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "src/styles/global.css"), "utf-8");

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function blockBody(source: string, openBraceIndex: number) {
  let depth = 1;

  for (let index = openBraceIndex + 1; index < source.length; index += 1) {
    const char = source[index];

    if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
    }

    if (depth === 0) {
      return source.slice(openBraceIndex + 1, index);
    }
  }

  throw new Error(`Unclosed CSS block starting at ${openBraceIndex}`);
}

function ruleBody(source: string, selector: string) {
  const match = new RegExp(`${escapeRegExp(selector)}\\s*\\{`, "m").exec(source);
  expect(match, `Expected CSS rule for ${selector}`).not.toBeNull();

  return blockBody(source, match!.index + match![0].length - 1);
}

function expectRuleToContain(source: string, selector: string, declaration: string) {
  const rulePattern = new RegExp(`${escapeRegExp(selector)}\\s*\\{`, "gm");
  const matches = [...source.matchAll(rulePattern)];
  expect(matches.length, `Expected CSS rule for ${selector}`).toBeGreaterThan(0);

  const hasDeclaration = matches.some((match) => blockBody(source, match.index! + match[0].length - 1).includes(declaration));
  expect(hasDeclaration, `Expected ${selector} to include ${declaration}`).toBe(true);
}

function mediaBody(source: string, query: string) {
  const match = new RegExp(`@media\\s*${escapeRegExp(query)}\\s*\\{`, "m").exec(source);
  expect(match, `Expected @media ${query} block`).not.toBeNull();

  return blockBody(source, match!.index + match![0].length - 1);
}

describe("conversation layout scroll guards", () => {
  it("defines the editorial shell as a fixed two-column layout with a quiet conversation stack", () => {
    const shellRule = ruleBody(css, ".appShell.editorialShell");
    const sidebarRule = ruleBody(css, ".sidebarColumn");
    const conversationRule = ruleBody(css, ".conversationColumn");
    const paneRule = ruleBody(css, ".conversationPane");

    expect(shellRule).toContain("position: fixed;");
    expect(shellRule).toContain("inset: 0;");
    expect(shellRule).toContain("grid-template-columns: minmax(16rem, var(--sidebar-w)) minmax(0, 1fr);");
    expect(sidebarRule).toContain("min-height: 0;");
    expect(sidebarRule).toContain("overflow: hidden;");
    expect(conversationRule).toContain("display: flex;");
    expect(conversationRule).toContain("flex-direction: column;");
    expect(conversationRule).toContain("min-height: 0;");
    expect(conversationRule).toContain("overflow: hidden;");
    expect(paneRule).toContain("flex: 1 1 0;");
    expect(paneRule).toContain("min-height: 0;");
  });

  it("positions the conversation navigation buttons as a bottom-right stacked overlay", () => {
    const timelineRule = ruleBody(css, ".conversationTimeline");
    const navRule = ruleBody(css, ".conversationNavButtons");
    const jumpButtonRule = ruleBody(css, ".conversationJumpButton");
    const mobileRules = mediaBody(css, "(max-width: 880px)");
    const mobileNavRule = ruleBody(mobileRules, ".conversationNavButtons");

    expect(timelineRule).toContain("position: relative;");
    expect(navRule).toContain("position: absolute;");
    expect(navRule).toContain("display: flex;");
    expect(navRule).toContain("flex-direction: column;");
    expect(navRule).toContain("right: 18px;");
    expect(navRule).toContain("bottom: 18px;");
    expect(navRule).toContain("pointer-events: none;");
    expect(navRule).toContain("gap: 10px;");
    expect(jumpButtonRule).toContain("width: 42px;");
    expect(jumpButtonRule).toContain("height: 42px;");
    expect(jumpButtonRule).toContain("pointer-events: auto;");
    expect(jumpButtonRule).toContain("backdrop-filter: blur(14px);");
    expect(mobileNavRule).toContain("right: 12px;");
    expect(mobileNavRule).toContain("bottom: 12px;");
    expect(mobileNavRule).toContain("gap: 8px;");
  });

  it("keeps composer todo selectors in the global shell stylesheet", () => {
    expect(css).toMatch(/\.composerStack\s*\{/);
    expect(css).toMatch(/\.composerTodoBar\s*\{/);
    expect(css).toMatch(/\.composerTodoBarButton\.isExpanded\s*\{/);
    expect(css).toMatch(/\.composerTodoPanel\s*\{/);
    expect(css).toMatch(/\.composerTodoStatus\.completed\s*\{/);
  });

  it("bounds the expanded composer todo panel without hiding the composer", () => {
    const stackRule = ruleBody(css, ".composerStack");
    const panelRule = ruleBody(css, ".composerTodoPanel");

    expect(stackRule).toContain("display: flex;");
    expect(stackRule).toContain("flex-direction: column;");
    expect(panelRule).toContain("max-height: min(32dvh, 260px);");
    expect(panelRule).toContain("overflow: auto;");
    expect(panelRule).toContain("overscroll-behavior: contain;");
  });

  it("lets long todo titles shrink and wrap inside the composer panel item header", () => {
    const itemHeadRule = ruleBody(css, ".composerTodoItemHead");
    const itemTitleRule = ruleBody(css, ".composerTodoItemHead strong");

    expect(itemHeadRule).toContain("min-width: 0;");
    expect(itemTitleRule).toContain("flex: 1 1 auto;");
    expect(itemTitleRule).toContain("min-width: 0;");
    expect(itemTitleRule).toContain("overflow-wrap: anywhere;");
  });

  it("contains long todo summary, description, and status text inside the composer panel", () => {
    const statusRule = ruleBody(css, ".composerTodoStatus");

    expectRuleToContain(css, ".composerTodoSummary", "overflow-wrap: anywhere;");
    expectRuleToContain(css, ".composerTodoItem p", "overflow-wrap: anywhere;");
    expect(statusRule).toContain("flex: 0 1 auto;");
    expect(statusRule).toContain("min-width: 0;");
    expect(statusRule).toContain("overflow-wrap: anywhere;");
  });

  it("adds bounded workspace dialog hooks and stable toolbar/composer sizing", () => {
    const dialogRule = ruleBody(css, ".workspaceDialog");
    const dialogBodyRule = ruleBody(css, ".workspaceDialogBody");
    const dialogHeaderRule = ruleBody(css, ".workspaceDialogHeader");
    const toolbarTextButtonRule = ruleBody(css, ".toolbarTextButton");
    const composerInputRule = ruleBody(css, ".composerInputWrap");
    const queueButtonRule = ruleBody(css, ".composerQueueButton");
    const sendButtonRule = ruleBody(css, ".sendButton");

    expect(dialogRule).toContain("width: min(72rem, calc(100vw - 32px));");
    expect(dialogRule).toContain("max-height: min(86dvh, 56rem);");
    expect(dialogRule).toContain("overflow: hidden;");
    expect(dialogBodyRule).toContain("min-height: 0;");
    expect(dialogBodyRule).toContain("overflow: hidden;");
    expect(dialogHeaderRule).toContain("flex: 0 0 auto;");
    expect(toolbarTextButtonRule).toContain("min-width: fit-content;");
    expect(composerInputRule).toContain("min-width: 0;");
    expect(queueButtonRule).toContain("min-width: fit-content;");
    expect(sendButtonRule).toContain("width: 44px;");
    expect(sendButtonRule).toContain("height: 44px;");
  });

  it("keeps session title clamped and session actions hover-gated so long text does not stretch the rail", () => {
    const titleRule = ruleBody(css, ".sessionTitle");
    const actionRowRule = ruleBody(css, ".sessionActionRowInline");

    expect(titleRule).toContain("display: -webkit-box;");
    expect(titleRule).toContain("overflow: hidden;");
    expect(titleRule).toContain("-webkit-line-clamp: 2;");
    expect(titleRule).toContain("overflow-wrap: anywhere;");
    expect(actionRowRule).toContain("pointer-events: none;");
    expect(css).toContain(".sessionCard:hover .sessionActionRowInline,");
    expect(css).toContain(".sessionCard:focus-within .sessionActionRowInline,");
    expect(css).toContain(".sessionCard.active .sessionActionRowInline {");
    expect(css).toContain("pointer-events: auto;");
  });

  it("keeps session footer meta and action areas vertically centered with each other", () => {
    const footerRowRule = ruleBody(css, ".sessionCardFooterRow");
    const footerAsideRule = ruleBody(css, ".sessionCardFooterAside");

    expect(footerRowRule).toContain("align-items: center;");
    expect(footerAsideRule).toContain("align-items: center;");
  });

  it("keeps the mobile composer and shell sizing rules scoped to the 880px media block", () => {
    const mobileRules = mediaBody(css, "(max-width: 880px)");
    const mobileShellRule = ruleBody(mobileRules, ".appShell.editorialShell");
    const mobileStackRule = ruleBody(mobileRules, ".composerStack");
    const mobilePanelRule = ruleBody(mobileRules, ".composerTodoPanel");
    const mobileShellComposerRule = ruleBody(mobileRules, ".composerShell");
    const mobileTextareaRule = ruleBody(mobileRules, ".composerTextarea");

    expect(mobileShellRule).toContain("grid-template-columns: 1fr;");
    expect(mobileStackRule).toContain("padding: 8px 10px calc(8px + env(safe-area-inset-bottom));");
    expect(mobilePanelRule).toContain("max-height: min(28dvh, 220px);");
    expect(mobileShellComposerRule).toContain("padding: 8px 10px calc(8px + env(safe-area-inset-bottom));");
    expect(mobileTextareaRule).toContain("min-height: 56px;");
    expect(mobileTextareaRule).toContain("max-height: 176px;");
  });

  it("keeps the new session footer visible on mobile with a sticky safe-area action bar", () => {
    const mobileRules = mediaBody(css, "(max-width: 880px)");
    const mobileDialogRule = ruleBody(mobileRules, ".newSessionDialog");
    const mobileFormBodyRule = ruleBody(mobileRules, ".newSessionFormBody");
    const mobileFooterRule = ruleBody(mobileRules, ".newSessionFooter");

    expect(mobileDialogRule).toContain("max-height: min(100dvh, 100svh);");
    expect(mobileFormBodyRule).toContain("padding-bottom:");
    expect(mobileFooterRule).toContain("position: sticky;");
    expect(mobileFooterRule).toContain("bottom: 0;");
    expect(mobileFooterRule).toContain("padding-bottom: calc(0.85rem + env(safe-area-inset-bottom));");
  });

  it("keeps mobile tool-result content bounded to the viewport and wraps long tokens", () => {
    const messageRowRule = ruleBody(css, ".messageRow");
    const messageSurfaceRule = ruleBody(css, ".messageSurface");
    const toolBlockRule = ruleBody(css, ".messageToolBlock");
    const toolDetailsRule = ruleBody(css, ".messageToolDetails");
    const bodyRule = ruleBody(css, ".messageBody");

    expect(messageRowRule).toContain("min-width: 0;");
    expect(messageSurfaceRule).toContain("min-width: 0;");
    expect(messageSurfaceRule).toContain("max-width: 100%;");
    expect(toolBlockRule).toContain("min-width: 0;");
    expect(toolDetailsRule).toContain("min-width: 0;");
    expect(toolDetailsRule).toContain("max-width: 100%;");
    expect(bodyRule).toContain("min-width: 0;");
    expect(bodyRule).toContain("max-width: 100%;");
    expect(bodyRule).toContain("overflow-wrap: anywhere;");
    expect(bodyRule).toContain("word-break: break-word;");
  });

  it("keeps the mobile message header compact with a two-line clamped title", () => {
    const mobileRules = mediaBody(css, "(max-width: 880px)");
    const headerShellRule = ruleBody(mobileRules, ".messageCardHeader");
    const headerRule = ruleBody(mobileRules, ".messageCardHeaderRow");
    const badgeRule = ruleBody(mobileRules, ".messageCardHeader .badge");
    const titleRule = ruleBody(mobileRules, ".messageCardTitle");
    const timestampRule = ruleBody(mobileRules, ".messageTimestamp");

    expect(headerShellRule).toContain("gap: 2px;");
    expect(headerShellRule).toContain("margin-bottom: 4px;");
    expect(headerRule).toContain("flex-wrap: nowrap;");
    expect(headerRule).toContain("min-width: 0;");
    expect(headerRule).toContain("align-items: flex-start;");
    expect(badgeRule).toContain("flex: 0 0 auto;");
    expect(badgeRule).toContain("font-size: 0.58rem;");
    expect(badgeRule).toContain("padding:");
    expect(titleRule).toContain("min-width: 0;");
    expect(titleRule).toContain("flex: 1 1 auto;");
    expect(titleRule).toContain("display: -webkit-box;");
    expect(titleRule).toContain("-webkit-line-clamp: 2;");
    expect(titleRule).toContain("-webkit-box-orient: vertical;");
    expect(titleRule).toContain("line-height: 1.1;");
    expect(titleRule).toContain("overflow: hidden;");
    expect(timestampRule).toContain("flex: 0 0 auto;");
  });
});
