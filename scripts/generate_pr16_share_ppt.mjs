import fs from "node:fs";
import path from "node:path";
import PptxGenJS from "pptxgenjs";

const outDir = path.resolve("docs");
const outPath = path.join(outDir, "pr16-share-deck.pptx");

fs.mkdirSync(outDir, { recursive: true });

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_16x9";
pptx.author = "Codex";
pptx.company = "Codoxear";
pptx.subject = "PR#16 technical sharing";
pptx.title = "PR#16: 一次面向演进的 Web 架构重构";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "zh-CN",
};

const C = {
  ink: "1F2937",
  slate: "475569",
  muted: "64748B",
  line: "CBD5E1",
  paper: "F8FAFC",
  snow: "FFFFFF",
  navy: "0F172A",
  ocean: "0F3D5E",
  teal: "0E7490",
  cyan: "E0F2FE",
  mint: "D1FAE5",
  sand: "FFF7ED",
  amber: "EA580C",
  gold: "F59E0B",
  red: "DC2626",
};

function addTopTag(slide, text, opts = {}) {
  slide.addText(text, {
    x: opts.x ?? 0.55,
    y: opts.y ?? 0.28,
    w: opts.w ?? 2.3,
    h: 0.28,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 9,
    bold: true,
    color: opts.color ?? C.teal,
    charSpacing: 1.2,
  });
}

function addPageTitle(slide, title, subtitle) {
  addTopTag(slide, "PR #16 / CODOXEAR WEB");
  slide.addText(title, {
    x: 0.55,
    y: 0.58,
    w: 8.6,
    h: 0.65,
    margin: 0,
    fontFace: "Aptos Display",
    fontSize: 24,
    bold: true,
    color: C.navy,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.55,
      y: 1.02,
      w: 8.6,
      h: 0.38,
      margin: 0,
      fontFace: "Aptos",
      fontSize: 10.5,
      color: C.slate,
    });
  }
}

function addFooter(slide, label = "PR#16 分享") {
  slide.addText(label, {
    x: 0.55,
    y: 5.16,
    w: 2.4,
    h: 0.2,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 8.5,
    color: C.muted,
  });
  slide.addText(String(slide._slideNum || ""), {
    x: 9.1,
    y: 5.13,
    w: 0.35,
    h: 0.22,
    margin: 0,
    align: "right",
    fontFace: "Aptos",
    fontSize: 8.5,
    color: C.muted,
  });
}

function addBulletList(slide, items, opts = {}) {
  const runs = [];
  items.forEach((item, index) => {
    runs.push({
      text: item,
      options: {
        bullet: { indent: 14 },
        breakLine: index !== items.length - 1,
      },
    });
  });
  slide.addText(runs, {
    x: opts.x,
    y: opts.y,
    w: opts.w,
    h: opts.h,
    margin: 0,
    fontFace: "Aptos",
    fontSize: opts.fontSize ?? 14,
    color: opts.color ?? C.ink,
    breakLine: false,
    paraSpaceAfterPt: 8,
    valign: "top",
  });
}

function addCard(slide, x, y, w, h, title, body, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    line: { color: opts.lineColor ?? C.line, width: 1 },
    fill: { color: opts.fillColor ?? C.snow },
    shadow: { type: "outer", color: "000000", blur: 1, offset: 1, angle: 45, opacity: 0.08 },
  });
  if (opts.accentColor) {
    slide.addShape(pptx.ShapeType.rect, {
      x: x + 0.15,
      y: y + 0.18,
      w: 0.12,
      h: h - 0.36,
      line: { color: opts.accentColor, width: 0 },
      fill: { color: opts.accentColor },
    });
  }
  slide.addText(title, {
    x: x + 0.35,
    y: y + 0.2,
    w: w - 0.5,
    h: 0.34,
    margin: 0,
    fontFace: "Aptos Display",
    fontSize: 15,
    bold: true,
    color: opts.titleColor ?? C.navy,
  });
  slide.addText(body, {
    x: x + 0.35,
    y: y + 0.55,
    w: w - 0.55,
    h: h - 0.72,
    margin: 0,
    fontFace: "Aptos",
    fontSize: opts.bodySize ?? 11,
    color: opts.bodyColor ?? C.slate,
    breakLine: false,
    valign: "top",
  });
}

function addPill(slide, x, y, text, fill, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w: 1.5,
    h: 0.34,
    rectRadius: 0.12,
    line: { color: fill, width: 0 },
    fill: { color: fill },
  });
  slide.addText(text, {
    x,
    y: y + 0.06,
    w: 1.5,
    h: 0.2,
    margin: 0,
    align: "center",
    fontFace: "Aptos",
    fontSize: 9.5,
    bold: true,
    color,
  });
}

// Slide 1: cover
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 10,
    h: 5.625,
    line: { color: C.navy, width: 0 },
    fill: { color: C.navy },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.75,
    y: 0,
    w: 3.25,
    h: 5.625,
    line: { color: C.ocean, width: 0 },
    fill: { color: C.ocean, transparency: 4 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.08,
    y: 0.75,
    w: 2.95,
    h: 4.1,
    line: { color: C.teal, width: 0 },
    fill: { color: C.teal, transparency: 18 },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.6,
    y: 4.75,
    w: 8.55,
    h: 0,
    line: { color: "94A3B8", width: 1.2, transparency: 45 },
  });
  slide.addText("PR#16", {
    x: 0.65,
    y: 0.72,
    w: 1.5,
    h: 0.34,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 12,
    bold: true,
    color: "7DD3FC",
    charSpacing: 1.5,
  });
  slide.addText("一次面向演进的\nWeb 架构重构", {
    x: 0.65,
    y: 1.22,
    w: 5.2,
    h: 1.55,
    margin: 0,
    fontFace: "Aptos Display",
    fontSize: 26,
    bold: true,
    color: C.snow,
    breakLine: false,
    fit: "shrink",
  });
  slide.addText("从 static 单体前端到 Vite + Preact + TypeScript\n并借此把移动端与 Pi 交互体验真正做顺", {
    x: 0.68,
    y: 3.0,
    w: 4.7,
    h: 0.75,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 12.5,
    color: "CBD5E1",
  });
  addPill(slide, 0.68, 4.18, "架构重构", "123A56", "E0F2FE");
  addPill(slide, 2.35, 4.18, "体验优化", "134E4A", "D1FAE5");
  addPill(slide, 4.02, 4.18, "工程演进", "7C2D12", "FFEDD5");
  slide.addText("Codoxear / 内部技术分享", {
    x: 0.68,
    y: 4.9,
    w: 2.8,
    h: 0.26,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 9.5,
    color: "94A3B8",
  });
}

// Slide 2: why refactor
{
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  addPageTitle(slide, "为什么这次必须重构，而不是继续打补丁", "旧前端已经从“功能载体”变成了“迭代阻力”。");

  addCard(
    slide,
    0.55,
    1.45,
    4.2,
    2.85,
    "旧形态：codoxear/static 单体前端",
    "app.js + app.css + index.html 混合承担 DOM 渲染、状态修改、轮询、API 调用、移动端处理与 backend 细节。\n\n结果不是不能用，而是每加一个能力，都要在同一团逻辑里穿线。",
    { accentColor: C.red, fillColor: C.snow }
  );
  addBulletList(slide, [
    "状态边界隐式：谁拥有会话状态、消息状态、UI 状态并不清楚",
    "功能耦合严重：移动端、轮询、消息、工作区逻辑互相牵连",
    "评审与测试困难：一个文件太大，很难在工作记忆里完整 hold 住",
    "新需求上来后，改动成本不是线性增长，而是放大增长",
  ], { x: 5.05, y: 1.58, w: 4.15, h: 2.45, fontSize: 12.5 });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 5.05,
    y: 4.1,
    w: 4.15,
    h: 0.8,
    rectRadius: 0.05,
    line: { color: "FCD34D", width: 0.8 },
    fill: { color: C.sand },
  });
  slide.addText("真正触发重构的不是“想换技术栈”，而是 Pi 交互、workspace、移动端体验已经在逼近旧结构上限。", {
    x: 5.25,
    y: 4.28,
    w: 3.78,
    h: 0.4,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 10.6,
    bold: true,
    color: C.amber,
  });
  addFooter(slide);
}

// Slide 3: how refactor
{
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  addPageTitle(slide, "怎么重构：先重画边界，再替换实现", "核心动作不是 JSX 重写，而是把职责拆开。 ");

  addCard(slide, 0.62, 1.55, 2.1, 1.05, "AppShell", "页面骨架、对话区、侧栏、overlay 编排", { accentColor: C.teal, fillColor: C.cyan });
  addCard(slide, 2.95, 1.55, 2.1, 1.05, "Domain Stores", "sessions / messages / session-ui / composer", { accentColor: C.teal, fillColor: C.cyan });
  addCard(slide, 5.28, 1.55, 1.95, 1.05, "API Client", "统一 fetch、类型和契约适配", { accentColor: C.teal, fillColor: C.cyan });
  addCard(slide, 7.45, 1.55, 1.95, 1.05, "Python Server", "继续托管 /api 与生产静态资源", { accentColor: C.teal, fillColor: C.cyan });

  slide.addShape(pptx.ShapeType.line, { x: 2.72, y: 2.07, w: 0.2, h: 0, line: { color: C.teal, width: 1.2, beginArrowType: "none", endArrowType: "triangle" } });
  slide.addShape(pptx.ShapeType.line, { x: 5.03, y: 2.07, w: 0.2, h: 0, line: { color: C.teal, width: 1.2, beginArrowType: "none", endArrowType: "triangle" } });
  slide.addShape(pptx.ShapeType.line, { x: 7.22, y: 2.07, w: 0.2, h: 0, line: { color: C.teal, width: 1.2, beginArrowType: "none", endArrowType: "triangle" } });

  addCard(slide, 0.62, 3.0, 2.76, 1.6, "工程化升级", "前端迁移到 web/\nVite 构建、TS 类型、可测试的组件与 store 结构", { accentColor: C.ocean, fillColor: C.snow, bodySize: 10.8 });
  addCard(slide, 3.6, 3.0, 2.76, 1.6, "状态边界清晰", "把会话、消息、工作区、composer 分域管理\n避免 UI 与副作用交叉污染", { accentColor: C.ocean, fillColor: C.snow, bodySize: 10.8 });
  addCard(slide, 6.58, 3.0, 2.76, 1.6, "为后续演进留接口", "轮询优化、Pi live state、未来 SSE / WebSocket\n都不再需要重写语义层", { accentColor: C.ocean, fillColor: C.snow, bodySize: 10.8 });
  addFooter(slide);
}

// Slide 4: experience
{
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  addPageTitle(slide, "重构不是为换栈，而是为了把体验做顺", "新架构真正释放出来的是可落地的用户体验改进。 ");

  addCard(slide, 0.55, 1.55, 2.85, 1.45, "移动端会话流", "从桌面式杂糅控制，优化到 conversation-first：\n打开会话、发消息、读回复更顺手。", { accentColor: C.gold, fillColor: C.snow });
  addCard(slide, 3.58, 1.55, 2.85, 1.45, "Pi AskUser / Todo", "不再只是底层 tool plumbing，Web 里可以直接承接 Pi 的 ask_user 与会话状态。", { accentColor: C.gold, fillColor: C.snow });
  addCard(slide, 6.61, 1.55, 2.85, 1.45, "Workspace 能力", "补齐分层文件浏览、workspace 视图与会话侧能力，形成更完整的辅助工作流。", { accentColor: C.gold, fillColor: C.snow });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.55,
    y: 3.32,
    w: 8.9,
    h: 1.15,
    rectRadius: 0.08,
    line: { color: C.line, width: 0.8 },
    fill: { color: "EEF2FF" },
  });
  slide.addText("一个很关键的例子：cwd group", {
    x: 0.82,
    y: 3.54,
    w: 3.1,
    h: 0.24,
    margin: 0,
    fontFace: "Aptos Display",
    fontSize: 16,
    bold: true,
    color: C.navy,
  });
  addBulletList(slide, [
    "后端负责持久化和恢复，前端负责分组渲染、重命名、折叠与异常兼容",
    "这类体验需求以前很难干净落地；新边界下则可以按职责自然拆分",
  ], { x: 0.82, y: 3.9, w: 8.15, h: 0.48, fontSize: 10.8, color: C.slate });
  addFooter(slide);
}

// Slide 5: benefits
{
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  addPageTitle(slide, "这次优化最值钱的，不是页面更现代，而是后续更便宜", "用户收益和工程收益同时发生。 ");

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.58,
    y: 1.55,
    w: 2.25,
    h: 2.2,
    rectRadius: 0.08,
    line: { color: C.line, width: 0.8 },
    fill: { color: C.snow },
  });
  slide.addText("209", { x: 0.92, y: 1.9, w: 1.2, h: 0.46, margin: 0, fontFace: "Aptos Display", fontSize: 28, bold: true, color: C.teal });
  slide.addText("files changed", { x: 0.92, y: 2.36, w: 1.4, h: 0.22, margin: 0, fontFace: "Aptos", fontSize: 11, color: C.muted });
  slide.addText("不是一个“修几个点”的 PR，而是一次架构层级的收束。", { x: 0.92, y: 2.8, w: 1.45, h: 0.5, margin: 0, fontFace: "Aptos", fontSize: 10.5, color: C.slate });

  addCard(slide, 3.08, 1.55, 3.0, 1.05, "对用户", "移动端路径更短、Pi 交互更直观、workspace 更完整", { accentColor: C.teal, fillColor: C.cyan });
  addCard(slide, 3.08, 2.8, 3.0, 1.05, "对研发", "状态边界更清楚，代码 review、测试和定位回归更容易", { accentColor: C.teal, fillColor: C.cyan });
  addCard(slide, 6.32, 1.55, 3.0, 1.05, "对后续演进", "继续补 Pi live state、轮询优化、更多交互能力时，不必再拆旧墙", { accentColor: C.amber, fillColor: C.sand });
  addCard(slide, 6.32, 2.8, 3.0, 1.05, "对团队协作", "功能开发从“在一个大文件里穿线”变成“在明确边界里扩展”", { accentColor: C.amber, fillColor: C.sand });

  slide.addText("一句话：这次优化把“迭代阻力”转成了“演进能力”。", {
    x: 0.6,
    y: 4.48,
    w: 8.75,
    h: 0.32,
    margin: 0,
    fontFace: "Aptos Display",
    fontSize: 17,
    bold: true,
    color: C.navy,
  });
  addFooter(slide);
}

// Slide 6: recap
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  slide.addText("复盘", {
    x: 0.68,
    y: 0.72,
    w: 1.2,
    h: 0.34,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 12,
    bold: true,
    color: "7DD3FC",
    charSpacing: 1.5,
  });
  slide.addText("这次 PR 真正完成的，\n不是“前端重写”", {
    x: 0.68,
    y: 1.22,
    w: 4.35,
    h: 1.0,
    margin: 0,
    fontFace: "Aptos Display",
    fontSize: 25,
    bold: true,
    color: C.snow,
  });
  slide.addText("而是重新定义了 Codoxear Web 的演进方式", {
    x: 0.7,
    y: 2.38,
    w: 4.95,
    h: 0.32,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 13,
    color: "CBD5E1",
  });

  addCard(slide, 5.72, 0.92, 3.45, 0.92, "01 先定义边界", "AppShell、store、API、server 各自负责什么先讲清楚。", { accentColor: "38BDF8", fillColor: "13253B", lineColor: "1E3A5F", titleColor: C.snow, bodyColor: "CBD5E1", bodySize: 10.5 });
  addCard(slide, 5.72, 2.03, 3.45, 0.92, "02 再做体验", "移动端、AskUser、workspace 这些体验，才能顺着边界自然落地。", { accentColor: "34D399", fillColor: "13253B", lineColor: "1E3A5F", titleColor: C.snow, bodyColor: "CBD5E1", bodySize: 10.5 });
  addCard(slide, 5.72, 3.14, 3.45, 0.92, "03 把一次优化变成长期收益", "后续继续演进时，不再为旧结构反复支付额外成本。", { accentColor: "F59E0B", fillColor: "13253B", lineColor: "1E3A5F", titleColor: C.snow, bodyColor: "CBD5E1", bodySize: 10.5 });

  slide.addText("谢谢", {
    x: 0.7,
    y: 4.62,
    w: 1.0,
    h: 0.28,
    margin: 0,
    fontFace: "Aptos Display",
    fontSize: 18,
    bold: true,
    color: C.snow,
  });
  slide.addText("PR#16 / refactor(web): Vite + Preact 重构前端架构，并接入 Pi AskUser / 工作区能力", {
    x: 0.7,
    y: 4.97,
    w: 8.25,
    h: 0.24,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 9,
    color: "94A3B8",
  });
}

await pptx.writeFile({ fileName: outPath });
console.log(outPath);
