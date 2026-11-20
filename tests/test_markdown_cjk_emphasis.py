"""Behavioral coverage for CJK-aware emphasis rescue in the markdown renderer.

CommonMark delimiter flanking treats CJK corner brackets and fullwidth
punctuation as Unicode punctuation, so marked pairs the wrong ``**`` delimiters
in prose like ``首推是**「欲与天公试比高」**：``. The renderer registers an
inline rescue tokenizer for delimiter runs adjacent to CJK punctuation; these
tests execute the real vendored marked + app_markdown.js in a node VM and pin
the rendered HTML contract (rescued spans and untouched built-in paths alike).
"""

from __future__ import annotations
from test_markdown_edge_cases import render_markdown

import json
import subprocess
import textwrap
from pathlib import Path

from frontend_module_loader import module_path


ROOT = Path(__file__).resolve().parents[1]
APP_MARKDOWN_JS = module_path("app_markdown.js")
MARKED_JS = ROOT / "codoxear" / "static" / "vendor" / "marked.min.js"


def render_line(markdown: str) -> str:
    """Single trailing newline emitted by marked is not part of the contract."""
    return render_markdown(markdown).rstrip("\n")


def render_markdown_twice(markdown: str) -> tuple[str, str]:
    """Render the same input twice in one process (registration idempotency)."""
    app_source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
    marked_source = MARKED_JS.read_text(encoding="utf-8")
    program = textwrap.dedent(
        f"""
        const vm = require("vm");
        const context = {{
          URL,
          location: {{ origin: "http://localhost", href: "http://localhost/" }},
          window: {{ CodoxearUrls: {{ resolveAppUrl: (path) => path }} }},
        }};
        vm.createContext(context);
        vm.runInContext({json.dumps(marked_source)}, context);
        context.window.marked = context.marked;
        vm.runInContext({json.dumps(app_source)}, context);
        const first = context.window.CodoxearMarkdown.mdToHtml({json.dumps(markdown)});
        const second = context.window.CodoxearMarkdown.mdToHtml({json.dumps(markdown)});
        process.stdout.write(JSON.stringify([first, second]));
        """
    )
    proc = subprocess.run(
        ["node", "-e", program],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return tuple(json.loads(proc.stdout))


def test_reported_two_title_line_bold_each_quoted_phrase_only() -> None:
    markdown = "首推是**「欲与天公试比高」**：它把竞争的坐标系。次推**「会当水击三千里」**，如果更想要动感。"

    html = render_markdown(markdown)

    assert "<strong>「欲与天公试比高」</strong>" in html
    assert "<strong>「会当水击三千里」</strong>" in html
    # The CommonMark mis-parse bolds the prose BETWEEN the quoted titles.
    assert "<strong>：它把竞争的坐标系。次推</strong>" not in html
    assert "**" not in html


def test_mirror_case_closer_before_letter_bold_only_the_phrase() -> None:
    html = render_line("**「敢教日月换新天」**很霸气")

    assert html == "<p><strong>「敢教日月换新天」</strong>很霸气</p>"


def test_cjk_period_closer_followed_by_letter_bolds_the_span() -> None:
    # The `**` between 。 and the next CJK letter is not right-flanking under
    # CommonMark (punctuation before, letter after), so the built-in can never
    # close this span; the rescue must pair it with the pending opener.
    html = render_line("**bold。**tail")

    assert html == "<p><strong>bold。</strong>tail</p>"


def test_reported_commander_sentence_bolds_only_the_verdict() -> None:
    markdown = "**成为合格的中场指挥官：概率很高。**他的底层数据（机会创造、推进、xG积累）在最差的时期都是队内顶级，这是能力信号；波动的是终结和默契，这是时间信号。能力为真、时间可解的问题，历史极少以失败告终。"

    html = render_line(markdown)

    assert html == (
        "<p><strong>成为合格的中场指挥官：概率很高。</strong>他的底层数据（机会创造、推进、xG积累）"
        "在最差的时期都是队内顶级，这是能力信号；波动的是终结和默契，这是时间信号。"
        "能力为真、时间可解的问题，历史极少以失败告终。</p>"
    )


def test_rejected_cjk_closer_cannot_open_and_swallow_the_next_span() -> None:
    # Regression: the stranded closer in `。**t` was reinterpreted as an
    # opener and paired with the next span's opener, bolding the prose between
    # the two intended spans and destroying the second one.
    html = render_line("**bold。**tail and **second** end")

    assert html == "<p><strong>bold。</strong>tail and <strong>second</strong> end</p>"


def test_unclosed_cjk_opener_stays_literal() -> None:
    html = render_line("**永远没有闭合的粗体。后续没有分隔符")

    assert html == "<p>**永远没有闭合的粗体。后续没有分隔符</p>"


def test_opener_after_cjk_period_keeps_builtin_path() -> None:
    # `。**n` followed by a letter is a valid CommonMark opener (letter after
    # the run), and its letter-adjacent closer is valid too — nothing needs
    # the rescue, so the built-in result must pass through unchanged.
    html = render_line("句子。**新粗体**后续")

    assert html == "<p>句子。<strong>新粗体</strong>后续</p>"


def test_single_star_cjk_emphasis_rescued() -> None:
    html = render_line("*「斜体」*测试")

    assert html == "<p><em>「斜体」</em>测试</p>"


def test_bold_adjacent_to_cjk_letters_only_keeps_builtin_path() -> None:
    # CJK ideographs are letters, not punctuation: built-in flanking already
    # opens and closes these, and the rescue must stay byte-identical.
    html = render_line("这是**加粗**文本")

    assert html == "<p>这是<strong>加粗</strong>文本</p>"


def test_cjk_bold_surrounded_by_brackets_keeps_builtin_result() -> None:
    html = render_line("「**bold**」")

    assert html == "<p>「<strong>bold</strong>」</p>"


def test_triple_delimiter_cjk_rescues_as_em_strong_like_the_ascii_twin() -> None:
    html = render_line("***「both」***混排")

    assert html == "<p><em><strong>「both」</strong></em>混排</p>"


def test_link_inside_cjk_bold_renders_inside_strong() -> None:
    html = render_line("**「看[文档](https://example.com)」**")

    assert html == '<p><strong>「看<a href="https://example.com">文档</a>」</strong></p>'


def test_star_inside_link_label_is_not_taken_as_the_closer() -> None:
    html = render_line("**「看[a*b](url)」**外")

    assert html == '<p><strong>「看<a href="url">a*b</a>」</strong>外</p>'


def test_consecutive_cjk_bold_spans_stay_separate() -> None:
    html = render_line("**「中文」**English**「更多」**")

    assert html == "<p><strong>「中文」</strong>English<strong>「更多」</strong></p>"


def test_ascii_emphasis_regressions_keep_builtin_output() -> None:
    assert render_line("**bold** and **another**") == "<p><strong>bold</strong> and <strong>another</strong></p>"
    # Intraword `*` is legal CommonMark emphasis; the vendored marked renders
    # it, and the rescue extension must not change that (underscore is the
    # intraword-literal delimiter, covered below).
    # 2*3*4 is now treated as multiplication, not emphasis
    assert render_line("2*3*4") == "<p>2*3*4</p>"
    assert render_line("a_b_c") == "<p>a_b_c</p>"
    assert render_line("foo_bar_baz") == "<p>foo_bar_baz</p>"


def test_strikethrough_behavior_unchanged_including_cjk_brackets() -> None:
    # marked's del rule has no flanking rules, so CJK punctuation never breaks
    # it — the rescue extension intentionally does not cover `~~`.
    assert render_line("~~strike~~ and ~~another~~") == "<p><del>strike</del> and <del>another</del></p>"
    assert render_line("~~「删除」~~测试") == "<p><del>「删除」</del>测试</p>"


def test_code_span_with_cjk_bold_markers_stays_literal() -> None:
    html = render_line("`**「not bold」**`")

    assert html == "<p><code>**「not bold」**</code></p>"


def test_math_placeholders_survive_alongside_cjk_bold() -> None:
    html = render_line("数学 $a*b$ 与 **「粗」**")

    assert "<strong>「粗」</strong>" in html
    assert "md-math-fallback" in html
    assert "**" not in html


def test_rescue_registration_is_idempotent_across_repeated_renders() -> None:
    first, second = render_markdown_twice("首推**「一次」**：再次**「二次」**，完")

    assert first == second
    assert "<strong>「一次」</strong>" in second
    assert "<strong>「二次」</strong>" in second


def test_multiplication_asterisks_not_emphasis() -> None:
    """digit*word patterns (2*H200, 4*A800) must render as literal asterisks."""
    # Same-line
    assert render_line("2*H200 and 4*A800") == "<p>2*H200 and 4*A800</p>"
    # Cross-line (the original bug: * after 2 opened, * after 4 closed)
    html = render_line("- 2*H200, the pod\n- 4*A800, now vacant")
    assert "<em>" not in html
    assert "2*H200" in html
    assert "4*A800" in html
    # Chained: 3*4*5
    assert render_line("3*4*5") == "<p>3*4*5</p>"
    # Does not affect real bold/italic
    assert "<strong>" in render_line("**bold** and 2*H200")
    assert "<em>" in render_line("*italic* and 2*H200")
