import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_MARKDOWN_JS = ROOT / "codoxear" / "static" / "app_markdown.js"


class TestMarkdownPostProcessors(unittest.TestCase):
    def test_local_file_references_are_dom_post_processors(self) -> None:
        source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        self.assertIn("function parseLocalFileRef", source)
        self.assertIn("function localFileRefFromRef", source)
        self.assertIn("function rewriteTextFileRefs", source)
        self.assertIn("function rewriteMarkedLinks", source)
        self.assertIn("dataset.candidateFilePath", source)
        self.assertIn("dataset.candidateFileLine", source)

    def test_inline_code_file_references_preserve_code_wrapper(self) -> None:
        source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        self.assertIn("function rewriteInlineCodeFileRefs", source)
        self.assertIn('root.querySelectorAll("code")', source)
        self.assertIn('code.replaceChildren(createFileRefSpan', source)
        self.assertIn('nodeHasAncestor(code, "PRE")', source)

    def test_marked_output_links_and_images_are_resolved_safely(self) -> None:
        source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        self.assertIn("function safeUrl", source)
        self.assertIn("function rewriteMarkedImages", source)
        self.assertIn("function previewImageUrlForRef", source)
        self.assertIn('link.target = "_blank"', source)
        self.assertIn('link.rel = "noreferrer noopener"', source)
        self.assertIn('image.loading = "lazy"', source)

    def test_post_processors_run_after_marked_parse(self) -> None:
        source = APP_MARKDOWN_JS.read_text(encoding="utf-8")
        self.assertIn("function postProcessMarkedHtml", source)
        self.assertIn("rewriteMarkedLinks(root, document, options)", source)
        self.assertIn("rewriteMarkedImages(root, document, options)", source)
        self.assertIn("rewriteInlineCodeFileRefs(root, document, options)", source)
        self.assertIn("rewriteTextFileRefs(root, document, options)", source)
        self.assertIn("decorateCodeBlocks(root, document)", source)
        self.assertIn("wrapMarkedTables(root, document)", source)
        self.assertIn("postProcessMarkedHtml(window.marked.parse(prepared), options)", source)


if __name__ == "__main__":
    unittest.main()
