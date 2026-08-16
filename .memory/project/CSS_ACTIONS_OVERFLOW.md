# CSS invariant: `.actions` chrome rows are never scroll containers

Two independently reasonable rules collided (fixed at the root in the commit
that introduced this invariant):

1. Base `.actions` (topbar, sidebar header, every dialog header action row)
   carried `overflow-x: auto` + `-webkit-overflow-scrolling: touch` +
   `::-webkit-scrollbar { display: none }`, added as generic mobile
   defensiveness so button rows would scroll instead of spilling.
2. The "hit area is not visual size" rule gives chrome `.icon-btn`s an
   absolutely positioned `::after { inset: -6px }` (form controls: `-2px`).

Absolutely positioned descendants inflate the **scrollable overflow area** of
any ancestor scroll container, and `overflow-x: auto` computes `overflow-y`
to `auto` as well. So every `.actions` row containing hit-area buttons had
permanent phantom overflow on both axes:

- Firefox (which ignores `::-webkit-scrollbar`) showed real horizontal and
  vertical scrollbars around button clusters — e.g. the sidebar notification /
  voice buttons — even when everything visually fit.
- In WebKit/Blink the bars were hidden by the `display: none` hack, but the
  row still clipped anything painted outside a button's border box. This is
  how the `#fileEditBtn.dirty` ring lost its top/bottom segments (patched
  per-instance in d1794f0b before the root fix).

The invariant now: `.actions` sets no `overflow` at all. Rows fit or wrap
(`.queueHeader .actions` and `.fileViewerHeader .actions` wrap). The one
genuine scroller, `.choiceChips`, keeps `overflow-x: auto` but absorbs the
2px hit-area outset with `padding: 2px` so scrollbars appear only on real
chip overflow.

Rule of thumb: never make a row that hosts hit-area pseudo elements or
outline/shadow state rings a scroll container. If scrolling is truly needed,
absorb the pseudo-element outset into the scroller's padding box.

Pinned by `tests/test_chrome_row_overflow_contract.py` (parsed-stylesheet
contract) and the `chrome_rows_not_scrollable` / `chip_rows_no_vertical_phantom`
checks in `scripts/docker_verify.sh` (real-browser measurement).
