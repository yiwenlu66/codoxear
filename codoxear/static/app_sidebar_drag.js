// Pure geometry and decision helpers for the sidebar drag gesture.
//
// The pointer session in app_sessions.js measures the live DOM and asks these
// functions what a drop means. Nothing here touches the DOM or holds state: the
// server owns order (session_listing.sort_session_rows), so a drop resolves to
// ids only — the dragged row's new neighbours in the Now block — and the server
// computes the offset.
//
// Gesture map (source row group × drop target group):
//
//   source │ within Now │ Now header    │ Later block/header  │ Waiting
//   Now    │ reorder    │ move to top   │ snooze indefinitely │ refuse
//   Later  │ —          │ wake, at slot │ cancel              │ refuse

const DRAGGABLE_GROUPS = new Set(["now", "later"]);

function dropKind({ sourceGroup, targetGroup } = {}) {
  if (!DRAGGABLE_GROUPS.has(sourceGroup)) return "none";
  if (targetGroup === "waiting") return "refuse";
  if (sourceGroup === "now") {
    if (targetGroup === "now") return "reorder";
    if (targetGroup === "later") return "snooze";
    return "none";
  }
  // Later keeps recency order: Later → Later is a cancel, not a reorder.
  if (targetGroup === "now") return "wake";
  return "none";
}

// Index the dragged row should occupy among `cards` (the destination block's
// other cards, in rendered order): the number of cards whose midpoint lies
// above the pointer. Upper half of a card → before it, lower half → after it.
function insertionIndex(cards, pointerY) {
  const list = Array.isArray(cards) ? cards : [];
  const y = Number(pointerY);
  if (!Number.isFinite(y)) return list.length;
  let index = 0;
  for (const card of list) {
    const top = Number(card && card.top);
    const height = Number(card && card.height);
    if (!Number.isFinite(top) || !Number.isFinite(height)) continue;
    if (top + height / 2 < y) index += 1;
    else break;
  }
  return index;
}

// The destination block's id order after the drop, and the API neighbours of
// the dragged id in it (null at either end of the block).
function nextOrder(ids, draggedId, index) {
  const rest = (Array.isArray(ids) ? ids : []).filter((id) => id !== draggedId);
  const at = Math.max(0, Math.min(rest.length, Math.trunc(Number(index)) || 0));
  const order = [...rest.slice(0, at), draggedId, ...rest.slice(at)];
  return {
    order,
    above: at > 0 ? rest[at - 1] : null,
    below: at < rest.length ? rest[at] : null,
  };
}

function sameOrder(a, b) {
  return a.length === b.length && a.every((id, index) => id === b[index]);
}

// What releasing the pointer here would write. `cards` are the destination
// block's measured cards ({id, top, height}, rendered order, the dragged card
// may be among them); `targetIds` the same block's ids. Returns
// {kind: "none"} for every drop that writes nothing: cancel, refuse, or an
// unchanged Now order.
function dragPayload({ sourceGroup, targetGroup, overHeader, sourceId, targetIds, cards, pointerY } = {}) {
  const kind = dropKind({ sourceGroup, targetGroup, overHeader });
  if (kind === "snooze") return { kind, order: null, above: null, below: null, index: null };
  if (kind !== "reorder" && kind !== "wake") return { kind: "none" };
  const ids = Array.isArray(targetIds) ? targetIds : [];
  const others = (Array.isArray(cards) ? cards : []).filter((card) => card && card.id !== sourceId);
  const index = overHeader ? 0 : insertionIndex(others, pointerY);
  const { order, above, below } = nextOrder(ids, sourceId, index);
  if (kind === "reorder" && sameOrder(order, ids)) return { kind: "none" };
  return { kind, order, above, below, index };
}

// Content-space Y of the insertion line for slot `index` among `cards` (the
// destination block's other cards): midway between the two neighbours, just
// above the first card, just below the last, or under the header of an empty
// block. Returns the line's centre; the caller offsets by half its thickness.
function insertionLineY(cards, index, headerBottom) {
  const list = Array.isArray(cards) ? cards : [];
  const at = Math.max(0, Math.min(list.length, Math.trunc(Number(index)) || 0));
  const bottomOf = (card) => Number(card.top) + Number(card.height);
  if (!list.length) return Number(headerBottom) + LINE_EDGE_INSET_PX;
  if (at === 0) return Number(list[0].top) - LINE_EDGE_INSET_PX;
  if (at === list.length) return bottomOf(list[list.length - 1]) + LINE_EDGE_INSET_PX;
  return (bottomOf(list[at - 1]) + Number(list[at].top)) / 2;
}

const LINE_EDGE_INSET_PX = 4;
const AUTO_SCROLL_EDGE_PX = 48;
const AUTO_SCROLL_MIN_PX = 6;
const AUTO_SCROLL_MAX_PX = 14;

// Signed per-frame scroll step for a pointer at `pointerY` in a container
// spanning [top, bottom] (client coordinates): 0 outside the 48px edge bands,
// 6px at the band's inner edge rising to 14px at (and beyond) the container
// edge, negative toward the top.
function autoScrollStep(top, bottom, pointerY) {
  const y = Number(pointerY);
  if (!Number.isFinite(y)) return 0;
  const speed = (distance) => {
    const depth = 1 - Math.max(0, distance) / AUTO_SCROLL_EDGE_PX;
    return Math.round(AUTO_SCROLL_MIN_PX + (AUTO_SCROLL_MAX_PX - AUTO_SCROLL_MIN_PX) * depth);
  };
  const fromTop = y - Number(top);
  const fromBottom = Number(bottom) - y;
  if (fromTop < AUTO_SCROLL_EDGE_PX && fromTop <= fromBottom) return -speed(fromTop);
  if (fromBottom < AUTO_SCROLL_EDGE_PX) return speed(fromBottom);
  return 0;
}

export { dropKind, insertionIndex, nextOrder, dragPayload, insertionLineY, autoScrollStep };
