// The tree's adapter for the label editor (af.util.label_editor; written by af.tree.draw.editor).
// A label hangs off its node by a leader to its box's right edge. The checks are the renderer's
// (render._clashes, hand_clashes, hand_leaders, place.placement_metrics), rerun as labels move,
// on each label's glyph width and the occupancy placement used, as the renderer measured them.
"use strict";

const ADAPTER = (() => {
  const P = D.adapter;
  const H = D.page.height;
  const INK = Uint8Array.from(atob(P.ink.bits), (c) => c.charCodeAt(0));

  function glyph(l) {
    const n = l.lines.length;
    const h = P.font_size * (P.pitch * (n - 1) + 0.72);
    const yc = (l.box[1] + l.box[3]) / 2;
    return [l.box[2] - 1 - l.glyph_w, yc - h / 2, l.box[2] - 1, yc + h / 2];
  }
  const leaderEnd = (b) => [b[2], (b[1] + b[3]) / 2];
  const leader = (l) => [leaderEnd(l.box), l.anchor];
  function clash(a, b) { // render._clashes, not moving: glyphs within the clearance, or side by side
    const ga = glyph(a), gb = glyph(b);
    if (hits(ga, gb, P.clearance)) return true;
    return P.beside_gap !== null && hits(ga, gb, P.beside_gap, P.row_gap);
  }
  function inkUnder(b) { // place.Grid.cost > 0
    const i0 = Math.max(0, Math.floor(b[1])), i1 = Math.min(P.ink.rows, Math.ceil(b[3]));
    const j0 = Math.max(0, Math.floor(b[0])), j1 = Math.min(P.ink.cols, Math.ceil(b[2]));
    for (let i = i0; i < i1; i++) {
      for (let j = j0; j < j1; j++) {
        const k = i * P.ink.cols + j;
        if ((INK[k >> 3] >> (7 - (k & 7))) & 1) return true;
      }
    }
    return false;
  }

  function refused(shown, willBeHand) {
    const out = [];
    for (let i = 0; i < shown.length; i++) {
      const a = shown[i];
      for (let j = i + 1; j < shown.length; j++) {
        const b = shown[j];
        if ((willBeHand(a) || willBeHand(b)) && clash(a, b)) {
          out.push({ ids: [a.id, b.id], text: `${a.text} and ${b.text} too close: the build refuses it` });
        }
      }
      if (willBeHand(a) && P.obstacles.some((o) => hits(glyph(a), o, P.clearance))) {
        out.push({ ids: [a.id], text: `${a.text} on a strain label: the build refuses it` });
      }
    }
    return out;
  }

  function warned(shown) {
    const out = [];
    const add = (ids, text) => out.push({ ids, text });
    for (let i = 0; i < shown.length; i++) {
      const a = shown[i];
      const la = leader(a);
      for (let j = i + 1; j < shown.length; j++) {
        const b = shown[j];
        const lb = leader(b);
        if (segmentsCross(la[0], la[1], lb[0], lb[1])) add([a.id, b.id], `${a.text} and ${b.text}: leaders cross`);
        const ends = Math.hypot(a.anchor[0] - b.anchor[0], a.anchor[1] - b.anchor[1]);
        const angle = Math.abs(Math.atan2(la[1][1] - la[0][1], la[1][0] - la[0][0]) - Math.atan2(lb[1][1] - lb[0][1], lb[1][0] - lb[0][0])) * 180 / Math.PI;
        const long = Math.min(Math.hypot(la[1][0] - la[0][0], la[1][1] - la[0][1]), Math.hypot(lb[1][0] - lb[0][0], lb[1][1] - lb[0][1]));
        if (ends > 0 && ends < P.font_size && angle < 12 && long > 3 * P.font_size) {
          add([a.id, b.id], `${a.text} and ${b.text}: leaders run together to nodes ${ends.toFixed(1)} apart`);
        }
      }
      for (const b of shown) {
        if (b !== a && segmentHitsBox(la[0], la[1], glyph(b), 0)) add([a.id, b.id], `${a.text}: leader through ${b.text}`);
      }
      if (inkUnder(a.box)) add([a.id], `${a.text} on the tree`);
      if (P.inset && segmentHitsBox(la[0], la[1], P.inset, 0)) add([a.id], `${a.text}: leader over the map`);
      if (a.box[0] < P.left || a.box[1] < P.top || a.box[3] > P.bottom) add([a.id], `${a.text} outside the label area`);
    }
    return out;
  }

  function draw(l) {
    const yc = (l.box[1] + l.box[3]) / 2, n = l.lines.length;
    const lines = l.lines.map((text, i) => ({
      x: l.box[2] - 1,
      y: yc + (i - (n - 1) / 2) * P.pitch * P.font_size + P.baseline,
      text,
      anchor: "end",
    }));
    return { leader: leader(l), lines };
  }

  return {
    // the box's top-left at the node + (dx, dy) x the page height (render._pinned_box)
    offset: (l) => ({ dx: (l.box[0] + 1 - l.anchor[0]) / H, dy: (l.box[1] - l.anchor[1]) / H }),
    refused,
    warned,
    draw,
    textStyle: () => ({ "font-size": P.font_size, "stroke-width": P.stroke || 0, fill: "#4d4d4d", stroke: "#4d4d4d" }),
  };
})();
