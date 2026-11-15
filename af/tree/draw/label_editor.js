// The tree's adapter for the label editor (af.util.label_editor; written by af.tree.draw.editor).
// A label hangs off its node by a leader to its box's right edge (under the ae placer, drawn as
// ae draws it). The checks are the renderer's
// (render.hand_clashes, hand_leaders, place.placement_metrics), rerun as labels move, on each
// label's glyph lines and the occupancy placement used, as the renderer measured them. A label
// that will be a hand pin is checked glyph line to glyph line (ae's ink rule; Q306, Q307).
"use strict";

const ADAPTER = (() => {
  const P = D.adapter;
  const H = D.page.height;
  const INK = Uint8Array.from(atob(P.ink.bits), (c) => c.charCodeAt(0));

  const SIDE = P.side_gap === undefined ? 0.08 * P.font_size : P.side_gap;
  const PAD = P.leader_pad === undefined ? 0.05 * P.font_size : P.leader_pad;
  function glyphLines(l) { // render._glyph_lines: each line its own width, capitals high
    const n = l.lines.length, yc = (l.box[1] + l.box[3]) / 2, right = l.box[2] - 1, half = 0.36 * P.font_size;
    const longest = Math.max(...l.lines.map((t) => t.length));
    const w = l.line_w || l.lines.map((t) => l.glyph_w * t.length / longest); // a page drawn before line_w: mono
    return l.lines.map((t, i) => {
      const cy = yc + (i - (n - 1) / 2) * P.pitch * P.font_size;
      return [right - w[i], cy - half, right, cy + half];
    });
  }
  const leaderEnd = (b) => [b[2], (b[1] + b[3]) / 2];
  const tether = (l) => l.tether || l.anchor; // where the leader ends (render._tether)
  function leader(l) { // render._leader: off the right edge, or a line's top or bottom when over its node
    const t = tether(l);
    if (t[0] >= l.box[2]) return [leaderEnd(l.box), t];
    const lines = glyphLines(l), top = lines[0], bottom = lines[lines.length - 1];
    if (t[1] < (top[1] + bottom[3]) / 2) return [[(top[0] + top[2]) / 2, top[1] - 1], t];
    return [[(bottom[0] + bottom[2]) / 2, bottom[3] + 1], t];
  }
  const AE = P.ae_leader; // under the ae placer: draw each leader as ae does (render._ae_leader)
  function aeLeader(l) { // ae's box (pen advance, AE.row a line), placer_ae attach_pt, drawn_leader
    const fs = P.font_size, x1 = l.box[2] - 1, cy = (l.box[1] + l.box[3]) / 2;
    const th = l.lines.length * AE.row * fs, x0 = x1 - l.advance_w, y0 = cy - th / 2, y1 = cy + th / 2;
    const e = tether(l);
    const c = e[0] > x1 ? [x1, cy] : [(x0 + x1) / 2, y1 < e[1] ? y1 - AE.inset * fs : y0 + AE.inset * fs];
    if (Math.abs(c[0] - e[0]) <= AE.none_within * fs && Math.abs(c[1] - e[1]) <= AE.none_within * fs) return null;
    const lx = e[0] - c[0], ly = e[1] - c[1], ll = Math.hypot(lx, ly), gap = Math.min(fs * AE.gap, ll * 0.4);
    return [e, [c[0] + lx / ll * gap, c[1] + ly / ll * gap]];
  }
  function clash(a, b) { // render._ink_clash: stacked closer than a stack's row gap, else SIDE
    for (const ra of glyphLines(a)) {
      for (const rb of glyphLines(b)) {
        const gx = Math.max(ra[0], rb[0]) - Math.min(ra[2], rb[2]), gy = Math.max(ra[1], rb[1]) - Math.min(ra[3], rb[3]);
        if (gx < 0 && gy < P.row_gap) return true;
        if (gy < 0 && gx < SIDE) return true;
        if (gx >= 0 && gy >= 0 && Math.hypot(gx, gy) < SIDE) return true;
      }
    }
    return false;
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
      if (willBeHand(a) && glyphLines(a).some((r) => P.obstacles.some((o) => hits(r, o, P.clearance)))) {
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
        const ends = Math.hypot(la[1][0] - lb[1][0], la[1][1] - lb[1][1]);
        const angle = Math.abs(Math.atan2(la[1][1] - la[0][1], la[1][0] - la[0][0]) - Math.atan2(lb[1][1] - lb[0][1], lb[1][0] - lb[0][0])) * 180 / Math.PI;
        const long = Math.min(Math.hypot(la[1][0] - la[0][0], la[1][1] - la[0][1]), Math.hypot(lb[1][0] - lb[0][0], lb[1][1] - lb[0][1]));
        if (ends > 0 && ends < P.font_size && angle < 12 && long > 3 * P.font_size) {
          add([a.id, b.id], `${a.text} and ${b.text}: leaders run together to nodes ${ends.toFixed(1)} apart`);
        }
      }
      for (const b of shown) {
        if (b !== a && glyphLines(b).some((r) => segmentHitsBox(la[0], la[1], r, PAD))) add([a.id, b.id], `${a.text}: leader through ${b.text}`);
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
    return { leader: AE ? aeLeader(l) : leader(l), lines }; // checks keep leader(l): the renderer's
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
