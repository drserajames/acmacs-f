// The map's adapter for the label editor (af.util.label_editor; written by af.map.editor).
// A map label is a vaccine mark's name: its box sits beside the mark, with no leader, and prints
// in the figure font with a white halo. The refusals are af.map.labels.hand_clashes, rerun as
// labels move: a hand label off the page, on the legend or title, on another hand label (with the
// placer's gap) or on a mark is what the build stops on, so red here means the build stops.
"use strict";

const ADAPTER = (() => {
  const P = D.adapter;
  const W = D.page.width, H = D.page.height;

  function overlap(a, b) { // af.map.labels._overlap: touching edges are clear
    return !(a[2] <= b[0] || a[0] >= b[2] || a[3] <= b[1] || a[1] >= b[3]);
  }
  const grown = (b, by) => [b[0] - by, b[1] - by, b[2] + by, b[3] + by];
  function onCircle(b, c, r) { // af.map.labels._on_circle
    const dx = Math.max(b[0] - c[0], 0, c[0] - b[2]);
    const dy = Math.max(b[1] - c[1], 0, c[1] - b[3]);
    return dx * dx + dy * dy < r * r - 1e-9;
  }

  function refused(shown, willBeHand) {
    const out = [];
    const held = shown.filter(willBeHand);
    const marks = shown.map((l) => l.anchor); // the marks of the labels the page prints
    for (let i = 0; i < held.length; i++) {
      const a = held[i], b = a.box;
      if (b[0] < 0 || b[1] < 0 || b[2] > W || b[3] > H) out.push({ ids: [a.id], text: `${a.text} runs off the page` });
      for (const f of P.furniture) {
        if (overlap(b, f.box)) out.push({ ids: [a.id], text: `${a.text} is on the ${f.name}` });
      }
      for (let j = i + 1; j < held.length; j++) {
        if (overlap(grown(b, P.gap), held[j].box)) {
          out.push({ ids: [a.id, held[j].id], text: `${a.text} is on the label ${held[j].text}` });
        }
      }
      if (marks.some((m) => onCircle(b, m, P.mark_radius))) out.push({ ids: [a.id], text: `${a.text} is on a vaccine mark` });
    }
    return out;
  }

  function warned(shown) { // drawn points under a label: the halo keeps it readable, so information only
    const out = [];
    for (const l of shown) {
      const n = P.points.filter((p) => p[0] > l.box[0] && p[0] < l.box[2] && p[1] > l.box[1] && p[1] < l.box[3]).length;
      if (n > 0) out.push({ ids: [l.id], text: `${l.text} covers ${n} point${n === 1 ? "" : "s"}` });
    }
    return out;
  }

  return {
    // the box's top-left at the point + (dx, dy) in fractions of the page (af.map.labels.pinned_box)
    offset: (l) => ({ dx: (l.box[0] - l.anchor[0]) / W, dy: (l.box[1] - l.anchor[1]) / H }),
    refused,
    warned,
    // on the baseline that puts the glyphs in the box, as af.map.render draws them
    draw: (l) => ({ leader: null, lines: [{ x: l.box[0], y: l.box[3] - l.descent, text: l.text, anchor: "start" }] }),
    textStyle: () => ({
      "font-size": P.font_size, fill: "#000000", stroke: "#ffffff", "stroke-width": P.halo,
      "paint-order": "stroke", "stroke-linejoin": "round",
    }),
  };
})();
