// The tree label editor (af.tree.draw.editor). Everything it needs is in the page: the figure
// without its substitution labels (PICTURE), and in #data each label's node, box and glyph width
// as the renderer placed them, the occupancy placement used, and the settings of the renderer's
// checks, which are rerun here as labels move (render._clashes, hand_clashes, hand_leaders,
// place.placement_metrics). Save writes the rows af.tree.draw.hand reads back.
"use strict";

const D = JSON.parse(document.getElementById("data").textContent);
const P = D.page;
const NS = "http://www.w3.org/2000/svg";
const FAMILY = D.font.data ? "af-label" : D.font.family;
const DONE_BY = /^([\s\S]*) \(([^,()]+), (\d{4}-\d{2}-\d{2})\)$/;

function splitReason(text) {
  const m = DONE_BY.exec(text || "");
  return m ? { reason: m[1], by: m[2], date: m[3] } : { reason: text || "", by: "", date: "" };
}

// ---- state --------------------------------------------------------------------------------

const labels = D.labels.map((l, i) => ({
  id: i,
  text: l.text,
  lines: l.text.split(" "),
  first: l.first,
  last: l.last,
  node: l.node,
  box0: l.box.slice(),
  box: l.box.slice(),
  glyphW: l.glyph_w,
  pin: l.pin ? { ...l.pin, ...splitReason(l.pin.reason) } : null,
  hidden: false,
  fileHide: null, // the hide row this label came with, when it was hidden by hand
  why: "",
}));
for (const h of D.hidden) {
  labels.push({
    id: labels.length,
    text: h.text,
    lines: h.text.split(" "),
    first: h.first,
    last: h.last,
    node: h.node,
    box0: h.box.slice(),
    box: h.box.slice(),
    glyphW: h.glyph_w,
    pin: null,
    hidden: true,
    fileHide: { first: h.first, last: h.last, ...splitReason(h.reason) },
    why: "",
  });
}
let selected = null;
let zoom = 1;
const ui = { boxes: false, freeze: true, freezeWhy: "", by: "" };
try { ui.by = localStorage.getItem("af-label-editor-by") || ""; } catch (e) { /* private window */ }

const moved = (l) => !l.hidden && (l.fileHide !== null || l.box.some((v, k) => Math.abs(v - l.box0[k]) > 1e-6));
const handFile = (l) => !l.hidden && l.pin !== null && l.pin.hand && !moved(l);
const frozen = (l) => ui.freeze && !l.hidden && !moved(l) && l.pin === null;
// a hand pin once saved: what the build will never move, and refuses to draw clashing
const willBeHand = (l) => moved(l) || handFile(l) || frozen(l);
const newlyHidden = (l) => l.hidden && l.fileHide === null;
const needsWhy = (l) => (moved(l) || newlyHidden(l)) && !l.why.trim();

// ---- geometry, as render.py and place.py --------------------------------------------------

function glyph(l) {
  const n = l.lines.length;
  const h = P.font_size * (P.pitch * (n - 1) + 0.72);
  const yc = (l.box[1] + l.box[3]) / 2;
  return [l.box[2] - 1 - l.glyphW, yc - h / 2, l.box[2] - 1, yc + h / 2];
}
const leaderEnd = (b) => [b[2], (b[1] + b[3]) / 2];
function hits(a, b, gap, gapY) {
  const gy = gapY === undefined ? gap : gapY;
  return Math.min(a[2], b[2]) + gap > Math.max(a[0], b[0]) && Math.min(a[3], b[3]) + gy > Math.max(a[1], b[1]);
}
function clash(a, b) { // render._clashes, not moving: glyphs within the clearance, or side by side
  const ga = glyph(a), gb = glyph(b);
  if (hits(ga, gb, P.clearance)) return true;
  return P.beside_gap !== null && hits(ga, gb, P.beside_gap, P.row_gap);
}
function segmentHitsBox(p, q, box, pad) {
  const x0 = box[0] - pad, y0 = box[1] - pad, x1 = box[2] + pad, y1 = box[3] + pad;
  const dx = q[0] - p[0], dy = q[1] - p[1];
  let t0 = 0, t1 = 1;
  for (const [pp, qq] of [[-dx, p[0] - x0], [dx, x1 - p[0]], [-dy, p[1] - y0], [dy, y1 - p[1]]]) {
    if (pp === 0) { if (qq < 0) return false; continue; }
    const t = qq / pp;
    if (pp < 0) t0 = Math.max(t0, t); else t1 = Math.min(t1, t);
    if (t0 > t1) return false;
  }
  return true;
}
function segmentsCross(p1, p2, q1, q2) {
  const orient = (a, b, c) => Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
  return orient(p1, p2, q1) !== orient(p1, p2, q2) && orient(q1, q2, p1) !== orient(q1, q2, p2);
}
const INK = Uint8Array.from(atob(D.ink.bits), (c) => c.charCodeAt(0));
function inkUnder(b) { // place.Grid.cost > 0
  const i0 = Math.max(0, Math.floor(b[1])), i1 = Math.min(D.ink.rows, Math.ceil(b[3]));
  const j0 = Math.max(0, Math.floor(b[0])), j1 = Math.min(D.ink.cols, Math.ceil(b[2]));
  for (let i = i0; i < i1; i++) {
    for (let j = j0; j < j1; j++) {
      const k = i * D.ink.cols + j;
      if ((INK[k >> 3] >> (7 - (k & 7))) & 1) return true;
    }
  }
  return false;
}

// ---- checks -------------------------------------------------------------------------------

function problems() {
  const shown = labels.filter((l) => !l.hidden);
  const out = [];
  const add = (level, ids, text) => out.push({ level, ids, text });
  for (let i = 0; i < shown.length; i++) {
    const a = shown[i];
    const la = [leaderEnd(a.box), a.node];
    for (let j = i + 1; j < shown.length; j++) {
      const b = shown[j];
      const lb = [leaderEnd(b.box), b.node];
      if ((willBeHand(a) || willBeHand(b)) && clash(a, b)) {
        add("refused", [a.id, b.id], `${a.text} and ${b.text} too close: the build refuses it`);
      }
      if (segmentsCross(la[0], la[1], lb[0], lb[1])) add("warned", [a.id, b.id], `${a.text} and ${b.text}: leaders cross`);
      const ends = Math.hypot(a.node[0] - b.node[0], a.node[1] - b.node[1]);
      const angle = Math.abs(Math.atan2(la[1][1] - la[0][1], la[1][0] - la[0][0]) - Math.atan2(lb[1][1] - lb[0][1], lb[1][0] - lb[0][0])) * 180 / Math.PI;
      const long = Math.min(Math.hypot(la[1][0] - la[0][0], la[1][1] - la[0][1]), Math.hypot(lb[1][0] - lb[0][0], lb[1][1] - lb[0][1]));
      if (ends > 0 && ends < P.font_size && angle < 12 && long > 3 * P.font_size) {
        add("warned", [a.id, b.id], `${a.text} and ${b.text}: leaders run together to nodes ${ends.toFixed(1)} apart`);
      }
    }
    for (const b of shown) {
      if (b !== a && segmentHitsBox(la[0], la[1], glyph(b), 0)) add("warned", [a.id, b.id], `${a.text}: leader through ${b.text}`);
    }
    if (willBeHand(a) && P.obstacles.some((o) => hits(glyph(a), o, P.clearance))) {
      add("refused", [a.id], `${a.text} on a strain label: the build refuses it`);
    }
    if (inkUnder(a.box)) add("warned", [a.id], `${a.text} on the tree`);
    if (P.inset && segmentHitsBox(la[0], la[1], P.inset, 0)) add("warned", [a.id], `${a.text}: leader over the map`);
    if (a.box[0] < P.left || a.box[1] < P.top || a.box[3] > P.bottom) add("warned", [a.id], `${a.text} outside the label area`);
  }
  return out;
}

// ---- drawing ------------------------------------------------------------------------------

function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const svg = el("svg", { viewBox: `0 0 ${P.width} ${P.height}` });
el("image", { href: PICTURE, x: 0, y: 0, width: P.width, height: P.height, preserveAspectRatio: "none" }, svg);
const leaders = el("g", {}, svg);
const layer = el("g", {}, svg);

function draw(found) {
  const level = {};
  for (const p of found) for (const id of p.ids) if (level[id] !== "refused") level[id] = p.level;
  leaders.replaceChildren();
  layer.replaceChildren();
  layer.setAttribute("class", ui.boxes ? "boxes" : "");
  for (const l of labels) {
    if (l.hidden) continue;
    const [x0, y0, x1, y1] = l.box;
    const end = leaderEnd(l.box);
    el("line", { class: "leader", x1: end[0], y1: end[1], x2: l.node[0], y2: l.node[1] }, leaders);
    const cls = ["lab", level[l.id] || "", moved(l) ? "moved" : "", selected === l.id ? "selected" : ""];
    const g = el("g", { class: cls.join(" "), "data-id": l.id }, layer);
    el("rect", { class: "hit", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, g);
    el("rect", { class: "box", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, g);
    const yc = (y0 + y1) / 2, n = l.lines.length;
    l.lines.forEach((t, i) => {
      const y = yc + (i - (n - 1) / 2) * P.pitch * P.font_size + P.baseline;
      const txt = el("text", {
        x: x1 - 1, y, "text-anchor": "end",
        "font-family": FAMILY, "font-size": P.font_size, "stroke-width": P.stroke || 0,
      }, g);
      txt.textContent = t;
    });
    el("circle", { class: "node" + (selected === l.id ? " on" : ""), cx: l.node[0], cy: l.node[1], r: 1.6 }, layer);
  }
}

// ---- panel --------------------------------------------------------------------------------

const panel = document.createElement("div");
panel.id = "panel";
const stage = document.createElement("div");
stage.id = "stage";
stage.appendChild(svg);
document.body.append(stage, panel);
if (D.font.data) {
  const face = new FontFace("af-label", `url(data:font/${D.font.format === "opentype" ? "otf" : "ttf"};base64,${D.font.data})`);
  face.load().then((f) => { document.fonts.add(f); refresh(); });
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function status(l) {
  if (l.hidden) return l.fileHide ? `hidden by hand: ${l.fileHide.reason} (${l.fileHide.by}, ${l.fileHide.date})` : "hidden now";
  if (l.fileHide) return "brought back: it was hidden by hand";
  const was = l.pin ? (l.pin.hand ? `placed by hand: ${l.pin.reason} (${l.pin.by}, ${l.pin.date})` : `pinned in the round's config: ${l.pin.reason}`) : "placed automatically";
  return moved(l) ? `moved; ${was}` : was;
}

function refresh() {
  const found = problems();
  draw(found);
  svg.setAttribute("width", P.width * zoom);
  svg.setAttribute("height", P.height * zoom);
  const n = (f) => labels.filter(f).length;
  const refused = found.filter((p) => p.level === "refused"), warned = found.filter((p) => p.level === "warned");
  const sel = selected === null ? null : labels[selected];
  const toFreeze = n(frozen);
  panel.innerHTML = `
    <h1>${esc(P.figure || "Tree")}: labels</h1>
    <div class="sub">${esc(D.tree)}</div>
    <div class="sub">Saves as <b>${esc(D.save_as)}</b>, the file the round's trees.toml names.</div>
    <label for="by">Your name or initials</label>
    <input type="text" id="by" value="${esc(ui.by)}" class="${ui.by.trim() ? "" : "need"}">
    <div class="row"><label style="margin:0;font-weight:400"><input type="checkbox" id="freeze" ${ui.freeze ? "checked" : ""}>
      Keep every label I have not moved where it is now (${toFreeze})</label></div>
    ${ui.freeze && toFreeze ? `<label for="freezewhy">Why keep them where they are</label>
      <textarea id="freezewhy" class="${ui.freezeWhy.trim() ? "" : "need"}">${esc(ui.freezeWhy)}</textarea>` : ""}
    <div class="help">${ui.freeze ? "What you see is what prints." : "Labels you have not moved are placed again by the next build, and may move."}</div>
    <div class="row"><button class="primary" id="save">Save</button>
      <button id="boxes">${ui.boxes ? "Hide" : "Show"} boxes</button>
      <button id="out">−</button><button id="fit">Fit</button><button id="in">+</button>
      <button id="resetall">Undo all</button></div>
    <div id="message"></div>
    <div class="counts"><span>${n((l) => !l.hidden)} labels</span><span class="moved">${n(moved)} moved</span>
      <span>${n((l) => l.hidden)} hidden</span><span class="refused">${refused.length} refused</span><span class="warned">${warned.length} warnings</span></div>
    ${sel ? `<h2>Selected</h2>
      <div><b>${esc(sel.text)}</b></div><div class="sub">${esc(status(sel))}</div>
      <div class="sub">node of ${esc(sel.first)} … ${esc(sel.last)}</div>
      ${moved(sel) || newlyHidden(sel) ? `<label for="why">Why (saved with the label)</label>
        <textarea id="why" class="${sel.why.trim() ? "" : "need"}" placeholder="${esc(sel.pin && sel.pin.reason ? "was: " + sel.pin.reason : "")}">${esc(sel.why)}</textarea>` : ""}
      <div class="row"><button id="reset">Undo</button><button id="hide">${sel.hidden ? "Bring back" : "Hide"}</button></div>
      <div class="help">Drag the label, or nudge it with the arrow keys (shift: 10 times as far).</div>` : `<div class="help">Click a label to select it; drag it to move it.</div>`}
    <h2>Problems</h2>
    <ul>${found.map((p, k) => `<li class="${p.level}" data-k="${k}">${esc(p.text)}</li>`).join("") || "<li>none</li>"}</ul>
    <h2>Hidden</h2>
    <ul>${labels.filter((l) => l.hidden).map((l) => `<li data-id="${l.id}">${esc(l.text)}</li>`).join("") || "<li>none</li>"}</ul>
    <h2>How this saves</h2>
    <div class="help">Every label you move or hide needs a why, and every row records your name and the date.
      The build reads the file back, puts each label exactly where you left it and never moves it;
      a label it would have had to move (red here) stops the build with its name.</div>`;
  bind(found);
}

function bind(found) {
  const on = (id, ev, f) => { const e = document.getElementById(id); if (e) e.addEventListener(ev, f); };
  on("by", "input", (e) => { ui.by = e.target.value; e.target.classList.toggle("need", !ui.by.trim()); try { localStorage.setItem("af-label-editor-by", ui.by); } catch (x) { /* ignore */ } });
  on("freeze", "change", (e) => { ui.freeze = e.target.checked; refresh(); });
  on("freezewhy", "input", (e) => { ui.freezeWhy = e.target.value; e.target.classList.toggle("need", !ui.freezeWhy.trim()); });
  on("why", "input", (e) => { labels[selected].why = e.target.value; e.target.classList.toggle("need", !e.target.value.trim()); });
  on("save", "click", save);
  on("boxes", "click", () => { ui.boxes = !ui.boxes; refresh(); });
  on("in", "click", () => { zoom *= 1.25; refresh(); });
  on("out", "click", () => { zoom /= 1.25; refresh(); });
  on("fit", "click", () => { fit(); refresh(); });
  on("resetall", "click", () => { if (confirm("Undo every move and hide made since this page was opened?")) { labels.forEach(undo); refresh(); } });
  on("reset", "click", () => { undo(labels[selected]); refresh(); });
  on("hide", "click", () => { const l = labels[selected]; l.hidden = !l.hidden; if (!l.hidden && !l.fileHide) l.why = ""; refresh(); });
  panel.querySelectorAll("li[data-k]").forEach((li) => li.addEventListener("click", () => { selected = found[+li.dataset.k].ids[0]; refresh(); }));
  panel.querySelectorAll("li[data-id]").forEach((li) => li.addEventListener("click", () => { selected = +li.dataset.id; refresh(); }));
}

function undo(l) {
  l.box = l.box0.slice();
  l.hidden = l.fileHide !== null;
  l.why = "";
}

function fit() { zoom = Math.max(0.2, (stage.clientHeight - 32) / P.height); }

// ---- moving -------------------------------------------------------------------------------

function toFigure(ev) {
  const pt = svg.createSVGPoint();
  pt.x = ev.clientX; pt.y = ev.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}
let drag = null;
svg.addEventListener("pointerdown", (ev) => {
  const g = ev.target.closest(".lab");
  if (!g) return;
  const l = labels[+g.dataset.id];
  selected = l.id;
  const p = toFigure(ev);
  drag = { l, x: p.x, y: p.y, box: l.box.slice() };
  ev.preventDefault();
  refresh();
});
window.addEventListener("pointermove", (ev) => {
  if (!drag) return;
  const p = toFigure(ev);
  const dx = Math.round((p.x - drag.x) * 20) / 20, dy = Math.round((p.y - drag.y) * 20) / 20;
  drag.l.box = [drag.box[0] + dx, drag.box[1] + dy, drag.box[2] + dx, drag.box[3] + dy];
  draw(problems());
});
window.addEventListener("pointerup", () => { if (drag) { drag = null; refresh(); } });
window.addEventListener("keydown", (ev) => {
  if (selected === null || ev.target.closest("input, textarea")) return;
  const step = ev.shiftKey ? 2.5 : 0.25;
  const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[ev.key];
  if (!d) return;
  const l = labels[selected];
  l.box = [l.box[0] + d[0], l.box[1] + d[1], l.box[2] + d[0], l.box[3] + d[1]];
  ev.preventDefault();
  refresh();
});

// ---- saving (af.tree.draw.hand reads this) ------------------------------------------------

const q = (s) => JSON.stringify(String(s)); // a TOML basic string
const num = (v) => (Math.round(v * 1e6) / 1e6).toString();

function save() {
  const msg = document.getElementById("message");
  const by = ui.by.trim();
  const missing = labels.filter(needsWhy);
  const refusals = [];
  if (!by) refusals.push("Write your name or initials: every row records who did it.");
  if (missing.length) refusals.push(`Write why for: ${missing.map((l) => l.text).join(", ")}.`);
  if (ui.freeze && labels.some(frozen) && !ui.freezeWhy.trim()) refusals.push("Write why the labels you have not moved are kept where they are.");
  if (refusals.length) {
    msg.className = "refused";
    msg.textContent = "Not saved. " + refusals.join(" ");
    if (missing.length) { selected = missing[0].id; refresh(); document.getElementById("message").className = "refused"; document.getElementById("message").textContent = "Not saved. " + refusals.join(" "); }
    return;
  }
  const red = problems().filter((p) => p.level === "refused");
  if (red.length && !confirm(`${red.length} problem(s) are red: the build will stop on them until they are moved.\n\n${red.map((p) => p.text).join("\n")}\n\nSave anyway?`)) return;
  const today = new Date();
  const date = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
  const lines = [
    `# Tree labels placed by hand: ${P.figure}. Written by af's label editor (af.tree.draw.editor),`,
    "# read back by af.tree.draw.hand; the round's trees.toml names this file. Each label is on the",
    "# common ancestor of leaves first and last, its box's top-left at the node + (dx, dy) x the page height.",
    `figure = ${q(P.figure)}`,
    `tree = ${q(D.tree)}`,
  ];
  for (const l of labels) {
    if (l.hidden) continue;
    let row = null;
    if (moved(l)) row = { reason: l.why.trim(), by, date };
    else if (handFile(l)) row = { reason: l.pin.reason, by: l.pin.by, date: l.pin.date };
    else if (frozen(l)) row = { reason: ui.freezeWhy.trim(), by, date };
    if (!row) continue;
    const keys = l.pin ? l.pin : l;
    lines.push("", "[[label_pins]]", `first = ${q(keys.first)}`, `last = ${q(keys.last)}`, `text = ${q(l.text)}`,
      `dx = ${num((l.box[0] + 1 - l.node[0]) / P.height)}`, `dy = ${num((l.box[1] - l.node[1]) / P.height)}`,
      `reason = ${q(row.reason)}`, `by = ${q(row.by)}`, `date = ${q(row.date)}`);
  }
  for (const l of labels) {
    if (!l.hidden) continue;
    const row = l.fileHide ? l.fileHide : { first: l.first, last: l.last, reason: l.why.trim(), by, date };
    lines.push("", "[[hide_labels]]", `first = ${q(row.first)}`, `last = ${q(row.last)}`, `text = ${q(l.text)}`,
      `reason = ${q(row.reason)}`, `by = ${q(row.by)}`, `date = ${q(row.date)}`);
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "application/toml" }));
  a.download = D.save_as;
  a.click();
  msg.className = "";
  msg.textContent = `Saved as ${D.save_as} (your downloads folder). Put it beside the round's trees.toml, replacing the old one, and rebuild.`;
}

fit();
refresh();
