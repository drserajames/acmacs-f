// A label editor (af.util.label_editor): the page is common, the figure's adapter (ADAPTER) knows
// its labels' geometry and its build's checks. Everything it needs is in the page: the figure
// without its movable labels (PICTURE), and in #data each label's anchor, box and keys as the
// build placed them. Save writes the rows the figure's build reads back.
"use strict";

const D = JSON.parse(document.getElementById("data").textContent);
const NS = "http://www.w3.org/2000/svg";
const FAMILY = D.font.data ? "af-label" : D.font.family;
const TABLES = { pin: "label_pins", hide: "hide_labels", ...(D.tables || {}) };

// ---- geometry the adapters share ----------------------------------------------------------

function hits(a, b, gap, gapY) { // boxes [x0, y0, x1, y1] closer than gap (gapY up and down)
  const gy = gapY === undefined ? gap : gapY;
  return Math.min(a[2], b[2]) + gap > Math.max(a[0], b[0]) && Math.min(a[3], b[3]) + gy > Math.max(a[1], b[1]);
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

// ---- the editor ---------------------------------------------------------------------------

function startLabelEditor(A) {
  const P = D.page;

  // ---- state ------------------------------------------------------------------------------

  const labels = D.labels.map((l) => ({
    ...l,
    box0: l.box.slice(),
    box: l.box.slice(),
    pin: l.hidden ? null : l.pin,
    fileHide: l.hidden ? l.hide : null, // the hide row this label came with
    why: "",
  }));
  let selected = null;
  let zoom = 1;
  let notice = ""; // said until the next change
  let edits = 0, noticeAt = 0; // every change counts; a notice shows while no change has followed it
  const say = (t) => { notice = t; noticeAt = edits; };
  const ui = { boxes: false, freeze: true, freezeWhy: "", by: "" };
  try { ui.by = localStorage.getItem("af-label-editor-by") || ""; } catch (e) { /* private window */ }

  // Undo steps back one change at a time (a drag, a nudge, a hide, a put back), Cmd/Ctrl-Z too.
  // A step is every label's place before the change, so undoing never depends on which is selected.
  const history = [];
  const snapshot = () => labels.map((l) => ({ box: l.box.slice(), hidden: l.hidden, why: l.why }));
  const same = (a, b) => a.every((s, k) => s.hidden === b[k].hidden && s.box.every((v, j) => v === b[k].box[j]));
  function remember(id, kind) { history.push({ id, kind, at: Date.now(), was: snapshot() }); edits++; }
  function stepBack() {
    const step = history.pop();
    if (!step) {
      say("Nothing to undo: no label has been moved, nudged, hidden or put back since this page was opened.");
      return;
    }
    edits++;
    step.was.forEach((s, k) => { labels[k].box = s.box; labels[k].hidden = s.hidden; labels[k].why = s.why; });
    selected = step.id;
  }

  // The page remembers its work in this browser, so a reload or a crash before Save loses nothing.
  // Keyed by the page's own data: a page drawn again (from a saved file) starts from its own build.
  const KEPT = "af-label-editor:" + hash(document.getElementById("data").textContent);
  function keep() {
    const changed = labels.filter((l) => l.hidden !== (l.fileHide !== null) || l.why || l.box.some((v, k) => v !== l.box0[k]));
    try {
      if (!changed.length && !ui.freezeWhy) localStorage.removeItem(KEPT);
      else localStorage.setItem(KEPT, JSON.stringify({ at: new Date().toISOString(), freeze: ui.freeze, freezeWhy: ui.freezeWhy,
        labels: changed.map((l) => ({ id: l.id, text: l.text, box: l.box, hidden: l.hidden, why: l.why })) }));
    } catch (e) { /* storage full or blocked: Save still works */ }
  }
  const fits = (kept) => kept && Array.isArray(kept.labels) && kept.labels.length > 0
    && kept.labels.every((k) => labels[k.id] && labels[k.id].text === k.text);
  function restore() {
    let kept = null;
    try { kept = JSON.parse(localStorage.getItem(KEPT) || "null"); } catch (e) { return null; }
    if (kept && !fits(kept)) return null; // not this page's: leave it alone
    if (!kept) kept = earlier();
    if (!kept) return null;
    for (const k of kept.labels) Object.assign(labels[k.id], { box: k.box, hidden: k.hidden, why: k.why });
    ui.freeze = kept.freeze; ui.freezeWhy = kept.freezeWhy || "";
    return kept;
  }
  // A page redrawn with the same labels (a new script, or a field added to its data) has a new key: take the
  // newest work kept by an earlier copy whose every label is this page's, by place and text. Never deleted.
  function earlier() {
    let best = null;
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (!key || !key.startsWith("af-label-editor:") || key === KEPT || key.endsWith(":test")) continue;
        const kept = JSON.parse(localStorage.getItem(key) || "null");
        if (fits(kept) && (!best || kept.at > best.at)) best = { ...kept, from: key };
      }
    } catch (e) { return null; }
    return best;
  }

  const moved = (l) => !l.hidden && (l.fileHide !== null || l.box.some((v, k) => Math.abs(v - l.box0[k]) > 1e-6));
  const handFile = (l) => !l.hidden && l.pin !== null && l.pin.hand && !moved(l);
  const frozen = (l) => ui.freeze && !l.hidden && !moved(l) && l.pin === null;
  // a hand pin once saved: what the build will never move, and refuses to draw clashing
  const willBeHand = (l) => moved(l) || handFile(l) || frozen(l);
  const newlyHidden = (l) => l.hidden && l.fileHide === null;
  // A why is optional: a row with none says what was done, with who and when (design rule 9).
  const AUTO = {
    moved: "placed by hand in the label editor",
    hidden: "hidden by hand in the label editor",
    frozen: "kept where the build placed it, in the label editor",
  };
  let askedBy = false; // initials are asked for at most once on a page, then remembered

  function problems() {
    const shown = labels.filter((l) => !l.hidden);
    return [
      ...A.refused(shown, willBeHand).map((p) => ({ level: "refused", ...p })),
      ...A.warned(shown).map((p) => ({ level: "warned", ...p })),
    ];
  }

  // ---- drawing ----------------------------------------------------------------------------

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
      const drawn = A.draw(l);
      if (drawn.leader) {
        const [p, q] = drawn.leader;
        el("line", { class: "leader", x1: p[0], y1: p[1], x2: q[0], y2: q[1] }, leaders);
      }
      const cls = ["lab", level[l.id] || "", moved(l) ? "moved" : "", selected === l.id ? "selected" : ""];
      const g = el("g", { class: cls.join(" "), "data-id": l.id }, layer);
      el("rect", { class: "hit", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, g);
      el("rect", { class: "box", x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, g);
      const style = A.textStyle(l);
      for (const line of drawn.lines) {
        const txt = el("text", { x: line.x, y: line.y, "text-anchor": line.anchor, "font-family": FAMILY, ...style }, g);
        txt.textContent = line.text;
      }
      el("circle", { class: "node" + (selected === l.id ? " on" : ""), cx: l.anchor[0], cy: l.anchor[1], r: 1.6 }, layer);
    }
  }

  // ---- panel ------------------------------------------------------------------------------

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
  const named = (l) => Object.values(l.keys).filter((v) => typeof v === "string").join(" … "); // a count checks, it does not name

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
    keep();
    panel.innerHTML = `
      <h1>${esc(D.title)}</h1>
      ${D.subtitle ? `<div class="sub">${esc(D.subtitle)}</div>` : ""}
      <div class="sub">Saves as <b>${esc(D.save_as)}</b>${D.saves ? ", " + esc(D.saves) : ""}.</div>
      <label for="by">Your initials (remembered)</label>
      <input type="text" id="by" value="${esc(ui.by)}">
      <div class="row"><label style="margin:0;font-weight:400"><input type="checkbox" id="freeze" ${ui.freeze ? "checked" : ""}>
        Keep every label I have not moved where it is now (${toFreeze})</label></div>
      ${ui.freeze && toFreeze ? `<label for="freezewhy">Why keep them where they are (optional)</label>
        <textarea id="freezewhy" placeholder="${esc(AUTO.frozen)}">${esc(ui.freezeWhy)}</textarea>` : ""}
      <div class="help">${ui.freeze ? "What you see is what prints." : "Labels you have not moved are placed again by the next build, and may move."}</div>
      <div class="row"><button class="primary" id="save">Save ${esc(D.save_as)}</button>
        <button id="boxes">${ui.boxes ? "Hide" : "Show"} boxes</button>
        <button id="out">−</button><button id="fit">Fit</button><button id="in">+</button></div>
      <div class="row"><button id="undo" title="Cmd-Z or Ctrl-Z">Undo${history.length ? " " + esc(stepName(history[history.length - 1])) : ""}</button>
        <button id="resetall">Put all back</button></div>
      <div id="message">${notice && noticeAt === edits ? esc(notice) : ""}</div>
      <div class="counts"><span>${n((l) => !l.hidden)} labels</span><span class="moved">${n(moved)} moved</span>
        <span>${n((l) => l.hidden)} hidden</span><span class="refused">${refused.length} refused</span><span class="warned">${warned.length} warnings</span></div>
      ${sel ? `<h2>Selected</h2>
        <div><b>${esc(sel.text)}</b></div><div class="sub">${esc(status(sel))}</div>
        <div class="sub">${esc(D.names || "of")} ${esc(named(sel))}</div>
        ${moved(sel) || newlyHidden(sel) ? `<label for="why">Why (optional)</label>
          <textarea id="why" placeholder="${esc(sel.hidden ? AUTO.hidden : AUTO.moved)}">${esc(sel.why)}</textarea>` : ""}
        <div class="row"><button id="reset" title="where the build placed it">Put back</button><button id="hide">${sel.hidden ? "Bring back" : "Hide"}</button></div>
        <div class="help">Drag the label, or nudge it with the arrow keys (shift: 10 times as far).</div>` : `<div class="help">Click a label to select it; drag it to move it.</div>`}
      <h2>Problems</h2>
      <ul>${found.map((p, k) => `<li class="${p.level}" data-k="${k}">${esc(p.text)}</li>`).join("") || "<li>none</li>"}</ul>
      <h2>Hidden</h2>
      <ul>${labels.filter((l) => l.hidden).map((l) => `<li data-id="${l.id}">${esc(l.text)}</li>`).join("") || "<li>none</li>"}</ul>
      <h2>How this saves</h2>
      <div class="help">Save always saves. A why is optional: a label without one is saved as "${esc(AUTO.moved)}"
        (or hidden, or kept), with your initials and the date. The build reads the file back, puts each label exactly
        where you left it and never moves it; a label it would have had to move (red here) stops the build with its name.</div>`;
    bind(found);
  }

  function bind(found) {
    const on = (id, ev, f) => { const e = document.getElementById(id); if (e) e.addEventListener(ev, f); };
    on("by", "input", (e) => { ui.by = e.target.value; try { localStorage.setItem("af-label-editor-by", ui.by); } catch (x) { /* ignore */ } });
    on("freeze", "change", (e) => { ui.freeze = e.target.checked; refresh(); });
    on("freezewhy", "input", (e) => { ui.freezeWhy = e.target.value; });
    on("why", "input", (e) => { labels[selected].why = e.target.value; });
    on("save", "click", save);
    on("boxes", "click", () => { ui.boxes = !ui.boxes; refresh(); });
    on("in", "click", () => { zoom *= 1.25; refresh(); });
    on("out", "click", () => { zoom /= 1.25; refresh(); });
    on("fit", "click", () => { fit(); refresh(); });
    on("undo", "click", () => { stepBack(); refresh(); });
    on("resetall", "click", () => { if (confirm("Put every label back where the build placed it? (Undo then steps back from this.)")) { remember(null, "put all back"); labels.forEach(putBack); refresh(); } });
    on("reset", "click", () => { remember(selected, "put back"); putBack(labels[selected]); refresh(); });
    on("hide", "click", () => { const l = labels[selected]; remember(l.id, l.hidden ? "bring back" : "hide"); l.hidden = !l.hidden; if (!l.hidden && !l.fileHide) l.why = ""; refresh(); });
    on("freezewhy", "input", keep);
    on("why", "input", keep);
    // a problem names two labels: clicking it again selects the other one
    panel.querySelectorAll("li[data-k]").forEach((li) => li.addEventListener("click", () => {
      const ids = found[+li.dataset.k].ids;
      selected = ids[(ids.indexOf(selected) + 1) % ids.length];
      refresh();
    }));
    panel.querySelectorAll("li[data-id]").forEach((li) => li.addEventListener("click", () => { selected = +li.dataset.id; refresh(); }));
  }

  function putBack(l) {
    l.box = l.box0.slice();
    l.hidden = l.fileHide !== null;
    l.why = "";
  }
  function stepName(step) {
    const what = step.id === null ? "" : " " + labels[step.id].text;
    return `(${step.kind}${what})`;
  }

  function fit() { zoom = Math.max(0.2, (stage.clientHeight - 32) / P.height); }

  // ---- moving -----------------------------------------------------------------------------

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
    drag = { l, x: p.x, y: p.y, box: l.box.slice(), was: snapshot() };
    try { svg.setPointerCapture(ev.pointerId); } catch (e) { /* not captured: window still ends it */ }
    ev.preventDefault();
    refresh();
  });
  function endDrag() {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (!same(d.was, snapshot())) { history.push({ id: d.l.id, kind: "move", at: Date.now(), was: d.was }); edits++; }
    refresh();
  }
  window.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    if (ev.pointerType === "mouse" && !(ev.buttons & 1)) { endDrag(); return; } // released where the page did not see it
    const p = toFigure(ev);
    const dx = Math.round((p.x - drag.x) * 20) / 20, dy = Math.round((p.y - drag.y) * 20) / 20;
    drag.l.box = [drag.box[0] + dx, drag.box[1] + dy, drag.box[2] + dx, drag.box[3] + dy];
    draw(problems());
  });
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);
  window.addEventListener("blur", endDrag);
  window.addEventListener("keydown", (ev) => {
    if (ev.target.closest && ev.target.closest("input, textarea")) return;
    if ((ev.metaKey || ev.ctrlKey) && !ev.shiftKey && ev.key.toLowerCase() === "z") { ev.preventDefault(); stepBack(); refresh(); return; }
    if (selected === null) return;
    const step = ev.shiftKey ? 2.5 : 0.25;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[ev.key];
    if (!d) return;
    const l = labels[selected];
    // nudges of one label in a row are one step
    const last = history[history.length - 1];
    if (last && last.kind === "nudge" && last.id === l.id && Date.now() - last.at < 2000) { last.at = Date.now(); edits++; }
    else remember(l.id, "nudge");
    l.box = [l.box[0] + d[0], l.box[1] + d[1], l.box[2] + d[0], l.box[3] + d[1]];
    ev.preventDefault();
    refresh();
  });

  // ---- saving -----------------------------------------------------------------------------

  const q = (s) => JSON.stringify(String(s)); // a TOML basic string
  const num = (v) => (Math.round(v * 1e6) / 1e6).toString();
  const keyLines = (keys) => Object.entries(keys).map(([k, v]) => `${k} = ${typeof v === "number" ? num(v) : q(v)}`);

  function save() {
    // Save always saves: nothing typed is required, and red only stops the build, never the save.
    if (!ui.by.trim() && !askedBy) {
      askedBy = true;
      let given = null;
      try { given = window.prompt("Your initials, saved with each label (asked once):", ""); } catch (e) { /* no dialogs */ }
      if (given && given.trim()) {
        ui.by = given.trim();
        try { localStorage.setItem("af-label-editor-by", ui.by); } catch (e) { /* ignore */ }
      }
    }
    const by = ui.by.trim() || "not given";
    const red = problems().filter((p) => p.level === "refused");
    const why = (l, auto) => l.why.trim() || auto;
    const today = new Date();
    const date = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
    const lines = [...(D.comment || []).map((c) => `# ${c}`), ...keyLines(D.header || {})];
    for (const l of labels) {
      if (l.hidden) continue;
      let r = null;
      if (moved(l)) r = { reason: why(l, AUTO.moved), by, date };
      else if (handFile(l)) r = { reason: l.pin.reason, by: l.pin.by, date: l.pin.date };
      else if (frozen(l)) r = { reason: ui.freezeWhy.trim() || AUTO.frozen, by, date };
      if (!r) continue;
      const o = A.offset(l);
      lines.push("", `[[${TABLES.pin}]]`, ...keyLines((l.pin && l.pin.keys) || l.keys), `text = ${q(l.text)}`,
        `dx = ${num(o.dx)}`, `dy = ${num(o.dy)}`, `reason = ${q(r.reason)}`, `by = ${q(r.by)}`, `date = ${q(r.date)}`);
    }
    for (const l of labels) {
      if (!l.hidden) continue;
      const r = l.fileHide ? l.fileHide : { reason: why(l, AUTO.hidden), by, date };
      lines.push("", `[[${TABLES.hide}]]`, ...keyLines(r.keys || l.keys), `text = ${q(l.text)}`,
        `reason = ${q(r.reason)}`, `by = ${q(r.by)}`, `date = ${q(r.date)}`);
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "application/toml" }));
    a.download = D.save_as;
    a.click();
    const at = today.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const placed = labels.filter((l) => !l.hidden && (moved(l) || handFile(l) || frozen(l))).length;
    say(`Saved at ${at}: ${D.save_as} went to your browser's downloads folder, normally Downloads (if a file of that `
      + `name was already there, the browser may have added a number to the new one's name). It places ${placed} label(s) and hides `
      + `${labels.filter((l) => l.hidden).length}.`
      + (red.length ? ` ${red.length} red problem(s) will stop the build until they are moved: save again after moving them.` : "")
      + (D.saved ? " " + D.saved : ""));
    refresh();
  }

  let keeps = true;
  try { localStorage.setItem(KEPT + ":test", "1"); keeps = localStorage.getItem(KEPT + ":test") === "1"; localStorage.removeItem(KEPT + ":test"); } catch (e) { keeps = false; }
  const kept = keeps ? restore() : null;
  if (!keeps) say("This browser will not keep unsaved work on this page: Save before you reload or close it.");
  if (kept) say(`Your unsaved work ${kept.from ? "from an earlier copy of this page" : "on this page"}, from ${new Date(kept.at).toLocaleString()}, is back (${kept.labels.length} label(s)). Put all back starts again from the build.`);
  fit();
  refresh();
}

function hash(s) { // FNV-1a: which page this is, for what the browser keeps
  let h = 0x811c9dc5;
  for (let k = 0; k < s.length; k++) { h ^= s.charCodeAt(k); h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16) + "-" + s.length;
}
