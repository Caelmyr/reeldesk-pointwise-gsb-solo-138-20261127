import {$, button, checkbox, describeDate, formValues, h, input, link, message, numberField, request, select, svg, table, textarea} from "./ui.js";
import {panelNodes} from "./panels.js";
import {resultNodes, showPacket} from "./results.js";

const state = {
  rolls: [], roll: null, scans: [], scan: null, marks: [], records: [], operationTitles: {},
  mode: "positive", tool: "box", tab: "inspect", inspectKind: "coverage",
  box: null, line: null, point: null, reference: null, drag: null, activeMark: null,
  peerId: "", alignment: null, alignMethod: "patch", landmarks: [],
  wedge: [], cutPlan: null, cutOptions: {kerf_mm: .16, reserve_mm: .7, retained: "both"},
  lastDust: null, restore: new Set(), pickedDust: null,
  calibration: cleanCalibration(), peerCalibration: cleanCalibration(),
  tone: {shadow: .08, highlight: 1.75, gamma: 1, neutralize: false, brightness: 0},
  gradePairs: [], gradePaper: {black8: 14, white8: 242}, adoptedGrade: null, proofSize: 1800,
  preview: null, requestId: 0, previewRequest: null, loadId: 0, working: false,
};

const defaultMarks = ["frame", "keep", "reference", "dust", "scratch", "perforation", "cut"];
const descriptions = {scan: "Film scan", flat: "Backlight flat", dark: "Shutter dark", reference: "Reference image"};
const media = [["negative", "Color or B&W negative"], ["transparency", "Transparency"], ["paper", "Paper positive"], ["control", "Scanner control"]];
const decisions = [["hold", "Hold"], ["accepted", "Accepted"], ["rejected", "Rejected"]];
const storage = {
  read(key) { try { return localStorage.getItem(key); } catch { return null; } },
  write(key, value) { try { localStorage.setItem(key, value); } catch { /* Storage can be disabled. */ } },
};

function cleanCalibration() {
  return {flat_id: "", dark_id: "", base: [1, 1, 1], reject_clipped: true};
}

function calibrationFor(scan) {
  if (!scan?.sample || scan.role !== "scan") return cleanCalibration();
  const number = scan.name.endsWith("02") ? "02" : "01";
  const controls = kind => state.scans.find(item => item.sample && item.role === kind && item.name.endsWith(number))?.id ?? "";
  return {flat_id: controls("flat"), dark_id: controls("dark"), base: [.91, .68, .43], reject_clipped: true};
}

let toastTimer;
function toast(text, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = text;
  $("toast").classList.toggle("error-toast", error);
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 7000 : 3200);
}

function errorText(error) {
  return error instanceof Error ? error.message : String(error);
}

function perform(action) {
  return (...args) => Promise.resolve().then(() => action(...args)).catch(error => toast(errorText(error), true));
}

function all() {
  const scan = state.scan;
  return {x: 0, y: 0, w: scan.width, h: scan.height};
}

function need(value, hint) {
  if (!value) throw Error(hint);
  return value;
}

const needBox = () => need(state.box, "Draw a region or choose a saved frame first.");
const needLine = () => need(state.line, "Use Line to draw between two positions on the scan.");
const needPoint = () => need(state.point, "Use Point to place a measurement on the scan.");
const needAlignment = () => need(state.alignment, "Align the repeat scan first.");

function pathFor(scan, name) {
  return `/api/scans/${scan.id}/${name}`;
}

function clickSource() {
  if (!state.scan) throw Error("Import or select a scan first.");
  return state.scan;
}

function localCal(form) {
  const value = formValues(form);
  return {
    flat_id: value.flat_id ?? "",
    dark_id: value.dark_id ?? "",
    base: [0, 1, 2].map(key => Number(value[`base${key}`])),
    reject_clipped: value.reject_clipped === "on",
  };
}

function pairInputs() {
  if (!state.peerId) throw Error("Select a repeat scan.");
  return {peer_id: state.peerId, source_calibration: state.calibration, peer_calibration: state.peerCalibration};
}

function pairWithAlignment() {
  return {...pairInputs(), transform: needAlignment().transform};
}

function dropResult() {
  $("result-section").hidden = true;
  $("result-section").replaceChildren();
}

function setActive(container, key, value) {
  for (const button of $(container).querySelectorAll("button")) {
    const active = button.dataset[key] === value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function renderSidebar() {
  const roll = state.roll;
  $("roll-title").textContent = roll?.title || "Film archive";
  $("roll-detail").textContent = roll ? `${roll.code} · ${roll.stock}` : "Choose a roll to begin.";
  $("roll-menu").disabled = !roll;
  $("import-open").disabled = !roll;
  const groups = [
    ["Scans", state.scans.filter(item => item.role === "scan")],
    ["Controls and references", state.scans.filter(item => item.role !== "scan")],
  ];
  const list = $("scan-list");
  list.replaceChildren();
  for (const [title, scans] of groups) {
    if (!scans.length) continue;
    list.append(h("p", {class: "group-label"}, title, h("span", {class: "number"}, scans.length)));
    for (const scan of scans) {
      const thumbnail = new URL(pathFor(scan, "thumbnail"), location.origin);
      if (scan.role !== "scan") thumbnail.searchParams.set("view", "raw");
      const active = scan.id === state.scan?.id;
      list.append(button([
        h("img", {class: "scan-thumb", alt: "", src: thumbnail.href, loading: "lazy"}),
        h("span", {}, h("span", {class: "scan-name", title: scan.name}, scan.name),
          h("span", {class: "scan-role"}, scan.role === "scan" ? `${scan.width} × ${scan.height}` : descriptions[scan.role])),
      ], perform(() => selectScan(scan)), `scan-entry ${active ? "active" : ""}`, {"aria-current": active ? "true" : undefined}));
    }
  }
  $("roll-foot").replaceChildren(
    h("strong", {}, state.scans.some(scan => scan.sample) ? "Synthetic practice roll" : "Catalog storage"),
    roll?.location ? h("p", {}, roll.location) : h("p", {}, "Stored in the local catalog."),
  );
}

function renderMarks() {
  const container = $("marks");
  container.replaceChildren();
  const scan = state.scan;
  if (!scan || !state.marks.length) {
    container.append(h("p", {class: "muted"}, "No saved marks."));
    return;
  }
  const priority = {frame: 0, reference: 1, keep: 2, cut: 3, perforation: 4, scratch: 5, dust: 6};
  const ordered = [...state.marks].sort((a, b) => priority[a.kind] - priority[b.kind] ||
    (a.geometry.x ?? a.geometry.start.x) - (b.geometry.x ?? b.geometry.start.x) ||
    a.label.localeCompare(b.label));
  for (const mark of ordered) {
    let preview = null;
    if (mark.kind === "frame") {
      preview = h("span", {class: "frame-pic", "aria-hidden": "true"});
      const {w, h: height, x, y} = mark.geometry;
      if (state.preview) {
        preview.style.backgroundImage = `url("${state.preview}")`;
        preview.style.backgroundSize = `${scan.width / w * 100}% ${scan.height / height * 100}%`;
        preview.style.backgroundPosition = `${scan.width === w ? 0 : x / (scan.width - w) * 100}% ${scan.height === height ? 0 : y / (scan.height - height) * 100}%`;
      }
    }
    container.append(button([preview, h("span", {class: "mark-details"},
      h("span", {class: `mark-dot ${mark.kind}`}), h("strong", {title: mark.label}, mark.label))],
      () => activateMark(mark), `frame-label ${state.activeMark === mark.id ? "active" : ""}`,
      {ondblclick: () => openMark(mark), "aria-label": `Select ${mark.label}`}));
  }
}

function renderHeader() {
  const scan = state.scan;
  $("scan-title").textContent = scan?.name ?? "Load a scan";
  $("current-type").textContent = scan ? descriptions[scan.role].toUpperCase() : "SCAN VIEWER";
  $("scan-size").textContent = scan ? `${scan.width} × ${scan.height} px${scan.dpi ? ` · ${scan.dpi} dpi` : ""}` : "—";
  $("scan-menu").disabled = !scan;
  $("mark-open").disabled = !scan;
  $("clear-selection").disabled = !scan;
  $("image-state").hidden = Boolean(state.preview);
  $("image-state").textContent = scan ? "Loading source pixels…" : "Import a scan to use the workbench";
  setActive("view-modes", "view", state.mode);
  setActive("geometry-tools", "tool", state.tool);
}

function renderPanel() {
  setActive("inspector-tabs", "tab", state.tab);
  $("panel").replaceChildren(...panelNodes(state, actions));
}

function rect(geometry, attrs) {
  return svg("rect", {x: geometry.x, y: geometry.y, width: geometry.w, height: geometry.h, ...attrs});
}

function svgLine(geometry, attrs) {
  return svg("line", {x1: geometry.start.x, y1: geometry.start.y, x2: geometry.end.x, y2: geometry.end.y, ...attrs});
}

function renderOverlay() {
  const overlay = $("scan-overlay");
  const scan = state.scan;
  overlay.replaceChildren();
  if (!scan) return;
  overlay.setAttribute("viewBox", `0 0 ${scan.width} ${scan.height}`);
  overlay.setAttribute("preserveAspectRatio", "none");
  for (const mark of state.marks) {
    const attrs = {class: `saved-box is-${mark.kind}`};
    overlay.append(mark.kind === "cut" ? svgLine(mark.geometry, attrs) : rect(mark.geometry, attrs));
  }
  if (state.reference) overlay.append(rect(state.reference, {id: "comparison-reference"}));
  if (state.box) overlay.append(rect(state.box, {class: "live-drawing"}));
  if (state.line) overlay.append(svgLine(state.line, {class: "live-drawing"}));
  if (state.point) {
    const scale = Math.max(2, scan.width * .005);
    overlay.append(svg("circle", {cx: state.point.x, cy: state.point.y, r: scale, class: "live-drawing"}));
  }
  if (state.cutPlan) {
    overlay.append(svgLine(state.cutPlan, {class: "saved-box is-reference"}));
  }
  const label = state.point ? `Point ${state.point.x}, ${state.point.y}`
    : state.line ? `Line ${state.line.start.x}, ${state.line.start.y} → ${state.line.end.x}, ${state.line.end.y}`
    : state.box ? `Region ${state.box.w} × ${state.box.h} px · x ${state.box.x}, y ${state.box.y}`
    : "Draw a region to inspect";
  $("selection-label").textContent = label;
}

function clearGeometry({all = false} = {}) {
  state.box = state.line = state.point = state.drag = state.activeMark = null;
  if (all) state.reference = state.cutPlan = null;
  renderOverlay();
  renderMarks();
}

function activateMark(mark) {
  clearGeometry();
  if (mark.kind === "cut") {
    state.tool = "line";
    state.line = mark.geometry;
  } else {
    state.tool = "box";
    state.box = mark.geometry;
  }
  state.activeMark = mark.id;
  setActive("geometry-tools", "tool", state.tool);
  renderOverlay();
  renderMarks();
}

function scanPoint(event, target = $("scan-overlay"), size = state.scan) {
  const bounds = target.getBoundingClientRect();
  if (!bounds.width || !bounds.height || !size) return null;
  const x = Math.min(size.width - 1, Math.max(0, Math.floor((event.clientX - bounds.left) / bounds.width * size.width)));
  const y = Math.min(size.height - 1, Math.max(0, Math.floor((event.clientY - bounds.top) / bounds.height * size.height)));
  return {x, y};
}

function regionFrom(start, end) {
  return {x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), w: Math.abs(end.x - start.x) + 1, h: Math.abs(end.y - start.y) + 1};
}

function pointerBegin(event) {
  if (!state.scan || !state.preview || event.button !== 0) return;
  const start = scanPoint(event);
  if (!start) return;
  state.activeMark = null;
  if (state.tool === "point") {
    state.point = start;
    state.line = null;
  } else {
    state.point = null;
    state.drag = {pointer: event.pointerId, start};
    event.currentTarget.setPointerCapture(event.pointerId);
    if (state.tool === "box") {
      state.line = null;
      state.box = regionFrom(start, start);
    }
  }
  renderMarks();
  renderOverlay();
}

function pointerMove(event) {
  if (state.drag?.pointer !== event.pointerId) return;
  const end = scanPoint(event);
  if (!end) return;
  if (state.tool === "box") state.box = regionFrom(state.drag.start, end);
  else state.line = {start: state.drag.start, end};
  renderOverlay();
}

function pointerEnd(event) {
  if (state.drag?.pointer !== event.pointerId) return;
  pointerMove(event);
  if (state.line && Math.hypot(state.line.end.x - state.line.start.x, state.line.end.y - state.line.start.y) < 1) state.line = null;
  state.drag = null;
  renderOverlay();
}

async function updatePreview() {
  const scan = state.scan;
  state.previewRequest?.abort();
  const current = state.preview;
  if (!scan) {
    $("scan-image").removeAttribute("src");
    if (current) URL.revokeObjectURL(current);
    state.preview = null;
    renderHeader();
    return;
  }
  const controller = new AbortController();
  state.previewRequest = controller;
  try {
    const blob = await request(pathFor(scan, "preview"), {
      method: "POST", binary: true, signal: controller.signal,
      data: {view: state.mode, calibration: state.calibration, tone: state.tone},
    });
    if (state.previewRequest !== controller || state.scan?.id !== scan.id) return;
    state.preview = URL.createObjectURL(blob);
    $("scan-image").src = state.preview;
    $("scan-image").style.aspectRatio = `${scan.width} / ${scan.height}`;
    $("film-canvas").style.maxWidth = `${Math.min(1350, Math.max(95, 520 * scan.width / scan.height))}px`;
    if (current) URL.revokeObjectURL(current);
    renderHeader();
    renderMarks();
  } catch (error) {
    if (!controller.signal.aborted) {
      if (!state.preview) $("image-state").textContent = "This preview could not be opened.";
      toast(errorText(error), true);
    }
  }
}

async function selectScan(scan, {persist = true} = {}) {
  const stamp = ++state.loadId;
  state.scan = scan || null;
  state.marks = [];
  state.peerId = state.scans.find(item => item.role === "scan" && item.id !== scan?.id)?.id || "";
  state.alignment = null;
  state.landmarks = [];
  state.wedge = [];
  state.calibration = calibrationFor(scan);
  state.peerCalibration = calibrationFor(state.scans.find(item => item.id === state.peerId));
  state.lastDust = null;
  state.restore.clear();
  state.pickedDust = null;
  state.gradePairs = [];
  state.adoptedGrade = null;
  if (state.tone?.curve) {
    const {curve, ...tone} = state.tone;
    state.tone = tone;
  }
  state.mode = scan?.role === "scan" ? "positive" : "raw";
  state.tool = "box";
  state.cutOptions = {kerf_mm: .16, reserve_mm: .7, retained: "both"};
  if (state.preview) URL.revokeObjectURL(state.preview);
  state.preview = null;
  $("scan-image").removeAttribute("src");
  clearGeometry({all: true});
  dropResult();
  renderSidebar();
  renderHeader();
  renderPanel();
  if (!scan) return;
  if (persist) storage.write("reeldesk.scan", scan.id);
  const preview = updatePreview();
  const marks = await request(pathFor(scan, "marks"));
  if (state.loadId === stamp) {
    state.marks = marks.items;
    renderMarks();
    renderOverlay();
  }
  await preview;
}

async function refreshMarks() {
  const scan = clickSource();
  const result = await request(pathFor(scan, "marks"));
  if (state.scan?.id === scan.id) {
    state.marks = result.items;
    renderMarks();
    renderOverlay();
  }
}

async function loadRoll(id, desiredScan = null) {
  const stamp = ++state.requestId;
  const [roll, collection, records] = await Promise.all([
    request(`/api/rolls/${id}`),
    request(`/api/rolls/${id}/scans`),
    request(`/api/rolls/${id}/records`),
  ]);
  if (stamp !== state.requestId) return;
  state.roll = roll;
  state.scans = collection.items;
  state.records = records.items;
  storage.write("reeldesk.roll", roll.id);
  const matching = state.scans.find(item => item.id === desiredScan);
  await selectScan(matching || state.scans.find(item => item.role === "scan") || state.scans[0]);
}

async function refreshRecords() {
  if (!state.roll) return;
  const id = state.roll.id;
  const data = await request(`/api/rolls/${id}/records`);
  if (state.roll?.id === id) {
    state.records = data.items;
    renderPanel();
  }
}

function operationArgs(kind, element, overrides) {
  const fields = element ? formValues(element) : {};
  const calibration = state.calibration;
  const region = state.box ?? all();
  const base = {kind, calibration};
  switch (kind) {
    case "density": return {...base, region: needBox()};
    case "point": return {...base, point: needPoint()};
    case "profile": return {...base, line: needLine()};
    case "balance": return {...base, region: needBox(), reference: need(state.reference, "Keep a reference region first.")};
    case "calibration": return {...base, region, rows: +fields.rows, columns: +fields.columns};
    case "coverage": return {...base, region, coverage: {channel: fields.channel ?? "rgb"}};
    case "grain": return {...base, region: needBox(), scales: [0, 1, 2].map(i => +fields[`scale${i}`])};
    case "focus": return {...base, region: needBox(), focus: {orientation: fields.orientation ?? "auto"}};
    case "wedge": return {...base, wedge: {
      patches: state.wedge, tolerance: +fields.tolerance, compensate: fields.compensate === "on",
    }};
    case "dust": return {...base, region: needBox(), survey: {radius: +fields.radius, threshold: +fields.threshold}};
    case "restore": return {kind, ...overrides};
    case "dust-check": return {kind, ...pairWithAlignment(), ...overrides};
    case "registration": {
      const common = {kind, ...pairInputs(), method: state.alignMethod, view: fields.view ?? "overlay"};
      if (state.alignMethod === "patch") return {...common, region: needBox(), maximum: +fields.maximum};
      return {...common, region, model: fields.model ?? "shift", pairs: state.landmarks, robust: fields.robust === "on"};
    }
    case "compare": return {kind, ...pairWithAlignment(), point: needPoint()};
    case "residual": return {kind, ...pairWithAlignment(), region};
    case "cut": {
      const options = state.cutOptions = {kerf_mm: +fields.kerf_mm, reserve_mm: +fields.reserve_mm,
        dpi: +fields.dpi, retained: fields.retained, guide: region};
      return {kind, cut: options, line: state.line};
    }
    case "cut-verified": return {kind, cut: state.cutOptions, planned: state.cutPlan, actual: needLine()};
    case "frames": case "perforations": case "carrier": return {kind, region};
    case "proof": {
      state.proofSize = +fields.max_side || 1800;
      const payload = {...base, region, tone: state.tone, view: state.mode,
        proof: {max_side: state.proofSize, annotate: fields.annotate === "on",
          include_histogram: fields.include_histogram === "on"}};
      if (state.adoptedGrade && fields.grade_audit === "on") payload.grade = state.adoptedGrade;
      return payload;
    }
    case "strip-grade": {
      state.gradePaper = {black8: +fields.paper_black || 14, white8: +fields.paper_white || 242};
      const keepRef = fields.keep_ref || "auto";
      const payload = {
        kind, calibration, domain: "density",
        paper: {black: state.gradePaper.black8, white: state.gradePaper.white8},
        pairs: state.gradePairs.map(pair => ({...pair})),
        proof_max_side: +fields.proof_size || 1200,
      };
      if (keepRef === "none") payload.use_saved_keep = false;
      else if (keepRef !== "auto") payload.keep = keepRef;
      return payload;
    }
    case "density-master": return {...base, region, density_export: {
      minimum: +fields.minimum, maximum: +fields.maximum, clip_outside: fields.clip_outside === "on",
    }};
    case "contact": return {...base, tone: state.tone, sheet: {paper: fields.paper ?? "a4", columns: +fields.columns || 3}};
    case "masters": return {kind, ...(fields.useRegion === "on" ? {region: needBox()} : {}),
      masters: {format: fields.format ?? "tiff", padding: +fields.padding || 0}};
    default: throw Error("This operation is not available.");
  }
}

function flatNodes(nodes) {
  return nodes.flat(Infinity).filter(node =>
    node !== null && node !== undefined && node !== false && node !== true);
}

function showResult(entry, saved = false) {
  const result = $("result-section");
  result.replaceChildren(...flatNodes(resultNodes(entry, state, actions, saved)));
  result.hidden = false;
  result.scrollIntoView({block: "nearest", behavior: "instant"});
}

async function runOperation(kind, element, overrides) {
  const scan = clickSource();
  if (state.working) throw Error("Wait for the current operation to finish.");
  const data = operationArgs(kind, element, overrides);
  const button = element?.querySelector('button[type="submit"]');
  if (button) button.disabled = true;
  state.working = true;
  try {
    const job = await request(pathFor(scan, "analysis"), {method: "POST", data});
    if (scan.id !== state.scan?.id) return;
    if (kind === "registration" && data.peer_id === state.peerId) {
      state.alignment = {transform: job.result.transform, method: data.method};
      toast(job.result.ambiguous ? "Check the alignment preview. This patch had a plausible second match." : "Repeat scan aligned.");
      renderPanel();
    }
    if (kind === "dust") {
      state.lastDust = job;
      state.restore.clear();
      state.pickedDust = null;
      renderPanel();
    }
    showResult(job);
  } finally {
    state.working = false;
    if (button) button.disabled = false;
  }
}

let dialogSubmit = null;

function dialog(title, contents, {submitText = "", submit = null, footer = []} = {}) {
  const element = $("dialog");
  $("dialog-title").textContent = title;
  $("dialog-body").replaceChildren(...contents);
  $("dialog-error").textContent = "";
  const controls = [...footer];
  if (submitText) controls.push(h("button", {type: "submit", class: "primary"}, submitText));
  $("dialog-footer").replaceChildren(...controls);
  dialogSubmit = submit;
  if (!element.open) element.showModal();
}

function closeDialog() {
  dialogSubmit = null;
  if ($("dialog").open) $("dialog").close();
}

function dialogAction(action) {
  return async () => {
    try {
      await action();
    } catch (error) {
      $("dialog-error").textContent = errorText(error);
    }
  };
}

async function library(search = "") {
  const reply = await request(`/api/rolls?q=${encodeURIComponent(search)}`);
  state.rolls = reply.items;
  const results = state.rolls.map(roll =>
    button(h("span", {}, roll.title, h("small", {}, `${roll.code} · ${roll.scan_count} scans`)),
      dialogAction(async () => { await loadRoll(roll.id); closeDialog(); }), "list-button"));
  dialog("Roll library", [
    input("q", "Find by title, stock, code, or location", search, {type: "search", autofocus: true}),
    ...(results.length ? results : [h("p", {class: "muted"}, "No matching rolls.")]),
  ], {
    submitText: "Search",
    submit: async form => library(formValues(form).q),
    footer: [
      button("New roll", () => rollEditor()),
      button("Load practice roll", dialogAction(async () => {
        const demo = await request("/api/demo", {method: "POST", data: {}});
        await loadRoll(demo.roll_id, demo.scans[0]);
        closeDialog();
      })),
    ],
  });
}

function rollEditor(existing = null) {
  dialog(existing ? "Edit roll" : "New film roll", [
    input("title", "Roll title", existing?.title, {required: true, maxlength: 100}),
    input("code", "Archive code", existing?.code, {required: true, maxlength: 35}),
    input("stock", "Film stock", existing?.stock, {required: true, maxlength: 100}),
    input("location", "Physical location", existing?.location),
    input("tags", "Tags, comma separated", existing?.tags?.join(", ")),
    textarea("notes", "Notes", existing?.notes),
  ], {
    submitText: existing ? "Save roll" : "Create roll",
    submit: async form => {
      const value = formValues(form);
      value.tags = value.tags.split(",").map(item => item.trim()).filter(Boolean);
      const roll = await request(existing ? `/api/rolls/${existing.id}` : "/api/rolls",
        {method: existing ? "PATCH" : "POST", data: {...value, revision: existing?.revision}});
      if (existing) {
        state.roll = roll;
        renderSidebar();
      } else {
        await loadRoll(roll.id);
      }
      closeDialog();
      toast(existing ? "Roll updated." : "Roll created.");
    },
  });
}

function scanEditor() {
  const scan = clickSource();
  const confirmation = () => dialog("Remove scan?", [
    h("p", {}, "Its unreferenced original and marks will be removed from the catalog."),
  ], {
    submitText: "Remove scan",
    submit: async () => {
      await request(`/api/scans/${scan.id}?revision=${scan.revision}`, {method: "DELETE"});
      await loadRoll(state.roll.id);
      closeDialog();
      toast("Scan removed.");
    },
    footer: [button("Cancel", scanEditor)],
  });
  dialog("Scan details", [
    input("name", "Name", scan.name, {required: true, maxlength: 100}),
    select("medium", "Film or control type", media, scan.medium),
    input("dpi", "Scanner DPI", scan.dpi ?? "", {type: "number", min: 30, max: 20000, step: .01}),
    input("scanner", "Device", scan.scanner),
    input("exposure", "Exposure or pass settings", scan.exposure),
    select("verdict", "Scan verdict", decisions, scan.verdict),
    textarea("notes", "Notes", scan.notes),
    h("p", {class: "muted"}, `${scan.source_format} · ${scan.mode} · ${scan.width} × ${scan.height} · SHA-256 ${scan.checksum.slice(0, 12)}…`),
  ], {
    submitText: "Save scan details",
    submit: async form => {
      const updated = await request(`/api/scans/${scan.id}`,
        {method: "PATCH", data: {...formValues(form), revision: scan.revision}});
      state.scans = state.scans.map(item => item.id === scan.id ? updated : item);
      state.scan = updated;
      renderSidebar();
      renderHeader();
      await updatePreview();
      closeDialog();
      toast("Scan details updated.");
    },
    footer: [link("Original", pathFor(scan, "original"), {download: true}), button("Remove scan", confirmation, "text-button")],
  });
}

function importDialog() {
  const roll = need(state.roll, "Create or choose a roll first.");
  dialog("Import scan or control", [
    input("file", "Original image file", "", {type: "file", accept: ".png,.tif,.tiff,.jpg,.jpeg,.webp,image/*", required: true}),
    select("role", "Image role", Object.entries(descriptions), "scan"),
    select("medium", "Film or control type", media, "negative"),
    input("name", "Catalog name (optional)", "", {maxlength: 100}),
    input("dpi", "Scanner DPI (optional)", "", {type: "number", min: 30, max: 20000, step: .01}),
    input("scanner", "Device", "", {maxlength: 100}),
    input("exposure", "Exposure or pass settings", "", {maxlength: 100}),
    textarea("notes", "Notes"),
  ], {
    submitText: "Store original",
    submit: async form => {
      const values = formValues(form);
      const payload = new FormData();
      payload.append("file", values.file);
      delete values.file;
      if (!values.name) delete values.name;
      if (!values.dpi) delete values.dpi;
      payload.append("metadata", JSON.stringify(values));
      const scan = await request(`/api/rolls/${roll.id}/scans`, {method: "POST", file: payload});
      await loadRoll(roll.id, scan.id);
      closeDialog();
      toast("Scan stored.");
    },
  });
}

function openMark(existing = null, givenBox = null, suggested = null) {
  const scan = clickSource();
  const geometry = existing?.geometry ?? givenBox ?? (state.tool === "line" ? state.line : state.box);
  if (!geometry) throw Error("Draw a region or a line before adding a mark.");
  const isLine = Boolean(geometry.start);
  const choices = isLine ? ["cut"] : defaultMarks.filter(item => item !== "cut");
  dialog(existing ? "Edit mark" : "Add scan mark", [
    select("kind", "Mark type", choices, existing?.kind || choices[0]),
    input("label", "Label", existing?.label ?? suggested ?? "", {required: true, maxlength: 80, autofocus: true}),
    textarea("notes", "Notes", existing?.notes ?? "", {maxlength: 1000}),
  ], {
    submitText: "Save mark",
    submit: async form => {
      const data = {...formValues(form), geometry, revision: existing?.revision};
      await request(existing ? `/api/marks/${existing.id}` : pathFor(scan, "marks"), {
        method: existing ? "PATCH" : "POST", data,
      });
      await refreshMarks();
      closeDialog();
      toast("Mark saved.");
    },
    footer: existing ? [button("Remove mark", () => dialog("Remove mark?", [h("p", {}, existing.label)], {
      submitText: "Remove mark",
      submit: async () => {
        await request(`/api/marks/${existing.id}?revision=${existing.revision}`, {method: "DELETE"});
        await refreshMarks();
        closeDialog();
        toast("Mark removed.");
      },
    }), "text-button")] : [],
  });
}

function addLandmark() {
  const source = needPoint();
  const peer = need(state.scans.find(item => item.id === state.peerId), "Select the repeat scan.");
  const image = h("img", {
    src: `${pathFor(peer, "thumbnail")}?side=520&view=raw`, alt: "Repeat scan landmark preview",
    style: "display:block;width:100%;cursor:crosshair;border-radius:5px;",
    onclick: event => {
      const mapped = scanPoint(event, event.currentTarget, peer);
      if (!mapped) return;
      state.landmarks.push({source, peer: mapped});
      closeDialog();
      renderPanel();
      toast(`Landmark ${state.landmarks.length} added.`);
    },
  });
  dialog("Pair the landmark", [h("p", {class: "muted"}, "Click the same image detail in the repeat scan."), image]);
}

async function saveJob(job) {
  const record = await request(`/api/jobs/${job.id}/save`, {method: "POST", data: {}});
  await refreshRecords();
  showResult(record, true);
  toast("Review saved.");
}

async function openRecord(id) {
  const data = await request(`/api/records/${id}`);
  if (state.scan?.id !== data.scan_id) {
    const source = state.scans.find(item => item.id === data.scan_id);
    if (source) await selectScan(source);
  }
  showResult(data, true);
}

function recordDecision(entry) {
  dialog("Review decision", [
    input("title", "Review title", entry.title, {required: true, maxlength: 100}),
    select("decision", "Decision", decisions, entry.decision),
  ], {
    submitText: "Save decision",
    submit: async form => {
      const updated = await request(`/api/records/${entry.id}`,
        {method: "PATCH", data: {...formValues(form), revision: entry.revision}});
      await refreshRecords();
      showResult(updated, true);
      closeDialog();
      toast("Review decision saved.");
    },
    footer: [button("Remove review", () => dialog("Remove this review?", [
      h("p", {}, "The review and its files will be removed from the catalog."),
    ], {
      submitText: "Remove review", submit: async () => {
        await request(`/api/records/${entry.id}?revision=${entry.revision}`, {method: "DELETE"});
        await refreshRecords();
        dropResult();
        closeDialog();
        toast("Review removed.");
      }
    }), "text-button")],
  });
}

async function verifyPacket(form) {
  const data = await request("/api/packets/verify", {method: "POST", file: new FormData(form)});
  const result = $("result-section");
  result.replaceChildren(...showPacket(data));
  result.hidden = false;
  result.scrollIntoView({block: "nearest"});
}

async function openActivity() {
  const reply = await request(`/api/rolls/${state.roll.id}/activity`);
  dialog("Roll activity", [table(["Operation", "Time"], reply.items.map(row => [row.operation, describeDate(row.created)]))]);
}

function setPeer(id) {
  state.peerId = id;
  state.peerCalibration = calibrationFor(state.scans.find(item => item.id === state.peerId));
  state.alignment = null;
  state.landmarks = [];
  renderPanel();
}

async function setBase() {
  const scan = clickSource();
  const result = await request(pathFor(scan, "base"), {method: "POST",
    data: {region: needBox(), calibration: state.calibration}});
  if (state.scan?.id !== scan.id) return;
  state.calibration.base = result.base.map(value => Number(value.toFixed(3)));
  await updatePreview();
  renderPanel();
  toast("Film base applied.");
}

async function setCalibration(form) {
  const previous = state.calibration;
  state.calibration = localCal(form);
  const current = state.preview;
  await updatePreview();
  if (state.preview === current) state.calibration = previous;
  renderPanel();
}

async function setTone(form) {
  const previous = state.tone;
  const fields = formValues(form);
  state.tone = {shadow: +fields.shadow, highlight: +fields.highlight, gamma: +fields.gamma,
    brightness: +fields.brightness, neutralize: fields.neutralize === "on"};
  const current = state.preview;
  await updatePreview();
  if (state.preview === current) state.tone = previous;
  renderPanel();
}

function geometryFromRef(reference) {
  if (reference === "drawn") return needBox();
  const mark = state.marks.find(item => item.id === reference);
  if (!mark?.geometry || mark.geometry.x === undefined) {
    throw Error("Choose a saved box mark or draw the region first.");
  }
  return mark.geometry;
}

function addGradePair(form) {
  const fields = formValues(form);
  const dark = geometryFromRef(fields.dark_ref);
  const bright = geometryFromRef(fields.bright_ref);
  const gap8 = Math.min(200, Math.max(2, +fields.minimum || 28));
  state.gradePairs.push({
    id: `ui-pair-${state.gradePairs.length + 1}-${Date.now().toString(36)}`,
    label: fields.label || `Texture ${state.gradePairs.length + 1}`,
    dark, bright, minimum: gap8 / 255,
  });
  renderPanel();
  toast("Reference pair added.");
}

function removeGradePair(index) {
  state.gradePairs.splice(index, 1);
  renderPanel();
}

async function adoptGrade(result) {
  state.adoptedGrade = result;
  state.tone = {...state.tone, curve: result.curve};
  state.mode = "positive";
  setActive("view-modes", "view", state.mode);
  renderHeader();
  renderPanel();
  await updatePreview();
  renderPanel();
  toast("Unified curve adopted. Open Archive ▸ Make scan proof.");
}

async function clearAdoptedGrade() {
  state.adoptedGrade = null;
  const {curve, ...tone} = state.tone;
  state.tone = tone;
  renderPanel();
  await updatePreview();
  renderPanel();
  toast("Adopted curve cleared.");
}

function openArchive() {
  state.tab = "archive";
  renderPanel();
}

function openStrip() {
  state.tab = "strip";
  renderPanel();
}

function restoreAlignment(entry) {
  const peer = need(state.scans.find(item => item.id === entry.peer_id), "The saved repeat scan is not available.");
  state.peerId = peer.id;
  state.alignment = {transform: entry.result.transform, method: entry.settings.method};
  state.peerCalibration = entry.settings.peer_calibration;
  state.calibration = entry.settings.source_calibration;
  state.tab = "repeat";
  renderPanel();
  updatePreview();
  toast("Saved alignment adopted.");
}

function restoreChoices() {
  const old = need(state.lastDust, "Run a dust survey first.");
  if (!state.restore.size) throw Error("Choose at least one unprotected finding.");
  return runOperation("restore", null, {
    calibration: old.settings.calibration, region: old.settings.region, survey: old.settings.survey,
    findings: [...state.restore].sort((left, right) => left - right).map(number => ({number, state: "restore"})),
  });
}

function checkDust() {
  const old = need(state.lastDust, "Run a dust survey first.");
  const selected = need(state.pickedDust, "Select a dust finding in the survey result.");
  return runOperation("dust-check", null, {
    region: old.settings.region, survey: old.settings.survey,
    source_calibration: old.settings.calibration, finding: selected,
  });
}

const actions = {
  run: perform(runOperation), setCalibration: perform(setCalibration), setTone: perform(setTone), setBase: perform(setBase),
  addGradePair: perform(addGradePair), removeGradePair, adoptGrade: perform(adoptGrade),
  clearAdoptedGrade: perform(clearAdoptedGrade), openArchive, openStrip,
  keepReference: perform(() => { state.reference = needBox(); renderOverlay(); renderPanel(); toast("Comparison reference set."); }),
  selectInspect: value => { state.inspectKind = value; renderPanel(); },
  addWedgePatch: perform(form => {
    const nominal = +formValues(form).nominal;
    state.wedge.push({label: `Step ${state.wedge.length + 1}`, nominal, box: needBox()});
    renderPanel();
    toast("Wedge patch added.");
  }),
  clearWedge: () => { state.wedge = []; renderPanel(); },
  setPeer,
  setPeerCalibration: form => { state.peerCalibration = localCal(form); toast("Repeat controls selected."); },
  setMethod: value => { state.alignMethod = value; renderPanel(); },
  addLandmark: perform(addLandmark),
  clearLandmarks: () => { state.landmarks = []; renderPanel(); },
  clearCut: () => { state.cutPlan = null; renderOverlay(); renderPanel(); },
  useCut: line => { state.cutPlan = line; renderOverlay(); renderPanel(); toast("Planned cut set. Draw the observed cut, then verify."); },
  markFrame: perform((box, index) => openMark(null, box, `Frame ${index + 1}`)),
  pickFinding: finding => {
    state.pickedDust = finding.number;
    state.box = finding.box;
    state.line = state.point = null;
    renderOverlay();
    toast(`Dust finding ${finding.number} selected.`);
  },
  toggleRestore: (number, checked) => { if (checked) state.restore.add(number); else state.restore.delete(number); },
  restoreChoices: perform(restoreChoices), checkDust: perform(checkDust),
  openRecord: perform(openRecord), saveJob: perform(saveJob), recordDecision, restoreAlignment: perform(restoreAlignment),
  verifyPacket: perform(verifyPacket), openActivity: perform(openActivity),
};
 
function attach() {
  $("scan-overlay").addEventListener("pointerdown", pointerBegin);
  $("scan-overlay").addEventListener("pointermove", pointerMove);
  $("scan-overlay").addEventListener("pointerup", pointerEnd);
  $("scan-overlay").addEventListener("pointercancel", () => { state.drag = null; });
  $("clear-selection").addEventListener("click", () => clearGeometry());
  $("view-modes").addEventListener("click", event => {
    const view = event.target.dataset.view;
    if (!view || view === state.mode || !state.scan) return;
    state.mode = view;
    renderHeader();
    updatePreview();
  });
  $("geometry-tools").addEventListener("click", event => {
    const tool = event.target.dataset.tool;
    if (!tool || !state.scan) return;
    state.tool = tool;
    if (tool !== "point") state.point = null;
    if (tool !== "line") state.line = null;
    setActive("geometry-tools", "tool", tool);
    renderOverlay();
  });
  $("inspector-tabs").addEventListener("click", event => {
    if (!event.target.dataset.tab) return;
    state.tab = event.target.dataset.tab;
    renderPanel();
  });
  $("library-open").addEventListener("click", perform(() => library()));
  $("import-open").addEventListener("click", perform(importDialog));
  $("roll-menu").addEventListener("click", () => rollEditor(state.roll));
  $("scan-menu").addEventListener("click", perform(scanEditor));
  $("mark-open").addEventListener("click", perform(() => openMark()));
  $("dialog-close").addEventListener("click", closeDialog);
  $("dialog").addEventListener("close", () => { dialogSubmit = null; });
  $("dialog-form").addEventListener("submit", async event => {
    event.preventDefault();
    if (!dialogSubmit) return;
    const selected = event.submitter;
    if (selected) selected.disabled = true;
    try {
      $("dialog-error").textContent = "";
      await dialogSubmit(event.currentTarget);
    } catch (error) {
      $("dialog-error").textContent = errorText(error);
    } finally {
      if (selected) selected.disabled = false;
    }
  });
}

async function initialize() {
  attach();
  const [server, entries] = await Promise.all([request("/api/status"), request("/api/rolls")]);
  state.operationTitles = server.operations;
  state.rolls = entries.items;
  const storedRoll = storage.read("reeldesk.roll");
  const storedScan = storage.read("reeldesk.scan");
  const roll = state.rolls.find(item => item.id === storedRoll) || state.rolls[0];
  if (roll) await loadRoll(roll.id, storedScan);
  else {
    renderSidebar();
    renderPanel();
    renderHeader();
  }
}
perform(initialize)();
