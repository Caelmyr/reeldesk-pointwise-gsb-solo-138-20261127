import {asPercent, button, describeDate, h, link, metrics, prettyNumber as p, table} from "./ui.js";

const channels = ["R", "G", "B"];

const presentation = {
  density: x => [["Valid RGB pixels", x.valid_rgb], ["Area", `${x.area} px`], ["Median green", p(x.channels.G.median, 4, " D")]],
  point: x => [["Source point", `${x.point.x}, ${x.point.y}`], ["Median green", p(x.channels.G.neighborhood_median, 4, " D")],
    ["Input depth", `${x.bit_depth} bit`]],
  profile: x => [["Usable stations", `${x.usable_stations} / ${x.samples}`], ["Profile length", p(x.distance_px.at(-1), 2, " px")]],
  balance: x => [["Result", x.status], ...channels.map(key => [`${key} density difference`, p(x.delta[key], 4, " D")])],
  calibration: x => [["Backlight flat", x.flat_name], ["P05–P95 drift", p(x.p05_p95_drift, 4, " D")], ["Tiles", x.tiles.length]],
  registration: x => [["Overlap coverage", asPercent(x.coverage)], ["Model", x.model],
    [x.score === undefined ? "Fit error" : "Patch score", p(x.score ?? x.rms_px, 4, x.rms_px === undefined ? "" : " px")],
    ["Translation", `${p(x.transform[2], 2)}, ${p(x.transform[5], 2)} px`]],
  compare: x => [["Mapped point", `${p(x.peer_point.x, 2)}, ${p(x.peer_point.y, 2)}`], ["Paired G samples", x.channels.G.joint],
    ["Green difference", p(x.channels.G.median_delta, 4, " D")]],
  residual: x => [["Full RGB overlap", asPercent(x.full_rgb_fraction)], ["Sampled pixels", x.samples], ["Green P95 absolute", p(x.channels.G.p95_absolute, 4, " D")]],
  dust: x => [["Candidates", x.findings.length], ["Usable region", asPercent(x.usable_fraction)], ["Omitted groups", x.cutoff_count]],
  "dust-check": x => [["Repeat check", x.verdict], ["Local change", p(x.local_difference, 4, " D")], ["Ring witnesses", x.ring]],
  restore: x => [["Restored pixels", x.restored_pixels], ["Held back", x.unfilled_pixels], ["Findings", x.restored_findings.join(", ")]],
  grain: x => [["Usable region", asPercent(x.coverage)], ["Measured scales", x.bands.length]],
  cut: x => [["Status", x.status], ["Closest mark", p(x.min_clearance_mm, 3, " mm")], ["Suggested positions", x.candidates.length]],
  "cut-verified": x => [["Status", x.status], ["Largest offset", p(x.offset_mm, 3, " mm")], ["Allowed offset", p(x.allowed_offset_mm, 3, " mm")]],
  frames: x => [["Frame boxes", x.frames.length], ["Possible gaps", x.gaps.length]],
  perforations: x => [["Median pitch", p(x.pitch_px, 3, " px")], ["Estimated DPI", p(x.dpi_estimate, 1)]],
  proof: x => [["Proof size", `${x.width} × ${x.height}`], ["Marks drawn", x.mark_count], ["View", x.type]],
  contact: x => [["Scans", x.scans.length], ["Pages", x.pages], ["Paper", x.paper.toUpperCase()]],
  coverage: x => [["Eligible", asPercent(x.eligible_fraction)], ["Status", x.status], ["Reported gaps", x.gaps.length]],
  focus: x => [["Status", x.status], ["MTF50", p(x.mtf50_cyc_px, 5, " cycles/px")], ["Edge width", p(x.edge_width_px, 3, " px")]],
  wedge: x => [["Steps in tolerance", x.within_tolerance_steps], ["Review", x.review_steps], ["Tolerance", p(x.tolerance, 3, " D")]],
  "density-master": x => [["Density range", `${p(x.minimum)}–${p(x.maximum)} D`],
    ["Green encoded samples", x.channels.G.samples], ["Outside range", x.outside_range]],
  masters: x => [["Native crops", x.files.length], ["Format", x.format.toUpperCase()]],
  carrier: x => [["Carrier status", x.status], ["Width", p(x.width_mm, 4, " mm")], ["Rail support", asPercent(x.support_fraction)]],
  "strip-grade": x => [
    ["Pairs held", `${x.recognized} / ${x.total_pairs}`],
    ["Discarded", x.discarded.length ? x.discarded.join(", ") : "none"],
    ["Paper range", `${x.paper.black_8bit}–${x.paper.white_8bit}`],
    ["Proof losses", x.proof_check ? x.proof_check.lost.length : "—"],
    ["Protected texture", x.protected ? (x.protected.preserved ? "retained" : "thin") : "—"],
  ],
  "proof-audit": x => [["Proof losses", x.grade_audit?.lost?.length ?? 0],
    ["Proof size", x.grade_audit ? `${x.grade_audit.proof_width}×${x.grade_audit.proof_height}` : "—"]],
};

function graph(data, colors, seriesNames, width = 560, height = 160) {
  const canvas = h("canvas", {width, height, class: "result-chart", "aria-label": "Density chart"});
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#191a16";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#4d4a3e";
  ctx.lineWidth = 1;
  const plot = {x: 34, y: 12, w: width - 47, h: height - 34};
  const all = data.flat().filter(x => x !== null && x !== undefined);
  if (!all.length) return canvas;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of all) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }
  if (minimum === maximum) { minimum -= 0.5; maximum += 0.5; }
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#ab9c87";
  ctx.fillText(p(maximum, 3), 5, 14);
  ctx.fillText(p(minimum, 3), 5, height - 22);
  for (let row = 0; row <= 4; row++) {
    const y = plot.y + plot.h * row / 4;
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }
  data.forEach((series, index) => {
    ctx.strokeStyle = colors[index];
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    let hasPrevious = false;
    series.forEach((value, step) => {
      if (value === null || value === undefined) { hasPrevious = false; return; }
      const x = plot.x + plot.w * step / Math.max(1, series.length - 1);
      const y = plot.y + plot.h * (maximum - value) / (maximum - minimum);
      if (hasPrevious) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      hasPrevious = true;
    });
    ctx.stroke();
    ctx.fillStyle = colors[index];
    ctx.fillText(seriesNames[index], plot.x + 22 * index, height - 7);
  });
  return canvas;
}

function extendedTable(kind, data, state, actions, interactive) {
  if (kind === "density") return [
    table(["Channel", "Median D", "P05–P95", "Eligible", "Clipped low / high"],
      channels.map(channel => {
        const value = data.channels[channel];
        return [channel, p(value.median), `${p(value.p05)} – ${p(value.p95)}`, asPercent(value.valid_fraction), `${value.clips_low} / ${value.clips_high}`];
      })),
    graph(channels.map(key => data.histogram.channels[key].counts.map(Math.sqrt)), ["#d7776e", "#a8c591", "#85aeda"], channels),
  ];
  if (kind === "point") return table(["Channel", "Raw", "Transmitted", "Density D", "Usable neighbors"],
    channels.map(channel => {
      const value = data.channels[channel];
      return [channel, p(value.raw), p(value.transmission), p(value.density), value.sample_support];
    }));
  if (kind === "profile") return graph(channels.map(key => data.channels[key].density), ["#d7776e", "#a8c591", "#85aeda"], channels);
  if (kind === "compare") {
    return table(["Channel", "Primary D", "Repeat D", "Difference D", "Paired"],
      channels.map(channel => {
        const value = data.channels[channel];
        return [channel, p(value.source), p(value.peer), p(value.median_delta), value.joint];
      }));
  }
  if (kind === "calibration") return table(["Row", "Column", "Drift D", "Light change"], data.tiles.map(tile =>
    [tile.row + 1, tile.column + 1, p(tile.density_drift), p(tile.flat_relative_percent, 2, "%")]));
  if (kind === "dust") return [
    table(["Finding", "Type", "Pixels", "Severity D", "Opt in"], data.findings.map(item => [
      interactive ? button(`#${item.number}`, () => actions.pickFinding(item), "text-button") : item.number,
      item.polarity, item.pixels, p(item.severity), item.can_restore
        ? (interactive ? h("input", {type: "checkbox", "aria-label": `Restore finding ${item.number}`,
          onchange: e => actions.toggleRestore(item.number, e.target.checked)}) : "Allowed")
        : "Protected",
    ])),
    interactive && data.findings.some(x => x.can_restore) ? button("Preview chosen retouch", actions.restoreChoices, "quiet") : null,
  ];
  if (kind === "cut" && data.candidates.length) return table(["Column", "From guide edge", "Clearance", "Choose"],
    data.candidates.map(item => [item.column, p(item.distance_from_guide_left_mm, 3, " mm"), p(item.clearance_mm, 3, " mm"),
      interactive ? button("Set planned cut", () => actions.useCut(item.line), "text-button") : "—"]));
  if (kind === "cut" && data.line) return [
    table(["Mark", "Kind", "Side", "Clearance"], [...data.conflicts, ...data.clear_marks].map(item =>
      [item.label, item.kind, item.side, p(item.clearance_mm, 3, " mm")])),
    interactive && data.status === "clear" ? button("Set planned cut", () => actions.useCut(data.line), "quiet") : null,
  ];
  if (kind === "frames") return table(["Frame", "Box", "Apply"], data.frames.map((item, index) =>
    [index + 1, `${item.box.x}, ${item.box.y} · ${item.box.w} × ${item.box.h}`,
      interactive ? button("Mark frame", () => actions.markFrame(item.box, index), "text-button") : "—"]));
  if (kind === "coverage") return table(["Mark", "Valid", "Review"], data.marks.map(item =>
    [item.name, asPercent(item.eligible_fraction), item.status]));
  if (kind === "wedge") return table(["Patch", "Certified D", "R measured", "G measured", "B measured"],
    data.steps.map(item => [item.label, p(item.nominal), ...channels.map(key => p(item.measurements[key].median))]));
  if (kind === "density-master") return table(["Channel", "Encoded", "Unavailable", "Below", "Above"],
    channels.map(channel => {
      const value = data.channels[channel];
      return [channel, value.samples, value.unavailable, value.below, value.above];
    }));
  if (kind === "masters") return table(["File", "Mode", "Size"],
    data.files.map(item => [item.file, item.output_mode, `${item.export_region.w} × ${item.export_region.h}`]));
  if (kind === "grain") return table(["Radius", "Samples", "MAD", "P95"], data.bands.map(item =>
    [item.radius, item.samples, p(item.mad), p(item.p95)]));
  if (kind === "strip-grade") return stripGradeView(data, actions, interactive);
  if (kind === "proof" && data.grade_audit) return proofGradeAudit(data.grade_audit);
  if (kind === "proof-audit") return proofGradeAudit(data.grade_audit);
  return null;
}

function gradeCurveChart(curve) {
  const knots = curve.knots || curve.points || [];
  const width = 560;
  const height = 170;
  const canvas = h("canvas", {width, height, class: "result-chart", "aria-label": "Shared monotone tone curve"});
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#191a16";
  ctx.fillRect(0, 0, width, height);
  if (knots.length < 2) return canvas;
  const xs = knots.map(point => point[0]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const spanX = Math.max(1e-6, maxX - minX);
  const plot = {x: 36, y: 12, w: width - 48, h: height - 34};
  ctx.strokeStyle = "#4d4a3e";
  for (let row = 0; row <= 4; row++) {
    const y = plot.y + plot.h * row / 4;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
  }
  const mapX = value => plot.x + (value - minX) / spanX * plot.w;
  const mapY = value => plot.y + plot.h * (1 - Math.max(0, Math.min(1, value)));
  ctx.strokeStyle = "#e6cf9b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  knots.forEach(([x, y], index) => {
    if (index) ctx.lineTo(mapX(x), mapY(y)); else ctx.moveTo(mapX(x), mapY(y));
  });
  ctx.stroke();
  ctx.fillStyle = "#9fc3fc";
  knots.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(mapX(x), mapY(y), 2.6, 0, Math.PI * 2); ctx.fill(); });
  ctx.fillStyle = "#ab9c87";
  ctx.font = "10px sans-serif";
  ctx.fillText(`paper ${curve.paper_black ?? ""}–${curve.paper_white ?? ""}`, plot.x, height - 8);
  ctx.fillText("shared R=G=B curve", plot.x + plot.w - 120, 14);
  return canvas;
}

function stripGradeView(data, actions, interactive) {
  const pairRows = data.pairs.map(item => [
    item.label,
    item.dark_median === undefined ? "—" : Math.round(item.dark_median * 255),
    item.bright_median === undefined ? "—" : Math.round(item.bright_median * 255),
    Math.round(item.contrast * 255),
    item.minimum_8bit,
    item.kept ? "kept" : `discarded · ${(item.reason || "conflict").replaceAll("_", " ")}`,
  ]);
  const nodes = [
    gradeCurveChart(data.curve),
    table(["Pair", "Dark 8b", "Bright 8b", "Δ 8b", "Min 8b", "Verdict"], pairRows),
  ];
  if (data.protected) {
    const keep = data.protected.boxes?.[0] || data.protected;
    nodes.push(h("p", {class: "muted result-details"},
      `Protected texture output spread ${keep.output_spread ?? "—"} (${keep.retained_fraction !== undefined ? Math.round(keep.retained_fraction * 100) + "% retained" : "tie-break used"}).`));
  }
  if (data.proof_check) nodes.push(proofGradeAudit(data.proof_check));
  if (interactive) {
    nodes.push(h("div", {class: "stack"},
      button("Adopt curve for Archive proof", () => actions.adoptGrade(data), "primary"),
      button("Open Archive · Make scan proof", () => actions.openArchive(), "quiet")));
  }
  return nodes;
}

function proofGradeAudit(audit) {
  if (!audit?.pairs?.length) return null;
  const rows = audit.pairs.map(item => [
    item.label,
    `${audit.proof_width}×${audit.proof_height}`,
    Math.round(item.contrast * 255),
    item.minimum_8bit,
    item.separable_pixels,
    item.lost_after_resize ? "lost after resize" : item.contrast_ok ? "holds" : "below target",
  ]);
  return [
    audit.lost.length
      ? h("p", {class: "error"}, `Originally sufficient but lost on the sized proof: ${audit.lost_labels.join(", ")}`)
      : h("p", {class: "muted"}, "All adopted pairs remain separable on the sized proof."),
    table(["Pair", "Proof px", "Δ 8b", "Min 8b", "Witness px", "Proof verdict"], rows),
  ];
}

function filesFor(entry, saved) {
  return entry.files.map(file => {
    const url = saved ? `/api/records/${entry.id}/files/${file.id}` : `/api/jobs/${entry.id}/files/${file.index}`;
    return {name: file.name, type: saved ? file.content_type : file.type, url};
  });
}

export function resultNodes(entry, state, actions, saved) {
  const caption = state.operationTitles[entry.kind] || entry.kind;
  const info = filesFor(entry, saved);
  const stats = presentation[entry.kind]?.(entry.result) || [];
  const report = entry.result;
  const headerActions = saved
    ? [button("Review decision", () => actions.recordDecision(entry)), link("Review ZIP", `/api/records/${entry.id}/packet`, {download: true})]
    : [button("Save review", () => actions.saveJob(entry), "primary")];
  return [
    h("header", {class: "result-heading"},
      h("div", {}, h("p", {class: "eyebrow"}, saved ? `SAVED REVIEW · ${entry.decision.toUpperCase()}` : "WORKING RESULT"),
        h("h3", {}, saved ? entry.title : caption)),
      h("div", {class: "stack"}, headerActions)),
    stats.length ? metrics(stats) : null,
    saved ? h("p", {class: "muted result-details"}, describeDate(entry.created)) : null,
    info.filter(file => file.type.startsWith("image/")).map(file => h("img", {src: file.url, alt: file.name})),
    extendedTable(entry.kind, report, state, actions, !saved),
    h("div", {class: "stack"},
      info.map(file => link(file.name, `${file.url}?download=1`, {download: true})),
      saved ? link("ZIP with originals", `/api/records/${entry.id}/packet?originals=1`, {download: true}) : null,
      saved && ["profile", "calibration", "density", "dust", "residual", "compare", "cut", "cut-verified", "grain",
        "coverage", "wedge", "carrier"].includes(entry.kind)
        ? link("Reading CSV", `/api/records/${entry.id}/csv`, {download: true}) : null,
      saved && entry.kind === "registration" && entry.scan_id === state.scan.id
        ? button("Use this alignment", () => actions.restoreAlignment(entry)) : null),
    h("details", {class: "result-details"}, h("summary", {}, "Reading details"),
      h("pre", {}, JSON.stringify(report, null, 2))),
  ];
}

export function showPacket(data) {
  return [
    h("header", {class: "result-heading"},
      h("div", {}, h("p", {class: "eyebrow"}, "PACKET CHECK"),
        h("h3", {}, data.status === "verified" ? "Checksums match" : "Review required"))),
    metrics([["Record", data.record.title], ["Evidence files", data.files.length], ["Source scans", (data.sources.length + (data.support_sources?.length ?? 0))]]),
    table(["Original", "In packet", "In catalog"], [...data.sources, ...(data.support_sources || [])].map(row => [row.name, row.packet, row.local])),
    table(["Evidence", "Checksum"], data.files.map(row => [row.name, row.status])),
    data.unexpected_files.length ? h("p", {class: "error"}, `Extra files: ${data.unexpected_files.join(", ")}`) : null,
  ];
}
