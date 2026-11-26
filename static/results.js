import {asPercent, button, describeDate, h, input, link, metrics, prettyNumber as p, table} from "./ui.js";

const channels = ["R", "G", "B"];
const edgeStatusChinese = {
  pass: "通过",
  "dark-jump": "新变暗跳变",
  "bright-jump": "新变亮跳变",
  insufficient: "证据不足 · 待复核",
};
const manualReasonChinese = {
  "no-clean-source-pixels": "无干净实测颗粒源",
  "outer-ring-density-limit": "外圈密度差超限",
  "donor-pixel-conflict": "源像素已被另一尘块借用",
  "selected-patch-overlaps-keep": "修补口占 Keep",
};
const verdictChinese = {
  "repair-edge-pass": "修口边检通过",
  "edge-jump-exceeds-tolerance": "修口出现超公差新亮/暗跳变",
  "edge-evidence-review": "部分边口证据不足，待复核",
  "manual-review": "该块未实际修补，需手工补",
};

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
  "dust-check": x => [
    ["Repeat check", x.verdict],
    ["Edge status", x.edge_status ?? "legacy"],
    ["Local change", p(x.local_difference, 4, " D")],
    ["New edge jump", p(x.edges?.length ? Math.max(...x.edges.map(e => Math.abs(e.new_jump ?? 0))) : null, 4, " D")],
    ["Ring witnesses", x.ring],
  ],
  restore: x => [
    ["Measured-grain repairs", `${x.summary?.restored_findings ?? x.restored_findings.length}/${x.summary?.selected_findings ?? "—"}`],
    ["Manual blocks", (x.manual_findings ?? []).length],
    ["Restored pixels", x.restored_pixels],
    ["Held back", x.unfilled_pixels],
    ["Worst seam D", p(x.summary?.worst_seam_density_difference, 4)],
    ["Worst outer-ring D", p(x.summary?.worst_ring_density_difference, 4)],
    ["Findings", (x.restored_findings ?? []).join(", ") || "—"],
  ],
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
  if (kind === "dust") {
    const repairable = data.findings.some(x => x.can_restore);
    return [
      table(["Finding", "Type", "Pixels", "Severity D", "Opt in"], data.findings.map(item => [
        interactive ? button(`#${item.number}`, () => actions.pickFinding(item), "text-button") : item.number,
        item.polarity, item.pixels, p(item.severity), item.can_restore
          ? (interactive ? h("input", {type: "checkbox", "aria-label": `Restore finding ${item.number}`,
            onchange: e => actions.toggleRestore(item.number, e.target.checked)}) : "Allowed")
          : "Protected",
      ])),
      interactive && repairable ? h("div", {class: "donor-limits"},
        input("ring_limit", "Outer-ring D limit", state.ringLimit, {
          type: "number", min: "0", max: "1.5", step: ".005",
          oninput: e => { state.ringLimit = +e.currentTarget.value; },
          onchange: e => { state.ringLimit = +e.currentTarget.value; },
        }),
        input("edge_tolerance", "Repeat edge tolerance D", state.edgeTolerance, {
          type: "number", min: "0", max: "1.5", step: ".005",
          oninput: e => { state.edgeTolerance = +e.currentTarget.value; },
          onchange: e => { state.edgeTolerance = +e.currentTarget.value; },
        }),
        button("Preview chosen retouch", actions.restoreChoices, "quiet")) : null,
    ];
  }
  if (kind === "restore") {
    const assignments = data.plan?.assignments ?? data.assignments ?? [];
    const manual = data.manual_findings ?? [];
    const summary = data.summary ?? {};
    return [
      h("p", {class: "help"}, data.headline ?? "Measured-grain donor plan."),
      h("p", {class: "help pass"},
        `可采用方案：自动补 ${summary.restored_findings ?? assignments.length}/${summary.selected_findings ?? "—"} 块；`
        + `手工 ${manual.length} 块；最坏修口 D=${p(summary.worst_seam_density_difference, 4)}，`
        + `外圈限值 D=${p(summary.ring_limit, 3)}。`),
      assignments.length ? table(
        ["Dust", "Donor", "Source box", "Ring D", "Seam D"],
        assignments.map(item => [
          `#${item.number}`, item.donor_name,
          `${item.source_box.x},${item.source_box.y} ${item.source_box.w}×${item.source_box.h}`,
          p(item.ring_density_difference, 4), p(item.seam_density_difference, 4),
        ])) : h("p", {class: "warning"}, "No block could use a valid matched grain source."),
      manual.length ? table(["Manual #", "Reason", "Pixels"], manual.map(item => [
        `#${item.number}`,
        h("span", {}, item.reason, h("br"), h("small", {class: "muted"}, manualReasonChinese[item.reason] ?? "需手工补")),
        item.pixels,
      ])) : null,
    ];
  }
  if (kind === "dust-check" && data.edges?.length) {
    return [
      h("p", {class: data.passed ? "help pass" : data.review_required ? "help warning" : "warning"},
        `${data.edge_check?.verdict ?? data.verdict} · ${verdictChinese[data.verdict] ?? ""}`),
      table(["Edge", "Pairs", "Repaired jump", "Repeat jump", "New jump", "Status"], data.edges.map(edge => [
        edge.side, `${edge.pairs}/${edge.required_pairs}`, p(edge.repaired_jump, 4),
        p(edge.repeat_jump, 4), p(edge.new_jump, 4),
        h("span", {}, edge.status, h("br"), h("small", {class: "muted"}, edgeStatusChinese[edge.status] ?? "")),
      ])),
    ];
  }
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
  return null;
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
