import {button, checkbox, describeDate, h, input, link, message, numberField as num, properties, section, select} from "./ui.js";

const channels = ["R", "G", "B"];
const props = properties;
const formOptions = (entries) => [["", "None"], ...entries.map(item => [item.id, item.name])];

function form(action, label, children, submit = "primary") {
  return h("form", {onsubmit: event => { event.preventDefault(); action(event.currentTarget); }},
    children, h("button", {type: "submit", class: submit}, label));
}

function calibrationOptions(state, actions) {
  const calibration = state.calibration;
  const flats = state.scans.filter(scan => scan.role === "flat");
  const darks = state.scans.filter(scan => scan.role === "dark");
  return section("Transmission controls",
    form(actions.setCalibration, "Apply to preview", [
      select("flat_id", "Backlight flat", formOptions(flats), calibration.flat_id),
      select("dark_id", "Shutter dark", formOptions(darks), calibration.dark_id),
      props("columns-3", channels.map((channel, i) => num(`base${i}`, `Base ${channel}`, calibration.base[i], .015, 1.5, .001))),
      checkbox("reject_clipped", "Exclude clipped samples", calibration.reject_clipped),
    ]),
    h("details", {}, h("summary", {}, "Positive tone"),
      form(actions.setTone, "Update tone", [
        props("columns-2", num("shadow", "Shadows D", state.tone.shadow, -.4, 3, .01),
          num("highlight", "Highlights D", state.tone.highlight, -.35, 3.6, .01)),
        props("columns-2", num("gamma", "Gamma", state.tone.gamma, .35, 3, .05),
          num("brightness", "Brightness", state.tone.brightness, -.4, .4, .01)),
        checkbox("neutralize", "Neutralize midtones", state.tone.neutralize),
      ], "quiet")));
}

function standardReading(state, actions) {
  return section("Density reading",
    message("Draw a region or choose a frame. Density is measured in the original scan."),
    props("stack",
      button("Measure region", () => actions.run("density"), "primary"),
      button("Set base from region", actions.setBase),
      button("Read source point", () => actions.run("point")),
      button("Read line", () => actions.run("profile")),
      button("Keep as reference", actions.keepReference),
      button("Compare regions", () => actions.run("balance"), "quiet", {disabled: !state.reference})),
  );
}

function imagingStudies(state, actions) {
  const options = [
    ["coverage", "Scanner coverage"],
    ["calibration", "Illumination map"],
    ["dust", "Dust survey"],
    ["grain", "Grain scales"],
    ["focus", "Slanted-edge focus"],
    ["wedge", "Transmission wedge"],
  ];
  const kind = state.inspectKind;
  let fields;
  if (kind === "calibration") {
    fields = props("columns-2", num("rows", "Rows", 4, 2, 12), num("columns", "Columns", 8, 2, 18));
  } else if (kind === "dust") {
    fields = props("columns-2", num("radius", "Grain radius (px)", 5, 2, 24), num("threshold", "Density difference", .12, .015, 1.2, .001));
  } else if (kind === "grain") {
    fields = props("columns-3", [1, 2, 4].map((n, i) => num(`scale${i}`, `Scale ${i + 1}`, n, 1, 10)));
  } else if (kind === "focus") {
    fields = select("orientation", "Edge direction", [["auto", "Automatic"], ["vertical", "Vertical"], ["horizontal", "Horizontal"]]);
  } else if (kind === "wedge") {
    fields = props("wedge-actions",
      message("Draw each wedge patch and enter its certified density."),
      props("columns-2", num("nominal", "Certified density", .5, -.15, 4.5, .01),
        num("tolerance", "Allowed error D", .08, .015, .8, .001)),
      button("Add selected patch", event => actions.addWedgePatch(event.currentTarget.form)),
      button(`Clear ${state.wedge.length} patch${state.wedge.length === 1 ? "" : "es"}`, actions.clearWedge, "text-button",
        {disabled: state.wedge.length === 0}),
      checkbox("compensate", "Show fitted compensation", false));
  } else {
    fields = select("channel", "Check channels", [["rgb", "All RGB"], ["red", "Red"], ["green", "Green"], ["blue", "Blue"]], "rgb");
  }
  return section("Scanner and film checks",
    form(formElement => actions.run(kind, formElement), "Run check", [
      select("operation", "Check", options, kind, {onchange: e => actions.selectInspect(e.target.value)}),
      fields,
    ]));
}

function repeatPass(state, actions) {
  const peers = state.scans.filter(scan => scan.role === "scan" && scan.id !== state.scan?.id);
  if (!peers.length) return section("Repeat-pass alignment", message("Import another scan of the same film."));
  const options = [
    ["patch", "Patch translation"],
    ["landmarks", "Landmark pairs"],
  ];
  const flat = state.scans.filter(it => it.role === "flat");
  const dark = state.scans.filter(it => it.role === "dark");
  const evidence = state.alignment
    ? message(`Alignment ready · ${state.alignment.method === "landmarks" ? "landmarks" : "patch"}`)
    : message("Align the repeat scan to use point and difference checks.");
  return [
    section("Repeat-pass alignment",
      select("peer_id", "Repeat scan", peers.map(peer => [peer.id, peer.name]), state.peerId, {onchange: e => actions.setPeer(e.target.value)}),
      form(el => actions.run("registration", el), "Align repeat scan", [
        select("method", "Alignment method", options, state.alignMethod, {onchange: e => actions.setMethod(e.target.value)}),
        state.alignMethod === "patch"
          ? [message("Draw a distinctive patch in the main scan."),
            num("maximum", "Search radius (px)", 28, 1, 160)]
          : [
            select("model", "Geometry", [["shift", "Shift"], ["similarity", "Rotate and scale"], ["affine", "Affine"]], "shift"),
            props("stack", button("Add landmark pair", actions.addLandmark),
              button(`Clear ${state.landmarks.length}`, actions.clearLandmarks, "text-button", {disabled: !state.landmarks.length})),
            checkbox("robust", "Reduce outlier weight", true),
          ],
        select("view", "Alignment preview", [["overlay", "Overlay"], ["checker", "Checkerboard"], ["difference", "Difference"], ["peer", "Aligned peer"]], "overlay"),
      ]),
    ),
    section("Repeat-scan controls",
      form(actions.setPeerCalibration, "Use repeat controls", [
        select("flat_id", "Repeat backlight", formOptions(flat), state.peerCalibration.flat_id),
        select("dark_id", "Repeat dark", formOptions(dark), state.peerCalibration.dark_id),
        props("columns-3", channels.map((channel, i) => num(`base${i}`, `Base ${channel}`, state.peerCalibration.base[i], .015, 1.5, .001))),
        checkbox("reject_clipped", "Exclude clipped samples", state.peerCalibration.reject_clipped),
      ], "quiet"),
    ),
    section("Check the aligned film", evidence,
      props("stack", button("Read both at point", () => actions.run("compare"), "primary", {disabled: !state.alignment}),
        button("Measure difference", () => actions.run("residual"), "quiet", {disabled: !state.alignment})),
      (state.lastDust && state.alignment) ? button("Repeat-check selected dust", actions.checkDust, "text-button") : null)
  ];
}

function stripDesk(state, actions) {
  const checks = [["frames", "Suggest frame boxes"], ["perforations", "Perforation pitch"], ["carrier", "Carrier rails"]];
  return [
    section("Strip measurements",
      form(element => actions.run(element.elements.check.value, element), "Run strip check", [
        select("check", "Measurement", checks, "frames"),
        message("Draw a strip region containing the relevant frames or film edges.")
      ]),
    ),
    section("Cut clearance",
      message("Use the selected region as the guide. With a line selected, inspect its clearance."),
      form(element => actions.run("cut", element), "Check cut positions", [
        props("columns-2",
          num("kerf_mm", "Blade width (mm)", state.cutOptions.kerf_mm, .015, 4, .005),
          num("reserve_mm", "Keep-away (mm)", state.cutOptions.reserve_mm, 0, 12, .05)),
        num("dpi", "Scanner DPI", state.cutOptions.dpi || state.scan.dpi || 1200, 30, 20000, .1),
        select("retained", "Keep film on", [["both", "Both sides"], ["left", "Left"], ["right", "Right"]], state.cutOptions.retained),
      ]),
      state.cutPlan ? props("stack",
        button("Use selected as actual", () => actions.run("cut-verified")),
        button("Clear planned line", actions.clearCut, "text-button")) : null,
    ),
    section("Native frame masters",
      form(element => actions.run("masters", element), "Export frame masters", [
        select("format", "Lossless format", [["tiff", "TIFF"], ["png", "PNG"]], "tiff"),
        num("padding", "Frame margin (px)", 0, 0, 250),
        checkbox("useRegion", "Export selection instead of saved frames", false),
      ], "quiet")),
  ];
}

function batchDesk(state, actions) {
  const readings = state.batchReadings ?? [];
  const selected = state.batchSelection;
  const ready = readings.filter(item => item.retestable).length;
  const rows = readings.map(item => h("tr", {},
    h("td", {}, h("input", {type: "checkbox", checked: selected.has(item.id),
      onchange: event => actions.toggleBatchRecord(item.id, event.target.checked), "aria-label": `Include ${item.title}`})),
    h("td", {}, h("span", {title: item.title}, item.title), h("small", {}, ` · ${item.scan_name ?? "scan"}`)),
    h("td", {}, h("input", {type: "checkbox", checked: state.batchPriority.has(item.id),
      onchange: event => actions.toggleBatchPriority(item.id, event.target.checked), "aria-label": `Protect ${item.title}`})),
    h("td", {}, item.retestable ? "ready" : (item.missing.join("; ") || "incomplete"))));
  return section("Offline re-check package",
    state.batchTruncated ? message("Only the newest 200 saved density readings are listed here.") : null,
    message("Packs saved density readings with the original scan, the flat and dark controls used, and the measurement box. "
      + "Files are stored once by content hash. Every chosen reading stays in the list; only readings with complete evidence can be re-measured."),
    readings.length
      ? [
        props("stack",
          button("Select every ready reading", actions.selectAllBatch, "text-button"),
          button("Clear selection", actions.clearBatch, "text-button")),
        h("div", {class: "small-table"}, h("table", {},
          h("thead", {}, h("tr", {}, ["Pack", "Saved reading", "Protect", "Evidence"].map(label => h("th", {scope: "col"}, label)))),
          h("tbody", {}, rows))),
        form(actions.planBatchPackage, "Plan package", [
          num("max_bytes", "Byte ceiling", state.batchLimit, 8192, 90 * 1024 * 1024, 1024),
          num("tolerance", "Median density tolerance D", state.batchTolerance, .001, 1, .001),
          message(`${selected.size} chosen · ${ready} ready to re-measure`),
        ]),
      ]
      : message("Save a region-density reading to build a package."));
}

function archiveDesk(state, actions) {
  const saved = state.records ?? [];
  return [
    batchDesk(state, actions),
    section("Proof and contact sheet",
      form(element => actions.run("proof", element), "Make scan proof", [
        checkbox("annotate", "Show saved marks", true),
        checkbox("include_histogram", "Include density distribution", false),
      ]),
      h("details", {}, h("summary", {}, "Roll contact sheet"),
        form(element => actions.run("contact", element), "Make contact sheet", [
          props("columns-2", select("paper", "Paper", [["a4", "A4"], ["letter", "Letter"]], "a4"),
            num("columns", "Columns", 3, 2, 5)),
        ], "quiet"))),
    section("16-bit density master",
      form(element => actions.run("density-master", element), "Export density master", [
        props("columns-2", num("minimum", "Lowest density D", -.12, -1.5, 2.3, .01),
          num("maximum", "Highest density D", 2.35, -.1, 3.9, .01)),
        checkbox("clip_outside", "Clamp outside-range samples", false),
      ], "quiet")),
    section("Saved reviews",
      saved.length ? saved.map(item =>
        button(h("span", {}, item.title, h("small", {}, `${item.decision} · ${describeDate(item.created)}`)),
          () => actions.openRecord(item.id), "list-button"))
        : message("Save a working result to keep its settings and evidence here.")),
    section("Review packet",
      form(actions.verifyPacket, "Verify ZIP", [
        input("packet", "ReelDesk review or re-check ZIP", "", {type: "file", accept: ".zip,application/zip", required: true})
      ], "quiet"),
      props("stack", button("Roll activity", actions.openActivity, "text-button"),
        link("Original file", `/api/scans/${state.scan.id}/original`, {download: true}))),
  ];
}

export function panelNodes(state, actions) {
  if (!state.scan) return [message("Select or import a scan to work with film." )];
  if (state.tab === "repeat") return repeatPass(state, actions);
  if (state.tab === "strip") return stripDesk(state, actions);
  if (state.tab === "archive") return archiveDesk(state, actions);
  return [standardReading(state, actions), calibrationOptions(state, actions), imagingStudies(state, actions)];
}
