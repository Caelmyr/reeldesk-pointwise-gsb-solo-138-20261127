import {button, checkbox, describeDate, h, input, link, message, numberField as num, properties, section, select, table} from "./ui.js";

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

function archiveDesk(state, actions) {
  const saved = state.records ?? [];
  const adopted = state.adoptedGrade;
  return [
    section("Proof and contact sheet",
      adopted
        ? message(`Adopted strip curve active · ${adopted.recognized}/${adopted.total_pairs} reference pairs held. Make the proof to re-check them at the finished size.`)
        : message("Tip: build and adopt a Unified strip grade in the Strip tab to re-check reference pairs on the sized proof."),
      adopted ? null : button("Open Unified strip grade", actions.openStrip, "quiet"),
      form(element => actions.run("proof", element), "Make scan proof", [
        num("max_side", "Finished image long side (px)", state.proofSize || 1800, 480, 3000, 20),
        checkbox("annotate", "Show saved marks", true),
        checkbox("include_histogram", "Include density distribution", false),
        adopted ? checkbox("grade_audit", "Re-check adopted reference pairs at this size", true) : null,
        adopted ? button("Clear adopted curve", actions.clearAdoptedGrade, "text-button") : null,
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
        input("packet", "ReelDesk review packet", "", {type: "file", accept: ".zip,application/zip", required: true})
      ], "quiet"),
      props("stack", button("Roll activity", actions.openActivity, "text-button"),
        link("Original file", `/api/scans/${state.scan.id}/original`, {download: true}))),
  ];
}

function gradePairBuilder(state, actions) {
  const boxMarks = state.marks.filter(mark => mark.geometry && Number.isFinite(mark.geometry.x));
  const options = [
    ...(state.box ? [["drawn", "Current drawn region"]] : []),
    ...boxMarks.map(mark => [mark.id, `${mark.kind} · ${mark.label}`]),
  ];
  const keepMarks = boxMarks.filter(mark => mark.kind === "keep");
  return section("Unified strip grade",
    message("Name dark and bright Reference boxes for the same strip. One monotone RGB curve is fitted across every frame and checked by transformed RGB-luma medians."),
    state.gradePairs.length
      ? table(["Pair", "Dark box", "Bright box", "Min Δ", ""], state.gradePairs.map((pair, index) => [
          pair.label,
          `${pair.dark.x},${pair.dark.y} · ${pair.dark.w}×${pair.dark.h}`,
          `${pair.bright.x},${pair.bright.y} · ${pair.bright.w}×${pair.bright.h}`,
          `${Math.round(pair.minimum * 255)}`,
          button("Remove", () => actions.removeGradePair(index), "text-button"),
        ]))
      : message("No pairs yet. Add Reference marks or draw the two boxes."),
    form(formElement => actions.addGradePair(formElement), "Add reference pair", [
      input("label", "Pair label", `Texture ${state.gradePairs.length + 1}`, {maxlength: 60}),
      select("dark_ref", "Dark texture box", options, state.box ? "drawn" : (boxMarks[0]?.id ?? ""), {required: true}),
      select("bright_ref", "Bright texture box", options, state.box ? "" : (boxMarks[1]?.id ?? boxMarks[0]?.id ?? ""), {required: true}),
      num("minimum", "Required grey gap (0–255)", 28, 2, 200, 1),
    ], "quiet"),
    form(element => actions.run("strip-grade", element), "Build unified curve", [
      props("columns-2",
        num("paper_black", "Paper black (0–255)", state.gradePaper.black8, 0, 120, 1),
        num("paper_white", "Paper white (0–255)", state.gradePaper.white8, 120, 255, 1)),
      props("columns-2",
        num("proof_size", "Proof size, long side (px)", 1200, 240, 3000, 10),
        select("keep_ref", "Protected texture", [
          ["auto", "Saved keep marks (auto)"],
          ...keepMarks.map(mark => [mark.id, mark.label]),
          ["none", "No protected texture"],
        ], "auto")),
      message("When the paper range cannot fund every gap, the curve keeps the largest possible number of pairs and favours the protected texture."),
    ]),
    state.adoptedGrade
      ? message(`Adopted curve · ${state.adoptedGrade.recognized}/${state.adoptedGrade.total_pairs} pairs held · use Archive ▸ Make scan proof.`)
      : null,
  );
}

export function panelNodes(state, actions) {
  if (!state.scan) return [message("Select or import a scan to work with film." )];
  if (state.tab === "repeat") return repeatPass(state, actions);
  if (state.tab === "strip") return [gradePairBuilder(state, actions), ...stripDesk(state, actions)];
  if (state.tab === "archive") return archiveDesk(state, actions);
  return [standardReading(state, actions), calibrationOptions(state, actions), imagingStudies(state, actions)];
}
