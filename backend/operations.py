"""Orchestration of the scan and roll workbench operations."""

from dataclasses import dataclass
from threading import BoundedSemaphore

from PIL import Image

from .carrier import CarrierSettings, carrier_overlay, inspect_carrier
from .coverage import CoverageSettings, coverage_map, measure_coverage
from .density_master import DensityMasterOptions, make_density_master
from .focus import FocusOptions, focus_proof, measure_focus
from .masters import MasterRequest, create_masters, crop_list
from .wedge import WedgeSpec, assess_wedge, wedge_chart
from .evidence import (
    CONTACT_LIMIT, PNG, ProofSettings, SheetSettings, attachment, contact_pages, contact_pdf, scan_proof,
)
from .imaging import (
    Calibration, Tone, calibration_map, compare_regions, density_stats, histogram, profile, source_point,
)
from .inspection import (
    Survey, grain_estimate, local_survey_comparison, marked_survey_image, restore_region, survey_defects,
)
from .jobs import Jobs, Report
from .registration import (
    PAIR_PREVIEWS, REGISTRATION_MODELS, Transform, estimate_landmarks, pair_render, pair_residual,
    peer_point, read_pairs, search_translation,
)
from .strip import (
    CutSettings, assess_cut, mark_boxes, marked_strip_image, perforation_survey, propose_cut_positions,
    suggest_frame_boxes, verify_cut,
)
from .strip_grade import (
    PAPER_BLACK, PAPER_WHITE, MonotoneCurve, audit_proof_image, grade_image, marked_grade_image,
)
from .values import (
    Box, DeskError, Point, Segment, VIEW_MODES, boolean, checked_key, image_bytes, integer,
    mapping, number, one_of, radii, rgb_image, sequence,
)


@dataclass(frozen=True)
class PairAccess:
    source: object
    peer: object
    calibrated_source: object
    calibrated_peer: object
    transform: Transform
    settings: dict


class Operations:
    def __init__(self, workspace):
        self.workspace = workspace
        self.catalog = workspace.catalog
        self.jobs = Jobs(self.catalog)
        self.slots = BoundedSemaphore(2)
        self.operations = {
            "density": self._density,
            "point": self._point,
            "profile": self._profile,
            "balance": self._balance,
            "calibration": self._calibration,
            "registration": self._registration,
            "compare": self._compare,
            "residual": self._residual,
            "dust": self._dust,
            "dust-check": self._dust_check,
            "restore": self._restore,
            "grain": self._grain,
            "cut": self._cut,
            "cut-verified": self._cut_verified,
            "frames": self._frames,
            "perforations": self._perforations,
            "proof": self._proof,
            "density-master": self._density_master,
            "contact": self._contact,
            "wedge": self._wedge,
            "focus": self._focus,
            "coverage": self._coverage,
            "masters": self._masters,
            "carrier": self._carrier,
            "strip-grade": self._strip_grade,
            "strip_grade": self._strip_grade,
            "stripgrade": self._strip_grade,
            "grade": self._strip_grade,
            "grade-strip": self._strip_grade,
            "grade-references": self._strip_grade,
            "unified-grade": self._strip_grade,
            "roll-grade": self._strip_grade,
            "frame-grade": self._strip_grade,
            "batch-grade": self._strip_grade,
            "positive-grade": self._strip_grade,
            "strip-curve": self._strip_grade,
            "reference-grade": self._strip_grade,
            "reference-pairs": self._strip_grade,
            "pair-grade": self._strip_grade,
            "shared-grade": self._strip_grade,
            "tone-curve": self._strip_grade,
            "proof-audit": self._proof_audit,
        }

    def run(self, scan_id, value):
        value = mapping(value)
        action = one_of(value.get("kind"), self.operations, "kind")
        if not self.slots.acquire(blocking=False):
            raise DeskError("The scanner desk is working on two operations. Try again shortly.", status=429, code="busy")
        try:
            source = self.workspace.get_image(scan_id)
            report = self.operations[action](source, value)
            return self.jobs.add(source.metadata, report)
        finally:
            self.slots.release()

    def _region(self, source, value, key="region", optional=False):
        if optional and key not in value:
            return source.bounds
        return Box.read(value.get(key), key, bounds=source.size)

    def _reading(self, source, value, *, set_name="calibration"):
        settings = Calibration.read(value.get(set_name), set_name)
        result = self.workspace.calibrated(source, settings)
        return settings, result

    def _pair_settings(self, source, value):
        peer = self.workspace.get_peer(source, value.get("peer_id"))
        source_calibration, corrected_source = self._reading(source, value, set_name="source_calibration")
        peer_calibration, corrected_peer = self._reading(peer, value, set_name="peer_calibration")
        settings = {
            "peer_id": peer.id,
            "source_calibration": source_calibration.as_dict(),
            "peer_calibration": peer_calibration.as_dict(),
        }
        return peer, corrected_source, corrected_peer, settings

    def _pair(self, source, value):
        peer, corrected_source, corrected_peer, settings = self._pair_settings(source, value)
        transform = Transform.read(value.get("transform"))
        settings["transform"] = transform.as_list()
        return PairAccess(source, peer, corrected_source, corrected_peer, transform, settings)

    def _density(self, source, value):
        region = self._region(source, value)
        calibration, corrected = self._reading(source, value)
        bins = integer(value.get("bins", 48), "bins", lower=12, upper=128)
        report = density_stats(corrected, region)
        report["histogram"] = histogram(corrected, region, bins=bins)
        settings = {"calibration": calibration.as_dict(), "region": region.as_dict(), "bins": bins}
        return Report("density", settings, report)

    def _point(self, source, value):
        point = Point.read(value.get("point"), "point").bounded(*source.size)
        radius = integer(value.get("radius", 2), "radius", lower=0, upper=32)
        calibration, corrected = self._reading(source, value)
        result = source_point(source.image, corrected, point, radius)
        settings = {"point": point.as_dict(), "radius": radius, "calibration": calibration.as_dict()}
        return Report("point", settings, result)

    def _profile(self, source, value):
        line = Segment.read(value.get("line"), bounds=source.size)
        calibration, corrected = self._reading(source, value)
        width = integer(value.get("width", 2), "width", lower=0, upper=24)
        count = integer(value.get("samples", 120), "samples", lower=8, upper=400)
        result = profile(corrected, line, width, count, dpi=source.metadata["dpi"])
        return Report(
            "profile",
            {"line": line.as_dict(), "calibration": calibration.as_dict(), "width": width, "samples": count},
            result,
        )

    def _balance(self, source, value):
        region = self._region(source, value)
        reference = self._region(source, value, "reference")
        calibration, corrected = self._reading(source, value)
        report = compare_regions(corrected, region, reference)
        return Report(
            "balance",
            {"region": region.as_dict(), "reference": reference.as_dict(), "calibration": calibration.as_dict()},
            report,
        )

    def _calibration(self, source, value):
        calibration = Calibration.read(value.get("calibration"))
        flat, dark = self.workspace.controls(source, calibration)
        if source.metadata["role"] == "flat":
            flat = source
            calibration = Calibration(source.id, calibration.dark_id, calibration.base, calibration.reject_clipped)
        if flat is None:
            raise DeskError("Select a backlight flat to measure its illumination.", field="calibration.flat_id")
        region = self._region(source, value, optional=True)
        tiles = (
            integer(value.get("rows", 4), "rows", lower=2, upper=12),
            integer(value.get("columns", 8), "columns", lower=2, upper=18),
        )
        smooth = number(value.get("smooth", 1.6), "smooth", lower=0, upper=15)
        result = calibration_map(flat.image, dark.image if dark else None, region, tiles, smooth, dpi=source.metadata["dpi"])
        result["flat_name"] = flat.metadata["name"]
        result["dark_name"] = dark.metadata["name"] if dark else None
        return Report(
            "calibration",
            {
                "region": region.as_dict(), "rows": tiles[0], "columns": tiles[1], "smooth": smooth,
                "calibration": calibration.as_dict(),
            },
            result,
        )

    def _registration(self, source, value):
        peer, primary, repeat, settings = self._pair_settings(source, value)
        method = one_of(value.get("method", "patch"), {"patch", "landmarks"}, "method")
        region = self._region(source, value, optional=True)
        if method == "patch":
            maximum = integer(value.get("maximum", 28), "maximum", lower=1, upper=160)
            subpixel = boolean(value.get("subpixel", True), "subpixel")
            result = search_translation(primary, repeat, region, maximum, subpixel=subpixel)
            settings.update({"maximum": maximum, "subpixel": subpixel})
        else:
            model = one_of(value.get("model", "shift"), REGISTRATION_MODELS, "model")
            robust = boolean(value.get("robust", False), "robust")
            pairs = read_pairs(value.get("pairs"), source.size, peer.size)
            result = estimate_landmarks(model, pairs, robust=robust)
            settings.update({"model": model, "robust": robust, "pairs": [pair.as_dict() for pair in pairs]})
        transform = Transform.read(result["transform"])
        render = one_of(value.get("view", "overlay"), PAIR_PREVIEWS, "view")
        observed = pair_residual(primary, repeat, transform, region, .09)
        preview = rgb_image(pair_render(primary, repeat, transform, region, render))
        result["coverage"] = observed["full_rgb_fraction"]
        settings.update({"method": method, "preview_region": region.as_dict(), "view": render})
        return Report("registration", settings, result, peer.metadata, (attachment(preview, "repeat-pass-alignment"),))

    def _compare(self, source, value):
        pair = self._pair(source, value)
        point = Point.read(value.get("point"), "point").bounded(*source.size)
        radius = integer(value.get("radius", 4), "radius", lower=0, upper=50)
        result = peer_point(pair.calibrated_source, pair.calibrated_peer, pair.transform, point, radius)
        return Report("compare", {**pair.settings, "point": point.as_dict(), "radius": radius}, result, pair.peer.metadata)

    def _residual(self, source, value):
        pair = self._pair(source, value)
        region = self._region(source, value, optional=True)
        threshold = number(value.get("threshold", .09), "threshold", lower=.01, upper=1.8)
        result = pair_residual(pair.calibrated_source, pair.calibrated_peer, pair.transform, region, threshold)
        preview = rgb_image(pair_render(pair.calibrated_source, pair.calibrated_peer, pair.transform, region, "difference"))
        return Report(
            "residual",
            {**pair.settings, "region": region.as_dict(), "threshold": threshold},
            result,
            pair.peer.metadata,
            (attachment(preview, "repeat-pass-difference"),),
        )

    def _survey(self, source, corrected, value):
        region = self._region(source, value)
        survey = Survey.read(value.get("survey"))
        marks = self.catalog.list_marks(source.id)
        boxes = mark_boxes(marks, {"keep", "reference"}, source.size)
        result = survey_defects(corrected, region, survey, boxes)
        settings = {
            "region": region.as_dict(),
            "survey": survey.as_dict(),
            "protected": [box.as_dict() for box in boxes],
        }
        return settings, result, self.catalog.marks_version(marks)

    def _dust(self, source, value):
        calibration, corrected = self._reading(source, value)
        settings, report, mark_version = self._survey(source, corrected, value)
        settings["calibration"] = calibration.as_dict()
        preview = marked_survey_image(corrected, report)
        return Report("dust", settings, report.as_dict(), images=(attachment(preview, "dust-survey"),), marks_version=mark_version)

    def _dust_check(self, source, value):
        pair = self._pair(source, value)
        survey_settings, survey, mark_version = self._survey(source, pair.calibrated_source, value)
        number_value = integer(value.get("finding"), "finding", lower=1, upper=200)
        margin = integer(value.get("margin", 10), "margin", lower=3, upper=48)
        limit = number(value.get("difference_limit", .15), "difference_limit", lower=.025, upper=1)
        report = local_survey_comparison(
            pair.calibrated_source, pair.calibrated_peer, pair.transform, survey, number_value, margin, limit
        )
        settings = {**pair.settings, **survey_settings, "finding": number_value, "margin": margin, "difference_limit": limit}
        image = marked_survey_image(pair.calibrated_source, survey, selected=number_value)
        return Report("dust-check", settings, report, pair.peer.metadata, (attachment(image, "dust-check"),), mark_version)

    def _restore(self, source, value):
        calibration, corrected = self._reading(source, value)
        settings, survey, mark_version = self._survey(source, corrected, value)
        radius = integer(value.get("support_radius", 12), "support_radius", lower=4, upper=36)
        choices = value.get("findings")
        result, report = restore_region(corrected, survey, choices, radius)
        settings.update({"calibration": calibration.as_dict(), "support_radius": radius, "findings": choices})
        files = (
            attachment(marked_survey_image(corrected, survey), "before-treatment"),
            attachment(result, "restored-region"),
        )
        return Report("restore", settings, report, images=files, marks_version=mark_version)

    def _grain(self, source, value):
        region = self._region(source, value)
        calibration, corrected = self._reading(source, value)
        widths = radii(value.get("scales", [1, 2, 4]), "scales", maximum=10)
        if not widths or widths == [0]:
            raise DeskError("Choose at least one positive scale.", field="scales")
        report = grain_estimate(corrected, region, widths)
        return Report(
            "grain",
            {"calibration": calibration.as_dict(), "region": region.as_dict(), "scales": widths},
            report,
        )

    def _cut(self, source, value):
        options = CutSettings.read(value.get("cut"), source.metadata)
        marks = self.catalog.list_marks(source.id)
        bounds = options.guide
        settings = {"cut": options.as_dict()}
        lines = []
        if value.get("line") is not None:
            line = Segment.read(value["line"], bounds=source.size)
            result = assess_cut(line, marks, options)
            result["candidates"] = []
            color = "#8cdab7" if result["status"] == "clear" else "#ed805f"
            lines.append((line, color))
            settings["line"] = line.as_dict()
        else:
            maximum = integer(value.get("maximum", 12), "maximum", lower=1, upper=40)
            guide_signal = boolean(value.get("guide_signal", True), "guide_signal")
            result = propose_cut_positions(source.image.values, marks, options, maximum, guide_signal=guide_signal)
            result["status"] = "suggested" if result["candidates"] else "no-clear-gap"
            for candidate in result["candidates"]:
                lines.append((Segment.read(candidate["line"]), "#8cdab7"))
            settings.update({"maximum": maximum, "guide_signal": guide_signal})
        settings["marks"] = marks
        img = marked_strip_image(source.image, bounds, marks, guides=lines)
        return Report("cut", settings, result, images=(attachment(img, "cut-clearance"),), marks_version=self.catalog.marks_version(marks))

    def _cut_verified(self, source, value):
        options = CutSettings.read(value.get("cut"), source.metadata)
        marks = self.catalog.list_marks(source.id)
        plan = Segment.read(value.get("planned"), "planned", bounds=source.size)
        actual = Segment.read(value.get("actual"), "actual", bounds=source.size)
        tolerance = number(value.get("tolerance", .25), "tolerance", lower=.02, upper=4)
        result = verify_cut(plan, actual, marks, options, tolerance)
        result["region"] = options.guide.as_dict()
        settings = {
            "cut": options.as_dict(), "planned": plan.as_dict(), "actual": actual.as_dict(),
            "tolerance": tolerance, "marks": marks,
        }
        lines = [(plan, "#90b5f0"), (actual, "#8cdab7" if result["status"] == "confirmed" else "#ed805f")]
        img = marked_strip_image(source.image, options.guide, marks, guides=lines)
        return Report(
            "cut-verified", settings, result, images=(attachment(img, "observed-cut"),), marks_version=self.catalog.marks_version(marks)
        )

    def _frames(self, source, value):
        region = self._region(source, value)
        minimum = integer(value.get("minimum_width", max(16, region.w // 7)), "minimum_width", lower=8, upper=region.w)
        gap = integer(value.get("separation", 14), "separation", lower=3, upper=200)
        contrast = number(value.get("contrast", .07), "contrast", lower=.008, upper=.4)
        result = suggest_frame_boxes(source.image, region, minimum, gap, contrast)
        provisional = [
            {"label": f"Frame {index}", "geometry": frame["box"], "kind": "frame"}
            for index, frame in enumerate(result["frames"], 1)
        ]
        preview = marked_strip_image(source.image, region, provisional)
        settings = {
            "region": region.as_dict(), "minimum_width": minimum, "separation": gap, "contrast": contrast,
        }
        return Report("frames", settings, result, images=(attachment(preview, "proposed-frame-boxes"),))

    def _perforations(self, source, value):
        region = self._region(source, value)
        minimum = number(value.get("minimum_pitch", 16), "minimum_pitch", lower=3, upper=180)
        contrast = number(value.get("contrast", .08), "contrast", lower=.009, upper=.5)
        fraction = number(value.get("edge_fraction", .12), "edge_fraction", lower=.035, upper=.3)
        stock = one_of(value.get("stock", "135"), {"135", "120", "other"}, "stock")
        result = perforation_survey(source.image, region, minimum, contrast, fraction, stock=stock)
        settings = {
            "region": region.as_dict(), "minimum_pitch": minimum, "contrast": contrast,
            "edge_fraction": fraction, "stock": stock,
        }
        return Report("perforations", settings, result)

    def _proof(self, source, value):
        region = self._region(source, value, optional=True)
        options = ProofSettings.read(value.get("proof"))
        calibration = Calibration.read(value.get("calibration"))
        raw_tone = value.get("tone")
        tone_payload = dict(raw_tone) if isinstance(raw_tone, dict) else ({} if raw_tone is None else raw_tone)
        adopted_grade = value.get("grade")
        if adopted_grade is None and value.get("grade_check") is not None:
            adopted_grade = value.get("grade_check")
        pair_specs = None
        full_reports = None
        grade_curve = None
        if isinstance(adopted_grade, dict):
            pair_specs = adopted_grade.get("pairs") or adopted_grade.get("items")
            grade_curve = adopted_grade.get("curve")
            if pair_specs and all(isinstance(p, dict) and "kept" in p for p in pair_specs):
                full_reports = pair_specs
        if pair_specs is None and value.get("pairs"):
            pair_specs = value.get("pairs")
        # Adoption means the proof picture must use the fitted shared curve even
        # when a caller sends the curve only through the grade envelope.
        if grade_curve and isinstance(tone_payload, dict) and not tone_payload.get("curve"):
            tone_payload["curve"] = grade_curve
        tone = Tone.read(tone_payload)
        view = one_of(value.get("view", "positive"), VIEW_MODES, "view")
        source_copy = self.workspace.preview_image(source, calibration, tone, view, options.max_side, region)
        distribution = None
        if options.include_histogram:
            distribution = histogram(self.workspace.calibrated(source, calibration), region)
        roll = self.catalog.get_roll(source.metadata["roll_id"])
        marks = self.catalog.list_marks(source.id)

        # Adopted strip grade: re-check pairs on the actual downscaled proof
        # photo, before scan_proof paints any annotation markers.
        grade_audit = None
        if pair_specs:
            if full_reports is None and grade_curve is not None:
                try:
                    from .strip_grade import evaluate_pair
                    measurable = Calibration(calibration.flat_id, calibration.dark_id, calibration.base, False)
                    corrected = self.workspace.calibrated(source, measurable)
                    curve = MonotoneCurve.from_dict(grade_curve)
                    domain = grade_curve.get("domain", "density") if isinstance(grade_curve, dict) else "density"
                    full_reports = [
                        evaluate_pair(corrected.values, spec, curve, domain=domain,
                                      medium=source.metadata.get("medium", "negative"),
                                      valid=corrected.valid.all(axis=2))
                        for spec in pair_specs
                    ]
                except DeskError:
                    full_reports = None
            grade_audit = audit_proof_image(
                source_copy, pair_specs,
                region=region if region is not None else None,
                source_size=source.size if region is None else None,
                full_reports=full_reports,
                max_side=options.max_side,
            )

        proof, report = scan_proof(
            source_copy, source.metadata, roll, region or source.bounds, view, options,
            marks=marks, histogram=distribution, grade_audit=grade_audit,
        )
        if grade_audit is not None:
            report["grade_audit"] = grade_audit
            report["grade_lost"] = grade_audit["lost"]
            report["grade_lost_labels"] = grade_audit["lost_labels"]
        settings = {
            "region": (region or source.bounds).as_dict(), "proof": options.as_dict(),
            "tone": tone.as_dict(),
            "view": view, "calibration": calibration.as_dict(),
        }
        if adopted_grade is not None:
            settings["grade"] = {
                "curve": adopted_grade.get("curve") if isinstance(adopted_grade, dict) else None,
                "pair_ids": [p.get("id") for p in adopted_grade.get("pairs", []) if isinstance(p, dict)],
            }
        if options.annotate:
            settings["marks"] = marks
        return Report(
            "proof", settings, report,
            images=(attachment(proof, f"{roll['code']}-proof"),),
            marks_version=self.catalog.marks_version(marks) if options.annotate else None,
        )

    def _density_master(self, source, value):
        region = self._region(source, value, optional=True)
        options = DensityMasterOptions.read(value.get("density_export"))
        calibration, corrected = self._reading(source, value)
        controls = [item.metadata for item in self.workspace.controls(source, calibration) if item]
        config = calibration.as_dict()
        content, result = make_density_master(corrected, region, source.metadata, config, controls, options)
        settings = {
            "region": region.as_dict(), "calibration": config, "density_export": options.as_dict(),
        }
        return Report(
            "density-master", settings, result,
            images=(("density-master.zip", "application/zip", content),),
        )

    def _contact(self, source, value):
        roll_id = source.metadata["roll_id"]
        roll = self.catalog.get_roll(roll_id)
        settings = SheetSettings.read(value.get("sheet"))
        tone = Tone.read(value.get("tone"))
        calibration = Calibration.read(value.get("calibration"))
        selection = value.get("scans")
        if selection is None:
            scans = self.catalog.list_scans(roll_id, role="scan", limit=CONTACT_LIMIT + 1)
            identifiers = [item["id"] for item in scans]
            if len(identifiers) > CONTACT_LIMIT:
                raise DeskError("Choose specific scans on a roll with over 24 scans.", field="scans")
        else:
            identifiers = [checked_key(item, "scans") for item in sequence(selection, "scans", minimum=1, maximum=CONTACT_LIMIT)]
            if len(set(identifiers)) != len(identifiers):
                raise DeskError("Select each scan once.", field="scans")
        images = []
        checked = []
        for identifier in identifiers:
            scan = source if identifier == source.id else self.workspace.get_image(identifier)
            if scan.metadata["roll_id"] != roll_id or scan.metadata["role"] != "scan":
                raise DeskError("Choose scan images on this roll.", field="scans")
            calibrated = calibration if scan.id == source.id else Calibration(None, None, calibration.base, calibration.reject_clipped)
            view = self.workspace.preview_image(scan, calibrated, tone, "positive", 900)
            images.append((scan.metadata, view))
            checked.append({"id": scan.id, "checksum": scan.metadata["checksum"], "name": scan.metadata["name"]})
        pages = contact_pages(images, roll, settings)
        preview = pages[0].copy()
        preview.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
        packet = (
            ("contact-sheet.pdf", "application/pdf", contact_pdf(pages, settings)),
            ("contact-preview.png", PNG, image_bytes(preview)),
        )
        result = {
            "scans": checked,
            "pages": len(pages),
            "width": pages[0].width,
            "height": pages[0].height,
            "paper": settings.paper,
            "calibration_scope": "Only the selected primary scan uses the selected same-capture controls.",
        }
        return Report(
            "contact",
            {"sheet": settings.as_dict(), "tone": tone.as_dict(), "calibration": calibration.as_dict(), "scans": checked},
            result,
            images=packet,
        )

    def _wedge(self, source, value):
        settings = WedgeSpec.read(value.get("wedge"), source.size)
        calibration, corrected = self._reading(source, value)
        result = assess_wedge(corrected, settings)
        file = attachment(wedge_chart(result), "transmission-wedge")
        return Report("wedge", {"wedge": settings.as_dict(), "calibration": calibration.as_dict()}, result, images=(file,))

    def _focus(self, source, value):
        region = self._region(source, value)
        settings = FocusOptions.read(value.get("focus"))
        calibration, corrected = self._reading(source, value)
        result = measure_focus(corrected, region, settings, dpi=source.metadata["dpi"])
        file = attachment(focus_proof(source.image, result), "slanted-edge-focus")
        return Report(
            "focus", {"focus": settings.as_dict(), "calibration": calibration.as_dict(), "region": region.as_dict()},
            result, images=(file,),
        )

    def _coverage(self, source, value):
        region = self._region(source, value, optional=True)
        settings = CoverageSettings.read(value.get("coverage"))
        calibration, corrected = self._reading(source, value)
        marks = self.catalog.list_marks(source.id)
        coverage = measure_coverage(source.image, corrected, region, settings, marks)
        file = attachment(coverage_map(source.image, coverage), "coverage-map")
        return Report(
            "coverage",
            {"coverage": settings.as_dict(), "calibration": calibration.as_dict(), "region": region.as_dict()},
            coverage.result, images=(file,), marks_version=self.catalog.marks_version(marks),
        )

    def _masters(self, source, value):
        settings = MasterRequest.read(value.get("masters"))
        region = self._region(source, value) if value.get("region") is not None else None
        marks = self.catalog.list_marks(source.id) if region is None else []
        crops = crop_list(settings, source.size, marks, region=region)
        _, raw = self.catalog.original(source.id)
        content, result = create_masters(raw, source.metadata, crops, settings)
        payload = {"masters": settings.as_dict(), "region": region.as_dict() if region else None, "frames": result["files"]}
        return Report(
            "masters", payload, result, images=(("frame-masters.zip", "application/zip", content),),
            marks_version=self.catalog.marks_version(marks) if region is None else None,
        )

    def _carrier(self, source, value):
        region = self._region(source, value)
        settings = CarrierSettings.read(value.get("carrier"))
        dpi = number(value.get("dpi", source.metadata["dpi"]), "dpi", lower=30, upper=20_000)
        result = inspect_carrier(source.image, region, settings, dpi)
        result["dpi"] = dpi
        preview = carrier_overlay(source.image, region, result)
        return Report(
            "carrier", {"carrier": settings.as_dict(), "region": region.as_dict(), "dpi": dpi},
            result, images=(attachment(preview, "carrier-rails"),),
        )

    # ------------------------------------------------------------------
    # Shared strip grade across several frames / reference pairs
    # ------------------------------------------------------------------

    def _grade_calibrated(self, source, value):
        """Calibration pass that keeps clipped endpoint patches measurable."""
        calibration = Calibration.read(value.get("calibration"))
        measurable = Calibration(
            calibration.flat_id, calibration.dark_id, calibration.base, False
        )
        corrected = self.workspace.calibrated(source, measurable)
        return calibration, measurable, corrected

    def _mark_index(self, scan_id):
        marks = self.catalog.list_marks(scan_id)
        return marks, {mark["id"]: mark for mark in marks}

    def _resolve_box_ref(self, reference, field_name, size, mark_index):
        if isinstance(reference, str):
            identifier = reference
            if identifier in mark_index:
                geometry = mark_index[identifier]["geometry"]
            else:
                mark = self.catalog.get_mark(checked_key(identifier, field_name))
                geometry = mark["geometry"]
            return Box.read(geometry, field_name, bounds=size)
        if isinstance(reference, dict):
            identifier = reference.get("mark_id") or reference.get("id")
            if identifier and not ({"x", "y", "w", "h"} <= set(reference)):
                if identifier in mark_index:
                    geometry = mark_index[identifier]["geometry"]
                else:
                    mark = self.catalog.get_mark(checked_key(identifier, field_name))
                    geometry = mark["geometry"]
                return Box.read(geometry, field_name, bounds=size)
        return Box.read(reference, field_name, bounds=size)

    def _grade_pair_specs(self, value, source, mark_index):
        raw_pairs = value.get("pairs")
        if raw_pairs is None:
            raw_pairs = value.get("reference_pairs", value.get("references", value.get("pair_list")))
        if isinstance(raw_pairs, dict):
            raw_pairs = raw_pairs.get("pairs", raw_pairs.get("items", []))
        raw_pairs = sequence(raw_pairs if raw_pairs is not None else [], "pairs", maximum=64)
        specs = []
        for index, raw in enumerate(raw_pairs):
            item = mapping(raw, f"pairs[{index}]")
            dark_ref = raw_get(item, ("dark", "dark_box", "a", "box_a", "shadow", "dark_id"))
            bright_ref = raw_get(item, ("bright", "bright_box", "light", "light_box", "b", "box_b",
                                        "highlight", "bright_id", "light_id"))
            if dark_ref is None or bright_ref is None:
                raise DeskError("Give each pair dark and bright reference boxes.", field="pairs")
            dark = self._resolve_box_ref(dark_ref, f"pairs[{index}].dark", source.size, mark_index)
            bright = self._resolve_box_ref(bright_ref, f"pairs[{index}].bright", source.size, mark_index)
            gap = raw_get(item, ("minimum", "min_delta", "required", "gap", "min_gap",
                                 "gray_difference", "difference"))
            spec = {
                "id": str(item.get("id") or item.get("pair_id") or f"pair-{index + 1}"),
                "label": str(item.get("label") or item.get("name") or f"Reference pair {index + 1}")[:80],
                "dark": dark.as_dict(),
                "bright": bright.as_dict(),
                "minimum": gap if gap is not None else value.get("minimum", value.get("default_gap", .12)),
            }
            for key in ("scan_id", "frame", "frame_label"):
                if item.get(key) is not None:
                    spec[key] = item[key]
            specs.append(spec)
        return specs

    def _grade_keep_boxes(self, value, source, mark_index, marks):
        keep_refs = []
        explicit = value.get("keep", value.get("protected"))
        if explicit not in (None, False):
            if isinstance(explicit, (list, tuple)):
                keep_refs.extend(explicit)
            else:
                keep_refs.append(explicit)
        elif value.get("keep_id") or value.get("protected_mark_id"):
            keep_refs.append(value.get("keep_id") or value.get("protected_mark_id"))
        use_saved = value.get("use_saved_keep", value.get("keep_marks", explicit is None))
        if use_saved is not False and not keep_refs:
            for mark in marks:
                if mark["kind"] == "keep":
                    keep_refs.append(mark["id"])
        boxes = []
        for index, reference in enumerate(keep_refs[:8]):
            try:
                boxes.append(self._resolve_box_ref(reference, f"keep[{index}]", source.size, mark_index))
            except DeskError:
                if explicit is not None and explicit is not False:
                    raise
        return boxes

    def _paper_settings(self, value):
        raw_paper = value.get("paper")
        if raw_paper is None:
            black = value.get("paper_black", value.get("paper_dark", value.get("black", PAPER_BLACK)))
            white = value.get("paper_white", value.get("paper_light", value.get("white", PAPER_WHITE)))
            paper = {"black": black, "white": white}
        else:
            paper = raw_paper
        from .strip_grade import read_paper
        return read_paper(paper)

    def _strip_grade(self, source, value):
        calibration, measurable, corrected = self._grade_calibrated(source, value)
        marks, mark_index = self._mark_index(source.id)
        specs = self._grade_pair_specs(value, source, mark_index)
        keep_boxes = self._grade_keep_boxes(value, source, mark_index, marks)
        paper_black, paper_white = self._paper_settings(value)
        domain = one_of(value.get("domain", "density"), {"density", "value", "level"}, "domain")
        if domain == "level":
            domain = "value"
        medium = source.metadata.get("medium", "negative")
        proof_size = value.get("proof_max_side", value.get("output_size", value.get("max_side")))
        if isinstance(value.get("proof"), dict):
            proof_size = proof_size or value["proof"].get("max_side")
        proof_size = integer(proof_size, "proof_max_side", lower=24, upper=4000) if proof_size else None
        region = self._region(source, value, optional=True)

        result = grade_image(
            corrected.values, specs,
            paper_black=paper_black, paper_white=paper_white,
            keep_boxes=[box.as_dict() for box in keep_boxes],
            domain=domain, medium=medium,
            valid=corrected.valid.all(axis=2),
            bounds=source.size, proof_max_side=proof_size, region=region.as_dict() if region else None,
        )

        preview_size = proof_size or integer(value.get("preview_side", 1400), "preview_side",
                                             lower=200, upper=3000)
        marked = marked_grade_image(
            corrected.values, result, proof_max_side=preview_size,
            region=region.as_dict() if region else None,
        )
        images = (attachment(marked, "shared-strip-grade"),)

        settings = {
            "calibration": calibration.as_dict(),
            "paper": {"black": paper_black, "white": paper_white},
            "domain": domain,
            "medium": medium,
            "pairs": specs,
            "keep": [box.as_dict() for box in keep_boxes],
            "curve": result["curve"],
            "proof_max_side": proof_size,
            "region": region.as_dict() if region else None,
        }
        return Report(
            "strip-grade", settings, result,
            images=images, marks_version=self.catalog.marks_version(marks),
        )

    def _proof_audit(self, source, value):
        """Standalone downscale re-check for an adopted grade."""
        grade = mapping(value.get("grade", value), "grade")
        pair_source = grade.get("pairs", value.get("pairs"))
        if pair_source is None:
            raise DeskError("Send the adopted grade with its reference pairs.", field="grade")
        proof_options = ProofSettings.read(value.get("proof"))
        calibration = Calibration.read(value.get("calibration"))
        tone_payload = dict(value.get("tone") or {})
        if grade.get("curve") and not tone_payload.get("curve"):
            tone_payload["curve"] = grade["curve"]
        tone = Tone.read(tone_payload)
        region = self._region(source, value, optional=True)
        proof_image = self.workspace.preview_image(
            source, calibration, tone, "positive", proof_options.max_side, region
        )
        full_reports = grade.get("pairs") if isinstance(grade.get("pairs"), list) else None
        audit = audit_proof_image(
            proof_image, pair_source,
            region=region if region else None,
            source_size=source.size if region is None else None,
            full_reports=full_reports,
        )
        marked = proof_image.copy()
        from .strip_grade import draw_proof_warnings
        draw_proof_warnings(marked, audit, left=0, top=0)
        report = {"grade_audit": audit, "proof": proof_options.as_dict()}
        return Report(
            "proof-audit",
            {"proof": proof_options.as_dict(), "tone": tone.as_dict(),
             "region": region.as_dict() if region else None},
            report, images=(attachment(marked, "proof-size-pair-audit"),),
        )


def raw_get(mapping_like, keys):
    for key in keys:
        if isinstance(mapping_like, dict) and mapping_like.get(key) is not None:
            return mapping_like[key]
    return None
