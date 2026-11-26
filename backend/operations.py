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
from .values import (
    Box, DeskError, Point, Segment, VIEW_MODES, boolean, checked_key, image_bytes, integer,
    mapping, number, one_of, radii, rgb_image, sequence, text_list,
)
from .witnesses import grid_regions, observations_from_marks, parse_cell, witness_plan


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
            "witness-calibration": self._witness_calibration,
            "witness": self._witness_calibration,
            "witness-compensation": self._witness_calibration,
            "reference-compensation": self._witness_calibration,
            "reference-calibration": self._witness_calibration,
            "illumination-witness": self._witness_calibration,
            "witness-map": self._witness_calibration,
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
        compensation = None
        for key in (
            "compensation", "witness_plan", "adopted_compensation", "compensation_plan",
            "witness_result", "plan", "witness_corrections", "calibration_regions",
        ):
            if value.get(key) is not None:
                compensation = value[key]
                break
        if isinstance(compensation, dict) and isinstance(compensation.get("result"), dict):
            compensation = compensation["result"].get("witness_plan", compensation["result"])
        if isinstance(compensation, dict) and "witness_plan" in compensation and "cells" not in compensation:
            compensation = compensation["witness_plan"]
        if compensation is None:
            job_identifier = value.get(
                "compensation_job_id",
                value.get("witness_job_id", value.get("adopted_job_id")),
            )
            if job_identifier:
                job = self.jobs.get(checked_key(job_identifier, "compensation_job_id"))
                job_result = job.report.result
                compensation = job_result.get("witness_plan", job_result)
        marks_version = None
        auto_witness_plan = False
        auto_witness_requested = bool(value.get("use_witnesses") or value.get("witness_compensation"))
        if compensation is None and (auto_witness_requested or any(value.get(key) is not None for key in (
            "reference_cell", "anchor_cell", "fixed_cell",
            "same_name_limit", "jump_limit", "witnesses", "observations",
        ))):
            witness_region = self._region(source, value, "witness_region", optional=True)
            if value.get("grid_region") is not None:
                witness_region = self._region(source, value, "grid_region")
            tiles = (
                integer(value.get("rows", 4), "rows", lower=2, upper=12),
                integer(value.get("columns", 8), "columns", lower=2, upper=18),
            )
            compensation, _, marks_version = self._build_witness_plan(
                source, value, corrected, witness_region, tiles
            )
            auto_witness_plan = True
        zone_level = value.get(
            "zone_level",
            value.get("calibration_level", value.get("region_level", value.get("provenance_level"))),
        )
        report = compare_regions(
            corrected, region, reference, compensation,
            zone_level=zone_level,
            calibration_level=value.get("calibration_level"),
            region_level=value.get("region_level"),
        )
        settings = {
            "region": region.as_dict(), "reference": reference.as_dict(), "calibration": calibration.as_dict(),
        }
        if isinstance(compensation, dict):
            settings["compensation"] = compensation
            settings["zone_level"] = zone_level or compensation.get("compare_zone_level") or compensation.get("zone_level") or "cell"
        if auto_witness_plan:
            report["witness_plan"] = compensation
        return Report("balance", settings, report, marks_version=marks_version)

    def _calibration(self, source, value):
        calibration = Calibration.read(value.get("calibration"))
        flat, dark = self.workspace.controls(source, calibration)
        if source.metadata["role"] == "flat":
            flat = source
            calibration = Calibration(source.id, calibration.dark_id, calibration.base, calibration.reject_clipped)
        witness_requested = bool(
            value.get("witness_compensation") is not None
            or value.get("witness_plan") is not None
            or value.get("reference_cell") is not None
            or value.get("anchor_cell") is not None
            or value.get("fixed_cell") is not None
            or value.get("witnesses") is not None
            or value.get("reference_witnesses") is not None
            or value.get("reference_marks") is not None
            or value.get("references") is not None
            or value.get("observations") is not None
            or value.get("compensate")
            or value.get("recommend")
        )
        if flat is None and not witness_requested:
            raise DeskError("Select a backlight flat to measure its illumination.", field="calibration.flat_id")
        region = self._region(source, value, optional=True)
        tiles = (
            integer(value.get("rows", 4), "rows", lower=2, upper=12),
            integer(value.get("columns", 8), "columns", lower=2, upper=18),
        )
        settings = {
            "region": region.as_dict(), "rows": tiles[0], "columns": tiles[1],
            "calibration": calibration.as_dict(),
        }
        if flat is not None:
            smooth = number(value.get("smooth", 1.6), "smooth", lower=0, upper=15)
            result = calibration_map(flat.image, dark.image if dark else None, region, tiles, smooth, dpi=source.metadata["dpi"])
            result["flat_name"] = flat.metadata["name"]
            result["dark_name"] = dark.metadata["name"] if dark else None
            settings["smooth"] = smooth
        else:
            result = {"region": region.as_dict(), "rows": tiles[0], "columns": tiles[1], "tiles": []}

        marks_version = None
        if witness_requested:
            _, corrected = self._reading(source, value)
            plan, witness_settings, marks_version = self._build_witness_plan(
                source, value, corrected, region, tiles
            )
            result["witness_plan"] = plan
            if flat is None:
                result.update({key: value for key, value in plan.items() if key not in {"region"}})
                result["witness_plan"] = plan
                result["flat_name"] = None
                result["dark_name"] = None
            settings["witnesses"] = witness_settings
        return Report("calibration", settings, result, marks_version=marks_version)

    def _witness_reference_cell(self, value, region, rows, columns):
        raw = value.get(
            "reference_cell",
            value.get("anchor_cell", value.get("fixed_cell", value.get("anchor", value.get("fixed")))),
        )
        if isinstance(raw, bool):
            raw = None
        if raw is None and value.get("reference") is not None:
            reference = Box.read(value.get("reference"), "reference", bounds=(region.x + region.w, region.y + region.h))
            center_x = reference.x + reference.w / 2
            center_y = reference.y + reference.h / 2
            column = min(columns - 1, max(0, int((center_x - region.x) * columns / region.w)))
            row = min(rows - 1, max(0, int((center_y - region.y) * rows / region.h)))
            return row, column
        if raw is None and value.get("reference_row") is not None and value.get("reference_column") is not None:
            raw = {
                "row": value.get("reference_row"),
                "column": value.get("reference_column"),
                "one_based": value.get("reference_one_based", True),
            }
        if raw is None:
            anchor_witness = value.get("anchor_witness", value.get("fixed_witness"))
            if anchor_witness is not None and value.get("witnesses"):
                for item in value["witnesses"]:
                    if str(item.get("id", item.get("witness_id", item.get("mark_id")))) == str(anchor_witness):
                        raw = item.get("cell", {"row": item.get("row"), "column": item.get("column")})
                        break
        if raw is None:
            return None
        return parse_cell(raw, rows, columns)

    def _witness_limits(self, value):
        nested = value.get("limits")
        nested = mapping(nested, "limits") if isinstance(nested, dict) else {}
        witness_options = value.get("witness_compensation")
        witness_options = mapping(witness_options, "witness_compensation") if isinstance(witness_options, dict) else {}

        def pick(*keys, default=None):
            containers = (value, witness_options, nested)
            for container in containers:
                for key in keys:
                    if isinstance(container, dict) and container.get(key) is not None:
                        return container[key]
            return default

        common = pick("limit", "density_limit", default=None)
        same = pick(
            "same_name_limit", "equal_limit", "same_name_difference_limit", "same_name_difference",
            "name_limit", "match_limit", "same_name_tolerance", "equal_density_limit",
            "match_tolerance", "same_name", default=common if common is not None else .04,
        )
        jump = pick(
            "jump_limit", "adjacent_limit", "neighbor_limit", "adjacent_jump_limit",
            "adjacent_jump", "compensation_jump_limit", "jump_tolerance",
            "neighbor_jump_limit", "neighbor_jump", "continuity_limit", "continuity_tolerance",
            "smoothness_limit", "jump", "adjacent",
            default=common if common is not None else .03,
        )
        return same, jump

    def _build_witness_plan(self, source, value, corrected, region, tiles):
        rows, columns = tiles
        grid = grid_regions(region, rows, columns)
        explicit_observations = value.get("observations")
        if explicit_observations is None:
            explicit_observations = value.get("witness_samples")
        raw_witnesses = value.get("witnesses")
        if raw_witnesses is None:
            for key in ("reference_witnesses", "reference_marks", "references", "reference_blocks", "blocks", "patches"):
                if value.get(key) is not None:
                    raw_witnesses = value[key]
                    break
        if explicit_observations is None and isinstance(raw_witnesses, list) and raw_witnesses:
            first_witness = mapping(raw_witnesses[0], "witnesses[0]")
            compact = any(key in first_witness for key in ("median", "density", "medians", "density_median"))
            located = any(key in first_witness for key in ("cell", "row", "r"))
            if compact and located:
                explicit_observations = raw_witnesses
        witness_details = None
        marks_version = None
        if explicit_observations is not None:
            observations = sequence(explicit_observations, "observations", maximum=5000)
        else:
            explicit_marks = raw_witnesses
            if explicit_marks is not None:
                marks = sequence(explicit_marks, "witnesses", maximum=500)
            else:
                marks = self.catalog.list_marks(source.id)
                marks_version = self.catalog.marks_version(marks)
            labels = value.get("witness_labels")
            if labels is not None:
                allowed = set(text_list(labels, "witness_labels"))
                marks = [mark for mark in marks if mark.get("label") in allowed]
            # Same-name groups are the useful equality witnesses.  Keep single
            # references only when the caller explicitly asks for them.
            include_single = boolean(value.get("include_single", False), "include_single") if "include_single" in value else False
            if not include_single:
                counts = {}
                for mark in marks:
                    if mark.get("kind", "reference") == "reference" and mark.get("label"):
                        counts[mark["label"]] = counts.get(mark["label"], 0) + 1
                marks = [
                    mark for mark in marks
                    if mark.get("label") and counts.get(mark["label"], 0) >= 2
                ]
            observations, witness_details = observations_from_marks(corrected, grid, marks)
            # Explicit compact cell witnesses may be supplied alongside marks.
            compact = value.get("cell_witnesses")
            if compact:
                observations = [*observations, *sequence(compact, "cell_witnesses", maximum=5000)]

        reference_cell = self._witness_reference_cell(value, region, rows, columns)
        same_limit, jump_limit = self._witness_limits(value)
        plan = witness_plan(
            rows=rows,
            columns=columns,
            observations=observations,
            region=region,
            reference_cell=reference_cell,
            same_name_limit=same_limit,
            jump_limit=jump_limit,
            witness_details=witness_details,
        )
        settings = {
            "region": region.as_dict(), "rows": rows, "columns": columns,
            "reference_cell": (
                {"row": reference_cell[0], "column": reference_cell[1]} if reference_cell else None
            ),
            "same_name_limit": same_limit, "jump_limit": jump_limit,
        }
        return plan, settings, marks_version

    def _witness_calibration(self, source, value):
        calibration, corrected = self._reading(source, value)
        region = self._region(source, value, optional=True)
        tiles = (
            integer(value.get("rows", 4), "rows", lower=2, upper=12),
            integer(value.get("columns", 8), "columns", lower=2, upper=18),
        )
        plan, witness_settings, marks_version = self._build_witness_plan(source, value, corrected, region, tiles)
        settings = {"calibration": calibration.as_dict(), **witness_settings}
        return Report("witness-calibration", settings, plan, marks_version=marks_version)

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
        tone = Tone.read(value.get("tone"))
        view = one_of(value.get("view", "positive"), VIEW_MODES, "view")
        source_copy = self.workspace.preview_image(source, calibration, tone, view, options.max_side, region)
        distribution = None
        if options.include_histogram:
            distribution = histogram(self.workspace.calibrated(source, calibration), region)
        roll = self.catalog.get_roll(source.metadata["roll_id"])
        marks = self.catalog.list_marks(source.id)
        proof, report = scan_proof(
            source_copy, source.metadata, roll, region, view, options, marks=marks, histogram=distribution
        )
        settings = {
            "region": region.as_dict(), "proof": options.as_dict(), "tone": tone.as_dict(),
            "view": view, "calibration": calibration.as_dict(),
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
