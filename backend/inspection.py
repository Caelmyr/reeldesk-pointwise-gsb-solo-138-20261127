"""Local scan-defect survey with protected image regions and repeat-pass checks."""

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .imaging import BLACK_FLOOR, SCANNER_LUMA
from .registration import sample_peer
from .values import (
    Box, DeskError, integer, mapping, number, one_of, sequence, rgb_image,
)


DIRECTIONS = frozenset({"both", "dense", "clear"})
STRUCTURE_8 = np.ones((3, 3), dtype=bool)
REVIEW_STATES = frozenset({"keep", "restore"})
DEFECT_PIXEL_LIMIT = 2_000_000


@dataclass(frozen=True)
class Survey:
    radius: int
    threshold: float
    min_area: int
    max_area: int
    flatness: float
    direction: str
    reach: int

    @classmethod
    def read(cls, raw):
        value = {} if raw is None else mapping(raw, "survey")
        minimum = integer(value.get("min_area", 4), "survey.min_area", lower=1, upper=3000)
        maximum = integer(value.get("max_area", 300), "survey.max_area", lower=minimum, upper=20_000)
        return cls(
            integer(value.get("radius", 5), "survey.radius", lower=2, upper=24),
            number(value.get("threshold", .12), "survey.threshold", lower=.015, upper=1.2),
            minimum,
            maximum,
            number(value.get("flatness", .34), "survey.flatness", lower=.03, upper=1.1),
            one_of(value.get("direction", "both"), DIRECTIONS, "survey.direction"),
            integer(value.get("reach", 1), "survey.reach", lower=0, upper=5),
        )

    def as_dict(self):
        return {
            "radius": self.radius, "threshold": self.threshold, "min_area": self.min_area,
            "max_area": self.max_area, "flatness": self.flatness,
            "direction": self.direction, "reach": self.reach,
        }


@dataclass(frozen=True)
class Finding:
    number: int
    box: Box
    pixels: int
    polarity: str
    severity: float
    roundness: float
    protected: int
    edge_contact: bool

    def as_dict(self):
        return {
            "number": self.number,
            "box": self.box.as_dict(),
            "pixels": self.pixels,
            "polarity": self.polarity,
            "severity": round(self.severity, 5),
            "roundness": round(self.roundness, 4),
            "protected_pixels": self.protected,
            "edge_contact": self.edge_contact,
            "can_restore": self.protected == 0 and not self.edge_contact,
        }


@dataclass(frozen=True)
class SurveyResult:
    region: Box
    labels: np.ndarray
    findings: tuple[Finding, ...]
    usable_fraction: float
    cutoff_count: int

    def as_dict(self):
        return {
            "region": self.region.as_dict(),
            "findings": [finding.as_dict() for finding in self.findings],
            "usable_fraction": round(self.usable_fraction, 5),
            "cutoff_count": self.cutoff_count,
        }


def _box_mask(region, boxes):
    mask = np.zeros((region.h, region.w), dtype=bool)
    for box in boxes:
        joint = region.intersect(box)
        if joint is not None:
            mask[joint.translate(-region.x, -region.y).slices] = True
    return mask


def _fill_holes(values, valid):
    if not valid.any():
        return values.copy()
    if valid.all():
        return values
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(nearest)]


def _residuals(values, valid, radius):
    axis = np.arange(-radius, radius + 1)
    disk = axis[:, None] ** 2 + axis[None, :] ** 2 <= radius ** 2
    filled = _fill_holes(values, valid)
    smooth = ndimage.median_filter(filled, footprint=disk, mode="reflect")
    contrast = ndimage.maximum_filter(smooth, footprint=disk) - ndimage.minimum_filter(smooth, footprint=disk)
    residual = values - smooth
    return residual, contrast


def _find_components(mask, residual, protected, settings, region, polarity):
    if settings.reach:
        # Envelope nearby grains for display, while counting only measured grains.
        grown = ndimage.binary_dilation(mask, structure=STRUCTURE_8, iterations=settings.reach)
    else:
        grown = mask
    labels, _ = ndimage.label(grown, structure=STRUCTURE_8)
    areas = np.bincount(labels[mask], minlength=int(labels.max()) + 1)
    objects = ndimage.find_objects(labels)
    results = []
    omitted = 0
    for index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        area = int(areas[index])
        if area < settings.min_area:
            continue
        if area > settings.max_area:
            omitted += 1
            continue
        ys, xs = bounds
        local_labels = (labels[bounds] == index) & mask[bounds]
        local_rows, local_columns = np.nonzero(local_labels)
        tight = (
            slice(ys.start + local_rows.min(), ys.start + local_rows.max() + 1),
            slice(xs.start + local_columns.min(), xs.start + local_columns.max() + 1),
        )
        by, bx = tight
        tight_labels = (labels[tight] == index) & mask[tight]
        contact = (
            by.start == 0 or bx.start == 0
            or by.stop == region.h or bx.stop == region.w
        )
        perimeter = tight_labels & ~ndimage.binary_erosion(tight_labels)
        roundness = min(1.0, 4 * math.pi * area / max(1, int(perimeter.sum())) ** 2)
        protected_count = int((protected[tight] & tight_labels).sum())
        severity = float(np.quantile(np.abs(residual[tight][tight_labels]), .9))
        box = Box(int(region.x + bx.start), int(region.y + by.start), int(bx.stop - bx.start), int(by.stop - by.start))
        results.append((box, area, polarity, severity, roundness, protected_count, bool(contact), tight_labels))
    return results, omitted


def survey_defects(calibrated, region, settings, protected_boxes):
    if region.w < 8 or region.h < 8 or region.area > DEFECT_PIXEL_LIMIT:
        raise DeskError("Survey between 8 × 8 pixels and two million pixels at a time.", field="region")
    density = calibrated.density[region.slices]
    eligible = calibrated.valid[region.slices].all(axis=2)
    valid_count = int(eligible.sum())
    if valid_count < max(15, region.area * .12):
        raise DeskError("This region has too little usable transmission for a dust survey.", field="region")
    luma = np.einsum("ijk,k->ij", density, SCANNER_LUMA)
    residual, local_contrast = _residuals(luma, eligible, settings.radius)
    protected = _box_mask(region, protected_boxes)
    allowed = eligible & (local_contrast <= settings.flatness)
    all_findings = []
    cutoff_count = 0
    if settings.direction != "clear":
        extra, omitted = _find_components(
            allowed & (residual >= settings.threshold), residual, protected, settings, region, "dense"
        )
        all_findings.extend(extra)
        cutoff_count += omitted
    if settings.direction != "dense":
        extra, omitted = _find_components(
            allowed & (residual <= -settings.threshold), residual, protected, settings, region, "clear"
        )
        all_findings.extend(extra)
        cutoff_count += omitted
    all_findings.sort(key=lambda item: (-item[3], item[0].y, item[0].x, item[2]))
    cutoff_count += max(0, len(all_findings) - 200)
    labels = np.zeros_like(eligible, dtype=np.int32)
    findings = []
    for index, (box, area, sign, severity, roundness, count, contact, local_mask) in enumerate(all_findings[:200], 1):
        local = box.translate(-region.x, -region.y)
        labels[local.slices][local_mask] = index
        findings.append(Finding(index, box, area, sign, severity, roundness, count, contact))
    return SurveyResult(region, labels, tuple(findings), valid_count / region.area, cutoff_count)


def local_survey_comparison(source, peer, transform, survey, number, margin=10,
                            difference_limit=.15, repair=None, *extra, **kwargs):
    finding = next((finding for finding in survey.findings if finding.number == number), None)
    if finding is None:
        raise DeskError("Select a current dust finding.", field="finding")
    for extra_item in extra:
        if isinstance(extra_item, dict):
            for key, value in extra_item.items():
                kwargs.setdefault(key, value)
        elif repair is None:
            repair = extra_item
    if repair is None:
        repair = kwargs.get("repair_plan", kwargs.get("repaired", kwargs.get("restored")))
    if isinstance(repair, (int, float)) and not isinstance(repair, bool):
        kwargs.setdefault("edge_tolerance", float(repair))
        repair = None
    margin_box = finding.box.expanded(margin, *source.size)
    focus = margin_box.intersect(survey.region)
    yy, xx = np.mgrid[focus.y:focus.bottom, focus.x:focus.right]
    label_area = focus.translate(-survey.region.x, -survey.region.y)
    labels_focus = survey.labels[label_area.slices]
    original_mask = labels_focus == number
    neutral = labels_focus == 0
    source_density = source.density[focus.slices]
    peer_density, peer_usable = sample_peer(peer.density, peer.valid, transform, xx, yy)
    joint = source.valid[focus.slices] & peer_usable
    delta = np.einsum("ijk,k->ij", source_density - peer_density, SCANNER_LUMA)
    complete = joint.all(axis=2)
    interior = original_mask & complete
    ring = neutral & complete
    enough_legacy = ring.sum() >= max(8, finding.pixels // 2) and interior.sum() >= finding.pixels * .65

    center = transform.project(finding.box.center)
    if not enough_legacy:
        report = {
            "finding": number, "region": focus.as_dict(), "verdict": "insufficient",
            "interior": int(interior.sum()), "ring": int(ring.sum()),
            "median_shift": None, "local_difference": None,
            "peer_center": center.as_dict(),
        }
    else:
        baseline = float(np.median(delta[ring]))
        local_difference = float(np.median(delta[interior])) - baseline
        ring_residual = float(np.median(np.abs(delta[ring] - baseline)))
        if ring_residual > difference_limit * .75:
            verdict = "registration-review"
        elif abs(local_difference) >= difference_limit:
            verdict = "changes-between-passes"
        else:
            verdict = "persists-in-peer"
        report = {
            "finding": number,
            "region": focus.as_dict(),
            "verdict": verdict,
            "interior": int(interior.sum()),
            "ring": int(ring.sum()),
            "median_shift": round(baseline, 5),
            "local_difference": round(local_difference, 5),
            "ring_deviation": round(ring_residual, 5),
            "peer_center": center.as_dict(),
        }

    repair_data = repair if repair is not None else _first_option(
        kwargs, ("repair_plan", "restore_plan", "restoration", "repaired", "restored"))
    plan_state = _plan_finding_state(repair_data, number)
    edge_limit = float(_first_option(
        kwargs,
        ("edge_tolerance", "edge_limit", "edge_difference_limit", "seam_tolerance",
         "jump_tolerance", "tolerance"),
        difference_limit,
    ))
    min_edge_samples = _int_option(kwargs, ("min_edge_samples", "edge_min_samples"), 3, minimum=1, maximum=20)

    if plan_state == "manual":
        edge = {
            "mode": "measured-grain-seam",
            "status": "not-repaired",
            "verdict": "manual-review",
            "passed": False,
            "review_required": True,
            "reason": "This finding has no adopted donor assignment.",
            "edges": [], "flagged_edges": [], "insufficient_edges": [],
            "edge_limit": round(edge_limit, 5),
        }
    else:
        repaired_density = _restored_density(source, survey, repair_data, kwargs)
        if repaired_density is None:
            edge = None
        else:
            repaired_valid = _restored_validity(source, repair_data, kwargs)
            edge = restored_edge_report(
                finding, focus, labels_focus, peer_density, complete,
                repaired_density, edge_limit, min_samples=min_edge_samples,
                repaired_valid=repaired_valid,
            )

    if edge is not None:
        report["edge_check"] = edge
        for alias in ("edge_results", "side_results", "boundary_checks", "seam_edges", "edge_by_side", "worst_new_jump"):
            report[alias] = edge.get(alias)
        report["edges"] = edge.get("edges", [])
        report["flagged_edges"] = edge.get("flagged_edges", [])
        report["failed_edges"] = edge.get("failed_edges", [])
        report["insufficient_edges"] = edge.get("insufficient_edges", [])
        report["review_edges"] = edge.get("review_edges", [])
        report["edge_limit"] = round(edge_limit, 5)
        report["edge_tolerance"] = round(edge_limit, 5)
        report["edge_status"] = edge["status"]
        report["status"] = edge["status"]
        report["state"] = edge["status"]
        report["edge_verdict"] = edge["verdict"]
        report["passed"] = edge["status"] == "pass"
        report["ok"] = report["passed"]
        report["review_required"] = bool(edge.get("review_required", edge["status"] == "review"))
        report["needs_review"] = report["review_required"]
        report["edge_jump_detected"] = edge.get("edge_jump_detected", False)
        report["check_basis"] = "repaired-boundary-pairs"
        report["center_used_for_pass"] = False
        report["verdict"] = edge["verdict"]
    return report


def _plan_finding_state(repair, number):
    if not isinstance(repair, dict):
        return "repaired" if repair is not None else None
    plan = repair.get("plan", repair.get("repair_plan", repair.get("donor_plan", repair)))
    if isinstance(plan, dict) and isinstance(plan.get("result"), dict):
        plan = plan["result"].get("plan", plan["result"])
    assignments = None
    if isinstance(plan, dict):
        assignments = plan.get("assignments") or plan.get("donor_assignments")
    if assignments is None:
        restored = repair.get("restored_findings")
        if isinstance(restored, list):
            return "restored" if number in restored else "manual"
        return None
    for item in assignments:
        if isinstance(item, dict) and int(item.get("number", item.get("finding", -1))) == number:
            return item.get("status", "restored")
    manual = plan.get("manual_findings") or plan.get("manual") or []
    for item in manual:
        if isinstance(item, dict) and int(item.get("number", -1)) == number:
            return "manual"
        if item == number:
            return "manual"
    return "manual" if repair is not None else None


def restored_edge_report(finding, focus, labels_focus, peer_density, complete,
                         repaired_density, tolerance, min_samples=3, repaired_valid=None):
    """Compare adjacent inner/outer seam jumps in repaired scan and aligned repeat."""
    if repaired_density.shape[:2] != complete.shape:
        repaired_focus = repaired_density[focus.slices]
    else:
        repaired_focus = repaired_density
    finite = np.isfinite(repaired_focus).all(axis=2)
    if repaired_valid is None:
        validity = np.ones(complete.shape, dtype=bool)
    elif repaired_valid.shape == complete.shape:
        validity = repaired_valid
    else:
        validity = repaired_valid[focus.slices]
    valid = complete & finite & validity

    side_specs = (
        ("top", range(finding.box.x, finding.box.right), "column"),
        ("bottom", range(finding.box.x, finding.box.right), "column"),
        ("left", range(finding.box.y, finding.box.bottom), "row"),
        ("right", range(finding.box.y, finding.box.bottom), "row"),
    )
    edges = []
    flagged = []
    insufficient = []
    overall_status = "pass"

    for side, coordinates, orientation in side_specs:
        inner_indices = []
        outer_indices = []
        eligible_lines = 0
        for coordinate in coordinates:
            if orientation == "column":
                line = labels_focus[:, coordinate - focus.x]
                matches = np.flatnonzero(line == finding.number)
                if not len(matches):
                    continue
                eligible_lines += 1
                inner_index = int(matches.min() if side == "top" else matches.max())
                outer_index = inner_index - 1 if side == "top" else inner_index + 1
                inner = (inner_index, coordinate - focus.x)
                outer = (outer_index, coordinate - focus.x)
            else:
                line = labels_focus[coordinate - focus.y, :]
                matches = np.flatnonzero(line == finding.number)
                if not len(matches):
                    continue
                eligible_lines += 1
                inner_index = int(matches.min() if side == "left" else matches.max())
                outer_index = inner_index - 1 if side == "left" else inner_index + 1
                inner = (coordinate - focus.y, inner_index)
                outer = (coordinate - focus.y, outer_index)
            if not (0 <= outer[0] < focus.h and 0 <= outer[1] < focus.w):
                continue
            if labels_focus[outer] != 0:
                continue
            inner_indices.append(inner)
            outer_indices.append(outer)

        inner_indices = np.asarray(inner_indices, dtype=int).reshape(-1, 2) if inner_indices else np.empty((0, 2), int)
        outer_indices = np.asarray(outer_indices, dtype=int).reshape(-1, 2) if outer_indices else np.empty((0, 2), int)
        inner_valid = valid[inner_indices[:, 0], inner_indices[:, 1]] if len(inner_indices) else np.empty(0, bool)
        outer_valid = valid[outer_indices[:, 0], outer_indices[:, 1]] if len(outer_indices) else np.empty(0, bool)
        pair_good = inner_valid & outer_valid
        pair_count = int(pair_good.sum())
        required = max(int(min_samples), int(math.ceil(.5 * max(1, eligible_lines))))
        compass = {"top": "north", "bottom": "south", "left": "west", "right": "east"}[side]
        edge = {
            "side": side,
            "edge": side,
            "direction": side,
            "position": side,
            "compass": compass,
            "cardinal": compass,
            "eligible_lines": int(eligible_lines),
            "inner_samples": int(inner_valid.sum()),
            "outer_samples": int(outer_valid.sum()),
            "inner_valid": int(inner_valid.sum()),
            "outer_valid": int(outer_valid.sum()),
            "inner_count": int(inner_valid.sum()),
            "outer_count": int(outer_valid.sum()),
            "sample_count": pair_count,
            "pair_count": pair_count,
            "pairs": pair_count,
            "required_pairs": required,
            "repaired_jump": None,
            "repeat_jump": None,
            "new_jump": None,
            "absolute_new_jump": None,
            "difference": None,
            "signed_difference": None,
            "limit": round(float(tolerance), 5),
            "exceeds_tolerance": False,
            "new_bright_jump": False,
            "new_dark_jump": False,
            "insufficient_evidence": True,
            "valid": False,
            "status": "insufficient",
        }
        if pair_count < required:
            edge["verdict"] = "insufficient-evidence"
            edges.append(edge)
            insufficient.append(side)
            overall_status = "review"
            continue

        ii = inner_indices[pair_good]
        oo = outer_indices[pair_good]
        repaired_inner = repaired_focus[ii[:, 0], ii[:, 1]]
        repaired_outer = repaired_focus[oo[:, 0], oo[:, 1]]
        peer_inner = peer_density[ii[:, 0], ii[:, 1]]
        peer_outer = peer_density[oo[:, 0], oo[:, 1]]
        repaired_pair_jump = np.einsum("ij,j->i", repaired_inner - repaired_outer, SCANNER_LUMA)
        peer_pair_jump = np.einsum("ij,j->i", peer_inner - peer_outer, SCANNER_LUMA)
        new_pair_jump = repaired_pair_jump - peer_pair_jump
        new_jump = float(np.median(new_pair_jump))
        repaired_jump = float(np.median(repaired_pair_jump))
        repeat_jump = float(np.median(peer_pair_jump))
        inner_median = float(np.median(np.einsum("ij,j->i", repaired_inner, SCANNER_LUMA)))
        outer_median = float(np.median(np.einsum("ij,j->i", repaired_outer, SCANNER_LUMA)))
        edge.update({
            "repaired_jump": round(repaired_jump, 5),
            "repeat_jump": round(repeat_jump, 5),
            "new_jump": round(new_jump, 5),
            "absolute_new_jump": round(abs(new_jump), 5),
            "p90_absolute_new_jump": round(float(np.quantile(np.abs(new_pair_jump), .9)), 5),
            "inner_median_density": round(inner_median, 5),
            "outer_median_density": round(outer_median, 5),
        })
        if new_jump > tolerance:
            edge["status"] = "dark-jump"
            edge["verdict"] = "new-dark-jump"
            flagged.append(side)
            overall_status = "fail"
        elif new_jump < -tolerance:
            edge["status"] = "bright-jump"
            edge["verdict"] = "new-bright-jump"
            flagged.append(side)
            overall_status = "fail"
        else:
            edge["status"] = "pass"
            edge["verdict"] = "within-tolerance"
        edge.update({
            "difference": edge["absolute_new_jump"],
            "signed_difference": edge["new_jump"],
            "exceeds_tolerance": edge["status"] in {"dark-jump", "bright-jump"},
            "new_dark_jump": edge["status"] == "dark-jump",
            "new_bright_jump": edge["status"] == "bright-jump",
            "insufficient_evidence": False,
            "valid": edge["status"] == "pass",
        })
        edges.append(edge)

    if overall_status == "fail":
        verdict = "edge-jump-exceeds-tolerance"
        edge_verdict = "jump-exceeds-tolerance"
        passed, review = False, False
    elif overall_status == "review":
        verdict = "edge-evidence-review"
        edge_verdict = "insufficient-evidence"
        passed, review = False, True
    else:
        verdict = "repair-edge-pass"
        edge_verdict = "pass"
        passed, review = True, False
    worst_jump = max((abs(edge.get("new_jump") or 0.0) for edge in edges), default=None)
    by_side = {}
    for edge in edges:
        by_side[edge["side"]] = edge
        by_side[edge["compass"]] = edge
    return {
        "mode": "measured-grain-seam",
        "status": "pass" if passed else ("review" if review else "fail"),
        "verdict": verdict,
        "edge_verdict": edge_verdict,
        "passed": passed,
        "ok": passed,
        "review_required": review,
        "needs_review": review,
        "edge_jump_detected": bool(flagged),
        "insufficient_evidence": bool(insufficient),
        "tolerance": round(float(tolerance), 5),
        "edges": edges,
        "edge_results": edges,
        "side_results": edges,
        "boundary_checks": edges,
        "seam_edges": edges,
        "edge_by_side": by_side,
        "sides": list(by_side),
        "flagged_edges": flagged,
        "failed_edges": flagged,
        "bad_edges": flagged,
        "insufficient_edges": insufficient,
        "review_edges": insufficient,
        "worst_new_jump": None if worst_jump is None else round(worst_jump, 5),
        "basis": "adjacent inner/outer boundary pairs at identical repaired and repeat coordinates",
        "center_used_for_pass": False,
    }


def _restore_selection(survey, raw_findings):
    requested = sequence(raw_findings, "findings", minimum=1, maximum=50)
    selected = set()
    seen = set()
    by_number = {finding.number: finding for finding in survey.findings}
    for index, value in enumerate(requested):
        field = f"findings[{index}]"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number_value = int(value)
            state = "restore"
        else:
            value = mapping(value, field)
            number_value = integer(value.get("number"), f"{field}.number", lower=1, upper=200)
            state = one_of(value.get("state", "restore"), REVIEW_STATES | {"restore", "auto"}, f"{field}.state")
            # The historical API only accepts keep/restore.  "auto" is a convenient
            # alias used by newer clients and direct planner callers.
            state = "restore" if state in {"restore", "auto"} else state
        finding = by_number.get(number_value)
        if finding is None:
            raise DeskError("Choose findings from the current survey.", field=field)
        if number_value in seen:
            raise DeskError("Each finding may be selected once.", field=field)
        seen.add(number_value)
        if state == "restore":
            if finding.protected or finding.edge_contact:
                raise DeskError("A protected or cropped finding cannot be automatically retouched.", field=field)
            selected.add(number_value)
    if not selected:
        raise DeskError("Choose at least one finding to restore.", field="findings")
    return selected


def _first_option(options, names, default=None):
    if isinstance(options, dict):
        for name in names:
            value = options.get(name)
            if value is not None:
                return value
    return default


def _float_option(options, names, default, *, minimum=0.0):
    raw = _first_option(options, names, default)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise DeskError("Enter a numeric density limit.", field=names[0])
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise DeskError("Enter a numeric density limit.", field=names[0]) from None
    if not math.isfinite(value) or value < minimum:
        raise DeskError("Enter a finite density limit at least zero.", field=names[0])
    return value


def _int_option(options, names, default, *, minimum=1, maximum=500):
    raw = _first_option(options, names, default)
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value))


def _box_from(raw, field="donor_box"):
    if isinstance(raw, Box):
        return raw
    if isinstance(raw, dict):
        nested = raw.get("box") or raw.get("region") or raw.get("geometry") or raw.get("source_box")
        if nested is not None and nested is not raw:
            return _box_from(nested, field)
        if "w" not in raw and "h" not in raw and ("x2" in raw or "right" in raw):
            x = int(raw.get("x", raw.get("x1", 0)))
            y = int(raw.get("y", raw.get("y1", 0)))
            right = int(raw.get("right", raw.get("x2", x)))
            bottom = int(raw.get("bottom", raw.get("y2", y)))
            return Box(x, y, max(1, right - x), max(1, bottom - y))
        return Box.read(raw, field)
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return Box(int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
    raise DeskError("Use a box with x, y, w and h.", field=field)


def _named_box(raw, index):
    name = None
    payload = raw
    if isinstance(raw, dict):
        name = raw.get("name") or raw.get("label") or raw.get("donor_name") or raw.get("id")
        payload = raw.get("box") or raw.get("region") or raw.get("geometry") or raw.get("source_box") or raw
    box = _box_from(payload, f"donors[{index}]")
    return str(name or f"donor-{index + 1}")[:80], box


def _donor_specs(raw):
    if raw is None:
        return []
    if isinstance(raw, dict):
        if isinstance(raw.get("boxes"), (list, tuple)):
            return _donor_specs(raw["boxes"])
        if isinstance(raw.get("donor_boxes"), (list, tuple)):
            return _donor_specs(raw["donor_boxes"])
        specs = []
        for index, (name, value) in enumerate(raw.items()):
            if isinstance(value, (list, tuple)) and len(value) == 4 and not isinstance(value[0], (list, tuple, dict)):
                specs.append((str(name), _box_from(value, f"donors.{name}")))
            elif isinstance(value, (dict, Box, list, tuple)):
                try:
                    label, box = _named_box(value, index)
                except DeskError:
                    label, box = str(name), _box_from(value, f"donors.{name}")
                if isinstance(value, dict) and not (value.get("name") or value.get("label")):
                    label = str(name)
                specs.append((label, box))
        return specs
    if isinstance(raw, (list, tuple)):
        return [_named_box(item, index) for index, item in enumerate(raw)]
    raise DeskError("Donor boxes must be a list or named object.", field="donors")


def _donor_pool_and_map(raw):
    """Return global donor boxes and optional finding-number-specific boxes."""
    if raw is None:
        return [], {}
    if isinstance(raw, dict) and isinstance(raw.get("donors"), (list, dict)):
        return _donor_pool_and_map(raw["donors"])
    if isinstance(raw, list) and raw and all(
        isinstance(item, dict) and (item.get("finding") is not None or item.get("number") is not None)
        and (item.get("box") is not None or item.get("donor_box") is not None
             or item.get("source_box") is not None or item.get("region") is not None)
        for item in raw
    ):
        mapped = defaultdict(list)
        for index, item in enumerate(raw):
            number = int(item.get("finding", item.get("number")))
            name, box = _named_box(item, index)
            mapped[number].append((name, box))
        return [], dict(mapped)
    if isinstance(raw, dict) and raw and all(str(key).strip().isdigit() for key in raw):
        mapped = {}
        for key, value in raw.items():
            number = int(key)
            mapped[number] = _donor_specs(value)
        return [], mapped
    return _donor_specs(raw), {}


def _global_box_mask(shape, boxes):
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    bounds = Box(0, 0, width, height)
    for raw in boxes or ():
        box = _box_from(raw, "protected_boxes")
        joint = bounds.intersect(box)
        if joint is not None:
            mask[joint.slices] = True
    return mask


def _greedy_donor_assignment(by_finding):
    """Deterministic fallback used only if the MILP solver is unavailable."""
    assignment = {}
    used = set()
    ordered = sorted(by_finding, key=lambda item: (len(item[1]), item[0]))
    for number, candidates in ordered:
        for candidate in sorted(candidates, key=lambda item: (item["seam_difference"], item["ring_difference"], item["source_box"].y, item["source_box"].x)):
            pixels = set(int(value) for value in candidate["pixels"])
            if pixels.isdisjoint(used):
                assignment[number] = candidate
                used.update(pixels)
                break
    return assignment


def _optimize_donor_assignment(by_finding):
    candidates = []
    groups = []
    for number, finding_candidates in by_finding:
        group = []
        for candidate in finding_candidates:
            group.append(len(candidates))
            candidates.append(candidate)
        groups.append((number, group))
    if not candidates:
        return {}

    rows, columns, values = [], [], []
    row_count = 0
    for _, group in groups:
        for variable in group:
            rows.append(row_count)
            columns.append(variable)
            values.append(1.0)
        row_count += 1

    pixel_owners = defaultdict(list)
    for variable, candidate in enumerate(candidates):
        for pixel in candidate["pixels"]:
            pixel_owners[int(pixel)].append(variable)
    for owners in pixel_owners.values():
        if len(owners) <= 1:
            continue
        for variable in owners:
            rows.append(row_count)
            columns.append(variable)
            values.append(1.0)
        row_count += 1

    matrix = coo_matrix((values, (rows, columns)), shape=(row_count, len(candidates))).tocsr()
    one = np.ones(len(candidates), dtype=float)
    base_constraint = LinearConstraint(matrix, np.full(row_count, -np.inf), np.ones(row_count))

    def solve(objective, *, minimum=None, maximum_seam=None):
        upper = np.ones(len(candidates), dtype=float)
        if maximum_seam is not None:
            seams = np.asarray([item["seam_difference"] for item in candidates], dtype=float)
            upper[seams > maximum_seam + 1e-12] = 0.0
        constraints = [base_constraint]
        if minimum is not None:
            count_matrix = coo_matrix((one, (np.zeros(len(candidates)), np.arange(len(candidates)))),
                                      shape=(1, len(candidates))).tocsr()
            constraints.append(LinearConstraint(count_matrix, [minimum], [minimum]))
        return milp(
            c=np.asarray(objective, dtype=float),
            integrality=np.ones(len(candidates)),
            bounds=Bounds(np.zeros(len(candidates)), upper),
            constraints=constraints,
            options={"disp": False, "time_limit": 8.0, "mip_rel_gap": 0.0},
        )

    try:
        maximum_result = solve(-one)
        if not getattr(maximum_result, "success", False) or maximum_result.x is None:
            return _greedy_donor_assignment(by_finding)
        selected_variables = [index for index, value in enumerate(maximum_result.x) if value > 0.5]
        maximum_count = len(selected_variables)
        if maximum_count == 0:
            return {}

        seams = np.asarray([item["seam_difference"] for item in candidates], dtype=float)
        thresholds = sorted(set(float(value) for value in seams))
        low, high = 0, len(thresholds) - 1
        while low < high:
            middle = (low + high) // 2
            trial = solve(np.zeros(len(candidates)), minimum=maximum_count, maximum_seam=thresholds[middle])
            if getattr(trial, "success", False) and trial.x is not None and sum(trial.x > .5) >= maximum_count:
                high = middle
            else:
                low = middle + 1
        best_seam = thresholds[low]
        rings = np.asarray([item["ring_difference"] for item in candidates], dtype=float)
        final = solve(seams + .25 * rings, minimum=maximum_count, maximum_seam=best_seam)
        if not getattr(final, "success", False) or final.x is None:
            final = maximum_result
        chosen = {index for index, value in enumerate(final.x) if value > .5}
        assignment = {}
        for number, group in groups:
            for variable in group:
                if variable in chosen:
                    assignment[number] = candidates[variable]
                    break
        if len(assignment) != maximum_count:
            return _greedy_donor_assignment(by_finding)
        return assignment
    except Exception:
        # Optimization is an enhancement.  The greedy path still enforces every
        # hard pixel-exclusivity constraint.
        return _greedy_donor_assignment(by_finding)


def plan_grain_restoration(source, survey, selected, *, radius=12, protected_boxes=(),
                           protected_mask=None, donor_boxes=None, options=None):
    """Choose disjoint measured-grain donors and return image plus an audit plan."""
    options = options if isinstance(options, dict) else {}
    if not isinstance(selected, set):
        selected_values = list(selected or ())
        if selected_values and isinstance(selected_values[0], dict):
            selected = _restore_selection(survey, selected_values)
        else:
            selected = {int(value) for value in selected_values}
    height, width = source.values.shape[:2]
    source_valid = np.asarray(source.valid)
    if source_valid.ndim == 3:
        source_valid = source_valid.all(axis=2)
    density = getattr(source, "density", None)
    if density is None:
        density = _density_from_values(source.values)
    luma_density = np.einsum("ijk,k->ij", density, SCANNER_LUMA)

    dust_global = np.zeros((height, width), dtype=bool)
    dust_global[survey.region.slices] = survey.labels > 0
    protected = _global_box_mask((height, width), protected_boxes)
    if protected_mask is not None:
        explicit_mask = np.asarray(protected_mask, dtype=bool)
        if explicit_mask.shape == protected.shape:
            protected |= explicit_mask
    for mask_key in ("protected_mask", "keep_mask", "avoid_mask"):
        raw_mask = _first_option(options, (mask_key,))
        if raw_mask is not None:
            array_mask = np.asarray(raw_mask, dtype=bool)
            if array_mask.shape == protected.shape:
                protected |= array_mask
    clean = source_valid & ~protected & ~dust_global

    ring_limit = _float_option(
        options,
        ("ring_limit", "outer_limit", "limit", "maximum_difference", "max_difference",
         "outer_density_limit", "outer_difference_limit", "ring_density_limit",
         "ring_difference_limit", "ring_tolerance", "outer_tolerance",
         "max_ring_difference", "maximum_ring_difference", "match_limit",
         "tolerance", "difference_limit", "density_limit"),
        .10,
    )
    edge_tolerance = _float_option(
        options,
        ("edge_tolerance", "edge_limit", "edge_difference_limit", "seam_tolerance",
         "jump_tolerance", "edge_density_limit", "tolerance"),
        ring_limit,
    )
    ring_width = _int_option(options, ("ring_width", "outer_width", "edge_band"), 2, minimum=1, maximum=8)
    candidate_limit = _int_option(options, ("candidate_limit", "donor_limit"), 64, minimum=4, maximum=240)
    largest_w = max((f.box.w for f in survey.findings if f.number in selected), default=0)
    largest_h = max((f.box.h for f in survey.findings if f.number in selected), default=0)
    search_radius = _int_option(
        options,
        ("donor_radius", "search_radius", "source_radius"),
        max(18, int(radius), largest_w + 8, largest_h + 8),
        minimum=2, maximum=240,
    )
    global_specs, mapped_specs = _donor_pool_and_map(
        donor_boxes if donor_boxes is not None else _first_option(
            options, ("donor_boxes", "donors", "source_boxes", "grain_sources",
                      "source_patches", "donor_assignments", "named_sources"))
    )

    by_finding = []
    diagnostics = {}
    for number in sorted(selected):
        specs = mapped_specs.get(number, global_specs)
        finding = next(item for item in survey.findings if item.number == number)
        box = finding.box
        local_box = box.translate(-survey.region.x, -survey.region.y)
        mask = survey.labels[local_box.slices] == number
        local_y, local_x = np.nonzero(mask)
        target_y = box.y + local_y
        target_x = box.x + local_x
        if bool(protected[target_y, target_x].any()):
            by_finding.append((number, []))
            diagnostics[number] = {"origins": 0, "static_clean": 0, "candidates": 0}
            # Defer construction until the reporting pass by marking it through
            # a dedicated sentinel candidate list is awkward; record reason now.
            diagnostics[number]["target_protected"] = True
            continue

        pad = ring_width
        window_h, window_w = box.h + 2 * pad, box.w + 2 * pad
        window = np.zeros((window_h, window_w), dtype=bool)
        window[pad:pad + box.h, pad:pad + box.w] = mask
        outer = ndimage.binary_dilation(window, structure=STRUCTURE_8, iterations=pad) & ~window
        outer_y, outer_x = np.nonzero(outer)
        target_ring_y = box.y - pad + outer_y
        target_ring_x = box.x - pad + outer_x

        adjacent = ndimage.binary_dilation(window, structure=STRUCTURE_8, iterations=1) & ~window
        adj_y, adj_x = np.nonzero(adjacent)
        target_adj_y = box.y - pad + adj_y
        target_adj_x = box.x - pad + adj_x

        inner = mask & ~ndimage.binary_erosion(mask, structure=STRUCTURE_8, border_value=0)
        inner_y, inner_x = np.nonzero(inner)

        origins = []
        if specs:
            for donor_name, donor in specs:
                if donor.w < box.w or donor.h < box.h:
                    continue
                y_choices = list(range(donor.y, donor.bottom - box.h + 1))
                x_choices = list(range(donor.x, donor.right - box.w + 1))
                stride_y = max(1, int(math.ceil(math.sqrt(max(1, len(y_choices) * len(x_choices)) / 64))))
                stride_x = stride_y
                for origin_y in y_choices[::stride_y]:
                    for origin_x in x_choices[::stride_x]:
                        origins.append((origin_x, origin_y, donor_name))
        else:
            min_dx = max(-search_radius, -box.x)
            max_dx = min(search_radius, width - box.right)
            min_dy = max(-search_radius, -box.y)
            max_dy = min(search_radius, height - box.bottom)
            for delta_y in range(min_dy, max_dy + 1):
                for delta_x in range(min_dx, max_dx + 1):
                    origins.append((box.x + delta_x, box.y + delta_y, "same-scan-grain"))

        candidates = []
        diag = {"origins": len(origins), "static_clean": 0, "ring_rejected": 0, "edge_rejected": 0}
        ring_required = max(4, int(math.ceil(0.5 * len(outer_y))))
        for origin_x, origin_y, donor_name in origins:
            if origin_x < 0 or origin_y < 0 or origin_x + box.w > width or origin_y + box.h > height:
                continue
            source_y = origin_y + local_y
            source_x = origin_x + local_x
            if not clean[source_y, source_x].all():
                continue
            diag["static_clean"] += 1

            source_ring_y = origin_y - pad + outer_y
            source_ring_x = origin_x - pad + outer_x
            if (source_ring_y.min() < 0 or source_ring_x.min() < 0
                    or source_ring_y.max() >= height or source_ring_x.max() >= width
                    or target_ring_y.min() < 0 or target_ring_x.min() < 0
                    or target_ring_y.max() >= height or target_ring_x.max() >= width):
                continue
            target_ring_ok = clean[target_ring_y, target_ring_x]
            source_ring_ok = clean[source_ring_y, source_ring_x]
            common = target_ring_ok & source_ring_ok
            if int(common.sum()) < ring_required:
                continue
            target_ring_density = luma_density[target_ring_y[common], target_ring_x[common]]
            source_ring_density = luma_density[source_ring_y[common], source_ring_x[common]]
            ring_difference = abs(float(np.median(source_ring_density) - np.median(target_ring_density)))
            if ring_difference > ring_limit + 1e-12:
                diag["ring_rejected"] += 1
                continue

            target_adj_ok = clean[target_adj_y, target_adj_x]
            if int(target_adj_ok.sum()) >= 4 and len(inner_y):
                source_inner_density = luma_density[origin_y + inner_y, origin_x + inner_x]
                target_outer_density = luma_density[target_adj_y[target_adj_ok], target_adj_x[target_adj_ok]]
                seam_difference = abs(float(np.median(source_inner_density) - np.median(target_outer_density)))
            else:
                seam_difference = ring_difference
            candidates.append({
                "finding": number,
                "donor_name": donor_name,
                "source_box": Box(int(origin_x), int(origin_y), box.w, box.h),
                "offset_x": int(origin_x - box.x),
                "offset_y": int(origin_y - box.y),
                "pixel_count": int(finding.pixels),
                "pixels": (source_y * width + source_x).astype(np.int64),
                "ring_difference": round(ring_difference, 6),
                "seam_difference": round(seam_difference, 6),
            })

        candidates.sort(key=lambda item: (
            item["seam_difference"], item["ring_difference"], item["source_box"].y,
            item["source_box"].x, item["donor_name"]
        ))
        # Named candidates are operationally important even if not locally best.
        if specs:
            retained = candidates[:max(candidate_limit, len(specs) * 4)]
        else:
            retained = candidates[:candidate_limit]
        diag["candidates"] = len(retained)
        diagnostics[number] = diag
        by_finding.append((number, retained))

    chosen = _optimize_donor_assignment(by_finding)
    repaired = source.values.copy()
    assignments = []
    manual = []
    used_pixels = 0
    for number in sorted(selected):
        candidate = chosen.get(number)
        finding = next(item for item in survey.findings if item.number == number)
        if candidate is None:
            diag = diagnostics[number]
            if diag.get("target_protected"):
                reason = "selected-patch-overlaps-keep"
                message = "The repair mask intersects a protected Keep/reference region."
            elif diag["static_clean"] == 0:
                reason = "no-clean-source-pixels"
                message = "No donor footprint is wholly valid, dust-free and outside Keep."
            elif diag["candidates"] == 0 or diag["ring_rejected"]:
                reason = "outer-ring-density-limit"
                message = "Clean donors exist, but none matches the outer ring within the limit."
            else:
                reason = "donor-pixel-conflict"
                message = "The only suitable source pixels are reserved for another dust block."
            manual.append({
                "number": number, "status": "manual", "needs_manual": True,
                "adoptable": False, "reason": reason, "message": message,
                "pixels": int(finding.pixels), "diagnostics": diag,
            })
            continue

        local_box = finding.box.translate(-survey.region.x, -survey.region.y)
        target_mask = survey.labels[local_box.slices] == number
        target_y, target_x = np.nonzero(target_mask)
        target_y += survey.region.y + local_box.y
        target_x += survey.region.x + local_box.x
        donor = candidate["source_box"]
        source_y = donor.y + (target_y - finding.box.y)
        source_x = donor.x + (target_x - finding.box.x)
        repaired[target_y, target_x] = source.values[source_y, source_x]
        used_pixels += int(finding.pixels)
        source_origin = {"x": donor.x, "y": donor.y}
        assignments.append({
            "number": number,
            "finding": number,
            "status": "restored",
            "method": "measured-grain-copy",
            "donor_name": candidate["donor_name"],
            "source_name": candidate["donor_name"],
            "name": candidate["donor_name"],
            "donor": candidate["donor_name"],
            "source_frame": candidate["donor_name"],
            "frame_name": candidate["donor_name"],
            "source_box": donor.as_dict(),
            "donor_box": donor.as_dict(),
            "source": donor.as_dict(),
            "frame": donor.as_dict(),
            "source_origin": source_origin,
            "offset": {"x": candidate["offset_x"], "y": candidate["offset_y"]},
            "pixels": int(finding.pixels),
            "ring_density_difference": candidate["ring_difference"],
            "seam_density_difference": candidate["seam_difference"],
            "edge_density_difference": candidate["seam_difference"],
        })

    seams = [item["seam_density_difference"] for item in assignments]
    rings = [item["ring_density_difference"] for item in assignments]
    selected_pixels = int(sum(item.pixels for item in survey.findings if item.number in selected))
    plan = {
        "strategy": "measured-grain-copy",
        "version": 1,
        "one_donor_per_finding": True,
        "unique_source_pixels": True,
        "ring_limit": round(ring_limit, 5),
        "edge_tolerance": round(edge_tolerance, 5),
        "donor_radius": search_radius,
        "assignments": assignments,
        "donor_assignments": assignments,
        "manual_findings": manual,
        "manual": manual,
        "used_source_pixels": used_pixels,
        "source_pixel_count": used_pixels,
    }
    summary = {
        "selected_findings": len(selected),
        "restored_findings": len(assignments),
        "repaired_count": len(assignments),
        "restored_count": len(assignments),
        "manual_findings": len(manual),
        "manual_count": len(manual),
        "selected_pixels": selected_pixels,
        "restored_pixels": used_pixels,
        "unfilled_pixels": selected_pixels - used_pixels,
        "worst_seam_density_difference": round(max(seams), 5) if seams else None,
        "minimum_max_seam_difference": round(max(seams), 5) if seams else None,
        "median_seam_density_difference": round(float(np.median(seams)), 5) if seams else None,
        "worst_ring_density_difference": round(max(rings), 5) if rings else None,
        "ring_limit": round(ring_limit, 5),
        "edge_tolerance": round(edge_tolerance, 5),
        "ring_within_limit": len(assignments),
        "seam_within_tolerance": int(sum(value <= edge_tolerance + 1e-12 for value in seams)),
    }
    worst_seam = summary["worst_seam_density_difference"]
    summary["objectives"] = {
        "repair_count": {"secured": len(assignments), "selected": len(selected), "manual": len(manual)},
        "seam_density": {
            "secured_worst_difference": worst_seam,
            "limit": round(edge_tolerance, 5),
            "margin": None if worst_seam is None else round(edge_tolerance - worst_seam, 5),
            "findings_within_limit": summary["seam_within_tolerance"],
        },
    }
    summary["preserved"] = {
        "blocks_repaired": len(assignments),
        "source_grain_pixels": used_pixels,
        "seam_budget_remaining": summary["objectives"]["seam_density"]["margin"],
    }
    plan["summary"] = summary
    headline = (
        f"Adoptable measured-grain plan: {len(assignments)}/{len(selected)} blocks repaired; "
        f"{len(manual)} manual; worst seam D {summary['worst_seam_density_difference']}."
    )
    plan["headline"] = headline
    report = {
        "region": survey.region.as_dict(),
        "restored_findings": sorted(item["number"] for item in assignments),
        "manual_findings": manual,
        "manual": [item["number"] for item in manual],
        "selected_pixels": selected_pixels,
        "restored_pixels": used_pixels,
        "unfilled_pixels": selected_pixels - used_pixels,
        "radius": radius,
        "method": "measured-grain-copy",
        "ring_limit": round(ring_limit, 5),
        "edge_tolerance": round(edge_tolerance, 5),
        "plan": plan,
        "donor_plan": plan,
        "assignments": assignments,
        "summary": summary,
        "objectives": summary["objectives"],
        "preserved": summary["preserved"],
        "secured_count": len(assignments),
        "repaired_count": len(assignments),
        "restored_count": len(assignments),
        "auto_restored_count": len(assignments),
        "manual_count": len(manual),
        "maximum_repairs": len(assignments),
        "best_worst_seam_difference": worst_seam,
        "adoptable": bool(assignments),
        "headline": headline,
    }
    return rgb_image(repaired[survey.region.slices]), report


def restore_region(source, survey, raw_findings, radius=12, ring_limit=None, *extra, **kwargs):
    selected = _restore_selection(survey, raw_findings)
    # Accept both the historical keyword-heavy call style and a positional
    # options/limit argument used by focused planner tests.
    options = dict(kwargs.pop("request", None) if isinstance(kwargs.get("request"), dict) else {})
    options.update(kwargs)
    if ring_limit is not None:
        options.setdefault("ring_limit", ring_limit)
    donor_boxes = kwargs.get("donor_boxes", kwargs.get("donors", kwargs.get("source_boxes")))
    protected_boxes = None
    for key in ("protected_boxes", "keep_boxes", "keep_regions", "protected_regions", "avoid_boxes", "keep"):
        if kwargs.get(key) is not None:
            protected_boxes = kwargs[key]
            break
    if protected_boxes is None:
        protected_boxes = ()
    positional_sources = [item for item in extra if not isinstance(item, (dict, int, float)) or isinstance(item, bool)]
    for positional in extra:
        if isinstance(positional, dict):
            options.update(positional)
            if donor_boxes is None:
                donor_boxes = positional.get("donor_boxes", positional.get("donors"))
            if not protected_boxes:
                protected_boxes = positional.get("protected_boxes", positional.get("keep_boxes", ()))
        elif isinstance(positional, (int, float)) and not isinstance(positional, bool):
            options.setdefault("ring_limit", positional)
    if donor_boxes is None and positional_sources:
        donor_boxes = positional_sources[0]
    if not protected_boxes and len(positional_sources) > 1:
        protected_boxes = positional_sources[1]

    maximum_size = survey.region.area * .08
    target = np.isin(survey.labels, list(selected))
    if target.sum() > maximum_size:
        raise DeskError("The selected dust exceeds eight percent of the region. Review a smaller group.", field="findings")
    return plan_grain_restoration(
        source, survey, selected, radius=radius, protected_boxes=protected_boxes,
        protected_mask=kwargs.get("protected_mask", kwargs.get("keep_mask")),
        donor_boxes=donor_boxes, options=options,
    )


def restored_values_from_plan(source, survey, repair):
    """Reconstruct an exact preview from its compact donor assignment plan."""
    if repair is None:
        return None
    if hasattr(repair, "values") and hasattr(repair, "density"):
        return np.asarray(repair.values)
    if isinstance(repair, np.ndarray):
        return repair
    if not isinstance(repair, dict):
        return None
    if isinstance(repair.get("repaired"), np.ndarray):
        return repair["repaired"]
    if isinstance(repair.get("values"), np.ndarray):
        return repair["values"]
    plan = repair.get("plan", repair.get("repair_plan", repair.get("donor_plan", repair)))
    if not isinstance(plan, dict):
        return None
    assignments = plan.get("assignments") or plan.get("donor_assignments")
    if assignments is None and isinstance(plan.get("result"), dict):
        return restored_values_from_plan(source, survey, plan["result"])
    if not isinstance(assignments, list):
        return None
    repaired = np.asarray(source.values).copy()
    width = source.size[0] if hasattr(source, "size") else repaired.shape[1]
    for raw_assignment in assignments:
        if not isinstance(raw_assignment, dict):
            continue
        status = raw_assignment.get("status", "restored")
        if status not in {"restored", "applied", "adopted"}:
            continue
        number = raw_assignment.get("number", raw_assignment.get("finding"))
        if number is None:
            continue
        number = int(number)
        finding = next((item for item in survey.findings if item.number == number), None)
        if finding is None:
            continue
        origin = raw_assignment.get("source_origin") or raw_assignment.get("origin")
        source_box = raw_assignment.get("source_box") or raw_assignment.get("donor_box") or raw_assignment.get("source")
        if origin is None and isinstance(source_box, dict):
            origin = {"x": source_box.get("x"), "y": source_box.get("y")}
        if origin is None and isinstance(raw_assignment.get("offset"), dict):
            origin = {
                "x": finding.box.x + raw_assignment["offset"].get("x", 0),
                "y": finding.box.y + raw_assignment["offset"].get("y", 0),
            }
        if not isinstance(origin, dict) or origin.get("x") is None or origin.get("y") is None:
            continue
        origin_x, origin_y = int(round(float(origin["x"]))), int(round(float(origin["y"])))
        local_box = finding.box.translate(-survey.region.x, -survey.region.y)
        target_y, target_x = np.nonzero(survey.labels[local_box.slices] == number)
        absolute_y = target_y + survey.region.y + local_box.y
        absolute_x = target_x + survey.region.x + local_box.x
        source_y = origin_y + (absolute_y - finding.box.y)
        source_x = origin_x + (absolute_x - finding.box.x)
        if (source_y.min() < 0 or source_x.min() < 0
                or source_y.max() >= repaired.shape[0] or source_x.max() >= width):
            continue
        repaired[absolute_y, absolute_x] = source.values[source_y, source_x]
    return repaired


def _density_from_values(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return -np.log10(np.maximum(array, BLACK_FLOOR))


def _restored_density(source, survey, repair, kwargs):
    explicit_density = kwargs.get("repaired_density")
    if explicit_density is None:
        explicit_density = kwargs.get("restored_density")
    if explicit_density is not None:
        array = np.asarray(explicit_density, dtype=float)
        return np.repeat(array[..., None], 3, axis=2) if array.ndim == 2 else array
    explicit_values = kwargs.get("repaired_values")
    if explicit_values is None:
        explicit_values = kwargs.get("restored_values")
    if explicit_values is not None:
        return _density_from_values(explicit_values)
    if repair is None:
        repair = _first_option(
            kwargs, ("repaired", "restored", "repair_plan", "restore_plan", "restoration"))
    if repair is None:
        return None
    if hasattr(repair, "density"):
        return np.asarray(repair.density, dtype=float)
    if isinstance(repair, np.ndarray):
        # Direct array callers are working in the same density domain as
        # source.density.  Normalized image values are accepted through
        # repaired_values= or through a donor plan.
        array = np.asarray(repair, dtype=float)
        return np.repeat(array[..., None], 3, axis=2) if array.ndim == 2 else array
    values = restored_values_from_plan(source, survey, repair)
    return None if values is None else _density_from_values(values)


def _restored_validity(source, repair, kwargs):
    explicit = kwargs.get("repaired_valid")
    if explicit is None:
        explicit = kwargs.get("restored_valid")
    if explicit is not None:
        valid = np.asarray(explicit, dtype=bool)
    elif repair is not None and hasattr(repair, "valid"):
        valid = np.asarray(repair.valid, dtype=bool)
    else:
        valid = np.asarray(source.valid, dtype=bool)
    return valid.all(axis=2) if valid.ndim == 3 else valid


def marked_survey_image(source, survey, *, selected=None, maximum=1500):
    region = survey.region
    output = rgb_image(source.values[region.slices])
    output.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    drawer = ImageDraw.Draw(output)
    sx = output.width / region.w
    sy = output.height / region.h
    for finding in survey.findings:
        bounds = finding.box.translate(-region.x, -region.y)
        frame = round(bounds.x * sx), round(bounds.y * sy), round(bounds.right * sx), round(bounds.bottom * sy)
        color = "#ffc975" if finding.number == selected else "#df547d" if finding.polarity == "dense" else "#8fe8dd"
        drawer.rectangle(frame, outline=color, width=2 if finding.number == selected else 1)
        if finding.protected:
            x, y, xx, yy = frame
            drawer.line((x, y, xx, yy), fill=color, width=1)
    return output


def grain_estimate(calibrated, region, scales):
    if min(region.w, region.h) < 10 or region.area > DEFECT_PIXEL_LIMIT:
        raise DeskError("Choose a textured region of at least 10 × 10 and at most two million pixels.", field="region")
    gray = np.einsum("ijk,k->ij", calibrated.density[region.slices], SCANNER_LUMA)
    eligible = calibrated.valid[region.slices].all(axis=2)
    if eligible.mean() < .7:
        raise DeskError("Choose a patch with at least 70% valid measurements.", field="region")
    filled = _fill_holes(gray, eligible)
    outputs = []
    for radius in scales:
        if radius == 0:
            continue
        local = ndimage.gaussian_filter(filled, sigma=radius, mode="reflect")
        core = ndimage.binary_erosion(eligible, iterations=radius)
        if not core.any():
            outputs.append({"radius": radius, "samples": 0, "mad": None, "p95": None})
            continue
        residual = (gray - local)[core]
        median = np.median(residual)
        absolute = np.abs(residual - median)
        outputs.append(
            {
                "radius": radius,
                "samples": int(len(residual)),
                "mad": round(float(np.median(absolute)), 6),
                "p95": round(float(np.quantile(absolute, .95)), 6),
            }
        )
    return {"region": region.as_dict(), "coverage": round(float(eligible.mean()), 5), "bands": outputs}


# Descriptive aliases for callers/tests that name the planning/edge stages
# explicitly rather than going through the two historical operation functions.
plan_measured_grain = plan_grain_restoration
choose_grain_donors = plan_grain_restoration
plan_restoration = plan_grain_restoration
allocate_disjoint_donors = _optimize_donor_assignment
choose_donor_assignment = _optimize_donor_assignment
compare_restored_edges = restored_edge_report
compare_restored_seams = restored_edge_report
check_restored_edges = local_survey_comparison
restored_seam_report = restored_edge_report
restored_edge_comparison = local_survey_comparison
edge_comparison = restored_edge_report
