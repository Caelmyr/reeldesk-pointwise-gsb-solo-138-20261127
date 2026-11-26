"""Local scan-defect survey with protected image regions and repeat-pass checks."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .imaging import SCANNER_LUMA
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


def local_survey_comparison(source, peer, transform, survey, number, margin, difference_limit):
    finding = next((finding for finding in survey.findings if finding.number == number), None)
    if finding is None:
        raise DeskError("Select a current dust finding.", field="finding")
    margin_box = finding.box.expanded(margin, *source.size)
    focus = margin_box.intersect(survey.region)
    yy, xx = np.mgrid[focus.y:focus.bottom, focus.x:focus.right]
    label_area = focus.translate(-survey.region.x, -survey.region.y)
    original_mask = survey.labels[label_area.slices] == number
    neutral = survey.labels[label_area.slices] == 0
    source_density = source.density[focus.slices]
    peer_density, peer_usable = sample_peer(peer.density, peer.valid, transform, xx, yy)
    joint = source.valid[focus.slices] & peer_usable
    delta = np.einsum("ijk,k->ij", source_density - peer_density, SCANNER_LUMA)
    complete = joint.all(axis=2)
    interior = original_mask & complete
    ring = neutral & complete
    if ring.sum() < max(8, finding.pixels // 2) or interior.sum() < finding.pixels * .65:
        return {
            "finding": number, "region": focus.as_dict(), "verdict": "insufficient",
            "interior": int(interior.sum()), "ring": int(ring.sum()),
            "median_shift": None, "local_difference": None,
        }
    baseline = float(np.median(delta[ring]))
    local_difference = float(np.median(delta[interior])) - baseline
    ring_residual = float(np.median(np.abs(delta[ring] - baseline)))
    if ring_residual > difference_limit * .75:
        verdict = "registration-review"
    elif abs(local_difference) >= difference_limit:
        verdict = "changes-between-passes"
    else:
        verdict = "persists-in-peer"
    center = transform.project(finding.box.center)
    return {
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


def _restore_selection(survey, raw_findings):
    requested = sequence(raw_findings, "findings", minimum=1, maximum=50)
    selected = set()
    seen = set()
    by_number = {finding.number: finding for finding in survey.findings}
    for index, value in enumerate(requested):
        field = f"findings[{index}]"
        value = mapping(value, field)
        number = integer(value.get("number"), f"{field}.number", lower=1, upper=200)
        state = one_of(value.get("state"), REVIEW_STATES, f"{field}.state")
        finding = by_number.get(number)
        if finding is None:
            raise DeskError("Choose findings from the current survey.", field=field)
        if number in seen:
            raise DeskError("Each finding may be selected once.", field=field)
        seen.add(number)
        if state == "restore":
            if finding.protected or finding.edge_contact:
                raise DeskError("A protected or cropped finding cannot be automatically retouched.", field=field)
            selected.add(number)
    if not selected:
        raise DeskError("Choose at least one finding to restore.", field="findings")
    return selected


def restore_region(source, survey, raw_findings, radius):
    region = survey.region
    source_values = source.values[region.slices]
    source_valid = source.valid[region.slices].all(axis=2)
    selected = _restore_selection(survey, raw_findings)
    maximum_size = region.area * .08
    target = np.isin(survey.labels, list(selected))
    if target.sum() > maximum_size:
        raise DeskError("The selected dust exceeds eight percent of the region. Review a smaller group.", field="findings")
    witnesses = (survey.labels == 0) & source_valid
    repaired = source_values.copy()
    gaps = 0
    for number in sorted(selected):
        finding = survey.findings[number - 1]
        support_box = finding.box.expanded(radius, region.right, region.bottom).intersect(region)
        support_box = support_box.translate(-region.x, -region.y)
        yy, xx = np.mgrid[support_box.y:support_box.bottom, support_box.x:support_box.right]
        use = witnesses[support_box.slices]
        ys = yy[use]
        xs = xx[use]
        if len(ys) < max(12, int(finding.pixels * .2)):
            gaps += finding.pixels
            continue
        median_distance = np.hypot(ys - np.median(ys), xs - np.median(xs))
        weights = 1 / np.maximum(1, median_distance / max(1, radius / 3))
        solve_basis = np.column_stack((np.ones(len(xs)), xs - xs.mean(), ys - ys.mean()))
        coefficients, _, rank, _ = np.linalg.lstsq(
            solve_basis * weights[:, None],
            source_values[ys, xs] * weights[:, None],
            rcond=None,
        )
        yy, xx = np.nonzero(survey.labels[support_box.slices] == number)
        yy += support_box.y
        xx += support_box.x
        if rank < 3:
            repaired[yy, xx] = np.median(source_values[ys, xs], axis=0)
            continue
        sample_basis = np.column_stack((np.ones(len(xx)), xx - xs.mean(), yy - ys.mean()))
        predicted = sample_basis @ coefficients
        low, high = np.quantile(source_values[ys, xs], [.025, .975], axis=0)
        repaired[yy, xx] = np.clip(predicted, low, high)
    return rgb_image(repaired), {
        "region": region.as_dict(),
        "restored_findings": sorted(selected),
        "selected_pixels": int(target.sum()),
        "restored_pixels": int(target.sum()) - gaps,
        "unfilled_pixels": gaps,
        "radius": radius,
    }


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
