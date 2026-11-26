"""Frame mapping and physical cut checks on an upright film strip."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from .imaging import SCANNER_LUMA
from .values import (
    Box, DeskError, MARK_KINDS, Point, Segment, clean_text, mapping,
    number, one_of, rgb_image,
)


BOX_MARKS = frozenset({"frame", "dust", "scratch", "perforation", "reference", "keep"})
CUT_SIDES = frozenset({"left", "right", "both"})
MARK_COLORS = {
    "frame": "#edd9a8",
    "cut": "#f1a970",
    "dust": "#f17196",
    "scratch": "#d6a3ff",
    "perforation": "#8ddacc",
    "reference": "#9fc3fc",
    "keep": "#a1e7a5",
}
STOCK_PITCH_MM = {"135": 4.75, "120": None, "other": None}


def read_mark(raw, scan, current=None):
    payload = mapping(raw, "mark")
    current = current or {}
    kind = one_of(payload.get("kind", current.get("kind")), MARK_KINDS, "kind")
    geometry = payload.get("geometry", current.get("geometry"))
    result = {"kind": kind, "geometry": read_geometry(geometry, kind, (scan["width"], scan["height"]))}
    result["label"] = clean_text(payload.get("label", current.get("label")), "label", maximum=80, required=True)
    result["notes"] = clean_text(payload.get("notes", current.get("notes")), "notes", maximum=1000)
    return result


def read_geometry(geometry, kind, size):
    if kind in BOX_MARKS:
        box = Box.read(geometry, "geometry", bounds=size)
        if kind in ("frame", "keep") and min(box.w, box.h) < 3:
            raise DeskError("Draw a box at least three pixels wide and tall.", field="geometry")
        return box.as_dict()
    return Segment.read(geometry, "geometry", bounds=size).as_dict()


def mark_boxes(marks, kinds, size):
    return [
        Box.read(mark["geometry"], "geometry", bounds=size)
        for mark in marks if mark["kind"] in kinds
    ]


@dataclass(frozen=True)
class CutSettings:
    guide: Box
    kerf_mm: float
    reserve_mm: float
    dpi: float
    retained: str

    @classmethod
    def read(cls, raw, scan):
        value = mapping(raw, "cut")
        guide = Box.read(value.get("guide"), "cut.guide", bounds=(scan["width"], scan["height"]))
        dpi = number(value.get("dpi", scan["dpi"]), "cut.dpi", lower=30, upper=20_000)
        return cls(
            guide,
            number(value.get("kerf_mm", .16), "cut.kerf_mm", lower=.015, upper=4),
            number(value.get("reserve_mm", .7), "cut.reserve_mm", lower=0, upper=12),
            dpi,
            one_of(value.get("retained", "both"), CUT_SIDES, "cut.retained"),
        )

    @property
    def pixels_per_mm(self):
        return self.dpi / 25.4

    @property
    def exclusion_pixels(self):
        return (self.kerf_mm / 2 + self.reserve_mm) * self.pixels_per_mm

    def as_dict(self):
        return {
            "guide": self.guide.as_dict(),
            "kerf_mm": self.kerf_mm,
            "reserve_mm": self.reserve_mm,
            "dpi": self.dpi,
            "retained": self.retained,
        }


def _corners(box):
    return (
        Point(box.x, box.y),
        Point(box.right, box.y),
        Point(box.x, box.bottom),
        Point(box.right, box.bottom),
    )


def _distance_from_rect(segment, box):
    if segment.crosses_box(box):
        return 0.0
    from_segment = [segment.distance_to(corner) for corner in _corners(box)]
    from_box = []
    for endpoint in (segment.start, segment.end):
        x = float(np.clip(endpoint.x, box.x, box.right))
        y = float(np.clip(endpoint.y, box.y, box.bottom))
        from_box.append(endpoint.distance(Point(x, y)))
    return min(from_segment + from_box)


def _side_of_cut(segment, point):
    dy = segment.end.y - segment.start.y
    if abs(dy) < 1e-9:
        return "left" if point.x < segment.start.x else "right"
    at_y = segment.start.x + (point.y - segment.start.y) / dy * (segment.end.x - segment.start.x)
    return "left" if point.x < at_y else "right"


def _inside_guide(line, guide):
    low_y, high_y = sorted((line.start.y, line.end.y))
    if low_y > guide.y + 1 or high_y < guide.bottom - 1:
        return False
    xs = line.start.x, line.end.x
    return all(guide.x <= x <= guide.right for x in xs)


def assess_cut(line, marks, settings):
    crossed = []
    edges = []
    outside = not _inside_guide(line, settings.guide)
    clearance = None
    for mark in marks:
        if mark["kind"] not in {"frame", "keep", "perforation", "reference"}:
            continue
        box = Box.read(mark["geometry"])
        if settings.guide.intersect(box) is None:
            continue
        distance = _distance_from_rect(line, box)
        clearance_here = distance / settings.pixels_per_mm - settings.kerf_mm / 2
        clearance = clearance_here if clearance is None else min(clearance, clearance_here)
        side = _side_of_cut(line, box.center)
        reserve_is_required = settings.retained in (side, "both") or mark["kind"] in {"keep", "reference"}
        crosses_image = distance <= settings.kerf_mm / 2 * settings.pixels_per_mm
        within_reserve = reserve_is_required and distance < settings.exclusion_pixels
        item = {
            "mark_id": mark["id"],
            "label": mark["label"],
            "kind": mark["kind"],
            "side": side,
            "clearance_mm": round(clearance_here, 4),
            "crosses": crosses_image,
            "within_reserve": within_reserve,
        }
        if crosses_image or within_reserve:
            crossed.append(item)
        else:
            edges.append(item)
    return {
        "line": line.as_dict(),
        "status": "outside-guide" if outside else "interferes" if crossed else "clear",
        "min_clearance_mm": round(clearance, 4) if clearance is not None else None,
        "conflicts": crossed,
        "clear_marks": edges,
        "has_reference": bool(crossed or edges),
        "cut_length_mm": round(line.length / settings.pixels_per_mm, 4),
    }


def _vertical_line(col, guide):
    return Segment(Point(float(col), float(guide.y)), Point(float(col), float(guide.bottom - 1)))


def propose_cut_positions(array, marks, settings, max_positions, *, guide_signal=True):
    guide = settings.guide
    keep_marks = [
        mark for mark in marks
        if mark["kind"] in {"frame", "keep", "reference"} and guide.intersect(Box.read(mark["geometry"]))
    ]
    ranges = sorted(
        (max(guide.x, box.x), min(guide.right, box.right))
        for box in (Box.read(mark["geometry"]) for mark in keep_marks)
    )
    holes = []
    position = guide.x
    for left, right in ranges:
        if position < left:
            holes.append((position, left))
        position = max(position, right)
    if position < guide.right:
        holes.append((position, guide.right))
    gray = np.einsum("ijk,k->ij", array[guide.slices], SCANNER_LUMA)
    strip_signal = np.median(gray, axis=0)
    gradient = np.abs(np.gradient(ndimage.gaussian_filter1d(strip_signal, sigma=1.8))) if guide.w > 2 else np.zeros(guide.w)
    candidates = []
    for start, end in holes:
        width = end - start
        reserve = math.ceil(settings.exclusion_pixels)
        if width < max(reserve * 2 + 1, 3):
            continue
        center = (start + end) / 2
        valid_columns = np.arange(start + reserve, end - reserve)
        distances = np.abs(valid_columns - center) / width
        appearance = gradient[valid_columns - guide.x]
        cost = distances + (appearance * 1.4 if guide_signal else 0)
        for rank in np.argsort(cost, kind="stable")[:max_positions]:
            column = int(valid_columns[rank])
            line = _vertical_line(column, guide)
            report = assess_cut(line, marks, settings)
            if report["status"] != "clear":
                continue
            candidates.append(
                {
                    "column": column,
                    "gap": {"left": start, "right": end},
                    "distance_from_guide_left_mm": round((column - guide.x) / settings.pixels_per_mm, 4),
                    "scan_gradient": round(float(appearance[rank]), 6),
                    "clearance_mm": report["min_clearance_mm"],
                    "line": report["line"],
                }
            )
            break
    return {
        "candidates": candidates[:max_positions],
        "guide": guide.as_dict(),
        "gaps_reviewed": len(holes),
        "reference_marks": len(keep_marks),
        "reason": "no-clear-gap" if not candidates else None,
    }


def verify_cut(planned, actual, marks, settings, tolerance):
    plan = assess_cut(planned, marks, settings)
    observed = assess_cut(actual, marks, settings)
    error = max(planned.distance_to(actual.start), planned.distance_to(actual.end)) / settings.pixels_per_mm
    far = error > tolerance
    if observed["status"] != "clear":
        status = "hold-clearance"
    elif far:
        status = "hold-offset"
    else:
        status = "confirmed"
    return {
        "status": status,
        "offset_mm": round(error, 4),
        "allowed_offset_mm": tolerance,
        "plan": plan,
        "observed": observed,
    }


def _edge_band(gray, region, edge, fraction):
    rows = max(2, round(region.h * fraction))
    if edge == "top":
        band = gray[:rows]
    else:
        band = gray[-rows:]
    return np.quantile(band, .84, axis=0)


def _perforation_peaks(trace, minimum_px, contrast):
    window = max(3, int(round(minimum_px * .5)))
    filtered = ndimage.gaussian_filter1d(trace, sigma=max(.7, minimum_px * .07))
    local_baseline = ndimage.minimum_filter1d(filtered, size=2 * window + 1)
    response = filtered - local_baseline
    peaks, props = signal.find_peaks(response, prominence=contrast, distance=max(2, minimum_px * .5))
    return peaks, props.get("prominences", np.array([]))


def perforation_survey(decoded, region, minimum_pitch, contrast, fraction, *, stock="135"):
    if region.w < 40 or region.h < 14:
        raise DeskError("Select a long film edge with room for several holes.", field="region")
    raw = decoded.values[region.slices]
    gray = np.einsum("ijk,k->ij", raw, SCANNER_LUMA)
    edge_results = {}
    intervals = []
    for edge in ("top", "bottom"):
        trace = _edge_band(gray, region, edge, fraction)
        peaks, prominence = _perforation_peaks(trace, minimum_pitch, contrast)
        interval = np.diff(peaks)
        if len(interval):
            intervals.extend(interval.tolist())
        edge_results[edge] = {
            "peaks": [
                {"x": int(region.x + point), "prominence": round(float(weight), 5)}
                for point, weight in zip(peaks, prominence)
            ],
            "intervals_px": interval.tolist(),
        }
    median = float(np.median(intervals)) if intervals else None
    pitch = STOCK_PITCH_MM[stock]
    if median and pitch:
        estimate = round(median / pitch * 25.4, 3)
    else:
        estimate = None
    return {
        "region": region.as_dict(),
        "edges": edge_results,
        "pitch_px": round(median, 4) if median else None,
        "dpi_estimate": estimate,
        "nominal_pitch_mm": pitch,
        "pitch_mad_px": round(float(np.median(np.abs(np.asarray(intervals) - median))), 4) if median else None,
    }


def suggest_frame_boxes(decoded, region, minimum_width, separation, contrast):
    if minimum_width * 2 + separation > region.w:
        raise DeskError("Select a strip long enough for at least two candidate frames.", field="region")
    data = decoded.values[region.slices]
    gray = np.einsum("ijk,k->ij", data, SCANNER_LUMA)
    vertical_core = gray[max(0, region.h // 8):max(1, region.h * 7 // 8)]
    top = np.quantile(vertical_core, .85, axis=0)
    bottom = np.quantile(vertical_core, .15, axis=0)
    activity = ndimage.gaussian_filter1d(top - bottom, sigma=max(1, separation / 7))
    quiet = activity < contrast
    labels, _ = ndimage.label(quiet)
    seams = []
    for bounds in ndimage.find_objects(labels):
        if not bounds:
            continue
        span = bounds[0]
        if span.stop - span.start >= separation:
            seams.append((span.start, span.stop))
    boundaries = [(0, 0), *seams, (region.w, region.w)]
    candidates = []
    for (_, first_right), (second_left, _) in zip(boundaries, boundaries[1:]):
        width = second_left - first_right
        if width < minimum_width:
            continue
        box = Box(region.x + first_right, region.y, width, region.h)
        candidates.append(
            {
                "box": box.as_dict(),
                "texture_contrast": round(float(np.median(activity[first_right:second_left])), 5),
            }
        )
    return {
        "region": region.as_dict(),
        "frames": candidates,
        "gaps": [{"left": region.x + start, "right": region.x + end} for start, end in seams],
    }


def marked_strip_image(decoded, source_region, marks, *, guides=None, max_side=1600):
    output = rgb_image(decoded.values[source_region.slices])
    output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    drawer = ImageDraw.Draw(output)
    sx = output.width / source_region.w
    sy = output.height / source_region.h

    def project(point):
        return round((point.x - source_region.x) * sx), round((point.y - source_region.y) * sy)

    for mark in marks:
        color = MARK_COLORS[mark["kind"]]
        if mark["kind"] == "cut":
            line = Segment.read(mark["geometry"])
            drawer.line((project(line.start), project(line.end)), fill=color, width=2)
            continue
        box = Box.read(mark["geometry"])
        if box.intersect(source_region) is None:
            continue
        left, top = project(Point(box.x, box.y))
        right, bottom = project(Point(box.right, box.bottom))
        drawer.rectangle((left, top, right, bottom), outline=color, width=2)
        drawer.text((left + 3, top + 3), mark["label"][:28], fill=color, stroke_width=2, stroke_fill="#24211e")
    for line, color in guides or ():
        drawer.line((project(line.start), project(line.end)), fill=color, width=3)
    return output
