"""Inspect missing physical film area, clipped channels, and per-frame coverage."""

from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .values import Box, DeskError, PRIMARY_CHANNELS, integer, mapping, number, one_of


COVERAGE_MODES = frozenset({"rgb", "red", "green", "blue"})
COVERAGE_COLORS = {
    "clear": [40, 48, 50],
    "alpha": [118, 89, 153],
    "low": [52, 105, 177],
    "high": [228, 124, 58],
    "controls": [202, 71, 100],
    "partial": [201, 169, 78],
}
COVERAGE_LIMIT = 5_000_000


@dataclass(frozen=True)
class CoverageSettings:
    channel: str
    min_gap: int
    minimum_fraction: float
    max_gap_count: int
    rows: int
    columns: int

    @classmethod
    def read(cls, value):
        value = {} if value is None else mapping(value, "coverage")
        return cls(
            one_of(value.get("channel", "rgb"), COVERAGE_MODES, "coverage.channel"),
            integer(value.get("min_gap", 5), "coverage.min_gap", lower=1, upper=50_000),
            number(value.get("minimum_fraction", .985), "coverage.minimum_fraction", lower=.3, upper=1),
            integer(value.get("max_gap_count", 25), "coverage.max_gap_count", lower=1, upper=100),
            integer(value.get("rows", 4), "coverage.rows", lower=2, upper=20),
            integer(value.get("columns", 8), "coverage.columns", lower=2, upper=28),
        )

    def as_dict(self):
        return {
            "channel": self.channel, "min_gap": self.min_gap,
            "minimum_fraction": self.minimum_fraction, "max_gap_count": self.max_gap_count,
            "rows": self.rows, "columns": self.columns,
        }


@dataclass(frozen=True)
class Coverage:
    result: dict
    classes: np.ndarray


def scan_classes(decoded, corrected, settings, region):
    raw_input = decoded.valid[region.slices]
    eligible = corrected.valid[region.slices]
    low = corrected.low_clip[region.slices]
    high = corrected.high_clip[region.slices]
    overlap = high & low
    suspect_controls = ~eligible & ~high & ~low
    if settings.channel == "rgb":
        eligible_pixel = eligible.all(axis=2)
        some_eligible = eligible.any(axis=2)
        saturated_low = low.any(axis=2)
        saturated_high = high.any(axis=2)
        calibration_gap = suspect_controls.any(axis=2)
        contradictory = overlap.any(axis=2) | (saturated_low & saturated_high)
    else:
        index = {"red": 0, "green": 1, "blue": 2}[settings.channel]
        eligible_pixel = eligible[..., index]
        some_eligible = eligible_pixel
        saturated_low = low[..., index]
        saturated_high = high[..., index]
        calibration_gap = suspect_controls[..., index]
        contradictory = overlap[..., index]
    distinct = np.zeros(raw_input.shape, dtype=np.uint8)
    rejected = ~eligible_pixel & raw_input
    distinct[~raw_input] = 1
    distinct[rejected & saturated_low & ~contradictory] = 2
    distinct[rejected & saturated_high & ~contradictory] = 3
    distinct[rejected & (calibration_gap | contradictory)] = 4
    distinct[rejected & some_eligible] = 5
    return distinct


def _border_length(mask):
    padded = np.pad(mask, (1, 1), mode="constant")
    vertical = np.count_nonzero(padded[1:] != padded[:-1])
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return int(vertical + horizontal)


def _coverage_gaps(classes, region, settings):
    union = classes > 0
    labels, count = ndimage.label(union)
    sizes = np.bincount(labels[union], minlength=count + 1)
    chosen = [index for index in range(1, count + 1) if sizes[index] >= settings.min_gap]
    chosen.sort(key=lambda index: (-sizes[index], index))
    bboxes = ndimage.find_objects(labels)
    entries = []
    reasons = list(COVERAGE_COLORS)
    for label in chosen[:settings.max_gap_count]:
        ys, xs = bboxes[label - 1]
        local = labels[ys, xs] == label
        local_values = classes[ys, xs][local]
        counts = np.bincount(local_values, minlength=len(reasons))
        center_x, center_y = np.mean(np.column_stack(np.nonzero(local)), axis=0)[::-1]
        box = Box(region.x + xs.start, region.y + ys.start, xs.stop - xs.start, ys.stop - ys.start)
        touching = []
        for test, label in ((xs.start == 0, "left"), (xs.stop == region.w, "right"),
                            (ys.start == 0, "top"), (ys.stop == region.h, "bottom")):
            if test:
                touching.append(label)
        entries.append(
            {
                "box": box.as_dict(),
                "center": {"x": round(float(region.x + xs.start + center_x), 2), "y": round(float(region.y + ys.start + center_y), 2)},
                "area": int(local.sum()),
                "boundary_edges": _border_length(local),
                "touches": touching,
                "reason": reasons[int(counts.argmax())],
                "reason_counts": {reason: int(amount) for reason, amount in zip(reasons, counts) if amount > 0},
            }
        )
    return entries, len(chosen)


def _tile_coverage(classes, region, settings):
    vertical = np.linspace(0, region.h, settings.rows + 1).round().astype(int)
    horizontal = np.linspace(0, region.w, settings.columns + 1).round().astype(int)
    if (np.diff(vertical) < 1).any() or (np.diff(horizontal) < 1).any():
        raise DeskError("Reduce the number of coverage tiles for this region.", field="coverage")
    rows = []
    for yi, (y0, y1) in enumerate(zip(vertical, vertical[1:])):
        for xi, (x0, x1) in enumerate(zip(horizontal, horizontal[1:])):
            local = classes[y0:y1, x0:x1]
            bins = np.bincount(local.ravel(), minlength=len(COVERAGE_COLORS))
            coverage = bins[0] / local.size
            reason = list(COVERAGE_COLORS)[int(np.argmax(bins[1:])) + 1] if coverage < 1 else None
            rows.append(
                {
                    "row": yi, "column": xi,
                    "box": Box(region.x + int(x0), region.y + int(y0), int(x1 - x0), int(y1 - y0)).as_dict(),
                    "eligible_fraction": round(float(coverage), 6),
                    "status": "accepted" if coverage >= settings.minimum_fraction else "review",
                    "dominant_gap": reason,
                }
            )
    return rows


def _frame_coverage(classes, region, marks, minimum):
    frames = []
    for mark in marks:
        if mark["kind"] not in {"frame", "keep", "reference"}:
            continue
        source_box = Box.read(mark["geometry"])
        joint = region.intersect(source_box)
        if joint is None:
            continue
        local = joint.translate(-region.x, -region.y)
        measured = classes[local.slices]
        fractions = np.bincount(measured.ravel(), minlength=len(COVERAGE_COLORS)) / measured.size
        coverage = float(fractions[0])
        frames.append(
            {
                "mark_id": mark["id"], "name": mark["label"], "kind": mark["kind"],
                "box": joint.as_dict(), "eligible_fraction": round(coverage, 6),
                "is_full_mark": joint == source_box,
                "status": "partial-area" if joint != source_box else "accepted" if coverage >= minimum else "review",
                "reason_fractions": {name: round(float(fraction), 5) for name, fraction in zip(COVERAGE_COLORS, fractions) if fraction > 0},
            }
        )
    return frames


def measure_coverage(decoded, corrected, region, settings, marks):
    if region.area > COVERAGE_LIMIT:
        raise DeskError("Choose up to five million pixels for the coverage check.", field="region")
    classes = scan_classes(decoded, corrected, settings, region)
    counts = np.bincount(classes.ravel(), minlength=len(COVERAGE_COLORS))
    gaps, gap_count = _coverage_gaps(classes, region, settings)
    rows = _tile_coverage(classes, region, settings)
    channels = {}
    for number, channel in enumerate(PRIMARY_CHANNELS):
        valid = corrected.valid[region.slices][..., number]
        channels[channel] = {"available": int(valid.sum()), "fraction": round(float(valid.mean()), 6)}
    coverage = int(counts[0]) / region.area
    result = {
        "region": region.as_dict(),
        "status": "accepted" if coverage >= settings.minimum_fraction else "review",
        "channel_mode": settings.channel,
        "minimum_fraction": settings.minimum_fraction,
        "eligible_fraction": round(coverage, 6),
        "reason_counts": dict(zip(COVERAGE_COLORS, counts.tolist())),
        "channels": channels,
        "gaps": gaps,
        "gap_count": gap_count,
        "tiles": rows,
        "marks": _frame_coverage(classes, region, marks, settings.minimum_fraction),
    }
    return Coverage(result, classes)


def coverage_map(decoded, coverage, *, maximum_side=1500):
    region = Box.read(coverage.result["region"])
    tile = decoded.values[region.slices]
    valid_values = np.mean(tile, axis=2)
    palette = np.asarray(list(COVERAGE_COLORS.values()), dtype=np.uint8)
    classes = coverage.classes
    rgb = palette[classes].copy()
    visible = classes == 0
    gray = np.clip(np.rint(70 + valid_values * 136), 0, 255).astype(np.uint8)
    rgb[visible] = np.stack((gray[visible], gray[visible], gray[visible]), axis=1)
    image = Image.fromarray(rgb)
    image.thumbnail((maximum_side, maximum_side), Image.Resampling.NEAREST)
    sx = image.width / region.w
    sy = image.height / region.h
    drawer = ImageDraw.Draw(image)
    for gap in coverage.result["gaps"]:
        box = Box.read(gap["box"]).translate(-region.x, -region.y)
        if box.area < 16:
            continue
        xy = (round(box.x * sx), round(box.y * sy), round((box.right - 1) * sx), round((box.bottom - 1) * sy))
        drawer.rectangle(xy, outline="#e8d5b5", width=1)
    return image
