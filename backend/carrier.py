"""Trace a film carrier's two edges and report physical bow and tilt."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .imaging import SCANNER_LUMA
from .values import DeskError, integer, mapping, number, one_of, rgb_image


MAX_CARRIER_AREA = 3_000_000


@dataclass(frozen=True)
class CarrierSettings:
    direction: str
    edge_depth: float
    min_step: float
    station_width: int
    tolerance: float

    @classmethod
    def read(cls, raw):
        value = {} if raw is None else mapping(raw, "carrier")
        return cls(
            one_of(value.get("direction", "horizontal"), {"horizontal", "vertical"}, "carrier.direction"),
            number(value.get("edge_depth", .26), "carrier.edge_depth", lower=.08, upper=.43),
            number(value.get("min_step", .035), "carrier.min_step", lower=.004, upper=.5),
            integer(value.get("station_width", 12), "carrier.station_width", lower=4, upper=80),
            number(value.get("tolerance", .15), "carrier.tolerance", lower=.015, upper=6),
        )

    def as_dict(self):
        return {
            "direction": self.direction, "edge_depth": self.edge_depth, "min_step": self.min_step,
            "station_width": self.station_width, "tolerance": self.tolerance,
        }


def _rail_edge(signal, offset, min_step):
    centered = ndimage.gaussian_filter1d(signal, sigma=1.1, mode="nearest")
    variation = np.diff(centered)
    absolute = np.abs(variation)
    peak = int(np.argmax(absolute))
    if absolute[peak] < min_step or peak in (0, len(absolute) - 1):
        return None, None, None
    left, center, right = absolute[peak - 1:peak + 2]
    denominator = left - 2 * center + right
    corrected = .5 * (left - right) / denominator if denominator < -1e-6 else 0
    return offset + peak + .5 + float(np.clip(corrected, -.5, .5)), float(center), int(np.sign(variation[peak]))


def _station_strips(decoded, region, settings):
    source = np.einsum("ijk,k->ij", decoded.values[region.slices], SCANNER_LUMA)
    valid = decoded.valid[region.slices]
    if settings.direction == "vertical":
        source = source.T
        valid = valid.T
    height, width = source.shape
    band = max(7, round(height * settings.edge_depth))
    if band * 2 >= height or width < settings.station_width * 4:
        raise DeskError("Choose a longer region that includes both rails.", field="region")
    observations = []
    for start in range(0, width - settings.station_width + 1, settings.station_width):
        end = start + settings.station_width
        span = source[:, start:end]
        support = valid[:, start:end]
        column = (start + end - 1) / 2
        if support.mean() < .82:
            observations.append((column, None, None, None, "gaps"))
            continue
        # Medians over a cross-section reduce the impact of holes in perforated film.
        median = np.median(np.where(support, span, np.median(span[support])), axis=1)
        first, first_strength, first_sign = _rail_edge(median[:band], 0, settings.min_step)
        last, last_strength, last_sign = _rail_edge(median[-band:], height - band, settings.min_step)
        if first is None or last is None:
            observations.append((column, first, last, None, "weak-edge"))
            continue
        if first_sign == last_sign:
            observations.append((column, first, last, min(first_strength, last_strength), "ambiguous"))
            continue
        strength = min(first_strength, last_strength)
        observations.append((column, first, last, strength, "observed"))
    return observations


def _fit_rail(x, y, strengths):
    mx = np.average(x, weights=strengths)
    center = x - mx
    design = np.column_stack((center, np.ones(len(x))))
    current = strengths.copy()
    for _ in range(5):
        root = np.sqrt(current)
        params, _, _, _ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        residual = y - design @ params
        spread = np.median(np.abs(residual - np.median(residual)))
        current = strengths / np.maximum(1, np.abs(residual) / max(.4, spread * 2.4))
    return float(params[0]), float(params[1] - params[0] * mx)


def _rail_report(records, index, pitch):
    observed = [row for row in records if row[-1] == "observed"]
    columns = np.array([row[0] for row in observed])
    readings = np.array([row[index] for row in observed])
    weights = np.array([row[3] for row in observed])
    slope, intercept = _fit_rail(columns, readings, weights)
    errors = readings - (slope * columns + intercept)
    return {
        "slope": round(slope, 7),
        "intercept_px": round(intercept, 6),
        "tilt_degrees": round(math.degrees(math.atan(slope)), 6),
        "bow_p90_px": round(float(np.quantile(np.abs(errors), .9)), 5),
        "bow_p90_mm": round(float(np.quantile(np.abs(errors), .9)) * pitch, 6),
    }


def _coordinate(region, direction, longitudinal, transverse):
    if transverse is None:
        return None
    if direction == "horizontal":
        return {"x": round(region.x + longitudinal, 5), "y": round(region.y + transverse, 5)}
    return {"x": round(region.x + transverse, 5), "y": round(region.y + longitudinal, 5)}


def inspect_carrier(decoded, region, settings, dpi):
    if region.area > MAX_CARRIER_AREA or min(region.w, region.h) < 24:
        raise DeskError("Use a strip region of at least 24 × 24 and three million pixels or less.", field="region")
    pitch = 25.4 / dpi
    results = _station_strips(decoded, region, settings)
    good = [row for row in results if row[-1] == "observed"]
    coverage = len(good) / len(results)
    rail_names = ("top", "bottom") if settings.direction == "horizontal" else ("left", "right")
    rows = []
    for x, leading, trailing, strength, status in results:
        rows.append(
            {
                "source_station": round(x, 3),
                rail_names[0]: _coordinate(region, settings.direction, x, leading),
                rail_names[1]: _coordinate(region, settings.direction, x, trailing),
                "width_mm": round((trailing - leading) * pitch, 5) if leading is not None and trailing is not None else None,
                "edge_strength": round(strength, 5) if strength is not None else None,
                "status": status,
            }
        )
    report = {
        "region": region.as_dict(), "direction": settings.direction,
        "rails": {}, "stations": rows, "rail_names": list(rail_names),
        "support_fraction": round(coverage, 5), "width_mm": None,
        "width_drift_mm": None, "status": "insufficient",
    }
    if len(good) < 4 or coverage < .32:
        return report
    first = _rail_report(results, 1, pitch)
    last = _rail_report(results, 2, pitch)
    widths = np.array([(hi - lo) * pitch for _, lo, hi, _, _ in good])
    width_drift = float(np.quantile(widths, .95) - np.quantile(widths, .05))
    if coverage < .7:
        status = "review-support"
    elif max(first["bow_p90_mm"], last["bow_p90_mm"], width_drift) > settings.tolerance:
        status = "review-carrier"
    else:
        status = "measured"
    report.update(
        {
            "rails": dict(zip(rail_names, (first, last))),
            "width_mm": round(float(np.median(widths)), 6),
            "width_drift_mm": round(width_drift, 6),
            "status": status,
        }
    )
    return report


def carrier_overlay(decoded, region, result):
    image = rgb_image(decoded.values[region.slices])
    image.thumbnail((1550, 1550))
    painter = ImageDraw.Draw(image)
    sx = image.width / region.w
    sy = image.height / region.h

    def point_on_image(value):
        return (round((value["x"] - region.x) * sx), round((value["y"] - region.y) * sy))

    count_longitudinal = region.w if result["direction"] == "horizontal" else region.h
    for rail, color in zip(result["rail_names"], ("#b0e3c2", "#eca894")):
        for station in result["stations"]:
            if station[rail] is None:
                continue
            x, y = point_on_image(station[rail])
            tint = color if station["status"] == "observed" else "#bf7957"
            painter.ellipse((x - 2, y - 2, x + 2, y + 2), outline=tint, width=2)
        if rail not in result["rails"]:
            continue
        model = result["rails"][rail]
        pixels = []
        for edge in (0, count_longitudinal - 1):
            measured = model["intercept_px"] + model["slope"] * edge
            coord = _coordinate(region, result["direction"], edge, measured)
            pixels.append(point_on_image(coord))
        painter.line(pixels, fill=color, width=1)
    return image
