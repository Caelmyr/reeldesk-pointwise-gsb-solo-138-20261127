"""Measure scanner density response against a certified transmission wedge."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .values import Box, DeskError, PRIMARY_CHANNELS, boolean, clean_text, mapping, number, sequence


MINIMUM_PATCHES = 3
MAXIMUM_PATCHES = 28
CURVE_COLORS = ("#e18476", "#9fcc8f", "#86bfe8")
MINIMUM_RGB_SUPPORT = .55


@dataclass(frozen=True)
class Patch:
    label: str
    box: Box
    nominal: float

    @classmethod
    def read(cls, value, index, size):
        field = f"patches[{index}]"
        value = mapping(value, field)
        bounds = Box.read(value.get("box"), f"{field}.box", bounds=size)
        if min(bounds.w, bounds.h) < 3:
            raise DeskError("Use a wedge patch at least 3 × 3 pixels.", field=f"{field}.box")
        return cls(
            clean_text(value.get("label", f"Step {index + 1}"), f"{field}.label", maximum=50, required=True),
            bounds,
            number(value.get("nominal"), f"{field}.nominal", lower=-.15, upper=4.5),
        )

    def as_dict(self):
        return {"label": self.label, "box": self.box.as_dict(), "nominal": self.nominal}


@dataclass(frozen=True)
class WedgeSpec:
    patches: tuple[Patch, ...]
    tolerance: float
    fit_floor: float
    fit_ceiling: float
    compensate: bool

    @classmethod
    def read(cls, data, size):
        value = mapping(data, "wedge")
        values = sequence(value.get("patches"), "patches", minimum=MINIMUM_PATCHES, maximum=MAXIMUM_PATCHES)
        patches = sorted((Patch.read(item, index, size) for index, item in enumerate(values)), key=lambda item: item.nominal)
        if len({patch.label.casefold() for patch in patches}) != len(patches):
            raise DeskError("Give each wedge step a distinct label.", field="patches")
        if any(right.nominal - left.nominal < .015 for left, right in zip(patches, patches[1:])):
            raise DeskError("Use distinct nominal densities, at least 0.015 apart.", field="patches")
        for index, patch in enumerate(patches[:-1]):
            if any(patch.box.intersect(other.box) for other in patches[index + 1:]):
                raise DeskError("Wedge patches must not overlap.", field="patches")
        tolerance = number(value.get("tolerance", .08), "wedge.tolerance", lower=.015, upper=.8)
        floor = number(value.get("fit_floor", -.1), "wedge.fit_floor", lower=-.15, upper=3.5)
        ceiling = number(value.get("fit_ceiling", 3.2), "wedge.fit_ceiling", lower=floor + .2, upper=4.5)
        return cls(tuple(patches), tolerance, floor, ceiling, boolean(value.get("compensate", False), "wedge.compensate"))

    def as_dict(self):
        return {
            "patches": [patch.as_dict() for patch in self.patches],
            "tolerance": self.tolerance,
            "fit_floor": self.fit_floor,
            "fit_ceiling": self.fit_ceiling,
            "compensate": self.compensate,
        }


def _density_channel(values, eligible, area):
    usable = values[eligible]
    extent = len(usable) / area
    if extent >= MINIMUM_RGB_SUPPORT:
        q05, q50, q95 = np.quantile(usable, [.05, .5, .95])
        spread = float(np.median(np.abs(usable - q50)))
    else:
        q05, q50, q95, spread = (None,) * 4
    return {
        "median": round(float(q50), 6) if q50 is not None else None,
        "range_90": round(float(q95 - q05), 6) if q05 is not None else None,
        "mad": round(spread, 6) if spread is not None else None,
        "fraction": round(extent, 5),
        "count": len(usable),
    }


def wedge_observations(calibrated, patches):
    all_density = calibrated.density
    output = []
    for patch in patches:
        area = patch.box.area
        values = all_density[patch.box.slices]
        valid = calibrated.valid[patch.box.slices]
        observed = {}
        for channel_index, channel in enumerate(PRIMARY_CHANNELS):
            observed[channel] = _density_channel(values[..., channel_index], valid[..., channel_index], area)
        output.append({**patch.as_dict(), "measurements": observed})
    return output


def _weighted_fit(x, y, weights):
    centered_x = np.average(x, weights=weights)
    centered_y = np.average(y, weights=weights)
    total = np.dot((x - centered_x) ** 2, weights)
    if total < 1e-6:
        return None
    slope = np.dot(weights * (x - centered_x), y - centered_y) / total
    return float(slope), float(centered_y - slope * centered_x)


def _robust_line(x, y, base_weights, tolerance):
    initial = _weighted_fit(x, y, base_weights)
    if initial is None:
        return None
    slope, intercept = initial
    weights = base_weights.copy()
    for _ in range(7):
        residual = y - (x * slope + intercept)
        center = np.median(residual)
        spread = np.median(np.abs(residual - center)) * 1.4826
        penalty = np.abs(residual) / max(.015, tolerance / 2, spread * 1.3)
        weights = base_weights / np.maximum(1, penalty)
        update = _weighted_fit(x, y, weights)
        if update is None:
            return None
        next_slope, next_intercept = update
        if abs(next_slope - slope) + abs(next_intercept - intercept) <= 1e-6:
            slope, intercept = next_slope, next_intercept
            break
        slope, intercept = next_slope, next_intercept
    return slope, intercept


def _observed_rows(observations, channel, floor, ceiling):
    rows = []
    for index, item in enumerate(observations):
        expected = item["nominal"]
        values = item["measurements"][channel]
        if values["median"] is None or not floor <= expected <= ceiling:
            continue
        median = values["median"]
        influence = values["fraction"] / max(.018, (values["range_90"] + .02))
        rows.append((index, expected, median, influence))
    return rows


def _fit_channel(observations, channel, settings):
    rows = _observed_rows(observations, channel, settings.fit_floor, settings.fit_ceiling)
    base = {"usable_steps": len(rows), "slope": None, "offset": None, "mae": None, "fit_range": None}
    if len(rows) < MINIMUM_PATCHES:
        return {**base, "status": "insufficient"}
    rows = np.asarray(rows, dtype=np.float64)
    _, nominal, measured, weights = rows.T
    fitted = _robust_line(nominal, measured, weights, settings.tolerance)
    if fitted is None:
        return {**base, "status": "insufficient"}
    slope, offset = fitted
    predictions = nominal * slope + offset
    error = measured - predictions
    status = "in-tolerance"
    if slope < .25 or slope > 1.75:
        status = "response-suspect"
    elif np.count_nonzero(np.abs(error) <= settings.tolerance) < len(rows) * .8:
        status = "review-steps"
    return {
        **base,
        "status": status,
        "slope": round(slope, 8),
        "offset": round(offset, 8),
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(np.sqrt(np.mean(error ** 2))), 6),
        "fit_range": [float(nominal.min()), float(nominal.max())],
    }


def _channel_status(patch, result, settings):
    value = patch["measurement"]
    if value is None:
        return {"error_to_nominal": None, "residual_to_fit": None, "corrected": None, "status": "invalid"}
    nominal = patch["nominal"]
    raw_error = value - nominal
    output = {"error_to_nominal": round(raw_error, 6), "residual_to_fit": None, "corrected": None}
    if result["slope"] is None:
        output["status"] = "in-tolerance" if abs(raw_error) <= settings.tolerance else "review"
        return output
    fitted_range = result["fit_range"]
    if not fitted_range[0] <= nominal <= fitted_range[1]:
        output["status"] = "outside-fit"
        return output
    discrepancy = value - (result["slope"] * nominal + result["offset"])
    output["residual_to_fit"] = round(discrepancy, 6)
    if settings.compensate and result["status"] != "response-suspect":
        effective = (value - result["offset"]) / result["slope"]
        output["corrected"] = round(effective, 6)
        comparison = effective - nominal
    else:
        comparison = raw_error
    output["status"] = "in-tolerance" if abs(comparison) <= settings.tolerance else "review"
    return output


def assess_wedge(calibrated, settings):
    observations = wedge_observations(calibrated, settings.patches)
    fits = {channel: _fit_channel(observations, channel, settings) for channel in PRIMARY_CHANNELS}
    okay = 0
    review = 0
    for row in observations:
        statuses = []
        for channel in PRIMARY_CHANNELS:
            entry = row["measurements"][channel]
            status = _channel_status({"measurement": entry["median"], "nominal": row["nominal"]}, fits[channel], settings)
            entry.update(status)
            statuses.append(status["status"])
        if all(value == "in-tolerance" for value in statuses):
            okay += 1
        else:
            review += 1
    return {
        "steps": observations,
        "fits": fits,
        "within_tolerance_steps": okay,
        "review_steps": review,
        "tolerance": settings.tolerance,
        "compensation_enabled": settings.compensate,
        "units": "optical-density",
    }


def _grid(drawer, bounds, minimum, maximum):
    left, top, right, bottom = bounds
    drawer.rectangle(bounds, fill="#1f2021", outline="#5a5048")
    std = max(.25, math.ceil((maximum - minimum) / 5 * 4) / 4)
    cursor = math.ceil(minimum / std) * std
    axis_font = ImageFont.load_default(size=14)
    while cursor <= maximum + 1e-9:
        x = round(left + (right - left) * (cursor - minimum) / (maximum - minimum))
        y = round(bottom - (bottom - top) * (cursor - minimum) / (maximum - minimum))
        drawer.line((x, top, x, bottom), fill="#333536")
        drawer.line((left, y, right, y), fill="#333536")
        drawer.text((x, bottom + 5), f"{cursor:.2g}", anchor="mt", font=axis_font, fill="#aeb5b5")
        drawer.text((left - 5, y), f"{cursor:.2g}", anchor="rm", font=axis_font, fill="#aeb5b5")
        cursor += std
    drawer.line((left, bottom, right, top), fill="#555454", width=2)


def wedge_chart(result):
    all_measured = [
        item["median"]
        for row in result["steps"]
        for item in row["measurements"].values()
        if item["median"] is not None
    ]
    all_nominal = [row["nominal"] for row in result["steps"]]
    minimum = min(0, min(all_nominal + all_measured))
    maximum = max(1, max(all_nominal + all_measured)) + .08
    image = Image.new("RGB", (780, 540), "#272320")
    drawer = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=25)
    label_font = ImageFont.load_default(size=14)
    drawer.text((34, 22), "Transmission wedge response", fill="#ece1d3", font=title_font)
    drawer.text((34, 55), "Measured density vs. certified step density", fill="#b3a997", font=label_font)
    left, top, right, bottom = 73, 95, 736, 463
    _grid(drawer, (left, top, right, bottom), minimum, maximum)

    def position(x, y):
        return (
            int(left + (right - left) * (x - minimum) / (maximum - minimum)),
            int(bottom - (bottom - top) * (y - minimum) / (maximum - minimum)),
        )

    for index, (channel, color) in enumerate(zip(PRIMARY_CHANNELS, CURVE_COLORS)):
        result_line = result["fits"][channel]
        if result_line["fit_range"]:
            source_x = result_line["fit_range"]
            source_y = [result_line["slope"] * x + result_line["offset"] for x in source_x]
            drawer.line([position(x, y) for x, y in zip(source_x, source_y)], fill=color, width=2)
        for row in result["steps"]:
            values = row["measurements"][channel]
            if values["median"] is None:
                continue
            x, y = position(row["nominal"], values["median"])
            brush = color if values["status"] == "in-tolerance" else "#292524"
            drawer.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2, fill=brush)
        drawer.text((36 + 24 * index, 511), channel, fill=color, font=label_font)
    drawer.text((395, 511), "Certified density  ·  D", fill="#c5b7a7", font=label_font, anchor="mt")
    return image
