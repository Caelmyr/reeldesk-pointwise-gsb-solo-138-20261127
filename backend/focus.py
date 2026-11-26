"""A repeatable slanted-edge focus check in native scanner coordinates."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .imaging import SCANNER_LUMA
from .values import Box, DeskError, integer, mapping, number, one_of, rgb_image


ORIENTATIONS = frozenset({"auto", "vertical", "horizontal"})
FOCUS_PIXEL_LIMIT = 1_000_000


@dataclass(frozen=True)
class FocusOptions:
    orientation: str
    oversample: int
    radius: float
    minimum_contrast: float

    @classmethod
    def read(cls, value):
        value = {} if value is None else mapping(value, "focus")
        return cls(
            one_of(value.get("orientation", "auto"), ORIENTATIONS, "focus.orientation"),
            integer(value.get("oversample", 4), "focus.oversample", lower=2, upper=8),
            number(value.get("radius", 9), "focus.radius", lower=3, upper=28),
            number(value.get("minimum_contrast", .09), "focus.minimum_contrast", lower=.015, upper=.85),
        )

    def as_dict(self):
        return {
            "orientation": self.orientation,
            "oversample": self.oversample,
            "radius": self.radius,
            "minimum_contrast": self.minimum_contrast,
        }


def _gray_region(corrected, region):
    measurements = corrected.values[region.slices]
    valid = corrected.valid[region.slices].all(axis=2)
    gray = np.einsum("ijk,k->ij", measurements, SCANNER_LUMA)
    if valid.mean() < .8:
        raise DeskError("Choose a focus edge with at least 80% usable pixels.", field="region")
    sample = gray[valid]
    lo, hi = np.quantile(sample, [.05, .95])
    gray = np.where(valid, gray, np.median(sample))
    return gray, valid, float(lo), float(hi)


def _edge_direction(gray, valid):
    smoothed = ndimage.gaussian_filter(gray, .9)
    dy, dx = np.gradient(smoothed)
    guard = ndimage.binary_erosion(valid)
    energy = np.maximum(np.hypot(dx, dy), 1e-6)
    selection = guard & (energy >= np.quantile(energy[guard], .82))
    x = float(np.sum(dx[selection] ** 2))
    y = float(np.sum(dy[selection] ** 2))
    return "vertical" if x >= y else "horizontal"


def _line_samples(gray, valid, minimum):
    blurred = ndimage.gaussian_filter1d(gray, 1, axis=1, mode="reflect")
    slope = np.diff(blurred, axis=1)
    usable = valid[:, 1:] & valid[:, :-1]
    widths = np.where(usable, np.abs(slope), 0)
    widths[:, :2] = 0
    widths[:, -2:] = 0
    rows = np.arange(gray.shape[0])
    peaks = np.argmax(widths, axis=1)
    peak_values = widths[rows, peaks]
    keep = peak_values > max(.007, minimum * .065)
    left = widths[rows, np.maximum(0, peaks - 1)]
    right = widths[rows, np.minimum(widths.shape[1] - 1, peaks + 1)]
    denominator = left - 2 * peak_values + right
    offset = np.divide(.5 * (left - right), denominator, out=np.zeros_like(denominator), where=denominator < -1e-6)
    sign = np.sign(slope[rows, peaks])
    if keep.any():
        dominance = np.sign(np.sum(sign[keep] * peak_values[keep]))
        keep &= sign == dominance
    return rows[keep], peaks[keep] + .5 + np.clip(offset[keep], -.45, .45), peak_values[keep]


def _fit(rows, locations, weights):
    centered = rows - rows.mean()
    matrix = np.column_stack((centered, np.ones_like(centered)))
    active = np.ones(len(rows), dtype=bool)
    fitted = None
    for _ in range(6):
        if active.sum() < 12:
            return None
        scales = np.sqrt(weights[active])
        coefficients, _, _, _ = np.linalg.lstsq(
            matrix[active] * scales[:, None], locations[active] * scales, rcond=None
        )
        fitted = coefficients[0], coefficients[1] - coefficients[0] * rows.mean()
        errors = np.abs(locations - (fitted[0] * rows + fitted[1]))
        keep = errors < max(.7, np.median(errors[active]) * 3.3)
        if np.array_equal(keep, active):
            break
        active = keep
    if not active.any():
        return None
    return fitted, active, errors


def _station_bins(gray, valid, slope, intercept, radius, oversample):
    yy, xx = np.mgrid[:gray.shape[0], :gray.shape[1]]
    offsets = (xx - (yy * slope + intercept)) / math.hypot(1, slope)
    groups = np.rint(offsets * oversample).astype(int)
    core = int(radius * oversample)
    eligible = valid & (groups >= -core) & (groups <= core)
    target = groups[eligible] + core
    if target.size == 0:
        raise DeskError("The fitted edge leaves no usable pixels in this region.", field="region")
    distances = np.arange(-core, core + 1) / oversample
    sums = np.bincount(target, weights=gray[eligible], minlength=distances.size)
    samples = np.bincount(target, minlength=distances.size)
    full = samples >= max(2, gray.shape[0] / oversample / 5)
    if full.sum() < len(distances) * .7:
        raise DeskError("The edge does not support enough subpixel positions. Choose a slightly slanted edge.", field="region")
    averaged = sums[full] / samples[full]
    curve = np.interp(distances, distances[full], averaged)
    return distances, curve, samples, full


def _frequency_crossing(frequencies, amplitudes, target):
    for index in range(1, len(amplitudes)):
        if amplitudes[index] <= target < amplitudes[index - 1]:
            previous = amplitudes[index - 1]
            current = amplitudes[index]
            blend = (previous - target) / max(1e-9, previous - current)
            return float(frequencies[index - 1] + blend * (frequencies[index] - frequencies[index - 1]))
    return None


def _edge_width(distances, esf):
    normalized = np.maximum.accumulate(np.clip(esf, 0, 1))
    if normalized[0] > .1 or normalized[-1] < .9:
        return None
    limits = np.interp([.1, .9], normalized, distances)
    return round(float(limits[1] - limits[0]), 5)


def _spectral_measurement(distances, flux, oversample, minimum, dpi):
    terminal = max(3, len(flux) // 7)
    start = np.median(flux[:terminal])
    end = np.median(flux[-terminal:])
    contrast = float(abs(end - start))
    if contrast < minimum:
        raise DeskError("Use an edge with stronger low-to-high contrast.", field="region")
    normalized = (flux - start) / (end - start)
    esf = ndimage.gaussian_filter1d(normalized, sigma=.58)
    derivative = np.gradient(esf, 1 / oversample)
    taper = np.hanning(len(derivative))
    windowed = derivative * taper
    size = 1 << (max(128, len(windowed) * 3) - 1).bit_length()
    response = np.abs(np.fft.rfft(windowed, n=size))
    if response[0] < 1e-8:
        raise DeskError("This edge has unstable light levels.", field="region")
    frequencies = np.fft.rfftfreq(size, d=1 / oversample)
    include = frequencies <= .5 + 1e-8
    frequencies = frequencies[include]
    response = np.clip(response[include] / response[0], 0, 1.4)
    mtf50 = _frequency_crossing(frequencies, response, .5)
    mtf10 = _frequency_crossing(frequencies, response, .1)
    return {
        "distance_px": distances.round(5).tolist(),
        "edge_spread": esf.round(5).tolist(),
        "frequency_cyc_px": frequencies.round(5).tolist(),
        "frequency_response": response.round(5).tolist(),
        "mtf50_cyc_px": round(mtf50, 6) if mtf50 is not None else None,
        "mtf10_cyc_px": round(mtf10, 6) if mtf10 is not None else None,
        "mtf50_lp_mm": round(mtf50 * dpi / 25.4, 5) if mtf50 is not None and dpi else None,
        "edge_width_px": _edge_width(distances, esf),
        "contrast": round(contrast, 5),
    }


def measure_focus(corrected, region, settings, *, dpi=None):
    if region.area > FOCUS_PIXEL_LIMIT or min(region.w, region.h) < 20:
        raise DeskError("Use an edge region of at least 20 × 20 and at most one million pixels.", field="region")
    gray, valid, low, high = _gray_region(corrected, region)
    if high - low < settings.minimum_contrast:
        raise DeskError("Choose a region with a clear light-to-dark edge.", field="region")
    orientation = _edge_direction(gray, valid) if settings.orientation == "auto" else settings.orientation
    if orientation == "horizontal":
        gray = gray.T
        valid = valid.T
    rows, locations, weights = _line_samples(gray, valid, settings.minimum_contrast)
    if len(rows) < max(12, gray.shape[0] // 3):
        raise DeskError("A straight edge must run through the selected region.", field="region")
    fitted = _fit(rows.astype(float), locations, weights)
    if fitted is None:
        raise DeskError("The edge positions cannot be fit to one reliable line.", field="region")
    model, inliers, errors = fitted
    slope, intercept = model
    angle = abs(math.degrees(math.atan(slope)))
    if angle < .4 or angle > 24:
        raise DeskError("Use an edge tilted between 0.4° and 24° from the chosen orientation.", field="region")
    edge_start = intercept
    edge_end = intercept + (gray.shape[0] - 1) * slope
    if max(edge_start, edge_end) >= gray.shape[1] - 2 or min(edge_start, edge_end) <= 2:
        raise DeskError("The edge leaves the selection. Move the box to include both sides throughout.", field="region")
    distances, flux, counts, full = _station_bins(gray, valid, slope, intercept, settings.radius, settings.oversample)
    report = _spectral_measurement(distances, flux, settings.oversample, settings.minimum_contrast, dpi)
    inlier_errors = errors[inliers]
    p90 = float(np.quantile(inlier_errors, .9))
    if orientation == "vertical":
        line = {"start": {"x": region.x + edge_start, "y": region.y}, "end": {"x": region.x + edge_end, "y": region.bottom - 1}}
    else:
        line = {"start": {"x": region.x, "y": region.y + edge_start}, "end": {"x": region.right - 1, "y": region.y + edge_end}}
    return {
        **report,
        "region": region.as_dict(),
        "orientation": orientation,
        "tilt_degrees": round(angle, 4),
        "line": line,
        "fit_residual_p90_px": round(p90, 5),
        "status": "review-edge" if p90 > .85 or inliers.sum() < gray.shape[0] * .68 else "measured",
        "edge_scanlines": int(inliers.sum()),
        "source_scanlines": gray.shape[0],
        "support_fraction": round(float(full.mean()), 5),
        "bin_counts": counts.tolist(),
    }


def _plot(drawer, origin, size, x, y, limits, color):
    left, top = origin
    width, height = size
    xmin, xmax, ymin, ymax = limits
    bounds = left, top, left + width, top + height
    drawer.rectangle(bounds, fill="#18191b", outline="#544b45")
    for index in range(1, 4):
        gx = left + width * index / 4
        gy = top + height * index / 4
        drawer.line((gx, top, gx, top + height), fill="#333235")
        drawer.line((left, gy, left + width, gy), fill="#333235")
    points = []
    for xx, yy in zip(x, y):
        px = left + width * (xx - xmin) / (xmax - xmin)
        py = top + height * (1 - (yy - ymin) / (ymax - ymin))
        points.append((round(px), round(np.clip(py, top + 1, top + height - 1))))
    if len(points) > 1:
        drawer.line(points, fill=color, width=2)


def focus_proof(image, result):
    region = Box.read(result["region"])
    photo = rgb_image(image.values[region.slices])
    photo.thumbnail((312, 275), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (900, 560), "#262322")
    drawer = ImageDraw.Draw(canvas)
    headline = ImageFont.load_default(size=26)
    normal = ImageFont.load_default(size=14)
    drawer.text((26, 20), "Slanted-edge focus check", font=headline, fill="#ece1d3")
    x, y = 25, 97
    canvas.paste(photo, (x, y))
    a, b = result["line"]["start"], result["line"]["end"]
    sx = photo.width / region.w
    sy = photo.height / region.h
    endpoints = [
        (x + round((point["x"] - region.x) * sx), y + round((point["y"] - region.y) * sy))
        for point in (a, b)
    ]
    drawer.line(endpoints, fill="#ecc897", width=2)
    mtf = result["mtf50_cyc_px"]
    reading = f"{mtf:.3f} cyc/px" if mtf is not None else "beyond measured band"
    rows = [
        f"MTF50  ·  {reading}",
        f"Edge tilt  ·  {result['tilt_degrees']:.2f}°",
        f"Fit residual p90  ·  {result['fit_residual_p90_px']:.2f} px",
        f"Scanlines  ·  {result['edge_scanlines']} / {result['source_scanlines']}",
        f"Status  ·  {result['status']}",
    ]
    for index, row in enumerate(rows):
        drawer.text((25, min(y + photo.height + 12, 381) + 24 * index), row, font=normal, fill="#d1c3ab")
    drawer.text((370, 75), "Edge response by distance (px)", font=normal, fill="#d1c3ab")
    distances = result["distance_px"]
    _plot(drawer, (370, 98), (495, 168), distances, result["edge_spread"],
          (distances[0], distances[-1], -.08, 1.08), "#e6a976")
    drawer.text((370, 305), "Frequency response (0–0.5 cycles/pixel)", font=normal, fill="#d1c3ab")
    _plot(drawer, (370, 329), (495, 168), result["frequency_cyc_px"], result["frequency_response"],
          (0, .5, 0, 1.06), "#98c5ad")
    return canvas
