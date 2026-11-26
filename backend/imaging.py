"""Decode scanner output and measure transmitted density in source coordinates."""

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from math import ceil
from threading import RLock
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .values import (
    Box, DeskError, PIXEL_LIMIT, PRIMARY_CHANNELS, UPLOAD_LIMIT, boolean,
    channel_vector, image_bytes, mapping, number, optional_key, rgb_image,
    round_channels,
)


FORMATS = frozenset({"JPEG", "PNG", "TIFF", "WEBP"})
GRAY16 = frozenset({"I;16", "I;16L", "I;16B", "I;16N"})
EDGE_GAP = 1 / 4096
SATURATION = 0.997
BLACK_FLOOR = 1 / 4096
DEFAULT_DENSITY_RANGE = (0.08, 1.75)
SCANNER_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class Decoded:
    values: np.ndarray
    valid: np.ndarray
    bit_depth: int

    @property
    def width(self):
        return self.values.shape[1]

    @property
    def height(self):
        return self.values.shape[0]

    @property
    def size(self):
        return self.width, self.height


@dataclass(frozen=True)
class Calibrated:
    values: np.ndarray
    valid: np.ndarray
    low_clip: np.ndarray
    high_clip: np.ndarray

    @property
    def size(self):
        return self.values.shape[1], self.values.shape[0]

    @property
    def density(self):
        return -np.log10(np.maximum(self.values, BLACK_FLOOR))


def _open_original(blob):
    if not blob:
        raise DeskError("Choose an image file.", field="file")
    if len(blob) > UPLOAD_LIMIT:
        raise DeskError("Use an image smaller than 36 MiB.", field="file", status=413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(blob))
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError) as error:
        raise DeskError("The image could not be decoded.", field="file") from error
    try:
        if image.format not in FORMATS:
            raise DeskError("Import a PNG, TIFF, JPEG or WebP scan.", field="file")
        if image.width * image.height > PIXEL_LIMIT:
            raise DeskError("Use a scan with at most 12 million pixels.", field="file")
        if image.width < 12 or image.height < 12:
            raise DeskError("The scan must be at least 12 × 12 pixels.", field="file")
        if getattr(image, "n_frames", 1) != 1:
            raise DeskError("Import one image per file. Multi-page TIFF and animated images are not supported.", field="file")
        image.load()
        return image
    except DeskError:
        image.close()
        raise
    except (OSError, ValueError, SyntaxError) as error:
        image.close()
        raise DeskError("The image is incomplete or damaged.", field="file") from error


def probe_original(blob):
    with _open_original(blob) as image:
        orientation = _exif_orientation(image)
        dpi = _image_dpi(image)
        source_format = image.format
        width, height = image.size
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        return {
            "source_format": source_format,
            "mode": image.mode,
            "width": width,
            "height": height,
            "orientation": orientation,
            "embedded_dpi": dpi,
        }


def _exif_orientation(image):
    try:
        raw = image.getexif().get(274, 1)
    except (OSError, ValueError, SyntaxError):
        return 1
    return raw if raw in range(1, 9) else 1


def _image_dpi(image):
    raw = image.info.get("dpi")
    if not raw or not isinstance(raw, (list, tuple)):
        return None
    try:
        x, y = map(float, raw[:2])
    except (TypeError, ValueError, OverflowError):
        return None
    if not (30 <= x <= 20_000 and 30 <= y <= 20_000) or abs(x - y) > .02 * max(x, y):
        return None
    return round((x + y) / 2, 2)


def _visible_rgb(image):
    if image.mode == "CMYK":
        profile = image.info.get("icc_profile")
        if profile:
            try:
                source = ImageCms.ImageCmsProfile(BytesIO(profile))
                return ImageCms.profileToProfile(image, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
            except (ImageCms.PyCMSError, OSError, ValueError):
                pass
    return image.convert("RGB")


def decode_original(blob):
    with _open_original(blob) as native:
        upright = ImageOps.exif_transpose(native)
        if upright.mode in GRAY16:
            gray = np.array(upright, dtype=np.float32) / 65535.0
            values = np.repeat(gray[..., None], 3, axis=2)
            valid = np.ones(gray.shape, dtype=bool)
            depth = 16
        elif upright.mode in ("I", "F"):
            scalar = np.asarray(upright, dtype=np.float32)
            finite = np.isfinite(scalar)
            if upright.mode == "I":
                valid = finite & (scalar >= 0) & (scalar <= 65535)
                scalar = scalar / 65535.0
                depth = 16
            else:
                valid = finite & (scalar >= 0) & (scalar <= 1)
                depth = 32
            values = np.repeat(np.where(valid, scalar, 0)[..., None], 3, axis=2)
        else:
            if "A" in upright.getbands() or "transparency" in upright.info:
                rgba = np.asarray(upright.convert("RGBA"), dtype=np.uint8)
                values = rgba[..., :3].astype(np.float32) / 255.0
                valid = rgba[..., 3] == 255
            else:
                values = np.asarray(_visible_rgb(upright), dtype=np.float32) / 255.0
                valid = np.ones(values.shape[:2], dtype=bool)
            depth = 8
    values.flags.writeable = False
    valid.flags.writeable = False
    return Decoded(values, valid, depth)


class DecodeCache:
    """Bound decoded arrays by bytes, using immutable original content hashes."""

    def __init__(self, byte_budget=150 * 1024 * 1024):
        self.byte_budget = byte_budget
        self.entries = OrderedDict()
        self.used = 0
        self.lock = RLock()

    def read(self, checksum, blob):
        with self.lock:
            cached = self.entries.get(checksum)
            if cached is not None:
                self.entries.move_to_end(checksum)
                return cached
        decoded = decode_original(blob)
        size = decoded.values.nbytes + decoded.valid.nbytes
        if size > self.byte_budget:
            return decoded
        with self.lock:
            if checksum in self.entries:
                return self.entries[checksum]
            while self.used + size > self.byte_budget:
                _, previous = self.entries.popitem(last=False)
                self.used -= previous.values.nbytes + previous.valid.nbytes
            self.entries[checksum] = decoded
            self.used += size
            return decoded


@dataclass(frozen=True)
class Calibration:
    flat_id: str | None
    dark_id: str | None
    base: tuple[float, float, float]
    reject_clipped: bool

    @classmethod
    def read(cls, value, field="calibration"):
        value = {} if value is None else mapping(value, field)
        return cls(
            optional_key(value.get("flat_id"), f"{field}.flat_id"),
            optional_key(value.get("dark_id"), f"{field}.dark_id"),
            channel_vector(value.get("base", [1, 1, 1]), f"{field}.base", lower=.015, upper=1.5),
            boolean(value.get("reject_clipped", True), f"{field}.reject_clipped"),
        )

    def as_dict(self):
        return {
            "flat_id": self.flat_id, "dark_id": self.dark_id,
            "base": list(self.base), "reject_clipped": self.reject_clipped,
        }


@dataclass(frozen=True)
class Tone:
    shadow: float
    highlight: float
    gamma: float
    neutralize: bool
    brightness: float

    @classmethod
    def read(cls, value):
        value = {} if value is None else mapping(value, "tone")
        shadow = number(value.get("shadow", DEFAULT_DENSITY_RANGE[0]), "tone.shadow", lower=-.4, upper=3.0)
        highlight = number(value.get("highlight", DEFAULT_DENSITY_RANGE[1]), "tone.highlight", lower=-.35, upper=3.6)
        if highlight - shadow < .025:
            raise DeskError("Move the highlight at least 0.025 density above the shadow.", field="tone.highlight")
        return cls(
            shadow, highlight,
            number(value.get("gamma", 1), "tone.gamma", lower=.35, upper=3),
            boolean(value.get("neutralize", False), "tone.neutralize"),
            number(value.get("brightness", 0), "tone.brightness", lower=-.4, upper=.4),
        )

    def as_dict(self):
        return {
            "shadow": self.shadow, "highlight": self.highlight, "gamma": self.gamma,
            "neutralize": self.neutralize, "brightness": self.brightness,
        }


def apply_calibration(scan, calibration, *, flat=None, dark=None):
    for control, label in ((flat, "flat"), (dark, "dark")):
        if control is not None and control.size != scan.size:
            raise DeskError("Controls must use the same scanner dimensions as the scan.", field=f"calibration.{label}_id")
    black = dark.values if dark is not None else 0
    white = flat.values if flat is not None else 1
    difference = white - black
    given_base = np.array(calibration.base, dtype=np.float32)
    denominator = np.maximum(difference, EDGE_GAP) * given_base
    normalized = (scan.values - black) / denominator
    control_valid = scan.valid.copy()
    if flat is not None:
        control_valid &= flat.valid
    if dark is not None:
        control_valid &= dark.valid
    high_clip = (scan.values >= SATURATION) & control_valid[..., None]
    low_clip = (scan.values <= black + BLACK_FLOOR) & control_valid[..., None]
    control_support = (difference > EDGE_GAP) & control_valid[..., None]
    valid = np.broadcast_to(control_support, scan.values.shape).copy()
    if calibration.reject_clipped:
        valid &= ~(high_clip | low_clip)
    return Calibrated(normalized.astype(np.float32, copy=False), valid, low_clip, high_clip)


def _near_gray(values, valid):
    complete = valid.all(axis=2)
    selected = values[complete]
    if len(selected) == 0:
        return np.zeros(3, dtype=np.float32)
    step = max(1, len(selected) // 90_000)
    middle = np.median(selected[::step], axis=0)
    return middle - float(np.dot(middle, SCANNER_LUMA))


def positive_pixels(calibrated, tone, medium):
    density = calibrated.density
    valid = calibrated.valid
    if tone.neutralize:
        density -= _near_gray(density, valid)
    level = (density - tone.shadow) / (tone.highlight - tone.shadow)
    if medium != "negative":
        level = 1 - level
    level = np.clip(level + tone.brightness, 0, 1)
    return np.power(level, 1 / tone.gamma)


def _blend_ineligible(values, eligible):
    # A neutral checker makes invalid film visible even against an empty image.
    y, x = np.ogrid[:values.shape[0], :values.shape[1]]
    checker = np.where(((x // 12) + (y // 12)) % 2 == 0, .16, .25)[..., None]
    return np.where(eligible[..., None], values, checker)


def _clip_view(scan, calibrated):
    base = scan.values.mean(axis=2)
    output = np.repeat((base * .62)[..., None], 3, axis=2)
    lo = calibrated.low_clip.any(axis=2)
    hi = calibrated.high_clip.any(axis=2)
    output[lo] = (.08, .52, .92)
    output[hi] = (.96, .36, .16)
    output[lo & hi] = (.77, .41, .92)
    return _blend_ineligible(output, scan.valid)


def _density_view(calibrated, density_range):
    density = calibrated.density
    weighted = np.einsum("ijk,k->ij", density, SCANNER_LUMA)
    low, high = density_range
    unit = np.clip((weighted - low) / (high - low), 0, 1)
    colors = np.array(
        [[.08, .12, .19], [.14, .43, .52], [.43, .68, .57], [.83, .68, .39], [.99, .88, .67]]
    )
    result = np.stack(
        [np.interp(unit, np.linspace(0, 1, len(colors)), colors[:, channel]) for channel in range(3)],
        axis=2,
    )
    return _blend_ineligible(result, calibrated.valid.all(axis=2))


def render_view(scan, calibrated, kind, tone, medium, *, region=None, max_side=1600):
    if kind == "raw":
        output = _blend_ineligible(scan.values, scan.valid)
    elif kind == "positive":
        output = _blend_ineligible(positive_pixels(calibrated, tone, medium), calibrated.valid.all(axis=2))
    elif kind == "density":
        output = _density_view(calibrated, (tone.shadow, tone.highlight))
    elif kind == "clip":
        output = _clip_view(scan, calibrated)
    else:
        raise DeskError("Choose a supported image view.", field="view")
    output = rgb_image(output[region.slices] if region else output)
    output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image_bytes(output)


def source_point(scan, calibrated, point, radius):
    x, y = point.pixel()
    region = Box(x, y, 1, 1).expanded(radius, *scan.size)
    summaries = density_stats(calibrated, region)["channels"]
    selected = {}
    for index, name in enumerate(PRIMARY_CHANNELS):
        is_usable = bool(calibrated.valid[y, x, index])
        sample = scan.values[y, x, index]
        transmission = float(calibrated.values[y, x, index]) if is_usable else None
        selected[name] = {
            "raw": round(float(sample), 6) if scan.valid[y, x] else None,
            "transmission": round(transmission, 6) if transmission is not None else None,
            "density": round(float(-np.log10(max(BLACK_FLOOR, transmission))), 6) if transmission is not None else None,
            "low_clip": bool(calibrated.low_clip[y, x, index]),
            "high_clip": bool(calibrated.high_clip[y, x, index]),
            "sample_support": summaries[name]["valid"],
            "neighborhood_median": summaries[name]["median"],
        }
    return {
        "point": point.as_dict(), "neighborhood": region.as_dict(),
        "channels": selected, "bit_depth": scan.bit_depth, "raw_pixel_available": bool(scan.valid[y, x]),
    }


def density_stats(calibrated, region):
    density = calibrated.density[region.slices]
    valid = calibrated.valid[region.slices]
    lows = calibrated.low_clip[region.slices]
    highs = calibrated.high_clip[region.slices]
    channels = {}
    for index, channel in enumerate(PRIMARY_CHANNELS):
        chosen = density[..., index][valid[..., index]]
        if len(chosen):
            q05, median, q95 = np.quantile(chosen, [.05, .5, .95])
            spread = np.median(np.abs(chosen - median))
            stats = {
                "mean": round(float(chosen.mean()), 5),
                "median": round(float(median), 5),
                "p05": round(float(q05), 5),
                "p95": round(float(q95), 5),
                "mad": round(float(spread), 5),
            }
        else:
            stats = dict.fromkeys(("mean", "median", "p05", "p95", "mad"))
        channels[channel] = {
            **stats,
            "valid": int(len(chosen)),
            "valid_fraction": round(len(chosen) / region.area, 5),
            "clips_low": int(lows[..., index].sum()),
            "clips_high": int(highs[..., index].sum()),
        }
    return {
        "region": region.as_dict(),
        "area": region.area,
        "valid_rgb": int(valid.all(axis=2).sum()),
        "channels": channels,
    }


def base_estimate(calibrated, region):
    values = calibrated.values[region.slices]
    valid = calibrated.valid[region.slices]
    estimates = []
    counts = []
    for index in range(3):
        chosen = values[..., index][valid[..., index]]
        counts.append(len(chosen))
        if len(chosen) < max(12, ceil(region.area / 2)):
            raise DeskError("Choose clear film base with usable pixels in all three channels.", field="region")
        estimates.append(float(np.median(chosen)))
    if not all(.015 <= value <= 1.5 for value in estimates):
        raise DeskError("The base reading is outside the usable transmission range.", field="region")
    return {"region": region.as_dict(), "base": round_channels(estimates), "counts": counts}


def histogram(calibrated, region, *, bins=48, low=-.2, high=2.8):
    values = calibrated.density[region.slices]
    valid = calibrated.valid[region.slices]
    edges = np.linspace(low, high, bins + 1)
    result = {}
    for index, channel in enumerate(PRIMARY_CHANNELS):
        chosen = values[..., index][valid[..., index]]
        counts, _ = np.histogram(chosen, bins=edges)
        result[channel] = {
            "counts": counts.tolist(),
            "below": int((chosen < low).sum()),
            "above": int((chosen > high).sum()),
            "invalid": int(region.area - len(chosen)),
        }
    return {"edges": edges.round(5).tolist(), "channels": result, "region": region.as_dict()}


def compare_regions(calibrated, region, reference):
    left = density_stats(calibrated, region)
    right = density_stats(calibrated, reference)
    differences = {}
    missing = []
    for channel in PRIMARY_CHANNELS:
        main = left["channels"][channel]
        ref = right["channels"][channel]
        if min(main["valid_fraction"], ref["valid_fraction"]) < .5:
            differences[channel] = None
            missing.append(channel)
        else:
            differences[channel] = round(main["median"] - ref["median"], 5)
    return {
        "region": left,
        "reference": right,
        "delta": differences,
        "missing_channels": missing,
        "status": "review" if missing else "measured",
    }


def _transverse_geometry(segment, width, samples):
    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y
    scale = np.linspace(0, 1, samples)
    transverse = np.arange(-width, width + 1, dtype=np.float64)
    x = segment.start.x + dx * scale[:, None] - (dy / segment.length) * transverse
    y = segment.start.y + dy * scale[:, None] + (dx / segment.length) * transverse
    return x, y


def _bilinear_valid(values, valid, x, y):
    height, width = values.shape
    safe = (x >= 0) & (y >= 0) & (x <= width - 1) & (y <= height - 1)
    sx = np.clip(x, 0, width - 1)
    sy = np.clip(y, 0, height - 1)
    left = np.floor(sx).astype(np.intp)
    top = np.floor(sy).astype(np.intp)
    right = np.minimum(left + 1, width - 1)
    bottom = np.minimum(top + 1, height - 1)
    tx = sx - left
    ty = sy - top
    total = np.zeros_like(x, dtype=np.float32)
    for rows, columns, weight in (
        (top, left, (1 - tx) * (1 - ty)),
        (top, right, tx * (1 - ty)),
        (bottom, left, (1 - tx) * ty),
        (bottom, right, tx * ty),
    ):
        sample_valid = valid[rows, columns]
        safe &= sample_valid | (weight < 1e-8)
        total += values[rows, columns] * weight
    return total, safe


def profile(calibrated, segment, transverse, samples, *, dpi=None):
    x, y = _transverse_geometry(segment, transverse, samples)
    density = calibrated.density
    distances = np.linspace(0, segment.length, samples)
    channel_results = {}
    usable = []
    for index, channel in enumerate(PRIMARY_CHANNELS):
        values, valid = _bilinear_valid(density[..., index], calibrated.valid[..., index], x, y)
        medians = []
        coverage = []
        for sample, kept in zip(values, valid):
            chosen = sample[kept]
            medians.append(round(float(np.median(chosen)), 5) if len(chosen) else None)
            coverage.append(int(len(chosen)))
        channel_results[channel] = {"density": medians, "support": coverage}
        usable.append(sum(count > 0 for count in coverage))
    return {
        "line": segment.as_dict(),
        "distance_px": distances.round(4).tolist(),
        "distance_mm": (distances * (25.4 / dpi)).round(4).tolist() if dpi else None,
        "channels": channel_results,
        "usable_stations": min(usable),
        "samples": samples,
        "transverse": transverse,
    }


def _masked_blur(values, valid, radius):
    support = ndimage.gaussian_filter(valid.astype(np.float32), sigma=radius, mode="nearest")
    total = ndimage.gaussian_filter(np.where(valid, values, 0), sigma=radius, mode="nearest")
    return np.divide(total, np.maximum(support, 1e-6)), support


def calibration_map(flat, dark, region, tiles, smooth, *, dpi=None):
    calibrated = apply_calibration(flat, Calibration(None, None, (1, 1, 1), False), dark=dark)
    gray = np.einsum("ijk,k->ij", calibrated.values, SCANNER_LUMA)
    valid = calibrated.valid.all(axis=2)
    if smooth > 0:
        gray, support = _masked_blur(gray, valid, smooth)
        valid &= support > .65
    rows, columns = tiles
    boundaries_y = np.rint(np.linspace(region.y, region.bottom, rows + 1)).astype(int)
    boundaries_x = np.rint(np.linspace(region.x, region.right, columns + 1)).astype(int)
    if np.min(np.diff(boundaries_x)) < 2 or np.min(np.diff(boundaries_y)) < 2:
        raise DeskError("Use fewer tiles for this small region.", field="tiles")
    chosen = gray[region.slices][valid[region.slices]]
    if len(chosen) < 10:
        raise DeskError("This flat scan has too few working pixels in the selected region.", field="region")
    global_median = float(np.median(chosen))
    if global_median <= EDGE_GAP:
        raise DeskError("The flat and dark controls have too little separation.", field="dark_id")
    table = []
    for row in range(rows):
        for column in range(columns):
            cell = Box(
                int(boundaries_x[column]), int(boundaries_y[row]),
                int(boundaries_x[column + 1] - boundaries_x[column]),
                int(boundaries_y[row + 1] - boundaries_y[row]),
            )
            sample = gray[cell.slices][valid[cell.slices]]
            median = float(np.median(sample)) if len(sample) >= .35 * cell.area else None
            result = {
                "row": row,
                "column": column,
                "region": cell.as_dict(),
                "samples": int(len(sample)),
                "flat_relative_percent": round((median / global_median - 1) * 100, 3) if median else None,
                "density_drift": round(-np.log10(median / global_median), 5) if median else None,
            }
            table.append(result)
    usable = [value["density_drift"] for value in table if value["density_drift"] is not None]
    return {
        "region": region.as_dict(),
        "tiles": table,
        "reference_transmission": round(global_median, 6),
        "p05_p95_drift": round(float(np.quantile(usable, .95) - np.quantile(usable, .05)), 5) if usable else None,
        "pixel_pitch_mm": round(25.4 / dpi, 6) if dpi else None,
    }
