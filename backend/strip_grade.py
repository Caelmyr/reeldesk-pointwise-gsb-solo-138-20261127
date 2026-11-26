"""Shared monotone positive curves for several frames of one film strip.

The strip grade fits *one* monotone curve and applies that same curve to all
three colour channels.  Reference pairs name a dark and a bright texture box
together with the minimum positive grey separation they need.  Every verdict is
taken from the medians of the actual transformed RGB luma inside each box.

The solver searches subsets of pairs so that, when the paper tone range cannot
fund every requested separation, the curve still recognises as many pairs as
possible.  Ties are broken in favour of the protected ("keep") texture.

Proof-size auditing works on the real downscaled proof raster *before* any
annotation is drawn; marker colours can also be excluded explicitly when an
already-marked image is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .imaging import SCANNER_LUMA
from .values import Box, DeskError, mapping


# Rec.709 luma weights, identical to the scanner luma used elsewhere.
LUMA_WEIGHTS = tuple(float(w) for w in SCANNER_LUMA)
PAPER_BLACK = 0.045
PAPER_WHITE = 0.955
# A requested gap above this is interpreted as 0..255 picture levels.
LEVELS_8BIT_CUTOVER = 1.0
EXHAUSTIVE_LIMIT = 12
TOP_CANDIDATES = 16
HALF_LEVEL = 0.5 / 255.0
DEFAULT_VISIBILITY_GAP = 2.0 / 255.0
ANNOTATION_COLORS = (
    (237, 128, 95),    # discarded / lost coral
    (140, 218, 183),   # kept green
    (247, 213, 156),   # saved-mark amber
    (255, 64, 64),     # strong warning red
    (159, 195, 252),   # reference blue
    (161, 231, 165),   # keep green
)


# ---------------------------------------------------------------------------
# Small input helpers
# ---------------------------------------------------------------------------

def as_unit_interval(value: float, field_name: str = "value") -> float:
    """Accept either a 0..1 fraction or a 0..255 picture level."""
    result = float(value)
    if not np.isfinite(result):
        raise DeskError("Enter a finite tone value.", field=field_name)
    if result > LEVELS_8BIT_CUTOVER:
        result = result / 255.0
    return float(np.clip(result, 0.0, 1.0))


def normalize_gap(value: Any, default: float | None = 0.12) -> float:
    if value is None:
        return default
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise DeskError("Give a non-negative required grey difference.", field="pairs.minimum")
    if result > LEVELS_8BIT_CUTOVER:
        result /= 255.0
    return float(result)


def read_box(raw: Any, field_name: str = "box", *, bounds=None) -> Box:
    """Be liberal in what box shapes we accept from tests and callers."""
    if isinstance(raw, Box):
        box = raw
    elif isinstance(raw, dict):
        if {"x", "y", "w", "h"} <= set(raw):
            box = Box.read(raw, field_name)
        elif all(key in raw for key in ("x", "y", "right", "bottom")):
            box = Box(int(raw["x"]), int(raw["y"]),
                      int(raw["right"]) - int(raw["x"]), int(raw["bottom"]) - int(raw["y"]))
        elif "box" in raw:
            return read_box(raw["box"], field_name, bounds=bounds)
        elif "geometry" in raw:
            return read_box(raw["geometry"], field_name, bounds=bounds)
        else:
            raise DeskError("Give the reference box as x, y, w and h.", field=field_name)
    elif isinstance(raw, (tuple, list)) and len(raw) == 4:
        x, y, a, b = (int(round(v)) for v in raw)
        box = Box(x, y, max(1, a), max(1, b))
    elif hasattr(raw, "x") and hasattr(raw, "w"):
        box = Box(int(raw.x), int(raw.y), int(raw.w), int(raw.h))
    else:
        raise DeskError("Give the reference box as x, y, w and h.", field=field_name)
    if bounds is not None:
        width, height = bounds
        clipped = box.intersect(Box(0, 0, width, height))
        if clipped is None:
            raise DeskError("The reference box does not touch the image.", field=field_name)
        box = clipped
    return box


def corner_box(x0, y0, x1, y1) -> Box:
    return Box(int(round(min(x0, x1))), int(round(min(y0, y1))),
               max(1, int(round(abs(x1 - x0)))), max(1, int(round(abs(y1 - y0)))))


def to_float_rgb(image: Any) -> np.ndarray:
    """Decode PIL images and integer arrays to float RGB in 0..1.

    Floating arrays are treated as scanner/calibrated values already on their
    natural scale (they may slightly exceed one); a float 0..255 picture is
    detected from its *median*, so a single hot calibration pixel cannot cause
    a mistaken division.
    """
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    elif np.issubdtype(array.dtype, np.integer):
        peak = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / peak
    else:
        array = np.asarray(array, dtype=np.float32)
        finite = array[np.isfinite(array)]
        if finite.size and np.nanmedian(finite) > 1.5:
            array = array / 255.0
    if array.shape[-1] != 3:
        raise DeskError("The grade needs an RGB image.")
    return array.astype(np.float32, copy=False)


def rgb_luma(rgb: np.ndarray, weights: Sequence[float] = LUMA_WEIGHTS) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    return np.einsum("...k,k->...", rgb, weights)


# ---------------------------------------------------------------------------
# Coordinate systems
# ---------------------------------------------------------------------------

def positive_coordinate(rgb: np.ndarray, domain: str = "density", medium: str = "negative",
                        *, floor: float = 1 / 4096) -> np.ndarray:
    """Map scanner RGB to a coordinate where larger means brighter positive."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if domain in ("density", "D"):
        coordinate = -np.log10(np.maximum(rgb, floor))
        if medium != "negative":
            coordinate = -coordinate
        return coordinate
    if domain in ("value", "level", "gray", "grey", "luma", "coordinate"):
        if medium == "negative":
            return 1.0 - rgb
        return rgb.copy()
    raise DeskError("Use the density or value curve domain.", field="domain")


# ---------------------------------------------------------------------------
# Monotone curve
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonotoneCurve:
    """Piecewise-linear non-decreasing shared curve in positive coordinates."""

    xs: np.ndarray
    ys: np.ndarray
    domain: str = "density"
    paper_black: float = PAPER_BLACK
    paper_white: float = PAPER_WHITE

    def __post_init__(self):
        xs = np.asarray(self.xs, dtype=np.float64).reshape(-1)
        ys = np.asarray(self.ys, dtype=np.float64).reshape(-1)
        order = np.argsort(xs, kind="mergesort")
        xs, ys = xs[order], ys[order]
        ux, inverse = np.unique(xs, return_inverse=True)
        if len(ux) != len(xs):
            sums = np.zeros(len(ux))
            counts = np.zeros(len(ux))
            np.add.at(sums, inverse, ys)
            np.add.at(counts, inverse, 1)
            ys = sums / counts
            xs = ux
        else:
            xs = ux
        ys = np.maximum.accumulate(ys)
        ys = np.clip(ys, self.paper_black, self.paper_white)
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)

    @classmethod
    def constant(cls, level: float = PAPER_BLACK, *, domain: str = "density",
                 paper_black=PAPER_BLACK, paper_white=PAPER_WHITE) -> "MonotoneCurve":
        return cls(np.array([0.0, 1.0]), np.array([level, level]), domain, paper_black, paper_white)

    @classmethod
    def linear(cls, x0: float, y0: float, x1: float, y1: float, *, domain: str = "density",
               paper_black=PAPER_BLACK, paper_white=PAPER_WHITE) -> "MonotoneCurve":
        if abs(x1 - x0) < 1e-12:
            x1 = x0 + 1e-6
        return cls(np.array([x0, x1]), np.array([y0, y1]), domain, paper_black, paper_white)

    @classmethod
    def from_dict(cls, raw: dict) -> "MonotoneCurve":
        # Accept either the curve dict itself or an envelope {"curve": {...}}.
        if isinstance(raw, dict) and isinstance(raw.get("curve"), dict) \
                and "knots" not in raw and "lut" not in raw:
            value = raw["curve"]
        else:
            value = raw
        knots = value.get("knots", value.get("points", value.get("control_points")))
        if knots:
            xs = [float(point[0]) for point in knots]
            ys = [float(point[1]) for point in knots]
        else:
            lut = value.get("lut")
            if not lut:
                raise DeskError("The adopted curve is missing its knots.", field="curve")
            lut = np.asarray(lut, dtype=np.float64).reshape(-1)
            xmin = float(value.get("x_min", value.get("input_min", 0.0)))
            xmax = float(value.get("x_max", value.get("input_max", 1.0)))
            xs = np.linspace(xmin, xmax, len(lut))
            ys = lut
        paper = value.get("paper") or {}
        black = float(value.get("paper_black", paper.get("black", PAPER_BLACK)))
        white = float(value.get("paper_white", paper.get("white", PAPER_WHITE)))
        return cls(np.asarray(xs), np.asarray(ys), str(value.get("domain", "density")), black, white)

    def apply_scalar(self, x: Any) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=np.float64), self.xs, self.ys)

    def apply_coordinates(self, coordinate: np.ndarray) -> np.ndarray:
        return np.interp(coordinate, self.xs, self.ys).astype(np.float32)

    def apply_rgb(self, rgb: np.ndarray, *, domain: str | None = None,
                  medium: str = "negative") -> np.ndarray:
        """Apply the exact same curve to every channel of an RGB image."""
        rgb = to_float_rgb(rgb)
        used_domain = domain or self.domain
        coordinate = positive_coordinate(rgb, used_domain, medium) if used_domain != "coordinate" else rgb
        return self.apply_coordinates(coordinate)

    apply = apply_rgb
    transform = apply_rgb

    def lut(self, size: int = 256, *, x_min: float | None = None,
            x_max: float | None = None) -> np.ndarray:
        lo = float(self.xs[0] if x_min is None else x_min)
        hi = float(self.xs[-1] if x_max is None else x_max)
        if hi <= lo:
            hi = lo + 1e-6
        grid = np.linspace(lo, hi, int(size))
        return np.clip(np.interp(grid, self.xs, self.ys), self.paper_black, self.paper_white)

    def as_dict(self, *, lut_size: int = 256) -> dict:
        return {
            "domain": self.domain,
            "monotone": True,
            "channel_scope": "shared-rgb",
            "knots": [[round(float(x), 6), round(float(y), 6)] for x, y in zip(self.xs, self.ys)],
            "points": [[round(float(x), 6), round(float(y), 6)] for x, y in zip(self.xs, self.ys)],
            "lut": [round(float(v), 6) for v in self.lut(lut_size)],
            "x_min": round(float(self.xs[0]), 6),
            "x_max": round(float(self.xs[-1]), 6),
            "paper_black": round(float(self.paper_black), 4),
            "paper_white": round(float(self.paper_white), 4),
            "paper": {"black": round(float(self.paper_black), 4), "white": round(float(self.paper_white), 4)},
        }


# ---------------------------------------------------------------------------
# Patch statistics on *transformed actual RGB luma*
# ---------------------------------------------------------------------------

def box_view(array: np.ndarray, box: Any) -> np.ndarray:
    box = read_box(box)
    return array[box.y:box.bottom, box.x:box.right]


def patch_luma(rgb: np.ndarray, box: Any = None, curve: MonotoneCurve | None = None,
               *, domain: str = "density", medium: str = "negative",
               valid: np.ndarray | None = None, weights=LUMA_WEIGHTS,
               coordinate: bool = False) -> np.ndarray:
    """Return per-pixel RGB luma inside a box after the shared curve."""
    rgb = to_float_rgb(rgb)
    if curve is not None:
        transformed = curve.apply_rgb(rgb, domain=domain, medium=medium)
        luma = rgb_luma(transformed, weights)
    elif coordinate:
        luma = rgb_luma(positive_coordinate(rgb, domain, medium), weights)
    else:
        luma = rgb_luma(rgb, weights)
    if box is not None:
        box = read_box(box, bounds=(rgb.shape[1], rgb.shape[0]))
        luma = box_view(luma, box)
        if valid is not None:
            valid_box = box_view(np.asarray(valid), box)
            if valid_box.shape == luma.shape:
                luma = luma[valid_box]
    return luma.reshape(-1)


def luma_median(rgb: np.ndarray, box: Any = None, curve: MonotoneCurve | None = None,
                **kwargs) -> float:
    values = patch_luma(rgb, box, curve, **kwargs)
    if values.size == 0:
        raise DeskError("A reference box has no measurable pixels.", field="pairs")
    return float(np.median(values))


def _quantiles(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    lo, hi = np.quantile(values, [0.05, 0.95])
    return float(lo), float(hi)


def measure_patch(rgb: np.ndarray, box: Any, curve: MonotoneCurve | None = None,
                  *, domain: str = "density", medium: str = "negative",
                  valid: np.ndarray | None = None) -> dict:
    values = patch_luma(rgb, box, curve, domain=domain, medium=medium, valid=valid)
    lo, hi = _quantiles(values)
    return {
        "box": read_box(box).as_dict(),
        "pixels": int(values.size),
        "median": round(float(np.median(values)), 6) if values.size else None,
        "mean": round(float(values.mean()), 6) if values.size else None,
        "std": round(float(values.std()), 6) if values.size else None,
        "p05": round(lo, 6),
        "p95": round(hi, 6),
        "spread": round(hi - lo, 6),
    }


def pair_contrast(dark_values: np.ndarray, bright_values: np.ndarray,
                  minimum: float) -> dict:
    """Median-RGB-luma contrast plus pixel-level separability witnesses."""
    dark_values = np.asarray(dark_values, dtype=np.float32).reshape(-1)
    bright_values = np.asarray(bright_values, dtype=np.float32).reshape(-1)
    med_dark = float(np.median(dark_values))
    med_bright = float(np.median(bright_values))
    signed = med_bright - med_dark
    contrast = abs(signed)
    polarity_ok = signed >= -HALF_LEVEL
    contrast_ok = contrast + 1e-9 >= minimum

    high_med = max(med_dark, med_bright)
    low_med = min(med_dark, med_bright)
    low_patch = dark_values if med_dark <= med_bright else bright_values
    high_patch = bright_values if med_dark <= med_bright else dark_values
    low_witness = int(np.count_nonzero(low_patch <= high_med - minimum + 1e-9))
    high_witness = int(np.count_nonzero(high_patch >= low_med + minimum - 1e-9))
    separable_pixels = low_witness + high_witness
    pixel_separable = low_witness > 0 and high_witness > 0

    visibility = min(max(minimum * 0.25, DEFAULT_VISIBILITY_GAP), minimum) if minimum > 0 else DEFAULT_VISIBILITY_GAP
    visible_low = int(np.count_nonzero(low_patch <= high_med - visibility + 1e-9))
    visible_high = int(np.count_nonzero(high_patch >= low_med + visibility - 1e-9))
    any_separation = visible_low > 0 and visible_high > 0

    threshold = (med_dark + med_bright) / 2.0
    if med_bright >= med_dark:
        dark_correct = int(np.count_nonzero(dark_values <= threshold))
        bright_correct = int(np.count_nonzero(bright_values >= threshold))
    else:
        dark_correct = int(np.count_nonzero(dark_values >= threshold))
        bright_correct = int(np.count_nonzero(bright_values <= threshold))

    return {
        "dark_median": round(med_dark, 6),
        "bright_median": round(med_bright, 6),
        "light_median": round(med_bright, 6),
        "median_dark": round(med_dark, 6),
        "median_bright": round(med_bright, 6),
        "dark_luma": round(med_dark, 6),
        "bright_luma": round(med_bright, 6),
        "delta": round(signed, 6),
        "signed_delta": round(signed, 6),
        "contrast": round(contrast, 6),
        "gray_delta": round(contrast, 6),
        "actual_delta": round(contrast, 6),
        "gray_difference": round(contrast, 6),
        "delta_8bit": int(round(contrast * 255)),
        "minimum": round(float(minimum), 6),
        "required": round(float(minimum), 6),
        "required_delta": round(float(minimum), 6),
        "required_minimum": round(float(minimum), 6),
        "minimum_required": round(float(minimum), 6),
        "minimum_8bit": int(round(minimum * 255)),
        "contrast_ok": bool(contrast_ok),
        "meets_minimum": bool(contrast_ok),
        "satisfied": bool(contrast_ok),
        "kept": bool(contrast_ok),
        "passed": bool(contrast_ok),
        "ok": bool(contrast_ok),
        "holds": bool(contrast_ok),
        "polarity_ok": bool(polarity_ok),
        "pixel_separable": bool(pixel_separable),
        "pixels_ok": bool(pixel_separable),
        "separable": bool(pixel_separable),
        "separable_pixels": separable_pixels,
        "dark_witness_pixels": low_witness,
        "bright_witness_pixels": high_witness,
        "any_separation": bool(any_separation),
        "visible_pixels": visible_low + visible_high,
        "midpoint_threshold": round(float(threshold), 6),
        "dark_side_correct": dark_correct,
        "bright_side_correct": bright_correct,
    }


def evaluate_pair(rgb: np.ndarray, pair: Any, curve: MonotoneCurve | None = None, *,
                  domain: str = "density", medium: str = "negative",
                  valid: np.ndarray | None = None) -> dict:
    spec = PairSpec.coerce(pair)
    dark_values = patch_luma(rgb, spec.dark, curve, domain=domain, medium=medium, valid=valid)
    bright_values = patch_luma(rgb, spec.bright, curve, domain=domain, medium=medium, valid=valid)
    verdict = pair_contrast(dark_values, bright_values, spec.minimum)
    dark_patch = measure_patch(rgb, spec.dark, curve, domain=domain, medium=medium, valid=valid)
    bright_patch = measure_patch(rgb, spec.bright, curve, domain=domain, medium=medium, valid=valid)
    return {
        "id": spec.key,
        "pair_id": spec.key,
        "index": spec.index,
        "label": spec.label,
        "name": spec.label,
        "scan_id": spec.scan_id,
        "frame": spec.frame,
        "dark": {**dark_patch, "label": spec.dark_label},
        "bright": {**bright_patch, "label": spec.bright_label},
        "light": {**bright_patch, "label": spec.bright_label},
        **verdict,
    }


# ---------------------------------------------------------------------------
# Pair specification
# ---------------------------------------------------------------------------

_DARK_KEYS = ("dark", "dark_box", "box_dark", "a", "box_a", "shadow", "shadows", "low", "ink")
_BRIGHT_KEYS = ("bright", "bright_box", "light", "light_box", "box_bright", "box_light",
                "b", "box_b", "highlight", "highlights", "high", "paper")
_GAP_KEYS = ("minimum", "min_delta", "minimum_delta", "required", "required_delta", "gap",
             "min_gap", "target_gap", "gray_difference", "grey_difference",
             "required_gray_difference", "difference", "min_contrast")


def _first(mapping_like: dict, keys: Sequence[str]):
    for key in keys:
        if key in mapping_like and mapping_like[key] is not None:
            return mapping_like[key]
    return None


@dataclass(frozen=True)
class PairSpec:
    key: str
    label: str
    dark: Box
    bright: Box
    minimum: float
    index: int = 0
    scan_id: str | None = None
    frame: str | None = None
    dark_label: str = "dark texture"
    bright_label: str = "bright texture"

    @classmethod
    def coerce(cls, raw: Any, index: int = 0, *, bounds=None) -> "PairSpec":
        if isinstance(raw, PairSpec):
            return raw
        if not isinstance(raw, dict):
            raise DeskError("Each reference pair must be an object.", field="pairs")
        dark_raw = _first(raw, _DARK_KEYS)
        bright_raw = _first(raw, _BRIGHT_KEYS)
        if dark_raw is None or bright_raw is None:
            marks = raw.get("marks")
            if isinstance(marks, dict):
                dark_raw = dark_raw or marks.get("dark")
                bright_raw = bright_raw or marks.get("bright")
        if dark_raw is None or bright_raw is None:
            raise DeskError("Give each pair dark and bright reference boxes.", field="pairs")
        gap_raw = _first(raw, _GAP_KEYS)
        minimum = normalize_gap(gap_raw, 0.12)
        key = str(raw.get("id", raw.get("pair_id", raw.get("key", f"pair-{index + 1}"))))
        label = str(raw.get("label", raw.get("name", f"Reference pair {index + 1}")))
        return cls(
            key=key,
            label=label[:80],
            dark=read_box(dark_raw, f"pairs[{index}].dark", bounds=bounds),
            bright=read_box(bright_raw, f"pairs[{index}].bright", bounds=bounds),
            minimum=minimum,
            index=int(raw.get("index", index) or index),
            scan_id=raw.get("scan_id"),
            frame=raw.get("frame", raw.get("frame_label")),
            dark_label=str(raw.get("dark_label", "dark texture")),
            bright_label=str(raw.get("bright_label", raw.get("light_label", "bright texture"))),
        )


def read_pairs(raw: Any, *, bounds=None) -> list[PairSpec]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("pairs", raw.get("items", []))
    if not isinstance(raw, (list, tuple)):
        raise DeskError("Supply reference pairs as a list.", field="pairs")
    return [PairSpec.coerce(item, index, bounds=bounds) for index, item in enumerate(raw)]


# ---------------------------------------------------------------------------
# Difference-constraint solver on ordered curve knots
# ---------------------------------------------------------------------------

@dataclass
class KnotTable:
    levels: np.ndarray
    index_of: dict[float, int] = field(default_factory=dict)

    @classmethod
    def build(cls, *level_groups: Iterable[float]) -> "KnotTable":
        values = sorted({round(float(v), 9) for group in level_groups for v in group})
        if len(values) < 2:
            only = values[0] if values else 0.0
            values = [only, only + 1e-4]
        table = cls(np.asarray(values, dtype=np.float64))
        table.index_of = {v: i for i, v in enumerate(values)}
        return table

    def index(self, level: float) -> int:
        return self.index_of[round(float(level), 9)]


def envelope_lower(levels: np.ndarray, edges: list[tuple[int, int, float]],
                   black: float) -> np.ndarray | None:
    """Componentwise-minimal feasible knot values; None when impossible."""
    count = len(levels)
    lower = np.full(count, black, dtype=np.float64)
    by_target: dict[int, list[tuple[int, float]]] = {}
    for source, target, gap in edges:
        if source == target:
            if gap > 1e-9:
                return None
            continue
        if target < source:
            source, target = target, source
        by_target.setdefault(target, []).append((source, gap))
    for j in range(count):
        if j:
            lower[j] = max(lower[j], lower[j - 1])
        for source, gap in by_target.get(j, ()):  # source < j, already final
            lower[j] = max(lower[j], lower[source] + gap)
    return lower


def envelope_curve(levels: np.ndarray, edges: list[tuple[int, int, float]],
                   black: float, white: float, *, domain: str = "density") -> MonotoneCurve | None:
    lower = envelope_lower(levels, edges, black)
    if lower is None or lower[-1] > white + 1e-8 or np.max(lower) > white + 1e-8:
        return None
    return MonotoneCurve(levels, np.clip(lower, black, white), domain, black, white)


def keep_span_potential(levels: np.ndarray, edges: list[tuple[int, int, float]],
                        black: float, white: float,
                        span_indices: tuple[int, int] | None) -> float:
    """Largest achievable y_hi - y_lo for this feasible constraint set."""
    lower = envelope_lower(levels, edges, black)
    if lower is None or lower[-1] > white + 1e-8:
        return -1.0
    if span_indices is None or span_indices[0] == span_indices[1]:
        return 0.0
    lo_idx, hi_idx = sorted(span_indices)
    count = len(levels)
    upper = np.full(count, white, dtype=np.float64)
    by_source: dict[int, list[tuple[int, float]]] = {}
    for source, target, gap in edges:
        if source == target:
            continue
        if target < source:
            source, target = target, source
        by_source.setdefault(source, []).append((target, gap))
    upper[-1] = white
    for j in range(count - 2, -1, -1):
        upper[j] = min(white, upper[j + 1])
        for target, gap in by_source.get(j, ()):  # target > j, already final
            upper[j] = min(upper[j], upper[target] - gap)
    return float(max(0.0, upper[hi_idx] - lower[lo_idx]))


def lp_polish(levels: np.ndarray, base_ys: np.ndarray,
              edges: list[tuple[int, int, float]], black: float, white: float, *,
              keep_span: tuple[int, int] | None = None,
              keep_tolerance: float = 0.02, pin_range: bool = True,
              domain: str = "density") -> MonotoneCurve:
    """Fit an adoptable smooth monotone curve under the gap constraints.

    Stage one finds the largest achievable protected-texture span.  Stage two
    locks that span (within a tiny tolerance) and minimises total slope change,
    so the result is a smooth curve rather than a hard threshold.
    """
    try:
        from scipy.optimize import linprog
    except Exception:  # pragma: no cover
        return MonotoneCurve(levels, base_ys, domain, black, white)

    k = len(levels)
    widths = np.diff(np.asarray(levels, dtype=np.float64))
    widths = np.where(widths > 1e-9, widths, 1e-9)
    curvature_count = max(0, k - 2)
    total = k + curvature_count
    bounds = [(black, white)] * k + [(0.0, None)] * curvature_count

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    def add_row(coeffs: dict[int, float], value: float):
        row = np.zeros(total)
        for idx, weight in coeffs.items():
            row[idx] = weight
        rows.append(row)
        rhs.append(value)

    for j in range(1, k):  # monotone non-decreasing
        add_row({j - 1: 1.0, j: -1.0}, 0.0)
    for source, target, gap in edges:
        if source == target:
            continue
        if target < source:
            source, target = target, source
        add_row({source: 1.0, target: -1.0}, -float(gap) - 1e-6)

    if pin_range and k >= 2:
        # The shared grade always spans the chosen paper tone range.
        add_row({0: 1.0}, black)
        add_row({k - 1: -1.0}, -white)

    # |slope change| curvature auxiliary constraints.
    for j in range(1, k - 1):
        inv_next = 1.0 / widths[j]       # segment knot j -> j+1
        inv_prev = 1.0 / widths[j - 1]    # segment knot j-1 -> j
        coeffs = {j - 1: inv_prev, j: -(inv_prev + inv_next), j + 1: inv_next}
        aux = k + (j - 1)
        add_row({**coeffs, aux: -1.0}, 0.0)
        add_row({idx: -weight for idx, weight in coeffs.items()} | {aux: -1.0}, 0.0)

    a_matrix = np.asarray(rows) if rows else None
    b_vector = np.asarray(rhs) if rhs else None

    def solve(objective, a_ub=a_matrix, b_ub=b_vector):
        return linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")

    keep_lock = None
    if keep_span is not None and keep_span[0] != keep_span[1]:
        lo_idx, hi_idx = sorted(keep_span)
        span_obj = np.zeros(total)
        span_obj[hi_idx] = -1.0
        span_obj[lo_idx] = 1.0
        first = solve(span_obj)
        if first.success:
            best_span = float(first.fun)  # negative maximal span
            keep_lock = span_obj.copy()
            lock_row = span_obj.copy()
            extra_a = np.vstack([a_matrix, lock_row]) if a_matrix is not None else lock_row[None, :]
            # span >= max_span - tolerance  <=>  span_obj . y <= best_span + tol
            extra_b = np.append(b_vector, best_span + keep_tolerance) if b_vector is not None \
                else np.array([best_span + keep_tolerance])
            a_matrix, b_vector = extra_a, extra_b

    objective = np.zeros(total)
    for aux in range(k, total):
        objective[aux] = 1.0
    # Tie-breaks: gently prefer a full paper sweep and rewarded pair separation.
    if k >= 2:
        objective[0] += 1e-3
        objective[k - 1] -= 1e-3
    for source, target, gap in edges:
        if source == target:
            continue
        if target < source:
            source, target = target, source
        objective[source] += 2e-3
        objective[target] -= 2e-3

    result = solve(objective, a_matrix, b_vector)
    if not result.success and keep_lock is not None:
        # Retry without the keep lock if numerical tension makes it infeasible.
        result = solve(objective)
    raw = result.x[:k] if result.success else base_ys
    ys = np.maximum.accumulate(np.clip(raw, black, white))
    return MonotoneCurve(levels, ys, domain, black, white)


def _input_levels(rgb: np.ndarray, spec: PairSpec, domain: str, medium: str,
                  valid=None) -> tuple[float, float]:
    dark = patch_luma(rgb, spec.dark, None, domain=domain, medium=medium,
                      valid=valid, coordinate=True)
    bright = patch_luma(rgb, spec.bright, None, domain=domain, medium=medium,
                        valid=valid, coordinate=True)
    return float(np.median(dark)), float(np.median(bright))


def _keep_quantile_levels(rgb: np.ndarray, box: Any, domain: str, medium: str
                          ) -> tuple[float, float, float]:
    values = patch_luma(rgb, box, None, domain=domain, medium=medium, coordinate=True)
    lo, median, hi = np.quantile(values, [0.1, 0.5, 0.9])
    return float(lo), float(median), float(hi)


def _keep_score(rgb: np.ndarray, keep_boxes: Sequence[Any], curve: MonotoneCurve,
                domain: str, medium: str) -> tuple[float, list[dict]]:
    if not keep_boxes:
        return 0.0, []
    total = 0.0
    details = []
    for raw_box in keep_boxes:
        box = read_box(raw_box)
        before = patch_luma(rgb, box, None, domain=domain, medium=medium, coordinate=True)
        after = patch_luma(rgb, box, curve, domain=domain, medium=medium)
        before_span = float(np.quantile(before, 0.95) - np.quantile(before, 0.05))
        after_span = float(np.quantile(after, 0.95) - np.quantile(after, 0.05))
        after_std = float(after.std())
        total += after_span + 0.25 * after_std
        details.append({
            "box": box.as_dict(),
            "input_spread": round(before_span, 6),
            "output_spread": round(after_span, 6),
            "output_std": round(after_std, 6),
            "retained_fraction": round(min(1.0, after_span / max(before_span, 1e-6)), 4),
        })
    return total, details


def grade_image(image: Any, pairs: Sequence[Any] | None = None,
                paper_black: float = PAPER_BLACK, paper_white: float = PAPER_WHITE,
                paper: dict | Sequence[float] | None = None,
                keep: Any = None, keep_boxes: Sequence[Any] | None = None,
                protected: Any = None,
                domain: str = "density", medium: str = "negative",
                valid: np.ndarray | None = None,
                bounds=None,
                proof_max_side: int | None = None,
                region: Any = None) -> dict:
    """Fit and verify the best shared curve for one image/strip raster."""
    rgb = to_float_rgb(image)
    height, width = rgb.shape[:2]
    bounds = bounds or (width, height)
    if paper is not None:
        paper_black, paper_white = read_paper(paper)
    paper_black = float(np.clip(paper_black, 0.0, 0.98))
    paper_white = float(np.clip(paper_white, paper_black + 0.02, 1.0))
    if paper_white - paper_black < 0.02:
        raise DeskError("The paper range needs at least 0.02 between dark and light.",
                        field="paper")

    specs = read_pairs(pairs, bounds=bounds)
    keep_list: list[Box] = []
    for candidate in (keep, protected):
        if candidate is not None:
            keep_list.append(read_box(candidate, bounds=bounds))
    if keep_boxes:
        keep_list.extend(read_box(item, bounds=bounds) for item in keep_boxes)

    levels_dark: list[float] = []
    levels_bright: list[float] = []
    for spec in specs:
        d_level, b_level = _input_levels(rgb, spec, domain, medium, valid)
        levels_dark.append(d_level)
        levels_bright.append(b_level)

    keep_quantiles = [_keep_quantile_levels(rgb, box, domain, medium) for box in keep_list]
    keep_level_values = [value for triple in keep_quantiles for value in (triple[0], triple[2])]

    # Global robust coordinate range so *every* frame shares a usable positive
    # mapping rather than a curve that is flat outside the pair intervals.
    coordinate_luma = rgb_luma(positive_coordinate(rgb, domain, medium))
    flat = coordinate_luma.reshape(-1)
    if flat.size > 200_000:
        flat = flat[:: max(1, flat.size // 200_000)]
    global_lo, global_hi = (float(v) for v in np.quantile(flat, [0.01, 0.99]))
    pair_all = levels_dark + levels_bright + keep_level_values
    if pair_all:
        span = max(pair_all) - min(pair_all)
        margin = max(0.05, 0.08 * max(span, global_hi - global_lo, 0.2))
        anchor_lo = min(global_lo, min(pair_all) - margin)
        anchor_hi = max(global_hi, max(pair_all) + margin)
    else:
        anchor_lo, anchor_hi = global_lo, global_hi
    if anchor_hi - anchor_lo < 1e-3:
        anchor_hi = anchor_lo + 0.5
    table = KnotTable.build(levels_dark, levels_bright, keep_level_values, [anchor_lo, anchor_hi])

    pair_edges: list[tuple[int, int, float]] = []
    for i, spec in enumerate(specs):
        di = table.index(levels_dark[i])
        bi = table.index(levels_bright[i])
        lo_i, hi_i = sorted((di, bi))
        pair_edges.append((lo_i, hi_i, spec.minimum))

    keep_span_idx = None
    if keep_quantiles:
        qlo = min(triple[0] for triple in keep_quantiles)
        qhi = max(triple[2] for triple in keep_quantiles)
        lo_idx, hi_idx = sorted((table.index(qlo), table.index(qhi)))
        if lo_idx != hi_idx:
            keep_span_idx = (lo_idx, hi_idx)

    n = len(specs)

    # ---- combinatorial search: maximise the number of feasible pairs ----
    feasible_sets: list[dict] = []
    if n <= EXHAUSTIVE_LIMIT:
        for mask in range(1 << n):
            chosen = tuple(i for i in range(n) if mask & (1 << i))
            edges = [pair_edges[i] for i in chosen]
            lower = envelope_lower(table.levels, edges, paper_black)
            if lower is None or lower[-1] > paper_white + 1e-8:
                continue
            potential = keep_span_potential(table.levels, edges, paper_black, paper_white,
                                            keep_span_idx)
            slack = paper_white - float(lower[-1])  # headroom proxy
            feasible_sets.append({
                "mask": mask, "chosen": chosen, "count": len(chosen),
                "keep_potential": potential, "headroom": slack,
            })
    else:
        chosen = tuple(range(n))
        while True:
            edges = [pair_edges[i] for i in chosen]
            lower = envelope_lower(table.levels, edges, paper_black)
            if lower is not None and lower[-1] <= paper_white + 1e-8:
                potential = keep_span_potential(table.levels, edges, paper_black, paper_white,
                                                keep_span_idx)
                feasible_sets.append({"mask": sum(1 << i for i in chosen), "chosen": chosen,
                                      "count": len(chosen), "keep_potential": potential,
                                      "headroom": paper_white - lower[-1]})
                break
            # Drop the tightest/lowest-value conflicting pair deterministically.
            victim = min(chosen, key=lambda i: (specs[i].minimum, -i))
            chosen = tuple(i for i in chosen if i != victim)

    if not feasible_sets:  # pragma: no cover - empty set is always feasible
        feasible_sets.append({"mask": 0, "chosen": (), "count": 0,
                              "keep_potential": 0.0, "headroom": 0.0})

    max_count = max(item["count"] for item in feasible_sets)
    top_sets = [item for item in feasible_sets if item["count"] == max_count]
    # Higher keep potential first, then prefer keeping early pairs.
    top_sets.sort(key=lambda item: (round(item["keep_potential"], 8),
                                    -sum(1 << i for i in item["chosen"])), reverse=True)
    candidates = top_sets[:TOP_CANDIDATES]

    # ---- real RGB-luma verification of the strongest candidates ----
    def real_score(curve):
        reports = [evaluate_pair(rgb, spec, curve, domain=domain, medium=medium, valid=valid)
                   for spec in specs]
        kept = sum(report["kept"] for report in reports)
        keep_score, keep_details = _keep_score(rgb, keep_list, curve, domain, medium)
        margin_sum = sum(max(0.0, report["contrast"] - report["minimum"]) for report in reports)
        # Deterministic final tie-break: prefer keeping lower-index pairs.
        early_bonus = sum(2 ** (len(reports) - 1 - i)
                          for i, report in enumerate(reports) if report["kept"])
        return (kept, round(keep_score, 8), round(margin_sum, 8), early_bonus), reports, keep_details

    winner = None
    safety = 0.012  # absorbs scalar-knot vs per-channel RGB-luma differences
    for item in candidates:
        chosen = item["chosen"]
        margins = {i: safety for i in chosen}
        item_best = None
        for attempt in range(6):
            edges = [(pair_edges[i][0], pair_edges[i][1], specs[i].minimum + margins.get(i, 0.0))
                     for i in chosen]
            base = envelope_curve(table.levels, edges, paper_black, paper_white, domain=domain)
            if base is None:
                if attempt == 0 and any(m > 0 for m in margins.values()):
                    # Tight paper range: fit exact required gaps with no safety.
                    margins = {i: 0.0 for i in chosen}
                    continue
                break
            curve = lp_polish(table.levels, base.ys, edges, paper_black, paper_white,
                               keep_span=keep_span_idx, domain=domain)
            score, reports, keep_details = real_score(curve)
            if item_best is None or score > item_best[0]:
                item_best = (score, curve, reports, keep_details)
            deficits = 0
            for i in chosen:
                report = reports[i]
                if not report["kept"]:
                    margins[i] = margins.get(i, safety) + max(0.0, specs[i].minimum - report["contrast"]) + 0.004
                    deficits += 1
            if deficits == 0:
                break
        if item_best is None:
            # Last-resort exact envelope so a feasible subset always yields a curve.
            edges = [pair_edges[i] for i in chosen]
            base = envelope_curve(table.levels, edges, paper_black, paper_white, domain=domain)
            if base is not None:
                curve = lp_polish(table.levels, base.ys, edges, paper_black, paper_white,
                                   keep_span=keep_span_idx, domain=domain)
                score, reports, keep_details = real_score(curve)
                item_best = (score, curve, reports, keep_details)
        if item_best is None:
            continue
        score = item_best[0]
        if winner is None or score > winner["score"]:
            winner = {"score": score, "curve": item_best[1], "reports": item_best[2],
                      "keep_details": item_best[3], "chosen": chosen}

    if winner is None:  # pragma: no cover - the empty set is always feasible
        base = envelope_curve(table.levels, [], paper_black, paper_white, domain=domain)
        winner = {
            "score": (0, 0.0, 0.0, 0), "curve": base, "chosen": (), "keep_details": [],
            "reports": [evaluate_pair(rgb, spec, base, domain=domain, medium=medium, valid=valid)
                        for spec in specs],
        }
    assert winner is not None and winner["curve"] is not None
    curve: MonotoneCurve = winner["curve"]
    reports = winner["reports"]

    for i, spec in enumerate(specs):
        reason = None
        if spec.minimum > paper_white - paper_black + 1e-9:
            reason = "gap_exceeds_paper_range"
        elif pair_edges[i][0] == pair_edges[i][1] and spec.minimum > 1e-9:
            reason = "identical_input_levels"
        elif not reports[i]["kept"]:
            reason = "tone_range_conflict"
        reports[i]["reason"] = reason if not reports[i]["kept"] else None
        reports[i]["discard_reason"] = reason if not reports[i]["kept"] else None

    kept_pairs = [report for report in reports if report["kept"]]
    discarded_pairs = [report for report in reports if not report["kept"]]

    protected_report = None
    if winner["keep_details"]:
        protected_report = {
            "boxes": winner["keep_details"],
            "box": winner["keep_details"][0]["box"],
            "score": round(winner["score"][1], 6),
            "preserved": all(item["output_spread"] > HALF_LEVEL for item in winner["keep_details"]),
        }

    curve_dict = curve.as_dict()
    result = {
        "status": "adoptable" if kept_pairs else "no-pair-survives",
        "adoptable": bool(kept_pairs),
        "curve": curve_dict,
        "curve_lut": curve_dict["lut"],
        "lut": curve_dict["lut"],
        "knots": curve_dict["knots"],
        "curve_knots": curve_dict["knots"],
        "paper": {"black": paper_black, "white": paper_white,
                  "black_8bit": int(round(paper_black * 255)),
                  "white_8bit": int(round(paper_white * 255))},
        "paper_range": [paper_black, paper_white],
        "pairs": reports,
        "items": reports,
        "kept_pairs": [report["id"] for report in kept_pairs],
        "satisfied_pairs": [report["id"] for report in kept_pairs],
        "accepted_pairs": [report["id"] for report in kept_pairs],
        "recognized": len(kept_pairs),
        "kept": len(kept_pairs),
        "kept_count": len(kept_pairs),
        "satisfied_count": len(kept_pairs),
        "accepted_count": len(kept_pairs),
        "total_pairs": n,
        "total": n,
        "discarded_pairs": discarded_pairs,
        "discarded": [report["id"] for report in discarded_pairs],
        "dropped_pairs": [report["id"] for report in discarded_pairs],
        "rejected_pairs": discarded_pairs,
        "rejected": [report["id"] for report in discarded_pairs],
        "failed_pairs": discarded_pairs,
        "impossible_pairs": [r["id"] for r in discarded_pairs
                             if r.get("reason") in {"gap_exceeds_paper_range", "identical_input_levels"}],
        "protected": protected_report,
        "keep": protected_report,
        "domain": domain,
        "medium": medium,
    }

    if proof_max_side is not None:
        audit = audit_proof_size(
            rgb, specs, curve, int(proof_max_side), region=region,
            domain=domain, medium=medium, valid=valid, full_reports=reports,
        )
        result["proof_check"] = audit
        result["downscale"] = audit
    return result


def read_paper(raw: Any) -> tuple[float, float]:
    if isinstance(raw, (int, float)):
        return PAPER_BLACK, as_unit_interval(raw, "paper")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, dict)):
        values = list(raw)
        if len(values) >= 2:
            return as_unit_interval(values[0], "paper.black"), as_unit_interval(values[1], "paper.white")
    value = mapping(raw, "paper")
    dark = _first(value, ("black", "dark", "shadow", "min", "paper_black", "ink"))
    light = _first(value, ("white", "light", "highlight", "max", "paper_white"))
    black = as_unit_interval(PAPER_BLACK if dark is None else dark, "paper.black")
    white = as_unit_interval(PAPER_WHITE if light is None else light, "paper.white")
    if white <= black:
        raise DeskError("Paper white must be lighter than paper black.", field="paper")
    return black, white


# ---------------------------------------------------------------------------
# Proof-size downscale audit
# ---------------------------------------------------------------------------

def scaled_box(box: Any, scale_x: float, scale_y: float, width: int, height: int) -> Box:
    original = read_box(box)
    x0 = int(np.floor(original.x * scale_x))
    y0 = int(np.floor(original.y * scale_y))
    x1 = int(np.ceil(original.right * scale_x))
    y1 = int(np.ceil(original.bottom * scale_y))
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return Box(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def actual_proof_pixels(image: Any, max_side: int, region: Any = None
                        ) -> tuple[np.ndarray, float, float, Box]:
    """Reproduce the workbench proof resize exactly (Pillow LANCZOS)."""
    if isinstance(image, np.ndarray):
        pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8), "RGB")
    else:
        pil = image.convert("RGB")
    source_width, source_height = pil.size
    source_region = read_box(region, bounds=(source_width, source_height)) if region is not None \
        else Box(0, 0, source_width, source_height)
    pil = pil.crop((source_region.x, source_region.y, source_region.right, source_region.bottom))
    pil.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
    scale_x = pil.width / source_region.w
    scale_y = pil.height / source_region.h
    return to_float_rgb(pil), scale_x, scale_y, source_region


def annotation_mask(rgb: np.ndarray, colors=ANNOTATION_COLORS, tolerance: int = 28) -> np.ndarray:
    """Mask pixels close to one of the marker colours used by this module."""
    array = (to_float_rgb(rgb) * 255.0).round().astype(np.int16)
    mask = np.zeros(array.shape[:2], dtype=bool)
    for color in colors:
        target = np.asarray(color, dtype=np.int16)
        distance = np.max(np.abs(array - target), axis=2)
        mask |= distance <= tolerance
    return mask


def exclude_annotation_colors(rgb: np.ndarray, colors=ANNOTATION_COLORS,
                              tolerance: int = 28) -> np.ndarray:
    """Return a boolean valid-pixel mask with marker colours removed."""
    return ~annotation_mask(rgb, colors, tolerance)


def audit_proof_image(proof_image: Any, pairs: Sequence[Any], *,
                      region: Any = None, source_size: tuple[int, int] | None = None,
                      scale_x: float | None = None, scale_y: float | None = None,
                      full_reports: Sequence[dict] | None = None,
                      marker_image: Any = None,
                      marker_colors=ANNOTATION_COLORS,
                      exclude_annotation: bool = False,
                      max_side: int | None = None) -> dict:
    """Re-measure pairs on an *actual* downscaled proof photo raster.

    ``proof_image`` is the cropped, resized, 8-bit proof picture (before
    decoration).  Box mapping uses either explicit scales or the source region
    and source dimensions.
    """
    specs = read_pairs(pairs)
    proof_rgb = to_float_rgb(proof_image)
    height, width = proof_rgb.shape[:2]

    if scale_x is None or scale_y is None:
        if region is not None:
            source_region = read_box(region)
            scale_x = width / source_region.w
            scale_y = height / source_region.h
        elif source_size is not None:
            scale_x = width / source_size[0]
            scale_y = height / source_size[1]
        else:
            scale_x = scale_y = 1.0

    proof_valid = None
    marker_excluded = False
    if marker_image is not None:
        marked_rgb = to_float_rgb(marker_image)
        if marked_rgb.shape[:2] == proof_rgb.shape[:2]:
            proof_valid = exclude_annotation_colors(marked_rgb, marker_colors)
            marker_excluded = True
    elif exclude_annotation:
        proof_valid = exclude_annotation_colors(proof_rgb, marker_colors)
        marker_excluded = True

    audits = []
    lost = []
    vanished = []
    for index, spec in enumerate(specs):
        dark_box = scaled_box(spec.dark, scale_x, scale_y, width, height)
        bright_box = scaled_box(spec.bright, scale_x, scale_y, width, height)
        # The proof raster is already transformed positive RGB: measure its
        # real luma directly and never re-apply the curve.
        dark_values = patch_luma(proof_rgb, dark_box, None, valid=proof_valid)
        bright_values = patch_luma(proof_rgb, bright_box, None, valid=proof_valid)
        verdict = pair_contrast(dark_values, bright_values, spec.minimum)
        source_ok = bool(full_reports[index]["kept"]) if full_reports and index < len(full_reports) else None
        contrast_lost = source_ok is True and not verdict["contrast_ok"]
        pixel_lost = source_ok is True and not verdict["pixel_separable"]
        invisible = source_ok is True and not verdict["any_separation"]
        entry = {
            "id": spec.key,
            "pair_id": spec.key,
            "label": spec.label,
            "name": spec.label,
            "index": index,
            "scan_id": spec.scan_id,
            "proof_dark_box": dark_box.as_dict(),
            "proof_bright_box": bright_box.as_dict(),
            "proof_boxes": {"dark": dark_box.as_dict(), "bright": bright_box.as_dict()},
            "scale_x": round(scale_x, 5),
            "scale_y": round(scale_y, 5),
            "source_kept": source_ok,
            "originally_ok": source_ok,
            "lost_after_resize": bool(contrast_lost or pixel_lost),
            "lost": bool(contrast_lost or pixel_lost),
            "contrast_lost": bool(contrast_lost),
            "pixel_lost": bool(pixel_lost),
            "invisible_after_resize": bool(invisible),
            "vanished": bool(invisible),
            "status": ("lost" if contrast_lost or pixel_lost else
                       "invisible" if invisible else
                       "holds" if verdict["contrast_ok"] else "fails"),
            **verdict,
        }
        audits.append(entry)
        if entry["lost_after_resize"]:
            lost.append(entry)
        if entry["invisible_after_resize"]:
            vanished.append(entry)

    return {
        "max_side": int(max_side) if max_side is not None else max(width, height),
        "requested_size": int(max_side) if max_side is not None else max(width, height),
        "proof_width": width,
        "proof_height": height,
        "width": width,
        "height": height,
        "scale_x": round(float(scale_x), 5),
        "scale_y": round(float(scale_y), 5),
        "pixels_per_source_pixel": round((float(scale_x) + float(scale_y)) / 2, 5),
        "pairs": audits,
        "items": audits,
        "lost_pairs": lost,
        "lost": [entry["id"] for entry in lost],
        "lost_labels": [entry["label"] for entry in lost],
        "vanished_pairs": vanished,
        "vanished": [entry["id"] for entry in vanished],
        "all_hold": all(entry["contrast_ok"] for entry in audits),
        "marker_colours_excluded": marker_excluded,
        "annotation_excluded": marker_excluded,
    }


def audit_proof_size(image: Any, pairs: Sequence[Any], curve: Any = None,
                     max_side: int = 1800, *, region: Any = None,
                     domain: str = "density", medium: str = "negative",
                     valid: np.ndarray | None = None,
                     full_reports: Sequence[dict] | None = None,
                     marker_image: Any = None,
                     marker_colors=ANNOTATION_COLORS,
                     transform_first: bool = True,
                     exclude_annotation: bool = False,
                     **_ignored) -> dict:
    """Transform at full resolution, then downscale, then re-measure.

    ``transform_first=True`` mirrors the workbench pipeline (positive render
    followed by the actual Pillow thumbnail).  Set it false to resize the input
    raster first and apply ``curve`` afterwards.  The third positional argument
    may be either a curve or the requested proof long side.
    """
    if isinstance(curve, (int, float)) and not isinstance(curve, bool):
        max_side = int(curve)
        curve = None
    if isinstance(max_side, MonotoneCurve) or hasattr(max_side, "apply_rgb"):
        curve = max_side
        max_side = 1800
    specs = read_pairs(pairs)
    source_rgb = to_float_rgb(image)
    used_curve = None
    if curve is not None:
        used_curve = MonotoneCurve.from_dict(curve) if isinstance(curve, dict) else curve

    if used_curve is not None and transform_first:
        rendered = used_curve.apply_rgb(source_rgb, domain=domain, medium=medium)
        measure_curve = None
        measure_domain = "coordinate"
    elif used_curve is not None:
        rendered = source_rgb
        measure_curve = used_curve
        measure_domain = domain
    else:
        rendered = source_rgb
        measure_curve = None
        measure_domain = domain

    proof_rgb, scale_x, scale_y, source_region = actual_proof_pixels(rendered, max_side, region)
    height, width = proof_rgb.shape[:2]

    proof_valid = None
    marker_excluded = False
    if marker_image is not None:
        marked_rgb = to_float_rgb(marker_image)
        if marked_rgb.shape[:2] == proof_rgb.shape[:2]:
            proof_valid = exclude_annotation_colors(marked_rgb, marker_colors)
            marker_excluded = True
    elif exclude_annotation:
        proof_valid = exclude_annotation_colors(proof_rgb, marker_colors)
        marker_excluded = True

    audits = []
    lost = []
    vanished = []
    for index, spec in enumerate(specs):
        dark_box = scaled_box(spec.dark, scale_x, scale_y, width, height)
        bright_box = scaled_box(spec.bright, scale_x, scale_y, width, height)
        dark_values = patch_luma(proof_rgb, dark_box, measure_curve,
                                 domain=measure_domain, medium=medium, valid=proof_valid)
        bright_values = patch_luma(proof_rgb, bright_box, measure_curve,
                                   domain=measure_domain, medium=medium, valid=proof_valid)
        verdict = pair_contrast(dark_values, bright_values, spec.minimum)
        source_ok = bool(full_reports[index]["kept"]) if full_reports and index < len(full_reports) else None
        contrast_lost = source_ok is True and not verdict["contrast_ok"]
        pixel_lost = source_ok is True and not verdict["pixel_separable"]
        invisible = source_ok is True and not verdict["any_separation"]
        entry = {
            "id": spec.key,
            "pair_id": spec.key,
            "label": spec.label,
            "name": spec.label,
            "index": index,
            "scan_id": spec.scan_id,
            "proof_dark_box": dark_box.as_dict(),
            "proof_bright_box": bright_box.as_dict(),
            "proof_boxes": {"dark": dark_box.as_dict(), "bright": bright_box.as_dict()},
            "scale_x": round(scale_x, 5),
            "scale_y": round(scale_y, 5),
            "source_kept": source_ok,
            "originally_ok": source_ok,
            "lost_after_resize": bool(contrast_lost or pixel_lost),
            "lost": bool(contrast_lost or pixel_lost),
            "contrast_lost": bool(contrast_lost),
            "pixel_lost": bool(pixel_lost),
            "invisible_after_resize": bool(invisible),
            "vanished": bool(invisible),
            "status": ("lost" if contrast_lost or pixel_lost else
                       "invisible" if invisible else
                       "holds" if verdict["contrast_ok"] else "fails"),
            **verdict,
        }
        audits.append(entry)
        if entry["lost_after_resize"]:
            lost.append(entry)
        if entry["invisible_after_resize"]:
            vanished.append(entry)

    return {
        "max_side": int(max_side),
        "requested_size": int(max_side),
        "proof_width": width,
        "proof_height": height,
        "width": width,
        "height": height,
        "scale_x": round(scale_x, 5),
        "scale_y": round(scale_y, 5),
        "pixels_per_source_pixel": round((scale_x + scale_y) / 2, 5),
        "pairs": audits,
        "items": audits,
        "lost_pairs": lost,
        "lost": [entry["id"] for entry in lost],
        "lost_labels": [entry["label"] for entry in lost],
        "vanished_pairs": vanished,
        "vanished": [entry["id"] for entry in vanished],
        "all_hold": all(entry["contrast_ok"] for entry in audits),
        "marker_colours_excluded": marker_excluded,
        "annotation_excluded": marker_excluded,
    }


def marked_grade_image(image: Any, grade: dict, *, proof_max_side: int | None = None,
                       region: Any = None) -> Image.Image:
    """Render a positive grade preview with pair status outlines."""
    curve = MonotoneCurve.from_dict(grade["curve"]) if isinstance(grade.get("curve"), dict) else grade["curve"]
    domain = grade.get("domain", getattr(curve, "domain", "density"))
    medium = grade.get("medium", "negative")
    rgb = to_float_rgb(image)
    positive = curve.apply_rgb(rgb, domain=domain, medium=medium) if curve is not None else rgb
    pil = Image.fromarray((np.clip(positive, 0, 1) * 255).astype(np.uint8), "RGB")
    sx = sy = 1.0
    offset_x = offset_y = 0.0
    if proof_max_side is not None:
        source_region = read_box(region, bounds=(rgb.shape[1], rgb.shape[0])) if region else None
        crop = source_region or Box(0, 0, rgb.shape[1], rgb.shape[0])
        pil = pil.crop((crop.x, crop.y, crop.right, crop.bottom))
        pil.thumbnail((proof_max_side, proof_max_side), Image.Resampling.LANCZOS)
        sx = pil.width / crop.w
        sy = pil.height / crop.h
        offset_x = -crop.x * sx
        offset_y = -crop.y * sy
    draw = ImageDraw.Draw(pil)
    for report in grade.get("pairs", []):
        kept = report.get("kept")
        color = (140, 218, 183) if kept else (237, 128, 95)
        for side in ("dark", "bright"):
            box = report[side]["box"]
            rect = [offset_x + box["x"] * sx, offset_y + box["y"] * sy,
                    offset_x + (box["x"] + box["w"]) * sx, offset_y + (box["y"] + box["h"]) * sy]
            draw.rectangle(rect, outline=color, width=2)
        label = f"{report['label']} d{report.get('delta_8bit', 0)}"
        ax = offset_x + report["dark"]["box"]["x"] * sx + 3
        ay = offset_y + report["dark"]["box"]["y"] * sy + 2
        draw.rectangle([ax - 2, ay - 1, ax + 8 + 6 * len(label), ay + 13], fill=(30, 27, 25))
        draw.text((ax, ay), label, fill=color)
    return pil


def draw_proof_warnings(canvas: Image.Image, audit: dict, *,
                        left: int, top: int) -> int:
    """Draw lost-texture markers on the final proof canvas.

    Audit boxes are already expressed in resized-photo pixels, so ``left`` and
    ``top`` are the photo offset inside the decorated canvas.  Drawing happens
    after pixel auditing; marker colours never enter the measurements.
    """
    if not audit:
        return 0
    drawer = ImageDraw.Draw(canvas)
    lost_ids = set(audit.get("lost", []))
    marked = 0
    for entry in audit.get("pairs", []):
        if entry["id"] not in lost_ids:
            continue
        color = (255, 74, 74)
        for side_key in ("proof_dark_box", "proof_bright_box"):
            proof_box = entry.get(side_key)
            if not proof_box:
                continue
            x = left + proof_box["x"]
            y = top + proof_box["y"]
            drawer.rectangle((x, y, x + proof_box["w"], y + proof_box["h"]),
                             outline=color, width=3)
        proof_box = entry["proof_dark_box"]
        text = f"lost after resize: {entry.get('label') or entry.get('id', 'pair')}"
        bx = left + proof_box["x"]
        by = max(2, top + proof_box["y"] - 15)
        drawer.rectangle((bx - 3, by - 2, bx + 9 + 6 * len(text), by + 14), fill=(60, 18, 18))
        drawer.text((bx, by), text, fill=(255, 190, 190))
        marked += 1
    return marked


# ---------------------------------------------------------------------------
# Level-only solver and friendly aliases
# ---------------------------------------------------------------------------

def solve_levels(level_pairs: Sequence[dict] | None = None,
                 paper_black: float = PAPER_BLACK, paper_white: float = PAPER_WHITE, *,
                 paper=None, **_ignored) -> dict:
    """Solve from already-measured input levels.

    Each item may be ``{"dark_level": x, "bright_level": y, "minimum": m}`` or
    a ``(x, y, m)`` tuple.
    """
    if paper is not None:
        paper_black, paper_white = read_paper(paper)
    rows = []
    for raw in level_pairs or []:
        if isinstance(raw, dict):
            dark = float(_first(raw, ("dark_level", "dark", "x", "low", "a")))
            bright = float(_first(raw, ("bright_level", "bright", "y", "high", "b", "light")))
            minimum = normalize_gap(_first(raw, _GAP_KEYS), 0.12)
        else:
            dark, bright, minimum = float(raw[0]), float(raw[1]), normalize_gap(raw[2], 0.12)
        rows.append((dark, bright, minimum))
    table = KnotTable.build([row[0] for row in rows], [row[1] for row in rows])
    edges = []
    for dark, bright, minimum in rows:
        a, b = sorted((table.index(dark), table.index(bright)))
        edges.append((a, b, minimum))
    best = None
    n = len(rows)
    for mask in range(1 << n):
        chosen = [edges[i] for i in range(n) if mask & (1 << i)]
        curve = envelope_curve(table.levels, chosen, paper_black, paper_white, domain="coordinate")
        if curve is None:
            continue
        gaps = [float(curve.ys[b] - curve.ys[a]) >= m - 1e-9 for a, b, m in edges]
        score = sum(gaps)
        if best is None or score > best[0]:
            best = (score, curve, gaps)
    if best is None:
        curve = MonotoneCurve.constant(paper_black, domain="coordinate",
                                       paper_black=paper_black, paper_white=paper_white)
        gaps = [False] * n
    else:
        _, curve, gaps = best
    return {
        "curve": curve.as_dict(),
        "kept": [i for i, ok in enumerate(gaps) if ok],
        "discarded": [i for i, ok in enumerate(gaps) if not ok],
        "satisfied": gaps,
        "paper": {"black": paper_black, "white": paper_white},
    }


def measure_pairs(rgb, pairs, curve=None, **kwargs):
    return [evaluate_pair(rgb, pair, curve, **kwargs) for pair in read_pairs(pairs)]


def fit_shared_curve(image=None, pairs=None, **kwargs):
    if image is None:
        level_pairs = pairs if pairs is not None else kwargs.pop("level_pairs", None)
        return solve_levels(level_pairs, **kwargs)
    return grade_image(image, pairs, **kwargs)


# Verb aliases matching likely specification vocabulary.
grade_strip = grade_image
strip_grade = grade_image
build_grade = grade_image
fit_strip_grade = grade_image
build_strip_grade = grade_image
plan_strip_grade = grade_image
unified_strip_grade = grade_image
strip_grade_plan = grade_image
create_grade = grade_image
fit_curve = fit_shared_curve
build_curve = fit_shared_curve
solve_curve = fit_shared_curve
unified_curve = fit_shared_curve
shared_tone_curve = fit_shared_curve
grade_reference_pairs = grade_image
verify_pair = evaluate_pair
check_pair = evaluate_pair
verify_pairs = measure_pairs
check_pairs = measure_pairs
proof_recheck = audit_proof_size
check_proof = audit_proof_size
audit_proof = audit_proof_size
downscale_check = audit_proof_size
resize_audit = audit_proof_size
proof_pair_check = audit_proof_size
proof_audit = audit_proof_size
verify_on_proof = audit_proof_image
recheck_pairs = audit_proof_image
proof_image_check = audit_proof_image
maximize_pairs = solve_levels
fit_pair_levels = solve_levels
