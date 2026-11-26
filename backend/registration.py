"""Repeat-pass registration, in-source coordinate inspection, and pair differencing."""

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal

from .imaging import SCANNER_LUMA, _bilinear_valid
from .values import (
    Box, DeskError, Point, PRIMARY_CHANNELS, mapping, number, one_of,
    round_channels, sequence,
)


REGISTRATION_MODELS = frozenset({"shift", "similarity", "affine"})
PAIR_PREVIEWS = frozenset({"overlay", "difference", "checker", "peer"})


@dataclass(frozen=True)
class Pair:
    source: Point
    peer: Point

    @classmethod
    def read(cls, value, index, source_size, peer_size):
        field = f"pairs[{index}]"
        value = mapping(value, field)
        source = Point.read(value.get("source"), f"{field}.source").bounded(*source_size, field=f"{field}.source")
        peer = Point.read(value.get("peer"), f"{field}.peer").bounded(*peer_size, field=f"{field}.peer")
        return cls(source, peer)

    def as_dict(self):
        return {"source": self.source.as_dict(), "peer": self.peer.as_dict()}


@dataclass(frozen=True)
class Transform:
    """Map canonical source centers to peer centers, before any viewing scale."""

    data: tuple[float, float, float, float, float, float]

    @classmethod
    def read(cls, value):
        data = sequence(value, "transform", minimum=6, maximum=6)
        return cls.checked(tuple(number(item, f"transform[{index}]") for index, item in enumerate(data)))

    @classmethod
    def checked(cls, values):
        instance = cls(tuple(float(value) for value in values))
        matrix = instance.matrix
        basis = matrix[:, :2]
        singular = np.linalg.svd(basis, compute_uv=False)
        if not np.isfinite(matrix).all() or singular[1] < .16 or singular[0] > 6:
            raise DeskError("These landmarks produce a collapsed or implausibly stretched image.", field="pairs")
        if np.linalg.det(basis) <= 0:
            raise DeskError("The alignment would mirror the film. Import a correctly oriented scan.", field="pairs")
        return instance

    @property
    def matrix(self):
        return np.asarray(self.data, dtype=np.float64).reshape(2, 3)

    def as_list(self):
        return [round(value, 10) for value in self.data]

    def project(self, point):
        x, y = self.grid(point.x, point.y)
        return Point(float(x), float(y))

    def grid(self, x, y):
        a, b, tx, c, d, ty = self.data
        return a * x + b * y + tx, c * x + d * y + ty



def read_pairs(value, source_size, peer_size):
    entries = sequence(value, "pairs", minimum=1, maximum=15)
    result = [Pair.read(item, index, source_size, peer_size) for index, item in enumerate(entries)]
    return result


def _fit_shift(source, peer, weights):
    dx, dy = np.average(peer - source, axis=0, weights=weights)
    return Transform((1, 0, float(dx), 0, 1, float(dy)))


def _fit_similarity(source, peer, weights):
    count = len(source)
    x, y = source.T
    matrix = np.zeros((2 * count, 4), dtype=np.float64)
    matrix[0::2, :2] = np.column_stack((x, -y))
    matrix[1::2, :2] = np.column_stack((y, x))
    matrix[0::2, 2] = 1
    matrix[1::2, 3] = 1
    expanded = np.repeat(np.sqrt(weights), 2)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix * expanded[:, None], peer.reshape(-1) * expanded, rcond=None)
    if rank < 4:
        raise DeskError("Place the landmarks farther apart for rotation and scale.", field="pairs")
    a, b, tx, ty = coefficients
    return Transform.checked((a, -b, tx, b, a, ty))


def _fit_affine(source, peer, weights):
    basis = np.column_stack((source, np.ones(len(source))))
    expanded = np.sqrt(weights)[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(basis * expanded, peer * expanded, rcond=None)
    if rank < 3:
        raise DeskError("Use at least three landmarks that do not lie on a line.", field="pairs")
    return Transform.checked(tuple(coefficients.T.ravel()))


def _fit(mode, source, peer, weights):
    if mode == "shift":
        return _fit_shift(source, peer, weights)
    if mode == "similarity":
        return _fit_similarity(source, peer, weights)
    return _fit_affine(source, peer, weights)


def _mapped_error(transform, source, peer):
    predicted = np.column_stack(transform.grid(source[:, 0], source[:, 1]))
    return predicted, np.linalg.norm(predicted - peer, axis=1)


def estimate_landmarks(mode, pairs, robust=False):
    one_of(mode, REGISTRATION_MODELS, "model")
    minimum = {"shift": 1, "similarity": 2, "affine": 3}[mode]
    if len(pairs) < minimum:
        raise DeskError(f"Add at least {minimum} landmark pairs for {mode}.", field="pairs")
    source = np.array([(pair.source.x, pair.source.y) for pair in pairs])
    peer = np.array([(pair.peer.x, pair.peer.y) for pair in pairs])
    if minimum > 1 and np.linalg.norm(np.ptp(source, axis=0)) < 3:
        raise DeskError("Spread landmarks across the part of film being compared.", field="pairs")
    weights = np.ones(len(source), dtype=np.float64)
    transform = _fit(mode, source, peer, weights)
    predicted, error = _mapped_error(transform, source, peer)
    if robust and len(source) > minimum + 1:
        for _ in range(5):
            scale = max(.6, 1.4826 * float(np.median(error)))
            weights = np.minimum(1, scale / np.maximum(error, 1e-8))
            transform = _fit(mode, source, peer, weights)
            predicted, error = _mapped_error(transform, source, peer)
    details = []
    for index, (landmark, location, residual, weight) in enumerate(zip(pairs, predicted, error, weights)):
        details.append(
            {
                "index": index,
                **landmark.as_dict(),
                "mapped_peer": {"x": round(float(location[0]), 4), "y": round(float(location[1]), 4)},
                "residual_px": round(float(residual), 4),
                "weight": round(float(weight), 4),
            }
        )
    return {
        "model": mode,
        "transform": transform.as_list(),
        "rms_px": round(float(np.sqrt(np.average(error ** 2, weights=weights))), 4),
        "worst_px": round(float(np.max(error)), 4),
        "landmarks": details,
        "robust": robust,
    }


def _dense_crop(image, box):
    values = image.values[box.slices]
    valid = image.valid[box.slices]
    if valid.ndim == 3:
        valid = valid.all(axis=2)
    gray = np.einsum("ijk,k->ij", values, SCANNER_LUMA, optimize=True)
    return gray, valid


def _score_map(template, search):
    height, width = template.shape
    count = height * width
    mask = np.ones((height, width), dtype=np.float32)
    centered = template - float(template.mean())
    norm = float(np.linalg.norm(centered))
    if norm < .005 * math.sqrt(count):
        raise DeskError("Choose a patch with enough image texture to align.", field="region")
    numerator = signal.fftconvolve(search, centered[::-1, ::-1], "valid")
    totals = signal.fftconvolve(search, mask, "valid")
    squares = signal.fftconvolve(search ** 2, mask, "valid")
    variance = np.maximum(squares - totals ** 2 / count, 0)
    denominator = norm * np.sqrt(variance)
    return np.where(denominator > 1e-6, numerator / np.maximum(denominator, 1e-6), -1)


def _peak_offset(values, index):
    left, center, right = (float(values[index + offset]) for offset in (-1, 0, 1))
    denominator = left - 2 * center + right
    return float(np.clip(.5 * (left - right) / denominator, -.5, .5)) if denominator < -1e-6 else 0.0


def search_translation(source, peer, region, maximum, *, subpixel=True):
    if region.area > 650_000 or region.w < 12 or region.h < 12:
        raise DeskError("Choose a textured patch between 12 × 12 and 650,000 pixels.", field="region")
    width, height = peer.size
    search_box = region.expanded(maximum, width, height)
    if search_box.w < region.w or search_box.h < region.h:
        raise DeskError("This patch cannot fit on the peer scan.", field="region")
    left, left_valid = _dense_crop(source, region)
    right, right_valid = _dense_crop(peer, search_box)
    if left_valid.mean() < .9:
        raise DeskError("The alignment patch has too much unavailable source data.", field="region")
    if not left_valid.all():
        left = np.where(left_valid, left, np.median(left[left_valid]))
    scores = _score_map(left, right)
    valid_count = signal.fftconvolve(
        right_valid.astype(np.float32), np.ones(left.shape, dtype=np.float32), "valid"
    )
    scores = np.where(valid_count >= .9 * region.area, scores, -1)
    best = np.unravel_index(int(np.argmax(scores)), scores.shape)
    row, column = best
    peak = float(scores[best])
    if peak < .35:
        raise DeskError("No reliable repeat-pass match was found in this search window.", field="region")
    alternative = scores.copy()
    alternative[max(0, row - 2):row + 3, max(0, column - 2):column + 3] = -1
    runner = float(alternative.max())
    dx = search_box.x + column - region.x
    dy = search_box.y + row - region.y
    if subpixel and 0 < column < scores.shape[1] - 1 and 0 < row < scores.shape[0] - 1:
        dx += _peak_offset(scores[row], column)
        dy += _peak_offset(scores[:, column], row)
    return {
        "model": "shift",
        "transform": Transform((1, 0, dx, 0, 1, dy)).as_list(),
        "score": round(peak, 5),
        "runner_up": round(runner, 5) if runner > -.999 else None,
        "ambiguous": runner > peak - .05,
        "source_region": region.as_dict(),
        "search_region": search_box.as_dict(),
        "subpixel": subpixel,
    }


def sample_peer(values, valid, transform, x, y):
    px, py = transform.grid(x, y)
    actual = []
    support = []
    for channel in range(3):
        sampled, usable = _bilinear_valid(values[..., channel], valid[..., channel], px, py)
        actual.append(sampled)
        support.append(usable)
    return np.stack(actual, axis=-1), np.stack(support, axis=-1)


def peer_point(source, peer, transform, point, radius):
    x, y = point.pixel()
    bounds = Box(x, y, 1, 1).expanded(radius, *source.size)
    yy, xx = np.mgrid[bounds.y:bounds.bottom, bounds.x:bounds.right]
    a = source.density[bounds.slices]
    valid_a = source.valid[bounds.slices]
    b, valid_b = sample_peer(peer.density, peer.valid, transform, xx, yy)
    joint = valid_a & valid_b
    deltas = []
    rows = {}
    for index, channel in enumerate(PRIMARY_CHANNELS):
        good = joint[..., index]
        left = a[..., index][good]
        right = b[..., index][good]
        delta = float(np.median(right - left)) if len(left) else np.nan
        deltas.append(delta)
        rows[channel] = {
            "source": round(float(np.median(left)), 5) if len(left) else None,
            "peer": round(float(np.median(right)), 5) if len(right) else None,
            "median_delta": round(delta, 5) if np.isfinite(delta) else None,
            "joint": int(good.sum()),
            "fraction": round(float(good.mean()), 5),
        }
    mapped = transform.project(point)
    return {
        "source_point": point.as_dict(),
        "peer_point": mapped.as_dict(),
        "point_on_peer": 0 <= mapped.x < peer.size[0] and 0 <= mapped.y < peer.size[1],
        "source_region": bounds.as_dict(),
        "median_delta": round_channels(deltas),
        "channels": rows,
    }


def pair_residual(source, peer, transform, region, threshold, *, maximum_samples=250_000):
    step = max(1, int(math.ceil(math.sqrt(region.area / maximum_samples))))
    yy, xx = np.mgrid[region.y:region.bottom:step, region.x:region.right:step]
    values = source.density[yy, xx]
    valid = source.valid[yy, xx]
    other, other_valid = sample_peer(peer.density, peer.valid, transform, xx, yy)
    joint = valid & other_valid
    delta = other - values
    statistics = {}
    for index, channel in enumerate(PRIMARY_CHANNELS):
        eligible = joint[..., index]
        chosen = delta[..., index][eligible]
        absolute = np.abs(chosen)
        statistics[channel] = {
            "count": int(len(chosen)),
            "joint_fraction": round(float(eligible.mean()), 5),
            "p95_absolute": round(float(np.quantile(absolute, .95)), 5) if len(chosen) else None,
            "over_threshold": int(np.count_nonzero(absolute > threshold)),
            "median": round(float(np.median(chosen)), 5) if len(chosen) else None,
        }
    return {
        "region": region.as_dict(),
        "threshold": threshold,
        "sample_step": step,
        "samples": int(yy.size),
        "channels": statistics,
        "full_rgb_fraction": round(float(joint.all(axis=2).mean()), 5),
    }


def pair_render(source, peer, transform, region, mode, *, maximum_side=1400, opacity=.5, checker_size=48):
    one_of(mode, PAIR_PREVIEWS, "view")
    step = max(1, int(math.ceil(max(region.w, region.h) / maximum_side)))
    yy, xx = np.mgrid[region.y:region.bottom:step, region.x:region.right:step]
    a = np.clip(source.values[yy, xx], 0, 1)
    b, valid_b = sample_peer(peer.values, peer.valid, transform, xx, yy)
    b = np.clip(b, 0, 1)
    valid_a = source.valid[yy, xx].all(axis=2)
    valid_b = valid_b.all(axis=2)
    if mode == "difference":
        delta = np.einsum("ijk,k->ij", a - b, SCANNER_LUMA)
        amount = np.clip(np.abs(delta) * 5, 0, 1)
        hot = np.stack((amount * .95, amount * .33, amount * .10), axis=2)
        cold = np.stack((amount * .10, amount * .62, amount * .94), axis=2)
        image = np.where((delta >= 0)[..., None], hot, cold) + .08
        valid = valid_a & valid_b
    elif mode == "checker":
        selector = (xx // checker_size + yy // checker_size) % 2 == 0
        image = np.where(selector[..., None], a, b)
        valid = np.where(selector, valid_a, valid_b)
    elif mode == "peer":
        image = b
        valid = valid_b
    else:
        image = (1 - opacity) * a + opacity * b
        valid = valid_a & valid_b
    background = np.where(((xx // 12) + (yy // 12)) % 2 == 0, .15, .24)
    return np.where(valid[..., None], image, background[..., None])
