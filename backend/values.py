"""Validation, serialization, and canonical scanner-space geometry."""

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import math
import re
from uuid import uuid4

import numpy as np
from PIL import Image


PIXEL_LIMIT = 12_000_000
UPLOAD_LIMIT = 36 * 1024 * 1024
LABEL_LIMIT = 100
KEY_PATTERN = re.compile(r"^[a-f0-9]{24}$")
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
IMAGE_ROLES = frozenset({"scan", "flat", "dark", "reference"})
MARK_KINDS = frozenset({"frame", "cut", "dust", "scratch", "perforation", "reference", "keep"})
SCAN_MEDIA = frozenset({"negative", "transparency", "paper", "control"})
VERDICTS = frozenset({"hold", "accepted", "rejected"})
VIEW_MODES = frozenset({"raw", "positive", "density", "clip"})
PRIMARY_CHANNELS = ("R", "G", "B")


class DeskError(Exception):
    """An actionable input or workflow error."""

    def __init__(self, message, *, field=None, code="invalid", status=422, details=None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.code = code
        self.status = status
        self.details = details

    def payload(self):
        result = {"error": self.code, "message": self.message}
        if self.field is not None:
            result["field"] = self.field
        if self.details is not None:
            result["details"] = self.details
        return result


def missing(label):
    return DeskError(f"{label} was not found.", code="not_found", status=404)


def conflict(message):
    return DeskError(message, code="conflict", status=409)


def key():
    return uuid4().hex[:24]


def checked_key(value, field="id"):
    if not isinstance(value, str) or not KEY_PATTERN.fullmatch(value):
        raise DeskError("Use a valid ReelDesk record ID.", field=field)
    return value


def optional_key(value, field):
    return checked_key(value, field) if value not in (None, "") else None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def clean_text(value, field, *, maximum=LABEL_LIMIT, required=False):
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise DeskError("Use text for this field.", field=field)
    value = value.strip()
    if required and not value:
        raise DeskError("Enter a value for this field.", field=field)
    if len(value) > maximum:
        raise DeskError(f"Use at most {maximum} characters.", field=field)
    if any(ord(character) < 32 and character not in "\r\n\t" for character in value):
        raise DeskError("Control characters are not allowed here.", field=field)
    return value


def text_list(value, field, *, maximum=20, item_limit=LABEL_LIMIT):
    options = sequence(value, field, maximum=maximum)
    return [
        clean_text(item, f"{field}[{index}]", maximum=item_limit, required=True)
        for index, item in enumerate(options)
    ]


def tags(value):
    values = text_list(value, "tags", maximum=16, item_limit=40)
    unique = []
    for value in values:
        value = value.lower()
        if not TAG_PATTERN.fullmatch(value):
            raise DeskError("Use letters, digits and hyphens for tag names.", field="tags")
        if value not in unique:
            unique.append(value)
    return unique


def one_of(value, options, field):
    if not isinstance(value, str) or value not in options:
        readable = ", ".join(sorted(options))
        raise DeskError(f"Choose one of {readable}.", field=field)
    return value


def number(value, field, *, lower=None, upper=None):
    if isinstance(value, bool):
        raise DeskError("Enter a number.", field=field)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise DeskError("Enter a number.", field=field) from None
    if not math.isfinite(result):
        raise DeskError("Enter a finite number.", field=field)
    if lower is not None and result < lower:
        raise DeskError(f"Enter a value of at least {lower:g}.", field=field)
    if upper is not None and result > upper:
        raise DeskError(f"Enter a value no greater than {upper:g}.", field=field)
    return result


def integer(value, field, *, lower=None, upper=None):
    result = number(value, field, lower=lower, upper=upper)
    if result != int(result):
        raise DeskError("Enter a whole number.", field=field)
    return int(result)


def boolean(value, field):
    if not isinstance(value, bool):
        raise DeskError("Use true or false for this setting.", field=field)
    return value


def mapping(value, field="body"):
    if not isinstance(value, dict):
        raise DeskError("Supply a JSON object.", field=field)
    return value


def sequence(value, field, *, minimum=0, maximum=1000):
    if not isinstance(value, list):
        raise DeskError("Supply a list.", field=field)
    if len(value) < minimum:
        raise DeskError(f"Provide at least {minimum} entries.", field=field)
    if len(value) > maximum:
        raise DeskError(f"Provide at most {maximum} entries.", field=field)
    return value


def as_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def parse_json(value):
    return json.loads(value)


def payload_hash(data):
    return hashlib.sha256(data).hexdigest()


def json_hash(value):
    return payload_hash(as_json(value).encode("utf-8"))


def channel_vector(value, field, *, lower=0, upper=1):
    values = sequence(value, field, minimum=3, maximum=3)
    return tuple(
        number(component, f"{field}.{channel}", lower=lower, upper=upper)
        for channel, component in zip(PRIMARY_CHANNELS, values)
    )


def round_channels(values, digits=5):
    return [round(float(value), digits) if np.isfinite(value) else None for value in values]


def rgb_image(array):
    values = np.clip(np.rint(array * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def image_bytes(image, format="PNG", **options):
    buffer = BytesIO()
    image.save(buffer, format=format, **options)
    return buffer.getvalue()


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def read(cls, value, field):
        value = mapping(value, field)
        return cls(number(value.get("x"), f"{field}.x"), number(value.get("y"), f"{field}.y"))

    def as_dict(self):
        return {"x": self.x, "y": self.y}

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def pixel(self):
        return math.floor(self.x), math.floor(self.y)

    def bounded(self, width, height, field="point"):
        if not 0 <= self.x < width or not 0 <= self.y < height:
            raise DeskError("The point must lie on the scan.", field=field)
        return self


@dataclass(frozen=True)
class Box:
    """A half-open rectangle in source pixels."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def read(cls, value, field="region", *, bounds=None):
        value = mapping(value, field)
        box = cls(
            x=integer(value.get("x"), f"{field}.x", lower=0),
            y=integer(value.get("y"), f"{field}.y", lower=0),
            w=integer(value.get("w"), f"{field}.w", lower=1),
            h=integer(value.get("h"), f"{field}.h", lower=1),
        )
        if bounds:
            box.bounded(*bounds, field=field)
        return box

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def area(self):
        return self.w * self.h

    @property
    def slices(self):
        return slice(self.y, self.bottom), slice(self.x, self.right)

    @property
    def pillow(self):
        return self.x, self.y, self.right, self.bottom

    @property
    def center(self):
        return Point(self.x + self.w / 2, self.y + self.h / 2)

    def as_dict(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def bounded(self, width, height, *, field="region"):
        if self.x < 0 or self.y < 0 or self.right > width or self.bottom > height:
            raise DeskError("The region extends outside the scan.", field=field)
        return self

    def contains(self, point):
        return self.x <= point.x < self.right and self.y <= point.y < self.bottom

    def intersect(self, other):
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if x >= right or y >= bottom:
            return None
        return Box(x, y, right - x, bottom - y)

    def expanded(self, margin, width, height):
        x = max(0, self.x - margin)
        y = max(0, self.y - margin)
        right = min(width, self.right + margin)
        bottom = min(height, self.bottom + margin)
        return Box(x, y, right - x, bottom - y)

    def translate(self, dx, dy):
        return Box(self.x + dx, self.y + dy, self.w, self.h)



@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point

    @classmethod
    def read(cls, value, field="line", *, bounds=None):
        value = mapping(value, field)
        start = Point.read(value.get("start"), f"{field}.start")
        end = Point.read(value.get("end"), f"{field}.end")
        if bounds:
            start.bounded(*bounds, field=f"{field}.start")
            end.bounded(*bounds, field=f"{field}.end")
        if start.distance(end) < 1:
            raise DeskError("Place the endpoints at least one pixel apart.", field=field)
        return cls(start, end)

    @property
    def length(self):
        return self.start.distance(self.end)

    def as_dict(self):
        return {"start": self.start.as_dict(), "end": self.end.as_dict()}

    def stations(self, count):
        return [
            Point(self.start.x + (self.end.x - self.start.x) * fraction,
                  self.start.y + (self.end.y - self.start.y) * fraction)
            for fraction in np.linspace(0, 1, count)
        ]

    def projection(self, point):
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return ((point.x - self.start.x) * dx + (point.y - self.start.y) * dy) / (dx * dx + dy * dy)

    def point_at(self, fraction):
        return Point(
            self.start.x + fraction * (self.end.x - self.start.x),
            self.start.y + fraction * (self.end.y - self.start.y),
        )

    def distance_to(self, point):
        return self.point_at(max(0, min(1, self.projection(point)))).distance(point)

    def crosses_box(self, box):
        low = 0.0
        high = 1.0
        for start, end, minimum, maximum in (
            (self.start.x, self.end.x, box.x, box.right),
            (self.start.y, self.end.y, box.y, box.bottom),
        ):
            delta = end - start
            if abs(delta) < 1e-12:
                if start < minimum or start > maximum:
                    return False
                continue
            first, last = sorted(((minimum - start) / delta, (maximum - start) / delta))
            low = max(low, first)
            high = min(high, last)
            if low > high:
                return False
        return True


def page_bounds(arguments, *, limit=40, maximum=100):
    size = integer(arguments.get("limit", limit), "limit", lower=1, upper=maximum)
    offset = integer(arguments.get("offset", 0), "offset", lower=0, upper=1_000_000)
    return size, offset


def read_revision(value):
    return integer(value, "revision", lower=1, upper=2_000_000_000)


def radii(value, field, *, maximum=24):
    options = sequence(value, field, minimum=1, maximum=8)
    result = sorted({integer(item, field, lower=0, upper=maximum) for item in options})
    return result

