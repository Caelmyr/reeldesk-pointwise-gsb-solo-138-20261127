"""Validated access to scan assets, roll metadata, marks, and rendered views."""

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath
from threading import RLock

from PIL import Image

from .demo import install_demo
from .imaging import (
    Calibration, DecodeCache, Tone, apply_calibration, base_estimate, probe_original, render_view,
)
from .strip import read_mark
from .values import (
    Box, DeskError, IMAGE_ROLES, SCAN_MEDIA, VERDICTS, VIEW_MODES, checked_key, clean_text, integer,
    json_hash, mapping, number, one_of, read_revision, tags,
)


THUMBNAIL_BUDGET = 45 * 1024 * 1024


@dataclass(frozen=True)
class ScanImage:
    metadata: dict
    image: object

    @property
    def id(self):
        return self.metadata["id"]

    @property
    def size(self):
        return self.image.size

    @property
    def bounds(self):
        return Box(0, 0, *self.size)


def roll_fields(value, current=None):
    value = mapping(value)
    current = current or {}
    result = {}
    for label, maximum, required in (
        ("title", 100, True),
        ("code", 35, True),
        ("stock", 100, True),
        ("location", 150, False),
        ("notes", 2000, False),
    ):
        result[label] = clean_text(value.get(label, current.get(label)), label, maximum=maximum, required=required)
    result["tags"] = tags(value.get("tags", current.get("tags", [])))
    return result


def scan_fields(value, current=None):
    value = mapping(value)
    current = current or {}
    result = {}
    for label, maximum, required in (
        ("name", 100, True),
        ("scanner", 100, False),
        ("exposure", 100, False),
        ("notes", 2000, False),
    ):
        result[label] = clean_text(value.get(label, current.get(label)), label, maximum=maximum, required=required)
    result["medium"] = one_of(value.get("medium", current.get("medium", "negative")), SCAN_MEDIA, "medium")
    dpi = value.get("dpi", current.get("dpi"))
    result["dpi"] = number(dpi, "dpi", lower=30, upper=20_000) if dpi not in ("", None) else None
    return result


def _original_file_name(value):
    value = value.replace("\\", "/")
    return clean_text(PurePath(value).name, "filename", maximum=250, required=True)


class EncodedCache:
    """Least-recently-used previews keyed by immutable inputs and display settings."""

    def __init__(self, size_limit):
        self.entries = OrderedDict()
        self.size_limit = size_limit
        self.current_size = 0
        self.lock = RLock()

    def get(self, identifier):
        with self.lock:
            if identifier not in self.entries:
                return None
            self.entries.move_to_end(identifier)
            return self.entries[identifier]

    def put(self, identifier, data):
        size = len(data)
        if size > self.size_limit:
            return data
        with self.lock:
            if identifier in self.entries:
                self.entries.move_to_end(identifier)
                return self.entries[identifier]
            while self.current_size > self.size_limit - size:
                _, removed = self.entries.popitem(last=False)
                self.current_size -= len(removed)
            self.entries[identifier] = data
            self.current_size += size
        return data


class Workspace:
    def __init__(self, catalog):
        self.catalog = catalog
        self.decoded = DecodeCache()
        self.previews = EncodedCache(THUMBNAIL_BUDGET)

    def start_demo(self):
        return install_demo(self.catalog)

    def create_roll(self, data):
        return self.catalog.create_roll(roll_fields(data))

    def update_roll(self, roll_id, data):
        current = self.catalog.get_roll(roll_id)
        revision = read_revision(mapping(data).get("revision"))
        fields = roll_fields(data, current)
        fields = {key: value for key, value in fields.items() if value != current[key]}
        return self.catalog.update_roll(roll_id, fields, revision)

    def import_scan(self, roll_id, metadata, blob, filename):
        filename = _original_file_name(filename)
        metadata = mapping(metadata, "metadata")
        role = one_of(metadata.get("role", "scan"), IMAGE_ROLES, "role")
        probe = probe_original(blob)
        default_name = PurePath(filename).stem
        if role in {"flat", "dark"}:
            default_medium = "control"
        else:
            default_medium = "negative"
        current = {"name": default_name[:100], "dpi": probe["embedded_dpi"], "medium": default_medium}
        fields = scan_fields(metadata, current)
        if role in {"flat", "dark"}:
            fields["medium"] = "control"
        return self.catalog.add_scan(
            roll_id,
            {"role": role, "original_name": filename, **fields},
            blob,
            probe,
        )

    def update_scan(self, scan_id, data):
        current = self.catalog.get_scan(scan_id)
        revision = read_revision(mapping(data).get("revision"))
        fields = scan_fields(data, current)
        if current["role"] in {"flat", "dark"}:
            fields["medium"] = "control"
        fields["verdict"] = one_of(data.get("verdict", current["verdict"]), VERDICTS, "verdict")
        fields = {key: value for key, value in fields.items() if value != current[key]}
        return self.catalog.update_scan(scan_id, fields, revision)

    def add_mark(self, scan_id, value):
        scan = self.catalog.get_scan(scan_id)
        return self.catalog.add_mark(scan_id, read_mark(value, scan))

    def update_mark(self, mark_id, value):
        previous = self.catalog.get_mark(mark_id)
        scan = self.catalog.get_scan(previous["scan_id"])
        revision = read_revision(mapping(value).get("revision"))
        return self.catalog.update_mark(mark_id, read_mark(value, scan, previous), revision)

    def get_image(self, scan_id):
        metadata, raw = self.catalog.original(scan_id)
        return ScanImage(metadata, self.decoded.read(metadata["checksum"], raw))

    def get_peer(self, source, peer_id):
        peer = self.get_image(checked_key(peer_id, "peer_id"))
        if peer.id == source.id:
            raise DeskError("Choose a different repeat scan.", field="peer_id")
        if source.metadata["roll_id"] != peer.metadata["roll_id"] or peer.metadata["role"] != "scan":
            raise DeskError("Choose a scan from this roll for repeat-pass review.", field="peer_id")
        return peer

    def controls(self, source, calibration):
        selected = []
        for role, scan_id in (("flat", calibration.flat_id), ("dark", calibration.dark_id)):
            if scan_id is None:
                selected.append(None)
                continue
            control = self.get_image(scan_id)
            actual = control.metadata
            if actual["roll_id"] != source.metadata["roll_id"] or actual["role"] != role:
                raise DeskError(f"Choose a {role} control on this roll.", field=f"calibration.{role}_id")
            if control.size != source.size:
                raise DeskError(f"The {role} control must match this scan's dimensions.", field=f"calibration.{role}_id")
            selected.append(control)
        return tuple(selected)

    def calibrated(self, source, calibration):
        flat, dark = self.controls(source, calibration)
        return apply_calibration(
            source.image,
            calibration,
            flat=flat.image if flat else None,
            dark=dark.image if dark else None,
        )

    def settings_for_preview(self, source, value, *, max_side=1600):
        value = mapping(value)
        calibration = Calibration.read(value.get("calibration"))
        tone = Tone.read(value.get("tone"))
        view = one_of(value.get("view", "raw"), VIEW_MODES, "view")
        maximum = integer(value.get("max_side", max_side), "max_side", lower=80, upper=3000)
        region = Box.read(value["region"], bounds=source.size) if "region" in value else None
        return calibration, tone, view, maximum, region

    def preview(self, source, calibration, tone, view, maximum, region=None):
        flat, dark = self.controls(source, calibration) if view != "raw" else (None, None)
        identity = json_hash(
            {
                "source": source.metadata["checksum"], "medium": source.metadata["medium"],
                "flat": flat.metadata["checksum"] if flat else None,
                "dark": dark.metadata["checksum"] if dark else None,
                "calibration": calibration.as_dict(), "tone": tone.as_dict(),
                "view": view, "size": maximum, "region": region.as_dict() if region else None,
            }
        )
        encoded = self.previews.get(identity)
        if encoded is not None:
            return encoded
        corrected = apply_calibration(
            source.image, calibration, flat=flat.image if flat else None, dark=dark.image if dark else None
        ) if view != "raw" else None
        result = render_view(
            source.image, corrected, view, tone, source.metadata["medium"], region=region, max_side=maximum
        )
        return self.previews.put(identity, result)

    def preview_image(self, source, calibration, tone, view, maximum, region=None):
        data = self.preview(source, calibration, tone, view, maximum, region)
        with Image.open(BytesIO(data)) as image:
            return image.copy()

    def estimate_base(self, source, value):
        value = mapping(value)
        region = Box.read(value.get("region"), bounds=source.size)
        calibration = Calibration.read(value.get("calibration"))
        unity = Calibration(calibration.flat_id, calibration.dark_id, (1, 1, 1), calibration.reject_clipped)
        corrected = self.calibrated(source, unity)
        return base_estimate(corrected, region)
