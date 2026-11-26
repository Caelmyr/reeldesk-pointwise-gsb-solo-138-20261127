"""Offline re-check packages for saved region-density readings.

A batch package is one ZIP containing the saved region-density readings of a
roll, the original scans they were measured on, the flat and dark controls used
for each reading, and the measurement boxes. Every selected reading stays in
the manifest list. Files are deduplicated by content hash, and the finished ZIP
never exceeds an operator-supplied byte ceiling.

Verification never treats matching checksums as a re-measurement. It decodes the
packed originals, re-applies the recorded calibration and measures the recorded
box again. A reading is reported as re-measured only when every channel median
density is within the operator's tolerance and every valid-pixel count matches.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import PurePosixPath
from threading import RLock
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile
import zlib

import numpy as np

from .evidence import json_document, safe_original_name
from .imaging import Calibration, DecodeCache, apply_calibration, density_stats
from .values import (
    Box, DeskError, PRIMARY_CHANNELS, checked_key, integer, key, mapping, number, now,
    payload_hash, sequence,
)


BATCH_FORMAT = "reeldesk-batch-review-v1"
MAX_READINGS = 200
MAX_PACKET_FILES = 3 * MAX_READINGS + 16
MIN_CAPACITY = 8 * 1024
MAX_CAPACITY = 90 * 1024 * 1024
ORIGINAL_LOAD_LIMIT = 300 * 1024 * 1024
MANIFEST_MARGIN = 1024
MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "TIFF": "image/tiff", "WEBP": "image/webp"}
ROLE_TITLES = {"primary": "original scan", "flat": "flat control", "dark": "dark control"}
ZIP_END = 22
ZIP_ENTRY = 76


def _deflated_size(content):
    packed = zlib.compressobj(5, zlib.DEFLATED, -15)
    return len(packed.compress(content) + packed.flush())


def member_cost(name, content, *, compressed=None):
    return ZIP_ENTRY + 2 * len(name.encode("utf-8")) + (_deflated_size(content) if compressed is None else compressed)


def _batch_error(message):
    return DeskError(message, field="packet", code="invalid_packet")


@dataclass(frozen=True)
class PackedFile:
    checksum: str
    name: str
    content_type: str
    content: bytes | None
    size: int
    compressed: int | None

    @property
    def loaded(self):
        return self.content is not None

    @property
    def cost(self):
        """Price an unloaded file at its stored bytes plus deflate's worst-case growth."""
        packed = self.compressed if self.loaded else self.size + self.size // 1000 + 32
        return ZIP_ENTRY + 2 * len(self.name.encode("utf-8")) + packed


class OriginalStore:
    """Size each distinct original from catalog metadata, loading only chosen files."""

    def __init__(self, catalog, limit=ORIGINAL_LOAD_LIMIT):
        self.catalog = catalog
        self.limit = limit
        self.loaded = 0
        self.by_checksum = {}
        self.names = set()

    def _name(self, scan):
        checksum = scan["checksum"]
        base = safe_original_name(scan["original_name"], scan["source_format"])
        name = f"originals/{checksum[:16]}-{base}"
        if name in self.names:
            name = f"originals/{checksum}-{base}"
        self.names.add(name)
        return name

    def reference(self, scan):
        """Cheap sizing from the stored byte count; rerun compression only once loaded."""
        checksum = scan["checksum"]
        existing = self.by_checksum.get(checksum)
        if existing is not None:
            return existing
        reference = PackedFile(
            checksum, self._name(scan),
            MIME.get(scan["source_format"], "application/octet-stream"), None,
            scan["byte_size"], None,
        )
        self.by_checksum[checksum] = reference
        return reference

    def load(self, checksums):
        """Decode-free load of only the packed files, refreshing their exact compressed size."""
        for checksum in sorted(checksums):
            reference = self.by_checksum[checksum]
            if reference.content is not None:
                continue
            content = self._original(checksum)
            self.loaded += len(content)
            if self.loaded > self.limit:
                raise DeskError(
                    "These readings reference too many large originals to pack at once. "
                    "Plan a smaller selection.",
                    field="record_ids",
                )
            packed = zlib.compressobj(5, zlib.DEFLATED, -15)
            compressed = len(packed.compress(content) + packed.flush())
            self.by_checksum[checksum] = PackedFile(
                checksum, reference.name, reference.content_type, content, len(content), compressed
            )
        return [self.by_checksum[checksum] for checksum in sorted(checksums)]

    def _original(self, checksum):
        with self.catalog.connection() as connection:
            row = connection.execute(
                "SELECT content FROM originals WHERE checksum=?", (checksum,)
            ).fetchone()
        if row is None:
            raise DeskError("An original for this reading is missing from the catalog.", field="record_ids")
        return row["content"]


@dataclass
class Reading:
    record: dict
    scan: dict
    priority: bool
    calibration: dict | None = None
    region: Box | None = None
    expected: dict | None = None
    controls: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    @property
    def identifier(self):
        return self.record["id"]

    @property
    def ready(self):
        return not self.missing

    @property
    def signature(self):
        return tuple(sorted(reference.checksum for reference in self.files.values()))


def _calibration_payload(calibration):
    return {
        "flat_id": calibration.flat_id,
        "dark_id": calibration.dark_id,
        "base": list(calibration.base),
        "reject_clipped": bool(calibration.reject_clipped),
    }


def _expected_channels(result):
    channels = result.get("channels") if isinstance(result, dict) else None
    if not isinstance(channels, dict):
        return None, ["the saved reading has no channel results"]
    expected = {}
    problems = []
    for channel in PRIMARY_CHANNELS:
        value = channels.get(channel)
        if not isinstance(value, dict):
            problems.append(f"saved {channel} result is missing")
            continue
        median = value.get("median")
        valid = value.get("valid")
        if not isinstance(median, (int, float)) or isinstance(median, bool):
            problems.append(f"saved {channel} median density is missing")
            continue
        if type(valid) is not int or valid < 0:
            problems.append(f"saved {channel} valid-pixel count is missing")
            continue
        expected[channel] = {"median": round(float(median), 5), "valid": valid}
    if problems:
        return None, problems
    return {"channels": expected}, []


def _analyse(catalog, record, priority):
    reading = Reading(record, catalog.get_scan(record["scan_id"]), priority)
    settings = record.get("settings") if isinstance(record.get("settings"), dict) else {}
    try:
        calibration = Calibration.read(settings.get("calibration"), "calibration")
    except DeskError as error:
        reading.missing.append(f"the saved calibration is invalid: {error.message}")
        calibration = None
    if calibration is not None:
        reading.calibration = _calibration_payload(calibration)
        for role, field_name in (("flat", "flat_id"), ("dark", "dark_id")):
            scan_id = getattr(calibration, field_name)
            if not scan_id:
                reading.missing.append(f"no {role} control was used for this reading")
                continue
            try:
                control = catalog.get_scan(scan_id)
            except DeskError:
                reading.missing.append(f"the {role} control is missing from the catalog")
                continue
            if control["roll_id"] != reading.scan["roll_id"] or control["role"] != role:
                reading.missing.append(f"the packed {role} is not a {role} control on this roll")
            elif (control["width"], control["height"]) != (reading.scan["width"], reading.scan["height"]):
                reading.missing.append(f"the {role} control dimensions differ from the scan")
            else:
                reading.controls[role] = control
    try:
        reading.region = Box.read(settings.get("region"), "region", bounds=(reading.scan["width"], reading.scan["height"]))
    except DeskError:
        reading.missing.append("the measurement box is missing or outside the scan")
    expected, problems = _expected_channels(record.get("result"))
    reading.expected = expected
    reading.missing.extend(problems)
    return reading


def _attach_files(reading, store):
    if not reading.ready:
        return
    reading.files = {"primary": store.reference(reading.scan)}
    for role in ("flat", "dark"):
        reading.files[role] = store.reference(reading.controls[role])


def _records(catalog, roll_id, value):
    capacity = integer(value.get("max_bytes"), "max_bytes", lower=MIN_CAPACITY, upper=MAX_CAPACITY)
    tolerance = number(value.get("tolerance", .02), "tolerance", lower=1e-5, upper=1.0)
    selection = value.get("record_ids")
    if selection is None:
        records = catalog.list_records(roll_id, kind="density", limit=MAX_READINGS + 1)
        if len(records) > MAX_READINGS:
            raise DeskError("Choose at most 200 saved density readings.", field="record_ids")
    else:
        identifiers = []
        for item in sequence(selection, "record_ids", minimum=1, maximum=MAX_READINGS):
            identifier = checked_key(item, "record_ids")
            if identifier not in identifiers:
                identifiers.append(identifier)
        records = []
        for identifier in identifiers:
            record = catalog.get_record(identifier)
            if record["roll_id"] != roll_id:
                raise DeskError("Choose saved readings from this roll.", field="record_ids")
            if record["kind"] != "density":
                raise DeskError("Offline re-check packages hold saved region-density readings.", field="record_ids")
            records.append(record)
    priorities = {
        checked_key(item, "priority_ids")
        for item in sequence(value.get("priority_ids", []), "priority_ids", maximum=MAX_READINGS)
    }
    available = {record["id"] for record in records}
    if priorities - available:
        raise DeskError("Priority readings must also be selected for the package.", field="priority_ids")
    return records, priorities, capacity, tolerance


def _groups(readings):
    grouped = {}
    for reading in readings:
        if not reading.ready:
            continue
        key_signature = reading.signature
        group = grouped.get(key_signature)
        if group is None:
            group = {"files": key_signature, "readings": [], "priority": 0}
            grouped[key_signature] = group
        group["readings"].append(reading)
        group["priority"] += int(reading.priority)
    return sorted(grouped.values(), key=lambda item: (-len(item["readings"]), -item["priority"], item["files"]))


def _incremental(group, used, costs):
    return sum(costs[checksum] for checksum in group["files"] if checksum not in used)


def _greedy(groups, costs, budget):
    """Fallback when the integer solver is unavailable: protected groups first."""
    chosen = []
    used = set()
    spent = 0
    remaining = list(groups)
    for protected_only in (True, False):
        while remaining:
            if protected_only and not any(group["priority"] for group in remaining):
                break
            best = None
            best_score = None
            best_extra = 0
            for group in remaining:
                if protected_only and not group["priority"]:
                    continue
                extra = _incremental(group, used, costs)
                if spent + extra > budget:
                    continue
                score = (len(group["readings"]) / max(extra, 1), group["priority"] / max(extra, 1))
                if best_score is None or score > best_score:
                    best, best_score, best_extra = group, score, extra
            if best is None:
                break
            chosen.append(best)
            used.update(best["files"])
            spent += best_extra
            remaining.remove(best)
    identifiers = set()
    for group in chosen:
        identifiers.update(reading.identifier for reading in group["readings"])
    return identifiers


def _valid_selection(groups, costs, budget, identifiers):
    used = set()
    for group in groups:
        if any(reading.identifier in identifiers for reading in group["readings"]):
            used.update(group["files"])
    return sum(costs[checksum] for checksum in used) <= budget


def _solve(groups, costs, budget):
    """Maximise evidence-complete readings, then protected readings, then spend."""
    total_files = {checksum for group in groups for checksum in group["files"]}
    if sum(costs[checksum] for checksum in total_files) <= budget:
        return {reading.identifier for group in groups for reading in group["readings"]}
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError:
        return _greedy(groups, costs, budget)
    files = sorted(total_files)
    file_index = {checksum: position for position, checksum in enumerate(files)}
    count = len(groups)
    width = count + len(files)
    rows = []
    lower = []
    upper = []
    for position, group in enumerate(groups):
        for checksum in group["files"]:
            row = np.zeros(width)
            row[position] = 1
            row[count + file_index[checksum]] = -1
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0)
    budget_row = np.zeros(width)
    for checksum, position in file_index.items():
        budget_row[count + position] = costs[checksum]
    rows.append(budget_row)
    lower.append(-np.inf)
    upper.append(budget)
    counts = np.array([len(group["readings"]) for group in groups] + [0] * len(files), dtype=float)
    priority = np.array([group["priority"] for group in groups] + [0] * len(files), dtype=float)
    file_cost = np.array([0] * count + [costs[checksum] for checksum in files], dtype=float)
    # Prefer fewer stored files once counts and priorities tie, so a smaller
    # package does not lock out later choices when the ceiling is loose.
    small = np.zeros(width)
    for checksum, position in file_index.items():
        small[count + position] = 1
    fixed_rows = []
    fixed_lower = []
    fixed_upper = []

    def run(objective):
        matrix = csr_matrix(np.vstack(rows + fixed_rows))
        result = milp(
            objective,
            constraints=LinearConstraint(matrix, np.array(lower + fixed_lower), np.array(upper + fixed_upper)),
            integrality=np.ones(width),
            bounds=Bounds(np.zeros(width), np.ones(width)),
            options={"time_limit": 15},
        )
        if result.x is None:
            raise ValueError("no solution")
        return result.x

    try:
        x = run(-counts)
        best_count = int(round(float(np.dot(counts, x))))
        row = np.zeros(width)
        row[:count] = counts[:count]
        fixed_rows.append(row)
        fixed_lower.append(best_count - 1e-6)
        fixed_upper.append(np.inf)
        x = run(-priority)
        best_priority = int(round(float(np.dot(priority, x))))
        row = np.zeros(width)
        row[:count] = priority[:count]
        fixed_rows.append(row)
        fixed_lower.append(best_priority - 1e-6)
        fixed_upper.append(np.inf)
        objective = small * 1e-6 - counts * 1e-9 - file_cost
        x = run(objective)
        chosen = [group for position, group in enumerate(groups) if x[position] > .5]
        identifiers = {reading.identifier for group in chosen for reading in group["readings"]}
        if _valid_selection(groups, costs, budget, identifiers):
            return identifiers
    except (ValueError, RuntimeError):
        pass
    return _greedy(groups, costs, budget)


def _reading_entry(reading, included, file_names):
    if included:
        status = "included"
    elif reading.missing:
        status = "incomplete"
    else:
        status = "not-included"
    entry = {
        "record_id": reading.identifier,
        "title": reading.record["title"],
        "created": reading.record["created"],
        "scan_id": reading.scan["id"],
        "priority": reading.priority,
        "status": status,
        "region": reading.region.as_dict() if reading.region else None,
        "calibration": reading.calibration,
        "source": {
            "name": reading.scan["name"],
            "width": reading.scan["width"],
            "height": reading.scan["height"],
            "orientation": reading.scan["orientation"],
            "dpi": reading.scan["dpi"],
            "sha256": reading.scan["checksum"],
        },
        "controls": {
            role: (
                {"name": reading.controls[role]["name"], "sha256": reading.controls[role]["checksum"]}
                if role in reading.controls else None
            )
            for role in ("flat", "dark")
        },
        "expected": reading.expected,
    }
    if included:
        entry["evidence"] = {role: file_names[reference.checksum] for role, reference in reading.files.items()}
    elif reading.missing:
        entry["missing"] = list(reading.missing)
    else:
        entry["reason"] = "not packed within the byte ceiling"
    return entry


def _manifest(readings, included, files, capacity, tolerance):
    file_names = {reference.checksum: reference.name for reference in files}
    entries = [_reading_entry(reading, reading.identifier in included, file_names) for reading in readings]
    return {
        "format": BATCH_FORMAT,
        "created": now(),
        "capacity_bytes": capacity,
        "tolerance": tolerance,
        "coordinates": "Region boxes use source pixels after applying image orientation; rectangles are half-open.",
        "recheck": (
            "Decode each packed original scan with its packed flat and dark controls, apply the recorded "
            "calibration and measure the recorded box again. Matching file checksums are not a re-measurement."
        ),
        "summary": {
            "requested": len(readings),
            "retestable_available": sum(1 for reading in readings if reading.ready),
            "included": sum(1 for reading in readings if reading.identifier in included),
            "incomplete": sum(1 for reading in readings if reading.missing),
            "priority_requested": sum(1 for reading in readings if reading.priority),
            "priority_included": sum(1 for reading in readings if reading.priority and reading.identifier in included),
        },
        "files": [
            {"name": reference.name, "sha256": reference.checksum, "bytes": reference.size,
             "content_type": reference.content_type}
            for reference in sorted(files, key=lambda item: item.name)
        ],
        "readings": entries,
    }


def _pack(manifest_bytes, files):
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=5) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        for reference in sorted(files, key=lambda item: item.name):
            archive.writestr(reference.name, reference.content)
    return buffer.getvalue()


def _selection_files(readings, included):
    checksums = set()
    for reading in readings:
        if reading.identifier in included:
            checksums.update(reference.checksum for reference in reading.files.values())
    return checksums


def _render(readings, included, store, capacity, tolerance):
    files = store.load(_selection_files(readings, included))
    manifest = _manifest(readings, included, files, capacity, tolerance)
    content = json_document(manifest)
    return manifest, content, files, _pack(content, files)


def _build(catalog, roll_id, value):
    catalog.get_roll(roll_id)
    records, priorities, capacity, tolerance = _records(catalog, roll_id, value)
    readings = [_analyse(catalog, record, record["id"] in priorities) for record in records]
    store = OriginalStore(catalog)
    for reading in readings:
        _attach_files(reading, store)
    groups = _groups(readings)
    costs = {checksum: reference.cost for checksum, reference in store.by_checksum.items()}

    everything = {reading.identifier for reading in readings if reading.ready}
    all_files = [store.by_checksum[checksum] for checksum in {checksum for group in groups for checksum in group["files"]}]
    upper = _manifest(readings, everything, all_files, capacity, tolerance)
    reserve = ZIP_END + member_cost("manifest.json", json_document(upper)) + MANIFEST_MARGIN
    if capacity <= reserve:
        raise DeskError("The byte ceiling is too small to hold the reading list.", field="max_bytes")

    budget = capacity - reserve
    manifest = content = files = data = None
    for _ in range(4):
        chosen = _solve(groups, costs, budget)
        # Loading a selection turns its estimates into exact packed sizes.
        manifest, content, files, data = _render(readings, chosen, store, capacity, tolerance)
        if len(data) <= capacity:
            break
        costs = {checksum: reference.cost for checksum, reference in store.by_checksum.items()}
        overshoot = len(data) - capacity
        if budget <= overshoot + MANIFEST_MARGIN:
            data = None
            break
        budget -= overshoot + MANIFEST_MARGIN
    if data is None or len(data) > capacity:
        raise DeskError(
            "The byte ceiling is too small for these originals. Plan fewer readings or raise the ceiling.",
            field="max_bytes",
        )

    report = {
        "capacity_bytes": capacity,
        "package_bytes": len(data),
        "within_capacity": True,
        "tolerance": tolerance,
        "summary": manifest["summary"],
        "readings": manifest["readings"],
        "files": manifest["files"],
    }
    return data, report


def build_batch_package(catalog, roll_id, value):
    return _build(catalog, roll_id, mapping(value))


def list_density_readings(catalog, roll_id):
    catalog.get_roll(roll_id)
    records = catalog.list_records(roll_id, kind="density", limit=MAX_READINGS + 1)
    truncated = len(records) > MAX_READINGS
    records = records[:MAX_READINGS]
    rows = []
    for record in records:
        reading = _analyse(catalog, record, False)
        rows.append(
            {
                "id": record["id"],
                "title": record["title"],
                "decision": record["decision"],
                "created": record["created"],
                "scan_id": record["scan_id"],
                "scan_name": record.get("scan_name"),
                "region": reading.region.as_dict() if reading.region else None,
                "retestable": reading.ready,
                "missing": list(reading.missing),
            }
        )
    return rows, truncated


class BatchStore:
    """Recently planned packages, so the preview and its download are identical."""

    def __init__(self, byte_limit=384 * 1024 * 1024, lifetime=20 * 60):
        self.byte_limit = byte_limit
        self.lifetime = lifetime
        self.entries = OrderedDict()
        self.used = 0
        self.lock = RLock()

    def _drop(self, token):
        data, _, _ = self.entries.pop(token)
        self.used -= len(data)

    def _prune(self):
        cutoff = monotonic() - self.lifetime
        for token in [token for token, (_, _, born) in self.entries.items() if born < cutoff]:
            self._drop(token)

    def put(self, data, report, filename):
        token = key()
        with self.lock:
            self._prune()
            while self.entries and (len(self.entries) >= 4 or self.used + len(data) > self.byte_limit):
                self._drop(next(iter(self.entries)))
            self.entries[token] = (data, filename, monotonic())
            self.used += len(data)
        return token

    def get(self, token):
        checked_key(token, "token")
        with self.lock:
            self._prune()
            entry = self.entries.get(token)
            if entry is None:
                raise DeskError("The planned package expired. Plan it again.", status=404, code="expired")
            data, filename, _ = entry
            return data, filename


def _number_or_none(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _file_maps(manifest):
    if not isinstance(manifest.get("files"), list) or not isinstance(manifest.get("readings"), list):
        raise _batch_error("The batch package is missing its file or reading list.")
    if len(manifest["readings"]) > MAX_READINGS or len(manifest["files"]) > MAX_PACKET_FILES:
        raise _batch_error("The batch package has too many entries.")
    tolerance = manifest.get("tolerance")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or not 0 < tolerance <= 1:
        raise _batch_error("The batch package has an invalid density tolerance.")
    capacity = manifest.get("capacity_bytes")
    if type(capacity) is not int or capacity < 0:
        raise _batch_error("The batch package has an invalid byte ceiling.")
    files = {}
    names = set()
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise _batch_error("The batch package lists an invalid file.")
        name = item.get("name")
        checksum = item.get("sha256")
        size = item.get("bytes")
        if not isinstance(name, str):
            raise _batch_error("The batch package lists an invalid file path.")
        if _original_member(name) is None:
            raise _batch_error("The batch package lists an invalid file path.")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise _batch_error("The batch package lists an invalid original checksum.")
        if type(size) is not int or size < 0:
            raise _batch_error("The batch package lists an invalid original size.")
        if checksum in files:
            raise _batch_error("The batch package repeats an original file.")
        if name in names:
            raise _batch_error("The batch package repeats a file path.")
        names.add(name)
        files[checksum] = {"name": name, "sha256": checksum, "bytes": size, "content_type": item.get("content_type")}
    return files, tolerance, capacity


def _original_member(name):
    if "\\" in name or name.startswith("/"):
        return None
    path = PurePosixPath(name)
    if str(path) != name or len(path.parts) != 2 or path.parts[0] != "originals" or ".." in path.parts:
        return None
    return path


def verify_batch_packet(archive, members, manifest, manifest_data, packet_bytes=None):
    files, tolerance, capacity = _file_maps(manifest)
    expected = {"manifest.json"}
    file_rows = {}
    contents = {}
    for checksum, item in files.items():
        name = item["name"]
        expected.add(name)
        row = {"name": name, "sha256": checksum, "status": "missing"}
        if name in members:
            try:
                data = archive.read(members[name])
            except (OSError, RuntimeError, NotImplementedError, ValueError):
                row["status"] = "unreadable"
            else:
                digest = payload_hash(data)
                if len(data) == item["bytes"] and digest == checksum:
                    row["status"] = "verified"
                    contents[checksum] = data
                else:
                    row["status"] = "changed"
                    row["found_sha256"] = digest
                    row["bytes"] = len(data)
        file_rows[checksum] = row
    extra = sorted(set(members) - expected - {"manifest.json"})
    cache = DecodeCache()

    def decoded(checksum):
        if checksum in contents:
            try:
                return cache.read(checksum, contents[checksum]), None
            except DeskError as error:
                return None, error.message
        return None, "the packed original is missing or changed"

    rows = []
    counts = {"retested": 0, "failed": 0, "not-included": 0, "incomplete": 0}
    priority_retested = 0
    for entry in manifest["readings"]:
        if not isinstance(entry, dict):
            raise _batch_error("The batch package lists an invalid reading.")
        record_id = checked_key(entry.get("record_id"), "reading.record_id")
        title = entry.get("title") if isinstance(entry.get("title"), str) else ""
        priority = entry.get("priority") is True
        result = {
            "record_id": record_id,
            "title": title[:100],
            "priority": priority,
            "status": "not-included",
            "reasons": [],
            "channels": [],
        }
        if entry.get("status") != "included":
            reason = entry.get("reason")
            declared = entry.get("missing")
            if isinstance(declared, list) and declared:
                result["status"] = "incomplete"
                result["reasons"] = [item for item in declared if isinstance(item, str)][:8]
                counts["incomplete"] += 1
            else:
                result["reasons"] = [reason if isinstance(reason, str) else "not packed"]
                counts["not-included"] += 1
            rows.append(result)
            continue

        reasons = []
        source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
        width, height = source.get("width"), source.get("height")
        region = None
        if type(width) is not int or type(height) is not int or width < 12 or height < 12:
            reasons.append("the recorded scan dimensions are invalid")
        else:
            try:
                region = Box.read(entry.get("region"), "reading.region", bounds=(width, height))
            except DeskError as error:
                reasons.append(error.message)

        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        roles = {}
        for role in ("primary", "flat", "dark"):
            name = evidence.get(role)
            if not isinstance(name, str):
                reasons.append(f"the {ROLE_TITLES[role]} is not packed")
                continue
            item = next((row for row in files.values() if row["name"] == name), None)
            if item is None:
                reasons.append(f"the {ROLE_TITLES[role]} is not listed")
                continue
            if role == "primary":
                wanted = source.get("sha256")
            else:
                controls = entry.get("controls") if isinstance(entry.get("controls"), dict) else {}
                control = controls.get(role) if isinstance(controls.get(role), dict) else {}
                wanted = control.get("sha256")
            if item["sha256"] != wanted:
                reasons.append(f"the packed {ROLE_TITLES[role]} does not match the recorded checksum")
                continue
            if file_rows[item["sha256"]]["status"] not in {"verified", "changed", "unreadable"}:
                reasons.append(f"the packed {ROLE_TITLES[role]} is missing")
                continue
            roles[role] = item["sha256"]

        try:
            calibration = Calibration.read(entry.get("calibration"), "reading.calibration")
        except DeskError as error:
            reasons.append(error.message)
            calibration = None
        expected_result = entry.get("expected") if isinstance(entry.get("expected"), dict) else {}
        channels = expected_result.get("channels") if isinstance(expected_result.get("channels"), dict) else {}

        if not reasons:
            primary, primary_error = decoded(roles["primary"])
            flat, flat_error = decoded(roles["flat"])
            dark, dark_error = decoded(roles["dark"])
            if primary_error:
                reasons.append(f"original scan: {primary_error}")
            if flat_error:
                reasons.append(f"flat control: {flat_error}")
            if dark_error:
                reasons.append(f"dark control: {dark_error}")
            if not reasons:
                if primary.size != (width, height):
                    reasons.append("the packed original scan dimensions differ from the recorded scan")
                elif flat.size != primary.size or dark.size != primary.size:
                    reasons.append("the packed controls do not match the scan dimensions")
                else:
                    try:
                        calibrated = apply_calibration(primary, calibration, flat=flat, dark=dark)
                        measured = density_stats(calibrated, region)
                    except DeskError as error:
                        reasons.append(error.message)
                    else:
                        for channel in PRIMARY_CHANNELS:
                            wanted = channels.get(channel) if isinstance(channels.get(channel), dict) else {}
                            expected_median = _number_or_none(wanted.get("median"))
                            expected_valid = wanted.get("valid") if type(wanted.get("valid")) is int else None
                            values = measured["channels"][channel]
                            actual_median = values.get("median")
                            actual_valid = values.get("valid")
                            passed = (
                                expected_median is not None
                                and actual_median is not None
                                and abs(actual_median - expected_median) <= tolerance + 1e-9
                                and expected_valid is not None
                                and actual_valid == expected_valid
                            )
                            result["channels"].append(
                                {
                                    "channel": channel,
                                    "expected_median": expected_median,
                                    "measured_median": actual_median,
                                    "delta": round(actual_median - expected_median, 6)
                                    if actual_median is not None and expected_median is not None else None,
                                    "expected_valid": expected_valid,
                                    "measured_valid": actual_valid,
                                    "passed": passed,
                                }
                            )
                            if passed:
                                continue
                            if expected_median is None or expected_valid is None:
                                reasons.append(f"{channel} has no saved median or valid-pixel count")
                            elif actual_median is None:
                                reasons.append(f"{channel} could not be measured in the packed original")
                            else:
                                if abs(actual_median - expected_median) > tolerance + 1e-9:
                                    reasons.append(
                                        f"{channel} median {actual_median:.5f} differs from saved "
                                        f"{expected_median:.5f} beyond {tolerance:g}"
                                    )
                                if actual_valid != expected_valid:
                                    reasons.append(
                                        f"{channel} valid pixels {actual_valid} differ from saved {expected_valid}"
                                    )
        if reasons:
            result["status"] = "failed"
            result["reasons"] = reasons
            counts["failed"] += 1
        else:
            result["status"] = "retested"
            counts["retested"] += 1
            if priority:
                priority_retested += 1
        rows.append(result)

    absent = [row["name"] for row in file_rows.values() if row["status"] == "missing"]
    changed = [row["name"] for row in file_rows.values() if row["status"] in {"changed", "unreadable"}]
    included = sum(1 for entry in manifest["readings"] if entry.get("status") == "included")
    complete = counts["retested"] == included and counts["retested"] > 0 and not (absent or changed or extra)
    return {
        "status": "retested" if complete else "review",
        "format": "batch",
        "record": {"kind": "batch", "title": "Offline re-check package"},
        "capacity_bytes": capacity,
        "package_bytes": packet_bytes if packet_bytes is not None else len(manifest_data),
        "tolerance": tolerance,
        "summary": {
            "requested": len(manifest["readings"]),
            "included": included,
            "retested": counts["retested"],
            "failed": counts["failed"],
            "not_included": counts["not-included"],
            "incomplete": counts["incomplete"],
            "priority_requested": sum(1 for entry in manifest["readings"] if entry.get("priority") is True),
            "priority_retested": priority_retested,
        },
        "readings": rows,
        "files": list(file_rows.values()),
        "missing_files": absent,
        "changed_files": changed,
        "unexpected_files": extra,
        "manifest_sha256": payload_hash(manifest_data),
    }
