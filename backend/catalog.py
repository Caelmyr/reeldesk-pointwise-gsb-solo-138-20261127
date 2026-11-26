"""SQLite catalog for physical rolls, immutable image originals and saved decisions."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3

from .values import (
    DeskError, as_json, checked_key, conflict, key, missing, now,
    json_hash, parse_json, payload_hash,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS rolls (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    stock TEXT NOT NULL,
    code TEXT NOT NULL,
    location TEXT NOT NULL,
    notes TEXT NOT NULL,
    tags TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    revision INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS rolls_code_lower ON rolls(lower(code));
CREATE TABLE IF NOT EXISTS originals (
    checksum TEXT PRIMARY KEY,
    content BLOB NOT NULL,
    bytes INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    roll_id TEXT NOT NULL REFERENCES rolls(id) ON DELETE CASCADE,
    checksum TEXT NOT NULL REFERENCES originals(checksum),
    name TEXT NOT NULL,
    original_name TEXT NOT NULL,
    role TEXT NOT NULL,
    medium TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    mode TEXT NOT NULL,
    source_format TEXT NOT NULL,
    orientation INTEGER NOT NULL,
    dpi REAL,
    scanner TEXT NOT NULL,
    exposure TEXT NOT NULL,
    notes TEXT NOT NULL,
    verdict TEXT NOT NULL DEFAULT 'hold',
    sample INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    revision INTEGER NOT NULL,
    UNIQUE (roll_id, checksum, role)
);
CREATE INDEX IF NOT EXISTS scans_roll ON scans(roll_id, created DESC);
CREATE TABLE IF NOT EXISTS marks (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    geometry TEXT NOT NULL,
    notes TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    revision INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS marks_scan ON marks(scan_id, kind, created);
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    roll_id TEXT NOT NULL REFERENCES rolls(id) ON DELETE CASCADE,
    scan_id TEXT NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    peer_id TEXT REFERENCES scans(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    settings TEXT NOT NULL,
    result TEXT NOT NULL,
    decision TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    revision INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS records_roll ON records(roll_id, kind, created DESC);
CREATE INDEX IF NOT EXISTS records_scan ON records(scan_id, created DESC);
CREATE INDEX IF NOT EXISTS records_peer ON records(peer_id) WHERE peer_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    checksum TEXT NOT NULL,
    content BLOB NOT NULL,
    bytes INTEGER NOT NULL,
    created TEXT NOT NULL,
    UNIQUE (record_id, name)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_id TEXT NOT NULL REFERENCES rolls(id) ON DELETE CASCADE,
    subject_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    details TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_roll ON events(roll_id, id DESC);
CREATE TABLE IF NOT EXISTS settings (
    setting_key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

ROLL_FIELDS = ("title", "stock", "code", "location", "notes", "tags")
SCAN_FIELDS = ("name", "medium", "dpi", "scanner", "exposure", "notes", "verdict")
MARK_FIELDS = ("kind", "label", "geometry", "notes")
RECORD_FIELDS = ("title", "decision")


def roll_dict(row):
    result = dict(row)
    result["tags"] = parse_json(result["tags"])
    return result


def scan_dict(row):
    result = dict(row)
    result["sample"] = bool(result["sample"])
    result.pop("content", None)
    return result


def mark_dict(row):
    result = dict(row)
    result["geometry"] = parse_json(result["geometry"])
    return result


def record_dict(row):
    result = dict(row)
    result["settings"] = parse_json(result["settings"])
    result["result"] = parse_json(result["result"])
    return result


def event_dict(row):
    result = dict(row)
    result["details"] = parse_json(result["details"])
    return result


def _escape_like(value):
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Catalog:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "reeldesk.sqlite3"
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
            connection.execute("PRAGMA user_version=1")

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.file, timeout=8, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=8000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def writing(self, connection):
        if connection is not None:
            yield connection
        else:
            with self.transaction() as active:
                yield active

    def health(self):
        try:
            with self.connection() as connection:
                connection.execute("SELECT id FROM rolls LIMIT 1").fetchone()
                schema = connection.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.Error as error:
            raise DeskError("The catalog is unavailable.", status=503, code="catalog_unavailable") from error
        return {"status": "ok", "schema": schema}

    def _one(self, connection, sql, args, label):
        row = connection.execute(sql, args).fetchone()
        if row is None:
            raise missing(label)
        return row

    def _get_roll(self, connection, roll_id):
        checked_key(roll_id, "roll_id")
        return self._one(connection, "SELECT * FROM rolls WHERE id=?", (roll_id,), "Roll")

    def _get_scan(self, connection, scan_id):
        checked_key(scan_id, "scan_id")
        return self._one(
            connection,
            """
            SELECT scans.*, originals.bytes AS byte_size FROM scans
            JOIN originals USING(checksum) WHERE scans.id=?
            """,
            (scan_id,),
            "Scan",
        )

    def _get_mark(self, connection, mark_id):
        checked_key(mark_id, "mark_id")
        return self._one(connection, "SELECT * FROM marks WHERE id=?", (mark_id,), "Mark")

    def _get_record(self, connection, record_id):
        checked_key(record_id, "record_id")
        return self._one(connection, "SELECT * FROM records WHERE id=?", (record_id,), "Record")

    def _audit(self, connection, roll_id, subject_id, operation, details):
        connection.execute(
            "INSERT INTO events(roll_id,subject_id,operation,details,created) VALUES(?,?,?,?,?)",
            (roll_id, subject_id, operation, as_json(details), now()),
        )

    def _touch_roll(self, connection, roll_id):
        connection.execute("UPDATE rolls SET updated=? WHERE id=?", (now(), roll_id))

    def _expect_revision(self, row, revision):
        if row["revision"] != revision:
            raise conflict(f"This item changed in another window. Reload revision {row['revision']} and try again.")

    def list_rolls(self, *, query="", tag="", limit=40, offset=0):
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT r.*,
                       (SELECT count(*) FROM scans s WHERE s.roll_id=r.id AND s.role='scan') AS scan_count,
                       (SELECT count(*) FROM scans s WHERE s.roll_id=r.id AND s.role!='scan') AS control_count,
                       (SELECT count(*) FROM records k WHERE k.roll_id=r.id) AS record_count,
                       (SELECT s.id FROM scans s WHERE s.roll_id=r.id AND s.role='scan'
                         ORDER BY s.created, s.id LIMIT 1) AS cover_id
                FROM rolls r
                WHERE (
                    ?='' OR lower(r.title) LIKE ? ESCAPE '\\'
                         OR lower(r.stock) LIKE ? ESCAPE '\\'
                         OR lower(r.code) LIKE ? ESCAPE '\\'
                         OR lower(r.location) LIKE ? ESCAPE '\\'
                ) AND (
                    ?='' OR EXISTS (SELECT 1 FROM json_each(r.tags) t WHERE t.value=?)
                )
                ORDER BY r.updated DESC, r.id
                LIMIT ? OFFSET ?
                """,
                (
                    query,
                    *([f"%{_escape_like(query.lower())}%"] * 4),
                    tag,
                    tag,
                    limit,
                    offset,
                ),
            ).fetchall()
            return [roll_dict(row) for row in rows]

    def get_roll(self, roll_id):
        with self.connection() as connection:
            return roll_dict(self._get_roll(connection, roll_id))

    def create_roll(self, fields, *, connection=None):
        roll_id = key()
        timestamp = now()
        payload = {field: fields[field] for field in ROLL_FIELDS}
        payload["tags"] = as_json(payload["tags"])
        with self.writing(connection) as active:
            try:
                active.execute(
                    """
                    INSERT INTO rolls(id,title,stock,code,location,notes,tags,created,updated,revision)
                    VALUES(?,?,?,?,?,?,?,?,?,1)
                    """,
                    (roll_id, *payload.values(), timestamp, timestamp),
                )
            except sqlite3.IntegrityError as error:
                raise conflict("That roll code is already in use.") from error
            self._audit(active, roll_id, roll_id, "roll.created", {"code": fields["code"]})
            return roll_dict(self._get_roll(active, roll_id))

    def update_roll(self, roll_id, fields, revision):
        with self.transaction() as connection:
            current = self._get_roll(connection, roll_id)
            self._expect_revision(current, revision)
            changes = {field: fields[field] for field in ROLL_FIELDS if field in fields}
            if "tags" in changes:
                changes["tags"] = as_json(changes["tags"])
            if not changes:
                return roll_dict(current)
            columns = ", ".join(f"{field}=?" for field in changes)
            try:
                connection.execute(
                    f"UPDATE rolls SET {columns}, updated=?, revision=revision+1 WHERE id=?",
                    (*changes.values(), now(), roll_id),
                )
            except sqlite3.IntegrityError as error:
                raise conflict("That roll code is already in use.") from error
            self._audit(connection, roll_id, roll_id, "roll.edited", {"fields": list(changes)})
            return roll_dict(self._get_roll(connection, roll_id))

    def list_scans(self, roll_id, *, role="", verdict="", limit=100, offset=0):
        with self.connection() as connection:
            self._get_roll(connection, roll_id)
            rows = connection.execute(
                """
                SELECT scans.*, originals.bytes AS byte_size,
                    (SELECT count(*) FROM marks m WHERE m.scan_id=scans.id) AS mark_count
                FROM scans JOIN originals USING(checksum)
                WHERE roll_id=? AND (?='' OR role=?) AND (?='' OR verdict=?)
                ORDER BY CASE role WHEN 'scan' THEN 0 ELSE 1 END, scans.created, scans.id
                LIMIT ? OFFSET ?
                """,
                (roll_id, role, role, verdict, verdict, limit, offset),
            ).fetchall()
            return [scan_dict(row) for row in rows]

    def get_scan(self, scan_id):
        with self.connection() as connection:
            return scan_dict(self._get_scan(connection, scan_id))

    def original(self, scan_id):
        with self.connection() as connection:
            scan = self._get_scan(connection, scan_id)
            original = connection.execute(
                "SELECT content FROM originals WHERE checksum=?", (scan["checksum"],)
            ).fetchone()
            return scan_dict(scan), original["content"]

    def add_scan(self, roll_id, fields, original, probe, *, sample=False, connection=None):
        timestamp = now()
        scan_id = key()
        checksum = payload_hash(original)
        with self.writing(connection) as active:
            self._get_roll(active, roll_id)
            duplicate = active.execute(
                "SELECT id FROM scans WHERE roll_id=? AND checksum=? AND role=?",
                (roll_id, checksum, fields["role"]),
            ).fetchone()
            if duplicate:
                raise DeskError(
                    "These exact image bytes are already stored for this role on the roll.",
                    status=409,
                    code="duplicate_scan",
                    details={"scan_id": duplicate["id"]},
                )
            active.execute(
                "INSERT OR IGNORE INTO originals(checksum, content, bytes) VALUES(?,?,?)",
                (checksum, original, len(original)),
            )
            active.execute(
                """
                INSERT INTO scans(
                    id,roll_id,checksum,name,original_name,role,medium,width,height,mode,
                    source_format,orientation,dpi,scanner,exposure,notes,sample,created,updated,revision
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
                """,
                (
                    scan_id, roll_id, checksum, fields["name"], fields["original_name"],
                    fields["role"], fields["medium"], probe["width"], probe["height"],
                    probe["mode"], probe["source_format"], probe["orientation"], fields["dpi"],
                    fields["scanner"], fields["exposure"], fields["notes"], int(sample),
                    timestamp, timestamp,
                ),
            )
            self._touch_roll(active, roll_id)
            self._audit(active, roll_id, scan_id, "scan.imported", {"role": fields["role"], "checksum": checksum})
            return scan_dict(self._get_scan(active, scan_id))

    def update_scan(self, scan_id, fields, revision):
        with self.transaction() as connection:
            current = self._get_scan(connection, scan_id)
            self._expect_revision(current, revision)
            changes = {field: fields[field] for field in SCAN_FIELDS if field in fields}
            if not changes:
                return scan_dict(current)
            columns = ", ".join(f"{field}=?" for field in changes)
            connection.execute(
                f"UPDATE scans SET {columns}, updated=?, revision=revision+1 WHERE id=?",
                (*changes.values(), now(), scan_id),
            )
            self._touch_roll(connection, current["roll_id"])
            self._audit(
                connection, current["roll_id"], scan_id, "scan.edited",
                {"fields": list(changes), "verdict": changes.get("verdict", current["verdict"])},
            )
            return scan_dict(self._get_scan(connection, scan_id))

    def delete_scan(self, scan_id, revision):
        with self.transaction() as connection:
            scan = self._get_scan(connection, scan_id)
            self._expect_revision(scan, revision)
            referenced = connection.execute(
                """
                SELECT id FROM records WHERE scan_id=? OR peer_id=?
                UNION SELECT r.id FROM records r, json_each(r.settings, '$.support_inventory') origin
                 WHERE json_extract(origin.value, '$.scan_id')=?
                UNION SELECT id FROM records
                 WHERE json_extract(settings, '$.calibration.flat_id')=?
                    OR json_extract(settings, '$.calibration.dark_id')=?
                    OR json_extract(settings, '$.source_calibration.flat_id')=?
                    OR json_extract(settings, '$.source_calibration.dark_id')=?
                    OR json_extract(settings, '$.peer_calibration.flat_id')=?
                    OR json_extract(settings, '$.peer_calibration.dark_id')=?
                LIMIT 1
                """,
                (scan_id,) * 9,
            ).fetchone()
            if referenced:
                raise conflict("A saved record cites this scan. Remove the record before removing the scan.")
            connection.execute("DELETE FROM scans WHERE id=?", (scan_id,))
            connection.execute(
                """
                DELETE FROM originals WHERE checksum=?
                 AND NOT EXISTS(SELECT 1 FROM scans WHERE checksum=?)
                """,
                (scan["checksum"], scan["checksum"]),
            )
            self._touch_roll(connection, scan["roll_id"])
            self._audit(connection, scan["roll_id"], scan_id, "scan.removed", {"name": scan["name"]})

    def _mark_rows(self, connection, scan_id, kind=""):
        return connection.execute(
            "SELECT * FROM marks WHERE scan_id=? AND (?='' OR kind=?) ORDER BY created,id",
            (scan_id, kind, kind),
        ).fetchall()

    def list_marks(self, scan_id, kind=""):
        with self.connection() as connection:
            self._get_scan(connection, scan_id)
            return [mark_dict(row) for row in self._mark_rows(connection, scan_id)]

    @staticmethod
    def marks_version(marks):
        return json_hash([{"id": mark["id"], "revision": mark["revision"]} for mark in marks])

    def get_mark(self, mark_id):
        with self.connection() as connection:
            return mark_dict(self._get_mark(connection, mark_id))

    def add_mark(self, scan_id, fields, *, connection=None):
        mark_id = key()
        timestamp = now()
        with self.writing(connection) as active:
            scan = self._get_scan(active, scan_id)
            count = active.execute("SELECT count(*) FROM marks WHERE scan_id=?", (scan_id,)).fetchone()[0]
            if count >= 300:
                raise conflict("This scan already has 300 marks.")
            active.execute(
                """
                INSERT INTO marks(id,scan_id,kind,label,geometry,notes,created,updated,revision)
                VALUES(?,?,?,?,?,?,?,?,1)
                """,
                (
                    mark_id, scan_id, fields["kind"], fields["label"], as_json(fields["geometry"]),
                    fields["notes"], timestamp, timestamp,
                ),
            )
            self._touch_roll(active, scan["roll_id"])
            self._audit(active, scan["roll_id"], mark_id, "mark.added", {"scan_id": scan_id, "kind": fields["kind"]})
            return mark_dict(self._get_mark(active, mark_id))

    def update_mark(self, mark_id, fields, revision):
        with self.transaction() as connection:
            current = self._get_mark(connection, mark_id)
            self._expect_revision(current, revision)
            scan = self._get_scan(connection, current["scan_id"])
            changes = {field: fields[field] for field in MARK_FIELDS if field in fields}
            if "geometry" in changes:
                changes["geometry"] = as_json(changes["geometry"])
            if not changes:
                return mark_dict(current)
            columns = ", ".join(f"{field}=?" for field in changes)
            connection.execute(
                f"UPDATE marks SET {columns}, updated=?, revision=revision+1 WHERE id=?",
                (*changes.values(), now(), mark_id),
            )
            self._touch_roll(connection, scan["roll_id"])
            self._audit(connection, scan["roll_id"], mark_id, "mark.edited", {"scan_id": scan["id"], "fields": list(changes)})
            return mark_dict(self._get_mark(connection, mark_id))

    def delete_mark(self, mark_id, revision):
        with self.transaction() as connection:
            mark = self._get_mark(connection, mark_id)
            self._expect_revision(mark, revision)
            scan = self._get_scan(connection, mark["scan_id"])
            connection.execute("DELETE FROM marks WHERE id=?", (mark_id,))
            self._touch_roll(connection, scan["roll_id"])
            self._audit(connection, scan["roll_id"], mark_id, "mark.removed", {"scan_id": scan["id"], "kind": mark["kind"]})

    def list_records(self, roll_id, *, scan_id=None, kind="", limit=40, offset=0):
        with self.connection() as connection:
            self._get_roll(connection, roll_id)
            if scan_id is not None:
                self._get_scan(connection, scan_id)
            rows = connection.execute(
                """
                SELECT records.*,
                    scans.name AS scan_name,
                    (SELECT count(*) FROM evidence WHERE record_id=records.id) AS file_count
                FROM records JOIN scans ON records.scan_id=scans.id
                WHERE records.roll_id=? AND (? IS NULL OR scan_id=?) AND (?='' OR kind=?)
                ORDER BY records.created DESC, records.id
                LIMIT ? OFFSET ?
                """,
                (roll_id, scan_id, scan_id, kind, kind, limit, offset),
            ).fetchall()
            return [record_dict(row) for row in rows]

    def get_record(self, record_id):
        with self.connection() as connection:
            result = record_dict(self._get_record(connection, record_id))
            files = connection.execute(
                """
                SELECT id,name,content_type,checksum,bytes,created FROM evidence
                WHERE record_id=? ORDER BY name
                """,
                (record_id,),
            ).fetchall()
            result["files"] = [dict(row) for row in files]
            return result

    @staticmethod
    def source_identity(scan, role):
        return {
            "role": role,
            "scan_id": scan["id"],
            "name": scan["name"],
            "sha256": scan["checksum"],
            "width": scan["width"],
            "height": scan["height"],
            "dpi": scan["dpi"],
            "orientation": scan["orientation"],
            "synthetic_demo": bool(scan["sample"]),
        }

    def _record_origins(self, connection, settings, roll_id):
        originals = {}
        seen = set()
        for section in ("calibration", "source_calibration", "peer_calibration"):
            selection = settings.get(section) or {}
            for field, role in (("flat_id", "flat"), ("dark_id", "dark")):
                scan_id = selection.get(field)
                if not scan_id:
                    continue
                control = self._get_scan(connection, scan_id)
                if control["roll_id"] != roll_id or control["role"] != role:
                    raise conflict("A calibration control ceased to be available before saving.")
                if scan_id not in originals:
                    originals[scan_id] = self.source_identity(control, role)
                originals[scan_id].setdefault("used_by", []).append(section)
                seen.add(scan_id)
        for additional in settings.get("scans", ()):
            scan_id = additional["id"]
            if scan_id in seen:
                continue
            scan = self._get_scan(connection, scan_id)
            if scan["roll_id"] != roll_id or scan["checksum"] != additional["checksum"]:
                raise conflict("A contact-sheet source changed before saving.")
            originals[scan_id] = self.source_identity(scan, "contact-image")
        return list(originals.values())

    def add_record(self, scan_id, kind, title, settings, result, *, peer_id=None, files=(), expected=None):
        record_id = key()
        timestamp = now()
        expected = expected or {}
        with self.transaction() as connection:
            scan = self._get_scan(connection, scan_id)
            if "source_revision" in expected and scan["revision"] != expected["source_revision"]:
                raise conflict("The primary scan metadata changed. Run the operation again before saving.")
            parent_origins = [self.source_identity(scan, "primary")]
            if peer_id:
                peer = self._get_scan(connection, peer_id)
                if peer["roll_id"] != scan["roll_id"] or peer_id == scan_id:
                    raise DeskError("Choose a different scan from this roll.", field="peer_id")
                if "peer_revision" in expected and peer["revision"] != expected["peer_revision"]:
                    raise conflict("The repeat scan metadata changed. Run the operation again before saving.")
                parent_origins.append(self.source_identity(peer, "repeat-pass"))
            if expected.get("marks_version") is not None:
                current = self.marks_version(self._mark_rows(connection, scan_id))
                if current != expected["marks_version"]:
                    raise conflict("Scan marks changed after the preview. Run the operation again before saving.")
            settings = {
                **settings,
                "source_inventory": parent_origins,
                "support_inventory": self._record_origins(connection, settings, scan["roll_id"]),
            }
            connection.execute(
                """
                INSERT INTO records(id,roll_id,scan_id,peer_id,kind,title,settings,result,decision,created,updated,revision)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,1)
                """,
                (
                    record_id, scan["roll_id"], scan_id, peer_id, kind, title,
                    as_json(settings), as_json(result), "hold", timestamp, timestamp,
                ),
            )
            for name, content_type, content in files:
                self._add_evidence(connection, record_id, name, content_type, content, timestamp)
            self._touch_roll(connection, scan["roll_id"])
            self._audit(
                connection, scan["roll_id"], record_id, "record.saved",
                {"kind": kind, "scan_id": scan_id, "peer_id": peer_id},
            )
        return self.get_record(record_id)

    def _add_evidence(self, connection, record_id, name, content_type, content, timestamp):
        connection.execute(
            """
            INSERT INTO evidence(id,record_id,name,content_type,checksum,content,bytes,created)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (key(), record_id, name, content_type, payload_hash(content), content, len(content), timestamp),
        )

    def update_record(self, record_id, fields, revision):
        with self.transaction() as connection:
            record = self._get_record(connection, record_id)
            self._expect_revision(record, revision)
            changes = {field: fields[field] for field in RECORD_FIELDS if field in fields}
            if changes:
                columns = ", ".join(f"{field}=?" for field in changes)
                connection.execute(
                    f"UPDATE records SET {columns}, updated=?, revision=revision+1 WHERE id=?",
                    (*changes.values(), now(), record_id),
                )
                self._touch_roll(connection, record["roll_id"])
                self._audit(
                    connection, record["roll_id"], record_id, "record.edited",
                    {"fields": list(changes), "decision": changes.get("decision", record["decision"])},
                )
        return self.get_record(record_id)

    def delete_record(self, record_id, revision):
        with self.transaction() as connection:
            record = self._get_record(connection, record_id)
            self._expect_revision(record, revision)
            connection.execute("DELETE FROM records WHERE id=?", (record_id,))
            self._touch_roll(connection, record["roll_id"])
            self._audit(connection, record["roll_id"], record_id, "record.removed", {"kind": record["kind"]})

    def evidence_file(self, record_id, file_id):
        checked_key(file_id, "file_id")
        with self.connection() as connection:
            self._get_record(connection, record_id)
            row = self._one(
                connection,
                "SELECT * FROM evidence WHERE record_id=? AND id=?",
                (record_id, file_id),
                "Evidence file",
            )
            return dict(row)

    def read_review_packet(self, record_id, include_originals):
        with self.connection() as connection:
            connection.execute("BEGIN")
            record = record_dict(self._get_record(connection, record_id))
            scan = scan_dict(self._get_scan(connection, record["scan_id"]))
            peer = scan_dict(self._get_scan(connection, record["peer_id"])) if record["peer_id"] else None
            exhibits = connection.execute(
                "SELECT * FROM evidence WHERE record_id=? ORDER BY name", (record_id,)
            ).fetchall()
            originals = []
            if include_originals:
                for role, source in (("primary", scan), ("repeat-pass", peer)):
                    if source is None:
                        continue
                    row = self._one(
                        connection, "SELECT content FROM originals WHERE checksum=?",
                        (source["checksum"],), "Original",
                    )
                    originals.append((role, source, row["content"]))
            return record, scan, peer, [dict(row) for row in exhibits], originals

    def activity(self, roll_id, limit=40, before=None):
        with self.connection() as connection:
            self._get_roll(connection, roll_id)
            rows = connection.execute(
                """
                SELECT * FROM events WHERE roll_id=? AND (? IS NULL OR id<?)
                ORDER BY id DESC LIMIT ?
                """,
                (roll_id, before, before, limit),
            ).fetchall()
            return [event_dict(row) for row in rows]

    def summary(self):
        with self.connection() as connection:
            scans = connection.execute(
                "SELECT role, count(*) AS amount FROM scans GROUP BY role ORDER BY role"
            ).fetchall()
            records = connection.execute(
                "SELECT decision, count(*) AS amount FROM records GROUP BY decision ORDER BY decision"
            ).fetchall()
            originals = connection.execute(
                "SELECT coalesce(sum(bytes),0) AS bytes, count(*) AS amount FROM originals"
            ).fetchone()
            rolls = connection.execute("SELECT count(*) FROM rolls").fetchone()[0]
            return {
                "rolls": rolls,
                "scans": {row["role"]: row["amount"] for row in scans},
                "records": {row["decision"]: row["amount"] for row in records},
                "original_bytes": originals["bytes"],
                "unique_originals": originals["amount"],
            }

    def setting_in_transaction(self, connection, name):
        row = connection.execute("SELECT value FROM settings WHERE setting_key=?", (name,)).fetchone()
        return None if row is None else parse_json(row["value"])

    def remember_setting(self, connection, name, value):
        connection.execute(
            "INSERT INTO settings(setting_key,value) VALUES(?,?) "
            "ON CONFLICT(setting_key) DO UPDATE SET value=excluded.value",
            (name, as_json(value)),
        )
