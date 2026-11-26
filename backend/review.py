"""Check exported review packets against their manifest and local originals."""

from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile
import re

from .batch import BATCH_FORMAT, verify_batch_packet
from .evidence import record_csv
from .values import DeskError, checked_key, clean_text, missing, parse_json, payload_hash


PACKET_UPLOAD_LIMIT = 95 * 1024 * 1024
PACKET_EXPANDED_LIMIT = 320 * 1024 * 1024
PACKET_MEMBER_LIMIT = 700
SHASUM = re.compile(r"^[0-9a-f]{64}$")


def _packet_error(message):
    return DeskError(message, field="packet", code="invalid_packet")


def _unpack(blob):
    if not blob or len(blob) > PACKET_UPLOAD_LIMIT:
        raise _packet_error("Use a ReelDesk ZIP of up to 95 MiB.")
    try:
        return ZipFile(BytesIO(blob))
    except BadZipFile as error:
        raise _packet_error("The file is not a readable ZIP packet.") from error


def _validate_member(info, seen):
    name = info.filename
    path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or ".." in path.parts or str(path) != name:
        raise _packet_error("The packet contains an invalid file path.")
    if info.flag_bits & 1 or info.is_dir():
        raise _packet_error("The packet contains a folder entry or an encrypted file.")
    if name in seen:
        raise _packet_error("The packet contains repeated paths.")
    if info.file_size > 40 * 1024 * 1024:
        raise _packet_error("A single packet entry exceeds 40 MiB.")
    return name


def _read_members(archive):
    seen = {}
    total = 0
    members = archive.infolist()
    if len(members) > PACKET_MEMBER_LIMIT:
        raise _packet_error("Use a packet with at most 700 files.")
    for info in members:
        name = _validate_member(info, seen)
        total += info.file_size
        if total > PACKET_EXPANDED_LIMIT:
            raise _packet_error("The packet expands beyond the supported 320 MiB.")
        seen[name] = info
    return seen


def _packet_manifest(archive, members):
    entry = members.get("manifest.json")
    if entry is None or entry.file_size > 4 * 1024 * 1024:
        raise _packet_error("The packet is missing a readable manifest.")
    try:
        data = archive.read(entry)
        manifest = parse_json(data)
    except (ValueError, TypeError, BadZipFile, RuntimeError, UnicodeError) as error:
        raise _packet_error("The packet manifest is invalid.") from error
    if not isinstance(manifest, dict) or manifest.get("format") not in {"reeldesk-review-v1", BATCH_FORMAT}:
        raise _packet_error("Choose an exported ReelDesk review packet.")
    if not isinstance(manifest.get("record"), dict) and manifest.get("format") != BATCH_FORMAT:
        raise _packet_error("The review packet is missing its record or result.")
    if manifest.get("format") == BATCH_FORMAT:
        return manifest, data
    if not isinstance(manifest.get("result"), dict):
        raise _packet_error("The review packet is missing its record or result.")
    if not isinstance(manifest.get("sources"), list) or not isinstance(manifest.get("files"), list):
        raise _packet_error("The review packet is missing source and file details.")
    return manifest, data


def _manifest_file(item, seen):
    if not isinstance(item, dict):
        raise _packet_error("The manifest contains an invalid evidence entry.")
    name = item.get("name")
    checksum = item.get("sha256")
    size = item.get("bytes")
    if not isinstance(name, str) or not name or str(PurePosixPath(name)) != name or len(PurePosixPath(name).parts) != 1:
        raise _packet_error("The manifest lists an invalid evidence name.")
    if not isinstance(checksum, str) or not SHASUM.fullmatch(checksum):
        raise _packet_error("The manifest lists an invalid evidence checksum.")
    if type(size) is not int or size < 0:
        raise _packet_error("The manifest lists an invalid evidence size.")
    path = "evidence/" + name
    if path in seen:
        raise _packet_error("The manifest repeats an evidence file.")
    seen.add(path)
    return name, path, checksum, size


def _checked_evidence(archive, members, manifest):
    rows = []
    expected = {"manifest.json"}
    for item in manifest["files"]:
        name, path, checksum, size = _manifest_file(item, expected)
        if path not in members:
            rows.append({"name": name, "status": "missing", "sha256": checksum})
            continue
        try:
            data = archive.read(members[path])
        except (BadZipFile, RuntimeError, NotImplementedError):
            rows.append({"name": name, "status": "unreadable", "sha256": checksum})
            continue
        digest = payload_hash(data)
        rows.append(
            {
                "name": name, "status": "verified" if len(data) == size and digest == checksum else "changed",
                "sha256": checksum, "found_sha256": digest, "bytes": len(data),
            }
        )
    try:
        csv_data = record_csv(manifest["record"].get("kind"), manifest["result"])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise _packet_error("The manifest result has invalid reading data.") from error
    if csv_data is not None:
        name = "reading.csv"
        expected.add(name)
        try:
            actual = archive.read(members[name]) if name in members else None
        except (BadZipFile, RuntimeError, NotImplementedError):
            actual = None
        if actual is None:
            status = "missing" if name not in members else "unreadable"
        else:
            status = "verified" if actual == csv_data else "changed"
        rows.append({"name": name, "status": status, "sha256": payload_hash(csv_data)})
    return rows, expected


def _source_state(member_name, archive, members, checksum):
    if member_name not in members:
        return "not-included"
    try:
        data = archive.read(members[member_name])
    except (BadZipFile, RuntimeError, NotImplementedError):
        return "unreadable"
    return "verified" if payload_hash(data) == checksum else "changed"


def _local_state(catalog, identifier, checksum):
    if catalog is None:
        return "unchecked"
    try:
        local = catalog.get_scan(identifier)
    except DeskError as error:
        if error.code != "not_found":
            raise
        return "missing"
    return "matched" if local["checksum"] == checksum else "different"


def _source_member(archive, members, source, catalog, expected):
    if not isinstance(source, dict) or source.get("role") not in {"primary", "repeat-pass"}:
        raise _packet_error("The manifest contains an unknown source role.")
    identifier = source.get("scan_id")
    checked_key(identifier, "source.scan_id")
    checksum = source.get("sha256")
    if not isinstance(checksum, str) or not SHASUM.fullmatch(checksum):
        raise _packet_error("The manifest contains an invalid original checksum.")
    role = source["role"]
    label = clean_text(source.get("name"), "source.name", maximum=100)
    prefix = f"originals/{role}-"
    paths = [name for name in members if name.startswith(prefix)]
    if len(paths) > 1:
        raise _packet_error("The packet includes multiple originals for a single source role.")
    member = paths[0] if paths else None
    if member:
        expected.add(member)
    return {
        "role": role,
        "scan_id": identifier,
        "name": label,
        "sha256": checksum,
        "packet": _source_state(member, archive, members, checksum),
        "local": _local_state(catalog, identifier, checksum),
    }


def _support_sources(manifest, catalog):
    rows = manifest.get("support_sources", [])
    if not isinstance(rows, list) or len(rows) > 30:
        raise _packet_error("The packet has an invalid list of support scans.")
    results = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("role") not in {"flat", "dark", "contact-image"}:
            raise _packet_error("The packet has an unknown support scan role.")
        key = checked_key(row.get("scan_id"), "support.scan_id")
        checksum = row.get("sha256")
        if not isinstance(checksum, str) or not SHASUM.fullmatch(checksum):
            raise _packet_error("The packet has an invalid support scan checksum.")
        if key in seen:
            raise _packet_error("The packet repeats a support scan.")
        seen.add(key)
        results.append(
            {
                "role": row["role"], "scan_id": key,
                "name": clean_text(row.get("name"), "support.name", maximum=100),
                "sha256": checksum, "local": _local_state(catalog, key, checksum),
                "packet": "metadata-only",
            }
        )
    return results


def verify_packet(blob, catalog=None):
    with _unpack(blob) as archive:
        members = _read_members(archive)
        manifest, data = _packet_manifest(archive, members)
        if manifest.get("format") == BATCH_FORMAT:
            return verify_batch_packet(archive, members, manifest, data, packet_bytes=len(blob))
        files, expected = _checked_evidence(archive, members, manifest)
        sources = [_source_member(archive, members, source, catalog, expected) for source in manifest["sources"]]
        roles = [source["role"] for source in sources]
        if roles not in (["primary"], ["primary", "repeat-pass"]):
            raise _packet_error("The packet must identify one primary original and at most one repeat-pass original.")
        support = _support_sources(manifest, catalog)
        extra = sorted(set(members) - expected)
    missing = [item["name"] for item in files if item["status"] == "missing"]
    changed = [item["name"] for item in files if item["status"] not in {"verified", "missing"}]
    changed.extend(source["role"] for source in sources if source["packet"] in {"changed", "unreadable"})
    status = "review" if missing or changed or extra or any(source["local"] == "different" for source in sources + support) else "verified"
    return {
        "status": status,
        "record": manifest["record"],
        "sources": sources,
        "support_sources": support,
        "files": files,
        "missing_files": missing,
        "changed_files": changed,
        "unexpected_files": extra,
        "manifest_sha256": payload_hash(data),
        "packet_sha256": payload_hash(blob),
    }
