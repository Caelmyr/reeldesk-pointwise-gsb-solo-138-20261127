"""Local web service for the ReelDesk workbench."""

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from threading import BoundedSemaphore

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from .batch import BatchStore, build_batch_package, list_density_readings
from .catalog import Catalog
from .evidence import KIND_TITLES, packet_zip, record_csv, record_manifest, safe_original_name, tagged_name
from .operations import Operations
from .review import PACKET_UPLOAD_LIMIT, verify_packet
from .values import (
    DeskError, IMAGE_ROLES, MARK_KINDS, UPLOAD_LIMIT, VERDICTS, clean_text, integer,
    mapping, one_of, optional_key, page_bounds, parse_json, payload_hash, read_revision, tags,
)
from .workspace import Workspace


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = APP_ROOT / "data"
STATIC_ROOT = APP_ROOT / "static"


@dataclass(frozen=True)
class Services:
    catalog: Catalog
    workspace: Workspace
    operations: Operations
    previews: BoundedSemaphore
    batches: BatchStore


def _body():
    try:
        value = request.get_json()
    except BadRequest as error:
        raise DeskError("Send a readable JSON object.", field="body", status=400) from error
    return mapping(value)


def _optional_choice(data, key, options):
    value = data.get(key, "")
    return one_of(value, options, key) if value != "" else ""


def _flag(arguments, key, default=False):
    value = arguments.get(key)
    if value is None:
        return default
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise DeskError("Use true or false for this option.", field=key)


def _binary(content, mime, *, filename=None, cache=0, etag=None):
    result = send_file(
        BytesIO(content),
        mimetype=mime,
        as_attachment=filename is not None,
        download_name=filename,
        max_age=cache,
        conditional=etag is not None,
        etag=etag if etag is not None else False,
    )
    return result


def _json_upload(field, maximum):
    raw = request.form.get(field)
    if raw is None or raw == "":
        return {}
    if len(raw) > maximum:
        raise DeskError(f"The {field} is too long.", field=field)
    try:
        value = parse_json(raw)
    except (TypeError, ValueError) as error:
        raise DeskError(f"The {field} is not readable JSON.", field=field) from error
    return mapping(value, field)


def _file(field, maximum):
    maximum_request = maximum + 32 * 1024
    if (request.content_length or 0) > maximum_request:
        raise DeskError("The selected file exceeds the upload limit.", field=field, status=413)
    incoming = request.files.get(field)
    if not incoming or not incoming.filename:
        raise DeskError("Choose a file to upload.", field=field)
    return incoming.filename, incoming.stream.read(maximum + 1)


def _display_original_mime(source_format):
    return {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "TIFF": "image/tiff",
        "WEBP": "image/webp",
    }.get(source_format, "application/octet-stream")


def create_app(storage=None, *, seed_demo=None):
    app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="/static")
    app.config.update(
        MAX_CONTENT_LENGTH=PACKET_UPLOAD_LIMIT + 32 * 1024,
        MAX_FORM_MEMORY_SIZE=384 * 1024,
        MAX_FORM_PARTS=20,
    )
    catalog = Catalog(storage or Path(os.environ.get("REELDESK_DATA", str(DEFAULT_DATA))))
    workspace = Workspace(catalog)
    services = Services(catalog, workspace, Operations(workspace), BoundedSemaphore(2), BatchStore())
    if seed_demo is None:
        seed_demo = os.environ.get("REELDESK_DEMO", "1") == "1"
    if seed_demo:
        workspace.start_demo()

    @app.errorhandler(DeskError)
    def desk_error(error):
        return jsonify(error.payload()), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify({"error": "upload_too_large", "message": "The upload exceeds the size limit."}), 413

    @app.errorhandler(404)
    def unknown(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not_found", "message": "This workbench endpoint does not exist."}), 404
        return error

    @app.after_request
    def protect(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if request.path.startswith("/api/") and "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return send_from_directory(STATIC_ROOT, "index.html", max_age=0)

    @app.get("/health")
    def health():
        return jsonify(catalog.health())

    register_catalog(app, services)
    register_imaging(app, services)
    register_review(app, services)
    return app


def register_catalog(app, services):
    catalog = services.catalog
    workspace = services.workspace

    @app.get("/api/status")
    def status():
        return jsonify({"product": "ReelDesk", "catalog": catalog.summary(), "operations": KIND_TITLES})

    @app.post("/api/demo")
    def install_demo():
        return jsonify(workspace.start_demo())

    @app.get("/api/rolls")
    def list_rolls():
        search = clean_text(request.args.get("q"), "q", maximum=100)
        tag = clean_text(request.args.get("tag"), "tag", maximum=40)
        if tag:
            tag = tags([tag])[0]
        limit, offset = page_bounds(request.args)
        data = catalog.list_rolls(query=search, tag=tag, limit=limit, offset=offset)
        return jsonify({"items": data, "limit": limit, "offset": offset})

    @app.post("/api/rolls")
    def add_roll():
        return jsonify(workspace.create_roll(_body())), 201

    @app.get("/api/rolls/<roll_id>")
    def roll_details(roll_id):
        return jsonify(catalog.get_roll(roll_id))

    @app.patch("/api/rolls/<roll_id>")
    def edit_roll(roll_id):
        return jsonify(workspace.update_roll(roll_id, _body()))

    @app.get("/api/rolls/<roll_id>/scans")
    def list_scans(roll_id):
        role = _optional_choice(request.args, "role", IMAGE_ROLES)
        decision = _optional_choice(request.args, "verdict", VERDICTS)
        limit, offset = page_bounds(request.args, limit=100, maximum=200)
        result = catalog.list_scans(roll_id, role=role, verdict=decision, limit=limit, offset=offset)
        return jsonify({"items": result, "offset": offset, "limit": limit})

    @app.post("/api/rolls/<roll_id>/scans")
    def import_scan(roll_id):
        name, content = _file("file", UPLOAD_LIMIT)
        data = _json_upload("metadata", 12_000)
        return jsonify(workspace.import_scan(roll_id, data, content, name)), 201

    @app.get("/api/scans/<scan_id>")
    def scan_details(scan_id):
        return jsonify(catalog.get_scan(scan_id))

    @app.patch("/api/scans/<scan_id>")
    def edit_scan(scan_id):
        return jsonify(workspace.update_scan(scan_id, _body()))

    @app.delete("/api/scans/<scan_id>")
    def remove_scan(scan_id):
        revision = read_revision(request.args.get("revision"))
        catalog.delete_scan(scan_id, revision)
        return "", 204

    @app.get("/api/scans/<scan_id>/original")
    def scan_download(scan_id):
        source, data = catalog.original(scan_id)
        name = safe_original_name(source["original_name"], source["source_format"])
        return _binary(data, _display_original_mime(source["source_format"]), filename=name, etag=source["checksum"])

    @app.get("/api/scans/<scan_id>/marks")
    def marks(scan_id):
        kind = _optional_choice(request.args, "kind", MARK_KINDS)
        return jsonify({"items": catalog.list_marks(scan_id, kind)})

    @app.post("/api/scans/<scan_id>/marks")
    def add_mark(scan_id):
        return jsonify(workspace.add_mark(scan_id, _body())), 201

    @app.patch("/api/marks/<mark_id>")
    def edit_mark(mark_id):
        return jsonify(workspace.update_mark(mark_id, _body()))

    @app.delete("/api/marks/<mark_id>")
    def remove_mark(mark_id):
        catalog.delete_mark(mark_id, read_revision(request.args.get("revision")))
        return "", 204

    @app.get("/api/rolls/<roll_id>/activity")
    def roll_activity(roll_id):
        limit, _ = page_bounds(request.args, limit=30)
        value = request.args.get("before")
        before = integer(value, "before", lower=1) if value is not None else None
        return jsonify({"items": catalog.activity(roll_id, limit=limit, before=before)})


def register_imaging(app, services):
    workspace = services.workspace
    operations = services.operations

    def guarded_preview(scan_id, settings):
        if not services.previews.acquire(timeout=4):
            raise DeskError("Two previews are already in progress. Try again shortly.", status=429, code="busy")
        try:
            source = workspace.get_image(scan_id)
            args = workspace.settings_for_preview(source, settings)
            return workspace.preview(source, *args)
        finally:
            services.previews.release()

    @app.get("/api/scans/<scan_id>/thumbnail")
    def scan_thumbnail(scan_id):
        maximum = integer(request.args.get("side", 280), "side", lower=80, upper=520)
        view = request.args.get("view", "positive")
        result = guarded_preview(scan_id, {"view": view, "max_side": maximum, "calibration": {"reject_clipped": False}})
        return _binary(result, "image/png", cache=120, etag=payload_hash(result))

    @app.post("/api/scans/<scan_id>/preview")
    def scan_preview(scan_id):
        return _binary(guarded_preview(scan_id, _body()), "image/png")

    @app.post("/api/scans/<scan_id>/base")
    def measure_base(scan_id):
        return jsonify(workspace.estimate_base(workspace.get_image(scan_id), _body()))

    @app.post("/api/scans/<scan_id>/analysis")
    def analyze(scan_id):
        return jsonify(operations.run(scan_id, _body()).summary()), 201

    @app.get("/api/jobs/<job_id>")
    def job_summary(job_id):
        return jsonify(operations.jobs.get(job_id).summary())

    @app.get("/api/jobs/<job_id>/files/<int:index>")
    def job_file(job_id, index):
        name, mime, data = operations.jobs.exhibit(job_id, index)
        return _binary(data, mime, filename=name if _flag(request.args, "download") else None)

    @app.post("/api/jobs/<job_id>/save")
    def save_job(job_id):
        record, created = operations.jobs.save(job_id, _body().get("title"))
        return jsonify(record), 201 if created else 200


def register_review(app, services):
    catalog = services.catalog

    @app.get("/api/rolls/<roll_id>/records")
    def records(roll_id):
        kind = _optional_choice(request.args, "kind", KIND_TITLES)
        scan_id = optional_key(request.args.get("scan_id"), "scan_id")
        limit, offset = page_bounds(request.args)
        return jsonify({"items": catalog.list_records(roll_id, scan_id=scan_id, kind=kind, limit=limit, offset=offset)})

    @app.get("/api/records/<record_id>")
    def record_details(record_id):
        return jsonify(catalog.get_record(record_id))

    @app.patch("/api/records/<record_id>")
    def edit_record(record_id):
        data = _body()
        changes = {}
        if "title" in data:
            changes["title"] = clean_text(data["title"], "title", required=True)
        if "decision" in data:
            changes["decision"] = one_of(data["decision"], VERDICTS, "decision")
        return jsonify(catalog.update_record(record_id, changes, read_revision(data.get("revision"))))

    @app.delete("/api/records/<record_id>")
    def remove_record(record_id):
        catalog.delete_record(record_id, read_revision(request.args.get("revision")))
        return "", 204

    @app.get("/api/records/<record_id>/files/<file_id>")
    def evidence_file(record_id, file_id):
        data = catalog.evidence_file(record_id, file_id)
        show = not _flag(request.args, "download")
        return _binary(data["content"], data["content_type"], filename=None if show else data["name"], etag=data["checksum"])

    @app.get("/api/records/<record_id>/csv")
    def export_csv(record_id):
        result = catalog.get_record(record_id)
        csv = record_csv(result["kind"], result["result"])
        if csv is None:
            raise DeskError("This record does not have a tabular export.", status=404, code="not_found")
        return _binary(csv, "text/csv", filename=tagged_name(result["title"], "reading", "csv"))

    @app.get("/api/records/<record_id>/packet")
    def export_packet(record_id):
        include = _flag(request.args, "originals")
        record, source, peer, files, raw_sources = catalog.read_review_packet(record_id, include)
        manifest = record_manifest(record, source, peer, files)
        originals = [
            (
                f"{label}-{safe_original_name(info['original_name'], info['source_format'])}",
                original,
            )
            for label, info, original in raw_sources
        ]
        zipped = packet_zip(manifest, files, originals)
        return _binary(zipped, "application/zip", filename=tagged_name(record["title"], "review-packet", "zip"))

    @app.post("/api/packets/verify")
    def check_packet():
        _, content = _file("packet", PACKET_UPLOAD_LIMIT)
        local = _flag(request.args, "local", True)
        return jsonify(verify_packet(content, catalog if local else None))

    @app.get("/api/rolls/<roll_id>/batch-readings")
    def batch_readings(roll_id):
        items, truncated = list_density_readings(catalog, roll_id)
        return jsonify({"items": items, "truncated": truncated, "limit": 200})

    @app.post("/api/rolls/<roll_id>/batch-package")
    def plan_batch_package(roll_id):
        data, report = build_batch_package(catalog, roll_id, _body())
        name = tagged_name(f"{catalog.get_roll(roll_id)['code']}-recheck", "recheck", "zip")
        token = services.batches.put(data, report, name)
        return jsonify({**report, "token": token, "name": name}), 201

    @app.get("/api/batch-packages/<token>/download")
    def download_batch_package(token):
        data, name = services.batches.get(token)
        return _binary(data, "application/zip", filename=name)


def main():
    app = create_app()
    host = os.environ.get("REELDESK_HOST", "127.0.0.1")
    port = integer(os.environ.get("REELDESK_PORT", "8080"), "port", lower=1024, upper=65535)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
