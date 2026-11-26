"""Inspection proofs, contact sheets, and portable review packets."""

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import csv
import math
import re
import unicodedata
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .values import (
    Box, DeskError, PRIMARY_CHANNELS, as_json, boolean, image_bytes, integer, mapping,
    one_of, payload_hash,
)


SHEET_PAPERS = {"a4": (210, 297), "letter": (215.9, 279.4)}
EXHIBIT_LIMIT = 20 * 1024 * 1024
CONTACT_LIMIT = 24
JPEG_QUALITY = 90
PNG = "image/png"

TITLE_COLOR = "#e9dcc4"
MUTED_COLOR = "#ada391"
RULER_BG = "#292525"
RULER_LINE = "#d0b58d"

KIND_TITLES = {
    "density": "Density reading",
    "point": "Source point reading",
    "profile": "Density profile",
    "balance": "Local contrast",
    "calibration": "Illumination check",
    "registration": "Repeat-pass alignment",
    "compare": "Repeat-pass point check",
    "residual": "Repeat-pass difference",
    "dust": "Dust survey",
    "dust-check": "Repeat-pass dust check",
    "restore": "Local dust retouch",
    "cut": "Cut clearance check",
    "cut-verified": "Observed cut",
    "frames": "Frame boxes",
    "perforations": "Perforation survey",
    "grain": "Grain survey",
    "proof": "Scan proof",
    "density-master": "16-bit density master",
    "contact": "Roll contact sheet",
    "wedge": "Transmission wedge response",
    "focus": "Slanted-edge focus",
    "coverage": "Scanner coverage",
    "masters": "Frame masters",
    "carrier": "Carrier rails",
}


def slug(value, default):
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", normalized).strip("-")[:48].lower()
    return cleaned or default


def tagged_name(value, fallback, suffix):
    return f"{slug(value, fallback)}.{suffix}"


def safe_original_name(name, format):
    stem = Path(name.replace("\\", "/")).name.rsplit(".", 1)[0]
    suffix = {"JPEG": "jpg", "PNG": "png", "TIFF": "tif", "WEBP": "webp"}.get(format, "bin")
    return tagged_name(stem, "original", suffix)


def json_document(value):
    return (as_json(value) + "\n").encode("utf-8")


def csv_document(header, rows):
    file = StringIO(newline="")
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow("" if item is None else item for item in row)
    return file.getvalue().encode("utf-8-sig")


def result_rows(kind, result):
    if kind == "profile":
        distances = result["distance_px"]
        distances_mm = result.get("distance_mm") or [None] * len(distances)
        return (
            ["station", "distance_px", "distance_mm", "density_r", "density_g", "density_b", "samples_r", "samples_g", "samples_b"],
            [
                [index, distance, distances_mm[index]]
                + [result["channels"][channel]["density"][index] for channel in PRIMARY_CHANNELS]
                + [result["channels"][channel]["support"][index] for channel in PRIMARY_CHANNELS]
                for index, distance in enumerate(distances)
            ],
        )
    if kind == "calibration":
        keys = ["row", "column", "samples", "flat_relative_percent", "density_drift"]
        return keys, [[cell[key] for key in keys] for cell in result["tiles"]]
    if kind == "dust":
        keys = [
            "number", "pixels", "polarity", "severity", "roundness",
            "protected_pixels", "edge_contact", "can_restore",
        ]
        return keys, [[finding[key] for key in keys] for finding in result["findings"]]
    if kind == "density":
        stats = ["median", "p05", "p95", "mean", "mad", "valid", "valid_fraction", "clips_low", "clips_high"]
        return ["channel", *stats], [[channel, *(result["channels"][channel][key] for key in stats)] for channel in PRIMARY_CHANNELS]
    if kind in {"residual", "compare"}:
        fields = list(result["channels"]["R"])
        return ["channel", *fields], [[channel, *(result["channels"][channel][field] for field in fields)] for channel in PRIMARY_CHANNELS]
    if kind in {"cut", "cut-verified"}:
        value = result.get("observed", result)
        keys = ["mark_id", "label", "kind", "side", "clearance_mm", "crosses", "within_reserve"]
        return keys, [[mark[key] for key in keys] for mark in value["conflicts"] + value["clear_marks"]]
    if kind == "grain":
        keys = ["radius", "samples", "mad", "p95"]
        return keys, [[band[key] for key in keys] for band in result["bands"]]
    if kind == "coverage":
        columns = ["row", "column", "eligible_fraction", "status", "dominant_gap"]
        return columns, [[row[key] for key in columns] for row in result["tiles"]]
    if kind == "wedge":
        keys = ["median", "fraction", "error_to_nominal", "residual_to_fit", "status"]
        rows = [
            [step["label"], step["nominal"], channel, *(step["measurements"][channel][key] for key in keys)]
            for step in result["steps"] for channel in PRIMARY_CHANNELS
        ]
        return ["step", "nominal", "channel", *keys], rows
    if kind == "carrier":
        return (
            ["position_px", "width_mm", "edge_strength", "status"],
            [[station["source_station"], station["width_mm"], station["edge_strength"], station["status"]]
             for station in result["stations"]],
        )
    return None


def record_csv(kind, result):
    table = result_rows(kind, result)
    return csv_document(*table) if table else None


def _font(size):
    return ImageFont.load_default(size=max(9, round(size)))


def _display_text(drawer, text, font, maximum):
    if drawer.textlength(text, font=font) <= maximum:
        return text
    clean = text
    while clean and drawer.textlength(clean + "…", font=font) > maximum:
        clean = clean[:-1]
    return clean + "…"


def _draw_caption(drawer, xy, text, font, maximum, *, fill=TITLE_COLOR):
    drawer.text(xy, _display_text(drawer, text, font, maximum), font=font, fill=fill)


@dataclass(frozen=True)
class ProofSettings:
    max_side: int
    annotate: bool
    scale_bar: bool
    include_histogram: bool

    @classmethod
    def read(cls, raw):
        value = {} if raw is None else mapping(raw, "proof")
        return cls(
            integer(value.get("max_side", 1800), "proof.max_side", lower=480, upper=3000),
            boolean(value.get("annotate", True), "proof.annotate"),
            boolean(value.get("scale_bar", True), "proof.scale_bar"),
            boolean(value.get("include_histogram", False), "proof.include_histogram"),
        )

    def as_dict(self):
        return {
            "max_side": self.max_side, "annotate": self.annotate, "scale_bar": self.scale_bar,
            "include_histogram": self.include_histogram,
        }


def _proof_scale(drawer, origin, image_width, original_width, dpi):
    if not dpi:
        return None
    pixels_per_mm = dpi / 25.4 * image_width / original_width
    maximum = image_width / 4 / pixels_per_mm
    reachable = [length for length in (.5, 1, 2, 5, 10, 20, 50) if length <= maximum]
    if not reachable:
        return None
    millimeters = reachable[-1]
    length = round(millimeters * pixels_per_mm)
    start_x, y = origin
    drawer.line((start_x, y, start_x + length, y), fill=RULER_LINE, width=3)
    for x in (start_x, start_x + length):
        drawer.line((x, y - 3, x, y + 4), fill=RULER_LINE, width=2)
    drawer.text((start_x + length + 9, y - 8), f"{millimeters:g} mm", fill=TITLE_COLOR, font=_font(12))
    return {"length_mm": millimeters, "proof_pixels": length}


def _proof_marks(drawer, marks, region, left, top, scale_x, scale_y):
    count = 0
    for mark in marks:
        if mark["kind"] not in {"frame", "keep", "reference"}:
            continue
        box = Box.read(mark["geometry"]).intersect(region)
        if box is None:
            continue
        x = int(left + (box.x - region.x) * scale_x)
        y = int(top + (box.y - region.y) * scale_y)
        right = int(left + (box.right - region.x) * scale_x) - 1
        bottom = int(top + (box.bottom - region.y) * scale_y) - 1
        drawer.rectangle((x, y, right, bottom), outline="#f7d59c", width=2)
        if right - x >= 36 and bottom - y >= 16:
            _draw_caption(drawer, (x + 3, y + 2), mark["label"], _font(11), right - x - 6, fill="#fff2d7")
        count += 1
    return count


def _draw_histogram(drawer, table, left, top, width, height):
    values = [table["channels"][channel]["counts"] for channel in PRIMARY_CHANNELS]
    max_count = max((count for channel in values for count in channel), default=1)
    if max_count == 0:
        return
    drawer.rectangle((left, top, left + width, top + height), fill="#201d1c", outline="#524a43")
    step = width / max(1, len(values[0]) - 1)
    for channel, color in zip(values, ("#e87368", "#a5c77b", "#78a8d2")):
        points = [
            (left + round(index * step), top + height - round(height * math.sqrt(count / max_count)))
            for index, count in enumerate(channel)
        ]
        drawer.line(points, fill=color, width=2)


def scan_proof(image, scan, roll, region, view, settings, *, marks=(), histogram=None):
    width, height = image.size
    image = image.copy()
    image.thumbnail((settings.max_side, settings.max_side), Image.Resampling.LANCZOS)
    pad = 22
    header = 64
    panel = 82 if settings.include_histogram and histogram is not None else 0
    footer = 59 + panel
    canvas = Image.new("RGB", (max(420, image.width + 2 * pad), image.height + header + footer), "#292525")
    left = (canvas.width - image.width) // 2
    canvas.paste(image, (left, header))
    drawer = ImageDraw.Draw(canvas)
    _draw_caption(drawer, (pad, 10), scan["name"], _font(22), canvas.width - 2 * pad)
    title = f"{roll['code']}  ·  {roll['title']}  ·  {view}"
    _draw_caption(drawer, (pad, 39), title, _font(12), canvas.width - 2 * pad, fill=MUTED_COLOR)
    y = header + image.height
    drawer.line((pad, y + 10, canvas.width - pad, y + 10), fill="#55483e")
    source_text = f"Source x {region.x}, y {region.y}  ·  {region.w} × {region.h} px  ·  {scan['checksum'][:12]}"
    _draw_caption(drawer, (pad, y + 18), source_text, _font(11), canvas.width - 2 * pad, fill=MUTED_COLOR)
    count = _proof_marks(drawer, marks, region, left, header, image.width / region.w, image.height / region.h) if settings.annotate else 0
    bar = _proof_scale(drawer, (pad, y + 42), image.width, region.w, scan["dpi"]) if settings.scale_bar else None
    if panel:
        drawer.text((pad, y + 61), "Density distribution", font=_font(11), fill=MUTED_COLOR)
        _draw_histogram(drawer, histogram, pad, y + 77, canvas.width - 2 * pad, 43)
    return canvas, {
        "width": canvas.width,
        "height": canvas.height,
        "type": view,
        "region": region.as_dict(),
        "mark_count": count,
        "scale": bar,
        "density_histogram": bool(panel),
        "source_pixels_per_preview_pixel": round(width / image.width, 4),
    }


@dataclass(frozen=True)
class SheetSettings:
    paper: str
    orientation: str
    columns: int
    dpi: int
    show_codes: bool

    @classmethod
    def read(cls, raw):
        value = {} if raw is None else mapping(raw, "sheet")
        return cls(
            one_of(value.get("paper", "a4"), SHEET_PAPERS, "sheet.paper"),
            one_of(value.get("orientation", "landscape"), {"portrait", "landscape"}, "sheet.orientation"),
            integer(value.get("columns", 3), "sheet.columns", lower=2, upper=5),
            integer(value.get("dpi", 125), "sheet.dpi", lower=72, upper=200),
            boolean(value.get("show_codes", True), "sheet.show_codes"),
        )

    @property
    def dimensions(self):
        millimeters = SHEET_PAPERS[self.paper]
        if self.orientation == "landscape":
            millimeters = millimeters[::-1]
        return tuple(round(length / 25.4 * self.dpi) for length in millimeters)

    def as_dict(self):
        return {
            "paper": self.paper, "orientation": self.orientation, "columns": self.columns,
            "dpi": self.dpi, "show_codes": self.show_codes,
        }


def _sheet_grid(settings, count):
    width, height = settings.dimensions
    columns = settings.columns
    margin = max(18, round(.045 * width))
    gutter = max(8, round(.016 * width))
    header = round(.108 * height)
    footer = round(.045 * height)
    inner_width = width - 2 * margin
    inner_height = height - header - footer
    cell_width = (inner_width - (columns - 1) * gutter) // columns
    possible_rows = max(1, (inner_height + gutter) // (round(cell_width * .59) + gutter))
    rows = min(max(1, math.ceil(count / columns)), possible_rows)
    cell_height = (inner_height - (rows - 1) * gutter) // rows
    return margin, gutter, header, footer, cell_width, cell_height, rows, columns


def contact_pages(items, roll, settings):
    if not items:
        raise DeskError("Add at least one scan before making a contact sheet.", field="scans")
    if len(items) > CONTACT_LIMIT:
        raise DeskError(f"Choose at most {CONTACT_LIMIT} images for one contact sheet.", field="scans")
    width, height = settings.dimensions
    margin, gap, header, footer, cell_width, cell_height, rows, columns = _sheet_grid(settings, len(items))
    capacity = rows * columns
    page_count = math.ceil(len(items) / capacity)
    pages = []
    for page_number in range(page_count):
        start = page_number * capacity
        canvas = Image.new("RGB", (width, height), "#f4eee2")
        drawer = ImageDraw.Draw(canvas)
        text_width = width - 2 * margin
        _draw_caption(drawer, (margin, round(header * .18)), roll["title"], _font(header * .34), text_width, fill="#32302b")
        second_line = f"{roll['code']}  ·  {roll['stock']}  ·  {len(items)} scans"
        _draw_caption(drawer, (margin, round(header * .58)), second_line, _font(header * .16), text_width, fill="#645b51")
        drawer.line((margin, header - 8, width - margin, header - 8), fill="#b6a587", width=1)
        for index, (scan, preview) in enumerate(items[start:start + capacity]):
            row, column = divmod(index, columns)
            x = margin + column * (cell_width + gap)
            y = header + row * (cell_height + gap)
            caption_height = min(42, max(20, round(cell_height * .14)))
            cell_box = x, y, x + cell_width, y + cell_height - caption_height
            drawer.rectangle(cell_box, fill="#312b2a")
            preview = ImageOps.contain(preview, (cell_width, cell_height - caption_height), Image.Resampling.LANCZOS)
            photo_x = x + (cell_width - preview.width) // 2
            photo_y = y + (cell_height - caption_height - preview.height) // 2
            canvas.paste(preview, (photo_x, photo_y))
            marker = f"{start + index + 1:02d}  {scan['name']}"
            if settings.show_codes:
                marker += f"  ·  {scan['checksum'][:7]}"
            _draw_caption(drawer, (x, cell_box[3] + 3), marker, _font(caption_height * .53), cell_width, fill="#40392d")
        page_label = f"{page_number + 1} / {page_count}"
        drawer.text((width - margin - drawer.textlength(page_label, font=_font(footer * .36)), height - footer + 4),
                    page_label, font=_font(footer * .36), fill="#665c4c")
        pages.append(canvas)
    return pages


def contact_pdf(pages, settings):
    buffer = BytesIO()
    pages[0].save(buffer, format="PDF", resolution=settings.dpi, save_all=True, append_images=pages[1:])
    return buffer.getvalue()


def record_manifest(record, scan, peer, files):
    sources = list(record["settings"].get("source_inventory", ()))
    for role, item in (("primary", scan), ("repeat-pass", peer)):
        if item is None or record["settings"].get("source_inventory"):
            continue
        sources.append(
            {
                "role": role, "scan_id": item["id"], "name": item["name"], "sha256": item["checksum"],
                "width": item["width"], "height": item["height"], "dpi": item["dpi"],
                "orientation": item["orientation"], "synthetic_demo": item["sample"],
            }
        )
    return {
        "format": "reeldesk-review-v1",
        "record": {field: record[field] for field in ("id", "kind", "title", "decision", "created", "revision")},
        "sources": sources,
        "support_sources": record["settings"].get("support_inventory", []),
        "coordinates": "Coordinates use source pixels after applying image orientation; rectangles are half-open.",
        "settings": record["settings"],
        "result": record["result"],
        "files": [
            {
                "name": file["name"], "content_type": file["content_type"], "bytes": file["bytes"],
                "sha256": file["checksum"],
            }
            for file in files
        ],
    }


def packet_zip(manifest, files, originals=()):
    original_total = sum(len(content) for _, content in originals)
    if original_total > 72 * 1024 * 1024:
        raise DeskError("Download the original scans individually for this record.", status=413, field="originals")
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=5) as archive:
        archive.writestr("manifest.json", json_document(manifest))
        table = record_csv(manifest["record"]["kind"], manifest["result"])
        if table is not None:
            archive.writestr("reading.csv", table)
        used = set()
        for file in files:
            name = Path(file["name"]).name
            if name in used:
                raise DeskError("The record has duplicate file names.", status=500, code="invalid_record")
            if payload_hash(file["content"]) != file["checksum"]:
                raise DeskError("An evidence file failed its checksum.", status=500, code="corrupt_evidence")
            used.add(name)
            archive.writestr(f"evidence/{name}", file["content"])
        for label, content in originals:
            archive.writestr(f"originals/{Path(label).name}", content)
    return buffer.getvalue()


def attachment(image, name, *, lossless=True):
    if lossless:
        data = image_bytes(image)
        mime = "image/png"
        ext = "png"
    else:
        data = image_bytes(image.convert("RGB"), format="JPEG", quality=JPEG_QUALITY)
        mime = "image/jpeg"
        ext = "jpg"
    if len(data) > EXHIBIT_LIMIT:
        raise DeskError("Reduce the proof dimensions before saving evidence.", field="proof.max_side")
    return tagged_name(name, "review", ext), mime, data
