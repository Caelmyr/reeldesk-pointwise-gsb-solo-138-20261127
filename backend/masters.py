"""Export frame masters from immutable originals in native raster coordinates."""

from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import ImageOps

from .evidence import json_document, slug
from .imaging import GRAY16, _open_original
from .values import Box, DeskError, boolean, checked_key, integer, mapping, one_of, payload_hash, sequence


MASTER_FORMATS = frozenset({"png", "tiff"})
MASTER_LIMIT = 45 * 1024 * 1024
MASTER_PIXELS = 16_000_000
FRAME_COUNT = 18


@dataclass(frozen=True)
class MasterRequest:
    selections: tuple
    padding: int
    output: str
    use_alpha: bool
    exact_padding: bool

    @classmethod
    def read(cls, data):
        value = {} if data is None else mapping(data, "masters")
        raw_marks = value.get("marks")
        chosen = ()
        if raw_marks is not None:
            raw_marks = sequence(raw_marks, "masters.marks", minimum=1, maximum=FRAME_COUNT)
            chosen = tuple(checked_key(identifier, "masters.marks") for identifier in raw_marks)
            if len(set(chosen)) != len(chosen):
                raise DeskError("Choose each frame once.", field="masters.marks")
        return cls(
            chosen,
            integer(value.get("padding", 0), "masters.padding", lower=0, upper=250),
            one_of(value.get("format", "tiff"), MASTER_FORMATS, "masters.format"),
            boolean(value.get("use_alpha", True), "masters.use_alpha"),
            boolean(value.get("exact_padding", False), "masters.exact_padding"),
        )

    def as_dict(self):
        return {
            "marks": list(self.selections), "padding": self.padding, "format": self.output,
            "use_alpha": self.use_alpha, "exact_padding": self.exact_padding,
        }


@dataclass(frozen=True)
class Crop:
    frame_id: str | None
    label: str
    base: Box
    padded: Box
    clipped: bool

    def as_dict(self):
        return {
            "frame_id": self.frame_id, "label": self.label, "source_region": self.base.as_dict(),
            "export_region": self.padded.as_dict(), "padding_truncated": self.clipped,
        }


def _prepare_crop(box, name, identifier, settings, size):
    expanded = box.expanded(settings.padding, *size)
    clipped = expanded.w != box.w + 2 * settings.padding or expanded.h != box.h + 2 * settings.padding
    if clipped and settings.exact_padding:
        raise DeskError(
            f"{name} has insufficient film at the scan boundary for the requested padding.",
            field="masters.padding",
        )
    return Crop(identifier, name, box, expanded, clipped)


def crop_list(settings, size, marks, region=None):
    if settings.selections and region:
        raise DeskError("Choose marked frames or one region for a master.", field="masters.marks")
    if region:
        return [_prepare_crop(region, "Selected region", None, settings, size)]
    frames = {mark["id"]: mark for mark in marks if mark["kind"] == "frame"}
    if settings.selections:
        if missing := [identifier for identifier in settings.selections if identifier not in frames]:
            raise DeskError("Select the current frame marks on this scan.", field="masters.marks", details={"missing": missing})
        frame_rows = [frames[identifier] for identifier in settings.selections]
    else:
        frame_rows = sorted(
            frames.values(),
            key=lambda mark: (mark["geometry"]["x"], mark["geometry"]["y"], mark["id"]),
        )
    if not frame_rows:
        raise DeskError("Mark a frame or select the scan region to export.", field="masters.marks")
    if len(frame_rows) > FRAME_COUNT:
        raise DeskError("Select up to 18 frame masters at once.", field="masters.marks")
    result = []
    for value in frame_rows:
        box = Box.read(value["geometry"], bounds=size)
        result.append(_prepare_crop(box, value["label"], value["id"], settings, size))
    if sum(crop.padded.area for crop in result) > MASTER_PIXELS:
        raise DeskError("Choose fewer frames or a smaller total export area.", field="masters.marks")
    return result


def _without_alpha(image):
    if image.mode in {"RGBA", "LA"}:
        if image.mode == "LA":
            return image.getchannel("L")
        return image.convert("RGB")
    if image.mode == "P" and "transparency" in image.info:
        return image.convert("RGB")
    return image


def _tiff_compatible(image):
    if image.mode in {"RGB", "RGBA", "L", "LA", "1", "I", "F", "CMYK", "P", *GRAY16}:
        return image
    raise DeskError(
        "TIFF master export cannot preserve this original pixel mode.",
        field="masters.format",
    )


def _png_compatible(image):
    if image.mode in {"I", "F", "CMYK"}:
        raise DeskError("Use TIFF to preserve the original scan mode.", field="masters.format")
    if image.mode in {"I;16B", "I;16N"}:
        return image.point(lambda value: value).convert("I;16")
    if image.mode not in {"RGB", "RGBA", "L", "LA", "1", "P", "I;16", "I;16L"}:
        raise DeskError("Choose a supported native color mode for PNG export.", field="masters.format")
    return image


def _encode_crop(image, output, dpi, icc, transparency):
    buffer = BytesIO()
    options = {}
    if dpi:
        options["dpi"] = (dpi, dpi)
    if icc and image.mode in {"RGB", "RGBA", "CMYK", "L", "LA"}:
        options["icc_profile"] = icc
    if output == "tiff":
        image = _tiff_compatible(image)
        if image.mode == "P" and transparency is not None:
            image = image.convert("RGBA")
            options.pop("icc_profile", None)
        image.save(buffer, format="TIFF", compression="tiff_deflate", **options)
        suffix = "tif"
    else:
        image = _png_compatible(image)
        if image.mode == "P" and transparency is not None:
            options["transparency"] = transparency
        image.save(buffer, format="PNG", compress_level=5, **options)
        suffix = "png"
    return suffix, image.mode, buffer.getvalue()


def _contact_labels(crops, source):
    prefix = slug(source["name"], "scan")
    for number, crop in enumerate(crops, 1):
        frame = slug(crop.label, "frame")
        yield f"{prefix}-{number:02d}-{frame}"


def create_masters(original, scan, crops, settings):
    buffer = BytesIO()
    used = 0
    results = []
    with _open_original(original) as native, ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=1) as archive:
        upright = ImageOps.exif_transpose(native)
        if upright.size != (scan["width"], scan["height"]):
            raise DeskError("The original's current orientation differs from its catalog record.", field="scan_id")
        icc = native.info.get("icc_profile")
        transparency = upright.info.get("transparency") if settings.use_alpha else None
        for crop, label in zip(crops, _contact_labels(crops, scan)):
            cropped = upright.crop(crop.padded.pillow)
            if not settings.use_alpha:
                cropped = _without_alpha(cropped)
            mode = cropped.mode
            try:
                suffix, output_mode, content = _encode_crop(cropped, settings.output, scan["dpi"], icc, transparency)
            except (ValueError, OSError) as error:
                raise DeskError("The selected raster mode could not be written in this format.", field="masters.format") from error
            used += len(content)
            if used > MASTER_LIMIT:
                raise DeskError("The generated masters exceed 45 MiB. Export fewer frames.", status=413, field="masters.marks")
            name = f"{label}.{suffix}"
            archive.writestr(name, content)
            results.append(
                {
                    **crop.as_dict(),
                    "file": name, "bytes": len(content),
                    "input_mode": mode, "output_mode": output_mode,
                    "sha256": payload_hash(content),
                }
            )
        manifest = {
            "schema": "reeldesk-masters-1",
            "source": {"id": scan["id"], "name": scan["name"], "sha256": scan["checksum"], "dpi": scan["dpi"]},
            "settings": settings.as_dict(),
            "coordinate_space": "upright scanner source pixels",
            "frames": results,
        }
        archive.writestr("frames.json", json_document(manifest))
    return buffer.getvalue(), {
        "files": results,
        "source": manifest["source"],
        "format": settings.output,
        "total_raster_bytes": used,
    }
