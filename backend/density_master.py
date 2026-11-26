"""Serialize calibrated optical density at 16-bit precision with explicit missing values."""

from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from .evidence import json_document
from .values import DeskError, PRIMARY_CHANNELS, boolean, mapping, number, payload_hash


DENSITY_MASTER_LIMIT = 3_000_000
MASTER_SIGNATURE = "reeldesk-optical-density-v1"
MASK_FLAGS = {
    "unavailable": 1,
    "low_sensor_clip": 2,
    "high_sensor_clip": 4,
    "below_export_range": 8,
    "above_export_range": 16,
}


@dataclass(frozen=True)
class DensityMasterOptions:
    minimum: float
    maximum: float
    clip_outside: bool

    @classmethod
    def read(cls, value):
        value = {} if value is None else mapping(value, "density_export")
        minimum = number(value.get("minimum", -.12), "density_export.minimum", lower=-1.5, upper=2.3)
        maximum = number(value.get("maximum", 2.35), "density_export.maximum", lower=minimum + .005, upper=3.9)
        return cls(minimum, maximum, boolean(value.get("clip_outside", False), "density_export.clip_outside"))

    @property
    def scale(self):
        return (self.maximum - self.minimum) / 65534

    def as_dict(self):
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "clip_outside": self.clip_outside,
        }


def _encoded_image(pixels, dpi=None):
    output = BytesIO()
    img = Image.fromarray(pixels)
    options = {"compress_level": 5}
    if dpi:
        options["dpi"] = (dpi, dpi)
    img.save(output, format="PNG", **options)
    return output.getvalue()


def _encode_channel(density, valid, low_clip, high_clip, options):
    # A zero value means no reading. A genuine reading at the lower limit starts at one.
    below = valid & (density < options.minimum)
    above = valid & (density > options.maximum)
    final_valid = valid.copy()
    if not options.clip_outside:
        final_valid &= ~(below | above)
    source = np.where(final_valid, density, options.minimum)
    codes = np.rint(np.clip((source - options.minimum) / options.scale, 0, 65534)).astype(np.uint16) + 1
    codes[~final_valid] = 0
    quality = np.zeros(valid.shape, dtype=np.uint8)
    for field, mask in (
        ("unavailable", ~valid),
        ("low_sensor_clip", low_clip),
        ("high_sensor_clip", high_clip),
        ("below_export_range", below),
        ("above_export_range", above),
    ):
        quality[mask] |= MASK_FLAGS[field]
    counts = {
        "samples": int(final_valid.sum()),
        "unavailable": int((~valid).sum()),
        "omitted_by_range": int((valid & ~final_valid).sum()),
        "below": int(below.sum()),
        "above": int(above.sum()),
        "sensor_low": int(low_clip.sum()),
        "sensor_high": int(high_clip.sum()),
    }
    return codes, quality, counts


def _source_snapshot(source):
    return {
        "scan_id": source["id"],
        "name": source["name"],
        "dimensions": [source["width"], source["height"]],
        "dpi": source["dpi"],
        "sha256": source["checksum"],
    }


def _file_record(archive, name, content, mode):
    archive.writestr(name, content)
    return {"filename": name, "mode": mode, "bytes": len(content), "sha256": payload_hash(content)}


def make_density_master(calibrated, region, source, calibration, control_list, options):
    if region.area > DENSITY_MASTER_LIMIT:
        raise DeskError("Choose three million pixels or less for a 16-bit density master.", field="region")
    density = -np.log10(np.maximum(calibrated.values[region.slices], 1 / 4096))
    valid = calibrated.valid[region.slices]
    low_clip = calibrated.low_clip[region.slices]
    high_clip = calibrated.high_clip[region.slices]
    metadata = _source_snapshot(source)
    buffer = BytesIO()
    channels = {}
    dpi = source["dpi"]
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for index, channel in enumerate(PRIMARY_CHANNELS):
            codes, quality, counts = _encode_channel(
                density[..., index],
                valid[..., index],
                low_clip[..., index],
                high_clip[..., index],
                options,
            )
            name = channel.lower()
            raster = _encoded_image(codes, dpi)
            flags = _encoded_image(quality, dpi)
            channels[channel] = {
                "density": _file_record(archive, f"density-{name}.png", raster, "uint16"),
                "quality": _file_record(archive, f"quality-{name}.png", flags, "uint8-bitmask"),
                "counts": counts,
            }
        manifest = {
            "schema": MASTER_SIGNATURE,
            "source": metadata,
            "region": region.as_dict(),
            "calibration": calibration,
            "controls": [_source_snapshot(control) for control in control_list],
            "range": options.as_dict(),
            "scale": options.scale,
            "offset": options.minimum,
            "unavailable_value": 0,
            "density_first_value": 1,
            "units": "optical density",
            "quality_flags": MASK_FLAGS,
            "channels": channels,
        }
        archive.writestr("density.json", json_document(manifest))
    result = {
        "source": metadata, "region": region.as_dict(),
        "minimum": options.minimum, "maximum": options.maximum, "scale": options.scale,
        "outside_range": "clipped" if options.clip_outside else "omitted",
        "channels": {channel: value["counts"] for channel, value in channels.items()},
    }
    return buffer.getvalue(), result
