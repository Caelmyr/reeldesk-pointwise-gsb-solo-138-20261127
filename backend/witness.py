"""Compatibility spelling for witness-based illumination helpers."""

from .witnesses import *  # noqa: F401,F403
from .witnesses import (  # noqa: F401
    compensation_maps,
    grid_regions,
    observations_from_marks,
    parse_cell,
    provenance_compare,
    witness_plan,
)

recommend_compensation = witness_plan
build_witness_plan = witness_plan
calibrate_from_references = witness_plan
compare_by_calibration_region = provenance_compare
