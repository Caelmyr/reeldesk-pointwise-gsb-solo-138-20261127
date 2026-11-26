"""Scenario checks for the two-clamp cut planner and pouch evaluation.

Run with:  python3 -I scripts/test_dissector.py
The script builds an upright four-frame strip entirely from in-memory
geometry; no catalog or image files are required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/opt/pointwise-env/site-packages")

from backend.dissector import Dissection, PlanOptions, evaluate_pouches
from backend.strip import CutSettings
from backend.values import Box, DeskError, Segment


DPI = 120.0
WIDTH, HEIGHT = 1180, 200
GUIDE = Box(0, 0, WIDTH, HEIGHT)


def frame_boxes():
    # Four frames, each 220 wide, with ~80 px blank gaps between them.
    return [
        Box(40, 60, 220, 80),    # frame 1: 40..260
        Box(340, 60, 220, 80),   # frame 2: 340..560
        Box(640, 60, 220, 80),   # frame 3: 640..860
        Box(940, 60, 220, 80),   # frame 4: 940..1160
    ]


def grip_boxes():
    # Two outer end grips and one grip on each side of every interior gap,
    # placed on the blank margin clear of the blade columns.
    return [
        Box(8, 10, 70, 18),      # g1 far left (start)
        Box(1102, 10, 70, 18),   # g2 far right (start)
        Box(266, 10, 26, 18),    # g3 right of gap 1 blade
        Box(308, 10, 26, 18),    # g4 left of gap 2 blade
        Box(566, 10, 26, 18),    # g5 right of gap 2 blade
        Box(608, 10, 26, 18),    # g6 left of gap 3 blade
        Box(866, 10, 26, 18),    # g7 right of gap 3 blade
        Box(908, 10, 26, 18),    # g8 left of frame 4 side
    ]


def make_marks(grips=None):
    grips = grip_boxes() if grips is None else grips
    marks = []
    for index, box in enumerate(frame_boxes(), start=1):
        marks.append({"id": f"f{index}", "kind": "frame", "label": f"Frame {index}",
                      "geometry": box.as_dict(), "notes": ""})
    for index, box in enumerate(grips, start=1):
        marks.append({"id": f"g{index}", "kind": "grip", "label": f"Grip {index}",
                      "geometry": box.as_dict(), "notes": ""})
    return marks


def settings():
    return CutSettings(GUIDE, kerf_mm=0.16, reserve_mm=0.0, dpi=DPI, retained="both")


def options(**overrides):
    base = {
        "min_grip_mm": 2.0,
        "reach_mm": 80.0,
        "bag_min_mm": 0.0,
        "bag_max_mm": 400.0,
        "bags": [],
        "start_grips": ["g1", "g2"],
        "preferred_frame": None,
    }
    base.update(overrides)
    return PlanOptions.read(base, frame_count=4)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        raise SystemExit(1)


def main():
    marks = make_marks()
    cut = settings()

    plan = Dissection(marks, cut, options())
    moves, steps = plan.plan_sequence()
    print("chosen columns:", plan.chosen_columns)
    for step in steps:
        print(step)
    check("produces three cuts", len(steps) == 3)
    check("both child sides clamped each cut",
          all(step["left_grip"] and step["right_grip"] for step in steps))
    check("at most two clamp moves per cut",
          all(step["moves"] <= 2 for step in steps))
    check("grip contact meets minimum",
          all(step["left_grip"]["contact_mm"] >= 2.0 and step["right_grip"]["contact_mm"] >= 2.0
              for step in steps))
    check("blade distance within reach",
          all(step["left_grip"]["blade_distance_mm"] <= 80.0
              and step["right_grip"]["blade_distance_mm"] <= 80.0 for step in steps))
    check("clamp moves total is consistent",
          moves == sum(step["moves"] for step in steps), f"total={moves}")
    check("a starting clamp is used on the first cut",
          {steps[0]["left_grip"]["mark_id"], steps[0]["right_grip"]["mark_id"]}
          & set(options().start_grips))

    # No clamp may ever reference a grip that the cut already detached.
    columns = plan.chosen_columns
    for step in steps:
        column = step["column"]
        for side in ("left_grip", "right_grip"):
            grip = next(g for g in plan.grips if g.mark_id == step[side]["mark_id"])
            if side == "left_grip":
                check(f"{grip.label} stays left of cut {step['cut']}", grip.box.right <= column)
            else:
                check(f"{grip.label} stays right of cut {step['cut']}", grip.box.x >= column)

    # Preferred frame should push cuts outward while keeping pouch limits.
    preferred = Dissection(marks, cut, options(preferred_frame=2))
    sacrifice = preferred.leader_sacrifice()
    print("leader sacrifice:", sacrifice)
    check("sacrifice non-negative", sacrifice["sacrificed_mm"] >= -1e-9)
    check("chosen leader within best",
          sacrifice["chosen_min_leader_mm"] <= sacrifice["best_min_leader_mm"] + 1e-9)

    # A pouch range narrower than any feasible cell must be rejected up front.
    tight = options(bags=[{"frame": 1, "min_mm": 1.0, "max_mm": 2.0}])
    try:
        Dissection(marks, cut, tight)
        check("impossible pouch rejected", False)
    except Exception as error:
        check("impossible pouch rejected", getattr(error, "code", "") == "pouch-impossible", str(error))

    # A grip overlapping a frame is refused.
    bad_marks = make_marks(grips=grip_boxes() + [Box(100, 70, 40, 20)])
    try:
        Dissection(bad_marks, cut, options())
        check("grip over frame refused", False)
    except DeskError as error:
        check("grip over frame refused", "blank margin" in str(error))

    # Pouch evaluation with both observed cuts present for an interior frame.
    observed = []
    for boundary, column in enumerate(columns, start=1):
        planned = Segment.read({"start": {"x": column, "y": GUIDE.y},
                                "end": {"x": column, "y": GUIDE.bottom - 1}})
        actual = Segment.read({"start": {"x": column + 1, "y": GUIDE.y},
                               "end": {"x": column + 1, "y": GUIDE.bottom - 1}})
        observed.append({"boundary": boundary,
                         "planned": planned.as_dict(), "actual": actual.as_dict()})
    result = evaluate_pouches(
        marks, cut, tolerance=0.5,
        bag_input={"bag_min_mm": 0, "bag_max_mm": 400}, observed=observed,
    )
    interior = [row for row in result["frames"] if row["frame"] in (2, 3)]
    check("interior frames measured with both cuts",
          all(row["status"] == "confirmed" for row in interior), str(interior))
    check("outer frames pending an edge cut",
          all(row["status"] == "pending" for row in result["frames"] if row["frame"] in (1, 4)))

    # Force a too-narrow max width: cuts pass offset but the cell fails pouch.
    target = interior[0]
    held = evaluate_pouches(
        marks, cut, tolerance=0.5,
        bag_input={"bag_min_mm": 0, "bag_max_mm": target["min_width_mm"] - 0.5},
        observed=observed,
    )
    held_row = next(row for row in held["frames"] if row["frame"] == target["frame"])
    check("cell over pouch width holds frame even when cuts pass",
          held_row["status"] == "hold" and target["frame"] in held["held_frames"], str(held_row))

    # One cut outside offset tolerance must hold every frame touching it.
    bad_observed = [dict(item) for item in observed]
    far_column = columns[1] + 40
    bad_observed[1] = {
        "boundary": 2,
        "planned": {"start": {"x": columns[1], "y": GUIDE.y},
                    "end": {"x": columns[1], "y": GUIDE.bottom - 1}},
        "actual": {"start": {"x": far_column, "y": GUIDE.y},
                   "end": {"x": far_column, "y": GUIDE.bottom - 1}},
    }
    offset_fail = evaluate_pouches(
        marks, cut, tolerance=0.25,
        bag_input={"bag_min_mm": 0, "bag_max_mm": 400}, observed=bad_observed,
    )
    check("offset failure reported on cut 2",
          any(c["boundary"] == 2 and c["status"] == "hold-offset" for c in offset_fail["cuts"]))
    check("frames touching bad cut are held",
          set(offset_fail["held_frames"]) >= {2, 3}, str(offset_fail["held_frames"]))

    print("all dissector checks passed")


if __name__ == "__main__":
    main()
