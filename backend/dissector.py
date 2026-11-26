"""Cut sequencing with two clamps and pouch-width checks on an upright film strip.

The strip runs left to right inside a guide. Frame boxes mark the pictures,
``grip`` marks label clampable blank leader at the top or bottom margin, and
each cut is a vertical blade position in the gap between two frames.

The planner never lets a clamp cross a finished cut: after a cut the two
children are separate pieces, and a grip only belongs to the piece it is
physically still attached to.
"""

from dataclasses import dataclass
from itertools import product
import math

from PIL import Image, ImageDraw

from .strip import CutSettings, _vertical_line, assess_cut, marked_strip_image, verify_cut
from .values import Box, DeskError, Segment, integer, mapping, number


CLAMP_COUNT = 2



@dataclass(frozen=True)
class Grip:
    mark_id: str
    label: str
    box: Box
    edge: str

    def segment_in(self, left, right):
        """Return the usable x-span of this grip inside a piece, or None."""
        lo = max(self.box.x, left)
        hi = min(self.box.right, right)
        if hi <= lo:
            return None
        return lo, hi


@dataclass(frozen=True)
class BagRange:
    minimum_mm: float
    maximum_mm: float


class Dissection:
    def __init__(self, marks, settings, plan):
        self.marks = marks
        self.settings = settings
        self.ppm = settings.pixels_per_mm
        self.kerf_px = settings.kerf_mm * self.ppm
        guide = settings.guide
        self.guide = guide
        self.plan = plan

        frame_marks = [
            mark for mark in marks
            if mark["kind"] == "frame" and guide.intersect(Box.read(mark["geometry"])) is not None
        ]
        self.frame_marks = sorted(frame_marks, key=lambda item: Box.read(item["geometry"]).x)
        self.frames = [Box.read(item["geometry"]) for item in self.frame_marks]
        if len(self.frames) < 2:
            raise DeskError("Draw at least two frame boxes inside the guide to plan cuts.", field="frames")
        if any(right.x <= left.right for left, right in zip(self.frames, self.frames[1:])):
            raise DeskError("Frame boxes must not overlap; order them along the strip.", field="frames")

        self.grips = self._read_grips()
        self.grip_by_id = {grip.mark_id: grip for grip in self.grips}
        self._check_start_grips()
        self.boundary_x = [guide.x] + [None] * (len(self.frames) - 1) + [guide.right]
        self.gap_columns = self._clear_columns()
        self.bags = self._bag_ranges()
        self.feasible_columns = self._pouch_feasible_sets()
        self.chosen_columns = self._choose_columns()
        for index, column in enumerate(self.chosen_columns, start=1):
            self.boundary_x[index] = column

    # ------------------------------------------------------------------ inputs
    def _read_grips(self):
        guide = self.guide
        frame_rows = [
            (frame.y, frame.bottom) for frame in self.frames
        ]
        grips = []
        for mark in self.marks:
            if mark["kind"] != "grip":
                continue
            box = Box.read(mark["geometry"])
            if guide.intersect(box) is None:
                raise DeskError(f"Grip {mark['label']} sits outside the guide.", field="plan.start_grips")
            overlaps = any(not (box.bottom <= top or box.y >= bottom) for top, bottom in frame_rows)
            if overlaps:
                raise DeskError(
                    f"Grip {mark['label']} overlaps a frame; grips belong on the blank margin.",
                    field="marks",
                )
            edge = "top" if box.center.y < (guide.y + guide.bottom) / 2 else "bottom"
            grips.append(Grip(mark["id"], mark["label"], box, edge))
        if len(grips) < CLAMP_COUNT:
            raise DeskError("Mark at least two Grip zones on the blank film margins.", field="marks")
        return sorted(grips, key=lambda grip: (grip.edge, grip.box.x))

    def _check_start_grips(self):
        starts = self.plan.start_grips
        if len(starts) != CLAMP_COUNT or len(set(starts)) != CLAMP_COUNT:
            raise DeskError("Choose two different starting Grip zones.", field="plan.start_grips")
        minimum_px = self.plan.min_grip_mm * self.ppm
        for mark_id in starts:
            grip = self.grip_by_id.get(mark_id)
            if grip is None:
                raise DeskError("Each starting clamp needs a Grip mark.", field="plan.start_grips")
            span = grip.segment_in(self.guide.x, self.guide.right)
            if span is None or span[1] - span[0] < minimum_px:
                raise DeskError(
                    f"Starting grip {grip.label} is shorter than the minimum clamp contact.",
                    field="plan.start_grips",
                )

    def _bag_ranges(self):
        overrides = {item["frame"]: BagRange(item["min_mm"], item["max_mm"]) for item in self.plan.bags}
        bags = []
        for index, mark in enumerate(self.frame_marks, start=1):
            bag = overrides.get(mark["id"]) or overrides.get(index) or self.plan.default_bag
            if bag.minimum_mm > bag.maximum_mm:
                raise DeskError(f"Bag range for frame {mark['label']} has its minimum above its maximum.", field="plan.bags")
            bags.append(bag)
        return bags

    # ------------------------------------------------------------- cut columns
    def _clear_columns(self):
        """Columns the existing cut clearance logic accepts, grouped by gap."""
        guide = self.guide
        keep = [
            Box.read(mark["geometry"])
            for mark in self.marks
            if mark["kind"] in {"frame", "keep", "reference"}
            and guide.intersect(Box.read(mark["geometry"])) is not None
        ]
        ranges = sorted((max(guide.x, box.x), min(guide.right, box.right)) for box in keep)
        holes = []
        position = guide.x
        for left, right in ranges:
            if position < left:
                holes.append((position, left))
            position = max(position, right)
        if position < guide.right:
            holes.append((position, guide.right))

        gap_columns = []
        for index in range(len(self.frames) - 1):
            gap_left = self.frames[index].right
            gap_right = self.frames[index + 1].x
            columns = []
            reserve = math.ceil(self.settings.exclusion_pixels)
            for hole_left, hole_right in holes:
                if hole_right <= gap_left or hole_left >= gap_right:
                    continue
                start_column = max(hole_left, gap_left) + reserve
                end_column = min(hole_right, gap_right) - reserve
                for column in range(start_column, end_column):
                    report = assess_cut(_vertical_line(column, guide), self.marks, self.settings)
                    if report["status"] == "clear":
                        columns.append(column)
            if not columns:
                raise DeskError(
                    f"No clear blade position between {self.frame_marks[index]['label']} "
                    f"and {self.frame_marks[index + 1]['label']}.",
                    field="cut",
                    code="no-clear-gap",
                )
            gap_columns.append(sorted(set(columns)))
        return gap_columns

    def _propagate(self, columns_per_gap):
        """Tighten boundary intervals until every pouch inequality is stable."""
        n = len(self.frames)
        low = [self.guide.x] + [min(columns) for columns in columns_per_gap] + [self.guide.right]
        high = [self.guide.x] + [max(columns) for columns in columns_per_gap] + [self.guide.right]
        widths = [
            (bag.minimum_mm * self.ppm + self.kerf_px, bag.maximum_mm * self.ppm + self.kerf_px)
            for bag in self.bags
        ]
        for _ in range(2 * (n + 1)):
            before = (low[:], high[:])
            for frame_index, (minimum, maximum) in enumerate(widths):
                left, right = frame_index, frame_index + 1
                low[right] = max(low[right], low[left] + minimum)
                high[left] = min(high[left], high[right] - minimum)
                low[left] = max(low[left], low[right] - maximum)
                high[right] = min(high[right], high[left] + maximum)
            if before == (low, high):
                break
        return low, high

    def _pouch_feasible_sets(self):
        low, high = self._propagate(self.gap_columns)
        sets = []
        for index, columns in enumerate(self.gap_columns, start=1):
            kept = [column for column in columns if low[index] <= column <= high[index]]
            if not kept:
                bag = self.bags[index - 1]
                raise DeskError(
                    f"Clear blade positions cannot keep a pouch between {bag.minimum_mm:g} and "
                    f"{bag.maximum_mm:g} mm around gap {index}.",
                    field="plan.bags",
                    code="pouch-impossible",
                )
            sets.append(kept)
        return sets

    def _leader_mm(self, frame_index, left_column, right_column):
        frame = self.frames[frame_index]
        half_kerf = self.settings.kerf_mm / 2
        left_leader = (frame.x - left_column) / self.ppm - half_kerf
        right_leader = (right_column - frame.right) / self.ppm - half_kerf
        return left_leader, right_leader

    def _choose_columns(self):
        """Center every cut, then push the preferred frame's cuts outward."""
        chosen = [min(columns, key=lambda column: abs(column - (min(columns) + max(columns)) / 2))
                  for columns in self.feasible_columns]
        preferred = self.plan.preferred_frame
        if preferred is None:
            return chosen

        def feasible_with(partial):
            columns = [
                [column] if column is not None else self.feasible_columns[gap]
                for gap, column in enumerate(partial)
            ]
            try:
                low, high = self._propagate(columns)
            except DeskError:
                return False
            return all(low[gap + 1] <= column <= high[gap + 1] for gap, column in enumerate(partial))

        partial = chosen[:]
        gaps = [g for g in (preferred - 1, preferred) if 0 <= g < len(chosen)]
        if not gaps:
            return chosen
        options = [self.feasible_columns[gap] for gap in gaps]
        best = None
        for candidate in product(*options):
            for gap, column in zip(gaps, candidate):
                partial[gap] = column
            if not feasible_with(partial):
                continue
            boundaries = [self.guide.x, *partial, self.guide.right]
            leaders = self._leader_mm(preferred - 1, boundaries[preferred - 1], boundaries[preferred])
            score = min(leaders)
            if best is None or score > best[0]:
                best = (score, candidate[:])
        if best is None:
            raise DeskError("The preferred frame cannot keep both pouch limits.", field="plan.preferred_frame")
        for gap, column in zip(gaps, best[1]):
            chosen[gap] = column
        return chosen

    # ------------------------------------------------------------- clamp logic
    def _valid_grip(self, grip, left, right, cut_column):
        """A grip can hold a child piece if it survives the cut and reaches it."""
        span = grip.segment_in(left, right)
        if span is None:
            return None
        length_mm = (span[1] - span[0]) / self.ppm
        if length_mm < self.plan.min_grip_mm:
            return None
        nearest = min(abs(cut_column - span[0]), abs(cut_column - span[1]))
        if cut_column in range(grip.box.x, grip.box.right) or nearest < self.kerf_px / 2:
            return None
        distance_mm = nearest / self.ppm
        if distance_mm > self.plan.reach_mm:
            return None
        return {"grip": grip, "length_mm": round(length_mm, 3), "distance_mm": round(distance_mm, 3)}

    def _piece_grips(self, left, right, cut_column):
        left_grips, right_grips = [], []
        for grip in self.grips:
            kept = self._valid_grip(grip, left, cut_column, cut_column)
            if kept:
                left_grips.append(kept)
            kept = self._valid_grip(grip, cut_column, right, cut_column)
            if kept:
                right_grips.append(kept)
        return left_grips, right_grips

    def plan_sequence(self):
        cuts = list(range(1, len(self.frames)))
        starts = self.plan.start_grips
        initial = tuple(
            (starts[clamp], (self.guide.x, self.guide.right)) for clamp in range(CLAMP_COUNT)
        )
        memo = {}
        failed = set()

        def split_piece(made, boundary):
            cuts_done = sorted([self.guide.x, *[self.chosen_columns[k - 1] for k in made], self.guide.right])
            column = self.chosen_columns[boundary - 1]
            for left, right in zip(cuts_done, cuts_done[1:]):
                if left < column < right:
                    return left, right, column
            raise DeskError("Cut position does not lie inside a remaining piece.", code="plan-state")

        def search(made, clamp_state):
            if len(made) == len(cuts):
                return 0, []
            key = (made, clamp_state)
            if key in failed:
                return None
            if key in memo:
                return memo[key]
            best = None
            for boundary in cuts:
                if boundary in made:
                    continue
                left, right, column = split_piece(made, boundary)
                left_grips, right_grips = self._piece_grips(left, right, column)
                if not left_grips or not right_grips:
                    continue
                for left_choice in left_grips:
                    for right_choice in right_grips:
                        if left_choice["grip"].mark_id == right_choice["grip"].mark_id:
                            continue
                        assignment, moves = self._assign_clamps(
                            clamp_state, left, right, column, left_choice, right_choice
                        )
                        if assignment is None:
                            continue
                        new_made = frozenset({*made, boundary})
                        sub = search(new_made, assignment)
                        if sub is None:
                            continue
                        sub_cost, sub_steps = sub
                        total = moves + sub_cost
                        if best is None or total < best[0]:
                            step = {
                                "cut": boundary,
                                "column": column,
                                "piece": [left, right],
                                "moves": moves,
                                "left_grip": self._grip_view(left_choice),
                                "right_grip": self._grip_view(right_choice),
                                "clamp_moves": [
                                    {"clamp": index + 1,
                                     "grip": self.grip_by_id[state[0]].label,
                                     "piece": list(state[1])}
                                    for index, state in enumerate(assignment)
                                ],
                            }
                            best = (total, [step, *sub_steps])
            if best is None:
                failed.add(key)
            else:
                memo[key] = best
            return best

        found = search(frozenset(), initial)
        if found is None:
            raise DeskError(
                "No cut order keeps both child pieces clamped with these Grip zones. Add another Grip mark.",
                code="no-sequence",
            )
        total_moves, steps = found
        cumulative = 0
        for step in steps:
            cumulative += step["moves"]
            step["cumulative_moves"] = cumulative
        return total_moves, steps

    def _assign_clamps(self, clamp_state, left, right, column, left_choice, right_choice):
        """Match the two clamps to the two child sides, counting grip changes.

        A clamp only "stays" when it already holds the active piece through a
        grip that physically ends up on the chosen child side and still meets
        the contact and reach limits. During a re-grip at least one clamp must
        remain on the active piece, so detaching both at once is rejected.
        """
        child_pieces = ((left, column), (column, right))
        choices = (left_choice, right_choice)
        on_active = sum(1 for _, piece in clamp_state if piece == (left, right))
        best = None
        for permutation in ((0, 1), (1, 0)):
            assignment = [None, None]
            moves = 0
            stays = 0
            for side_index, clamp_index in enumerate(permutation):
                choice = choices[side_index]
                grip_id, current_piece = clamp_state[clamp_index]
                unchanged = grip_id == choice["grip"].mark_id and current_piece == (left, right)
                if unchanged:
                    stays += 1
                else:
                    moves += 1
                assignment[clamp_index] = (choice["grip"].mark_id, child_pieces[side_index])
            # Never detach the last clamp that is currently holding this piece.
            if on_active >= 1 and stays == 0:
                continue
            state = tuple(assignment)
            if best is None or moves < best[0]:
                best = (state, moves)
        return best if best is not None else (None, 0)

    @staticmethod
    def _grip_view(choice):
        grip = choice["grip"]
        return {
            "mark_id": grip.mark_id,
            "label": grip.label,
            "edge": grip.edge,
            "contact_mm": choice["length_mm"],
            "blade_distance_mm": choice["distance_mm"],
        }

    # ------------------------------------------------------------------ report
    def frame_report(self):
        rows = []
        for index, (frame, mark, bag) in enumerate(
            zip(self.frames, self.frame_marks, self.bags), start=1
        ):
            left_column = self.boundary_x[index - 1]
            right_column = self.boundary_x[index]
            width_mm = (right_column - left_column - self.kerf_px) / self.ppm
            left_leader, right_leader = self._leader_mm(index - 1, left_column, right_column)
            rows.append({
                "frame": index,
                "mark_id": mark["id"],
                "label": mark["label"],
                "cell_width_mm": round(width_mm, 3),
                "bag_min_mm": bag.minimum_mm,
                "bag_max_mm": bag.maximum_mm,
                "fits": bag.minimum_mm <= width_mm <= bag.maximum_mm,
                "left_leader_mm": round(left_leader, 3),
                "right_leader_mm": round(right_leader, 3),
            })
        return rows

    def leader_sacrifice(self):
        preferred = self.plan.preferred_frame
        if preferred is None:
            return None
        # The theoretical best ignores pouch limits and only uses blade-clear
        # columns. Comparing against it shows what the pouch constraints cost.
        raw_gaps = {
            gap: self.gap_columns[gap]
            for gap in (preferred - 1, preferred)
            if 0 <= gap < len(self.gap_columns)
        }
        boundaries = [self.guide.x, *self.chosen_columns, self.guide.right]
        chosen_leaders = self._leader_mm(preferred - 1, boundaries[preferred - 1], boundaries[preferred])
        chosen_min = min(chosen_leaders)
        best_min = chosen_min
        gap_indices = sorted(raw_gaps)
        for columns in product(*(raw_gaps[gap] for gap in gap_indices)):
            trial = {gap: column for gap, column in zip(gap_indices, columns)}
            left_column = trial.get(preferred - 1, boundaries[preferred - 1])
            right_column = trial.get(preferred, boundaries[preferred])
            leaders = self._leader_mm(preferred - 1, left_column, right_column)
            best_min = max(best_min, min(leaders))
        return {
            "frame": preferred,
            "label": self.frame_marks[preferred - 1]["label"],
            "chosen_min_leader_mm": round(chosen_min, 3),
            "best_min_leader_mm": round(best_min, 3),
            "sacrificed_mm": round(max(0.0, best_min - chosen_min), 3),
            "chosen_left_leader_mm": round(chosen_leaders[0], 3),
            "chosen_right_leader_mm": round(chosen_leaders[1], 3),
        }

    def cut_lines(self):
        return [
            {"boundary": index, "line": _vertical_line(column, self.guide).as_dict(), "column": column}
            for index, column in enumerate(self.chosen_columns, start=1)
        ]


@dataclass(frozen=True)
class PlanOptions:
    min_grip_mm: float
    reach_mm: float
    default_bag: BagRange
    bags: tuple
    start_grips: tuple
    preferred_frame: int | None

    @classmethod
    def read(cls, raw, frame_count):
        value = mapping(raw, "plan")
        default = BagRange(
            number(value.get("bag_min_mm", 0), "plan.bag_min_mm", lower=0, upper=400),
            number(value.get("bag_max_mm", 400), "plan.bag_max_mm", lower=0, upper=400),
        )
        bags = []
        for item in value.get("bags", []):
            entry = mapping(item, "plan.bags")
            frame = entry.get("frame")
            if isinstance(frame, str) and frame.isdigit():
                frame = int(frame)
            if not isinstance(frame, int) or not 1 <= frame <= frame_count:
                raise DeskError("Refer each pouch range to a frame number.", field="plan.bags")
            bags.append({
                "frame": frame,
                "min_mm": number(entry.get("min_mm"), "plan.bags.min_mm", lower=0, upper=400),
                "max_mm": number(entry.get("max_mm"), "plan.bags.max_mm", lower=0, upper=400),
            })
        preferred = value.get("preferred_frame")
        if preferred not in (None, ""):
            preferred = integer(preferred, "plan.preferred_frame", lower=1, upper=frame_count)
        starts = value.get("start_grips")
        if not isinstance(starts, list) or not starts:
            raise DeskError("Pick the two starting Grip zones.", field="plan.start_grips")
        return cls(
            number(value.get("min_grip_mm", 3), "plan.min_grip_mm", lower=.1, upper=120),
            number(value.get("reach_mm", 60), "plan.reach_mm", lower=.1, upper=400),
            default,
            tuple(bags),
            tuple(starts),
            preferred,
        )


def build_cut_plan(marks, cut_raw, plan_raw, scan):
    settings = CutSettings.read(cut_raw, scan)
    options = PlanOptions.read(plan_raw, _frame_count(marks, settings.guide))
    dissection = Dissection(marks, settings, options)
    total_moves, steps = dissection.plan_sequence()
    result = {
        "status": "planned",
        "cut_count": len(dissection.chosen_columns),
        "clamp_moves": total_moves,
        "frames": dissection.frame_report(),
        "steps": steps,
        "cuts": dissection.cut_lines(),
        "leader": dissection.leader_sacrifice(),
        "grips": [
            {"mark_id": grip.mark_id, "label": grip.label, "edge": grip.edge,
             "x": grip.box.x, "w": grip.box.w}
            for grip in dissection.grips
        ],
        "start_grips": list(options.start_grips),
    }
    return settings, options, result


def _frame_count(marks, guide):
    return sum(
        1 for mark in marks
        if mark["kind"] == "frame" and guide.intersect(Box.read(mark["geometry"])) is not None
    )


# --------------------------------------------------------------------- pouches
def _line_x_at(line, y):
    start, end = line.start, line.end
    if abs(end.y - start.y) < 1e-9:
        return (start.x + end.x) / 2
    fraction = (y - start.y) / (end.y - start.y)
    return start.x + fraction * (end.x - start.x)


def evaluate_pouches(marks, cut_settings, tolerance, bag_input, observed):
    """Verify observed cuts, then judge each finished cell against its pouch."""
    guide = cut_settings.guide
    ppm = cut_settings.pixels_per_mm
    frame_marks = [
        mark for mark in marks
        if mark["kind"] == "frame" and guide.intersect(Box.read(mark["geometry"])) is not None
    ]
    frame_marks.sort(key=lambda item: Box.read(item["geometry"]).x)
    frames = [Box.read(item["geometry"]) for item in frame_marks]

    overrides = {}
    for item in bag_input.get("bags", []):
        entry = mapping(item, "pouch.bags")
        frame = entry.get("frame")
        if isinstance(frame, str) and frame.isdigit():
            frame = int(frame)
        overrides[frame] = BagRange(
            number(entry.get("min_mm"), "pouch.bags.min_mm", lower=0, upper=400),
            number(entry.get("max_mm"), "pouch.bags.max_mm", lower=0, upper=400),
        )
    default = BagRange(
        number(bag_input.get("bag_min_mm", 0), "pouch.bag_min_mm", lower=0, upper=400),
        number(bag_input.get("bag_max_mm", 400), "pouch.bag_max_mm", lower=0, upper=400),
    )

    cuts_by_boundary = {}
    cut_checks = []
    for item in observed:
        entry = mapping(item, "observed")
        boundary = integer(entry.get("boundary"), "observed.boundary", lower=1, upper=len(frames) - 1)
        planned = Segment.read(entry.get("planned"), "observed.planned", bounds=None)
        actual = Segment.read(entry.get("actual"), "observed.actual", bounds=None)
        verification = verify_cut(planned, actual, marks, cut_settings, tolerance)
        cuts_by_boundary[boundary] = (planned, actual, verification)
        cut_checks.append({"boundary": boundary, **verification})

    rows = []
    held_frames = []
    for index, (frame, mark) in enumerate(zip(frames, frame_marks), start=1):
        bag = overrides.get(index, default)
        left = cuts_by_boundary.get(index - 1)
        right = cuts_by_boundary.get(index)
        row = {
            "frame": index,
            "label": mark["label"],
            "bag_min_mm": bag.minimum_mm,
            "bag_max_mm": bag.maximum_mm,
            "left_boundary": index - 1,
            "right_boundary": index,
        }
        if left is None or right is None:
            row.update({"status": "pending", "reason": "Awaiting both observed cuts."})
            rows.append(row)
            continue
        left_line = left[1]
        right_line = right[1]
        stations = sorted({frame.y, (frame.y + frame.bottom) // 2, frame.bottom - 1})
        widths = []
        for y in stations:
            x_left = guide.x if index == 1 and left is None else _line_x_at(left_line, y)
            x_right = guide.right if index == len(frames) and right is None else _line_x_at(right_line, y)
            widths.append((x_right - x_left - cut_settings.kerf_mm * ppm) / ppm)
        minimum_width = min(widths)
        maximum_width = max(widths)
        row.update({
            "min_width_mm": round(minimum_width, 3),
            "max_width_mm": round(maximum_width, 3),
            "stations": len(stations),
        })
        bad_cuts = [
            check["boundary"] for check in cut_checks
            if check["boundary"] in (index - 1, index) and check["status"] != "confirmed"
        ]
        if bad_cuts:
            row.update({"status": "hold", "reason": f"Observed cut {', '.join(map(str, bad_cuts))} failed its tolerance."})
            held_frames.append(index)
        elif minimum_width < bag.minimum_mm or maximum_width > bag.maximum_mm:
            row.update({
                "status": "hold",
                "reason": "Finished cell falls outside the pouch range even though each cut passed its offset.",
            })
            held_frames.append(index)
        else:
            row.update({"status": "confirmed"})
        rows.append(row)

    return {
        "status": "hold" if held_frames else "measured",
        "held_frames": held_frames,
        "frames": rows,
        "cuts": cut_checks,
    }


def plan_preview(source, settings, result, marks):
    """Render the ordered cuts with their step number and active Grip zones."""
    guides = []
    ordered = sorted(result["steps"], key=lambda step: step["cut"])
    for step in ordered:
        line = next(item for item in result["cuts"] if item["boundary"] == step["cut"])
        guides.append((Segment.read(line["line"]), "#8cdab7"))
    image = marked_strip_image(source.image, settings.guide, marks, guides=guides)
    draw = ImageDraw.Draw(image)
    sx = image.width / settings.guide.w
    sy = image.height / settings.guide.h

    def project_x(column):
        return round((column - settings.guide.x) * sx)

    for step in ordered:
        x = project_x(step["column"])
        draw.ellipse((x - 9, 4, x + 9, 22), fill="#24211e", outline="#f1a970", width=2)
        draw.text((x - 4, 7), str(step["cut"]), fill="#f1d9a8")
    return image


def pouch_preview(source, settings, result, marks, observed_lines):
    """Render planned cuts in blue and observed cuts in green/red."""
    guides = []
    colors = {"confirmed": "#8cdab7", "hold-offset": "#ed805f", "hold-clearance": "#e98185"}
    for entry in observed_lines:
        planned = Segment.read(entry["planned"])
        actual = Segment.read(entry["actual"])
        status = entry["status"]
        guides.append((planned, "#90b5f0"))
        guides.append((actual, colors.get(status, "#ed805f")))
    return marked_strip_image(source.image, settings.guide, marks, guides=guides)
