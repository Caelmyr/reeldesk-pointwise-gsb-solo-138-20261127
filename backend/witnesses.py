"""Witness-based illumination compensation and provenance-aware region comparison.

The solver works in optical-density space.  A per-cell correction ``q`` is
added to density, which is equivalent to multiplying transmission by
``10 ** -q``.  Constraints are difference constraints, so feasibility can be
checked exactly with Bellman-Ford before a small linear program chooses the
least-moving feasible corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .values import PRIMARY_CHANNELS, Box, DeskError, mapping


CHANNEL_COUNT = 3
EPS = 1e-8
SEARCH_NODE_LIMIT = 20_000
EXHAUSTIVE_LIMIT = 14


def _limit_vector(value: Any, field: str, default: float = 0.04) -> tuple[float, float, float]:
    if value is None:
        values = [default] * 3
    elif isinstance(value, bool):
        raise DeskError("Enter a non-negative density limit.", field=field)
    elif isinstance(value, (int, float)):
        number = float(value)
        if not np.isfinite(number) or number < 0 or number > 4:
            raise DeskError("Enter a density limit from 0 to 4.", field=field)
        values = [number] * 3
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        values = []
        for index, item in enumerate(value):
            try:
                number = float(item)
            except (TypeError, ValueError):
                raise DeskError("Enter a numeric density limit.", field=f"{field}[{index}]") from None
            if not np.isfinite(number) or number < 0 or number > 4:
                raise DeskError("Enter a density limit from 0 to 4.", field=f"{field}[{index}]")
            values.append(number)
    else:
        raise DeskError("Use one density limit or three RGB limits.", field=field)
    return tuple(values)  # type: ignore[return-value]


def grid_regions(region: Box, rows: int, columns: int) -> list[dict[str, Any]]:
    """Return the same half-open tile geometry used by the illumination map."""
    y_bounds = np.rint(np.linspace(region.y, region.bottom, rows + 1)).astype(int)
    x_bounds = np.rint(np.linspace(region.x, region.right, columns + 1)).astype(int)
    cells = []
    for row in range(rows):
        for column in range(columns):
            box = Box(
                int(x_bounds[column]), int(y_bounds[row]),
                int(x_bounds[column + 1] - x_bounds[column]),
                int(y_bounds[row + 1] - y_bounds[row]),
            )
            cells.append({"row": row, "column": column, "index": row * columns + column, "region": box.as_dict()})
    return cells


def parse_cell(value: Any, rows: int, columns: int, field: str = "reference_cell") -> tuple[int, int]:
    if value is None:
        raise DeskError("Choose the fixed reference grid cell.", field=field)
    one_based = False
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise DeskError("Use row and column for the reference cell.", field=field)
        row, column = value
    elif isinstance(value, int) and not isinstance(value, bool):
        if not 0 <= value < rows * columns:
            raise DeskError("The reference cell is outside the illumination grid.", field=field)
        return divmod(value, columns)
    elif isinstance(value, dict):
        value = mapping(value, field)
        if "index" in value and value["index"] is not None:
            index = int(value["index"])
            if not 0 <= index < rows * columns:
                raise DeskError("The reference cell is outside the illumination grid.", field=field)
            return divmod(index, columns)
        row = value.get("row", value.get("r", value.get("row_index")))
        column = value.get("column", value.get("c", value.get("col", value.get("column_index"))))
        one_based = bool(value.get("one_based", value.get("oneBased", False)))
        if row is None or column is None:
            raise DeskError("Use row and column for the reference cell.", field=field)
    else:
        raise DeskError("Use row and column for the reference cell.", field=field)
    try:
        row = int(row)
        column = int(column)
    except (TypeError, ValueError):
        raise DeskError("Reference cell row and column must be whole numbers.", field=field) from None
    if one_based:
        row -= 1
        column -= 1
    if not (0 <= row < rows and 0 <= column < columns):
        raise DeskError("The reference cell is outside the illumination grid.", field=field)
    return row, column


def _as_vector(value: Any, field: str, *, default: Any = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise DeskError(f"{field} is required.", field=field)
        value = default
    if isinstance(value, np.ndarray):
        vector = value.astype(np.float64, copy=False)
    elif isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise DeskError(f"{field} must contain three channel values.", field=field)
        vector = np.asarray(value, dtype=np.float64)
    else:
        raise DeskError(f"{field} must contain three channel values.", field=field)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise DeskError(f"{field} must contain finite RGB values.", field=field)
    return vector


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    weight: tuple[float, float, float]
    kind: str
    witnesses: tuple[str, ...] = ()
    label: str | None = None
    pair: tuple[int, int] | None = None


def _observation_id(index: int) -> str:
    return f"obs-{index + 1}"


def normalize_observations(
    raw: Iterable[dict[str, Any]],
    rows: int,
    columns: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        item = mapping(item, f"witnesses[{index}]")
        name = item.get("name", item.get("label", item.get("witness_name")))
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if "cell" in item and item["cell"] is not None:
            row, column = parse_cell(item["cell"], rows, columns, f"witnesses[{index}].cell")
        elif {"row", "column"} <= set(item):
            row = int(item["row"])
            column = int(item["column"])
            if not (0 <= row < rows and 0 <= column < columns):
                raise DeskError("A witness is outside the illumination grid.", field=f"witnesses[{index}]")
        elif {"r", "c"} <= set(item):
            row = int(item["r"])
            column = int(item["c"])
            if not (0 <= row < rows and 0 <= column < columns):
                raise DeskError("A witness is outside the illumination grid.", field=f"witnesses[{index}]")
        else:
            raise DeskError("Give each witness a grid cell.", field=f"witnesses[{index}].cell")
        median = item.get("median", item.get("density", item.get("medians", item.get("density_median"))))
        median_vector = _as_vector(median, f"witnesses[{index}].median")
        samples = item.get("samples", item.get("counts", item.get("sample_count", [1, 1, 1])))
        if isinstance(samples, (int, float)):
            sample_vector = np.array([int(samples)] * 3, dtype=int)
        else:
            sample_vector = np.asarray(samples, dtype=int)
            if sample_vector.shape != (3,):
                raise DeskError("Witness samples must be one count or three RGB counts.", field=f"witnesses[{index}].samples")
        witness_id = str(item.get("witness_id", item.get("id", item.get("mark_id", f"w{index + 1}"))))
        observations.append({
            "id": str(item.get("observation_id", _observation_id(index))),
            "witness_id": witness_id,
            "name": name,
            "row": row,
            "column": column,
            "cell": row * columns + column,
            "median": median_vector,
            "samples": np.maximum(sample_vector, 0),
            "detail": item,
        })
    return observations


def _build_edges(
    active_observations: list[dict[str, Any]],
    active_cells: list[int],
    columns: int,
    equal_limit: tuple[float, float, float],
    jump_limit: tuple[float, float, float],
    anchor: int | None,
    node_count: int,
    z_node: int,
) -> list[Edge]:
    local = {global_cell: index for index, global_cell in enumerate(active_cells)}
    edges: list[Edge] = []

    by_name: dict[str, list[dict[str, Any]]] = {}
    for observation in active_observations:
        by_name.setdefault(observation["name"], []).append(observation)

    for name, group in by_name.items():
        for left_index in range(len(group)):
            left = group[left_index]
            for right_index in range(left_index + 1, len(group)):
                right = group[right_index]
                u = local[left["cell"]]
                v = local[right["cell"]]
                delta = left["median"] - right["median"]
                # |delta_c + q_u - q_v| <= E_c gives two difference edges.
                weights_uv = tuple(equal_limit[c] - float(delta[c]) for c in range(3))
                weights_vu = tuple(equal_limit[c] + float(delta[c]) for c in range(3))
                witnesses = tuple(dict.fromkeys([left["witness_id"], right["witness_id"]]))
                pair = (left_index, right_index)
                edges.append(Edge(v, u, weights_uv, "same_name", witnesses, name, pair))
                edges.append(Edge(u, v, weights_vu, "same_name", witnesses, name, pair))

    active_set = set(active_cells)
    for global_cell in active_cells:
        row, column = divmod(global_cell, columns)
        for neighbor in (global_cell + 1, global_cell + columns):
            if neighbor not in active_set:
                continue
            # Only right and down are visited, so each grid edge appears once.
            if neighbor == global_cell + 1 and (column + 1 >= columns):
                continue
            u = local[global_cell]
            v = local[neighbor]
            edges.append(Edge(v, u, jump_limit, "adjacent"))
            edges.append(Edge(u, v, jump_limit, "adjacent"))

    if anchor is not None and anchor in local:
        anchor_node = local[anchor]
        edges.append(Edge(anchor_node, z_node, (0.0, 0.0, 0.0), "anchor"))
        edges.append(Edge(z_node, anchor_node, (0.0, 0.0, 0.0), "anchor"))
    return edges


def _negative_cycle(nodes: int, edges: list[Edge], channel: int) -> list[int] | None:
    if nodes <= 0:
        return None
    distance = [0.0] * nodes
    parent = [-1] * nodes
    parent_edge = [-1] * nodes
    last_updated = -1
    for _ in range(nodes):
        last_updated = -1
        for edge_index, edge in enumerate(edges):
            candidate = distance[edge.u] + edge.weight[channel]
            if candidate < distance[edge.v] - 1e-9:
                distance[edge.v] = candidate
                parent[edge.v] = edge.u
                parent_edge[edge.v] = edge_index
                last_updated = edge.v
        if last_updated == -1:
            return None
    if last_updated == -1:
        return None
    cycle_node = last_updated
    for _ in range(nodes):
        cycle_node = parent[cycle_node]
        if cycle_node < 0:
            return None
    start = cycle_node
    cycle_edges: list[int] = []
    seen = set()
    current = start
    while current >= 0 and current not in seen:
        seen.add(current)
        edge_index = parent_edge[current]
        if edge_index < 0:
            return None
        cycle_edges.append(edge_index)
        current = parent[current]
        if current == start:
            return cycle_edges
    return None


def _feasibility_edges(
    observations: list[dict[str, Any]],
    used: set[str],
    rows: int,
    columns: int,
    equal_limit,
    jump_limit,
    anchor: int | None,
):
    active = [item for item in observations if item["witness_id"] in used]
    cells = sorted({item["cell"] for item in active} | ({anchor} if anchor is not None else set()))
    local_count = len(cells)
    z_node = local_count
    edges = _build_edges(active, cells, columns, equal_limit, jump_limit, anchor, local_count + 1, z_node)
    return active, cells, edges, z_node


def feasible(
    observations,
    used,
    rows,
    columns,
    equal_limit,
    jump_limit,
    anchor,
    *,
    bounds: float | None = None,
):
    active, cells, edges, z_node = _feasibility_edges(
        observations, used, rows, columns, equal_limit, jump_limit, anchor
    )
    node_count = len(cells) + 1
    if bounds is not None:
        for local_cell in range(len(cells)):
            edges.append(Edge(z_node, local_cell, (bounds,) * 3, "bound"))
            edges.append(Edge(local_cell, z_node, (bounds,) * 3, "bound"))
    for channel in range(CHANNEL_COUNT):
        if _negative_cycle(node_count, edges, channel) is not None:
            return False
    return True


def _conflict_witnesses(
    observations,
    used,
    rows,
    columns,
    equal_limit,
    jump_limit,
    anchor,
) -> list[str]:
    _, _, edges, _ = _feasibility_edges(
        observations, used, rows, columns, equal_limit, jump_limit, anchor
    )
    node_count = len({item["cell"] for item in observations if item["witness_id"] in used} | ({anchor} if anchor is not None else set())) + 1
    candidates: list[str] = []
    for channel in range(CHANNEL_COUNT):
        cycle = _negative_cycle(node_count, edges, channel)
        if not cycle:
            continue
        for edge_index in cycle:
            edge = edges[edge_index]
            if edge.kind == "same_name":
                candidates.extend(edge.witnesses)
    return list(dict.fromkeys(item for item in candidates if item in used))


def _bellman_solution(node_count: int, edges: list[Edge], channel: int) -> np.ndarray:
    distance = np.zeros(node_count, dtype=np.float64)
    for _ in range(max(1, node_count - 1)):
        changed = False
        for edge in edges:
            candidate = distance[edge.u] + edge.weight[channel]
            if candidate < distance[edge.v] - 1e-10:
                distance[edge.v] = candidate
                changed = True
        if not changed:
            break
    return distance


def _components(cells: list[int], edges: list[Edge], z_node: int, anchor: int | None) -> list[list[int]]:
    parent = list(range(len(cells) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        if edge.kind in {"same_name", "adjacent", "anchor"}:
            union(edge.u, edge.v)
    groups: dict[int, list[int]] = {}
    for local in range(len(cells)):
        groups.setdefault(find(local), []).append(local)
    return list(groups.values())


def _rough_cost(observations, used, rows, columns, equal_limit, jump_limit, anchor):
    active, cells, edges, z_node = _feasibility_edges(
        observations, used, rows, columns, equal_limit, jump_limit, anchor
    )
    if not cells:
        return 0.0, 0.0
    node_count = len(cells) + 1
    corrections = np.zeros((len(cells), 3), dtype=np.float64)
    for channel in range(3):
        distance = _bellman_solution(node_count, edges, channel)
        values = distance[:-1].copy()
        # Recenter each unanchored component; anchored component is shifted to anchor.
        for component in _components(cells, edges, z_node, anchor):
            idx = np.array(component, dtype=int)
            if anchor is not None:
                anchor_local = cells.index(anchor) if anchor in cells else None
                if anchor_local in component:
                    values[idx] -= values[anchor_local]
                    continue
            lo, hi = float(np.min(values[idx])), float(np.max(values[idx]))
            values[idx] -= (lo + hi) / 2
        corrections[:, channel] = values
    return float(np.max(np.abs(corrections))), float(np.sum(np.abs(corrections)))


def _score_set(observations, used, rows, columns, equal_limit, jump_limit, anchor):
    held = len({item["cell"] for item in observations if item["witness_id"] in used} | ({anchor} if anchor is not None else set()))
    max_cost, sum_cost = _rough_cost(observations, used, rows, columns, equal_limit, jump_limit, anchor)
    return {
        "held": held,
        "used": len(used),
        "max_cost": -round(max_cost, 9),
        "sum_cost": -round(sum_cost, 9),
    }


def _greedy_selection(witness_ids, observations, rows, columns, equal_limit, jump_limit, anchor):
    discarded: set[str] = set()
    used = set(witness_ids)
    for _ in range(len(witness_ids)):
        candidates = _conflict_witnesses(observations, used, rows, columns, equal_limit, jump_limit, anchor)
        if not candidates:
            break
        # Prefer removing the witness that resolves the conflict with the least
        # damage to cell coverage.  Ties prefer fewer samples.
        best_id = None
        best_key = None
        for witness_id in candidates:
            trial = used - {witness_id}
            resolves = feasible(observations, trial, rows, columns, equal_limit, jump_limit, anchor)
            cells_lost = 0 if any(
                item["witness_id"] != witness_id and item["witness_id"] in trial
                for item in observations
            ) else 1
            sample_sum = int(sum(int(np.sum(item["samples"])) for item in observations if item["witness_id"] == witness_id))
            key = (0 if resolves else 1, cells_lost, sample_sum, witness_id)
            if best_key is None or key < best_key:
                best_key = key
                best_id = witness_id
        used.discard(best_id)
        discarded.add(best_id)
    return used, discarded


def select_witnesses(
    observations: list[dict[str, Any]],
    rows: int,
    columns: int,
    equal_limit,
    jump_limit,
    anchor: int | None,
):
    witness_ids = sorted({item["witness_id"] for item in observations})
    if not witness_ids:
        return set(), set(), {"strategy": "none", "search_nodes": 0}

    initial_used, initial_discarded = _greedy_selection(
        witness_ids, observations, rows, columns, equal_limit, jump_limit, anchor
    )
    best_used = set(initial_used)
    best_score = _score_set(observations, best_used, rows, columns, equal_limit, jump_limit, anchor)

    # For small witness counts, enumerate every subset.  This gives a hard
    # guarantee that the largest possible supported-cell set is recommended.
    if len(witness_ids) <= EXHAUSTIVE_LIMIT:
        all_set = set(witness_ids)
        for mask in range(1 << len(witness_ids)):
            discarded = {witness_ids[i] for i in range(len(witness_ids)) if mask & (1 << i)}
            used = all_set - discarded
            if not feasible(observations, used, rows, columns, equal_limit, jump_limit, anchor):
                continue
            score = _score_set(observations, used, rows, columns, equal_limit, jump_limit, anchor)
            if (score["held"], score["used"], score["max_cost"], score["sum_cost"]) > (
                best_score["held"], best_score["used"], best_score["max_cost"], best_score["sum_cost"]
            ):
                best_score = score
                best_used = used
        # Restore evidence whenever it remains jointly feasible.  Enumeration
        # already usually finds this, but this also makes the result robust.
        for witness_id in sorted(set(witness_ids) - best_used):
            trial = best_used | {witness_id}
            if feasible(observations, trial, rows, columns, equal_limit, jump_limit, anchor):
                best_used = trial
        return best_used, set(witness_ids) - best_used, {"strategy": "exhaustive", "search_nodes": 1 << len(witness_ids)}

    # Conflict-directed DFS for larger marks sets.
    nodes = 0

    def search(discarded: set[str], last_index: int):
        nonlocal best_used, best_score, nodes
        nodes += 1
        if nodes > SEARCH_NODE_LIMIT:
            return
        used = set(witness_ids) - discarded
        if feasible(observations, used, rows, columns, equal_limit, jump_limit, anchor):
            score = _score_set(observations, used, rows, columns, equal_limit, jump_limit, anchor)
            if (score["held"], score["used"], score["max_cost"], score["sum_cost"]) > (
                best_score["held"], best_score["used"], best_score["max_cost"], best_score["sum_cost"]
            ):
                best_score = score
                best_used = set(used)
            return

        candidates = _conflict_witnesses(observations, used, rows, columns, equal_limit, jump_limit, anchor)
        candidate_indexes = [witness_ids.index(item) for item in candidates if witness_ids.index(item) > last_index]
        upper_held = len({
            item["cell"] for item in observations
            if item["witness_id"] not in discarded
        } | ({anchor} if anchor is not None else set()))
        upper_used = len(witness_ids) - len(discarded)
        if (upper_held, upper_used) < (best_score["held"], best_score["used"]):
            return
        for index in sorted(set(candidate_indexes)):
            search(discarded | {witness_ids[index]}, index)

    search(set(), -1)
    for witness_id in sorted(set(witness_ids) - best_used):
        trial = best_used | {witness_id}
        if feasible(observations, trial, rows, columns, equal_limit, jump_limit, anchor):
            best_used = trial
    return best_used, set(witness_ids) - best_used, {"strategy": "conflict_search", "search_nodes": nodes}


def minimum_corrections(
    observations,
    used,
    rows,
    columns,
    equal_limit,
    jump_limit,
    anchor,
):
    active, cells, edges, z_node = _feasibility_edges(
        observations, used, rows, columns, equal_limit, jump_limit, anchor
    )
    if not cells:
        return cells, np.zeros((0, 3)), [], float("nan")

    low, high = 0.0, 1.0
    for _ in range(8):
        if feasible(observations, used, rows, columns, equal_limit, jump_limit, anchor, bounds=high):
            break
        high *= 2
    else:
        high = 64.0
    for _ in range(42):
        middle = (low + high) / 2
        if feasible(observations, used, rows, columns, equal_limit, jump_limit, anchor, bounds=middle):
            high = middle
        else:
            low = middle
    max_abs = high
    component_groups = _components(cells, edges, z_node, anchor)
    anchor_local = cells.index(anchor) if anchor is not None and anchor in cells else None

    corrections = None
    try:
        from scipy.optimize import linprog

        n = len(cells)
        # Variables: q[n*3], t[n*3].
        size = 6 * n
        objective = np.zeros(size)
        objective[3 * n:] = 1.0
        a_ub: list[np.ndarray] = []
        b_ub: list[float] = []

        def add_linear(coeff: dict[int, float], rhs: float):
            row = np.zeros(size)
            for index, value in coeff.items():
                row[index] = value
            a_ub.append(row)
            b_ub.append(rhs)

        local = {global_cell: index for index, global_cell in enumerate(cells)}
        for edge in edges:
            if edge.kind not in {"same_name", "adjacent"}:
                continue
            for channel in range(3):
                u = 3 * edge.u + channel
                v = 3 * edge.v + channel
                add_linear({u: -1.0, v: 1.0}, edge.weight[channel])

        for local_cell in range(n):
            for channel in range(3):
                q_index = 3 * local_cell + channel
                t_index = 3 * n + q_index
                add_linear({q_index: 1.0, t_index: -1.0}, 0.0)
                add_linear({q_index: -1.0, t_index: -1.0}, 0.0)

        bounds = [(None, None)] * (3 * n) + [(0.0, max_abs + 2e-5)] * (3 * n)
        if anchor is not None and anchor in local:
            for channel in range(3):
                bounds[3 * local[anchor] + channel] = (0.0, 0.0)
        result = linprog(
            objective,
            A_ub=np.asarray(a_ub) if a_ub else None,
            b_ub=np.asarray(b_ub) if b_ub else None,
            bounds=bounds,
            method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        if result.success:
            corrections = result.x[:3 * n].reshape(n, 3)
    except Exception:
        corrections = None

    if corrections is None:
        node_count = len(cells) + 1
        corrections = np.zeros((len(cells), 3))
        bounded_edges = list(edges)
        for local_cell in range(len(cells)):
            bounded_edges.append(Edge(z_node, local_cell, (max_abs + 1e-5,) * 3, "bound"))
            bounded_edges.append(Edge(local_cell, z_node, (max_abs + 1e-5,) * 3, "bound"))
        for channel in range(3):
            values = _bellman_solution(node_count, bounded_edges, channel)[:-1]
            corrections[:, channel] = values

    # Gauge freedom does not affect constraints.  Pin the requested cell to
    # zero; center every disconnected block to minimize its maximum |shift|.
    for group in component_groups:
        if not group:
            continue
        for channel in range(3):
            values = corrections[group, channel]
            if anchor_local is not None and anchor_local in group:
                shift = float(corrections[anchor_local, channel])
            else:
                shift = (float(np.min(values)) + float(np.max(values))) / 2
            corrections[group, channel] = values - shift

    corrections = np.round(corrections, 6)
    return cells, corrections, component_groups, float(np.max(np.abs(corrections))) if corrections.size else 0.0


def witness_plan(
    *,
    rows: int,
    columns: int,
    observations: list[dict[str, Any]],
    region: Box | dict[str, Any] | None = None,
    reference_cell: tuple[int, int] | None = None,
    same_name_limit: Any = 0.04,
    jump_limit: Any = 0.03,
    witness_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = int(rows)
    columns = int(columns)
    if not (2 <= rows <= 12 and 2 <= columns <= 18):
        raise DeskError("Use 2–12 rows and 2–18 columns for witness compensation.", field="grid")
    equal_limit = _limit_vector(same_name_limit, "same_name_limit", 0.04)
    adjacent_limit = _limit_vector(jump_limit, "jump_limit", 0.03)
    anchor_global = reference_cell[0] * columns + reference_cell[1] if reference_cell is not None else None

    observations = normalize_observations(observations or [], rows, columns)
    used, discarded, search_info = select_witnesses(
        observations, rows, columns, equal_limit, adjacent_limit, anchor_global
    )
    cells, corrections, component_groups, max_correction = minimum_corrections(
        observations, used, rows, columns, equal_limit, adjacent_limit, anchor_global
    )
    local_by_global = {global_cell: index for index, global_cell in enumerate(cells)}

    component_by_local: dict[int, str] = {}
    component_records = []
    anchored_component = None
    ordered_groups = sorted(component_groups, key=lambda group: min(cells[i] for i in group))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for component_index, group in enumerate(ordered_groups):
        component_id = letters[component_index] if component_index < len(letters) else f"B{component_index + 1}"
        global_members = [cells[i] for i in group]
        is_anchored = anchor_global is not None and anchor_global in global_members
        if is_anchored:
            component_id = "REF"
            anchored_component = component_id
        for local in group:
            component_by_local[local] = component_id
        component_max = 0.0
        max_at = None
        for local in group:
            for channel in range(3):
                magnitude = abs(float(corrections[local, channel]))
                if magnitude > component_max:
                    component_max = magnitude
                    max_at = (cells[local], channel)
        component_records.append({
            "id": component_id,
            "anchored": is_anchored,
            "connected_to_reference": is_anchored,
            "reference_cell": {"row": reference_cell[0], "column": reference_cell[1]} if is_anchored and reference_cell else None,
            "cells": [{"row": divmod(cell, columns)[0], "column": divmod(cell, columns)[1]} for cell in global_members],
            "cell_indexes": global_members,
            "max_correction": round(component_max, 6),
            "max_correction_cell": (
                {"row": divmod(max_at[0], columns)[0], "column": divmod(max_at[0], columns)[1],
                 "channel": PRIMARY_CHANNELS[max_at[1]]}
                if max_at else None
            ),
        })

    grid = grid_regions(region if isinstance(region, Box) else Box.read(region, "region") if region else Box(0, 0, 1, 1), rows, columns)
    cell_records: list[dict[str, Any]] = []
    for grid_cell in grid:
        global_cell = grid_cell["index"]
        row, column = grid_cell["row"], grid_cell["column"]
        is_anchor = global_cell == anchor_global
        local = local_by_global.get(global_cell)
        cell_observations = [
            item for item in observations
            if item["cell"] == global_cell and item["witness_id"] in used
        ]
        if local is None and not is_anchor:
            cell_records.append({
                **grid_cell,
                "id": f"r{row}c{column}",
                "status": "blank",
                "calibrated": False,
                "component": None,
                "calibration_region": f"r{row}c{column}",
                "correction": None,
                "offset": None,
                "gain": None,
                "witnesses": [],
                "samples": {channel: 0 for channel in PRIMARY_CHANNELS},
            })
            continue

        correction = corrections[local].tolist() if local is not None else [0.0, 0.0, 0.0]
        comp_id = component_by_local.get(local, "REF" if is_anchor else None)
        if is_anchor and comp_id is None:
            comp_id = "REF"
        samples = np.zeros(3, dtype=int)
        weighted = np.zeros(3, dtype=np.float64)
        cell_witness_ids = []
        for observation in cell_observations:
            cell_witness_ids.append(observation["witness_id"])
            counts = observation["samples"]
            samples += counts
            weighted += observation["median"] * np.maximum(counts, 1)
        observed = np.zeros(3, dtype=np.float64)
        np.divide(weighted, np.maximum(samples, 1), out=observed, where=samples > 0)
        corrected = observed + np.asarray(correction)
        status = "reference" if is_anchor else ("anchored" if comp_id == "REF" else "separate")
        cell_records.append({
            **grid_cell,
            "id": f"r{row}c{column}",
            "status": status,
            "calibrated": True,
            "has_evidence": bool(cell_observations),
            "connected_to_reference": comp_id == "REF",
            "component": comp_id,
            "block": comp_id,
            "calibration_region": f"r{row}c{column}",
            "correction": [round(float(x), 6) for x in correction],
            "offset": [round(float(x), 6) for x in correction],
            "density_offset": [round(float(x), 6) for x in correction],
            "gain": [round(float(10 ** (-x)), 6) for x in correction],
            "transmission_gain": [round(float(10 ** (-x)), 6) for x in correction],
            "factor": [round(float(10 ** (-x)), 6) for x in correction],
            "scale": [round(float(10 ** (-x)), 6) for x in correction],
            "max_abs_correction": round(float(np.max(np.abs(correction))), 6),
            "witnesses": list(dict.fromkeys(cell_witness_ids)),
            "samples": {PRIMARY_CHANNELS[c]: int(samples[c]) for c in range(3)},
            "observed_median": [round(float(x), 6) if samples[c] else None for c, x in enumerate(observed)],
            "corrected_median": [round(float(x), 6) if samples[c] else None for c, x in enumerate(corrected)],
        })

    # Boundary edges separate calibrated components from blanks/other blocks.
    boundaries = []
    calibrated_map = {cell["index"]: cell for cell in cell_records if cell["calibrated"]}
    for global_cell, record in calibrated_map.items():
        row, column = divmod(global_cell, columns)
        for dr, dc, side in ((0, 1, "right"), (1, 0, "bottom"), (0, -1, "left"), (-1, 0, "top")):
            nr, nc = row + dr, column + dc
            if not (0 <= nr < rows and 0 <= nc < columns):
                neighbor_id = None
                neighbor_component = "outside"
            else:
                neighbor = nr * columns + nc
                neighbor_record = cell_records[neighbor]
                neighbor_id = f"r{nr}c{nc}"
                neighbor_component = neighbor_record.get("component") if neighbor_record["calibrated"] else "blank"
            if neighbor_component != record.get("component"):
                geometry = record["region"]
                x, y = geometry["x"], geometry["y"]
                right, bottom = x + geometry["w"], y + geometry["h"]
                if side == "right":
                    line = {"start": {"x": right, "y": y}, "end": {"x": right, "y": bottom}}
                elif side == "left":
                    line = {"start": {"x": x, "y": y}, "end": {"x": x, "y": bottom}}
                elif side == "bottom":
                    line = {"start": {"x": x, "y": bottom}, "end": {"x": right, "y": bottom}}
                else:
                    line = {"start": {"x": x, "y": y}, "end": {"x": right, "y": y}}
                boundaries.append({
                    "cell": f"r{row}c{column}",
                    "side": side,
                    "neighbor": neighbor_id,
                    "neighbor_component": neighbor_component,
                    "component": record.get("component"),
                    "region": geometry,
                    "line": line,
                })

    all_witness_ids = sorted({item["witness_id"] for item in observations})
    witness_records = []
    details_by_id = {str(item.get("id", item.get("witness_id", item.get("mark_id")))): item for item in (witness_details or [])}
    for witness_id in all_witness_ids:
        related = [item for item in observations if item["witness_id"] == witness_id]
        detail = details_by_id.get(witness_id, related[0]["detail"] if related else {})
        used_here = witness_id in used
        cells_here = sorted({item["cell"] for item in related})
        record = {
            "id": witness_id,
            "name": related[0]["name"] if related else detail.get("label"),
            "label": related[0]["name"] if related else detail.get("label"),
            "status": "used" if used_here else "discarded",
            "used": used_here,
            "cell_indexes": cells_here,
            "cells": [{"row": divmod(cell, columns)[0], "column": divmod(cell, columns)[1]} for cell in cells_here],
            "samples": {
                PRIMARY_CHANNELS[c]: int(sum(int(item["samples"][c]) for item in related)) for c in range(3)
            },
            "mark_id": detail.get("mark_id", detail.get("id", witness_id)),
            "region": detail.get(
                "region",
                detail.get("geometry", detail.get("box", (detail.get("detail") or {}).get("region"))),
            ),
        }
        if not used_here:
            trial = used | {witness_id}
            conflicts = []
            for channel in range(3):
                _, trial_cells, trial_edges, _ = _feasibility_edges(
                    observations, trial, rows, columns, equal_limit, adjacent_limit, anchor_global
                )
                cycle = _negative_cycle(len(trial_cells) + 1, trial_edges, channel)
                if cycle:
                    conflicts.append(PRIMARY_CHANNELS[channel])
            record["reason"] = "outside the filled limits with the other recommended witnesses"
            record["conflict_channels"] = conflicts
        witness_records.append(record)

    same_name_checks = []
    for name in sorted({item["name"] for item in observations if item["witness_id"] in used}):
        group = [item for item in observations if item["name"] == name and item["witness_id"] in used]
        worst = [0.0, 0.0, 0.0]
        worst_pair = None
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                li = local_by_global[left["cell"]]
                ri = local_by_global[right["cell"]]
                difference = np.abs(left["median"] + corrections[li] - right["median"] - corrections[ri])
                for channel in range(3):
                    if float(difference[channel]) > worst[channel]:
                        worst[channel] = float(difference[channel])
                        worst_pair = (left["cell"], right["cell"])
        same_name_checks.append({
            "name": name,
            "max_difference": [round(x, 6) for x in worst],
            "limit": list(equal_limit),
            "within_limit": all(worst[c] <= equal_limit[c] + 1e-5 for c in range(3)),
            "worst_pair": worst_pair,
        })

    adjacent_checks = []
    for global_cell in cells:
        row, column = divmod(global_cell, columns)
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = row + dr, column + dc
            if not (0 <= nr < rows and 0 <= nc < columns):
                continue
            neighbor = nr * columns + nc
            if neighbor not in local_by_global:
                continue
            li = local_by_global[global_cell]
            ri = local_by_global[neighbor]
            jump = np.abs(corrections[li] - corrections[ri])
            adjacent_checks.append({
                "cells": [f"r{row}c{column}", f"r{nr}c{nc}"],
                "max_jump": [round(float(x), 6) for x in jump],
                "limit": list(adjacent_limit),
                "within_limit": all(float(jump[c]) <= adjacent_limit[c] + 1e-5 for c in range(3)),
            })

    max_cell = None
    max_channel = None
    max_value = 0.0
    if corrections.size:
        flat_index = int(np.argmax(np.abs(corrections)))
        local_index, channel_index = divmod(flat_index, 3)
        max_value = float(abs(corrections[local_index, channel_index]))
        max_channel = PRIMARY_CHANNELS[channel_index]
        max_cell_global = cells[local_index]
        max_cell = {"row": divmod(max_cell_global, columns)[0], "column": divmod(max_cell_global, columns)[1]}

    correction_grid = [
        [cell_records[row * columns + column]["correction"] for column in range(columns)]
        for row in range(rows)
    ]

    used_records = [item for item in witness_records if item["status"] == "used"]
    discarded_records = [item for item in witness_records if item["status"] == "discarded"]
    calibrated_count = len(cells) + (1 if anchor_global is not None and anchor_global not in local_by_global else 0)
    return {
        "status": "recommended" if observations else "no_witnesses",
        "state": "recommended" if observations else "no_witnesses",
        "verdict": "recommended" if observations else "no_witnesses",
        "recommended": bool(observations),
        "adoptable": bool(observations),
        "rows": rows,
        "columns": columns,
        "region": (region.as_dict() if isinstance(region, Box) else region) if region is not None else None,
        "reference_cell": {"row": reference_cell[0], "column": reference_cell[1]} if reference_cell else None,
        "limits": {"same_name": list(equal_limit), "jump": list(adjacent_limit)},
        "cells": cell_records,
        "tiles": cell_records,
        "corrections": correction_grid,
        "correction_grid": correction_grid,
        "components": component_records,
        "calibration_blocks": component_records,
        "boundaries": boundaries,
        "witnesses": witness_records,
        "used_witnesses": used_records,
        "discarded_witnesses": discarded_records,
        "discarded": discarded_records,
        "dropped_witnesses": discarded_records,
        "excluded_witnesses": discarded_records,
        "rejected_witnesses": discarded_records,
        "blank_cells_list": [cell for cell in cell_records if not cell["calibrated"]],
        "empty_cells": [cell for cell in cell_records if not cell["calibrated"]],
        "same_name_checks": same_name_checks,
        "equal_name_checks": same_name_checks,
        "adjacent_checks": adjacent_checks,
        "neighbor_checks": adjacent_checks,
        "constraints": {"same_name": same_name_checks, "adjacent": adjacent_checks},
        "violations": [
            check for check in [*same_name_checks, *adjacent_checks] if not check["within_limit"]
        ],
        "held_cells": calibrated_count,
        "calibrated_cells": calibrated_count,
        "blank_cells": rows * columns - calibrated_count,
        "used_witness_count": len(used_records),
        "discarded_witness_count": len(discarded_records),
        "max_correction": round(max_value, 6),
        "maximum_correction": round(max_value, 6),
        "max_abs_correction": round(max_value, 6),
        "max_adjustment": round(max_value, 6),
        "maximum_adjustment": round(max_value, 6),
        "maximum_correction_density": round(max_value, 6),
        "max_correction_detail": {"value": round(max_value, 6), "channel": max_channel, "cell": max_cell},
        "search": search_info,
        "zone_level": "cell",
        "cell_zone_level": "cell",
        "compare_zone_level": "cell",
    }


def observations_from_marks(calibrated, grid: list[dict[str, Any]], marks: list[dict[str, Any]]):
    """Sample every saved Reference mark against every illumination tile."""
    density = calibrated.density
    valid = calibrated.valid
    height, width = density.shape[:2]
    observations: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    observation_index = 0
    for mark_index, mark in enumerate(marks):
        mark = mapping(mark, f"marks[{mark_index}]")
        if mark.get("kind", "reference") not in {"reference", "keep"}:
            continue
        label = mark.get("label", mark.get("name"))
        if not isinstance(label, str) or not label.strip():
            continue
        geometry = mark.get("geometry", mark.get("region", mark.get("box", mark)))
        try:
            box = Box.read(geometry, "geometry", bounds=(width, height))
        except DeskError:
            continue
        witness_id = str(mark.get("id", mark.get("mark_id", f"mark-{mark_index + 1}")))
        details.append({
            "id": witness_id,
            "mark_id": witness_id,
            "label": label.strip(),
            "region": box.as_dict(),
        })
        intersected = False
        for cell in grid:
            tile = Box.read(cell["region"], "cell")
            cut = box.intersect(tile)
            if cut is None:
                continue
            medians = []
            counts = []
            for channel in range(3):
                channel_values = density[cut.slices][..., channel]
                channel_valid = valid[cut.slices][..., channel]
                chosen = channel_values[channel_valid]
                counts.append(int(chosen.size))
                if chosen.size:
                    medians.append(float(np.median(chosen)))
            if not all(counts):
                continue
            observations.append({
                "observation_id": f"{witness_id}-{cell['id'] if 'id' in cell else cell['index']}",
                "witness_id": witness_id,
                "name": label.strip(),
                "row": cell["row"],
                "column": cell["column"],
                "cell": cell["index"],
                "median": np.asarray(medians, dtype=float),
                "samples": np.asarray(counts, dtype=int),
                "detail": {"id": witness_id, "mark_id": witness_id, "label": label.strip(), "region": box.as_dict()},
            })
            observation_index += 1
            intersected = True
        if not intersected:
            continue
    return observations, details


def _correction_vector(entry: dict[str, Any]) -> np.ndarray | None:
    raw = entry.get(
        "correction",
        entry.get("offset", entry.get("density_offset", entry.get("correction_density", entry.get("delta")))),
    )
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None
        return np.asarray([float(x) for x in raw], dtype=np.float64)
    gain = entry.get("gain", entry.get("factor", entry.get("scale", entry.get("transmission_gain"))))
    if gain is not None and isinstance(gain, (list, tuple)) and len(gain) == 3:
        values = np.asarray([float(x) for x in gain], dtype=np.float64)
        return -np.log10(np.maximum(values, 1e-12))
    return None


def compensation_maps(compensation: Any, height: int, width: int, *, zone_level: str | None = None):
    if compensation is None:
        return None
    if not isinstance(compensation, dict):
        raise DeskError("Adopt a witness compensation plan before comparing regions.", field="compensation")
    level = (
        zone_level
        or compensation.get("compare_zone_level")
        or compensation.get("calibration_level")
        or compensation.get("zone_level")
        or "cell"
    )
    offset = np.zeros((height, width, 3), dtype=np.float64)
    cell_map = np.full((height, width), -1, dtype=np.int32)
    block_map = np.full((height, width), -1, dtype=np.int32)
    cell_keys: list[str] = []
    block_keys: list[str] = []

    def paint(box: Box, vector: np.ndarray, cell_key: str, block_key: str | None = None):
        x0 = max(0, box.x)
        y0 = max(0, box.y)
        x1 = min(width, box.right)
        y1 = min(height, box.bottom)
        if x0 >= x1 or y0 >= y1:
            return
        if cell_key not in cell_keys:
            cell_keys.append(cell_key)
        cell_code = cell_keys.index(cell_key)
        use_block = block_key if level in {"block", "component", "calibration_block"} else cell_key
        if use_block not in block_keys:
            block_keys.append(use_block)
        block_code = block_keys.index(use_block)
        offset[y0:y1, x0:x1, :] = vector
        cell_map[y0:y1, x0:x1] = cell_code
        block_map[y0:y1, x0:x1] = block_code

    entries = compensation.get("calibration_regions", compensation.get("zones", compensation.get("regions")))
    if isinstance(entries, list):
        for entry in entries:
            entry = mapping(entry, "calibration_regions[]")
            vector = _correction_vector(entry)
            geometry = entry.get("region", entry.get("box", entry.get("geometry")))
            if vector is None or geometry is None:
                continue
            box = Box.read(geometry, "calibration_region").intersect(Box(0, 0, width, height))
            if box is None:
                continue
            key = str(entry.get("id", entry.get("calibration_region", entry.get("name", f"zone-{len(cell_keys)}"))))
            paint(box, vector, key, str(entry.get("component", entry.get("block", key))))

    cells = compensation.get("cells", compensation.get("tiles"))
    if cells is None and isinstance(compensation.get("corrections"), list):
        rows = int(compensation.get("rows", 1))
        columns = int(compensation.get("columns", len(compensation["corrections"])))
        region = Box.read(compensation.get("region", {"x": 0, "y": 0, "w": width, "h": height}), "compensation.region")
        generated = grid_regions(region, rows, columns)
        raw_corrections = compensation["corrections"]
        correction_array = np.asarray(raw_corrections, dtype=float)
        if correction_array.ndim == 3:
            flat_corrections = correction_array.reshape(-1, 3)
        elif correction_array.ndim == 2 and correction_array.shape[1] == 3:
            flat_corrections = correction_array
        elif correction_array.ndim == 1 and correction_array.size == rows * columns * 3:
            flat_corrections = correction_array.reshape(-1, 3)
        else:
            flat_corrections = correction_array.reshape(-1, 3) if correction_array.size % 3 == 0 else []
        cells = []
        for index, geometry in enumerate(generated):
            vector = flat_corrections[index].tolist() if index < len(flat_corrections) else None
            cells.append({**geometry, "correction": vector})

    if isinstance(cells, list):
        for entry in cells:
            entry = mapping(entry, "compensation.cells[]")
            vector = _correction_vector(entry)
            if vector is None or not all(np.isfinite(vector)):
                continue
            geometry = entry.get("region", entry.get("box"))
            if geometry is None and {"row", "column"} <= set(entry):
                rows = int(compensation.get("rows", 1))
                columns = int(compensation.get("columns", 1))
                base_region = compensation.get("region", {"x": 0, "y": 0, "w": width, "h": height})
                geometry = grid_regions(Box.read(base_region, "compensation.region"), rows, columns)[
                    int(entry["row"]) * columns + int(entry["column"])
                ]["region"]
            if geometry is None:
                continue
            box = Box.read(geometry, "compensation.cell").intersect(Box(0, 0, width, height))
            if box is None:
                continue
            row = entry.get("row")
            column = entry.get("column")
            key = str(entry.get("calibration_region", entry.get("id", f"r{row}c{column}")))
            block = entry.get("component", entry.get("block", entry.get("calibration_block", key)))
            paint(box, vector, key, str(block))

    active_map = block_map if level in {"block", "component", "calibration_block"} else cell_map
    active_keys = block_keys if level in {"block", "component", "calibration_block"} else cell_keys
    if not active_keys:
        raise DeskError("The adopted compensation has no usable calibration regions.", field="compensation")
    return {
        "offset": offset,
        "zone_map": active_map,
        "zone_keys": active_keys,
        "cell_map": cell_map,
        "cell_keys": cell_keys,
        "block_map": block_map,
        "block_keys": block_keys,
        "zone_level": level,
    }


def _zone_summaries(density_corrected, valid_slice, zone_slice, zone_keys, channel, minimum_fraction=0.5):
    summaries = []
    total = int(valid_slice.sum())
    for code, key in enumerate(zone_keys):
        mask = valid_slice & (zone_slice == code)
        count = int(mask.sum())
        if not count:
            continue
        values = density_corrected[..., channel][mask]
        summaries.append({
            "calibration_region": key,
            "samples": count,
            "measured_fraction": round(count / total, 5) if total else None,
            "median": round(float(np.median(values)), 5),
            "majority": total > 0 and count + 1e-12 >= minimum_fraction * total,
        })
    summaries.sort(key=lambda item: (-item["samples"], item["calibration_region"]))
    return total, summaries


def provenance_compare(density, valid, region: Box, reference: Box, compensation: Any, *, zone_level: str | None = None, minimum_fraction: float = .5):
    height, width = density.shape[:2]
    maps = compensation_maps(compensation, height, width, zone_level=zone_level)
    if maps is None:
        return None
    corrected_density = density + maps["offset"]
    zone_map = maps["zone_map"]
    zone_keys = maps["zone_keys"]
    result = {
        "zone_level": maps["zone_level"],
        "delta": {},
        "missing_channels": [],
        "comparable_channels": [],
        "common_regions": {},
        "parts": {},
        "separate": {},
    }
    for channel_index, channel in enumerate(PRIMARY_CHANNELS):
        r_slice = region.slices
        f_slice = reference.slices
        r_valid = valid[r_slice][..., channel_index]
        f_valid = valid[f_slice][..., channel_index]
        r_zones = zone_map[r_slice]
        f_zones = zone_map[f_slice]
        r_total, r_parts = _zone_summaries(corrected_density[r_slice], r_valid, r_zones, zone_keys, channel_index, minimum_fraction)
        f_total, f_parts = _zone_summaries(corrected_density[f_slice], f_valid, f_zones, zone_keys, channel_index, minimum_fraction)
        r_major = {item["calibration_region"]: item for item in r_parts if item["majority"]}
        f_major = {item["calibration_region"]: item for item in f_parts if item["majority"]}
        common = [key for key in r_major if key in f_major]
        result["parts"][channel] = {
            "region": r_parts,
            "reference": f_parts,
            "region_samples": r_total,
            "reference_samples": f_total,
            "region_unassigned_samples": r_total - sum(item["samples"] for item in r_parts),
            "reference_unassigned_samples": f_total - sum(item["samples"] for item in f_parts),
        }
        if not common:
            result["delta"][channel] = None
            result["missing_channels"].append(channel)
            left_zone = r_parts[0]["calibration_region"] if r_parts else None
            right_zone = f_parts[0]["calibration_region"] if f_parts else None
            result["separate"][channel] = {
                "left_dominant_region": left_zone,
                "reference_dominant_region": right_zone,
                "left_majority_regions": list(r_major),
                "reference_majority_regions": list(f_major),
                "reason": "No single calibration region supplies at least half of measured pixels in both boxes.",
            }
            continue
        # Prefer the region with the largest balanced sample support.
        chosen = max(common, key=lambda key: (
            min(r_major[key]["measured_fraction"] or 0, f_major[key]["measured_fraction"] or 0),
            r_major[key]["samples"] + f_major[key]["samples"],
        ))
        left_median = r_major[chosen]["median"]
        right_median = f_major[chosen]["median"]
        result["delta"][channel] = round(left_median - right_median, 5)
        result["comparable_channels"].append(channel)
        result["common_regions"][channel] = {
            "calibration_region": chosen,
            "region_samples": r_major[chosen]["samples"],
            "reference_samples": f_major[chosen]["samples"],
            "region_measured_fraction": r_major[chosen]["measured_fraction"],
            "reference_measured_fraction": f_major[chosen]["measured_fraction"],
            "region_median": left_median,
            "reference_median": right_median,
        }
    if not result["missing_channels"]:
        result["status"] = "measured"
        result["comparable"] = True
    elif not result["comparable_channels"]:
        result["status"] = "separate"
        result["comparable"] = False
        result["message"] = "The boxes are supported by separate calibration regions; no merged difference is reported."
    else:
        result["status"] = "partial"
        result["comparable"] = False
        result["message"] = "Some channels do not share a majority calibration region."
    result["state"] = result["status"]
    result["verdict"] = result["status"]
    result["separate_channels"] = list(result["missing_channels"])
    result["incomparable_channels"] = list(result["missing_channels"])
    result["separate_parts"] = result["separate"]
    result["separations"] = result["separate"]
    result["split_parts"] = result["separate"]
    result["is_separate"] = result["status"] == "separate"
    result["no_merged_delta"] = result["status"] == "separate"
    result["merged_delta"] = None if result["missing_channels"] else result["delta"]
    return result
